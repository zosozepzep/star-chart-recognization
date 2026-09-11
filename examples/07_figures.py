# -*- coding: utf-8 -*-
"""教程 07：出 8 张成果图（用真实数据，不用测试夹具）。

容器内运行：
    SC_IMAGE=star-chart:cpu-interim bash scripts/dr.sh python examples/07_figures.py

约 3 分钟（含一次完整主链 + 一次 6 帧阈值扫描的最小版）。PNG 落在 output/figures/。

**中文字体**：setup_matplotlib() 要求字体能覆盖 FIGURE_CJK 的 307 个字符
（304 个汉字）。当前镜像 star-chart:cpu-interim 里的 SimHei 缺 0 个，可用；
DejaVu Sans 缺 304 个，不可用。若抛 FontUnavailable，说明镜像里没有中文字体
——参见 README「已知限制」。
"""
import time
from pathlib import Path

import numpy as np

from src.analysis.orbit import estimate_from_trajectory
from src.analysis.sensor_health import (
    build_report,
    estimate_seeing,
    sky_brightness,
)
from src.astrometry.photometry import (
    calibrate_table,
    limiting_magnitude,
    zero_point_from_truth,
)
from src.calib.background import background_stats, model_background
from src.calib.hotpixel import build_from_sequence
from src.calib.segment import tracking_segment
from src.config import load_config
from src.dataio.fits_loader import FrameSequence
from src.detect.psf import fit_table
from src.detect.segmentation import detect_sequence, detect_sources_in_frame
from src.register.solver import register_sequence
from src.target.dual_frame import find_targets
from src.target.trajectory import build_trajectory
from src.validate.repeatability import scan_thresholds
from src.validate.truth import compare, load_truth, match_frames
from src.viz import figures as F

DIR = "data/images/60385_20260722_1485695076008039_PIC_POS"
OUT = Path("output/figures")
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 6.179
t0 = time.time()

font = F.setup_matplotlib()
print("图上中文字体 = %s（FIGURE_CJK %d 字符，缺 %d 个）"
      % (font, len(F.FIGURE_CJK), len(F.glyphs_missing(font))))

conf = load_config()
seq = FrameSequence.from_directory(DIR)
seg = tracking_segment(seq.headers)
frames = seg.frames
hpm = build_from_sequence(seq, frames[:30])
dets = detect_sequence(seq, frames, n_sigma=4.0, npixels=5, hot_clusters=hpm.clusters)
reg = register_sequence(dets, frames, reference=seg.start)
verdict = find_targets(dets, frames, reg)[0]
track = verdict.track
print("主链完成 t=%.1fs" % (time.time() - t0))

truth = load_truth(seq.truth_path)
report = compare(track, truth, seq, reg)
_, rows = match_frames(truth, seq, track.frames)
zp = zero_point_from_truth(track.flux[:len(rows)], truth.mag[rows])
traj = build_trajectory(track, seq, SCALE)
orbit = estimate_from_trajectory(traj, seq.headers)

# 图 1：原图 + 探测源，叠加目标位置与热像素簇
img30 = seq.image(30)
model30 = model_background(img30)
table30 = detect_sources_in_frame(img30, model30, n_sigma=5.0, npixels=5, frame=30)
F.plot_detections(img30, table30, OUT / "fig1-detections.png",
                  target_xy=tuple(verdict.to_dict()["xy_end"]),
                  hot_clusters=list(hpm.clusters))
print("图 1 探测叠加            t=%.1fs" % (time.time() - t0))

# 图 2：阈值-复现率曲线（教程 06 的最小版，六帧、段首 +1）
cf = list(range(seg.start + 1, seg.start + 1 + conf["repeatability"]["frames"]))
cdets = detect_sequence(seq, cf, n_sigma=conf["detect"]["search_n_sigma"],
                        hot_clusters=hpm.clusters)
creg = register_sequence(cdets, cf, reference=cf[0], config=dict(conf["register"]))
curve = scan_thresholds(seq, cf, creg, sigmas=conf["repeatability"]["sigmas"],
                        npixels=conf["detect"]["npixels"],
                        hot_clusters=hpm.clusters,
                        match_radius_px=conf["repeatability"]["match_radius_px"])
F.plot_threshold_curve(curve, OUT / "fig2-threshold-curve.png",
                       title="数据集 B 阈值-复现率曲线")
print("图 2 阈值曲线            t=%.1fs" % (time.time() - t0))

F.plot_registration_residuals(reg, OUT / "fig3-registration-residuals.png")
F.plot_dual_frame_verdict(verdict, OUT / "fig4-dual-frame-verdict.png")
F.plot_trajectory(traj, OUT / "fig5-trajectory.png")
F.plot_height_interval(orbit, OUT / "fig6-height-interval.png")

N = int(conf["sensor_health"]["report_frames"])
sensor = build_report(seq.dataset_id, hpm, background_stats(seq.image(N - 1)),
                     epoch_isot=seq.headers[0].date_obs.isot,
                     background_frame_index=N - 1,
                     background_exposure_ms=seq.headers[N - 1].exposure_ms)
F.plot_hotpixels(hpm, OUT / "fig7-hotpixels.png", shape=img30.shape, report=sensor)

_, fits = fit_table(model30.subtract(img30), table30.brightest(80))
seeing = estimate_seeing(fits, SCALE)
bkg_median = float(np.median(np.asarray(img30, dtype=np.float64)))
mu = sky_brightness(bkg_median, zero_point=zp, scale_arcsec_px=SCALE)
F.plot_magnitude_histogram(calibrate_table(table30, zp),
                           OUT / "fig8-magnitude-histogram.png",
                           limiting=limiting_magnitude(table30, zp),
                           seeing=seeing, sky_mag_arcsec2=mu)

print("--- 8 张图已落在 %s ---" % OUT)
for p in sorted(OUT.glob("*.png")):
    print("  %-34s %d bytes" % (p.name, p.stat().st_size))
print("total t=%.1fs" % (time.time() - t0))
