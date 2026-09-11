# -*- coding: utf-8 -*-
"""教程 05：派生数据集 4 与 5 —— 传感器健康报告、观测条件量化。

容器内运行：
    SC_IMAGE=star-chart:cpu-interim bash scripts/dr.sh python examples/05_sensor_seeing.py

约 60 秒。这一步不需要真值（除天光面亮度用到零点），两个历元都能跑。
"""
import json

import numpy as np

from src.analysis.sensor_health import (
    build_report,
    compare_epochs,
    estimate_seeing,
    sky_brightness,
)
from src.astrometry.photometry import ZeroPoint
from src.calib.background import background_stats, model_background
from src.calib.hotpixel import build_from_sequence
from src.config import load_config
from src.dataio.fits_loader import FrameSequence
from src.detect.psf import fit_table
from src.detect.segmentation import detect_sources_in_frame

DIRS = {
    "B": "data/images/60385_20260722_1485695076008039_PIC_POS",
    "A": "data/images/60394_20260309_1437117414637934_PIC",
}
SCALE = 6.179                       # 与教程 04 同一约定
ZP, ZP_STD, ZP_N = 15.335125, 0.217907, 54   # 真值零点，来自教程 03

# report_frames 从配置读，出厂 20：它是簇数收敛平台的起点
# （5 帧 -> 115 簇、10 -> 8、15 -> 6、20 -> 4、25/30/36 -> 4），不是随手选的数
N = int(load_config()["sensor_health"]["report_frames"])
print("report_frames = %d（来自 src/config/default.yaml）" % N)

reports = {}
for tag, d in DIRS.items():
    seq = FrameSequence.from_directory(d)
    frames = list(range(N))
    hpm = build_from_sequence(seq, frames)
    stats = background_stats(seq.image(frames[-1]))
    reports[tag] = build_report(
        seq.dataset_id, hpm, stats,
        epoch_isot=seq.headers[0].date_obs.isot,
        background_frame_index=frames[-1],
        background_exposure_ms=seq.headers[frames[-1]].exposure_ms,
    )
    print("=== 派生 4：历元 %s 传感器健康 ===" % tag)
    print(json.dumps(reports[tag].to_dict(), ensure_ascii=False, indent=2, default=float))

print("=== 派生 4：跨历元对比（相隔约 4.5 个月）===")
print(json.dumps(compare_epochs(reports["A"], reports["B"]).to_dict(),
                 ensure_ascii=False, indent=2, default=float))

for tag, d in DIRS.items():
    seq = FrameSequence.from_directory(d)
    img = seq.image(30)
    model = model_background(img)
    table = detect_sources_in_frame(img, model, n_sigma=5.0, frame=30).brightest(80)
    _, fits = fit_table(model.subtract(img), table)

    est = estimate_seeing(fits, SCALE)
    bkg_median = float(np.median(np.asarray(img, dtype=np.float64)))
    zp = ZeroPoint(ZP, ZP_STD, ZP_N, "truth", exposure_s=None)
    mu = sky_brightness(bkg_median, zero_point=zp, scale_arcsec_px=SCALE)

    print("=== 派生 5：历元 %s 观测条件（第 30 帧，5σ，最亮 80 源）===" % tag)
    print("  n_stars / n_rejected = %d / %d" % (est.n_stars, est.n_rejected))
    print("  fwhm_px              = %.6f   ← 纯像素量，可作独立主张" % est.fwhm_px)
    print("  elongation_median    = %.6f   ← fwhm_px 可读性的前提条件" % est.elongation_median)
    print("  fwhm_arcsec          = %.6f   ← 乘了真值板比例，是定标量" % est.fwhm_arcsec)
    print("  全帧背景中位         = %.1f ADU" % bkg_median)
    print("  天光面亮度           = %.2f mag/arcsec^2   ← 条件命题，报告里不写小数" % mu)
