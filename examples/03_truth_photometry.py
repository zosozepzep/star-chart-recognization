# -*- coding: utf-8 -*-
"""教程 03：真值比对、星等零点、极限星等。

容器内运行：
    bash scripts/dr.sh python examples/03_truth_photometry.py

约 110 秒。这是全教程唯一读 .DAT 真值的一步——识别链本身不读真值，
真值只用于「验证」与「星等零点标定」两处。
"""
import json
import time

from src.astrometry.photometry import limiting_magnitude, zero_point_from_truth
from src.calib.background import model_background
from src.calib.hotpixel import build_from_sequence
from src.calib.segment import tracking_segment
from src.dataio.fits_loader import FrameSequence
from src.detect.segmentation import detect_sequence, detect_sources_in_frame
from src.register.solver import register_sequence
from src.target.dual_frame import find_targets
from src.validate.truth import compare, load_truth, match_frames

DIR = "data/images/60385_20260722_1485695076008039_PIC_POS"
t0 = time.time()

seq = FrameSequence.from_directory(DIR)
seg = tracking_segment(seq.headers)
frames = seg.frames
hpm = build_from_sequence(seq, frames[:30])
dets = detect_sequence(seq, frames, n_sigma=4.0, npixels=5, hot_clusters=hpm.clusters)
reg = register_sequence(dets, frames, reference=seg.start)
track = find_targets(dets, frames, reg)[0].track
print("主链完成 t=%.1fs" % (time.time() - t0))

truth = load_truth(seq.truth_path)
report = compare(track, truth, seq, reg)
print("--- 真值比对 ---")
print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=float))

# 零点：流量与真值星等必须按 match_frames 的配对下标对齐，不能按位置硬切
matched_frames, matched_rows = match_frames(truth, seq, track.frames)
print("配对帧数 = %d" % len(matched_frames))
flux_by_frame = dict(zip(track.frames, track.flux))
zp = zero_point_from_truth([flux_by_frame[int(f)] for f in matched_frames], truth.mag[matched_rows])
print("--- 星等零点 ---")
print(json.dumps(zp.to_dict(), ensure_ascii=False, indent=2))

# 极限星等：单帧源表的经验暗端百分位，不是「最暗可复现源」
img = seq.image(30)
table = detect_sources_in_frame(img, model_background(img), n_sigma=5.0, npixels=5, frame=30)
print("--- 极限星等（第 30 帧，5σ）---")
print("源数 = %d" % len(table))
print("limiting_magnitude(95pct) = %.4f  ← 经验暗端百分位" % limiting_magnitude(table, zp))
print("total t=%.1fs" % (time.time() - t0))
