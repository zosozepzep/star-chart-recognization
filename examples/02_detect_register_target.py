# -*- coding: utf-8 -*-
"""教程 02：探测 → 配准 → 双坐标系目标判决（项目主链）。

容器内运行：
    bash scripts/dr.sh python examples/02_detect_register_target.py

约 100 秒（4096² 单帧背景建模 1.48 s，55 帧探测是主要开销）。
输出应与 docs/reports/measurements.md「配准」「目标判决」两节逐位一致。
"""
import json
import time

from src.calib.hotpixel import build_from_sequence
from src.calib.segment import tracking_segment
from src.dataio.fits_loader import FrameSequence
from src.detect.segmentation import detect_sequence
from src.register.solver import register_sequence
from src.target.dual_frame import find_targets

DIR = "data/images/60385_20260722_1485695076008039_PIC_POS"
t0 = time.time()

seq = FrameSequence.from_directory(DIR)
seg = tracking_segment(seq.headers)
frames = seg.frames

# 热像素图只用前 30 帧建：跨历元 2/4 个簇逐位相同，30 帧已在收敛平台上
hpm = build_from_sequence(seq, frames[:30])
print("热像素簇 = %d   t=%.1fs" % (len(hpm.clusters), time.time() - t0))

# 搜索用 4.0σ（detect.search_n_sigma），报数用 5.0σ（detect.report_n_sigma）
dets = detect_sequence(seq, frames, n_sigma=4.0, npixels=5, hot_clusters=hpm.clusters)
print("探测完成          t=%.1fs" % (time.time() - t0))

reg = register_sequence(dets, frames, reference=seg.start)
rep = reg.report()
print("--- 配准（以段首帧 f%d 为参考）---" % seg.start)
print(json.dumps({k: v for k, v in rep.items() if k != "pairs"},
                 ensure_ascii=False, indent=2, default=float))

hits = find_targets(dets, frames, reg)
print("--- 目标判决 ---")
print("判为目标的航迹数 = %d" % len(hits))
print(json.dumps(hits[0].to_dict(), ensure_ascii=False, indent=2, default=float))
print("total t=%.1fs" % (time.time() - t0))
