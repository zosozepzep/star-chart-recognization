# -*- coding: utf-8 -*-
"""教程 04：派生数据集 1 与 2 —— 目标轨迹角速度、圆轨道高度反演。

容器内运行：
    SC_IMAGE=star-chart:cpu-interim bash scripts/dr.sh python examples/04_trajectory_orbit.py

约 110 秒。

**板比例尺约定**：本项目全程用四舍五入的 SCALE = 6.179 ″/px。真值比对给出的
全精度值是 6.179046984546331；用全精度会得到 770.156602 ″/s，而报告里的
770.150746 ″/s 是 6.179 的结果。两者都对，但**必须选一个**，否则你自己跑出来的
数与 docs/reports/measurements.md 对不上。
"""
import json
import time

from src.analysis.orbit import estimate_from_trajectory
from src.calib.hotpixel import build_from_sequence
from src.calib.segment import tracking_segment
from src.dataio.fits_loader import FrameSequence
from src.detect.segmentation import detect_sequence
from src.register.solver import register_sequence
from src.target.dual_frame import find_targets
from src.target.trajectory import build_trajectory
from src.validate.truth import compare, load_truth

DIR = "data/images/60385_20260722_1485695076008039_PIC_POS"
SCALE = 6.179          # 交付口径：四舍五入值，见模块 docstring
t0 = time.time()

seq = FrameSequence.from_directory(DIR)
seg = tracking_segment(seq.headers)
frames = seg.frames
hpm = build_from_sequence(seq, frames[:30])
dets = detect_sequence(seq, frames, n_sigma=4.0, npixels=5, hot_clusters=hpm.clusters)
reg = register_sequence(dets, frames, reference=seg.start)
track = find_targets(dets, frames, reg)[0].track

truth = load_truth(seq.truth_path)
full_scale = compare(track, truth, seq, reg).plate_scale_arcsec_px
print("主链完成 t=%.1fs" % (time.time() - t0))

traj = build_trajectory(track, seq, SCALE)
print("--- 派生 1：轨迹与角速度 ---")
print("  轨迹点数            = %d" % len(traj))
print("  板比例（全精度）    = %.12f arcsec/px" % full_scale)
print("  板比例（本次使用）  = %.6f arcsec/px" % SCALE)
print("  mean_rate_arcsec_s  = %.6f" % traj.mean_rate_arcsec_s)
print("  rate_std_arcsec_s   = %.6f" % traj.rate_std_arcsec_s)
print("  相对散度 std/mean   = %.6f" % (traj.rate_std_arcsec_s / traj.mean_rate_arcsec_s))
print("  （速率取 53 个相邻对，points[0] 是前向填充值、不参与）")

# estimate_from_trajectory 按 point.frame 取每点的俯仰角，
# 不给调用方「传哪些头部」的自由度——这是高度带能复现的前提
est = estimate_from_trajectory(traj, seq.headers)
print("--- 派生 2：圆轨道高度反演 ---")
print(json.dumps(est.to_dict(), ensure_ascii=False, indent=2, default=float))
print("  只报自相容区间，**不报**「与公开两行根数吻合」——TLE 尚未抄进"
      " docs/reference/tle.txt")
print("total t=%.1fs" % (time.time() - t0))
