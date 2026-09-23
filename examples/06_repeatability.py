# -*- coding: utf-8 -*-
"""教程 06：阈值-复现率曲线（「为什么只报 10² 颗星」的定量依据）。

容器内运行：
    bash scripts/dr.sh python examples/06_repeatability.py

**约 100 秒**（两个数据集合计）。九档 sigma × 2 个数据集 × 6 帧，
背景模型逐帧建一次跨全档复用，但仍然是 GB 级驻留（4096² float64 单幅 134 MB）。
`frames` 不是随手可加大的参数：它同时决定驻留量与复现率的结构性上限。

这一节不需要真值：可复现性对着数据自身量。src/validate/repeatability.py
被硬约束禁止 import src.validate.truth——一旦引入真值，报出的星数就变成
答案文件的函数。
"""
from src.calib.hotpixel import build_from_sequence
from src.calib.segment import tracking_segment
from src.config import load_config
from src.dataio.fits_loader import FrameSequence
from src.detect.segmentation import detect_sequence
from src.register.solver import register_sequence
from src.validate.repeatability import min_hits_for, scan_thresholds

DIRS = {
    "A": "data/images/60394_20260309_1437117414637934_PIC",
    "B": "data/images/60385_20260722_1485695076008039_PIC_POS",
}

conf = load_config()
sigmas = conf["repeatability"]["sigmas"]
radius = conf["repeatability"]["match_radius_px"]
n_frames = conf["repeatability"]["frames"]
min_hits = min_hits_for(0.6, n_frames)
print("min_hits = %d of %d（含参考帧本身），match_radius = %.1f px" % (min_hits, n_frames, radius))
print("sigmas = %s" % sigmas)

for tag, d in DIRS.items():
    seq = FrameSequence.from_directory(d)
    seg = tracking_segment(seq.headers)
    # 段首 +1：段首帧机架未停稳，实测使同一 sigma 的复现率相差 3.1 倍
    frames = list(range(seg.start + 1, seg.start + 1 + n_frames))
    hpm = build_from_sequence(seq, frames)
    dets = detect_sequence(seq, frames, n_sigma=conf["detect"]["search_n_sigma"],
                           hot_clusters=hpm.clusters)
    reg = register_sequence(dets, frames, reference=frames[0], config=dict(conf["register"]))
    curve = scan_thresholds(seq, frames, reg, sigmas=sigmas,
                            npixels=conf["detect"]["npixels"],
                            hot_clusters=hpm.clusters, match_radius_px=radius)

    print("=== 历元 %s，帧 f%02d-f%02d ===" % (tag, frames[0], frames[-1]))
    print("| sigma | 各帧均值探测数 | 参考帧数 | 可复现数 | 复现率 | 纯度 |")
    print("|---|---|---|---|---|---|")
    for p in curve.points:
        print("| %.1f | %.4f | %d | %d | %.6f | %.6f |"
              % (p.n_sigma, p.n_detected_mean, p.n_reference,
                 p.n_reproducible, p.reproducibility, p.purity))

    # recommended() 用后缀判据：取「自身及所有更高档都达标」的最低 sigma。
    # 两个数据集在 0.85 门限下都达不到——这是如实报告的实测结果，不放宽阈值。
    for gate in (0.85, 0.75, 0.60):
        try:
            best = curve.recommended(gate)       # 返回 RepeatabilityPoint，不是 float
            print("  recommended(%.2f) -> %.1f sigma（复现率 %.6f，可复现 %d 颗）"
                  % (gate, best.n_sigma, best.reproducibility, best.n_reproducible))
        except ValueError as exc:
            print("  recommended(%.2f) -> 抛错: %s" % (gate, exc))
