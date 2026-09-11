# -*- coding: utf-8 -*-
"""教程 01：读序列、看头部、切跟踪段。

容器内运行：
    SC_IMAGE=star-chart:cpu-interim bash scripts/dr.sh python examples/01_load.py

这一步不做任何探测，只回答「这批数据是什么」。约 3 秒。
"""
from src.calib.segment import tracking_segment
from src.dataio.fits_loader import FrameSequence

DIRS = {
    "B": "data/images/60385_20260722_1485695076008039_PIC_POS",
    "A": "data/images/60394_20260309_1437117414637934_PIC",
}

for tag, d in DIRS.items():
    seq = FrameSequence.from_directory(d)
    seg = tracking_segment(seq.headers)
    h0 = seq.headers[0]

    print("=== 历元 %s ===" % tag)
    print("  dataset_id     = %s" % seq.dataset_id)
    print("  帧数           = %d" % len(seq.headers))
    print("  首帧 DATE-OBS  = %s" % h0.date_obs.isot)
    print("  方位 / 俯仰    = %.5f / %.5f deg" % (h0.azimuth_deg, h0.elevation_deg))
    print("  曝光           = %.1f ms" % h0.exposure_ms)
    print("  真值 .DAT      = %s" % (seq.truth_path.name if seq.truth_path else "无"))
    print("  跟踪段         = f%02d-f%02d，共 %d 帧" % (seg.start, seg.end, seg.length))
    print("  段首帧要跳过   = 机架尚未停稳，配准 rms 会抬高（见 docs/reports/measurements.md）")
