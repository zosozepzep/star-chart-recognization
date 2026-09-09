from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time

from src.calib.segment import (
    NoTrackingSegment,
    Segment,
    azimuth_rates,
    segment_sequence,
    tracking_segment,
)
from src.dataio.fits_loader import FrameHeader, FrameSequence


def make_headers(azimuths, cadence=1.0):
    t0 = Time("2026-07-21T17:26:28.000", scale="utc")
    return [
        FrameHeader(
            index=i,
            path=f"synthetic_{i:03d}.fits",
            date_obs=t0 + i * cadence * u.s,
            azimuth_deg=float(az) % 360.0,
            elevation_deg=15.0,
            exposure_ms=30.0,
            site_lon_deg=88.3438,
            site_lat_deg=43.312,
            site_alt_m=1801.0,
        )
        for i, az in enumerate(azimuths)
    ]


def test_segment_end_is_inclusive():
    """end 含端：length 与 frames 都必须按含端约定来。"""
    seg = Segment(label="tracking", start=16, end=70, daz_mean=0.222, daz_std=0.003)
    assert seg.length == 55
    assert seg.frames[0] == 16 and seg.frames[-1] == 70
    assert len(seg.frames) == seg.length


def test_azimuth_rates_basic():
    hdrs = make_headers([10.0, 10.2, 10.4, 10.6])
    r = azimuth_rates(hdrs)
    assert r.shape == (3,)
    assert r == pytest.approx([0.2, 0.2, 0.2], abs=1e-9)


def test_azimuth_rates_handles_wraparound():
    hdrs = make_headers([359.8, 0.1, 0.4])
    r = azimuth_rates(hdrs)
    assert r == pytest.approx([0.3, 0.3], abs=1e-9)


def test_azimuth_rates_handles_negative_wraparound():
    hdrs = make_headers([0.2, 359.9, 359.6])
    r = azimuth_rates(hdrs)
    assert r == pytest.approx([-0.3, -0.3], abs=1e-9)


def test_segment_synthetic_three_states():
    """5 帧摆扫 + 30 帧稳定跟踪 + 5 帧摆扫。"""
    az = [0.0]
    for _ in range(5):
        az.append(az[-1] + 6.0)
    for _ in range(30):
        az.append(az[-1] + 0.222)
    for _ in range(5):
        az.append(az[-1] - 6.0)
    segs = segment_sequence(make_headers(az))
    tracking = [s for s in segs if s.label == "tracking"]
    assert len(tracking) == 1
    seg = tracking[0]
    assert seg.start == 5 and seg.end == 35
    assert seg.daz_mean == pytest.approx(0.222, abs=1e-6)
    assert seg.daz_std < 1e-6


def test_segments_cover_whole_sequence_without_gaps():
    az = [0.0]
    for _ in range(4):
        az.append(az[-1] + 6.0)
    for _ in range(20):
        az.append(az[-1] + 0.3)
    segs = segment_sequence(make_headers(az))
    assert segs[0].start == 0
    assert segs[-1].end == len(az) - 1
    for a, b in zip(segs, segs[1:]):
        assert b.start == a.end + 1


def test_segments_are_sorted_and_non_overlapping_and_exhaustive():
    """覆盖全序列、按 start 升序、无重叠——三条性质逐一断言。"""
    az = [0.0]
    for _ in range(5):
        az.append(az[-1] + 6.0)
    for _ in range(20):
        az.append(az[-1] + 0.222)
    for _ in range(5):
        az.append(az[-1] - 6.0)
    for _ in range(15):
        az.append(az[-1] + 0.35)
    segs = segment_sequence(make_headers(az))
    assert len(segs) >= 3
    starts = [s.start for s in segs]
    assert starts == sorted(starts)
    # end 含端：把每段的 frames 摊平后应恰好等于 0..n-1，既无缺口也无重复
    covered = [f for s in segs for f in s.frames]
    assert covered == list(range(len(az)))
    assert sum(s.length for s in segs) == len(az)
    # 相邻标签必须交替，否则说明该合并的段没合并
    labels = [s.label for s in segs]
    assert all(x != y for x, y in zip(labels, labels[1:]))


def test_segment_sequence_rejects_too_short_input():
    """单帧序列没有增量可言，显式报错而非返回 nan 段。"""
    with pytest.raises(ValueError):
        segment_sequence(make_headers([12.0]))


def test_single_frame_tail_segment_has_no_nan_stats():
    """末尾单帧段没有段内增量，统计量必须是有限值。"""
    az = [0.0]
    for _ in range(12):
        az.append(az[-1] + 0.2)
    az.append(az[-1] + 50.0)
    segs = segment_sequence(make_headers(az))
    tail = segs[-1]
    assert tail.start == tail.end == len(az) - 1
    assert np.isfinite(tail.daz_mean) and np.isfinite(tail.daz_std)


def test_tracking_segment_returns_longest_not_first():
    """两个跟踪平台时必须返回更长的那个（此处是第二个）。"""
    az = [0.0]
    for _ in range(12):  # 短跟踪平台，13 帧
        az.append(az[-1] + 0.20)
    for _ in range(5):  # 摆扫隔开
        az.append(az[-1] + 6.0)
    for _ in range(25):  # 长跟踪平台
        az.append(az[-1] + 0.44)
    segs = segment_sequence(make_headers(az))
    tracking = [s for s in segs if s.label == "tracking"]
    assert len(tracking) == 2
    assert tracking[0].length < tracking[1].length  # 第一个更短，取首个即为错
    seg = tracking_segment(make_headers(az))
    assert (seg.start, seg.end) == (tracking[1].start, tracking[1].end)
    assert seg.length > tracking[0].length
    assert seg.daz_mean == pytest.approx(0.44, abs=1e-6)


def test_constant_rate_slew_does_not_merge_into_tracking():
    """匀速摆扫的窗口标准差也是 0，不能因此和跟踪平台并成一段。

    这是"整段标准差"判据存在的理由：逐窗口打标记会让 6.0°/帧 的匀速摆扫和
    0.222°/帧 的跟踪平台连成一片，跟踪段边界随之失效。
    """
    az = [0.0]
    for _ in range(5):
        az.append(az[-1] + 6.0)
    for _ in range(30):
        az.append(az[-1] + 0.222)
    segs = segment_sequence(make_headers(az))
    tracking = [s for s in segs if s.label == "tracking"]
    assert len(tracking) == 1
    assert tracking[0].start == 5
    assert tracking[0].daz_mean == pytest.approx(0.222, abs=1e-6)


def test_short_stable_run_is_not_tracking():
    """5 帧静止段（数据集 A 的 f00-f04）不应被当成跟踪段。"""
    az = [0.0, 0.0, 0.0, 0.0, 0.0]
    for _ in range(20):
        az.append(az[-1] + 0.47)
    segs = segment_sequence(make_headers(az), min_length=10)
    tracking = [s for s in segs if s.label == "tracking"]
    assert len(tracking) == 1
    assert tracking[0].start >= 4


def test_window_rejects_stable_run_shorter_than_window():
    """稳定段短于 window 时不算跟踪段——window 是估计标准差的最少增量数。

    平台恰好 6 个增量，因此把边界钉死在 window=6 与 window=7 之间：window=6 时
    `n_rates >= window` 取等成立而通过，window=7 时落回 slew。用「恰好等于」这一
    对取值，才能同时排除 `>` 与 `>= window - 1` 这两种写错的比较；window=5 与 8
    这样的宽松取值对三者都通过，区分不出来。摆扫故意带抖动，避免摆扫自己形成稳定
    段而与平台相连。
    """
    jitter = [6.0, 5.3, 6.7, 5.6, 6.4]
    az = [0.0]
    for d in jitter:
        az.append(az[-1] + d)
    for _ in range(6):
        az.append(az[-1] + 0.25)
    for d in jitter:
        az.append(az[-1] + d)
    hdrs = make_headers(az)
    kept = segment_sequence(hdrs, window=6, min_length=2)
    assert [s.label for s in kept] == ["slew", "tracking", "slew"]
    assert (kept[1].start, kept[1].end) == (5, 11)
    dropped = segment_sequence(hdrs, window=7, min_length=2)
    assert [s.label for s in dropped] == ["slew"]


def test_two_frame_sequence_is_the_minimum_accepted():
    """2 帧是分段的下界：只有一个增量，不报错，但也构不成跟踪段。"""
    segs = segment_sequence(make_headers([10.0, 10.2]))
    assert len(segs) == 1
    assert segs[0].label == "slew"
    assert (segs[0].start, segs[0].end) == (0, 1)
    assert segs[0].daz_mean == pytest.approx(0.2)
    assert segs[0].daz_std == pytest.approx(0.0)


def test_all_tracking_sequence_is_one_segment_covering_all_frames():
    """整条序列都在稳定跟踪时只有一段，且必须覆盖到末帧。"""
    az = [0.0]
    for _ in range(19):
        az.append(az[-1] + 0.222)
    segs = segment_sequence(make_headers(az))
    assert len(segs) == 1
    assert segs[0].label == "tracking"
    assert (segs[0].start, segs[0].end) == (0, len(az) - 1)
    assert segs[0].length == len(az)


def test_tracking_segment_tie_break_takes_the_earlier_segment():
    """两个跟踪段等长时取靠前的那个——固定 max() 的稳定性，避免行为随实现漂移。"""
    az = [0.0]
    for _ in range(14):
        az.append(az[-1] + 0.20)
    for d in [6.0, 5.3, 6.7, 5.6, 6.4]:
        az.append(az[-1] + d)
    for _ in range(14):
        az.append(az[-1] + 0.44)
    segs = segment_sequence(make_headers(az))
    tracking = [s for s in segs if s.label == "tracking"]
    assert len(tracking) == 2
    assert tracking[0].length == tracking[1].length  # 等长才是这条断言的前提
    seg = tracking_segment(make_headers(az))
    assert (seg.start, seg.end) == (tracking[0].start, tracking[0].end)


def test_no_tracking_segment_raises():
    az = [0.0]
    for i in range(20):
        az.append(az[-1] + (6.0 if i % 2 else -6.0))
    with pytest.raises(NoTrackingSegment):
        tracking_segment(make_headers(az))


def test_dataset_b_tracking_segment_is_f16_to_f70(dataset_b_dir):
    seq = FrameSequence.from_directory(dataset_b_dir)
    seg = tracking_segment(seq.headers)
    assert (seg.start, seg.end) == (16, 70)
    assert seg.length == 55
    assert seg.daz_mean == pytest.approx(0.222, abs=0.005)
    # 实测 0.01442，而非任务书预估的 <0.01：段内首个增量 r16(f16->f17)=+0.119 是
    # 机架停稳前的余量，其余 53 个增量的标准差只有 0.0032。这一帧确实属于跟踪段
    # （.DAT 真值第 1 行的时标正是 f16），所以修正预期值而不是挪动分段边界。
    assert seg.daz_std == pytest.approx(0.0144, abs=0.001)


def test_dataset_a_has_one_long_tracking_segment(dataset_a_dir):
    """数据集 A 的跟踪段实测 f04-f35，钉死实测值而非任务书的宽松下限。

    任务书写的是 `start <= 6`、`length >= 28`，实测是 4 与 32；宽松界会让边界左右
    漂两帧都算通过，回归就看不出来了。实测数字见 docs/reports/measurements.md。
    """
    seq = FrameSequence.from_directory(dataset_a_dir)
    seg = tracking_segment(seq.headers)
    assert (seg.start, seg.end) == (4, 35)
    assert seg.length == 32
    assert seg.daz_mean == pytest.approx(0.4749, abs=0.001)
    assert seg.daz_std == pytest.approx(0.0160, abs=0.001)
