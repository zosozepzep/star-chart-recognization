"""机架状态分段。

只读 FITS 头的 AZIMUTH 与 DATE-OBS，不碰像素，因此绝不会被噪声干扰——
这是整条流水线中最可靠的一步，放在最前面。

判据：把逐帧方位角增量序列切成"整段标准差低于阈值"的极大连续段。之所以看
**整段**的标准差、而不是逐个滑动窗口打标记，是因为匀速摆扫在窗口内的标准差
同样为 0；只要允许相邻两个稳定窗口的标记连成一片，紧邻的摆扫平台就会和跟踪
平台并成一段。整段判据天然排除了这种合并：两者速率不同，一旦跨过交界，整段
标准差立刻超阈值。

段的取舍用两个长度门槛：window 是"够不够估计一个标准差"的最少增量数，
min_length 是跟踪段的最短帧数。这样不需要给 |ΔAZ| 设人为下限：数据集 A 开头
的 5 帧静止段自然被 min_length=10 排除，而不必区分"静止"与"慢速跟踪"。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.dataio.fits_loader import FrameHeader


class NoTrackingSegment(RuntimeError):
    """序列中不存在满足条件的稳定跟踪段。"""


@dataclass(frozen=True)
class Segment:
    label: str
    start: int
    end: int
    daz_mean: float
    daz_std: float

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def frames(self) -> list[int]:
        return list(range(self.start, self.end + 1))


def azimuth_rates(headers: list[FrameHeader]) -> np.ndarray:
    """逐帧方位角增量（°/帧），已把差值折到 (-180, 180]。"""
    az = np.array([h.azimuth_deg for h in headers], dtype=np.float64)
    d = np.diff(az)
    return (d + 180.0) % 360.0 - 180.0


def _stable_runs(rates: np.ndarray, std_threshold: float) -> list[tuple[int, int]]:
    """把增量序列切成极大连续段，每段整体标准差低于阈值。

    贪心地从左向右扩张：能并进当前段就并，一超阈值就在此断开。返回的
    (start, end_inclusive) 覆盖全部增量且互不重叠。
    """
    runs: list[tuple[int, int]] = []
    n = rates.size
    i = 0
    while i < n:
        best = i
        j = i
        while j < n and float(rates[i : j + 1].std()) < std_threshold:
            best = j
            j += 1
        runs.append((i, best))
        i = best + 1
    return runs


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """把布尔序列压成 (start, end_inclusive, value) 的连续段列表。"""
    out: list[tuple[int, int, bool]] = []
    if mask.size == 0:
        return out
    start = 0
    for i in range(1, mask.size + 1):
        if i == mask.size or mask[i] != mask[start]:
            out.append((start, i - 1, bool(mask[start])))
            start = i
    return out


def segment_sequence(
    headers: list[FrameHeader],
    *,
    window: int = 5,
    std_threshold: float = 0.05,
    min_length: int = 10,
) -> list[Segment]:
    """把序列切成交替的 tracking / slew 段，覆盖全部帧且互不重叠。"""
    n_frames = len(headers)
    if n_frames < 2:
        raise ValueError(f"分段至少需要 2 帧: n={n_frames}")

    rates = azimuth_rates(headers)

    # 帧级标签：增量 k 属于帧 k 与 k+1 之间。稳定增量段 [a, b] 覆盖帧 [a, b+1]。
    frame_tracking = np.zeros(n_frames, dtype=bool)
    for a, b in _stable_runs(rates, std_threshold):
        n_rates = b - a + 1
        if n_rates >= window and (n_rates + 1) >= min_length:
            frame_tracking[a : b + 2] = True

    segments: list[Segment] = []
    for a, b, value in _runs(frame_tracking):
        # 段内增量是 rates[a:b]（右端开，因为 rates[b] 跨出本段）。单帧段没有
        # 段内增量，退一格取通向它的那个增量，避免对空数组求均值得到静默 nan。
        if b > a:
            chunk = rates[a:b]
        else:
            k = min(a, rates.size - 1)
            chunk = rates[k : k + 1]
        segments.append(
            Segment(
                label="tracking" if value else "slew",
                start=a,
                end=b,
                daz_mean=float(chunk.mean()),
                daz_std=float(chunk.std()),
            )
        )
    return segments


def tracking_segment(headers: list[FrameHeader], **kwargs) -> Segment:
    """返回最长的稳定跟踪段。"""
    candidates = [s for s in segment_sequence(headers, **kwargs) if s.label == "tracking"]
    if not candidates:
        raise NoTrackingSegment(
            "未找到稳定跟踪段；序列可能全为摆扫，或 segment.std_threshold_deg 过严"
        )
    return max(candidates, key=lambda s: s.length)
