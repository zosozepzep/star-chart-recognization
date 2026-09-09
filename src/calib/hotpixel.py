"""热像素识别。

判据：跨全序列位置固定 且 数值方差极小的像素——
    median > 背景 + n_sigma * 背景 sigma
    (max - min) / median < flat_ratio
恒星与目标因视场移动只在少数帧经过某像素，不可能同时满足两条。

必须按簇（连通域）建模：实测热像素以 2-10 个相邻像素成团出现，
按单像素处理会在簇边缘留下残点，进而被探测阶段当成高置信目标。

实测结论（数据集 B 帧 16-45 与数据集 A 帧 5-35，相隔约 4.5 个月的两次独立观测）：
flat_ratio=0.10 时两个数据集都给出同样 5 个簇、位置互差不超过 0.25 px。
按输出次序（npix 降序，同 npix 时 cx 升序）：
    npix 9 (327.0, 3210.1)、8 (244.0, 3173.5)、2 (312.0, 3206.5)、
    1 (0, 0)、1 (2192.0, 3223.0)
其中 (0, 0) 中位值约 26982，正是全图统计里那个 max=26978 的来源。
两个数据集都是 seed 21 像素、dilate=1 后 mask 57 像素。完整实测表见
docs/reports/measurements.md 的热像素节。

flat_ratio 取 0.10 而非最初设想的 0.05，是因为 0.05 会漏掉 (312, 3206.5)——它实测
0.0743，但中位值 134（背景 6）且在两个数据集同一像素复现，是无可争议的真缺陷。
0.10 位于一段实测平台的中部（数据集 B 扫描结果）：
    0.05 -> 4 簇（漏 312）   0.07~0.15 -> 5 簇（稳定平台）
    0.20 -> 18 簇            0.30 -> 242 簇（开始收进真实恒星）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation, center_of_mass, label
from scipy.spatial import cKDTree

from src.calib.background import background_stats

logger = logging.getLogger(__name__)

# 逐像素统计的分块上限（字节）。分块只为控内存，不改变结果：median / max / min
# 都是逐像素独立的，按行切开与整块计算逐位一致（见 test_pixel_stats_chunking_is_exact）。
_MAX_CHUNK_BYTES = 256 * 1024 * 1024

# build_from_sequence 累积立方体所用的 dtype 上限。
_UINT16_MAX = 65535


@dataclass(frozen=True)
class HotCluster:
    cx: float
    cy: float
    npix: int
    median_value: float
    flat_ratio: float


@dataclass
class HotPixelMap:
    mask: np.ndarray
    clusters: list[HotCluster] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_pixels": int(self.mask.sum()),
            "n_clusters": len(self.clusters),
            "clusters": [
                {
                    "cx": c.cx,
                    "cy": c.cy,
                    "npix": c.npix,
                    "median_value": c.median_value,
                    "flat_ratio": c.flat_ratio,
                }
                for c in self.clusters
            ],
        }


def _pixel_stats(
    cube: np.ndarray, *, max_chunk_bytes: int = _MAX_CHUNK_BYTES
) -> tuple[np.ndarray, np.ndarray]:
    """逐像素时间轴中位数与极差，按行分块以限制峰值内存。

    不能直接 `np.median(cube, axis=0)`：np.median 内部要 partition，会整份复制输入。
    30 帧 4096^2 的 uint16 立方体本身就是 1 GB，整块走会再多 1 GB 峰值。
    极差在 float64 上做减法而非原 dtype：uint16 里 max >= min 恒成立所以不会绕，
    但显式转换后这份代码对任何输入 dtype 都是安全的。
    """
    n, ny, nx = cube.shape
    # 预填 NaN 而非 np.empty：分块循环若漏写某几行，np.empty 留下的是上一次分配的
    # 残留内存——数量级往往看着像正常像素值，判据照样跑得出"合理"结果，缺陷完全无声。
    # 填 NaN 让任何漏写立刻显形（NaN 参与比较恒为假，且与整块参考比对必然不相等）。
    # 代价是每次调用多一遍 4096^2 float64（约 134 MB）的写入，相对实测 1.76 GiB 峰值可忽略。
    med = np.full((ny, nx), np.nan, dtype=np.float64)
    span = np.full((ny, nx), np.nan, dtype=np.float64)

    row_bytes = max(int(n) * int(nx) * cube.dtype.itemsize, 1)
    rows = max(int(max_chunk_bytes) // row_bytes, 1)
    for y0 in range(0, ny, rows):
        y1 = min(y0 + rows, ny)
        chunk = cube[:, y0:y1, :]
        med[y0:y1] = np.median(chunk, axis=0)
        span[y0:y1] = chunk.max(axis=0).astype(np.float64) - chunk.min(axis=0).astype(
            np.float64
        )
    return med, span


def build_hotpixel_map(
    cube: np.ndarray,
    *,
    bkg_median: float,
    bkg_sigma: float,
    n_sigma: float = 5.0,
    flat_ratio: float = 0.10,
    dilate: int = 1,
) -> HotPixelMap:
    """由帧立方体构建热像素图；默认值与 src/config/default.yaml 的 hotpixel 节一致。

    刻意**不**把 cube 转成 float64：调用方传进来的往往已是 uint16 的 1 GB 立方体，
    转 float64 会变成 8 GB。逐像素统计交给 _pixel_stats 分块处理。
    """
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"需要 (n_frames, ny, nx) 立方体，收到 {cube.shape}")
    if cube.shape[0] == 0:
        raise ValueError("立方体不含任何帧，无法做时间轴统计")

    med, span = _pixel_stats(cube)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(med > 0, span / med, np.inf)

    # 两条判据都是**严格**比较，各由一个用例看守：
    # `med > thr`：恰好等于阈值的像素不算热像素
    #     (test_pixel_exactly_at_brightness_threshold_is_excluded，合成——真数据没有
    #      任何像素中位值恰在阈值上，改成 >= 活过全部实数据用例)
    # `ratio < flat_ratio`：极差比恰好等于 flat_ratio 的像素不算热像素
    #     (test_dataset_b_flat_ratio_boundary_is_strict——数据集 B 的 x=245,y=3174
    #      中位值 100.0、极差 10.0，ratio 恰为 0.100000，改成 <= 会让相邻那个 8 像素
    #      簇变 9 像素、质心从 (244.0, 3173.5) 移到 (244.111, 3173.556))
    seed = (med > bkg_median + n_sigma * bkg_sigma) & (ratio < flat_ratio)

    # 4 连通（label 的默认结构元），刻意不用 8 连通：热像素成团源于读出电路上物理
    # 相邻的单元一起漂，物理相邻即共边。只共角的两个像素更可能是两个独立缺陷，并成
    # 一簇会把质心推到两者之间的正常像素上，reject_hot 就按错的中心去剔点。
    # 实测数据集 A/B 两种连通性都给 5 簇（那里的簇本来就共边连通），故此约定由
    # test_diagonal_neighbours_are_separate_clusters 用合成用例看守。
    labels, n = label(seed)
    clusters: list[HotCluster] = []
    for k in range(1, n + 1):
        sel = labels == k
        cy, cx = center_of_mass(sel)
        clusters.append(
            HotCluster(
                cx=float(round(cx, 3)),
                cy=float(round(cy, 3)),
                npix=int(sel.sum()),
                median_value=float(np.median(med[sel])),
                flat_ratio=float(np.median(ratio[sel])),
            )
        )
    clusters.sort(key=lambda c: (-c.npix, c.cx, c.cy))

    mask = binary_dilation(seed, iterations=dilate) if dilate > 0 else seed
    logger.info(
        "hot pixel map: %d clusters, %d masked pixels (n_sigma=%.1f flat_ratio=%.3f)",
        len(clusters),
        int(mask.sum()),
        n_sigma,
        flat_ratio,
    )
    return HotPixelMap(mask=mask, clusters=clusters)


def _to_uint16(frame: np.ndarray, index: int) -> np.ndarray:
    """把单帧安全地降到 uint16，任何越界都显式报错。

    有符号→无符号转换是静默的：-1 会变成 65535，若该像素在多数帧都是负值，
    绕完之后 median 极高、span 为 0，**恰好同时满足两条热像素判据**，
    于是凭空造出一个传感器缺陷。NaN / inf 的转换结果更是未定义，而 `nan < 0`
    为假，光查负值兜不住，所以先查有限性。
    实测数据集 A 帧 5-35 与数据集 B 帧 16-45 共 55 帧无任何负像素，
    这条守卫当前不会误伤真数据；它防的是换数据集之后的无声崩坏。
    """
    frame = np.asarray(frame)
    if not np.all(np.isfinite(frame)):
        raise ValueError(f"第 {index} 帧含非有限像素（NaN 或 inf），无法转 uint16")
    if frame.min() < 0:
        raise ValueError(
            f"第 {index} 帧含负值像素（最小 {float(frame.min())}），"
            "转 uint16 会绕成大正数并伪造出热像素簇"
        )
    if frame.max() > _UINT16_MAX:
        raise ValueError(
            f"第 {index} 帧最大像素 {float(frame.max())} 超出 uint16 上限 {_UINT16_MAX}"
        )
    return frame.astype(np.uint16)


def build_from_sequence(
    sequence,
    frame_indices,
    *,
    n_sigma: float = 5.0,
    flat_ratio: float = 0.10,
    dilate: int = 1,
    max_frames: int = 40,
) -> HotPixelMap:
    """从 FrameSequence 抽取若干帧构建热像素图。

    max_frames 限制内存：4096^2 float64 单帧 128 MB，40 帧 float64 约 5 GB 峰值过高，
    因此按 uint16 累积（40 帧约 1.3 GB），再由 _pixel_stats 分块求逐像素统计量。
    背景基准取首帧的全图统计——热像素只占几个像素，对 sigma-clip 中位数无影响。
    """
    picked = list(frame_indices)[:max_frames]
    if not picked:
        raise ValueError("frame_indices 为空")

    first = np.asarray(sequence.image(picked[0]), dtype=np.float64)
    cube = np.empty((len(picked),) + first.shape, dtype=np.uint16)
    cube[0] = _to_uint16(first, picked[0])
    for i, idx in enumerate(picked[1:], start=1):
        cube[i] = _to_uint16(sequence.image(idx), idx)

    stats = background_stats(first)
    logger.info(
        "hot pixel input: %d frames, bkg median=%.4f sigma=%.4f",
        len(picked),
        stats["median"],
        stats["std"],
    )
    return build_hotpixel_map(
        cube,
        bkg_median=stats["median"],
        bkg_sigma=stats["std"],
        n_sigma=n_sigma,
        flat_ratio=flat_ratio,
        dilate=dilate,
    )


def reject_hot(
    xy: np.ndarray, clusters: list[HotCluster], *, radius: float = 8.0
) -> np.ndarray:
    """返回布尔保留掩膜：距任一热像素簇中心不超过 radius 的点被剔除。

    xy 为 (N, 2) float64 的 [x, y] 点集，返回长度为 N 的 bool 数组。

    空点集必须显式早退。`np.array([])` / `[]` / `np.empty((0,))` 过 atleast_2d 后
    形状都是 **(1, 0)**：size 为 0 但 len() 是 1，于是有簇时 cKDTree.query 直接
    ValueError，无簇时更隐蔽——会为 0 个点返回长度 1 的掩膜，静默错位下游数组。
    """
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    if xy.size == 0:
        return np.zeros(0, dtype=bool)
    if not clusters:
        return np.ones(len(xy), dtype=bool)
    centers = np.array([[c.cx, c.cy] for c in clusters], dtype=np.float64)
    dist, _ = cKDTree(centers).query(xy)
    return dist > radius
