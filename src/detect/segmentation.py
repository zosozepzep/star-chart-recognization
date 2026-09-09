"""全项目统一的源表数据结构。

本模块定义检测、配准和目标跟踪之间共享的 ``SourceTable``。构造时强制
所有列转换为无单位的 NumPy ``float64`` 或 ``int64`` 数组，因为 astropy
``Column`` 可能携带单位（实测 ``area`` 携带 ``Unit("pix2")``，
``elongation`` 携带无量纲单位）；这些单位泄漏到下游会使 ``cKDTree``
抛出 ``UnitConversionError``。本模块另提供两个阈值分割探测函数：
``detect_sources_in_frame``（单帧）与 ``detect_sequence``（多帧，逐帧建背景）。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import operator
from typing import Any

import numpy as np
from photutils.segmentation import SourceCatalog, detect_sources

from src.calib.background import BackgroundModel, model_background
from src.calib.hotpixel import HotCluster, reject_hot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceTable:
    """Represent one frame's detected source measurements."""

    frame: int
    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    peak: np.ndarray
    elongation: np.ndarray
    npix: np.ndarray

    def __post_init__(self) -> None:
        """Normalize columns and reject malformed tables at construction time."""
        # Frozen dataclasses disallow normal assignment; object.__setattr__ is
        # used here so every input column is normalized exactly once.
        columns: tuple[tuple[str, Any, Any], ...] = (
            ("x", self.x, np.float64),
            ("y", self.y, np.float64),
            ("flux", self.flux, np.float64),
            ("peak", self.peak, np.float64),
            ("elongation", self.elongation, np.float64),
            ("npix", self.npix, np.int64),
        )
        normalized: dict[str, np.ndarray] = {}
        for name, value, dtype in columns:
            try:
                array = np.asarray(value, dtype=dtype)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"列 {name} 无法转换为要求的数组类型") from exc
            if array.ndim != 1:
                raise ValueError(f"列 {name} 必须是一维数组")
            normalized[name] = array

        lengths = {name: len(array) for name, array in normalized.items()}
        if len(set(lengths.values())) != 1:
            details = ", ".join(f"{name}={length}" for name, length in lengths.items())
            raise ValueError(f"列长度不一致: {details}")

        for name, array in normalized.items():
            object.__setattr__(self, name, array)

    @property
    def xy(self) -> np.ndarray:
        """Return a new ``(N, 2)`` float64 array in ``[x, y]`` order."""
        return np.column_stack((self.x, self.y))

    def __len__(self) -> int:
        """Return the number of sources in the table."""
        return len(self.x)

    def select(self, mask: Any) -> SourceTable:
        """Select rows using either a boolean mask or an integer index array.

        The boolean-mask and integer-index contracts are both intentional;
        ``brightest`` relies on the integer-index path and must not lose it.
        """
        try:
            selector = np.asarray(mask)
        except (TypeError, ValueError) as exc:
            raise ValueError("选择器必须是布尔掩码或整数索引数组") from exc

        if selector.ndim != 1:
            raise ValueError("选择器必须是一维布尔掩码或整数索引数组")

        if np.issubdtype(selector.dtype, np.bool_):
            if len(selector) != len(self):
                raise ValueError(
                    f"布尔掩码长度不正确: 期望 {len(self)}，实际 {len(selector)}"
                )
        elif np.issubdtype(selector.dtype, np.integer):
            pass
        else:
            raise ValueError("选择器必须是布尔掩码或整数索引数组")

        return SourceTable(
            frame=self.frame,
            x=self.x[selector],
            y=self.y[selector],
            flux=self.flux[selector],
            peak=self.peak[selector],
            elongation=self.elongation[selector],
            npix=self.npix[selector],
        )

    def brightest(self, n: int) -> SourceTable:
        """Return up to ``n`` rows ordered by descending flux."""
        count = operator.index(n)
        if count <= 0:
            return self.select(np.empty(0, dtype=np.int64))
        order = np.argsort(self.flux)[::-1][:count]
        return self.select(order)

    def to_records(self) -> list[dict[str, int | float]]:
        """Return JSON-friendly Python scalar records."""
        return [
            {
                "frame": int(self.frame),
                "x": float(self.x[index]),
                "y": float(self.y[index]),
                "flux": float(self.flux[index]),
                "peak": float(self.peak[index]),
                "elongation": float(self.elongation[index]),
                "npix": int(self.npix[index]),
            }
            for index in range(len(self))
        ]

    @classmethod
    def empty(cls, frame: int = -1) -> SourceTable:
        """Return a fully formed empty source table."""
        return cls(
            frame=frame,
            x=np.empty(0, dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            flux=np.empty(0, dtype=np.float64),
            peak=np.empty(0, dtype=np.float64),
            elongation=np.empty(0, dtype=np.float64),
            npix=np.empty(0, dtype=np.int64),
        )


# photutils SourceCatalog.to_table() 取的六列。顺序与 SourceTable 的字段一一对应。
# 实测 photutils 2.0.2 的类型混搭：前四列是 Column 且 unit=None，elongation 是
# 无量纲 Quantity，area 是带 Unit("pix2") 的 Quantity 且 dtype 为 float64。
# 六者都是 ndarray 子类，所以必须逐列 np.asarray 转换，否则单位会漏进 cKDTree。
_COLUMNS = ("xcentroid", "ycentroid", "segment_flux", "max_value", "elongation", "area")

# Per-column target dtypes, paired positionally with _COLUMNS. `area` becomes int64
# because photutils hands it back as float64 pixel counts carrying Unit("pix2").
_COLUMN_DTYPES = (
    np.float64,
    np.float64,
    np.float64,
    np.float64,
    np.float64,
    np.int64,
)


def detect_sources_in_frame(
    image: np.ndarray,
    model: BackgroundModel,
    *,
    n_sigma: float = 5.0,
    npixels: int = 5,
    hot_clusters: list[HotCluster] | None = None,
    reject_radius: float = 8.0,
    frame: int = -1,
) -> SourceTable:
    """单帧阈值分割源探测；默认值与 src/config/default.yaml 的 detect 节一致。

    阈值**相对扣背景后的残差图**：``BackgroundModel.threshold(n_sigma)`` 返回的是
    ``n_sigma * rms``，把它直接拿去和原始图比较是错的——实测这样做在 dataset B
    第 30 帧的 3σ 上给出 52488 个源（正确值 169）。因此分割图与测光数据图都用
    ``model.subtract(image)``；``segment_flux`` 要的正是扣背景后的流量。

    阈值必须逐像素（``rms`` 是一整幅图）而非退化成标量，否则 dataset B 的
    实测计数（5σ→95、4σ→116、3σ→169）复现不了。

    ``detect_sources`` 在一无所获时返回 ``None`` 而不是空分割图，并发出
    ``NoDetectionsWarning``；此处返回空表并**让该警告照常传播**，因为
    "这一帧什么都没探到"是调用方需要看到的信息。
    """
    residual = model.subtract(image)
    segments = detect_sources(residual, model.threshold(n_sigma), npixels=npixels)
    if segments is None:
        logger.info("frame %d: no sources above %.1f sigma", frame, n_sigma)
        return SourceTable.empty(frame=frame)

    table = SourceCatalog(residual, segments).to_table(list(_COLUMNS))
    # 逐列转换是硬性的：astropy 的 Column 与 Quantity 都是 np.ndarray 子类，
    # 原样透传会把 Unit("pix2") 之类的单位带进下游的 cKDTree 并抛
    # UnitConversionError（见 tests 里的 test_naive_column_merge_really_does_raise）。
    columns = [
        np.asarray(table[name], dtype=dtype)
        for name, dtype in zip(_COLUMNS, _COLUMN_DTYPES)
    ]
    x, y, flux, peak, elongation, npix = columns

    # Guard against photutils emitting NaN centroids for degenerate segments (e.g. a
    # segment whose weights sum to zero). NOT dead code: it never fires on the real
    # frames measured so far (all 116 rows of dataset B frame 30 are finite), but a
    # NaN centroid would silently poison cKDTree queries and every downstream match.
    # Do not delete it just because the real-data path never triggers it.
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.all():
        logger.warning(
            "frame %d: dropping %d source(s) with non-finite centroid",
            frame,
            int((~finite).sum()),
        )
        x, y, flux, peak, elongation, npix = (
            column[finite] for column in (x, y, flux, peak, elongation, npix)
        )

    if hot_clusters:
        # reject_hot expects an (N, 2) [x, y] array; scipy's center_of_mass hands back
        # (row, col) = (y, x), so the column order here is load-bearing.
        keep = reject_hot(
            np.column_stack((x, y)), list(hot_clusters), radius=reject_radius
        )
        x, y, flux, peak, elongation, npix = (
            column[keep] for column in (x, y, flux, peak, elongation, npix)
        )

    logger.info(
        "frame %d: %d sources at %.1f sigma (npixels=%d)", frame, len(x), n_sigma, npixels
    )
    return SourceTable(
        frame=frame,
        x=x,
        y=y,
        flux=flux,
        peak=peak,
        elongation=elongation,
        npix=npix,
    )


def detect_sequence(
    sequence,
    frames,
    *,
    n_sigma: float = 5.0,
    npixels: int = 5,
    hot_clusters: list[HotCluster] | None = None,
    reject_radius: float = 8.0,
    background_kwargs: dict[str, Any] | None = None,
) -> dict[int, SourceTable]:
    """对指定帧逐帧探测，返回 ``{帧号: SourceTable}``。

    背景是**逐帧无条件建模**的，没有共享模型的开关。物理理由：同一序列内曝光
    可能变化——实测 dataset B 的 80 帧只有两个曝光值，f00–f07 为 80.0 ms、
    f08–f79 为 30.0 ms，分界恰在 f08；共用一个背景模型会让其中一段整体偏置。
    曝光取值用 ``sequence.headers[i].exposure_ms``（``FrameSequence`` 没有
    ``header()`` 方法，``headers`` 是 ``FrameHeader`` dataclass 的列表）。

    这是整条流水线最贵的一步：实测 4096² 单帧约 1.99 s，其中背景建模占 1.48 s。
    """
    kwargs = dict(background_kwargs or {})
    results: dict[int, SourceTable] = {}
    for f in frames:
        index = operator.index(f)
        image = sequence.image(index)
        model = model_background(image, **kwargs)
        results[index] = detect_sources_in_frame(
            image,
            model,
            n_sigma=n_sigma,
            npixels=npixels,
            hot_clusters=hot_clusters,
            reject_radius=reject_radius,
            frame=index,
        )
    logger.info("detected sources in %d frame(s)", len(results))
    return results
