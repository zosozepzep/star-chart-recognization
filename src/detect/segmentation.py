"""全项目统一的源表数据结构。

本模块定义检测、配准和目标跟踪之间共享的 ``SourceTable``。构造时强制
所有列转换为无单位的 NumPy ``float64`` 或 ``int64`` 数组，因为 astropy
``Column`` 可能携带单位（实测 ``area`` 携带 ``Unit("pix2")``，
``elongation`` 携带无量纲单位）；这些单位泄漏到下游会使 ``cKDTree``
抛出 ``UnitConversionError``。探测函数将在后续任务中加入本模块。
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Any

import numpy as np


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
