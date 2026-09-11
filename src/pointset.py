# -*- coding: utf-8 -*-
"""点集入参的统一校验：`(N, 2)` 的 `[x, y]`。

全局约束规定点集一律是 `(N, 2)` `float64` 的 `[x, y]`。原先各模块都写
`np.asarray(xy, dtype=np.float64).reshape(-1, 2)`，共十处。`reshape` 会
**静默接受任何元素数为偶数的形状**，这是终审第 1 项要修的洞：

    np.array([[1., 2., 3., 4.],        ->   [[1., 2.],
              [5., 6., 7., 8.]])            [3., 4.],
                                            [5., 6.],
                                            [7., 8.]]

一个 `(2, 4)` 的输入（两个点、每点四个量，例如误把 `x, y, flux, snr` 整表传进来）
静默变成**四个点**，且第二个点 `(3, 4)` 是把两个不同量当成了坐标。结果不报错、
量级正常、下游一路算到底。`(N, 3)` 同理：`(2, 3)` 变成三个点。

不能一律要求 `ndim == 2`：扁平的 `(2N,)` 输入在本项目里是**合法且有意义**的
（`[x0, y0, x1, y1, ...]`），有现成调用方依赖它。所以判据是**逐形状**的：

- `(N, 2)`  —— 原样接受（唯一的标准形）
- `(2N,)`   —— 接受，按 `[x, y]` 交替解读
- `(0,)` / `(0, 2)` 等空输入 —— 接受，归一到 `(0, 2)`
- 其余（`(N, 4)`、`(N, 3)`、三维及以上）—— 抛 `ValueError`

`(2, 2)` 在两条规则下解读相同（第一行就是第一个点），无歧义。
`(1, 2)` 是一个点，不是两个。
"""
from __future__ import annotations

import numpy as np


def as_xy(points, *, name: str = "点集") -> np.ndarray:
    """把点集入参归一成 `(N, 2)` `float64`，形状非法时抛中文 ``ValueError``。

    ``name`` 只进错误消息，用于指出是哪个实参错了——十个调用点共用一条消息时，
    「点集形状非法」远不如「dst_xy 形状非法」有用。
    """
    arr = np.asarray(points, dtype=np.float64)

    if arr.size == 0:
        # 空输入统一成 (0, 2)：调用方的 len()、切片与 KDTree 都按第 0 维工作。
        return np.zeros((0, 2), dtype=np.float64)

    if arr.ndim == 2:
        if arr.shape[1] == 2:
            return arr
        raise ValueError(
            f"{name} 形状非法: {arr.shape}，二维输入的第二维必须是 2（[x, y]）。"
            f"若本意是传 {arr.shape[0]} 个点的多列表格，请先取出 x/y 两列"
        )

    if arr.ndim == 1:
        if arr.shape[0] % 2 == 0:
            # 扁平 [x0, y0, x1, y1, ...]，本项目允许。
            return arr.reshape(-1, 2)
        raise ValueError(
            f"{name} 形状非法: {arr.shape}，一维输入按 [x, y] 交替解读，"
            f"元素数必须是偶数"
        )

    raise ValueError(
        f"{name} 形状非法: {arr.shape}，点集必须是 (N, 2) 的 [x, y] "
        f"或扁平的 (2N,)"
    )
