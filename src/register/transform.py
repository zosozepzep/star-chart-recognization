"""4 参数相似变换（各向同性缩放 + 旋转 + 平移）。

为什么不用 6 参数仿射：望远镜配准的物理自由度只有场旋转与平移，比例尺由光学
固定。多给两个自由度会让最小二乘去拟合噪声，在只有几十颗匹配星时表现为
病态解——本项目早期用 RANSAC 仿射就是这样失败的。用一对反射（镜像）点集对比
两个模型可以看到这一点被量化：4 参数相似变换无法吸收行列式变号，rms 停在
20*sqrt(5) ≈ 44.72；同一份数据的 6 参数仿射拟合能把 rms 压到 ~1.6e-14——多两个
自由度确实能拟合出物理上不该出现的形变。
"""
from __future__ import annotations

import numpy as np

from src.pointset import as_xy

DEFAULT_CENTER = (2047.5, 2047.5)


def similarity_matrix(
    *,
    scale: float = 1.0,
    rotation_deg: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0,
    center: tuple[float, float] = DEFAULT_CENTER,
) -> np.ndarray:
    """构造齐次矩阵：先绕 center 缩放旋转，再平移 (tx, ty)。"""
    t = np.deg2rad(rotation_deg)
    a = scale * np.cos(t)
    b = scale * np.sin(t)
    cx, cy = center
    return np.array(
        [
            [a, -b, cx - a * cx + b * cy + tx],
            [b, a, cy - b * cx - a * cy + ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def apply_transform(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    xy = as_xy(xy, name="xy")
    if xy.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    return xy @ matrix[:2, :2].T + matrix[:2, 2]


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """最小二乘拟合把 src 映到 dst 的相似变换。

    已知局限：当对应点退化（例如全部重合）时，`lstsq`（`rcond=None`）仍会返回一个
    有限的最小范数解，且该解在自身输入上的 rms 接近 0——看起来像完美拟合，实际上
    套到其他点上是任意的。这里不做秩检验：通用最小二乘已经覆盖了这种情形，而
    Task 10 的 `solve_pair` 会先用 `min_inliers` 门限过滤到足够多的匹配点才调用
    本函数，真正的防护在那一层；逐对做秩检验只会在流水线第二耗时的循环里白花
    运行时间去防一个下游不会发生的场景。
    """
    src = as_xy(src, name="src")
    dst = as_xy(dst, name="dst")
    if len(src) != len(dst):
        raise ValueError(f"点数不一致: {len(src)} vs {len(dst)}")
    n = len(src)
    if n < 2:
        raise ValueError("相似变换拟合至少需要 2 个对应点")

    A = np.zeros((2 * n, 4), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)
    A[0::2, 0] = src[:, 0]
    A[0::2, 1] = -src[:, 1]
    A[0::2, 2] = 1.0
    A[1::2, 0] = src[:, 1]
    A[1::2, 1] = src[:, 0]
    A[1::2, 3] = 1.0
    b[0::2] = dst[:, 0]
    b[1::2] = dst[:, 1]

    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, tx, ty = p
    return np.array([[a, -bb, tx], [bb, a, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def decompose(matrix: np.ndarray) -> dict:
    """从齐次矩阵读出 scale / rotation_deg / tx / ty。

    `tx`/`ty` 是矩阵平移列的原始读数，**不是** `similarity_matrix` 的输入实参本身。
    `similarity_matrix(tx=..., ty=..., center=c)` 会把绕 `center` 的旋转折叠进平移
    列；`decompose` 原样读出该列，因此两者只有在 `center=(0, 0)` 时才相等。例如
    绕 `DEFAULT_CENTER` 转 30°、tx=ty=0，`decompose` 会读出约 (+1298, -749) 的
    平移——这不是 bug，是矩阵本身携带的绕心旋转位移，也正是 Task 10 需要的原始
    累积矩阵平移量。因此 `similarity_matrix(**decompose(M))` 要重建出 M，必须传
    `center=(0, 0)`。

    `rotation_deg` 落在闭区间 `[-180, 180]`（两端都可达，例如 180.0 与 -180.0
    都会原样返回，不会被规整成对方），因为 `arctan2` 在 ±180° 处的分支边界正好
    落在两个端点上；不要把返回范围当成半开区间 `[-180, 180)` 来判断。
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    a, bb = matrix[0, 0], matrix[1, 0]
    return {
        "scale": float(np.hypot(a, bb)),
        "rotation_deg": float(np.rad2deg(np.arctan2(bb, a))),
        "tx": float(matrix[0, 2]),
        "ty": float(matrix[1, 2]),
    }


def residuals(matrix: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    diff = apply_transform(matrix, src) - as_xy(dst, name="dst")
    return np.hypot(diff[:, 0], diff[:, 1])


def rms(matrix: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    """均方根残差；空输入返回 `nan`（故意），不是 0.0。

    一个静默的 0.0 在 Task 10 的 `rms_max_px` 报告里会被误读成完美拟合；`nan` 让
    调用方能区分「没有匹配点」和「拟合得极好」。
    """
    r = residuals(matrix, src, dst)
    return float(np.sqrt(np.mean(r ** 2))) if r.size else float("nan")


def invert(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(matrix, dtype=np.float64))
