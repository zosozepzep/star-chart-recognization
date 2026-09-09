"""孔径测光与一阶矩质心精化。

分割给出的质心受阈值影响：``photutils`` 的 ``xcentroid``/``ycentroid`` 只对
**阈值以上**的像素求矩，所以同一颗星在 5σ 与 3σ 下会落在不同位置，暗源尤甚
——阈值切掉的是外围翼，剩下的核心又被分割边界削成不对称形状。配准、跟踪和
定轨都要求跨帧、跨阈值一致的坐标，因此本模块在固定窗口内对扣背景后的图像
重新求一阶矩，并用统一孔径重测流量，把测量与探测阈值解耦。

坐标一律 0 基、``x`` = 列、``y`` = 行；点集统一为 ``(N, 2)`` 的 float64
``[x, y]``。图像数组按 NumPy 惯例索引为 ``image[y, x]``。
"""

from __future__ import annotations

import logging

import numpy as np

from src.detect.segmentation import SourceTable

logger = logging.getLogger(__name__)


def cutout(
    image: np.ndarray, x: float, y: float, box: int
) -> tuple[np.ndarray, int, int]:
    """Return a ``box``-sized patch centred on ``(x, y)`` plus its origin.

    Returns ``(patch, x0, y0)`` where ``patch = image[y0:y1, x0:x1]``, so the
    absolute coordinate of ``patch[j, i]`` is ``(x0 + i, y0 + j)``. The window is
    clipped at the image borders, so an edge cutout is smaller than ``box``;
    callers must use the returned origin rather than assuming ``box``.

    ``box`` must be odd. With ``half = box // 2`` the span ``xc - half`` ..
    ``xc + half`` is inclusive, i.e. ``2 * half + 1`` pixels, so an even ``box``
    would silently yield ``box + 1``（实测 box=8 给出 (9, 9)、box=10 给出
    (11, 11)）。The window feeds centroiding and PSF fitting, where a silent
    off-by-one becomes a half-pixel systematic nobody can trace later, and every
    call site already passes an odd box (``psf.box`` = 9, this module's default
    9, and the 13 that ``aperture_flux`` derives), so rejecting even values
    costs nothing.
    """
    if box % 2 == 0:
        raise ValueError(f"box 必须是奇数，实际收到 {box}；偶数会静默返回 {box + 1} 像素的窗口")
    if box <= 0:
        raise ValueError(f"box 必须是正奇数，实际收到 {box}")

    half = box // 2
    # round() before int() so that x=20.6 centres on column 21, not 20.
    xc = int(round(float(x)))
    yc = int(round(float(y)))
    x0 = max(0, xc - half)
    y0 = max(0, yc - half)
    x1 = min(image.shape[1], xc + half + 1)
    y1 = min(image.shape[0], yc + half + 1)
    return image[y0:y1, x0:x1], x0, y0


def aperture_flux(
    image_sub: np.ndarray, xy: np.ndarray, *, radius: float = 5.0
) -> np.ndarray:
    """Sum a fixed circular aperture around each point of ``xy``.

    ``image_sub`` must already be background-subtracted; the sum is taken as-is
    with no further background estimate.

    The aperture is a **hard-edge pixel mask**, not subpixel-weighted: a pixel
    is included in full when its centre satisfies
    ``(xx - x)**2 + (yy - y)**2 <= radius**2`` and excluded in full otherwise.
    There is no partial-area weighting at the aperture boundary.

    实测后果：``radius=5``、σ=1.6 的高斯上本函数给出 15970.93，而解析总流量是
    ``amp * 2π * σ²`` = 16084.95，**偏低 0.709%**。这个亏损是有意接受的：

    - 同一个孔径同样地作用在恒星和目标上，所以亏损对所有源是共同的，在
      Task 24 的**差分**星等定标里整体抵消（定标量是 star 与 target 的流量比，
      共同的乘性因子约掉）；
    - ``photutils`` 的精确面积孔径要付的运行时间这条流水线负担不起
      （Task 6 实测单帧已经 1.99 s）。

    亏损方向与量级由 ``test_aperture_is_hard_edged_and_the_deficit_is_quantified``
    钉住，所以这里的说明不会与实现漂移。

    The window is ``ceil(radius) * 2 + 3`` pixels（radius=5 时为 13，高于容纳
    直径 10 所需的 11），always odd, and clipped at the borders — a source at
    the very edge therefore loses part of its aperture but still returns a
    finite positive sum（实测 (1.5, 1.5) 处给出 12937.9）。
    """
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("点集必须是 (N, 2) 的 [x, y] 数组")

    box = int(np.ceil(radius)) * 2 + 3
    radius_sq = float(radius) ** 2
    out = np.empty(len(points), dtype=np.float64)
    for index, (x, y) in enumerate(points):
        patch, x0, y0 = cutout(image_sub, x, y, box)
        # Absolute pixel-centre coordinates of the patch, so the aperture is
        # measured against the true source position rather than the patch centre
        # (they differ whenever the window is clipped at a border).
        yy, xx = np.mgrid[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius_sq
        out[index] = patch[mask].sum()
    return out


def centroid_of_mass(
    image_sub: np.ndarray, xy: np.ndarray, *, box: int = 9
) -> np.ndarray:
    """Refine each point to the first-moment centroid of its local window.

    Returns a new ``(N, 2)`` float64 array; the input is never modified.

    Weights are ``np.clip(patch, 0.0, None)``——负值截零，避免噪声把质心拉出
    窗口。扣背景后的残差图里负像素是常态（背景估计的涨落两侧对称），带符号的
    权重会让一阶矩的分母被抵消、分子被拉向随机方向。实测有噪声窗口（amp=60
    信号、σ=20 噪声、81 像素里 22 个为负）：截零给出 x=31.8153，不截零给出
    x=31.4911，差 0.324 px，而截零的那个更靠近真值。无噪声高斯上两者逐字节
    相同，所以这条特性只有噪声 fixture 能守卫（见
    ``test_negative_clip_matters_on_a_noisy_fixture``）。

    If a window's clipped weights sum to zero or less（全非正的窗口），the point
    is **left at its input value**: keeping the segmentation's position is more
    useful downstream than returning NaN or silently snapping to the window
    centre.
    """
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("点集必须是 (N, 2) 的 [x, y] 数组")

    out = points.copy()
    for index in range(len(out)):
        x, y = out[index]
        patch, x0, y0 = cutout(image_sub, x, y, box)
        weights = np.clip(patch, 0.0, None)
        total = weights.sum()
        if total <= 0.0:
            # Leave the seed untouched; see the docstring.
            continue
        yy, xx = np.mgrid[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]]
        out[index, 0] = (weights * xx).sum() / total
        out[index, 1] = (weights * yy).sum() / total
    return out


def remeasure(
    image_sub: np.ndarray,
    table: SourceTable,
    *,
    radius: float = 5.0,
    box: int = 9,
) -> SourceTable:
    """Return a new table with centroid, flux and peak all re-measured.

    ``elongation`` and ``npix`` are morphology from the segmentation and carry
    over unchanged. The input table is not modified, and every column of the
    result is an independent copy — mixing views and copies inside one
    constructor call would be a trap for the downstream tasks that share
    ``SourceTable``.

    ``peak`` is the maximum of a ``box``-sized cutout centred on the
    **re-measured** centroid, i.e. a **local** maximum of the
    background-subtracted image — not the segmentation's ``max_value`` and not
    ``image_sub.max()``. Local is the required definition because ``peak`` must
    describe the same aperture as ``flux``: 实测两源图（1000-amp 在 (20, 20)、
    300-amp 在 (45, 45)）全局最大是 1000.0，而暗源处的局部峰值是 300.0，
    差 3.3 倍；用全局最大会把每一个源的 peak 都写成最亮那个源的值。
    """
    if len(table) == 0:
        return SourceTable.empty(frame=table.frame)

    xy = centroid_of_mass(image_sub, table.xy, box=box)
    flux = aperture_flux(image_sub, xy, radius=radius)

    peak = np.empty(len(xy), dtype=np.float64)
    for index, (x, y) in enumerate(xy):
        patch, _, _ = cutout(image_sub, x, y, box)
        peak[index] = patch.max()

    logger.info(
        "frame %d: remeasured %d source(s) with r=%.1f px, box=%d",
        table.frame,
        len(xy),
        radius,
        box,
    )
    # Every column is an explicit copy: xy[:, 0] and xy[:, 1] are views into the
    # same array, so handing them over directly would make out.x and out.y share
    # a base with each other and with `xy`.
    return SourceTable(
        frame=table.frame,
        x=xy[:, 0].copy(),
        y=xy[:, 1].copy(),
        flux=flux,
        peak=peak,
        elongation=table.elongation.copy(),
        npix=table.npix.copy(),
    )
