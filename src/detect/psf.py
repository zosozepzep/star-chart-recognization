"""二维椭圆高斯 PSF 亚像素拟合。

一阶矩质心（``src.detect.centroid.centroid_of_mass``）在窗口内对所有权重求平均，
邻近源、不对称的翼和残余背景都会把它拉偏；PSF 拟合把整颗星的形状写进模型，
中心是模型参数而不是加权平均，因此在有噪声的真实帧上更稳。

**拟合失败必须显式标记。** ``least_squares`` 几乎总能返回一组数——纯噪声窗口上
它照样交回一个"中心"和一个"σ"，看起来和真星的输出毫无区别。若失败被静默地
当成成功，坏拟合会带着一个看似合理的亚像素坐标一路流进配准与定轨，事后无从
追溯。因此本模块的每次拟合都带一个 ``success`` 标志与一个可复算的
``residual_ratio``，失败的拟合把形状字段全部写成 NaN（不是 0，0 会被下游当成
真实测量值），并由 ``fit_table`` 从源表中剔除。

坐标一律 0 基、``x`` = 列、``y`` = 行；图像数组按 NumPy 惯例索引为 ``image[y, x]``。
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import warnings

import numpy as np
from scipy.optimize import least_squares

from src.detect.centroid import cutout
from src.detect.segmentation import SourceTable

logger = logging.getLogger(__name__)

# FWHM = 2 * sqrt(2 * ln 2) * sigma，高斯的解析换算，不是经验系数。
FWHM_PER_SIGMA = 2.3548200450309493


@dataclass(frozen=True)
class GaussianFit:
    """One source's 2D elliptical gaussian fit result.

    ``x``/``y`` are absolute 0-based image coordinates (x = column, y = row).
    ``fwhm_px`` is ``FWHM_PER_SIGMA * sqrt(sigma_x * sigma_y)``.
    ``residual_ratio`` is ``|residual|.sum() / |patch - background|.sum()``.
    When ``success`` is False every shape field is NaN and ``x``/``y`` carry the
    input seed unchanged.

    ``sigma_x``/``sigma_y``/``theta_deg`` 只有作为**一组**才有意义，单独比较是
    错的：``(σx, σy, θ)`` 参数化是退化的——``(3.0, 1.4, 30°)`` 与
    ``(1.4, 3.0, -60°)`` 描述**同一个椭圆**，而实测本实现对植入前者的 fixture
    返回的正是后者。所以：

    - **不要跨拟合比较 ``sigma_x``**，也不要假设 ``sigma_x >= sigma_y``；
      要长短轴就用 ``max(...)``/``min(...)``。
    - 要一个可比较的标量就用 ``fwhm_px``：几何平均 ``sqrt(σx·σy)`` 在上述
      重标记下**逐字节不变**（因为乘法可交换），是唯一免疫退化的形状字段，
      也是下游任务（视宁度统计、目标判别）实际消费的那一个。
    """

    x: float
    y: float
    amplitude: float
    sigma_x: float
    sigma_y: float
    theta_deg: float
    background: float
    fwhm_px: float
    residual_ratio: float
    success: bool


def _failed(x: float, y: float, residual_ratio: float = np.nan) -> GaussianFit:
    """Return a failure marker that keeps the seed position and NaNs the shape.

    NaN rather than 0.0 is deliberate: a zero amplitude or zero FWHM is a value
    downstream code would happily average into a seeing statistic, whereas NaN
    propagates visibly and is filtered explicitly by ``median_fwhm``.
    """
    return GaussianFit(
        x=float(x),
        y=float(y),
        amplitude=np.nan,
        sigma_x=np.nan,
        sigma_y=np.nan,
        theta_deg=np.nan,
        background=np.nan,
        fwhm_px=np.nan,
        residual_ratio=float(residual_ratio),
        success=False,
    )


def _model(
    xx: np.ndarray,
    yy: np.ndarray,
    amp: float,
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    theta: float,
    bkg: float,
) -> np.ndarray:
    """Evaluate a rotated elliptical gaussian plus a constant background.

    ``theta`` is in radians and measures the ``sigma_x`` axis from +x, with
    ``xr = dx cos t + dy sin t`` and ``yr = -dx sin t + dy cos t``.
    """
    dx = xx - cx
    dy = yy - cy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    xr = dx * cos_t + dy * sin_t
    yr = -dx * sin_t + dy * cos_t
    return bkg + amp * np.exp(-(xr**2 / (2.0 * sx**2) + yr**2 / (2.0 * sy**2)))


def fit_gaussian2d(
    image: np.ndarray,
    x: float,
    y: float,
    *,
    box: int = 9,
    max_nfev: int = 200,
    max_residual_ratio: float = 0.55,
) -> GaussianFit:
    """Fit one 2D elliptical gaussian in a ``box``-sized window around ``(x, y)``.

    ``image`` may be either the raw or the background-subtracted frame — the
    model carries its own constant background term. Seeds always come from 5σ
    segmentation; this function never searches.

    **判据是什么。** ``residual_ratio = |residual|.sum() / |patch - bkg|.sum()``。
    分子是整个窗口上的绝对残差和，主要由光子噪声贡献，几乎不随源的亮度变化；
    分母随流量线性增长。所以这个比值实际表现为 **≈1/SNR，不是形状检验**。
    三组实测支持这个判断（dataset B 第 30 帧、95 个 5σ 源、box=9）：

    - 它与亮度相关而与形状无关：实测 ``corr(log flux, rr) = -0.665``，分流量档的
      中位数从 flux<200 的 0.529 单调降到 >3000 的 0.273；
    - 窗口越大它越**差**：同样 95 颗星在 0.35 判据下 box=9 通过 38 个（40%）、
      box=11 只通过 22 个（23%），因为更宽的窗口只往分子里加噪声像素、
      分母一点不涨。真正的形状检验会随像素变多而变好；
    - 实测 ``corr(rr, fwhm) = -0.171``，0.55 判据下通过组（87 个）的中位 FWHM
      是 3.4380 px、被拒组（8 个）是 3.2360 px，差 6%——而流量中位数差了一个
      数量级。这个判据不是在挑形状不同的星，是在挑更亮的星。

    **因此它是对"已探测源"的合理性过滤器，绝不是探测器。** 位置永远来自 5σ
    分割；本函数只负责把灾难性失败的拟合标出来。

    **默认阈值 0.55 的来历（裁决 33）。** 原值 0.35 从未对真星测过，实测它在
    上述帧上只接受 38/95（40%），也就是静默丢掉本项目自己 60% 的 5σ 探测；
    0.55 保留 **87/95**，同时仍拒掉灾难性失败区——300 个 64² 空天窗口在 box=9
    上的 residual_ratio 中位数实测 0.9187，而最差的真星是 0.6504。

    **阈值依赖窗口尺寸。** 同一批星在 0.35 判据下 box=9 通过率 40%、
    box=11 只有 23%（0.55 判据下分别是 92% 与 81%）；任何改 ``box`` 的调用方
    都必须重新测量保留率，不能假定默认阈值随之成立。

    **诚实的局限（记录而非修补）：** 在 box=9 上这个判据原则上无法把信号与
    噪声完全分开——真星的 rr 实测跨 0.1807–0.6504，300 个空天窗口跨
    0.1808–1.0022，两个区间**重叠**，所以不存在任何"接受全部真星、
    拒绝全部空窗"的阈值（0.55 下 300 个空窗里有 19 个会被接受）。
    这可以接受，恰恰因为输入永远不是空天区；但要说清楚，
    不要暗示存在干净的分界。
    """
    patch, x0, y0 = cutout(image, x, y, box)
    # 9 是 7 参数模型所需的像素数下限（自由度 >= 2），**不是**对窗口形状的断言。
    # 生产路径上永远不会触发：cutout 在两个维度上对称裁剪，box=9 在图角上仍给
    # 25 个像素、在图边上给 45 个。唯一可达的情形是中心落在图外——那时切片是空的。
    if patch.size < 9:
        logger.debug("fit at (%.2f, %.2f) skipped: patch size %d", x, y, patch.size)
        return _failed(x, y)

    yy, xx = np.mgrid[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]]
    yy = yy.astype(np.float64)
    xx = xx.astype(np.float64)

    bkg0 = float(np.median(patch))
    amp0 = float(patch.max() - bkg0)
    if not np.isfinite(amp0) or amp0 <= 0.0:
        amp0 = max(abs(bkg0), 1.0)
    # σ 的上界取"拟合出的 FWHM 不得超过窗口本身"，即 sigma <= box / FWHM_PER_SIGMA。
    # 这不是随手挑的数：FWHM 大于窗口的高斯在该窗口内与一道背景斜坡无法区分，
    # 于是纯噪声窗口会被拟合成一个巨大、平坦的"源"并给出很小的 residual_ratio。
    # 实测（box=11、σ_noise=3.8、40 个种子）：上界取 box 时 rr 跨 0.3807–0.9917，
    # 其中 seed 24 给 0.3807（σ=11.0 顶到上界、amp 9.3）、seed 26 给 0.4827——
    # 两者都会被 0.55 的判据**错误接受**；上界改成 box / FWHM_PER_SIGMA 后
    # 同样 40 个种子跨 0.9080–0.9931，噪声全部被拒。
    # 真星不受影响：dataset B 第 30 帧 95 个源实测最大 σ 为 2.607，而 box=9 的
    # 上界是 3.822，95 个源里 0 个顶到上界，保留率与 FWHM 中位数逐位不变。
    sigma_max = float(box) / FWHM_PER_SIGMA
    sigma0 = min(max(box / 4.0, 1.0), sigma_max * 0.9)
    p0 = np.array([amp0, float(x), float(y), sigma0, sigma0, 0.0, bkg0])

    half = box / 2.0
    lower = np.array(
        [0.0, float(x) - half, float(y) - half, 0.3, 0.3, -np.pi, bkg0 - 10.0 * abs(amp0) - 1.0]
    )
    upper = np.array(
        [
            10.0 * abs(amp0) + 1.0,
            float(x) + half,
            float(y) + half,
            sigma_max,
            sigma_max,
            np.pi,
            bkg0 + 10.0 * abs(amp0) + 1.0,
        ]
    )
    p0 = np.clip(p0, lower + 1e-9, upper - 1e-9)

    def residuals(params: np.ndarray) -> np.ndarray:
        amp, cx, cy, sx, sy, theta, bkg = params
        return (_model(xx, yy, amp, cx, cy, sx, sy, theta, bkg) - patch).ravel()

    # least_squares 在退化窗口上会发 RuntimeWarning/OptimizeWarning；这些不是
    # 调用方能处理的信息，失败已经由 success 标志表达，所以在此局部静音。
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = least_squares(
                residuals, p0, bounds=(lower, upper), max_nfev=max_nfev
            )
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.debug("fit at (%.2f, %.2f) raised %s", x, y, exc)
        return _failed(x, y)

    amp, cx, cy, sx, sy, theta, bkg = (float(v) for v in result.x)
    residual = residuals(result.x)
    denominator = float(np.abs(patch - bkg).sum())
    if denominator <= 0.0:
        return _failed(x, y)
    ratio = float(np.abs(residual).sum() / denominator)

    fwhm = FWHM_PER_SIGMA * float(np.sqrt(sx * sy))
    ok = (
        bool(result.success)
        and amp > 0.0
        and np.isfinite(fwhm)
        and ratio <= max_residual_ratio
    )
    if not ok:
        return _failed(x, y, residual_ratio=ratio)

    return GaussianFit(
        x=cx,
        y=cy,
        amplitude=amp,
        sigma_x=sx,
        sigma_y=sy,
        theta_deg=float(np.rad2deg(theta)),
        background=bkg,
        fwhm_px=fwhm,
        residual_ratio=ratio,
        success=True,
    )


def fit_table(
    image_sub: np.ndarray,
    table: SourceTable,
    *,
    box: int = 9,
    max_nfev: int = 200,
    max_residual_ratio: float = 0.55,
) -> tuple[SourceTable, list[GaussianFit]]:
    """Fit every row of ``table`` and return the successful subset plus its fits.

    只保留 ``success`` 的源：失败的拟合没有可用的坐标，把它留在表里等于把一个
    编造的位置交给配准。返回的表与返回的列表**逐行对应**，长度相等。

    ``x``/``y`` are replaced by the fitted centres; ``flux``, ``peak``,
    ``elongation`` and ``npix`` carry over from the selected input rows unchanged.

    实现依赖 ``SourceTable.select`` **返回全新数组**这一性质：被带过来的四列
    直接交给新表的构造函数，而 ``select`` 里每一列都是花式索引的结果
    （因此必然是副本），所以下游改写输出表的列不会波及输入表。这一点由
    ``test_fit_table_drops_failures_and_keeps_order`` 钉住——若 ``select``
    哪天改成返回视图，输入表会被静默污染。

    注：``SourceTable`` 是 **frozen** dataclass，所以任务书 Ruling 37 描述的
    ``out.x = ...`` 属性赋值在本仓库里根本不成立（会抛 ``FrozenInstanceError``）。
    这里改为显式构造一张新表——语义相同、且不必绕过 frozen 保护。
    """
    if len(table) == 0:
        return SourceTable.empty(frame=table.frame), []

    keep: list[int] = []
    fits: list[GaussianFit] = []
    for index, (px, py) in enumerate(table.xy):
        fit = fit_gaussian2d(
            image_sub,
            float(px),
            float(py),
            box=box,
            max_nfev=max_nfev,
            max_residual_ratio=max_residual_ratio,
        )
        if fit.success:
            keep.append(index)
            fits.append(fit)

    # dtype=int 是必需的：裸 np.array([]) 是 float64，空的 keep 会让 select 抛
    # IndexError: arrays used as indices must be of integer (or boolean) type。
    selected = table.select(np.array(keep, dtype=int))
    xs = np.array([f.x for f in fits], dtype=np.float64)
    ys = np.array([f.y for f in fits], dtype=np.float64)
    out = SourceTable(
        frame=selected.frame,
        x=xs,
        y=ys,
        flux=selected.flux,
        peak=selected.peak,
        elongation=selected.elongation,
        npix=selected.npix,
    )

    logger.info(
        "frame %d: fitted %d/%d source(s) with box=%d, max_residual_ratio=%.2f",
        table.frame,
        len(fits),
        len(table),
        box,
        max_residual_ratio,
    )
    return out, fits


def median_fwhm(fits: list[GaussianFit]) -> float:
    """Return the median finite ``fwhm_px``, or NaN when there is none.

    NaN rather than 0.0 for the empty case: a zero seeing value would silently
    poison any statistic built on top of it, while NaN propagates visibly.
    """
    values = np.array([f.fwhm_px for f in fits], dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))
