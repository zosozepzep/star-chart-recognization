from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.calib.background import model_background
from src.dataio.fits_loader import FrameSequence
from src.detect.psf import (
    FWHM_PER_SIGMA,
    GaussianFit,
    fit_gaussian2d,
    fit_table,
    median_fwhm,
)
from src.detect.segmentation import SourceTable, detect_sources_in_frame

SIGMA_PSF = 1.6


def make_image(sources, ny=64, nx=64, sigma=SIGMA_PSF, background=0.0):
    """Return a noiseless image with circular gaussians planted at ``(cx, cy, amp)``."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    img = np.full((ny, nx), float(background), dtype=np.float64)
    for cx, cy, amp in sources:
        img += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return img


def make_elliptical(cx, cy, amp, sigma_x, sigma_y, theta_deg, ny=64, nx=64):
    """Plant one rotated elliptical gaussian.

    旋转约定必须与 ``src.detect.psf`` 的模型一致：``theta`` 是 ``sigma_x`` 主轴
    相对 +x 轴的角度，``xr = dx cosθ + dy sinθ``、``yr = -dx sinθ + dy cosθ``。
    """
    yy, xx = np.mgrid[0:ny, 0:nx]
    t = np.deg2rad(theta_deg)
    dx = xx - cx
    dy = yy - cy
    xr = dx * np.cos(t) + dy * np.sin(t)
    yr = -dx * np.sin(t) + dy * np.cos(t)
    return amp * np.exp(-(xr**2 / (2 * sigma_x**2) + yr**2 / (2 * sigma_y**2)))


def make_table(seeds, frame=3, flux=1000.0):
    """Build a SourceTable from integer seed positions ``[(x, y), ...]``."""
    xs = np.array([s[0] for s in seeds], dtype=np.float64)
    ys = np.array([s[1] for s in seeds], dtype=np.float64)
    n = len(seeds)
    return SourceTable(
        frame=frame,
        x=xs,
        y=ys,
        flux=np.full(n, flux, dtype=np.float64),
        peak=np.full(n, 100.0, dtype=np.float64),
        elongation=np.full(n, 1.05, dtype=np.float64),
        npix=np.full(n, 40, dtype=np.int64),
    )


# --------------------------------------------------------------------------
# fit_gaussian2d — noiseless recovery
# --------------------------------------------------------------------------


def test_fit_recovers_subpixel_centre():
    """无噪声高斯上，拟合必须把整数种子精化到真实亚像素中心。

    控制器实测：真值 (32.37, 29.62)、box=11、种子 (32, 30) -> (32.37000, 29.62000)，
    偏差 0.00000（机器精度）。abs=0.02 因此有极大余量。

    **坐标刻意不对称**：把返回的 (x, y) 互换会给出 (29.62, 32.37)，与真值相差
    2.75 px，本测试会响亮失败（Step 5 变异 1）。不要把 fixture "整理"成对称坐标。
    """
    img = make_image([(32.37, 29.62, 1000.0)])
    fit = fit_gaussian2d(img, 32.0, 30.0, box=11)
    assert fit.success is True
    assert fit.x == pytest.approx(32.37, abs=0.02)
    assert fit.y == pytest.approx(29.62, abs=0.02)


def test_fit_recovers_amplitude_sigma_and_fwhm():
    """振幅、σ 与 FWHM 必须一并复原。

    控制器实测：amp=800、σ=2.1、box=13 -> amp 800.0000、sigma_x 2.10000、
    fwhm 4.94512（= FWHM_PER_SIGMA * 2.1），机器精度。
    """
    img = make_image([(30.0, 30.0, 800.0)], sigma=2.1)
    fit = fit_gaussian2d(img, 30.0, 30.0, box=13)
    assert fit.success is True
    assert fit.amplitude == pytest.approx(800.0, rel=0.05)
    assert fit.sigma_x == pytest.approx(2.1, rel=0.05)
    assert fit.sigma_y == pytest.approx(2.1, rel=0.05)
    assert fit.fwhm_px == pytest.approx(FWHM_PER_SIGMA * 2.1, rel=0.08)


def test_fit_recovers_elliptical_shape():
    """椭圆 σ=(3.0, 1.4)、θ=30° 必须被复原为同一个椭圆。

    (σx, σy, θ) 参数化是**退化**的：实测本仓库的 ``_model`` 在
    (3.0, 1.4, 30°) 与 (1.4, 3.0, -60°) 两组参数上给出**同一张图**
    （max|a-b| = 1.99e-13，纯浮点噪声）。因此长短轴哪一个被叫做 ``sigma_x``
    完全取决于优化器从哪一侧收敛，是实现细节。

    **本实现实测返回 sx=3.0000、sy=1.4000、θ=30.0000°**（与任务书
    Ruling 35 记录的 sx=1.40/sy=3.00/θ=-60° 是同一个椭圆的另一种标记；
    控制器那次测量的初值与本实现不同）。所以这里比较 ``max``/``min``
    而不是逐字段比较：这条断言在两种标记下都成立，而
    ``fit.sigma_x == 3.0`` 会在一个同样**正确**的实现上失败。
    """
    img = make_elliptical(32.0, 32.0, 900.0, 3.0, 1.4, 30.0)
    fit = fit_gaussian2d(img, 32.0, 32.0, box=17)
    assert fit.success is True
    assert max(fit.sigma_x, fit.sigma_y) == pytest.approx(3.0, rel=0.12)
    assert min(fit.sigma_x, fit.sigma_y) == pytest.approx(1.4, rel=0.12)


def test_fwhm_is_invariant_under_the_sigma_theta_relabelling():
    """裁决 35：``fwhm_px`` 用几何平均，因此对 σ/θ 的重标记免疫。

    (3.0, 1.4, 30°) 与 (1.4, 3.0, -60°) 是同一个椭圆（实测两张图逐像素相差
    1.99e-13），所以 ``sigma_x`` 单独毫无意义——它取决于优化器落在哪一侧。
    ``fwhm_px = FWHM_PER_SIGMA * sqrt(sx * sy)`` 在重标记下不变（乘法可交换），
    这就是下游任务只消费 ``fwhm_px`` 的原因。

    本测试给两种标记各造一张图分别拟合，断言两次的 ``fwhm_px`` 相等
    **且**等于 ``FWHM_PER_SIGMA * sqrt(3.0 * 1.4)``——这样无论实现落在哪一侧
    都成立。这也是 Step 5 变异 2（``fwhm_px`` 改成 ``sx``）的守卫：几何平均是
    2.04939σ，而两张图实测分别给 sx=3.0000 与 sx=1.4000，相差 1.46 倍与
    1.46 倍，都远超 rel=0.05；换句话说变异 2 在**任何**一侧标记上都会被抓到。
    """
    want = FWHM_PER_SIGMA * np.sqrt(3.0 * 1.4)

    fit_a = fit_gaussian2d(
        make_elliptical(32.0, 32.0, 900.0, 3.0, 1.4, 30.0), 32.0, 32.0, box=17
    )
    fit_b = fit_gaussian2d(
        make_elliptical(32.0, 32.0, 900.0, 1.4, 3.0, -60.0), 32.0, 32.0, box=17
    )
    assert fit_a.success is True and fit_b.success is True
    assert fit_a.fwhm_px == pytest.approx(want, rel=0.05)
    assert fit_b.fwhm_px == pytest.approx(want, rel=0.05)
    # 重标记不变性：两次拟合的 fwhm_px 必须一致到 1e-9。
    assert fit_a.fwhm_px == pytest.approx(fit_b.fwhm_px, rel=1e-9)


def test_fit_recovers_background():
    """常数背景必须被拟合出来，而不是被吸进振幅里。

    控制器实测：background=25.0、box=13 -> 25.00000（机器精度），abs=3.0 余量充足。
    """
    img = make_image([(30.0, 30.0, 800.0)], sigma=2.1, background=25.0)
    fit = fit_gaussian2d(img, 30.0, 30.0, box=13)
    assert fit.success is True
    assert fit.background == pytest.approx(25.0, abs=3.0)


# --------------------------------------------------------------------------
# fit_gaussian2d — 噪声与失败
# --------------------------------------------------------------------------


def test_fit_on_a_noisy_source_stays_subpixel():
    """有噪声时中心仍必须落在 0.15 px 内。

    fixture：真值 (32.4, 31.7)、amp=200、σ_noise=3.8、seed 5。
    实测中心 (32.36291, 31.72020)，偏差 0.0371/0.0202，相对 abs=0.15 约 4 倍余量
    （任务书记录 (32.36320, 31.72032)，第 4 位小数起的差异来自本实现的 σ 上界
    与初值，见 ``test_noise_cannot_be_fitted_as_a_broad_flat_source``）。
    residual_ratio 实测 0.0738，远低于 0.55 的默认判据（任务书记录 0.1088，
    同一原因）。
    """
    truth_x, truth_y = 32.4, 31.7
    rng = np.random.default_rng(5)
    img = rng.normal(0.0, 3.8, size=(64, 64))
    img += make_image([(truth_x, truth_y, 200.0)])
    fit = fit_gaussian2d(img, 32.0, 32.0)
    assert fit.success is True
    assert fit.x == pytest.approx(truth_x, abs=0.15)
    assert fit.y == pytest.approx(truth_y, abs=0.15)
    assert fit.residual_ratio < 0.55


def test_pure_noise_is_marked_failed():
    """纯噪声窗口必须被显式标记为失败，而不是返回一个看似合理的拟合。

    fixture：seed 9、σ_noise=3.8、box=11，实测 residual_ratio 0.950085
    （任务书记录 0.955858）。

    这**不是**运气好挑的种子：40 个种子在 box=11 上的 residual_ratio 实测跨
    0.9080–0.9931（中位数 0.9488；任务书记录 0.9072–0.9866），全都远高于任何
    在讨论中的阈值。

    这里显式传 ``max_residual_ratio=0.35``（而非默认 0.55）是有意的：本测试
    钉的是**机制**——超过判据就必须 success=False——而 0.9501 在两个取值下
    都会被拒。默认值本身由 ``test_signature_defaults_match_config`` 与
    ``test_dataset_b_frame30_retention_and_fwhm`` 守卫。
    """
    rng = np.random.default_rng(9)
    img = rng.normal(0.0, 3.8, size=(64, 64))
    fit = fit_gaussian2d(img, 32.0, 32.0, box=11, max_residual_ratio=0.35)
    assert fit.success is False
    assert fit.residual_ratio > 0.35


def test_noise_cannot_be_fitted_as_a_broad_flat_source():
    """σ 上界必须是 ``box / FWHM_PER_SIGMA``，否则纯噪声能拟合成一道背景斜坡。

    这是本实现相对任务书原码的一处**必要修正**，实测发现：σ 的上界若放到
    ``box``（即允许 FWHM 达到窗口的 2.35 倍），优化器会把纯噪声窗口拟合成一个
    巨大而平坦的"源"——它在窗口内与一道线性背景无法区分，于是残差被压得很小。
    实测 box=11、σ_noise=3.8：

        上界 = box              -> 40 个种子 rr 跨 0.3807–0.9917，
                                   其中 seed 24 给 0.3807（σx 顶到 11.0、amp 9.3）、
                                   seed 26 给 0.4827 —— **两者都会被 0.55 错误接受**
        上界 = box / 2          -> 跨 0.8578–0.9919
        上界 = box / FWHM_PER_SIGMA -> 跨 0.9080–0.9931，噪声全部被拒（本实现）

    真星完全不受影响：dataset B 第 30 帧 95 个源实测最大 σ 为 2.607，而 box=9
    的上界是 3.822，95 个源里 **0 个**顶到上界；保留率（0.35→38、0.55→87）与
    accepted 组的中位 FWHM（3.4380 px）在四种上界下**逐位相同**。

    本测试直接钉住那两个曾经漏网的种子：它们的 rr 必须仍在 0.55 以上。
    """
    for seed in (9, 24, 26):
        rng = np.random.default_rng(seed)
        img = rng.normal(0.0, 3.8, size=(64, 64))
        fit = fit_gaussian2d(img, 32.0, 32.0, box=11)
        assert fit.success is False, f"seed {seed} 的纯噪声被接受了"
        assert fit.residual_ratio > 0.55, (
            f"seed {seed} 的纯噪声 rr={fit.residual_ratio:.4f}，σ 上界疑似被放宽"
        )
    # 40 个种子的整体分布也钉住下界，避免只有这三个被特判修好。
    ratios = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        img = rng.normal(0.0, 3.8, size=(64, 64))
        ratios.append(fit_gaussian2d(img, 32.0, 32.0, box=11).residual_ratio)
    ratios = np.array(ratios)
    assert ratios.min() > 0.85, f"40 种子最小 rr {ratios.min():.4f}，实测应为 0.9080"
    assert ratios.min() == pytest.approx(0.9080, abs=0.01)
    assert ratios.max() == pytest.approx(0.9931, abs=0.01)


def test_corner_source_fits_despite_the_clipped_window():
    """裁决 36：贴角的源窗口被裁小，但仍必须拟合成功。

    实测 ``cutout(img, 1.5, 1.5, 9)`` 给出 **(7, 7) = 49 个像素**、原点 (0, 0)
    ——不是任务书写的 (6, 6)：``cutout`` 先做 ``int(round(1.5))``，而 Python 的
    ``round`` 是四舍六入五取偶，``round(1.5) = 2``，于是窗口是 ``0 .. 6``。
    49 个像素远多于 7 个参数，所以拟合必须成功而不是走 ``patch.size < 9``。
    """
    from src.detect.centroid import cutout

    img = make_image([(1.5, 1.5, 1000.0)])
    patch, x0, y0 = cutout(img, 1.5, 1.5, 9)
    # 窗口确实被裁了（否则本测试没在测裁剪路径），且形状钉死而不是只说"比 9 小"。
    assert patch.shape == (7, 7)
    assert (x0, y0) == (0, 0)
    assert patch.size == 49

    fit = fit_gaussian2d(img, 1.5, 1.5, box=9)
    assert fit.success is True
    assert fit.x == pytest.approx(1.5, abs=0.15)
    assert fit.y == pytest.approx(1.5, abs=0.15)


def test_centre_outside_the_image_fails_without_raising():
    """裁决 36：中心落在图外必须返回 success=False，**不得抛异常**。

    这是 ``patch.size < 9`` 唯一真正可达的场景：``cutout`` 在 x=5000、64×64 上
    给出的切片是空的（size 0）。生产路径上的边角裁剪永远给不出这种窗口——
    两个维度都是对称裁剪，box=9 在角上仍有 25 个像素。9 是 7 参数模型的
    像素数下限，不是对窗口形状的断言。
    """
    img = make_image([(32.0, 32.0, 1000.0)])
    fit = fit_gaussian2d(img, 5000.0, 32.0)
    assert isinstance(fit, GaussianFit)
    assert fit.success is False
    assert not np.isfinite(fit.amplitude)
    assert not np.isfinite(fit.fwhm_px)
    # 失败的拟合把种子位置原样带回，方便调用方定位。
    assert fit.x == pytest.approx(5000.0)
    assert fit.y == pytest.approx(32.0)


# --------------------------------------------------------------------------
# median_fwhm
# --------------------------------------------------------------------------


def test_median_fwhm_of_empty_is_nan():
    """空列表必须给 nan，而不是 0.0——0 会被下游当成一个真实的 FWHM。"""
    value = median_fwhm([])
    assert isinstance(value, float)
    assert np.isnan(value)


def test_median_fwhm_ignores_non_finite_entries():
    """失败的拟合带的是 NaN 字段，必须被 ``isfinite`` 过滤掉。

    三个成功拟合（fwhm 3.0/4.0/5.0）加两个失败拟合，中位数必须是 4.0；
    若不过滤 NaN，``np.median`` 会返回 nan。
    """
    img = make_image([(32.0, 32.0, 1000.0)])
    failed = fit_gaussian2d(img, 5000.0, 32.0)
    assert failed.success is False

    good = [
        GaussianFit(
            x=1.0,
            y=1.0,
            amplitude=100.0,
            sigma_x=1.0,
            sigma_y=1.0,
            theta_deg=0.0,
            background=0.0,
            fwhm_px=value,
            residual_ratio=0.1,
            success=True,
        )
        for value in (3.0, 4.0, 5.0)
    ]
    assert median_fwhm(good + [failed, failed]) == pytest.approx(4.0)
    # 全部非有限时退回 nan。
    assert np.isnan(median_fwhm([failed, failed]))


# --------------------------------------------------------------------------
# fit_table
# --------------------------------------------------------------------------


def _noisy_two_sources(seed=5):
    """两个亮源 + 噪声；(5, 5) 处刻意留空，用来触发失败剔除。"""
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, 3.8, size=(64, 64))
    img += make_image([(20.3, 30.4, 1000.0), (45.2, 18.1, 600.0)])
    return img


def test_fit_table_drops_failures_and_keeps_order():
    """失败的源必须被剔除，保留下来的必须保持原顺序，且输入表不得被改动。

    三个种子 (20, 30)、(45, 18)、(5, 5)：前两个是亮源，第三个是空天区
    （实测 residual_ratio 0.7309，正确被拒；任务书记录 0.9284，量级一致）。

    裁决 37：``fit_table`` 用 ``table.select(...)`` 的返回值构造输出，
    而 ``select`` 交回的是全新数组——因此输入表必须完好，输出的 x 也不能
    是输入 x 的同一个对象。此前没有任何测试钉住这一点。

    注：``SourceTable`` 是 **frozen** dataclass，所以任务书 Ruling 37 描述的
    ``out.x = ...`` 属性赋值在本仓库里会抛 ``FrozenInstanceError``；实现改为
    显式构造新表。本测试断言的是同一份契约，与内部写法无关。
    """
    img = _noisy_two_sources()
    tbl = make_table([(20.0, 30.0), (45.0, 18.0), (5.0, 5.0)], frame=11)
    saved = {
        name: getattr(tbl, name).copy()
        for name in ("x", "y", "flux", "peak", "elongation", "npix")
    }

    out, fits = fit_table(img, tbl)

    assert isinstance(out, SourceTable)
    assert len(out) == 2
    assert len(fits) == 2
    assert all(f.success for f in fits)
    assert out.frame == 11
    # 顺序保持：先 (20.3, 30.4) 再 (45.2, 18.1)。
    assert out.x[0] == pytest.approx(20.3, abs=0.15)
    assert out.y[0] == pytest.approx(30.4, abs=0.15)
    assert out.x[1] == pytest.approx(45.2, abs=0.15)
    assert out.y[1] == pytest.approx(18.1, abs=0.15)
    # 未重测的列原样带过。
    assert out.elongation.tolist() == [1.05, 1.05]
    assert out.npix.dtype == np.int64

    # 裁决 37：输入表必须完好。
    assert tbl.x[0] == 20.0
    for name, before in saved.items():
        assert np.array_equal(getattr(tbl, name), before), f"输入表的 {name} 被改动了"
    # 裁决 37：输出的 x 不能是输入 x 的同一个对象。
    assert out.x is not tbl.x
    assert not np.shares_memory(out.x, tbl.x)

    # 带过来的四列同样不得与输入表共享内存。这里必须用**行为断言**而不是
    # `base is None`（Task 7 的教训）：`select` 的花式索引已经造了副本，
    # 而 `SourceTable.__post_init__` 的 `np.asarray` 在连续 float64/int64 上是
    # 恒等操作，所以 `base` 天然为 None，对"是否共享输入内存"完全是瞎的。
    out.flux[0] = -1.0
    out.peak[0] = -2.0
    out.elongation[0] = -3.0
    out.npix[0] = -4
    assert tbl.flux[0] == pytest.approx(1000.0), "out.flux 与输入表共享内存"
    assert tbl.peak[0] == pytest.approx(100.0), "out.peak 与输入表共享内存"
    assert tbl.elongation[0] == pytest.approx(1.05), "out.elongation 与输入表共享内存"
    assert tbl.npix[0] == 40, "out.npix 与输入表共享内存"


def test_fit_table_empty_keep_path():
    """裁决 37：``keep == []`` 这条路径必须给出一张形状完整的空表。

    ``np.array(keep, dtype=int)`` 的 ``dtype=int`` 是必需的——裸的
    ``np.array([])`` 是 float64，``select`` 会抛
    ``IndexError: arrays used as indices must be of integer (or boolean) type``。
    本测试就是拦住有人把 ``dtype=int`` 删掉的那道守卫（Step 5 变异 6）。
    """
    rng = np.random.default_rng(9)
    img = rng.normal(0.0, 3.8, size=(64, 64))
    tbl = make_table([(20.0, 30.0), (45.0, 18.0), (5.0, 5.0)], frame=17)

    out, fits = fit_table(img, tbl)

    assert len(out) == 0
    assert len(fits) == 0
    assert out.frame == 17
    assert out.xy.shape == (0, 2)
    for name in ("x", "y", "flux", "peak", "elongation"):
        arr = getattr(out, name)
        assert arr.shape == (0,) and arr.dtype == np.float64
    assert out.npix.shape == (0,) and out.npix.dtype == np.int64


def test_fit_table_on_an_empty_table():
    """空输入表必须原样给出空表与空列表，不得抛异常。"""
    img = make_image([(20.0, 30.0, 1000.0)])
    out, fits = fit_table(img, SourceTable.empty(frame=5))
    assert len(out) == 0
    assert fits == []
    assert out.frame == 5
    assert out.xy.shape == (0, 2)


# --------------------------------------------------------------------------
# 真实数据（裁决 34）
# --------------------------------------------------------------------------


def test_dataset_b_frame30_retention_and_fwhm(dataset_b_dir):
    """裁决 34：钉住真实数据上的保留率与 FWHM。

    ``max_residual_ratio`` 原为 0.35，从未对真星测过；实测它在
    dataset B 第 30 帧 box=9 上**拒掉约 60% 的本项目自己的 5σ 探测**
    （38/95 通过；任务书记录 39/95）。0.55 保留 87/95，同时仍拒掉灾难性失败区
    （300 个 64² 空天窗口 box=9 的 rr 中位数实测 0.9187，最差的真星 0.6504）。

    实测阈值曲线（同一帧、95 个 5σ 源、box=9）：
        0.35 -> 38/95 (40%)   0.45 -> 72/95   0.50 -> 82/95
        0.55 -> 87/95 (92%)   0.70 -> 95/95
    与任务书记录（39/72/82/87/95）只在 0.35 处差 1 个源。
    因此 ">= 85" 既有余量、又不允许退回 38。

    FWHM：实测被接受的 87 个源的中位 FWHM 为 3.4380 px（任务书记录 3.40±0.15，
    实测值落在带内），× 6.179 arcsec/px ≈ 21.2 arcsec。
    """
    seq = FrameSequence.from_directory(dataset_b_dir)
    img = seq.image(30)
    model = model_background(img, backend="cpu")
    tbl = detect_sources_in_frame(img, model, n_sigma=5.0, npixels=5, frame=30)
    # Task 6 钉死了这个数；拟合的输入必须正是那 95 个源。
    assert len(tbl) == 95

    out, fits = fit_table(model.subtract(img), tbl)
    assert len(out) == len(fits)
    assert len(fits) >= 85, f"默认判据下只保留 {len(fits)}/95，疑似阈值回退"
    assert out.frame == 30

    fwhm = median_fwhm(fits)
    # 实测 3.4380 px；3.40 ± 0.15 是任务书给的带子，实测值落在带内。
    # 3.40 px × 6.179 arcsec/px ≈ 21.0 arcsec。
    assert fwhm == pytest.approx(3.40, abs=0.15), f"实测中位 FWHM {fwhm:.4f} px"


# --------------------------------------------------------------------------
# 配置一致性
# --------------------------------------------------------------------------


def test_signature_defaults_match_config(cfg):
    """签名默认值必须来自 src/config/default.yaml，src/** 里不留魔数。

    ``max_residual_ratio`` 在裁决 33 里由 0.35 改为 0.55：签名与 YAML 必须同时
    是 0.55，否则真实数据的保留率会从 87/95 掉回 39/95。
    """
    psf_cfg = cfg["psf"]
    assert psf_cfg["box"] == 9
    assert psf_cfg["box"] % 2 == 1, "psf.box 必须是奇数，否则 cutout 会拒绝它"
    assert psf_cfg["max_nfev"] == 200
    assert psf_cfg["max_residual_ratio"] == 0.55

    for fn in (fit_gaussian2d, fit_table):
        params = inspect.signature(fn).parameters
        assert params["box"].default == psf_cfg["box"]
        assert params["max_nfev"].default == psf_cfg["max_nfev"]
        assert params["max_residual_ratio"].default == psf_cfg["max_residual_ratio"]


def test_fwhm_per_sigma_constant():
    """FWHM_PER_SIGMA 必须是 2*sqrt(2 ln 2)，写死的字面量不能与它漂移。"""
    assert FWHM_PER_SIGMA == pytest.approx(
        2.0 * np.sqrt(2.0 * np.log(2.0)), rel=1e-15
    )
    assert FWHM_PER_SIGMA == pytest.approx(2.3548200450309493, abs=1e-15)
