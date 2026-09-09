from __future__ import annotations

from dataclasses import FrozenInstanceError
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


def make_table(seeds, frame=3):
    """Build a SourceTable from integer seed positions ``[(x, y), ...]``.

    携带的四列 **每行互异**（第 ``i`` 行为 flux ``111*(i+1)``、peak ``11*(i+1)``、
    elongation ``1.01+0.01*i``、npix ``10*(i+1)``），这不是装饰：常量列会让
    "携带列错位" 这类缺陷原理上不可观测。此前这四列每行相同
    （1000.0/100.0/1.05/40），于是把 ``fit_table`` 里的
    ``table.select(np.array(keep, dtype=int))`` 换成 ``table.select(np.arange(len(fits)))``
    （即 ``[:n]`` 切片顶替按 keep 选行）在**全套件 178 项下全绿**。
    """
    xs = np.array([s[0] for s in seeds], dtype=np.float64)
    ys = np.array([s[1] for s in seeds], dtype=np.float64)
    n = len(seeds)
    return SourceTable(
        frame=frame,
        x=xs,
        y=ys,
        flux=np.array([111.0 * (i + 1) for i in range(n)], dtype=np.float64),
        peak=np.array([11.0 * (i + 1) for i in range(n)], dtype=np.float64),
        elongation=np.array([1.01 + 0.01 * i for i in range(n)], dtype=np.float64),
        npix=np.array([10 * (i + 1) for i in range(n)], dtype=np.int64),
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

    **本实现实测在这张 fixture 上返回 sx=3.0000、sy=1.4000、θ=+30.0000°**
    ——与植入值同一侧（任务书 Ruling 35 记录的 sx=1.40/sy=3.00/θ=-60° 是同一个
    椭圆的另一种标记，控制器那次测量的初值与本实现不同）。但这一侧不可依赖：
    同样的 σ 对换个朝向植入（θ=60°）就翻到 sx=1.4000 一侧，见
    ``test_fwhm_is_invariant_under_the_sigma_theta_relabelling``。
    所以这里比较 ``max``/``min`` 而不是逐字段比较：这条断言在两种标记下都成立，
    而 ``fit.sigma_x == 3.0`` 会在一个同样**正确**的实现上失败。
    """
    img = make_elliptical(32.0, 32.0, 900.0, 3.0, 1.4, 30.0)
    fit = fit_gaussian2d(img, 32.0, 32.0, box=17)
    assert fit.success is True
    assert max(fit.sigma_x, fit.sigma_y) == pytest.approx(3.0, rel=0.12)
    assert min(fit.sigma_x, fit.sigma_y) == pytest.approx(1.4, rel=0.12)


def test_fwhm_is_invariant_under_the_sigma_theta_relabelling():
    """裁决 35：``fwhm_px`` 用几何平均，因此对 σ/θ 的重标记免疫。

    **要真正测到不变性，两张 fixture 必须让优化器落在退化参数化的两侧。**
    此前这条测试用的是植入 (3.0, 1.4, 30°) 与 (1.4, 3.0, -60°)——那两组参数
    描述**同一个椭圆**，实测两张图逐像素只差 1.99e-13，于是优化器对两张图
    落在**同一侧**（都给 sx=3.0000 / sy=1.4000 / θ=+30.0000），``fwhm_a ==
    fwhm_b`` 逐位成立。结果 ``fwhm_px`` 改成 ``sx``、``sy``、``sx + sy``、
    ``0.0`` 这四种变异**全部**能通过不变性那条断言（相对差 0–3.4e-16）。
    实测：本实现对该椭圆的**任何**等价参数化（(3,1.4,30/210/-150)、
    (1.4,3,-60/120/300/-240)）都返回 sx=3.0000 一侧——图是逐位相同的，
    优化器当然给同一个答案。

    改法：换一张**朝向不同**的椭圆。同样的 σ 对 (3.0, 1.4)，植入 θ=30° 与
    植入 θ=60° 实测分别给

        θ=30° -> sx=3.0000、sy=1.4000、theta_deg=+30.0000
        θ=60° -> sx=1.4000、sy=3.0000、theta_deg=-30.0000

    ——两张图落在**两侧**，而 ``fwhm_px`` 实测**逐位相等**
    （都是 0x1.34dc487898469p+2 = 4.82594501282538）。

    四种变异现在的下场（实测）：``fwhm=sx`` 与 ``fwhm=sy`` 被不变性断言杀掉
    （相对差 0.533 / 1.143，远超 rel=1e-9）；``fwhm=sx+sy`` 与 ``fwhm=0.0``
    在两侧仍相等，由两条 ``== want`` 断言杀掉（相对偏差 0.0883 / 1.0000，
    超 rel=0.05）。四条一起才封住这个字段。
    """
    want = FWHM_PER_SIGMA * np.sqrt(3.0 * 1.4)

    fit_a = fit_gaussian2d(
        make_elliptical(32.0, 32.0, 900.0, 3.0, 1.4, 30.0), 32.0, 32.0, box=17
    )
    fit_b = fit_gaussian2d(
        make_elliptical(32.0, 32.0, 900.0, 3.0, 1.4, 60.0), 32.0, 32.0, box=17
    )
    assert fit_a.success is True and fit_b.success is True
    # 前提：两张图必须真的落在退化参数化的两侧，否则不变性无从谈起。
    assert fit_a.sigma_x > fit_a.sigma_y, "fixture a 应落在 sigma_x 是长轴的一侧"
    assert fit_b.sigma_x < fit_b.sigma_y, "fixture b 应落在 sigma_y 是长轴的一侧"

    assert fit_a.fwhm_px == pytest.approx(want, rel=0.05)
    assert fit_b.fwhm_px == pytest.approx(want, rel=0.05)
    # 重标记不变性：两次拟合的 fwhm_px 必须一致到 1e-9（实测逐位相等）。
    assert fit_a.fwhm_px == pytest.approx(fit_b.fwhm_px, rel=1e-9), (
        f"重标记下 fwhm_px 变了：{fit_a.fwhm_px!r} vs {fit_b.fwhm_px!r}"
    )


def test_theta_deg_is_in_degrees_and_the_orientation_is_recovered():
    """``theta_deg`` 的单位必须是**度**，符号必须与模型的旋转约定一致。

    此前测试文件里没有任何一处断言 ``theta_deg``：把 ``_model`` 的旋转符号
    翻过来（θ 由 +30° 变 −30°）、或让 ``theta_deg`` 返回弧度（+0.5236 而不是
    +30），两种变异都在全套件下存活，真实数据的保留率与中位 FWHM 逐位不变。

    **不能断言成一个裸数字**：``(σx, σy, θ)`` 参数化是退化的，
    ``(3.0, 1.4, 30°)`` ≡ ``(1.4, 3.0, -60°)``，哪一侧被叫做 ``sigma_x``
    取决于优化器。所以这里先按 ``σx > σy`` **归一化**——长轴是 ``sigma_y``
    时把角度 +90——再对 180° 取模（椭圆的朝向本来就只有 180° 的周期）。
    得到的"长轴方位角"是与标记无关的物理量。

    实测（box=17、amp=900、σ=(3.0, 1.4)）：

        植入 +30° -> sx=3.0000 sy=1.4000 theta_deg=+30.0000 -> 长轴方位  30°
        植入 -30° -> sx=3.0000 sy=1.4000 theta_deg=-30.0000 -> 长轴方位 150°
        植入 +60° -> sx=1.4000 sy=3.0000 theta_deg=-30.0000 -> 长轴方位  60°
        植入 -60° -> sx=1.4000 sy=3.0000 theta_deg=+30.0000 -> 长轴方位 120°

    两种变异都被这四条一起杀死：符号翻转把 +30° 的那张变成长轴方位 150°；
    返回弧度把它变成 0.5236°（而不是 30°）。
    """
    for planted in (30.0, -30.0, 60.0, -60.0):
        img = make_elliptical(32.0, 32.0, 900.0, 3.0, 1.4, planted)
        fit = fit_gaussian2d(img, 32.0, 32.0, box=17)
        assert fit.success is True, f"植入 θ={planted}° 的椭圆拟合失败了"
        # 按 σx > σy 归一化，再对 180° 取模，得到与标记无关的长轴方位角。
        major_angle = (
            fit.theta_deg if fit.sigma_x >= fit.sigma_y else fit.theta_deg + 90.0
        )
        major_angle %= 180.0
        assert major_angle == pytest.approx(planted % 180.0, abs=0.1), (
            f"植入 θ={planted}°：实测 theta_deg={fit.theta_deg:.5f}、"
            f"长轴方位 {major_angle:.5f}°，期望 {planted % 180.0:.5f}°"
            "（单位是否成了弧度？旋转符号是否翻了？）"
        )
        # 单位必须是度：同一个角度的弧度值在 |θ| = 30/60 上与度值差 30 倍以上。
        assert abs(fit.theta_deg) > 1.0, (
            f"theta_deg={fit.theta_deg!r} 太小，疑似返回的是弧度而不是度"
        )


def test_sigma_upper_bound_is_box_over_fwhm_per_sigma():
    """裁决 245：σ 的上界必须**恰好**是 ``box / FWHM_PER_SIGMA``。

    ``test_noise_cannot_be_fitted_as_a_broad_flat_source`` 里那条 40 种子的带子
    （min ≈ 0.9080、max ≈ 0.9931，abs=0.01）钉的是"纯噪声全部被拒"这个**行为**，
    它能杀掉上界 ``box/2``（实测 min 掉到 0.8578），但**杀不掉 ``box/3``**
    （实测 0.9092 / 0.9963，两端都落在带内）。也就是说它证明的是"上界不比
    box/2 松"，不是裁决 245 批准的那个公式。收紧那条 ``abs`` 不是修法——
    那正是本项目禁止的"照着观测值反推容差"。

    本测试直接钉公式：在一张**远宽于窗口**的高斯（σ=8.0）上拟合，优化器必然
    顶到 σ 的上界，于是 ``sigma_x`` 就**是**那个上界，可以对着闭式值比。
    这是纯闭式量，没有任何拟合残差进来，所以容差取 rel=1e-12：

        box=9  -> 9 / 2.3548200450309493  = 3.8219481012960856（实测逐位相等）
        box=11 -> 11 / 2.3548200450309493 = 4.671269901584105 （实测差 1.9e-16）

    三个候选上界全部被区分开（相对偏差）：``box`` 差 0.575、``box/2`` 差 0.151、
    ``box/3`` 差 0.274——都比 1e-12 大 11 个数量级以上。
    """
    for box in (9, 11):
        closed_form = float(box) / FWHM_PER_SIGMA
        # σ=8.0 的高斯在 9/11 像素的窗口里就是一片平缓的坡，优化器只能一路
        # 撑到上界。amp 与 rr 顺带断言，确认这不是一次失败的拟合。
        img = make_image([(32.0, 32.0, 1000.0)], sigma=8.0)
        fit = fit_gaussian2d(img, 32.0, 32.0, box=box)
        assert fit.success is True, f"box={box} 的宽高斯拟合失败了，无法读出上界"
        assert fit.sigma_x == pytest.approx(closed_form, rel=1e-12), (
            f"box={box}：σ 上界实测 {fit.sigma_x!r}，"
            f"闭式 box / FWHM_PER_SIGMA = {closed_form!r}"
        )
        assert fit.sigma_y == pytest.approx(closed_form, rel=1e-12)
        # 顶到上界意味着拟合出的 FWHM 恰好等于窗口边长——这就是上界的物理含义。
        assert fit.fwhm_px == pytest.approx(float(box), rel=1e-12)


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

    **这条测试证明什么、不证明什么。** 它钉的是**行为**——纯噪声在默认判据下
    全部被拒——不是上界的**公式**。文末那条 40 种子的带子（min ≈ 0.9080、
    max ≈ 0.9931，abs=0.01）能杀掉上界 ``box/2``（实测 min 掉到 0.8578），
    但**杀不掉 ``box/3``**：实测 ``box/3`` 给 0.9092 / 0.9963，两端都落在带内。
    所以它的强度只到"上界不比 box/2 松"。公式本身由
    ``test_sigma_upper_bound_is_box_over_fwhm_per_sigma`` 用闭式值钉住
    （rel=1e-12）。两条测试都需要：这一条保证噪声被拒，那一条保证用的是
    裁决 245 批准的那个上界。
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


@pytest.mark.parametrize("value", [0.0, 7.0, -3.5])
def test_flat_window_is_marked_failed_not_fitted(value):
    """恰好平坦的窗口必须显式失败，**不得**兜一个 amp0 继续拟合。

    ``amp0 = patch.max() - median(patch)`` 在平坦窗口上是 0。曾经的代码在
    这一支写 ``amp0 = max(abs(bkg0), 1.0)`` 继续拟合，实测后果是**凭空造出
    一个看起来完全合理的源**：

        全零窗口     -> success=True、amp=1.95e-08、rr=0.2301、fwhm=**8.9954**
        恒为 7.0 窗口 -> success=True、amp=3.40e-05、rr=0.3251、fwhm=**7.4996**

    8.99 px 与 7.50 px 都是毫无破绽的 FWHM，rr 又远低于 0.55——下游没有任何
    办法把它与真星区分开。这正是本模块开头那段"拟合失败必须显式标记"
    所禁止的东西。现在这一支直接 ``return _failed(x, y)``。

    这条路径只在**恰好**平坦的窗口上可达（死列、被常数填充的掩膜区、被削平的
    饱和平台）——评审在 200 个纯噪声窗口里 0/200 触发——所以只有常数帧能守卫它，
    而此前测试文件里没有任何 ``np.zeros``/``np.ones``/常数帧测试。
    """
    img = np.full((32, 32), value, dtype=np.float64)
    fit = fit_gaussian2d(img, 16.0, 16.0)
    assert fit.success is False, (
        f"恒为 {value} 的平坦窗口被拟合成了 success=True，"
        f"amp={fit.amplitude!r}、fwhm={fit.fwhm_px!r}、rr={fit.residual_ratio!r}"
    )
    # 形状字段必须全 NaN，不能留下任何可被下游当成测量值的数。
    assert not np.isfinite(fit.amplitude)
    assert not np.isfinite(fit.sigma_x)
    assert not np.isfinite(fit.sigma_y)
    assert not np.isfinite(fit.fwhm_px)
    # 种子原样带回。
    assert fit.x == pytest.approx(16.0)
    assert fit.y == pytest.approx(16.0)


def test_fit_table_on_a_flat_frame_keeps_nothing():
    """经公开 API：平坦帧上必须保留 **0** 个源，不是 2 个伪造源。

    这是 ``test_flat_window_is_marked_failed_not_fitted`` 的端到端对照。
    ``amp0`` 兜底版本实测在恒为 7.0 的帧上让 ``fit_table`` **保留 2 个源**
    （fwhm 7.4996），带着两个纯属虚构的亚像素中心流进配准。
    """
    tbl = make_table([(10.0, 10.0), (20.0, 20.0)], frame=4)
    for value in (0.0, 7.0):
        img = np.full((32, 32), value, dtype=np.float64)
        out, fits = fit_table(img, tbl)
        assert len(fits) == 0, f"恒为 {value} 的平坦帧上保留了 {len(fits)} 个伪造源"
        assert len(out) == 0
        assert out.frame == 4


def test_residual_ratio_denominator_excludes_the_background():
    """分母必须是 ``|patch - bkg|``，不是 ``|patch|``。

    这是 ``residual_ratio`` 定义的核心、本模块唯一的判据，此前**完全没有守卫**：
    把分母换成 ``|patch|`` 后全套件绿，而真实数据上它把保留率从 **87 抬到 92**、
    95 个源上 ``max|Δrr| = 0.161463``。

    遮住它的是两种输入同时失效：真实数据那条测试喂的是 ``model.subtract(img)``
    （已减背景，bkg≈0，两种分母只差千分之几），合成 fixture 无噪声、rr 恒为 0
    （0/x 与 0/y 都是 0）。唯一能区分的输入是**非零常数背景 + 噪声**——现有那张
    background=25.0 的 fixture 不够用，它无噪声。

    fixture：σ_noise=4.0、seed 7、amp=200 的源在 (32.3, 31.6)、box=9。实测

        background=0.0   -> ``|patch - bkg|`` 给 0.076530、``|patch|`` 给 0.076727（差 0.26%）
        background=400.0 -> ``|patch - bkg|`` 给 0.076530、``|patch|`` 给 0.006930（差 **11 倍**）

    也就是说正确的分母让 rr 与背景电平**无关**（两个电平下逐 6 位小数相同），
    而 ``|patch|`` 让同一颗星的 rr 随背景任意缩小。两条断言分别钉住这两点：
    绝对值 rel=1e-3（``|patch|`` 版本会给 0.0069，差 11 倍），以及
    "换背景电平 rr 不变" rel=1e-6（``|patch|`` 版本相差 11 倍）。
    """
    def noisy_with_background(background):
        rng = np.random.default_rng(7)
        img = rng.normal(0.0, 4.0, size=(64, 64)) + background
        img += make_image([(32.3, 31.6, 200.0)])
        return img

    fit_zero = fit_gaussian2d(noisy_with_background(0.0), 32.0, 32.0, box=9)
    fit_high = fit_gaussian2d(noisy_with_background(400.0), 32.0, 32.0, box=9)
    assert fit_zero.success is True and fit_high.success is True
    # 拟合出的背景确实是那个电平（否则下面比的不是同一件事）。
    assert fit_zero.background == pytest.approx(0.0, abs=1.0)
    assert fit_high.background == pytest.approx(400.0, abs=1.0)

    # 实测 0.0765295；``|patch|`` 分母在 background=400 上会给 0.0069301。
    assert fit_high.residual_ratio == pytest.approx(0.0765295, rel=1e-3), (
        f"background=400 上 rr={fit_high.residual_ratio!r}，"
        "分母疑似用了 |patch| 而不是 |patch - bkg|"
    )
    # 正确的分母让 rr 与背景电平无关。
    assert fit_high.residual_ratio == pytest.approx(
        fit_zero.residual_ratio, rel=1e-6
    ), (
        f"rr 随背景电平变了：bkg=0 给 {fit_zero.residual_ratio!r}、"
        f"bkg=400 给 {fit_high.residual_ratio!r}；分母没有扣掉背景"
    )


def test_failed_fit_residual_ratio_is_nan_so_success_must_be_the_filter():
    """未走到拟合的失败带 NaN 残差比——下游**只能**用 ``success`` 过滤。

    ``_failed`` 的 ``residual_ratio`` 默认值是 ``np.nan`` 而不是 ``np.inf``：
    对中心落在图外、窗口小于 9 像素、分母 ≤ 0 这三条路径，残差比是
    **未测量**，不是"无穷大误差"。语义更对，但有一个必须写下来的后果——
    NaN 的比较全是 ``False``：

        np.nan > 0.55  ->  False
        np.nan > 1e9   ->  False

    所以 ``[f for f in fits if f.residual_ratio <= thr]`` 会把这些行**静默留下**
    （换成 ``inf`` 时 ``inf > thr`` 为 True，写 ``rr > thr`` 的过滤器反而会剔除
    它们）。本仓库的 ``fit_table`` 过滤 ``fit.success``，所以陷阱目前只是潜在的；
    这条测试连同 ``_failed`` 的 docstring 一起把它钉住。

    注意区分：超出判据的失败（``ratio > max_residual_ratio``）**会**带回实测的
    残差比——本测试也断言这一条，否则"把所有失败都写成 NaN"的变异会存活。
    """
    img = make_image([(32.0, 32.0, 1000.0)])
    outside = fit_gaussian2d(img, 5000.0, 32.0)
    assert outside.success is False
    assert np.isnan(outside.residual_ratio), (
        f"图外拟合的 residual_ratio 是 {outside.residual_ratio!r}，期望 nan"
    )
    # NaN 的比较语义：任何阈值比较都给 False，所以 rr 不能当过滤器用。
    assert not (outside.residual_ratio > 0.55)
    assert not (outside.residual_ratio > 1e9)
    assert not (outside.residual_ratio <= 0.55)

    # 平坦窗口走的是同一条"未测量"路径。
    flat = fit_gaussian2d(np.zeros((32, 32), dtype=np.float64), 16.0, 16.0)
    assert flat.success is False
    assert np.isnan(flat.residual_ratio)

    # 反面：超出判据的失败必须带回**实测**的残差比，不是 NaN。
    rng = np.random.default_rng(9)
    noise = rng.normal(0.0, 3.8, size=(64, 64))
    rejected = fit_gaussian2d(noise, 32.0, 32.0, box=11, max_residual_ratio=0.35)
    assert rejected.success is False
    assert np.isfinite(rejected.residual_ratio), (
        "超出判据的失败也把 residual_ratio 写成了 NaN，实测残差比丢了"
    )
    assert rejected.residual_ratio > 0.35


@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int32, np.uint16, np.int16]
)
def test_fit_accepts_integer_and_single_precision_input(dtype):
    """整数与单精度输入必须照样拟合成功，且中心一致到 1e-3 px。

    实现里**没有** ``np.asarray(image, dtype=np.float64)`` 强转——依赖的是
    NumPy 在 ``patch - bkg``、``xx - cx`` 这些运算里自动提升到 float64。
    这是一处契约：调用方喂什么 dtype 都不该改变结果。此前无守卫，
    下次有人喂 float16 或掩膜数组时行为未定义。

    fixture：真值 (32.37, 29.62)、amp=1000、background=100（整数 dtype 需要
    非负且有足够动态范围）、box=11。实测

        float64/float32 -> x=32.370000、y=29.620000
        int32/uint16/int16 -> x=32.370213、y=29.619776（量化到整数后偏 2.1e-04 px）

    整数量化只把中心挪了 2.1e-04 px、rr 从 3e-16 抬到 1.9e-03，所以 abs=1e-3
    的中心容差对五种 dtype 都成立且不宽松。
    """
    base = make_image([(32.37, 29.62, 1000.0)], background=100.0)
    fit = fit_gaussian2d(base.astype(dtype), 32.0, 30.0, box=11)
    assert fit.success is True, f"dtype={np.dtype(dtype).name} 上拟合失败了"
    assert fit.x == pytest.approx(32.37, abs=1e-3)
    assert fit.y == pytest.approx(29.62, abs=1e-3)
    assert fit.residual_ratio < 0.01


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


def test_median_fwhm_takes_the_median_not_the_mean():
    """必须取**中位数**，均值会被个别极大 FWHM 拖走。

    真实数据上两者差得不多（第 30 帧 87 个源：median **3.4380**、mean
    **3.5084**），而 ``test_dataset_b_frame30_retention_and_fwhm`` 的带子是
    3.40±0.15——两个值都落在带内，所以把 median 换成 mean **在全套件下存活**。
    只有偏斜的列表能守卫这一点。

    fixture：``[3.0, 3.1, 3.2, 3.3, 30.0]``——一个拟合到邻近源或宇宙线上的
    极大值。median 是 **3.2**、mean 是 **8.52**，差 2.66 倍，rel=1e-12 的容差
    也远远够用。
    """
    values = [3.0, 3.1, 3.2, 3.3, 30.0]
    fits = [
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
        for value in values
    ]
    got = median_fwhm(fits)
    assert got == pytest.approx(3.2, rel=1e-12), (
        f"median_fwhm 给 {got!r}；中位数是 3.2、均值是 8.52，疑似取了均值"
    )
    # 显式排除均值，免得将来有人把 3.2 这个数当巧合。
    assert got != pytest.approx(float(np.mean(values)), rel=1e-3)


def test_gaussian_fit_is_frozen():
    """``GaussianFit`` 必须是 frozen，字段写不进去。

    这个契约此前无守卫：去掉 ``frozen=True`` 后全套件绿。它要紧是因为拟合结果
    会被放进列表、与 ``SourceTable`` 的行一一对应地传下去——任何一处就地改写
    都会让"表的第 i 行由 fits[i] 精化而来"这条不变式静默失效，而且改的人不会
    收到任何提示。
    """
    fit = GaussianFit(
        x=1.0,
        y=2.0,
        amplitude=100.0,
        sigma_x=1.5,
        sigma_y=1.6,
        theta_deg=10.0,
        background=0.0,
        fwhm_px=3.6,
        residual_ratio=0.1,
        success=True,
    )
    with pytest.raises(FrozenInstanceError):
        fit.x = 99.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        fit.success = False  # type: ignore[misc]
    assert fit.x == pytest.approx(1.0)
    assert fit.success is True


def test_truncated_optimisation_is_rejected_even_when_the_ratio_is_small():
    """``result.success`` 这一项不是冗余的：半收敛的拟合能给出很低的 rr。

    ``least_squares`` 撞到 ``max_nfev`` 时返回 ``status=0``/``success=False``，
    但那组半收敛的参数照样能算出一个通得过 0.55 判据的残差比。实测
    （无噪声高斯 (32.37, 29.62)、amp=1000、box=11）：

        max_nfev=1   -> res.success=False、rr=1.7493、fwhm=6.4758（rr 自己就超了）
        max_nfev=2   -> res.success=False、rr=**0.2717**、fwhm=6.4758 → 5.9716
        max_nfev=3   -> res.success=False、rr=**0.2546**、fwhm=**4.2209**
        max_nfev=200 -> res.success=True、 rr=2.0e-16、fwhm=3.7677（真值 3.7677）

    ``max_nfev=2/3`` 的 rr 都远低于 0.55，fwhm 也没有明显破绽（4.22 vs 真值
    3.77，只差 12%），**只有 ``bool(result.success)`` 拦得住它们**。从 ``ok``
    里删掉这一项在评审的 200 个纯噪声窗口上改变 5 个判定。

    这里显式传 ``max_residual_ratio=1e9`` 把判据这一项让开，好让断言只压在
    ``result.success`` 上——否则 max_nfev=1 那一档会被 rr 顺手拒掉，
    测不出这一项的作用。
    """
    img = make_image([(32.37, 29.62, 1000.0)])
    for nfev in (2, 3):
        fit = fit_gaussian2d(
            img, 32.0, 30.0, box=11, max_nfev=nfev, max_residual_ratio=1e9
        )
        assert fit.success is False, (
            f"max_nfev={nfev} 的半收敛拟合被接受了："
            f"rr={fit.residual_ratio!r}、fwhm={fit.fwhm_px!r}"
            "——ok 里的 bool(result.success) 疑似被删掉了"
        )
        # 前提：它确实**不是**被 rr 判据拒掉的，rr 远低于默认的 0.55。
        assert fit.residual_ratio < 0.55, (
            f"max_nfev={nfev} 的 rr={fit.residual_ratio!r} 超过 0.55，"
            "本测试的前提不再成立（它要证明的是 rr 通过时 success 仍拦得住）"
        )
    # 对照：迭代给够时同一张图必须成功。
    ok = fit_gaussian2d(img, 32.0, 30.0, box=11, max_nfev=200)
    assert ok.success is True
    assert ok.residual_ratio < 1e-6


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
    """失败的源必须被剔除，保留下来的必须保持原顺序，且携带列必须与行对齐。

    三个种子按顺序是 (5, 5)、(20, 30)、(45, 18)：**第一个**是空天区（实测
    residual_ratio 0.7309，正确被拒），后两个是亮源。失败源放在 **index 0**
    而不是末尾，且 ``make_table`` 的四列各行互异——这两点一起才让
    "用 ``[:len(fits)]`` 切片顶替按 keep 选行" 这个缺陷可观测：正确输出是
    flux ``[222.0, 333.0]`` / npix ``[20, 30]``，切片会给
    ``[111.0, 222.0]`` / ``[10, 20]``（即把第二、三颗星的坐标配上第一、二行
    的流量）。失败源在末尾 + 常量列时两者恰好一致，全套件都看不见错位。

    裁决 37：``fit_table`` 用 ``table.select(...)`` 的返回值构造输出，
    而 ``select`` 交回的是全新数组——因此输入表必须完好，输出的 x 也不能
    是输入 x 的同一个对象。

    注：``SourceTable`` 是 **frozen** dataclass，所以任务书 Ruling 37 描述的
    ``out.x = ...`` 属性赋值在本仓库里会抛 ``FrozenInstanceError``；实现改为
    显式构造新表。本测试断言的是同一份契约，与内部写法无关。
    """
    img = _noisy_two_sources()
    tbl = make_table([(5.0, 5.0), (20.0, 30.0), (45.0, 18.0)], frame=11)
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
    # 返回的表与返回的列表逐行对应。
    for index, fit in enumerate(fits):
        assert out.x[index] == pytest.approx(fit.x, rel=1e-12), f"第 {index} 行的 x 与 fits[{index}] 不符"
        assert out.y[index] == pytest.approx(fit.y, rel=1e-12), f"第 {index} 行的 y 与 fits[{index}] 不符"

    # **行对齐**：携带列必须来自被保留的那两个输入行（index 1、2），
    # 不是前两行。这是本测试相对切片变异的唯一守卫。
    assert out.flux.tolist() == [222.0, 333.0], "携带的 flux 与保留的行不对齐"
    assert out.peak.tolist() == [22.0, 33.0], "携带的 peak 与保留的行不对齐"
    assert out.npix.tolist() == [20, 30], "携带的 npix 与保留的行不对齐"
    assert out.elongation[0] == pytest.approx(1.02), "携带的 elongation 与保留的行不对齐"
    assert out.elongation[1] == pytest.approx(1.03), "携带的 elongation 与保留的行不对齐"
    assert out.npix.dtype == np.int64

    # 裁决 37：输入表必须完好。
    assert tbl.x[0] == 5.0
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
    assert tbl.flux[1] == pytest.approx(222.0), "out.flux 与输入表共享内存"
    assert tbl.peak[1] == pytest.approx(22.0), "out.peak 与输入表共享内存"
    assert tbl.elongation[1] == pytest.approx(1.02), "out.elongation 与输入表共享内存"
    assert tbl.npix[1] == 20, "out.npix 与输入表共享内存"


def test_fit_table_passes_box_and_max_nfev_through():
    """``box`` 与 ``max_nfev`` 必须原样透传给 ``fit_gaussian2d``。

    在 ``fit_table`` 内部硬写 ``box=9`` 或 ``max_nfev=200`` 忽略入参，此前
    **全套件绿**——没有任何测试用非默认值调用过 ``fit_table``。

    两条断言都刻意做成**整数差异**（保留数），不是浮点尾数：

    - ``max_nfev=1``：优化器一步就被截断，``least_squares`` 返回
      ``status=0``/``success=False``，两个亮源全部被剔除 -> 保留 **0** 个。
      硬写 ``max_nfev=200`` 的变异体保留 2 个。
    - ``box=63``：窗口大到把整张 64² 图几乎全吞进来，噪声像素压倒信号，
      两个源的 rr 实测升到 0.5834 与 0.5672，**都超过 0.55** -> 保留 **0** 个。
      硬写 ``box=9`` 的变异体保留 2 个（rr 0.0159 / 0.0250）。

    **为什么不断言 fwhm 尾数**：评审量到 ``box=17`` 与 ``box=9`` 在椭圆
    fixture 上给 fwhm 3.7672 vs 3.7706，差 0.0034（相对 9e-4）。这个差异比
    优化器在不同窗口下的收敛抖动大不了多少，钉它需要一条只能靠观测值反推的
    紧容差；保留数是整数，任何一侧都不可能因浮点抖动翻转。
    """
    img = _noisy_two_sources()
    tbl = make_table([(20.0, 30.0), (45.0, 18.0)], frame=11)

    out_default, fits_default = fit_table(img, tbl)
    assert len(fits_default) == 2, "默认参数下两个亮源都应保留，否则本测试的对照失效"

    _, fits_nfev = fit_table(img, tbl, max_nfev=1)
    assert len(fits_nfev) == 0, (
        f"max_nfev=1 下保留了 {len(fits_nfev)} 个源，max_nfev 疑似未透传"
    )

    out_box, fits_box = fit_table(img, tbl, box=63)
    assert len(fits_box) == 0, (
        f"box=63 下保留了 {len(fits_box)} 个源，box 疑似未透传"
    )
    assert len(out_box) == 0


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
    （空天窗口 box=9 的 rr 中位数实测 0.9227，最差的真星 0.6504；空窗配方见
    ``fit_gaussian2d`` 的 docstring）。

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
