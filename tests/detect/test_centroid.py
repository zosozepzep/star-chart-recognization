from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.detect.centroid import aperture_flux, centroid_of_mass, cutout, remeasure
from src.detect.segmentation import SourceTable

SIGMA_PSF = 1.6
# 单个 amp=1000、σ=1.6 的高斯的解析总流量：amp * 2π * σ² = 16084.95。
ANALYTIC_TOTAL = 1000.0 * 2.0 * np.pi * SIGMA_PSF**2


def make_image(sources, ny=64, nx=64, sigma=SIGMA_PSF):
    """Return a noiseless background-subtracted image with planted gaussians.

    `sources` entries are `(cx, cy, amp)` with cx = column, cy = row.
    """
    yy, xx = np.mgrid[0:ny, 0:nx]
    img = np.zeros((ny, nx), dtype=np.float64)
    for cx, cy, amp in sources:
        img += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return img


def make_table(sources, frame=3):
    """Build a SourceTable seeded at the rounded (integer) source positions.

    Seeding on integers rather than the true subpixel centres is deliberate: it
    is what a threshold segmentation actually hands over, and it gives
    `remeasure` something to improve on.
    """
    xs = np.array([round(s[0]) for s in sources], dtype=np.float64)
    ys = np.array([round(s[1]) for s in sources], dtype=np.float64)
    return SourceTable(
        frame=frame,
        x=xs,
        y=ys,
        flux=np.array([s[2] * 10.0 for s in sources], dtype=np.float64),
        peak=np.array([s[2] for s in sources], dtype=np.float64),
        elongation=np.full(len(sources), 1.05, dtype=np.float64),
        npix=np.full(len(sources), 40, dtype=np.int64),
    )


# --------------------------------------------------------------------------
# cutout
# --------------------------------------------------------------------------


def test_cutout_interior_shape_origin_and_pixel_alignment():
    """An interior cutout must be exactly (box, box) and correctly registered.

    实测 64×64 图、(20, 30)、box=9 -> shape (9, 9)、origin (16, 26)，且
    patch[4, 4] 必须**就是** img[30, 20]。patch 索引是 [row, col] = [y, x]，
    origin 是 (x0, y0)；这条断言是这两个约定唯一的守卫。
    """
    img = make_image([(20.0, 30.0, 1000.0)])
    patch, x0, y0 = cutout(img, 20.0, 30.0, 9)
    assert patch.shape == (9, 9)
    assert (x0, y0) == (16, 26)
    # 中心像素必须对上：x 与 y 互换会让这条失败（img[20, 30] != img[30, 20]）。
    assert patch[4, 4] == pytest.approx(img[30, 20])
    assert patch[0, 0] == pytest.approx(img[26, 16])


def test_cutout_clips_at_border_with_exact_shapes_and_origins():
    """裁决 32：钉死精确形状与原点，而不是只断言 "比 9 小"。

    原任务书只查 `patch.shape[0] < 9 and patch.shape[1] < 9`，那样 (1,1) 或
    (8,8) 之类的错窗口一样能过。实测（64×64 图，box=9）：

        (1, 1)   -> shape (6, 6)  origin (0, 0)
        (62, 61) -> shape (7, 6)  origin (58, 57)   <- 触发 min(image.shape) 分支
        (62, 30) -> shape (9, 6)  origin (58, 26)

    (62, 61) 这一行是关键：原任务书从未覆盖过上界裁剪分支。形状是
    (行, 列) = (y 跨度, x 跨度)，所以 (7, 6) 与 (6, 7) 不等价——它同时钉住了
    x/y 不被互换。
    """
    img = make_image([(20.0, 30.0, 1000.0)])

    patch, x0, y0 = cutout(img, 1.0, 1.0, 9)
    assert patch.shape == (6, 6)
    assert (x0, y0) == (0, 0)

    # 远端边界：x 与 y 的裁剪量不同，形状因此不对称。
    patch, x0, y0 = cutout(img, 62.0, 61.0, 9)
    assert patch.shape == (7, 6)
    assert (x0, y0) == (58, 57)

    patch, x0, y0 = cutout(img, 62.0, 30.0, 9)
    assert patch.shape == (9, 6)
    assert (x0, y0) == (58, 26)


def test_cutout_box_larger_than_image_returns_whole_image():
    """box 超过图幅时退化为整幅图，原点为 (0, 0)。用奇数 box（偶数已改为报错）。"""
    img = make_image([(20.0, 30.0, 1000.0)])
    patch, x0, y0 = cutout(img, 32.0, 32.0, 201)
    assert patch.shape == (64, 64)
    assert (x0, y0) == (0, 0)


def test_cutout_rejects_even_box_in_chinese():
    """裁决 32：偶数 box 必须报错，不能静默返回 box+1。

    实测 box=8 返回 (9, 9)、box=10 返回 (11, 11)——因为 `half = box // 2`
    随后取 `xc-half .. xc+half` 闭区间，共 `2*half+1` 个像素。于是 box 只在
    奇数时才是窗口尺寸。所有调用点都传奇数（config 的 psf.box=9、本模块默认
    9、aperture_flux 内部推得的 13），所以没人需要偶数支持，而喂给 PSF 拟合
    和质心的窗口里藏半个像素的系统偏差事后无人能查。
    """
    img = make_image([(20.0, 30.0, 1000.0)])
    for bad in (8, 10, 2):
        with pytest.raises(ValueError) as exc:
            cutout(img, 20.0, 30.0, bad)
        msg = str(exc.value)
        assert any("一" <= ch <= "鿿" for ch in msg), f"box={bad} 的报错不是中文"
        assert str(bad) in msg, f"box={bad} 的报错未给出实际取值"
    # 奇数 box 必须仍然正常工作，且恰好是 (box, box)。
    assert cutout(img, 20.0, 30.0, 9)[0].shape == (9, 9)


def test_cutout_rejects_non_positive_box_in_chinese():
    """B-3：`box <= 0` 这道守卫此前没有任何测试——删掉它整套 23 项照样全绿。

    `box=0` 会先被偶数分支拦下（0 是偶数），负奇数（-1、-3）才走到
    `box <= 0` 那一条。两条路径都必须给出中文 ValueError，且不能静默返回
    空切片：`half = -1 // 2` 在 Python 里是 -1，于是 box=-1 若不拦会得到一个
    形状诡异的窗口而不是报错。
    """
    img = make_image([(20.0, 30.0, 1000.0)])
    for bad in (0, -1, -3, -9):
        with pytest.raises(ValueError) as exc:
            cutout(img, 20.0, 30.0, bad)
        msg = str(exc.value)
        assert any("一" <= ch <= "鿿" for ch in msg), f"box={bad} 的报错不是中文"
        assert str(bad) in msg, f"box={bad} 的报错未给出实际取值"


# --------------------------------------------------------------------------
# aperture_flux
# --------------------------------------------------------------------------


def test_aperture_flux_recovers_most_of_the_analytic_total():
    """r=5、σ=1.6 的孔径必须取到解析总流量的绝大部分。

    实测 15970.93 对解析值 16084.95，相对偏差 0.709%，相对 rel=0.03 有约 4 倍余量。
    """
    img = make_image([(20.0, 30.0, 1000.0)])
    flux = aperture_flux(img, np.array([[20.0, 30.0]]))
    assert flux.shape == (1,)
    assert flux.dtype == np.float64
    assert flux[0] == pytest.approx(ANALYTIC_TOTAL, rel=0.03)


def test_aperture_is_hard_edged_and_the_deficit_is_quantified():
    """裁决 29：硬边缘孔径的量化亏损必须被钉住，而不是只写在 docstring 里。

    实现用布尔掩码 `(xx-x)**2 + (yy-y)**2 <= radius**2`，是**硬像素边界**，
    没有部分面积加权。实测后果：15970.93 对解析 16084.95，正好低 0.709%。
    这个方向和量级要钉死，因为「我们的孔径是亚像素精确的」是评委三十秒就能
    验的说法，而一个诚实量化的 0.71% 是站得住的。

    **B-2 修正：这个 0.5%–1.0% 的带子钉的是「截断比例」，不是边缘处理方式。**
    原 docstring 声称它也能排除真正的亚像素加权孔径，评审实测证伪：
        photutils method="exact"（面积精确）-> 亏损 0.8877%   <- 落在带子里
        photutils method="center"（等价于 `<`）-> 亏损 1.2740%  <- 越界
        本实现（硬边缘 `<=`）              -> 亏损 0.7089%
    物理根因：r=5、σ=1.6 即 3.125σ，二维高斯在该半径外的解析截断量本身就是
    `exp(-r²/2σ²)` = **0.7576%**。也就是说这三种实现的亏损绝大部分来自
    「孔径不够大」，边缘加权方式只贡献零点几个百分点的差异，所以带子容纳
    0.888% 是必然的，不是缺陷。

    **约束孔径形状的是 `test_aperture_flux_is_monotonic_in_radius`**：r=2 时
    本实现给 8828.18，而 photutils exact 给 8477.54、center 给 6996.85，
    三者在 rel=0.01 下互不兼容——小半径上边缘像素占比大，加权方式才显形。

    按裁决 243，**不得**为了排除 0.888% 而收窄带子：那正是从两个测量值倒推
    容差的做法。带子诚实地钉住截断量，形状由单调性测试钉。
    """
    img = make_image([(20.0, 30.0, 1000.0)])
    flux = aperture_flux(img, np.array([[20.0, 30.0]]))[0]
    deficit = (ANALYTIC_TOTAL - flux) / ANALYTIC_TOTAL
    # 方向：孔径只截取有限半径，一定偏低，绝不会偏高。
    assert flux < ANALYTIC_TOTAL
    # 量级：实测 0.00709，解析截断量 0.007576，钉在 0.005~0.010 之间。
    assert 0.005 < deficit < 0.010, f"实测亏损 {deficit:.5f} 不在预期的 0.5%~1.0% 区间"


def test_aperture_flux_is_monotonic_in_radius():
    """更大的孔径必须收到更多流量。实测 r=2 -> 8828.2、r=6 -> 16070.2。

    B-2：**本测试是约束孔径形状（边缘加权方式）的那一条**，而不是那条
    0.5%–1.0% 亏损带——后者钉的是截断比例，容纳面积精确孔径的 0.888%。
    小半径上边缘像素占周长的比例大，加权方式才显形，实测 r=2：
        本实现（硬边缘 `<=`）      -> 8828.18
        photutils method="exact"  -> 8477.54
        photutils method="center" -> 6996.85
    三者在 rel=0.01 下互不兼容，所以这里的 r=2 断言能判别实现换代。
    """
    img = make_image([(20.0, 30.0, 1000.0)])
    xy = np.array([[20.0, 30.0]])
    small = aperture_flux(img, xy, radius=2.0)[0]
    large = aperture_flux(img, xy, radius=6.0)[0]
    assert small == pytest.approx(8828.2, rel=0.01)
    assert large == pytest.approx(16070.2, rel=0.01)
    assert large > small


def test_aperture_flux_at_border_is_finite_and_positive():
    """紧贴边角的源必须给出有限正值，而不是 NaN 或抛异常。

    实测 (1.5, 1.5) 处的 amp=1000 源给出 12937.9——孔径被图边裁掉一部分，
    所以显著低于 15970.93，但仍是有限正值。
    """
    img = make_image([(1.5, 1.5, 1000.0)])
    flux = aperture_flux(img, np.array([[1.5, 1.5]]))
    assert np.isfinite(flux).all()
    assert flux[0] > 0.0
    assert flux[0] == pytest.approx(12937.9, rel=0.01)


def test_aperture_flux_handles_empty_input():
    img = make_image([(20.0, 30.0, 1000.0)])
    flux = aperture_flux(img, np.empty((0, 2), dtype=np.float64))
    assert flux.shape == (0,)
    assert flux.dtype == np.float64


# --------------------------------------------------------------------------
# centroid_of_mass
# --------------------------------------------------------------------------


def test_centroid_recovers_subpixel_position_from_integer_seed():
    """整数种子必须被精化到真实的亚像素位置。

    真值 (32.37, 29.62)，种子 (32, 30)，实测精化结果 (32.354587, 29.635870)，
    最大偏差 0.0159 px，相对 abs=0.05 有 3 倍余量。

    **坐标刻意不对称**：把输出的 (cx, cy) 互换会给出 (29.6359, 32.3546)，
    与真值差 2.73 px，本测试会响亮失败。不要把这个 fixture "整理"成对称坐标。
    """
    img = make_image([(32.37, 29.62, 1000.0)])
    got = centroid_of_mass(img, np.array([[32.0, 30.0]]))
    assert got.shape == (1, 2)
    assert got.dtype == np.float64
    assert got[0, 0] == pytest.approx(32.37, abs=0.05)
    assert got[0, 1] == pytest.approx(29.62, abs=0.05)


def test_centroid_batch_refines_every_source():
    """批量精化：两个源都必须被独立精化。

    实测（64×64，两源分别 amp=1000 与 600）：
        (20.2, 30.4) 种子 (20, 30) -> 偏差 (0.00806, 0.01679)
        (45.2, 18.1) 种子 (45, 18) -> 偏差 (0.00806, 0.00399)
    最大偏差 0.0168，相对 abs=0.06 有 3.5 倍余量。

    两个源的 x、y 坐标都不相等（20.2/30.4、45.2/18.1），所以互换输出会让
    偏差跳到 10.2 px 与 27.1 px，本测试同样能抓到。
    """
    sources = [(20.2, 30.4, 1000.0), (45.2, 18.1, 600.0)]
    img = make_image(sources)
    seeds = np.array([[20.0, 30.0], [45.0, 18.0]])
    got = centroid_of_mass(img, seeds)
    want = np.array([[20.2, 30.4], [45.2, 18.1]])
    assert got.shape == (2, 2)
    err = np.abs(got - want)
    assert err.max() < 0.06, f"实测最大偏差 {err.max():.5f}"
    # 逐个源单独钉，避免一个源精化正确就掩盖另一个没被处理。
    assert err[0].max() < 0.06
    assert err[1].max() < 0.06


def test_centroid_does_not_modify_the_input_array():
    """输入的种子数组必须保持原样——调用方可能还要用它。"""
    img = make_image([(32.37, 29.62, 1000.0)])
    seeds = np.array([[32.0, 30.0]])
    before = seeds.copy()
    out = centroid_of_mass(img, seeds)
    assert np.array_equal(seeds, before)
    assert out is not seeds


def test_negative_clip_matters_on_a_noisy_fixture():
    """裁决 30：负值截零这个特性必须由一个**有噪声**的 fixture 来守卫。

    在无噪声高斯上，clip 与不 clip 给出**逐字节相同**的结果
    （两者都是 32.354587, 29.635870）——纯高斯没有负像素，于是删掉整个
    `np.clip(patch, 0.0, None)` 能通过原任务书的每一条测试。

    有噪声的 fixture（amp=60 信号、σ=20 高斯噪声、seed 5、窗口内实测 81 个
    像素里有 22 个为负）实测：
        带 clip : (31.8153454494, 30.0751601927)
        无 clip : (31.4910792836, 29.9997906607)   <- 变异 2 的实测值
    x 方向差 0.324266 px，约为别处所用 abs=0.05 容差的 6 倍，判别决定性。

    **注意本测试钉的是实测的带 clip 坐标本身，不是"落在真值 0.05 内"**：
    在这个信噪比下带 clip 的结果距真值 (32.37, 29.62) 仍有 0.5547 px，
    任何能容纳 0.5547 又能排除无 clip 的 0.8789 的容差都是硬凑的。钉死实测
    坐标既诚实又能决定性地杀掉变异 2。（此偏离已由裁决 243 批准。）

    **B-5：本测试唯一的守卫是上面那两条 ±1e-3 的坐标钉。** 变异 2（删掉
    `np.clip`）使 x 变为 31.4911，偏离钉住值 0.3243，是 1e-3 容差的 **324 倍**。
    最后那条 `< 0.6` 是**意图文档，不是守卫**：给定 ±1e-3 的钉，它的可达区间
    只有 0.5537–0.5557，因此不可能失败。保留它是为了写明"截零让质心更靠近
    真值"这个方向性事实，不要把它当成检验手段。
    """
    truth_x, truth_y = 32.37, 29.62
    rng = np.random.default_rng(5)
    img = rng.normal(0.0, 20.0, size=(64, 64))
    yy, xx = np.mgrid[0:64, 0:64]
    img += 60.0 * np.exp(
        -((xx - truth_x) ** 2 + (yy - truth_y) ** 2) / (2 * SIGMA_PSF**2)
    )

    # 该窗口确实含负像素，否则这个 fixture 判别不了 clip。
    patch, _, _ = cutout(img, 32.0, 30.0, 9)
    assert (patch < 0.0).sum() > 0, "fixture 没有负像素，无法守卫 clip"

    got = centroid_of_mass(img, np.array([[32.0, 30.0]]))
    # 守卫：这两条钉死实测坐标，变异 2 会偏离 324 倍容差。
    assert got[0, 0] == pytest.approx(31.8153454494, abs=1e-3)
    assert got[0, 1] == pytest.approx(30.0751601927, abs=1e-3)
    # 文档，非守卫（B-5）：给定上面的 ±1e-3，可达区间仅 0.5537~0.5557，不可能失败。
    # 记录的事实是：带 clip 的结果在 x 上比无 clip 更靠近真值，0.5547 对 0.8789。
    assert abs(got[0, 0] - truth_x) < 0.6


def test_centroid_leaves_seed_unchanged_on_non_positive_patch():
    """裁决 30：全非正窗口走 `total <= 0` 分支，返回点必须**等于输入种子**。

    这是有意的行为——没有可用权重时保留分割给出的位置，而不是返回 NaN、0
    或把点挪到窗口中心。此前没有任何测试钉住它。
    """
    seeds = np.array([[20.0, 30.0], [45.0, 18.0]])

    all_negative = np.full((64, 64), -5.0, dtype=np.float64)
    got = centroid_of_mass(all_negative, seeds)
    assert np.array_equal(got, seeds)

    all_zero = np.zeros((64, 64), dtype=np.float64)
    got = centroid_of_mass(all_zero, seeds)
    assert np.array_equal(got, seeds)


def test_centroid_handles_empty_input():
    img = make_image([(20.0, 30.0, 1000.0)])
    got = centroid_of_mass(img, np.empty((0, 2), dtype=np.float64))
    assert got.shape == (0, 2)
    assert got.dtype == np.float64


# --------------------------------------------------------------------------
# remeasure
# --------------------------------------------------------------------------


def test_remeasure_refines_centroid_and_rewrites_flux_and_peak():
    """remeasure 必须同时更新质心、流量与峰值，并保留 frame 与其余列。

    实测单源 (20.2, 30.4)、amp=1000：局部峰值 961.690602 与 img.max() 相对
    差 0.000000（该 fixture 里这个源**就是**全局最大，所以两者恰好相等）。
    """
    sources = [(20.2, 30.4, 1000.0)]
    img = make_image(sources)
    table = make_table(sources, frame=11)
    out = remeasure(img, table)

    assert isinstance(out, SourceTable)
    assert out.frame == 11
    assert len(out) == 1
    # 质心被精化到亚像素。
    assert out.x[0] == pytest.approx(20.2, abs=0.05)
    assert out.y[0] == pytest.approx(30.4, abs=0.05)
    # 流量换成孔径测光值。
    assert out.flux[0] == pytest.approx(ANALYTIC_TOTAL, rel=0.03)
    # 该 fixture 下局部峰值恰等于全局最大值。
    assert out.peak[0] == pytest.approx(img.max(), rel=0.02)
    # 未重测的列原样带过。
    assert out.elongation[0] == pytest.approx(1.05)
    assert out.npix[0] == 40
    assert out.npix.dtype == np.int64


def test_remeasure_peak_is_local_not_global():
    """裁决 31b：peak 是**局部**极大值，不是整幅图的最大值。

    原任务书的 `rel=0.02` 对 `img.max()` 的比较之所以成立，只因为它的 fixture
    只有一个源、而那个源就是全局最大。两源 fixture 实测：1000-amp 在 (20, 20)、
    300-amp 在 (45, 45)，全局最大 1000.0，而 (45, 45) 处的局部峰值是 300.0，
    相差 3.3 倍。局部定义必须保留——peak 要和 flux 对应同一个孔径。
    """
    sources = [(20.0, 20.0, 1000.0), (45.0, 45.0, 300.0)]
    img = make_image(sources)
    out = remeasure(img, make_table(sources))

    assert img.max() == pytest.approx(1000.0)
    order = np.argsort(out.x)          # (20,20) 在前，(45,45) 在后
    bright, faint = order[0], order[1]
    assert out.peak[bright] == pytest.approx(1000.0, rel=0.02)
    # 关键断言：暗源处的 peak 必须是 300 而不是全局的 1000。
    assert out.peak[faint] == pytest.approx(300.0, rel=0.02)
    assert out.peak[faint] < 0.5 * img.max()


def test_remeasure_does_not_modify_the_input_table():
    """输入表必须完好——八个下游任务共用 SourceTable。"""
    sources = [(20.2, 30.4, 1000.0)]
    img = make_image(sources)
    table = make_table(sources)
    before = {
        name: getattr(table, name).copy()
        for name in ("x", "y", "flux", "peak", "elongation", "npix")
    }
    remeasure(img, table)
    for name, saved in before.items():
        assert np.array_equal(getattr(table, name), saved), f"输入表的 {name} 被改动了"


def test_remeasure_output_columns_are_independent_copies():
    """裁决 31a：输出表的每一列都必须是独立副本，不能是任何输入的视图。

    原实现把 `x=xy[:, 0]`、`y=xy[:, 1]` 直接交给构造函数——那是
    `centroid_of_mass` 返回数组的视图——而 `elongation`、`npix` 却显式
    `.copy()`。同一个构造调用里混用副本与视图，对八个消费 SourceTable 的
    下游任务是个陷阱。

    **下面四条 `base is None` 是本测试唯一的守卫，任何情况下都不要删。**
    （B-1 修正：原 docstring 把强弱关系写反了，评审已实测证伪。）

    理由：`xy[:, 0]` 与 `xy[:, 1]` 是同一个 `(N, 2)` 数组的**互不重叠**的两列，
    实测 `np.shares_memory(xy[:, 0], xy[:, 1])` 为 False。所以写 `out.x[0]`
    **永远不可能**碰到 `out.y`，也碰不到 `table.x`（后者另有一层：x 来自
    `centroid_of_mass` 的返回值，那本身已是输入的副本）。评审实测：把 x/y 改回
    视图、同时删掉这四行 `base is None`，整套 **23 项全绿**——裁决 31a 的变异
    活了下来。

    因此下面那三条行为断言是**契约文档，不是守卫**：它们描述"改一列不影响
    另一列"这个下游可以依赖的性质，但在 x/y 取自同一数组的两列时它们不可能
    失败。保留是为了写明意图，别把它们当成检验手段。
    """
    sources = [(20.2, 30.4, 1000.0), (45.2, 18.1, 600.0)]
    img = make_image(sources)
    table = make_table(sources)
    out = remeasure(img, table)

    # 唯一的守卫：每一列都必须自有内存，而不是别人的视图。
    assert out.x.base is None, "out.x 仍是某个数组的视图"
    assert out.y.base is None, "out.y 仍是某个数组的视图"
    assert out.elongation.base is None, "out.elongation 仍是某个数组的视图"
    assert out.npix.base is None, "out.npix 仍是某个数组的视图"

    # 契约文档（见 docstring：以下三条在当前实现下不可能失败，不是守卫）。
    saved_y = out.y.copy()
    saved_input_x = table.x.copy()
    out.x[0] = -999.0
    assert np.array_equal(out.y, saved_y)
    assert np.array_equal(table.x, saved_input_x)

    # B-4：elongation/npix 与输入表**是同一形状同一 dtype**，`base is None`
    # 对它们是瞎的——`SourceTable.__post_init__` 的 `np.asarray` 在已经是
    # float64/int64 连续数组上是恒等操作（实测返回同一对象、base 仍为 None）。
    # 所以去掉 remeasure 里的 `.copy()` 后，输入输出会是**同一个数组**，
    # 这时只有下面这种行为断言能抓到（与上面 x/y 的情形不同：那是兄弟列，
    # 这里是同一块内存）。
    out.elongation[0] = -1.0
    assert table.elongation[0] == pytest.approx(1.05), "out.elongation 与输入表共享内存"
    out.npix[0] = -7
    assert table.npix[0] == 40, "out.npix 与输入表共享内存"


def test_remeasure_handles_empty_table():
    img = make_image([(20.0, 30.0, 1000.0)])
    out = remeasure(img, SourceTable.empty(frame=5))
    assert len(out) == 0
    assert out.frame == 5
    assert out.xy.shape == (0, 2)
    assert out.npix.dtype == np.int64


def test_remeasure_columns_are_plain_arrays():
    """输出必须是无单位的纯 ndarray，能直接喂 cKDTree。"""
    from scipy.spatial import cKDTree

    sources = [(20.2, 30.4, 1000.0), (45.2, 18.1, 600.0)]
    img = make_image(sources)
    out = remeasure(img, make_table(sources))
    for name in ("x", "y", "flux", "peak", "elongation", "npix"):
        arr = getattr(out, name)
        assert type(arr) is np.ndarray, f"{name} 不是纯 ndarray"
        assert not hasattr(arr, "unit"), f"{name} 泄漏了单位"
    cKDTree(out.xy).query(np.array([[20.2, 30.4]]))


# --------------------------------------------------------------------------
# 配置一致性
# --------------------------------------------------------------------------


def test_signature_defaults_match_config(cfg):
    """签名默认值必须来自 src/config/default.yaml，src/** 里不留魔数。"""
    radius = cfg["detect"]["aperture_radius_px"]
    box = cfg["psf"]["box"]
    assert radius == 5.0
    assert box == 9
    assert box % 2 == 1, "psf.box 必须是奇数，否则 cutout 会拒绝它"

    for fn in (aperture_flux, remeasure):
        assert inspect.signature(fn).parameters["radius"].default == radius
    for fn in (centroid_of_mass, remeasure):
        assert inspect.signature(fn).parameters["box"].default == box


def test_detect_package_does_not_import_truth():
    """真值隔离（硬约束）：src/detect/ 不得引入 src.validate.truth。

    B-6：原实现只读 `centroid.py` 一个文件，而 docstring 声称覆盖整个
    `src/detect/`——于是 `segmentation.py` 实际无人看守。这是全仓库**唯一**
    一条真值隔离测试，所以它必须遍历 `src/detect/` 下的每个 `*.py`。
    真值 `.DAT` 只允许在校验/定标环节使用；探测环节若能读到真值，
    整条流水线的复现率数字就失去意义。
    """
    from pathlib import Path

    import src.detect as pkg

    package_dir = Path(pkg.__file__).parent
    modules = sorted(package_dir.rglob("*.py"))
    # 目录本身必须非空，否则这条测试会在什么都没查的情况下通过。
    assert len(modules) >= 2, f"src/detect/ 下只找到 {len(modules)} 个模块，遍历可能失效"

    for path in modules:
        source = path.read_text(encoding="utf-8")
        assert "validate.truth" not in source, f"{path.name} 引用了 src.validate.truth"
        assert "validate import truth" not in source, (
            f"{path.name} 从 src.validate 引入了 truth"
        )
