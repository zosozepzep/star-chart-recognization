"""跨帧可复现性验证器的测试。

本文件的核心职责是把两个**同名不同义**的量分开钉住：``reproducibility``
（参考帧的源里有多少在多帧复现）与 ``purity``（一帧典型探测里有多少是真源）。
裁决 66 记录的缺陷正是二者共用一个名字，而它们只在「参考帧探测数恰等于各帧
均值」时相等——本文件所有合成 fixture 里大多恰好满足这个条件，所以
``test_reproducibility_and_purity_diverge_on_unequal_frames`` 这一条
（100/200 不对称）才是区分二者的那条测试，其余几条不区分。
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.validate.repeatability import (
    RepeatabilityCurve,
    RepeatabilityPoint,
    chance_match_probability,
    cross_frame_repeatability,
    expected_neighbours_per_source,
    min_hits_for,
    scan_thresholds,
    single_frame_match_probability,
)

FIELD = (4096, 4096)


# --------------------------------------------------------------------------
# 原任务书 Step 1 的八条（裁决 66 把返回值改成 3 元组，断言数字不变）
# --------------------------------------------------------------------------


def test_all_real_sources_are_reproducible():
    """三颗真源在 6 帧里各自抖动 0.3 px，应全部复现。"""
    base = np.array([[100.0, 100.0], [500.0, 700.0], [2000.0, 3000.0]])
    rng = np.random.default_rng(1)
    frames = [base + rng.normal(0.0, 0.3, size=base.shape) for _ in range(6)]
    n, rep, purity = cross_frame_repeatability(frames, match_radius=3.0)
    assert n == 3
    assert rep == pytest.approx(1.0, abs=0.01)
    assert purity == pytest.approx(1.0, abs=0.01)


def test_pure_noise_is_not_reproducible():
    """每帧 200 个均匀随机点，复现率应崩到 0.1 以下——这是「263822 颗星」的反证。"""
    rng = np.random.default_rng(2)
    frames = [rng.uniform(0.0, 4096.0, size=(200, 2)) for _ in range(6)]
    n, rep, purity = cross_frame_repeatability(frames, match_radius=3.0)
    assert rep < 0.1
    assert purity < 0.1
    assert n < 20


def test_mixed_population_recovers_real_fraction():
    """30 颗真源 + 每帧 170 个随机假源，复现率应接近 30/200 = 0.15。

    本任务最强的一条证据：判据从 200 个探测里挑出的正是那 30 颗，
    而不是靠调容差凑出的比例。
    """
    rng = np.random.default_rng(3)
    real = rng.uniform(0.0, 4096.0, size=(30, 2))
    frames = [
        np.vstack([real + rng.normal(0.0, 0.3, size=real.shape),
                   rng.uniform(0.0, 4096.0, size=(170, 2))])
        for _ in range(6)
    ]
    n, rep, purity = cross_frame_repeatability(frames, match_radius=3.0)
    assert n == pytest.approx(30, abs=4)
    assert rep == pytest.approx(0.15, abs=0.04)
    # 每帧都是 200 个源，所以此处 purity 与 rep 相等，无法区分两者（见模块 docstring）。
    assert purity == pytest.approx(0.15, abs=0.04)


def test_min_fraction_controls_strictness():
    """min_fraction 0.4 与 0.9 在 6 帧上分别解析成 hits>=3 与 hits>=6，给出 20 与 10。"""
    rng = np.random.default_rng(4)
    real = rng.uniform(0.0, 4096.0, size=(20, 2))
    frames = []
    for k in range(6):
        pts = real.copy()
        if k >= 3:
            pts = pts[:10]  # 后半段只有一半源出现
        frames.append(pts + rng.normal(0.0, 0.2, size=pts.shape))
    loose, _, _ = cross_frame_repeatability(frames, match_radius=3.0, min_fraction=0.4)
    strict, _, _ = cross_frame_repeatability(frames, match_radius=3.0, min_fraction=0.9)
    assert loose > strict
    # 钉住这条测试真正在测的两个门限值，否则 loose>strict 在很多实现下恒真。
    assert min_hits_for(0.4, 6) == 3
    assert min_hits_for(0.9, 6) == 6
    assert (loose, strict) == (20, 10)


def test_empty_input_raises():
    """少于 2 帧无法谈「跨帧」。"""
    with pytest.raises(ValueError, match="至少需要 2 帧"):
        cross_frame_repeatability([])
    with pytest.raises(ValueError, match="至少需要 2 帧"):
        cross_frame_repeatability([np.zeros((3, 2))])


def test_curve_recommended_picks_lowest_sigma_above_threshold():
    """良态曲线上，单调合格后缀与原来的 min() 给出同一个答案 4.0（裁决 69）。"""
    curve = _curve(
        [(2.0, 2284.0, 0, 0.088), (3.0, 169.0, 0, 0.769),
         (4.0, 116.0, 0, 0.888), (5.0, 95.0, 0, 0.905)]
    )
    pick = curve.recommended(min_reproducibility=0.85)
    assert pick.n_sigma == 4.0


def test_curve_recommended_raises_when_none_qualify():
    """无一档达标时必须抛错，不得退化成「返回最好的那个」。"""
    curve = _curve([(3.0, 500.0, 20, 0.04)])
    with pytest.raises(ValueError, match="没有阈值达到复现率"):
        curve.recommended(min_reproducibility=0.85)


def test_curve_to_dict_is_json_ready():
    """to_dict 必须可 json 序列化，且带上裁决 66 新增的两个字段。

    **fixture 用 numpy 标量构造 ``frames`` 与 ``match_radius_px``，这是有意的。**
    原来这条整个用 python 原生 int/float 构造，于是 ``json.dumps`` **必然**成功
    ——它测不到真实通路。真实调用方给的是 numpy 标量：帧号来自 numpy 索引
    （``np.int64``），半径来自配置解析或算术（``np.float64``）。实测未转型的
    ``to_dict()`` 在 ``frames=[np.int64(16), np.int64(17)]`` 上抛
    ``TypeError: Object of type int64 is not JSON serializable``。
    point 内六个字段在 ``scan_thresholds`` 里已转过，这两个字段原先漏了。

    ``match_radius_px`` 那一格必须断言 ``type(...) is float`` 而不能只靠
    ``json.dumps`` 兜住：``np.float64`` **是** ``float`` 的子类，未转型也能序列化，
    所以只有类型断言钉得住它。``np.int64`` 不是 ``int`` 的子类，``frames`` 那一格
    ``dumps`` 就会炸，两格的牙齿来源不同。
    """
    curve = RepeatabilityCurve(
        points=[
            RepeatabilityPoint(
                n_sigma=5.0, n_detected_mean=95.0, n_reproducible=76,
                n_reference=95, reproducibility=0.8, purity=0.762,
            )
        ],
        frames=[np.int64(16), np.int64(17)],
        match_radius_px=np.float64(3.0),
    )
    d = curve.to_dict()
    json.loads(json.dumps(d))
    # 两个曾经原样透传的字段必须已是原生类型，否则上一行在真实调用方那里会炸。
    assert [type(f) for f in d["frames"]] == [int, int]
    assert type(d["match_radius_px"]) is float
    assert d["points"][0]["n_sigma"] == 5.0
    assert set(d["points"][0]) == {
        "n_sigma", "n_detected_mean", "n_reproducible",
        "n_reference", "reproducibility", "purity",
    }


def _curve(rows) -> RepeatabilityCurve:
    """从 (n_sigma, n_detected_mean, n_reproducible, reproducibility) 造曲线。

    ``n_reference`` 取 ``n_detected_mean`` 的整数值、``purity`` 由 ``n / mean`` 反算，
    仅为让字段都有值；裁决 286 已裁定这些数字是示意而非实测，
    所以不得把它们当作任何数据集上的测量值引用。

    ``n_reproducible`` 槽位上的填充值是**任意的**：本文件所有用 ``_curve`` 的测试
    只断言 ``recommended()`` 选出的 ``n_sigma``，没有一条读这个字段。它必须任意，
    因为它无法从任何实测推出——而它恰是会被误读为「星数」的那个字段，裁决 286
    的明文禁令（「不得把示意星数写进任何测试、docstring 或报告」）正是针对它。
    所以这里一律填 ``0`` 或与实测无关的小整数，而**不是**任务书草稿里的示意星数。

    与此相对，``(n_sigma, reproducibility)`` 对**不是**任意的：
    ``test_curve_recommended_picks_lowest_sigma_above_threshold`` 必须仍返回 4.0
    （裁决 69），``test_recommended_skips_a_dip_...`` 的凹陷位置也由该裁决固定。

    **注意：本函数造的曲线不满足字段间的自洽关系**
    （``n_reproducible / n_reference != reproducibility``，因为前者是任意填充值、
    后者是手填的示意值）。所以裁决 375 要求的自洽断言只能挂在由 ``scan_thresholds``
    真实构造的曲线上，见 ``test_real_curve_fields_are_self_consistent``。
    """
    points = [
        RepeatabilityPoint(
            n_sigma=s,
            n_detected_mean=mean,
            n_reproducible=n,
            n_reference=int(round(mean)),
            reproducibility=rep,
            purity=n / mean,
        )
        for s, mean, n, rep in rows
    ]
    return RepeatabilityCurve(points=points, frames=[16, 17], match_radius_px=3.0)


# --------------------------------------------------------------------------
# 裁决 66：reproducibility 与 purity 必须是两个量
# --------------------------------------------------------------------------


def test_reproducibility_equals_purity_on_equal_frame_counts():
    """各帧探测数相等时两者相等——**这是恒等式，不是性质**。

    ``mean(len(f) for f in frames) == len(frames[0])`` 在此成立，两个分母就
    是同一个数，所以这条断言在任何实现下都真。它的价值只在于记录二者相等的
    **充分条件**：裁决 66 的缺陷之所以在原有测试里全程不可见，正是因为每个
    fixture 都落在这个条件上。牙齿在下一条测试里。
    """
    real = np.random.default_rng(10).uniform(0.0, 4096.0, size=(100, 2))
    frames = [real.copy() for _ in range(6)]
    n, rep, purity = cross_frame_repeatability(frames, match_radius=3.0)
    assert n == 100
    assert rep == pytest.approx(1.0)
    assert purity == pytest.approx(1.0)
    assert rep == purity
    assert float(np.mean([len(f) for f in frames])) == float(len(frames[0]))


def test_reproducibility_and_purity_diverge_on_unequal_frames():
    """参考帧 100 源、其余五帧各 200 源且全部复现：复现率 1.000，纯度 0.545。

    这是区分两个量的那一条。原公式只报 0.545，读起来像「45% 没复现」，
    而实际上参考帧的每一个源都在全部 6 帧里出现了（裁决 66 的第 2 条反例）。
    """
    rng = np.random.default_rng(11)
    real = rng.uniform(0.0, 4096.0, size=(100, 2))
    frames = [real.copy()]
    for _ in range(5):
        extra = rng.uniform(0.0, 4096.0, size=(100, 2))
        frames.append(np.vstack([real.copy(), extra]))

    n, rep, purity = cross_frame_repeatability(frames, match_radius=3.0)
    assert n == 100
    assert rep == pytest.approx(1.0)
    assert purity == pytest.approx(0.545, abs=0.01)
    # 分母确实不同：1100/6 = 183.3333 对 100。
    assert float(np.mean([len(f) for f in frames])) == pytest.approx(183.3333, abs=1e-3)


def test_one_empty_frame_does_not_inflate_purity_above_one():
    """五帧健康 + 一帧全空：复现率 1.000，纯度 1.000，**不得**是 1.200。

    裁决 66 第 1 条：空帧在命中统计里被跳过，却仍以 0 参与均值，把原公式的
    分母压到 83.3333，纯度因此报成 1.2000——一个完全失败的帧把质量指标
    **抬高 20%**。实现必须把空帧从纯度分母里排除，本条断言 ``purity == 1.0``
    在保留空帧的实现上会拿到 1.2 而失败，牙齿就在这里。
    """
    real = np.random.default_rng(12).uniform(0.0, 4096.0, size=(100, 2))
    frames = [real.copy() for _ in range(5)] + [np.zeros((0, 2))]

    n, rep, purity = cross_frame_repeatability(frames, match_radius=3.0)
    assert n == 100
    assert rep == pytest.approx(1.0)
    assert purity == pytest.approx(1.0)
    assert purity <= 1.0
    # 把「含空帧的分母」这个错法算出来，钉住它与正确值差 20%。
    naive_mean = float(np.mean([len(f) for f in frames]))
    assert naive_mean == pytest.approx(83.3333, abs=1e-3)
    assert n / naive_mean == pytest.approx(1.2, abs=1e-3)


@pytest.mark.parametrize("seed,n_real,n_noise", [(20, 100, 0), (21, 30, 170), (22, 0, 200)])
def test_reproducibility_never_exceeds_one(seed, n_real, n_noise):
    """复现率的分母是参考帧自身的源数，所以恒在 [0, 1]。

    **本断言是结构恒等式，不构成对分母选择的约束**（如实标注，非缺陷）：
    ``n_reproducible = count_nonzero(hits >= min_hits)`` 而 ``len(hits) ==
    len(reference) == n_reference``，所以 ``0 <= rep <= 1`` 在任何分母**只要仍是
    ``n_reference``** 的实现下恒成立。它不会因为分母被换成各帧均值而变红——
    本条三组参数的各帧探测数实测全部相等（[100]×6、[200]×6、[200]×6），
    两个分母同值。这也解释了「复现率改用帧均值分母」那个变异体为何只被三条
    **帧长不等**的测试杀掉（``..._diverge_on_unequal_frames``、
    ``..._empty_frame_does_not_inflate_purity``、``..._dropout_caps_the_count``）。

    保留它是因为它是一条廉价的回归护栏（值域越界、异常、返回类型变化都会被它抓到），
    但**它不算作两个量分开的证据**。
    """
    rng = np.random.default_rng(seed)
    real = rng.uniform(0.0, 4096.0, size=(n_real, 2))
    frames = []
    for _ in range(6):
        pts = [real + rng.normal(0.0, 0.3, size=real.shape)] if n_real else []
        if n_noise:
            pts.append(rng.uniform(0.0, 4096.0, size=(n_noise, 2)))
        frames.append(np.vstack(pts) if pts else np.zeros((0, 2)))
    _, rep, _ = cross_frame_repeatability(frames, match_radius=3.0)
    assert 0.0 <= rep <= 1.0


def test_empty_reference_frame_returns_zeros():
    """参考帧为空时统计**未定义**，按约定返回 (0, 0.0, 0.0)。

    这不是「复现率为零」——其余五帧可能满是真源。Step 5 因此不得挑一个
    探测数为零的帧做参考帧。
    """
    rng = np.random.default_rng(13)
    frames = [np.zeros((0, 2))] + [rng.uniform(0.0, 4096.0, size=(50, 2)) for _ in range(5)]
    assert cross_frame_repeatability(frames, match_radius=3.0) == (0, 0.0, 0.0)


def test_reference_dropout_caps_the_count():
    """100 颗真源、参考帧只探到 80：计数上限就是 80，所以报的是**下界**。

    这是参考帧法固有的不对称，方向对保守论证有利（宁少报不多报），
    所以按设计保留并在 docstring 里写明，而不是「修掉」。
    """
    rng = np.random.default_rng(14)
    real = rng.uniform(0.0, 4096.0, size=(100, 2))
    frames = [real[:80] + rng.normal(0.0, 0.2, size=(80, 2))]
    for _ in range(5):
        frames.append(real + rng.normal(0.0, 0.2, size=real.shape))

    n, rep, _ = cross_frame_repeatability(frames, match_radius=3.0)
    assert n == 80
    assert rep == pytest.approx(1.0)  # 80/80——参考帧漏掉的那 20 个不在分母里


def test_many_to_one_matching_is_possible_and_tolerated():
    """两个相距 1 px 的参考源可被同一个探测同时匹配上，实测 n=2。

    最近邻匹配不是互为最近邻匹配。按实测的偶然密度这种情形可忽略
    （见 chance_match_probability 的量级），故如实记录而不加双向匹配。
    """
    reference = np.array([[100.0, 100.0], [101.0, 100.0]])
    frames = [reference] + [np.array([[100.5, 100.0]]) for _ in range(5)]
    n, _, _ = cross_frame_repeatability(frames, match_radius=3.0)
    assert n == 2


# --------------------------------------------------------------------------
# 裁决 69：recommended() 必须取单调合格后缀
# --------------------------------------------------------------------------


def test_recommended_skips_a_dip_and_picks_the_monotonic_suffix():
    """曲线在 4σ 凹到 0.60 时，推荐值必须是 5.0 而不是 3.0。

    原实现取合格集里最小的 sigma，会选中 3.0——而它的**上邻居** 4.0
    只有 0.60，低于门限。一个上邻居不合格的点不是操作点（裁决 69）。
    """
    curve = _curve(
        [(2.0, 3000.0, 300, 0.10), (3.0, 200.0, 176, 0.88),
         (4.0, 150.0, 90, 0.60), (5.0, 100.0, 91, 0.91)]
    )
    assert curve.recommended(min_reproducibility=0.85).n_sigma == 5.0
    # 钉住「原实现会选 3.0」这一事实，否则本测试与上一条无法区分。
    lowest_qualifying = min(
        (p for p in curve.points if p.reproducibility >= 0.85), key=lambda p: p.n_sigma
    )
    assert lowest_qualifying.n_sigma == 3.0


def test_recommended_selects_on_reproducibility_not_purity():
    """两个字段取值相反时，recommended 必须听 reproducibility 的。

    参数名就叫 min_reproducibility，字段必须与名字对上（裁决 66 末条）。
    """
    points = [
        RepeatabilityPoint(
            n_sigma=3.0, n_detected_mean=100.0, n_reproducible=20,
            n_reference=100, reproducibility=0.20, purity=0.95,
        ),
        RepeatabilityPoint(
            n_sigma=5.0, n_detected_mean=100.0, n_reproducible=90,
            n_reference=100, reproducibility=0.90, purity=0.10,
        ),
    ]
    curve = RepeatabilityCurve(points=points, frames=[0, 1], match_radius_px=3.0)
    assert curve.recommended(min_reproducibility=0.85).n_sigma == 5.0


# --------------------------------------------------------------------------
# 裁决 70 / 284 / 285：偶然重合下界——这是论证本身
# --------------------------------------------------------------------------


def test_min_hits_is_inclusive_of_the_reference_frame():
    """min_fraction -> min_hits 的换算表（6 帧），「>=4 of 6」= min_fraction 0.6。

    裁决 284 的全部分歧就是这个 off-by-one：min_hits **含参考帧本身**。
    """
    assert [min_hits_for(f, 6) for f in (0.5, 0.6, 0.9, 1.0)] == [3, 4, 6, 6]
    assert min_hits_for(0.0, 6) == 2  # 下限 2：至少要有两帧才谈得上跨帧
    assert min_hits_for(0.6, 2) == 2


def test_chance_match_table_reproduces():
    """裁决 70 的三行：4096² 场、3 px 半径下的期望邻居数与单帧偶然匹配概率。"""
    rows = {124: (0.00020897, 0.00020895),
            200: (0.00033706, 0.00033700),
            2284: (0.00384918, 0.00384178)}
    for n_det, (lam, p_one) in rows.items():
        assert expected_neighbours_per_source(n_det, 3.0, FIELD) == pytest.approx(lam, abs=5e-9)
        assert single_frame_match_probability(n_det, 3.0, FIELD) == pytest.approx(p_one, abs=5e-9)


def test_chance_reproducible_count_at_the_report_threshold():
    """报数阈值（124 源/帧、>=4 of 6）的偶然可复现源期望是 1.13e-8，远低于 1e-6。

    这条是「3 像素半径跨 4 帧不可能是偶然」这句话的定量形式，也是
    「为什么只报 10² 颗星」的正面回答。
    """
    got = chance_match_probability(124, 3.0, FIELD, n_frames=6, min_hits=min_hits_for(0.6, 6))
    assert got < 1e-6
    assert got == pytest.approx(1.13e-8, rel=0.02)


def test_chance_reproducible_count_at_two_sigma():
    """2σ（2284 源/帧）下仍只有 1.29e-3——**不是** 1.9e-9。

    1.9e-9 是 min_hits=6（每一个邻居帧都偶然匹配）那一档；本模块用的判据是
    >=4 of 6，诚实的数字是 1.29e-3，比它大六个量级（裁决 284）。
    所以断言写 < 5e-3，不写 < 1e-3——后者在本判据下是红的。
    """
    got = chance_match_probability(2284, 3.0, FIELD, n_frames=6, min_hits=min_hits_for(0.6, 6))
    assert got < 5e-3
    assert got == pytest.approx(1.29e-3, rel=0.02)


@pytest.mark.parametrize("min_hits,expected", [(4, 1.287624e-03), (5, 2.480055e-06), (6, 1.911442e-09)])
def test_chance_count_falls_steeply_with_min_hits(min_hits, expected):
    """min_hits 每加一档，偶然重合期望掉两到三个量级（裁决 284 的表）。"""
    got = chance_match_probability(2284, 3.0, FIELD, n_frames=6, min_hits=min_hits)
    assert got == pytest.approx(expected, rel=1e-4)


def test_real_to_chance_margin_is_five_orders_not_nine():
    """2σ 上真实可复现源约 201 个、偶然 1.29e-3，余量 **5.2** 个量级。

    裁决 285：原先写的「九阶」在任何 min_hits 下都不成立（4/5/6 三档分别是
    5.2 / 7.9 / 11.0）。5.2 阶已足以说明均匀密度近似不承重，但九不是这个
    方法能产出的数字。
    """
    real = 0.088 * 2284  # 2σ 纯度 x 探测数 = 201.0
    assert real == pytest.approx(201.0, abs=0.5)
    false = chance_match_probability(2284, 3.0, FIELD, n_frames=6, min_hits=4)
    assert math.log10(real / false) == pytest.approx(5.2, abs=0.05)


def test_chance_count_scales_with_field_area():
    """偶然密度按场面积缩放，所以 image_shape 必须取自数据而不能假定 4096²。

    同样 2284 个探测放进 2048² 场，期望邻居数从 0.00385 涨到 0.01540（恰 4 倍）；
    而 4096x4136 只挪到 0.00381——面积的量级要紧，具体帧尺寸不要紧。
    """
    big = expected_neighbours_per_source(2284, 3.0, (4096, 4096))
    small = expected_neighbours_per_source(2284, 3.0, (2048, 2048))
    tall = expected_neighbours_per_source(2284, 3.0, (4136, 4096))
    assert big == pytest.approx(0.00385, abs=5e-6)
    assert small == pytest.approx(0.01540, abs=5e-6)
    assert small / big == pytest.approx(4.0, rel=1e-9)
    assert tall == pytest.approx(0.00381, abs=5e-6)


# --------------------------------------------------------------------------
# 裁决 372：四个参数守卫的覆盖。被保护的性质是**非法输入不得静默产出可流下游的
# 数值**——不是「守卫存在」。四处的失效模式都不留痕迹，所以只有守卫拦得住。
# --------------------------------------------------------------------------


def test_negative_field_dimension_cannot_yield_a_negative_density():
    """负的图幅尺寸不得静默给出负的期望邻居数——那会让下游算出负概率。

    被保护的性质：非法 ``image_shape`` 不得产出可流下游的数值。这里的失效**完全
    隐形**：实测去掉守卫后 ``(-4096, 4096)`` 给出 λ = **−0.0038491832367892136**，
    即负的期望邻居数，而 ``single_frame_match_probability`` 的 ``1 - exp(-λ)`` 把它
    变成 **−0.003856600856790404**——一个负概率，没有任何异常。更隐形的是
    ``(-4096, -4096)``：两负相乘，λ 与正确尺寸 ``(4096, 4096)``
    **逐位相同**（0.0038491832367892136），符号错误不留一丝痕迹。

    ``(0, 4096)`` 那一档去掉守卫会立刻抛 ``ZeroDivisionError``（实测，不是 numpy 的
    inf + warning），性质上同样是「异常终止、不产出数值」，所以裁决 372 判它是等价
    变异、**不为它写断言**——但守卫仍保留，因为带 ``image_shape`` 元组的消息比
    ``float division by zero`` 有诊断价值。
    """
    with pytest.raises(ValueError):
        expected_neighbours_per_source(2284, 3.0, (-4096, 4096))


def test_negative_match_radius_cannot_be_laundered_by_squaring():
    """负半径不得静默通过——r 被平方，符号错误在结果里完全看不见。

    被保护的性质：非法 ``match_radius_px`` 不得产出可流下游的数值。实测去掉守卫后
    ``-3.0`` 与 ``+3.0`` 给出的 λ **逐位相同**（0.0038491832367892136），
    所以一个符号写错的半径会一路算到底、给出看起来完全正常的数字。

    ``0.0`` 那一档去掉守卫给出 ``0.0``，**恰是正确的极限值**（半径为零时半径内确实
    没有邻居），裁决 372 判它是等价变异、不为它写断言。
    """
    with pytest.raises(ValueError):
        expected_neighbours_per_source(2284, -3.0, FIELD)


def test_single_frame_cannot_be_reported_as_zero_chance_coincidence():
    """帧数 < 2 时偶然重合判据未定义，不得静默返回 0.0。

    被保护的性质：非法 ``n_frames`` 不得产出可流下游的数值。实测去掉守卫后
    ``n_frames = 1 / 0 / -5`` **全部静默返回 0.0**——读作「绝无偶然重合」，
    这是本项目最不能容忍的方向（整个可复现性判据的存在理由就是不许高报星数），
    而真相是「跨帧」判据在一帧上根本没有定义。
    """
    with pytest.raises(ValueError):
        chance_match_probability(2284, 3.0, FIELD, n_frames=1, min_hits=2)


def test_min_hits_outside_the_valid_range_cannot_produce_a_count():
    """``min_hits`` 越出 ``[2, n_frames]`` 时不得静默产出源数。

    被保护的性质：非法 ``min_hits`` 不得产出可流下游的数值。两侧都危险，
    且**两个断言合在一条测试里**，这样它同时钉住「守卫存在」与「区间下界是 2 而
    不是 1」（裁决 372 判下界挪一格**也是**漏洞，不是等价变异）：

    - ``min_hits = 1`` 去掉守卫实测返回 **2284.0**，即 ``n_detections`` 本身
      （二项尾从 k=0 起求和恒为 1）。读作「每一个探测都算偶然可复现」，
      而 ``min_hits = 1`` 意味着「只要参考帧自己看到就算复现」——判据退化成单帧探测。
    - ``min_hits = 7 > n_frames = 6`` 去掉守卫实测静默返回 **0.0**，
      与「判据合法而偶然重合确实极低」同形；真相是判据结构上恒为空
      （``hits`` 的上界就是 ``n_frames``）。
    """
    with pytest.raises(ValueError):
        chance_match_probability(2284, 3.0, FIELD, n_frames=6, min_hits=1)
    with pytest.raises(ValueError):
        chance_match_probability(2284, 3.0, FIELD, n_frames=6, min_hits=7)


# --------------------------------------------------------------------------
# 裁决 373 / 374：cross_frame_repeatability 的两个入口守卫
# --------------------------------------------------------------------------


@pytest.mark.parametrize("min_fraction", [1.0000001, 1.1, 1.5, 2.0])
def test_min_fraction_above_one_is_rejected_not_reported_as_zero(min_fraction):
    """``min_fraction > 1`` 时判据结构上恒为空，不得静默返回 ``(0, 0.0, 0.0)``。

    被保护的性质：非法 ``min_fraction`` 不得产出可流下游的比值。实测加守卫前
    ``1.0000001 / 1.1 / 1.5 / 2.0`` 全部返回 ``(0, 0.0, 0.0)``，而同一 fixture 换成
    纯噪声（判据完全合法、只是一个源都没复现）返回的**也是** ``(0, 0.0, 0.0)``
    ——调用方无从分辨「数据全军覆没」与「你的参数使判据不可能被满足」。

    而这不是「碰巧为 0」，是结构性的：``hits`` 的上界可证为 ``len(frames)``
    （参考帧自记 1 次，其余至多 ``n_frames - 1`` 次命中），所以
    ``min_hits > n_frames`` 时 ``hits >= min_hits`` 恒为空集。
    ``1.0`` 与 ``1.0000001`` 之间没有任何缓冲，而从 YAML 读浮点数出现这个量级的
    漂移很常见（裁决 373）。
    """
    real = np.random.default_rng(30).uniform(0.0, 4096.0, size=(10, 2))
    frames = [real.copy() for _ in range(6)]
    with pytest.raises(ValueError, match=r"min_fraction 必须落在"):
        cross_frame_repeatability(frames, match_radius=3.0, min_fraction=min_fraction)
    # 合法上界 1.0 仍须放行，守卫不得把区间收得比 (0, 1] 更窄。
    assert cross_frame_repeatability(frames, match_radius=3.0, min_fraction=1.0) == (
        10, 1.0, 1.0,
    )


@pytest.mark.parametrize("min_fraction", [0.0, -1.0])
def test_min_fraction_at_or_below_zero_cannot_inflate_reproducibility(min_fraction):
    """``min_fraction <= 0`` 不得被静默夹成最松判据——那个方向会**虚高**复现率。

    被保护的性质同上条。实测加守卫前 ``0.0`` 与 ``-1.0`` 都返回
    ``(10, 1.0, 1.0)``：``min_hits_for`` 的 ``max(2, ...)`` 把它们夹成 2，
    即最松的「≥2 of 6」判据。方向恰好是本项目最怕的那一侧——整个可复现性判据
    的存在理由就是不许高报星数（裁决 373）。
    """
    real = np.random.default_rng(31).uniform(0.0, 4096.0, size=(10, 2))
    frames = [real.copy() for _ in range(6)]
    with pytest.raises(ValueError, match=r"min_fraction 必须落在"):
        cross_frame_repeatability(frames, match_radius=3.0, min_fraction=min_fraction)


@pytest.mark.parametrize(
    "match_radius", [0.0, -3.0, float("nan"), float("inf"), float("-inf")]
)
def test_non_positive_or_infinite_match_radius_is_rejected(match_radius):
    """半径必须为正的有限值——``inf`` 那一档会给出「完美」的 100% 复现率。

    被保护的性质：非法 ``match_radius`` 不得产出可流下游的比值。这个函数原先
    **完全没有**半径守卫（守卫只在 ``expected_neighbours_per_source``，而本函数
    不调它）。实测加守卫前，6 帧 × 10 个全同源的 fixture 上：

    - ``0.0 / -3.0 / nan`` → ``(0, 0.0, 0.0)``，与「没复现」同形；
    - ``inf`` → ``(10, 1.0, 1.0)``——**把判据放到最松的极限，输出的却是最漂亮的
      数字**：完美的 100% 复现率与纯度。这个数会直接进对外材料（裁决 374）。

    ``nan`` 那一档是守卫**写法**的牙齿所在：条件必须写
    ``not match_radius > 0.0`` 而不是 ``match_radius <= 0.0``，因为
    ``nan <= 0.0`` 实测为 ``False``，正向写法会把 nan 放行。
    """
    real = np.random.default_rng(32).uniform(0.0, 4096.0, size=(10, 2))
    frames = [real.copy() for _ in range(6)]
    with pytest.raises(ValueError, match=r"匹配半径必须为正的有限值"):
        cross_frame_repeatability(frames, match_radius=match_radius, min_fraction=0.6)


def test_distance_exactly_at_the_match_radius_is_not_a_hit():
    """``dist == match_radius`` 恰好相等时**不算**命中：判据是半开区间 ``[0, r)``。

    实现用严格不等号 ``dist < match_radius``。半开区间本身是合理选择，问题在于
    这个约定原先**不是有意钉住的**——把 ``<`` 改成 ``<=`` 全模块无一条测试变红。
    本条就是那个约定的锚。

    构造是精确的、不依赖浮点容差：``(100.0 + 3.0) - 100.0 == 3.0`` 在双精度下
    精确成立，而 KD-tree 对轴向单点查询给出的距离实测也精确等于 ``3.0``
    （``np.float64(3.0)``）。所以边界那一侧不存在「差一个 ulp」的模糊。

    两个方向都断言，否则「恰好在边界上不命中」与「fixture 根本没构造对」不可区分：
    距离恰 3.0 给 ``(0, 0.0, 0.0)``，把它挪到 2.999 就给 ``(1, 1.0, 1.0)``。
    """
    reference = np.array([[100.0, 100.0]])
    assert (100.0 + 3.0) - 100.0 == 3.0
    on_edge = np.array([[103.0, 100.0]])
    assert cKDTree(on_edge).query(reference)[0][0] == 3.0

    frames_on_edge = [reference] + [on_edge.copy() for _ in range(5)]
    assert cross_frame_repeatability(
        frames_on_edge, match_radius=3.0, min_fraction=0.6
    ) == (0, 0.0, 0.0)

    just_inside = np.array([[102.999, 100.0]])
    frames_inside = [reference] + [just_inside.copy() for _ in range(5)]
    assert cross_frame_repeatability(
        frames_inside, match_radius=3.0, min_fraction=0.6
    ) == (1, 1.0, 1.0)


def test_nan_reproducibility_is_not_treated_as_qualifying():
    """``recommended()`` 不得把 ``nan`` 复现率当作达标。

    实测加修法前：曲线 ``[(3.0, 0.90), (5.0, nan)]``、门限 0.85 → 返回
    **``n_sigma = 3.0``**。机理是 ``nan < 0.85`` 为 ``False``，正向写法不 break，
    于是 nan 档被当作达标并进入合格后缀，循环继续往低 σ 走把 3.0 也纳进来。
    反向写法 ``not (nan >= 0.85)`` 实测为 ``True``，确实 break。

    **条件性措辞（裁决 377）：nan 在当前通路上不可达，这不是今天会出错的缺陷。**
    ``cross_frame_repeatability`` 对空参考帧走的是早返回、返回 ``(0, 0.0, 0.0)``
    而不是按定义算 ``0/0``（本文件
    ``test_empty_reference_frame_returns_zeros`` 钉住了这一点）。本条断言防的是
    **那条早返回被改动**，或从 ``to_dict()`` 反序列化重建曲线时 nan 从外部流进来。

    只断言异常类型与「没有阈值达到复现率」这个前缀，**不断言消息里的最高值**：
    实测本 fixture 上消息是「…最高为 0.900」，因为 ``max()`` 在有 nan 时的返回值
    取决于比较顺序，于是消息自称最高 0.900 ≥ 0.85 而又说没有阈值达标、自相矛盾。
    这是消息措辞的既有瑕疵（与裁决 377 附记的重复 ``n_sigma`` 那条同源），
    不是本条要钉的性质，也不该被断言锁住。
    """
    curve = RepeatabilityCurve(
        points=[
            RepeatabilityPoint(
                n_sigma=3.0, n_detected_mean=100.0, n_reproducible=90,
                n_reference=100, reproducibility=0.90, purity=0.90,
            ),
            RepeatabilityPoint(
                n_sigma=5.0, n_detected_mean=100.0, n_reproducible=0,
                n_reference=100, reproducibility=float("nan"), purity=float("nan"),
            ),
        ],
        frames=[0, 1],
        match_radius_px=3.0,
    )
    # 最高档是 nan，它不达标 -> 后缀为空 -> 抛错。不得返回 3.0σ。
    with pytest.raises(ValueError, match="没有阈值达到复现率"):
        curve.recommended(min_reproducibility=0.85)
    # 钉住写法：正向比较放行 nan，反向比较拦住它。
    assert not math.nan < 0.85
    assert not (math.nan >= 0.85)


# --------------------------------------------------------------------------
# 裁决 68：配置的报数阈值必须至少和方法推荐值一样保守
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset_a_curve(dataset_a_dir):
    """数据集 A 跟踪段的 6 帧（f05–f10）全九档阈值曲线。

    取帧理由（裁决 355 要求明确记录）：

    1. ``tracking_segment`` 在 A 上给出 f04–f35（32 帧），f00–f03 是机架静止待命段，
       被 ``min_length=10`` 排除，所以候选只能从 f04 起。
    2. **跳过段首帧 f04。** ``measurements.md`` 已记录数据集 B 的段首帧 f16
       「机架尚未停稳」、逐对 rms 抬升到 1.8567。实测 A 的段首帧同病且更重：
       f04→f05 这一对 rms=2.8217、旋转 −0.59493°，而其后各对稳定在
       rms≈0.73、旋转 −0.3805°±0.0015。因为 ``register_sequence`` 是**链式**累积
       （``matrices[nxt] = matrices[cur] @ sol.matrix``），首对的偏差进入其后每一帧
       的变换：以 f04 为参考帧时，仍在场内的参考源最近邻中位距离是
       4.83/4.51/4.63/4.53/4.68 px——**整条链系统性地超出 3.0 px 匹配半径**，
       5σ 复现率被压到 0.2602；改用 f05 起，同一串中位距离降到
       0.59/0.86/1.20/1.50/1.82 px，复现率 0.8000。这不是阈值问题而是参考系问题。
    3. 帧数 6 取自 ``repeatability.frames``；B 用同样的规则（段首 +1，即 f17–f22），
       两条曲线因此可以对照。
    4. f05 探测数非零（5σ 下 95 个），满足「参考帧不得为空」。
    """
    from src.calib.hotpixel import build_from_sequence
    from src.calib.segment import tracking_segment
    from src.config import load_config
    from src.dataio.fits_loader import FrameSequence
    from src.detect.segmentation import detect_sequence
    from src.register.solver import register_sequence

    conf = load_config()
    seq = FrameSequence.from_directory(dataset_a_dir)
    # 分段与热像素按各自模块的出厂默认值调用（与 default.yaml 的 segment /
    # hotpixel 节一致），项目内其余数据集测试也是这个用法。
    seg = tracking_segment(seq.headers)
    start = seg.start + 1  # 跳过未停稳的段首帧，见 docstring 第 2 条
    frames = list(range(start, start + conf["repeatability"]["frames"]))
    hpm = build_from_sequence(seq, frames)
    dets = detect_sequence(
        seq, frames, n_sigma=conf["detect"]["search_n_sigma"], hot_clusters=hpm.clusters
    )
    reg = register_sequence(dets, frames, reference=frames[0], config=dict(conf["register"]))
    return frames, scan_thresholds(
        seq,
        frames,
        reg,
        sigmas=conf["repeatability"]["sigmas"],
        npixels=conf["detect"]["npixels"],
        hot_clusters=hpm.clusters,
        match_radius_px=conf["repeatability"]["match_radius_px"],
    )


def test_default_threshold_is_unreachable_on_the_measured_curve(dataset_a_curve):
    """如实断言**今天观测到的事实**：数据集 A 九档无一达到默认门限 0.85。

    这条测的是「``recommended()`` 在实测曲线上求不出值」这个**事实**，
    **不是**「裁决 68 的关系不成立」。A 九档最高复现率是 0.8289（6σ），默认门限
    0.85 无一档达到，于是 ``recommended()`` 按设计抛错。门限 0.85 与实测曲线的差距
    是控制器要裁的事项，见 ``task-14-report.md``。

    **为什么不标 xfail（裁决 368）。** 原来这条与裁决 68 的关系断言写在同一个函数体
    里，整条标 ``xfail(strict=True)``——而 strict 只能隔离「整条测试」，不能隔离
    「测试里的某一步」。那个函数体有两个可失败点：``recommended()`` 抛错、
    以及 ``assert pick.n_sigma <= report``。今天失败在第一步，可一旦复现率越过 0.85，
    ``recommended()`` 会返回 6.0σ 而 ``assert 6.0 <= 5.0`` 失败——**仍被记为预期失败**，
    而那正是裁决 68 存在的全部理由。这个假想并不遥远：最高档只差 0.0211，
    且 A 的结构上限 0.8842 > 0.85，所以 0.85 在 A 上是可达的。

    改成如实断言事实后，它今天就是**绿的**，而一旦复现率真的越过 0.85 它会转红、
    强制重裁——这正是原 xfail 想要的效果，且没有那个盲区。
    门限 0.85 是 ``recommended()`` 的默认值，此处无参调用，**一字未动**。
    """
    _, curve = dataset_a_curve
    with pytest.raises(ValueError, match="没有阈值达到复现率"):
        curve.recommended()


def test_recommended_relationship_holds_at_the_measured_threshold(dataset_a_curve, cfg):
    """把裁决 68 的关系在**曲线真能求值的门限**上验一遍，不动 0.85 这个数。

    **这是裁决 68 关系的唯一守卫**（裁决 368）：原来那条同名断言与「默认门限
    是否可达」写在一个函数体里、整条标 strict xfail，现已拆成
    ``test_default_threshold_is_unreachable_on_the_measured_curve``——它只断言
    「默认门限在实测曲线上不可达」这个事实，不再带这条关系断言。

    被隔离的从来是「0.85 这个门限」，不是「推荐值不得超过报数阈值」这个关系。
    关系本身与门限取值无关，所以这里对曲线实际达到的每一个门限逐一验证：
    只要 ``recommended(t)`` 有解，它就必须 <= ``detect.report_n_sigma``。
    """
    _, curve = dataset_a_curve
    report = float(cfg["detect"]["report_n_sigma"])
    evaluated = 0
    for threshold in (0.5, 0.6, 0.7, 0.75, 0.8):
        try:
            pick = curve.recommended(min_reproducibility=threshold)
        except ValueError:
            continue
        evaluated += 1
        assert pick.n_sigma <= report, (
            f"门限 {threshold} 下推荐 {pick.n_sigma}σ，超过配置报数阈值 {report}σ；"
            "对外报数阈值必须至少和推荐值一样保守"
        )
    assert evaluated > 0, "没有任何门限能在实测曲线上求出推荐值，本条未验到任何东西"


def test_dataset_a_curve_is_monotonic_and_bounded(dataset_a_curve):
    """数据集 A 曲线的结构性质：sigma 升序、复现率有界、探测数随 sigma 下降。"""
    frames, curve = dataset_a_curve
    assert len(frames) == 6
    sigmas = [p.n_sigma for p in curve.points]
    assert sigmas == sorted(sigmas)
    assert all(0.0 <= p.reproducibility <= 1.0 for p in curve.points)
    counts = [p.n_detected_mean for p in curve.points]
    assert counts == sorted(counts, reverse=True)
    # 参考帧不得为空（裁决 355）：否则整条曲线的复现率按约定全是 0。
    assert all(p.n_reference > 0 for p in curve.points)
    json.loads(json.dumps(curve.to_dict()))


def test_real_curve_fields_are_self_consistent(dataset_a_curve):
    """曲线上六个字段之间的**代数关系**必须成立：两个比值等于各自的分子除分母。

    钉的是关系而不是数值，所以对真实数据的波动免疫，也不占用任何基线数字。
    被保护的性质：``reproducibility`` 与 ``purity`` 的分母必须**就是**同一个 point
    里报出的 ``n_reference`` / ``n_detected_mean``，不得各自独立漂移。

    为什么需要它（裁决 375 / 376）：``scan_thresholds`` 里 ``n_reference=int(counts[0])``
    与 ``cross_frame_repeatability`` 里 ``reproducibility = n_reproducible / n_reference``
    是**两条独立赋值**，原先没有任何断言把两者绑在一起。把前者换成
    ``int(round(float(np.mean(counts))))``，**全套 332 项无一变红**——而
    ``n_reference`` 这一列正是写进 ``measurements.md`` 的那一列，实测在数据集 A 上
    5σ 会把复现率从 0.8000 挪到 0.7600、6σ 从 0.8289 挪到 0.7778。这就是裁决 66 要
    分开的两个量从 ``scan_thresholds`` 这条侧门重新混一。第二条断言同样覆盖
    ``n_detected_mean``：``:327`` 的「空帧不进分母」是
    ``cross_frame_repeatability`` 里同一逻辑的**第二处独立实现**，把它改成
    ``list(counts)`` 也是全模块 0 杀（裁决 376）。

    **必须挂在由 ``scan_thresholds`` 真实构造的曲线上**，不得挂在 ``_curve()`` 的
    示意曲线上：后者的 ``n_reproducible`` 是任意填充值、``reproducibility`` 是手填的
    示意值，实测四档差 6.9e-05 到 4.3e-04（修法 1 把填充值换成 0 后差到 0.888），
    挂上去会红——而按裁决 243 那条红不许靠放宽容差消掉。
    """
    _, curve = dataset_a_curve
    assert curve.points, "曲线为空则本条未验到任何东西"
    checked = 0
    for p in curve.points:
        if p.n_reference:
            assert p.n_reproducible / p.n_reference == pytest.approx(
                p.reproducibility, abs=1e-12
            ), f"{p.n_sigma}σ: 复现率与 n_reproducible/n_reference 不自洽"
            checked += 1
        if p.n_detected_mean:
            assert p.n_reproducible / p.n_detected_mean == pytest.approx(
                p.purity, abs=1e-12
            ), f"{p.n_sigma}σ: 纯度与 n_reproducible/n_detected_mean 不自洽"
    assert checked == len(curve.points), (
        "有档位的 n_reference 为 0，复现率那一侧未被检查"
    )


def test_scroll_off_caps_reproducibility_below_one(dataset_a_curve, dataset_a_dir):
    """场滚动给复现率设了一个**结构性上限**：滚出探测器的源不可能复现。

    这是 0.85 达不到的主因，也是本任务最重要的归因，所以要有守卫而不只是报告
    里的一句话。数据集 A 的 6 帧累计场移实测 845.13 px（5 步），即约 169 px/帧；
    一个源要满足「≥4 of 6 帧」至少要活过 3 步，即 3×169 = 507 px，
    占 4096 图幅宽的 12.4%——单是这一项就把上限压到约 0.88（实测 0.8842，
    二维还要再减一点）。

    **5σ 一档**的实测复现率与该档上限之比是 0.8000 / 0.8842 = **0.9048**，
    即在结构上可复现的源里方法找回了 90.5%。**这个比值只对 5σ 这一档成立，
    不得推广到别档**（裁决 370）：结构上限是**每一档各自的量**，它取决于该档参考帧
    的源数与各源的滚出情况，而各档参考帧源数差别很大（2σ 5791、4σ 126、
    5σ 95、8σ 62）。拿一档的上限去除另一档的复现率，得到的比值不代表任何东西。
    ``measurements.md`` 的上限表只给 5σ 一行，正是这个理由。要给多档结论，
    必须对每一档各跑一次逆变换统计、算出各自的上限。

    下面 ``measured / ceiling > 0.85`` 这条断言的方向是保守的（实测 0.9048），
    留作护栏；但它不构成「多档都在 0.90 以上」这类更强的结论。
    """
    import numpy as np
    from scipy.spatial import cKDTree

    from src.calib.hotpixel import build_from_sequence
    from src.calib.segment import tracking_segment
    from src.config import load_config
    from src.dataio.fits_loader import FrameSequence
    from src.detect.segmentation import detect_sequence
    from src.register.solver import register_sequence

    frames, curve = dataset_a_curve
    conf = load_config()
    seq = FrameSequence.from_directory(dataset_a_dir)
    seg = tracking_segment(seq.headers)
    assert frames == list(range(seg.start + 1, seg.start + 1 + 6))

    hpm = build_from_sequence(seq, frames)
    # 配准用 search_n_sigma 的探测（与 dataset_a_curve 同一条流水线），复现率统计
    # 用 report_n_sigma 的探测。两者必须一致，否则得到的是另一个配准解：实测在
    # 5σ 探测上重建配准会把本条的复现率挪到 0.7895，与曲线报的 0.8000 不符。
    reg_dets = detect_sequence(
        seq, frames, n_sigma=conf["detect"]["search_n_sigma"], hot_clusters=hpm.clusters
    )
    reg = register_sequence(
        reg_dets, frames, reference=frames[0], config=dict(conf["register"])
    )
    dets = detect_sequence(
        seq, frames, n_sigma=conf["detect"]["report_n_sigma"], hot_clusters=hpm.clusters
    )
    ny, nx = seq.image(frames[0]).shape
    min_hits = min_hits_for(0.6, len(frames))

    ref_det = dets[frames[0]].xy
    ref_sky = reg.to_sky(frames[0], ref_det)
    # 每个参考源在多少帧里仍落在探测器画幅内（参考帧自身记 1）。
    on_detector = np.ones(len(ref_sky), dtype=int)
    for f in frames[1:]:
        xy = reg.to_detector(f, ref_sky)
        on_detector += (
            (xy[:, 0] >= 0) & (xy[:, 0] < nx) & (xy[:, 1] >= 0) & (xy[:, 1] < ny)
        ).astype(int)
    ceiling = float((on_detector >= min_hits).mean())

    # 实际命中数，用与 cross_frame_repeatability 相同的判据。
    hits = np.ones(len(ref_sky), dtype=int)
    for f in frames[1:]:
        sky = reg.to_sky(f, dets[f].xy)
        dist, _ = cKDTree(sky).query(ref_sky)
        hits += (dist < conf["repeatability"]["match_radius_px"]).astype(int)
    measured = float((hits >= min_hits).mean())

    assert ceiling < 1.0, "若无源滚出画幅，本条测试测不到它要测的东西"
    assert measured <= ceiling + 1e-12, (
        f"复现率 {measured:.4f} 超过了场滚动给出的结构性上限 {ceiling:.4f}，"
        "说明有源在滚出画幅后仍被判为命中——匹配或配准有误"
    )
    # 上限本身把 0.85 挡在外面，所以 5σ 达不到 0.85 是结构性的而非阈值选得不好。
    assert ceiling == pytest.approx(0.8842, abs=0.02)
    # 5σ 这一档：结构上可复现的那批源里方法找回了九成（实测 0.9048）。
    # 断言方向保守，且**只针对本档**——见 docstring 关于「上限是每档各自的量」。
    assert measured / ceiling > 0.85
    # 与曲线自身报的 5σ 复现率一致（同一判据的两条独立算路）。
    at_report = {p.n_sigma: p for p in curve.points}[conf["detect"]["report_n_sigma"]]
    assert at_report.reproducibility == pytest.approx(measured, abs=1e-12)


def test_dataset_b_threshold_scan_shows_noise_cliff(dataset_b_cliff_curve):
    """2σ 处探测数暴涨而纯度崩塌——这是「263822 颗星」的实测反证。

    原任务书这条测试的第四条断言是 ``by_sigma[5.0].reproducibility > 0.85``，
    实测 0.6304，属控制器所有的基线数字，故单独隔离到下一条测试里，
    不在此处放宽。其余三条断言实测全部成立。
    """
    _, curve = dataset_b_cliff_curve
    by_sigma = {p.n_sigma: p for p in curve.points}
    assert by_sigma[2.0].n_detected_mean > 5 * by_sigma[5.0].n_detected_mean
    assert by_sigma[2.0].purity < 0.3
    assert by_sigma[5.0].reproducibility > by_sigma[2.0].reproducibility
    assert by_sigma[5.0].purity > by_sigma[2.0].purity


@pytest.mark.xfail(
    strict=True,
    reason="实测数据集 B 5σ 复现率 0.6304 < 0.85；差额由场滚出（结构性上限 0.8261）"
    "与探测阈值抖动共同造成，属控制器所有的基线数字（见 task-14-report.md）",
)
def test_dataset_b_five_sigma_reproducibility_reaches_085(dataset_b_cliff_curve):
    """原任务书要求 5σ 复现率 > 0.85；实测 0.6304。

    隔离而不放宽：0.85 这个数**一字未动**留在断言里。一旦复现率真的越过它，
    strict xfail 会转红，强制重新裁决。归因见 ``task-14-report.md``——
    6 帧里场移累计约 600 px，参考帧的源有约 17% 在满足 ≥4 帧之前就滚出探测器，
    所以 0.85 在这个取帧方案下**结构上不可达**（实测上限 0.8261）。
    """
    _, curve = dataset_b_cliff_curve
    by_sigma = {p.n_sigma: p for p in curve.points}
    assert by_sigma[5.0].reproducibility > 0.85


@pytest.fixture(scope="module")
def dataset_b_cliff_curve(dataset_b_dir):
    """数据集 B 的 2/3/5σ 三档曲线；module 作用域让上面两条测试共跑一次流水线。

    取帧规则与数据集 A 一致：跳过未停稳的段首帧 f16（``measurements.md`` 已记录
    其逐对 rms 抬升到 1.8567），从 f17 起取 ``repeatability.frames`` = 6 帧。
    """
    from src.calib.hotpixel import build_from_sequence
    from src.calib.segment import tracking_segment
    from src.config import load_config
    from src.dataio.fits_loader import FrameSequence
    from src.detect.segmentation import detect_sequence
    from src.register.solver import register_sequence

    conf = load_config()
    seq = FrameSequence.from_directory(dataset_b_dir)
    seg = tracking_segment(seq.headers)
    start = seg.start + 1
    frames = list(range(start, start + conf["repeatability"]["frames"]))
    hpm = build_from_sequence(seq, frames)
    dets = detect_sequence(
        seq, frames, n_sigma=conf["detect"]["search_n_sigma"], hot_clusters=hpm.clusters
    )
    reg = register_sequence(dets, frames, reference=frames[0], config=dict(conf["register"]))

    return frames, scan_thresholds(
        seq,
        frames,
        reg,
        sigmas=[2.0, 3.0, 5.0],
        npixels=conf["detect"]["npixels"],
        hot_clusters=hpm.clusters,
        match_radius_px=conf["repeatability"]["match_radius_px"],
    )
