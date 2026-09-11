from __future__ import annotations

import json
import logging
import warnings

import numpy as np
import pytest

from src.astrometry.photometry import (
    MAX_CLIP_ROUNDS,
    MIN_ZP_POINTS,
    PhotometryReport,
    ZeroPoint,
    calibrate_table,
    instrumental_mag,
    limiting_magnitude,
    zero_point_from_truth,
)
from src.config import load_config
from src.detect.segmentation import SourceTable


def make_table(flux):
    flux = np.asarray(flux, dtype=np.float64)
    n = len(flux)
    return SourceTable(
        frame=0,
        x=np.arange(n, dtype=np.float64),
        y=np.arange(n, dtype=np.float64),
        flux=flux,
        peak=flux / 4.0,
        elongation=np.full(n, 1.1),
        npix=np.full(n, 9, dtype=np.int64),
    )


def test_instrumental_mag_follows_pogson():
    """100 倍流量比对应 2.5 星等差（10 倍一级）。"""
    m = instrumental_mag([100.0, 1000.0])
    assert m[0] - m[1] == pytest.approx(2.5, abs=1e-9)


def test_instrumental_mag_marks_nonpositive_flux_as_nan():
    """流量 <= 0 无定义星等，给 nan 而不是抛错——一帧里有几个坏源是常态。"""
    m = instrumental_mag([-5.0, 0.0, 10.0])
    assert np.isnan(m[0]) and np.isnan(m[1])
    assert np.isfinite(m[2])


def test_instrumental_mag_marks_nonpositive_flux_without_warning():
    """流量 <= 0 的 log10 被 errstate 包住，不得漏出 RuntimeWarning。

    这一条与 test_negative_exposure_raises_before_any_log10 是一对：流量的 nan
    是**合法输出**（所以警告要压掉），曝光的非正值是**非法输入**（所以要抛错而
    不是压掉警告）。两者不能互换。
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = instrumental_mag([-1.0, 0.0, 1000.0])
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []
    assert np.isnan(m[0]) and np.isnan(m[1])


def test_instrumental_mag_normalises_by_exposure():
    """归一化项的符号：``m + 2.5*log10(t)`` 等于 ``-2.5*log10(flux/t)``。

    实测 ``-2.5*log10(1000/0.08) = -10.242275``，与 ``instrumental_mag(1000,
    0.08)`` 逐位相同。符号写反会把每一个星等平移 5.4846 mag。
    """
    a = instrumental_mag([1000.0], exposure_s=None)[0]
    b = instrumental_mag([1000.0], exposure_s=0.08)[0]
    assert b == pytest.approx(a + 2.5 * np.log10(0.08), abs=1e-9)
    assert b == pytest.approx(-2.5 * np.log10(1000.0 / 0.08), abs=1e-9)


def test_instrumental_mag_treats_exposure_one_as_identity():
    """``exposure_s=1.0`` 的归一化项恰为 0，与不归一化数值相同但语义不同。"""
    assert instrumental_mag([1000.0], exposure_s=1.0)[0] == pytest.approx(
        instrumental_mag([1000.0])[0], abs=1e-12
    )


@pytest.mark.parametrize("bad", [0.0, 0, -0.08, -1.0])
def test_nonpositive_exposure_raises_chinese_value_error(bad):
    """曝光 <= 0 必须抛中文 ValueError，不得静默。

    修复前 ``if exposure_s:`` 是真值测试：实测 ``0.0`` 与 ``0`` 都返回
    ``-7.500000``，即**静默地不做归一化**，与「未请求归一化」完全同形；而
    ``-0.08`` 返回 ``nan``，与「流量 <= 0」的 nan **无法区分**，于是一次
    ms/s 单位混用看起来就像一次未探测。``FrameHeader.exposure_s`` 由
    ``exposure_ms`` 换算而来，这个混用在本仓是活风险而非假想。
    """
    with pytest.raises(ValueError, match="曝光"):
        instrumental_mag([1000.0], exposure_s=bad)


def test_negative_exposure_raises_before_any_log10():
    """校验必须放在所有 log10 之上，使行为不依赖 warnings 过滤器配置。

    曝光那一行的 ``log10`` 原先在 ``np.errstate`` 块**之外**：实测
    ``exposure_s=-0.08`` 漏出 1 条 RuntimeWarning（而 ``flux=[-1, 0]`` 漏出 0
    条）。当前 ``pytest.ini`` 的 ``filterwarnings`` 只有两条 ``ignore::``、没有
    ``error``，所以今天返回 nan；一旦有人加上全局 ``error``，同一行就会改为抛
    ``RuntimeWarning``。把校验提到最前面，两种配置下都是同一个 ValueError。
    因此本测试断言的是「抛 ValueError 且不留任何警告」，既不断言 nan，也不断言
    RuntimeWarning——那两者都是修复前的事故行为。
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="曝光"):
            instrumental_mag([1000.0], exposure_s=-0.08)
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


def test_zero_point_recovers_known_offset():
    """只验证偏移量的减法与 source/n_points 字段，**不验证星等律**。

    本测试用 ``instrumental_mag(flux) + 15.335`` 造真值，而函数算的是
    ``mean(truth_mag - instrumental_mag(flux))``，所以 15.335 是**构造出来的**：
    实测把星等律换成荒谬的 ``flux ** 0.5`` 并同样代入两边，本测试依然给出
    ``value=15.335000000``。任何函数只要在减法两侧同时使用就能通过。约束星等律
    的是 test_zero_point_is_anchored_to_pogson_ratios，不是这一条。
    """
    flux = np.array([100.0, 300.0, 1000.0, 3000.0])
    m_inst = instrumental_mag(flux)
    zp = zero_point_from_truth(flux, m_inst + 15.335)
    assert zp.value == pytest.approx(15.335, abs=1e-6)
    assert zp.std == pytest.approx(0.0, abs=1e-6)
    assert zp.source == "truth"
    assert zp.n_points == 4


def test_zero_point_is_anchored_to_pogson_ratios():
    """真值星等独立于 instrumental_mag 构造：100 倍流量比必须对应 5.00 星等差。

    四点（而非三点）是硬要求：``MIN_ZP_POINTS`` 现为 3，三点时
    ``len(d) > MIN_ZP_POINTS`` 为假，剪裁循环根本不进，且裕度为零——实测把
    ``MIN_ZP_POINTS`` 提到 4 会让三点 fixture 抛
    ``ValueError(零点定标至少需要 4 个有效配对，实际 3)``，红得与被测性质无关。
    四点下剪裁循环进入并在 ``sd <= 0.0`` 处退出。

    实测杀伤表（断言 value=17.5、std=0）：
    ``f**0.5`` value=-870.822620 std=1.33e+03；``-2.0*log10(f)`` 15.500000 /
    1.1180；``-3.0*log10(f)`` 19.500000 / 1.1180；``-2.5*ln(f)`` 30.525851 /
    7.2817；``-2.5*log2(f)`` 40.719281 / 12.980；``+2.5*log10(f)`` -2.500000 /
    11.180；``-2.5*log10(f)+7`` 10.500000 / **0.0**。
    """
    flux = np.array([10.0, 1000.0, 100000.0, 10000000.0])  # 每级 100 倍
    truth = np.array([15.0, 10.0, 5.0, 0.0])  # 每级 5.00 mag，独立给出
    zp = zero_point_from_truth(flux, truth)
    # std 只钉住对数律的系数 2.5；凡 a*log10(f)+b 中 a=-2.5 者，无论 b 取何值
    # std 都是 0，所以它对加性常数完全不敏感——而零点正是那个加性常数。
    assert zp.std == pytest.approx(0.0, abs=1e-9)
    # value 钉住零点常数本身，是杀掉「系数对、常数错」（-2.5*log10(f)+7）的
    # 唯一断言。上面那条 std 对它给出 0.0，放不倒它。两条都不可删。
    assert zp.value == pytest.approx(truth[0] + 2.5 * np.log10(10.0), abs=1e-9)
    # 捕捉「本不该剪裁却剪裁了」。
    assert zp.n_points == 4


def test_zero_point_clips_outliers():
    """一个 +5 mag 的错误配对必须被 3σ 剪掉。

    实测 ``value=15.000000, n=19``：diff 数组是 19 个 15.0 加一个 20.0，
    ``std=1.089725``，3σ 门限 3.2692 小于偏差 5.0，所以离群点被拒。在
    ``sigma_clip`` 从 1.0 到 3.0 的每个取值上都剪掉。基于 std 的剪裁看着脆弱，
    但在带真实散度的算例（40 点、σ=0.218、一个 +5 mag 错配）上与 MAD 剪裁**逐位
    一致**，不要换成 MAD。
    """
    flux = np.full(20, 1000.0)
    truth = instrumental_mag(flux) + 15.0
    truth[0] += 5.0  # 一个错误配对
    zp = zero_point_from_truth(flux, truth, sigma_clip=3.0)
    assert zp.value == pytest.approx(15.0, abs=0.05)
    assert zp.n_points == 19


def test_zero_point_requires_enough_points():
    """点数不足时抛错而不是返回一个无意义的零点。"""
    with pytest.raises(ValueError, match="零点定标至少需要"):
        zero_point_from_truth([1000.0], [8.0])


@pytest.fixture
def photometry_cfg_override(monkeypatch):
    """临时替换 ``photometry`` 配置段并清掉模块缓存，退出时恢复。

    ``_CFG_CACHE`` 必须两头都清：进入时清是为了让替换生效（前面的测试可能已经填过
    缓存），退出时清是为了不把改过的段留给后面的测试——一个模块级缓存被污染后，
    失败会出现在与本测试无关的地方。
    """
    import src.astrometry.photometry as mod

    def _apply(**keys):
        monkeypatch.setattr(mod, "_CFG_CACHE", None, raising=False)
        base = dict(load_config()["photometry"])
        base.update(keys)
        monkeypatch.setattr(mod, "_CFG_CACHE", base, raising=False)
        return base

    yield _apply
    mod._CFG_CACHE = None


def test_min_zp_points_comes_from_default_yaml():
    """配置键必须存在，且与模块的回退默认值一致。

    这条只钉「两处数字相等」，牙齿到此为止——它**不能**证明模块真的读了配置
    （把配置值和常量一起改成 9，它照样绿）。真读配置由
    ``test_min_zp_points_override_changes_behaviour`` 钉住，两条不可互相替代。
    """
    conf = load_config()
    assert conf["photometry"]["min_zp_points"] == MIN_ZP_POINTS


def test_max_clip_rounds_comes_from_default_yaml():
    """剪裁轮数上限同样必须落在配置里，模块常量只是回退值。

    原先这个 5 是**裸写在循环里**的字面量（``for _ in range(5)``），既不在配置里
    也没有名字。同上条：这只钉数字相等，真读配置见下面那条覆盖测试。
    """
    conf = load_config()
    assert conf["photometry"]["max_clip_rounds"] == MAX_CLIP_ROUNDS


def test_min_zp_points_override_changes_behaviour(photometry_cfg_override):
    """把配置里的 ``min_zp_points`` 调到 5，4 个有效配对必须被拒。

    被保护的性质：**阈值改在配置里就要真的生效**。原先模块只有常量、从不读配置，
    靠上面那条「相等」断言维持同步——而那条测试对「模块不读配置」这个事实**零区分
    力**：调用方改了 ``default.yaml`` 以为调紧了门限，实测行为一点没变。

    断言分两半：调到 5 时 4 个配对被拒（消息里的数字必须是 **5** 而不是模块常量 3，
    否则消息在说谎），调回 3 时同一份输入通过。少了后一半，一个「永远抛错」的实现
    也能让前一半变绿。模块常量本身不动（仍是 3），这条同时证明拒绝的依据是配置。
    """
    flux = np.full(4, 1000.0)
    truth = instrumental_mag(flux) + 15.0

    photometry_cfg_override(min_zp_points=5)
    with pytest.raises(ValueError, match="至少需要 5 个有效配对，实际 4"):
        zero_point_from_truth(flux, truth)
    assert MIN_ZP_POINTS == 3  # 模块常量未被改动，拒绝的依据只能是配置

    photometry_cfg_override(min_zp_points=3)
    assert zero_point_from_truth(flux, truth).n_points == 4


def test_max_clip_rounds_override_changes_behaviour(photometry_cfg_override):
    """把配置里的 ``max_clip_rounds`` 放开到 40，几何尾算例的结果必须跟着变。

    被保护的性质同上条。用 30 点几何尾 ``3**k``，因为它是实测能咬住 5 轮上限的
    输入：5 轮时剩 **25** 点、均值 **1.694577e+10**；放开到 40 轮时走完 **24** 轮
    收敛、剩 **7** 点、均值 **156.142857**。两个结果相差 1.69e10 mag，所以「配置
    有没有生效」在这个算例上一望可辨，不需要任何容差判断。

    警告那一半同时反向验证：上限咬住时有 WARNING，放开后收敛则**没有**。
    """
    geometric = np.array([3.0 ** k for k in range(30)])
    flux = np.full(len(geometric), 1000.0)
    truth = instrumental_mag(flux) + geometric

    photometry_cfg_override(max_clip_rounds=5)
    capped = zero_point_from_truth(flux, truth, sigma_clip=3.0)
    assert capped.n_points == 25
    assert capped.value == pytest.approx(1.694577e10, rel=1e-6)

    photometry_cfg_override(max_clip_rounds=40)
    freed = zero_point_from_truth(flux, truth, sigma_clip=3.0)
    assert freed.n_points == 7
    assert freed.value == pytest.approx(156.142857, abs=1e-6)
    assert MAX_CLIP_ROUNDS == 5  # 模块常量未动，轮数只能来自配置


def test_photometry_falls_back_to_module_constants_when_keys_are_missing():
    """配置段缺键时回退到模块常量，而不是抛 ``KeyError``。

    一个残缺的配置不该让整条测光链无法运行。回退值与出厂值相同，所以行为与配置
    完整时**逐位一致**——这正是断言的内容：空配置段下 4 个配对通过（回退门限 3），
    2 个配对被拒且消息报的是 3。

    这里直接写 ``_CFG_CACHE`` 而不用上面那个 fixture：本条要的恰是「段里什么都没
    有」，而 fixture 是在完整段上叠加键。``finally`` 里清回 ``None``，否则空段会
    留给后面的测试。
    """
    import src.astrometry.photometry as mod

    mod._CFG_CACHE = {}
    try:
        flux = np.full(4, 1000.0)
        truth = instrumental_mag(flux) + 15.0
        assert zero_point_from_truth(flux, truth).n_points == 4
        with pytest.raises(ValueError, match="至少需要 3 个有效配对，实际 2"):
            zero_point_from_truth(np.full(2, 1000.0), np.full(2, 8.0))
    finally:
        mod._CFG_CACHE = None


def test_clip_round_cap_logs_a_warning_when_it_bites(caplog):
    """轮数上限咬住时必须留一条 WARNING，不得静默发出未剪完的零点。

    被保护的性质：**剪裁没做完而结果照发，量级不可预估，必须留痕**。上限原先
    耗尽后直接落到 ``return``，没有任何痕迹。

    构造 30 点几何尾 ``3**k``：实测 5 轮时剩 **25** 点、均值 **1.694577e+10**，
    放开上限要 **24** 轮、剩 **7** 点、均值 **156.142857**——**差 1.69e10 mag 而
    原先无警告**。这不是本项目会遇到的输入（真实算例 1 轮即收敛），所以上限不必
    提高；要紧的是它不再静默。

    第二组断言（干净 12 点、1 轮收敛、**零条** WARNING）是本条的另一半牙齿：
    少了它，一个「每次都记 WARNING」的实现也能让上一半变绿，而那会把日志淹掉。
    """
    geometric = np.array([3.0 ** k for k in range(30)])
    flux = np.full(len(geometric), 1000.0)
    truth = instrumental_mag(flux) + geometric

    with caplog.at_level(logging.WARNING, logger="src.astrometry.photometry"):
        zp = zero_point_from_truth(flux, truth, sigma_clip=3.0)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "without converging" in message
    # 消息必须报出剪到剩几点，否则读者无从判断这个零点偏了多少。
    assert "25 of 30" in message
    # 钉住「上限确实咬住了」这个前提：咬住时存活点数就是 25。
    assert zp.n_points == 25

    caplog.clear()
    clean = np.full(12, 1000.0)
    with caplog.at_level(logging.WARNING, logger="src.astrometry.photometry"):
        zp_clean = zero_point_from_truth(clean, instrumental_mag(clean) + 15.0)
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
    assert zp_clean.n_points == 12
    assert zp_clean.value == pytest.approx(15.0, abs=1e-9)


def test_zero_point_ignores_nan_truth():
    """真值里的 nan 行不参与拟合，也不污染均值。"""
    flux = np.array([100.0, 300.0, 1000.0, 3000.0])
    truth = instrumental_mag(flux) + 15.0
    truth[1] = np.nan
    zp = zero_point_from_truth(flux, truth)
    assert zp.n_points == 3
    assert zp.value == pytest.approx(15.0, abs=1e-6)


def test_zero_point_rejects_length_mismatch():
    """流量与真值长度不一致时抛中文 ValueError。"""
    with pytest.raises(ValueError, match="数量不一致"):
        zero_point_from_truth([100.0, 1000.0, 10000.0], [8.0, 9.0])


def test_apply_and_calibrate_agree():
    """``calibrate_table`` 只是 ``apply(instrumental_mag(flux))`` 的组合。"""
    zp = ZeroPoint(value=15.335, std=0.218, n_points=54, source="truth")
    table = make_table([100.0, 1000.0])
    direct = zp.apply(instrumental_mag(table.flux))
    assert calibrate_table(table, zp) == pytest.approx(direct)


def test_calibrate_table_uses_zero_point_exposure():
    """定标必须用 ``zp.exposure_s``，调用方不能另给一个。

    原设计里 ``zero_point_from_truth`` 与 ``calibrate_table`` 各拿自己的
    ``exposure_s``，而 ``ZeroPoint`` 两个都不记：实测用 ``exposure_s=0.08`` 拟合
    却不带曝光定标，每一个星等平移 **2.7423 mag** = ``2.5*log10(0.08)``，且十一
    条原测试没有一条能发现。现在曝光跟着零点走。

    抵消的前提是**真值星等在两次拟合里是同一组数**：真值是固定的物理量，不随你
    选不选归一化而变。按 0.08 s 归一化把流量率放大 12.5 倍，仪器星等因此亮
    2.7423 mag（更负），拟合出的零点相应**高** 2.7423 mag，与定标时加回的那一项
    精确对消。若按各自约定重新构造真值，两条路本就不该相等，那测的是别的东西。
    """
    flux = np.array([100.0, 300.0, 1000.0, 3000.0])
    truth = np.array([12.6, 11.4, 10.1, 8.9])  # 固定的真值星等，与曝光约定无关
    zp = zero_point_from_truth(flux, truth, exposure_s=0.08)
    zp_no_exp = zero_point_from_truth(flux, truth)
    assert zp.exposure_s == pytest.approx(0.08)
    assert zp_no_exp.exposure_s is None
    # 带曝光拟合的零点恰高 |2.5*log10(0.08)| = 2.7423 mag。
    assert zp.value == pytest.approx(zp_no_exp.value - 2.5 * np.log10(0.08), abs=1e-12)
    assert zp.value - zp_no_exp.value == pytest.approx(2.7423, abs=1e-4)

    table = make_table(flux)
    mags = calibrate_table(table, zp)
    # 曝光项精确对消：定标星等与「全程不带曝光」逐位相同。
    assert mags == pytest.approx(calibrate_table(table, zp_no_exp), abs=1e-12)
    # 而漏掉曝光的定标会整体偏 2.7423 mag——这就是被堵住的那条静默路径。
    naive = zp.apply(instrumental_mag(flux))
    assert float(np.mean(naive - mags)) == pytest.approx(-2.5 * np.log10(0.08), abs=1e-9)
    assert float(np.mean(naive - mags)) == pytest.approx(2.7423, abs=1e-4)


def test_calibrate_table_raises_when_exposure_override_disagrees():
    """显式覆盖值与 ``zp.exposure_s`` 不一致时抛中文 ValueError。

    把 2.7423 mag 的静默系统误差换成一次响亮的失败。
    """
    zp = ZeroPoint(value=15.335, std=0.218, n_points=54, source="truth", exposure_s=0.08)
    table = make_table([100.0, 1000.0])
    with pytest.raises(ValueError, match="曝光时间"):
        calibrate_table(table, zp, exposure_s=0.03)


def test_calibrate_table_accepts_consistent_exposure_override():
    """覆盖值与零点一致时照常工作（允许调用方显式复述）。"""
    zp = ZeroPoint(value=15.335, std=0.218, n_points=54, source="truth", exposure_s=0.08)
    table = make_table([100.0, 1000.0])
    assert calibrate_table(table, zp, exposure_s=0.08) == pytest.approx(
        calibrate_table(table, zp)
    )


def test_limiting_magnitude_is_pinned_to_the_fixture_percentile():
    """极限星等钉死为字面量 10.7126，不在测试里用 nanpercentile 重算。

    fixture 是 ``geomspace(50, 50000, 200)`` 配 ``zp=15.335``，**其中没有任何
    随机数**，所以配得上一个字面量而不是一条区间。实测十位小数：
    p75=9.2125749892、p90=10.3375749892、p95=10.7125749892、p99=11.0125749892、
    中位数 7.3375749892。

    原测试的两条断言都是空的：一条是 ``nanpercentile(mags, 95.0)`` 逐字重述实现，
    另一条 ``lim > median`` 对**任何**大于 50 的百分位都成立——实测 51 → 7.4126、
    60 → 8.0876、75 → 9.2126、95 → 10.7126、99 → 11.0126 全部通过。也就是说默认
    百分位从 95 悄悄改成 51 什么也不会红。字面量会红，裕度 3.3 mag。
    """
    zp = ZeroPoint(value=15.335, std=0.2, n_points=50, source="truth")
    table = make_table(np.geomspace(50.0, 50000.0, 200))
    lim = limiting_magnitude(table, zp)
    assert lim == pytest.approx(10.7126, abs=1e-4)
    # 顺序也钉成字面量：9.2126 < 10.3376 < 10.7126 < 11.0126。
    assert limiting_magnitude(table, zp, percentile=75.0) == pytest.approx(9.2126, abs=1e-4)
    assert limiting_magnitude(table, zp, percentile=90.0) == pytest.approx(10.3376, abs=1e-4)
    assert limiting_magnitude(table, zp, percentile=99.0) == pytest.approx(11.0126, abs=1e-4)
    assert 9.2126 < 10.3376 < lim < 11.0126


def test_limiting_magnitude_returns_builtin_float():
    """返回内建 ``float`` 而非 ``np.float64``，供 json 直接序列化。"""
    zp = ZeroPoint(value=15.335, std=0.2, n_points=50, source="truth")
    table = make_table(np.geomspace(50.0, 50000.0, 200))
    assert type(limiting_magnitude(table, zp)) is float


def test_limiting_magnitude_on_empty_table_is_nan():
    """空表没有暗端可言，返回 nan。"""
    zp = ZeroPoint(value=15.0, std=0.1, n_points=10, source="truth")
    assert np.isnan(limiting_magnitude(SourceTable.empty(0), zp))


def test_limiting_magnitude_on_all_nan_table_is_nan():
    """全部流量 <= 0 时返回 nan 且不留警告——这是扣背景失败那一帧的形态。

    原测试只测了空表；全 nan 才是真正会遇到的情形。
    """
    zp = ZeroPoint(value=15.0, std=0.1, n_points=10, source="truth")
    table = make_table([-5.0, 0.0, -1.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lim = limiting_magnitude(table, zp)
    assert np.isnan(lim)
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


def test_limiting_magnitude_on_mixed_table_is_the_instrumental_magnitude():
    """混合 ``[-1, 0, 1000]``：两个 nan 被跳过，只剩 1000 那一个源。

    这里断言的是**仪器星等**口径（零点取 0），实测 ``|-2.5*log10(1000)| = 7.5``，
    返回值本身是 ``-7.5``（星等越小越亮，未加零点时是负数）。若改用定标星等口径、
    显式带上 15.335 的零点，同一输入给的是 ``15.335 - 2.5*log10(1000) =
    7.8350``。两个数都对，差别只在加不加零点——本测试把两者都钉住，读者不必猜
    7.5 是哪来的。
    """
    table = make_table([-1.0, 0.0, 1000.0])
    zp_instrumental = ZeroPoint(value=0.0, std=0.0, n_points=3, source="none")
    assert limiting_magnitude(table, zp_instrumental) == pytest.approx(-7.5, abs=1e-9)
    assert abs(limiting_magnitude(table, zp_instrumental)) == pytest.approx(7.5, abs=1e-9)

    zp_calibrated = ZeroPoint(value=15.335, std=0.218, n_points=54, source="truth")
    assert limiting_magnitude(table, zp_calibrated) == pytest.approx(7.8350, abs=1e-4)


def test_zero_point_to_dict_is_json_ready():
    """``to_dict()`` 可直接过 json，且带上曝光时间字段。"""
    d = ZeroPoint(15.335, 0.218, 54, "truth", 0.08).to_dict()
    assert json.loads(json.dumps(d)) == d
    assert d["exposure_s"] == pytest.approx(0.08)
    assert set(d) == {"value", "std", "n_points", "source", "exposure_s"}


def test_photometry_report_round_trips_from_real_numpy_arrays():
    """报告字段必须是内建 ``float``，因为 numpy 标量的 json 兼容性是偶然的。

    ``np.float64`` 是 ``float`` 的**子类**，所以 ``json.dumps`` 恰好放它过去；
    但 ``np.float32`` 与 ``np.int64`` 都抛
    ``TypeError: Object of type float32 is not JSON serializable``（实测）。
    也就是说这是一个潜伏 bug：任何一条产出 float32 流量列的代码路径都会把它变成
    即时 bug。所以本测试从**真实 numpy 数组**取 nanmin/nanmax 来构造报告，并断言
    ``type(...) is float``（用 ``isinstance`` 断言不到——``np.float64`` 会混过去）。
    """
    zp = ZeroPoint(value=15.335, std=0.218, n_points=54, source="truth", exposure_s=0.03)
    table = make_table(np.geomspace(50.0, 50000.0, 200))
    mags = calibrate_table(table, zp)
    report = PhotometryReport(
        zero_point=zp,
        n_calibrated=int(np.isfinite(mags).sum()),
        mag_min=np.nanmin(mags),
        mag_max=np.nanmax(mags),
        limiting_mag=limiting_magnitude(table, zp),
        note="数据集 B 真值零点",
    )
    d = report.to_dict()
    assert json.loads(json.dumps(d)) == d
    for key in ("mag_min", "mag_max", "limiting_mag"):
        assert type(d[key]) is float, key
    assert type(d["n_calibrated"]) is int
    # 未定标时三个星等字段为 None，不能被 float(None) 打断。
    empty = PhotometryReport(zero_point=None, note="未定标").to_dict()
    assert json.loads(json.dumps(empty)) == empty
    assert empty["zero_point"] is None
    assert empty["mag_min"] is None


def test_dataset_b_zero_point_matches_baseline(dataset_b_dir):
    """复现基线：零点 15.335 mag，散度收紧到 ``abs=0.08``。

    ``std`` 的容差从 ``abs=0.15`` 收到 ``abs=0.08``。原来的 ``abs=0.15`` 接受
    0.068~0.368，是 5.4 倍的散度跨度，分不出 0.1 mag 测光和 0.35 mag 测光。
    ``abs=0.08`` 名义区间是 ``[0.138, 0.298]``，按实测拟合值反推，真实可分辨边缘
    约为 **0.145 <= σ <= 0.305**（σ=0.140 拟合 0.1373 在界外，σ=0.145 拟合
    0.1412 在界内；σ=0.305 拟合 0.2980 在界内，σ=0.310 拟合 0.3014 在界外）。

    拟合散度比真实 σ **低约 2~3%，两个成因量级相当**（每档 σ 独立抽样、2000 次、
    n=54 实测）：``np.std`` 默认 ``ddof=0`` 加上样本标准差的有限 n 偏差占约
    **1.4%**（n=54 的理论值 ``sqrt((n-1)/n) * c4(n) = 0.9860``），3σ 剪裁的
    **单侧性**占约 **1.2%**。实测「剪裁开 vs 剪裁关」之差在 σ=0.100/0.218/0.305/
    0.500 四档分别为 1.11% / 1.26% / 1.27% / 1.24%，合计欠量 2.22% / 2.78% /
    2.73% / 3.03%。

    **不要用「比值在 σ 上恒定」论证成因是 ddof 而非剪裁**：3σ 剪裁的门限是
    ``|d - median| <= 3*std(d)``，分子分母同尺度，所以剪裁是**尺度不变**的——
    实测同一抽样按 σ 缩放 50 倍，剪裁开与剪裁关的比值逐位相同。σ 扫描对这两个
    解释的区分力恰好为零。同理不要用「剪裁数中位为 0」论证剪裁无影响：剪裁是
    单侧效应，实测有剪裁的比例 0.147、剪裁数均值 0.168，足以移动均值。

    ``n_points >= 50`` 保留：同样 400 次试验里 ``n_points`` 最小 53、中位 55，
    剪裁只在有粗大离群点时才动手，不吃这点裕度。
    """
    from src.calib.hotpixel import build_from_sequence
    from src.calib.segment import tracking_segment
    from src.dataio.fits_loader import FrameSequence
    from src.detect.segmentation import detect_sequence
    from src.register.solver import register_sequence
    from src.target.dual_frame import find_targets
    from src.validate.truth import load_truth, match_frames

    seq = FrameSequence.from_directory(dataset_b_dir)
    seg = tracking_segment(seq.headers)
    frames = list(range(seg.start, seg.end + 1))
    hpm = build_from_sequence(seq, frames[:30])
    dets = detect_sequence(seq, frames, n_sigma=4.0, hot_clusters=hpm.clusters)
    reg = register_sequence(dets, frames, reference=seg.start)
    trk = find_targets(dets, frames, reg)[0].track

    truth = load_truth(seq.truth_path)
    mf, rows = match_frames(truth, seq, trk.frames)
    order = {f: i for i, f in enumerate(trk.frames)}
    zp = zero_point_from_truth(trk.flux[[order[f] for f in mf]], truth.mag[rows])
    print(
        f"\n[dataset B zero point] value={zp.value:.6f} std={zp.std:.6f} "
        f"n_points={zp.n_points} n_input={len(mf)} clipped={len(mf) - zp.n_points}"
    )
    assert zp.value == pytest.approx(15.335, abs=0.15)
    assert zp.std == pytest.approx(0.218, abs=0.08)
    assert zp.n_points >= 50
