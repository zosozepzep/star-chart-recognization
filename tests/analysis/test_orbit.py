from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.orbit import (
    TLEReference,
    cos_theta_sat,
    cross_check_tle,
    estimate_from_trajectory,
    estimate_height,
    height_from_mean_motion,
    load_tle,
    rate_bounds,
    slant_range_km,
    topocentric_rate_max,
    topocentric_rate_min,
)

# Task 19 交付实测的三个输入（唯一有效版本，见 docs/reports/measurements.md）。
ELEV_B_TRACKED = 15.784004      # dataset B 目标轨迹 f17..f70 的 ELEVATIO 均值，跨度 0.519 度
RATE_B = 770.150746             # Task 19 交付实测 mean_rate_arcsec_s（53 对口径）
SITE_LAT_B = 43.312             # SITELATI，两数据集同站
SIDEREAL_ARCSEC_S = 360.0 * 3600.0 / 86164.0905     # 15.0411 ''/s

# Truth-derived plate scale, 与 tests/target/test_trajectory.py:23 及台账一致。
SCALE = 6.179

# 旧任务书的仰角，只用在单调性与 cos(theta_sat) 那两条纯几何检验里（裁决 124 的
# 原始网格就是在这个仰角上跑的，换成 15.784004 会让下面照抄的端点值全部对不上）。
ELEV_LEGACY = 15.7775

TLE_PATH = Path(__file__).resolve().parents[2] / "docs" / "reference" / "tle.txt"


def _tle_available(norad_id: int) -> bool:
    try:
        load_tle(TLE_PATH, norad_id)
        return True
    except (KeyError, ValueError, OSError):
        return False


requires_tle = pytest.mark.skipif(
    not _tle_available(60385),
    reason="docs/reference/tle.txt 尚未抄录 NORAD 60385 的真实两行根数；"
           "容器无网络，需人工从 celestrak 抄入后本组校验自动生效",
)


# --------------------------------------------------------------------------
# 5.1 区间断言：两条必须成对（裁决 407）
# --------------------------------------------------------------------------


def test_estimate_reports_a_band_not_a_point():
    """轨道面取向未知 + 测站自转符号未定，实测区间宽 634.92 km——报单值是虚报精度。

    **两条端点断言是互补的，不是冗余的。** 控制器实测的十体变异杀伤表
    （参照 = [213.4411, 848.3592]）里有三体各自只动一个界，另一个端点逐位不变：

    | 变异体 | h_min | h_max | h_min 断言 | h_max 断言 |
    |---|---|---|---|---|
    | 正确实现 | 213.4411 | 848.3592 | 通过 | 通过 |
    | 丢掉测站项（裸带） | 258.3995 | 804.1031 | 拒 | 拒 |
    | 测站项用 cos(仰角) | 无根 | 862.7803 | converged=False | — |
    | 测站项**加到两个界**上 | 304.7285 | 848.3592 | **拒** | **通过** |
    | 测站项符号翻转（从上界减） | 213.4411 | 760.5519 | **通过** | **拒** |
    | 丢掉 cos(theta_sat) | 760.5519 | 848.3592 | **拒** | **通过** |
    | sin(theta_sat) 分母用 R | 无根 | 848.3592 | converged=False | — |
    | 斜距换成高度 | 1002.4935 | 1944.1552 | 拒 | 拒 |
    | 斜距丢曲率项 | 无根 | 353.1338 | converged=False | — |
    | 环绕速度在地表算 | 219.8279 | 916.8212 | 拒 | 拒 |

    删掉任何一条，上表三体里就有一体活下来。

    ``rel=1e-4`` 的依据：正确实现与钉值相对差 < 1e-7（同一套公式同一组输入，钉值
    本身只写到 4 位小数），而最近的变异体（「环绕速度在地表算」的
    ``h_min = 219.8279``）相对差 2.99e-02——浮点余量 2 个数量级，杀伤余量 299 倍。
    """
    est = estimate_height(RATE_B, ELEV_B_TRACKED, site_lat_deg=SITE_LAT_B)
    assert est.converged
    assert est.height_min_km == pytest.approx(213.4411, rel=1e-4)
    assert est.height_max_km == pytest.approx(848.3592, rel=1e-4)
    assert est.height_min_km < est.height_km < est.height_max_km
    # 噪声全宽 15.7048 km，几何 634.92 km：实测比值 40.43，故 20 倍有 102% 余量
    assert (est.height_max_km - est.height_min_km) > 20.0 * 15.7048


def test_true_height_must_lie_inside_the_band():
    """带内任意高度都能产生实测角速度；带外不能。实测 800 km 在带内、900 km 在带外。

    实测宽带 800 -> [365.941, 808.561] 含 770.15；900 -> [344.830, 733.348] 不含。
    裸带同判（800 -> [401.066, 773.436] 含；900 -> [376.899, 701.279] 不含），
    故本测试不依赖测站项的取舍。
    """
    lo, hi = rate_bounds(800.0, ELEV_B_TRACKED, site_lat_deg=SITE_LAT_B)
    assert lo <= RATE_B <= hi
    lo, hi = rate_bounds(900.0, ELEV_B_TRACKED, site_lat_deg=SITE_LAT_B)
    assert not (lo <= RATE_B <= hi)


def test_rate_bounds_reproduce_the_exclusion_table():
    """第 6 节排除表的宽带口径实测值（仰角 15.784004、角速度 770.150746）。

    精确截止 848.3592 km：h<=800 自相容，h>=850 已被排除。
    """
    expected = {
        500.0: (466.508, 1197.440),
        600.0: (423.858, 1027.410),
        700.0: (391.583, 903.424),
        800.0: (365.941, 808.561),
        850.0: (354.912, 768.918),
        900.0: (344.830, 733.348),
        1000.0: (326.976, 672.059),
        1080.0: (314.475, 630.539),
        1200.0: (298.018, 577.761),
    }
    for height_km, (want_lo, want_hi) in expected.items():
        lo, hi = rate_bounds(height_km, ELEV_B_TRACKED, site_lat_deg=SITE_LAT_B)
        assert lo == pytest.approx(want_lo, rel=1e-5), height_km
        assert hi == pytest.approx(want_hi, rel=1e-5), height_km
        inside = lo <= RATE_B <= hi
        assert inside is (height_km <= 800.0), height_km


# --------------------------------------------------------------------------
# 5.2 数据集测试
# --------------------------------------------------------------------------


def test_dataset_b_rate_excludes_high_orbits(dataset_b_dir):
    """实测 770.15 ''/s、仰角 15.784 度：h<=800 km 自相容，h>=900 km 被排除。

    这是本反演能给出的最强结论，也是最**稳健**的一条：控制器实测十体变异里
    **六体给出与正确实现完全相同的判定**（含裁决 116 抓的 cos(仰角) 原缺陷），
    只有四体（测站项符号翻转、两种斜距错、地表速度）改变它。
    所以它答辩站得住，但**不能只守它**——定量端点断言（5.1）必须同时在。

    若 TLE 显示更高轨道，须在报告中如实说明三种可能原因
    （目标更低 / ELEVATIO 是机架指向 / 圆轨道假设不成立），不得调参迁就。

    区间收到 ±3 km 是为了杀伤力：实测「环绕速度在地表算」这一体
    ``h_min = 219.8279`` 落在旧任务书的 (200, 230) 内、会通过。
    """
    est = estimate_height(RATE_B, ELEV_B_TRACKED, site_lat_deg=SITE_LAT_B)
    assert est.converged
    assert 210.0 < est.height_min_km < 217.0        # 实测 213.4411
    assert 845.0 < est.height_max_km < 852.0        # 实测 848.3592
    assert 88.0 < est.period_min_min < 90.0         # 实测 88.7654
    assert 101.0 < est.period_max_min < 103.0       # 实测 101.8947


def test_dataset_b_end_to_end_reproduces_the_band(dataset_b_dir):
    """从 FITS 一路跑到高度带，确认 estimate_from_trajectory 取的是轨迹自己的帧。

    实测：轨迹 f17..f70（54 点），仰角均值 15.784004、跨度 0.519000，
    带 [213.4411, 848.3592]。传全 80 帧 headers 结果必须**不变**（裁决 409）：
    若实现对整个 headers 取均值，均值会变成 15.700203，h_max 变 845.9494，差 2.41 km。
    """
    from src.calib.hotpixel import build_from_sequence
    from src.calib.segment import tracking_segment
    from src.dataio.fits_loader import FrameSequence
    from src.detect.segmentation import detect_sequence
    from src.register.solver import register_sequence
    from src.target.dual_frame import find_targets
    from src.target.trajectory import build_trajectory

    seq = FrameSequence.from_directory(dataset_b_dir)
    seg = tracking_segment(seq.headers)
    frames = list(range(seg.start, seg.end + 1))
    hpm = build_from_sequence(seq, frames[:30])
    dets = detect_sequence(seq, frames, n_sigma=4.0, hot_clusters=hpm.clusters)
    reg = register_sequence(dets, frames, reference=seg.start)
    trk = find_targets(dets, frames, reg)[0].track
    assert trk.length == 54 and trk.start == 17 and trk.end == 70

    trj = build_trajectory(trk, seq, SCALE)
    assert len(seq.headers) == 80                   # 传的是全 80 帧，不是跟踪窗
    est = estimate_from_trajectory(trj, seq.headers)
    assert est.elevation_deg == pytest.approx(15.784004, abs=1e-5)
    assert est.rate_arcsec_s == pytest.approx(770.150746, rel=1e-6)
    assert est.height_min_km == pytest.approx(213.4411, rel=1e-4)
    assert est.height_max_km == pytest.approx(848.3592, rel=1e-4)
    assert "0.519" in est.note or "0.52" in est.note          # 真实跨度，见裁决 409
    # 误差预算的三个非几何项由**轨迹自己**的散度与仰角窗决定，不是模块缺省值：
    # 实测 std/mean = 5.908433/770.150746 -> 15.7048，窗端 15.43476/15.95376 -> 14.9325
    budget = est.error_budget
    assert budget["noise_km"] == pytest.approx(15.7048, abs=5e-5)
    assert budget["elevation_km"] == pytest.approx(14.9325, abs=5e-5)
    assert budget["projection_km"] == pytest.approx(0.1637, abs=5e-5)
    assert budget["dominant"] == "geometry"


# --------------------------------------------------------------------------
# 5.3 estimate_from_trajectory 的夹具（裁决 409）
# --------------------------------------------------------------------------


def test_estimate_from_trajectory_indexes_headers_by_frame():
    """按 point.frame 索引，不对传进来的整个 headers 取均值。

    夹具给 4 个 header 但轨迹只占第 1、2 两帧：若实现对全部 4 个取均值，
    仰角会是 20.0 而不是 15.690075，断言会红。
    """
    class H:
        def __init__(self, e):
            self.elevation_deg, self.site_lat_deg = e, 43.312

    class P:
        def __init__(self, f):
            self.frame = f

    class T:
        mean_rate_arcsec_s = 770.150746
        points = [P(1), P(2)]

    headers = [H(99.0), H(15.42639), H(15.95376), H(99.0)]
    est = estimate_from_trajectory(T(), headers)
    assert est.elevation_deg == pytest.approx(15.690075, abs=1e-6)
    assert "0.527" in est.note or "0.53" in est.note   # 该夹具的跨度是 0.52737


def test_estimate_from_trajectory_guards_the_frame_index():
    class P:
        def __init__(self, f):
            self.frame = f

    class T:
        mean_rate_arcsec_s = 770.150746
        points = [P(0), P(7)]

    with pytest.raises(ValueError, match="帧号超出"):
        estimate_from_trajectory(
            T(), [type("H", (), {"elevation_deg": 15.0, "site_lat_deg": 43.312})()]
        )


def test_estimate_from_trajectory_honours_an_explicit_site_latitude():
    """``site_lat_deg`` 缺省时取 ``picked[0].site_lat_deg``，显式给出时以显式值为准。

    测站项 ∝ cos(纬度)：纬度越低测站项越大，带上界越高。实测 43.312 度 -> 848.3592、
    20 度 -> 861.3882 km。这条守的是「缺省值不是硬编码」。

    **不用纬度 0 度做对照**：实测纬度低于约 10 度时宽带下界在 200 km 处仍低于实测
    角速度，``height_min`` 落在搜索区间外，``estimate_height`` 判不收敛、两个界都是
    ``None``（正是那条「界受搜索区间截断」的分支）。
    """
    class H:
        elevation_deg = 15.784004
        site_lat_deg = 43.312

    class P:
        frame = 0

    class T:
        mean_rate_arcsec_s = 770.150746
        points = [P()]

    default = estimate_from_trajectory(T(), [H()])
    lower_lat = estimate_from_trajectory(T(), [H()], site_lat_deg=20.0)
    assert default.height_max_km == pytest.approx(848.3592, rel=1e-4)
    assert lower_lat.height_max_km == pytest.approx(861.3882, rel=1e-4)
    assert lower_lat.height_max_km > default.height_max_km
    # 纬度 10 度：下界跑出搜索区间，判不收敛而不是给一个截断的假区间
    equator = estimate_from_trajectory(T(), [H()], site_lat_deg=10.0)
    assert not equator.converged
    assert equator.height_min_km is None and equator.height_max_km is None
    assert "搜索区间" in equator.note


# --------------------------------------------------------------------------
# 5.4 三条不收敛分支各一测试（裁决 121）
# --------------------------------------------------------------------------


def test_estimate_flags_nonpositive_rate():
    for bad in (0.0, -5.0):
        est = estimate_height(bad, 45.0, site_lat_deg=SITE_LAT_B)
        assert not est.converged and est.height_km is None
        assert "非正" in est.note


def test_estimate_flags_rate_below_band():
    est = estimate_height(0.001, 45.0, site_lat_deg=SITE_LAT_B)
    assert not est.converged and est.height_km is None
    assert "低于" in est.note        # 与"高于"必须可区分
    assert "12.77" in est.note       # 带下界写进消息，便于诊断（实测 12.7743）


def test_estimate_flags_rate_above_band():
    est = estimate_height(1e6, 45.0, site_lat_deg=SITE_LAT_B)
    assert not est.converged and est.height_km is None
    assert "高于" in est.note


def test_estimate_flags_nonfinite_rate():
    est = estimate_height(float("nan"), 45.0, site_lat_deg=SITE_LAT_B)
    assert not est.converged and est.height_km is None


def test_the_three_nonconvergence_notes_are_distinct():
    """三条分支三句**不同**的中文 note，否则诊断时分不开。"""
    notes = {
        estimate_height(0.0, 45.0, site_lat_deg=SITE_LAT_B).note,
        estimate_height(0.001, 45.0, site_lat_deg=SITE_LAT_B).note,
        estimate_height(1e6, 45.0, site_lat_deg=SITE_LAT_B).note,
        estimate_height(float("nan"), 45.0, site_lat_deg=SITE_LAT_B).note,
    }
    assert len(notes) == 4


def test_wide_band_at_45_degrees_matches_the_measured_limits():
    """E=45 度的宽带 [12.7743, 6012.2563]（裸带 [14.4503, 5761.7634]）。

    这两个数就是上面三条不收敛分支的判据边界：0.001 低于 12.7743，1e6 高于
    6012.2563。写成断言是为了「12.77」那个 note 断言不至于成为孤证。
    """
    lo_at_top, _ = rate_bounds(40000.0, 45.0, site_lat_deg=SITE_LAT_B)
    _, hi_at_bottom = rate_bounds(200.0, 45.0, site_lat_deg=SITE_LAT_B)
    assert lo_at_top == pytest.approx(12.7743, rel=1e-4)
    assert hi_at_bottom == pytest.approx(6012.2563, rel=1e-4)
    assert topocentric_rate_min(40000.0, 45.0) == pytest.approx(14.4503, rel=1e-4)
    assert topocentric_rate_max(200.0, 45.0) == pytest.approx(5761.7634, rel=1e-4)


# --------------------------------------------------------------------------
# 5.5 恒星速率（裁决 122 + 310）
# --------------------------------------------------------------------------


def test_geostationary_rate_is_consistent_with_the_sidereal_rate():
    """同步目标相对恒星以恒星速率移动，但它的视运动几乎全部来自测站自转
    （实测 omega_E*R*cos(lat)=0.338421 km/s 占 v_GEO=3.074661 km/s 的 11.01%），
    故必须用并入测站项的区间来判：裸的 v/rho 上界在实测的每一个仰角都排除
    恒星速率（45 度处 16.9515 vs 15.0411，差 12.70%）。
    仰角取 39 度而非 45 度：从纬度 43.312 度看同步轨道最大可达仰角实测 40.0373 度，
    所以"35786 km @ 45 度"不可能是同步目标。
    """
    lo, hi = rate_bounds(35786.0, 39.0, site_lat_deg=SITE_LAT_B)
    assert lo <= SIDEREAL_ARCSEC_S <= hi          # 实测 [14.7920, 18.5958]
    assert lo == pytest.approx(14.7920, rel=1e-4)
    assert hi == pytest.approx(18.5958, rel=1e-4)
    # 裸上界在 45 度处实测 16.9515，比恒星速率高 12.70%——单靠它会排除同步目标
    assert topocentric_rate_max(35786.0, 45.0) == pytest.approx(16.9515, rel=1e-4)
    assert topocentric_rate_max(35786.0, 45.0) > SIDEREAL_ARCSEC_S
    # 实测 LEO 550 km 上界 2088.4541 ''/s = 恒星速率的 138.85 倍，> 100x 有 39% 余量
    assert topocentric_rate_max(550.0, 45.0) > 100.0 * SIDEREAL_ARCSEC_S


# --------------------------------------------------------------------------
# 5.6 单调性与守卫（裁决 124）
# --------------------------------------------------------------------------


def test_all_four_bound_functions_decrease_with_height():
    """brentq 的唯一性前提。实测 6000 点 n_increase=0：
    rate_max_bare 2540.7378 -> 13.6698、rate_min_bare 913.9129 -> 13.5496、
    宽带上界 2651.1964 -> 15.2478、宽带下界 803.4542 -> 11.9716（E=15.7775）。
    **裁决 124 当年只验了裸的两个，而 estimate_height 反演的是宽带那两个。**
    """
    heights = np.linspace(200.0, 40000.0, 6000)
    curves = {
        "rate_max_bare": (
            np.array([topocentric_rate_max(float(h), ELEV_LEGACY) for h in heights]),
            2540.7378, 13.6698,
        ),
        "rate_min_bare": (
            np.array([topocentric_rate_min(float(h), ELEV_LEGACY) for h in heights]),
            913.9129, 13.5496,
        ),
        "wide_hi": (
            np.array([rate_bounds(float(h), ELEV_LEGACY, site_lat_deg=SITE_LAT_B)[1]
                      for h in heights]),
            2651.1964, 15.2478,
        ),
        "wide_lo": (
            np.array([rate_bounds(float(h), ELEV_LEGACY, site_lat_deg=SITE_LAT_B)[0]
                      for h in heights]),
            803.4542, 11.9716,
        ),
    }
    for name, (values, first, last) in curves.items():
        assert int((np.diff(values) > 0).sum()) == 0, name
        # 容差取 5e-5 = 四位小数的舍入半宽：钉值本身只写到 4 位，实测端点是
        # 2540.7377536 / 13.6697919 / 913.9128931 / 13.5495515 / 2651.1964411 /
        # 15.2477924 / 803.4542057 / 11.9715510，与钉值最大差 4.9e-05。
        assert values[0] == pytest.approx(first, abs=5e-5), name
        assert values[-1] == pytest.approx(last, abs=5e-5), name


def test_slant_range_degenerates_at_zenith():
    """仰角 90 度时斜距退化为高度，实测 slant(1080, 90) = 1080.000。"""
    assert slant_range_km(1080.0, 90.0) == pytest.approx(1080.0, rel=1e-9)
    # 同一高度在低仰角上斜距大得多：15.784004 度处实测 2502.1883 km = 高度的 2.3168 倍。
    # 「把斜距当成高度用」这个变异体实测把带推到 [1002.4935, 1944.1552]。
    assert slant_range_km(1080.0, ELEV_B_TRACKED) == pytest.approx(2502.1883, abs=5e-5)
    assert slant_range_km(1080.0, 45.0) == pytest.approx(1429.9670, abs=5e-5)


def test_topocentric_rate_rejects_nonpositive_height():
    """实测 height_km <= 0 抛 ValueError（经 slant_range_km 的守卫）。"""
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="高度"):
            topocentric_rate_max(bad, 45.0)
        with pytest.raises(ValueError, match="高度"):
            slant_range_km(bad, 45.0)


def test_cos_theta_sat_depends_on_height_not_just_elevation():
    """裁决 315：cos(theta_sat) 同时依赖高度与仰角，0.568 只是 h=1080 那一行。
    实测 E=15.7775：h=400 -> 0.42427、800 -> 0.51851、1080 -> 0.56808、2000 -> 0.68066。
    """
    expected = {400.0: 0.42427, 800.0: 0.51851, 1080.0: 0.56808, 2000.0: 0.68066}
    for height_km, want in expected.items():
        assert cos_theta_sat(height_km, ELEV_LEGACY) == pytest.approx(want, abs=5e-6)
    # 单调升：高度越高，视线越接近垂直于 r_sat
    values = [cos_theta_sat(h, ELEV_LEGACY) for h in sorted(expected)]
    assert values == sorted(values)


# --------------------------------------------------------------------------
# to_dict 与配置
# --------------------------------------------------------------------------


def test_to_dict_reports_the_band_and_the_error_budget():
    """``to_dict()`` 必须能过 ``json.dumps(..., allow_nan=False)``，且四项误差预算
    **全宽口径**、几何项主导（实测 634.9181 vs 15.7048+14.9325+0.1637=30.8011，
    比值 20.61）。
    """
    est = estimate_height(RATE_B, ELEV_B_TRACKED, site_lat_deg=SITE_LAT_B)
    out = est.to_dict()
    text = json.dumps(out, allow_nan=False, ensure_ascii=False)
    assert len(json.loads(text)) == len(out)
    assert out["model"] == "圆轨道假设 + 轨道面取向未知，故给出高度区间；非定轨结果"
    for key in ("height_min_km", "height_max_km", "range_min_km", "range_max_km",
                "period_min_min", "period_max_min", "height_km", "velocity_km_s"):
        assert type(out[key]) is float, key

    budget = out["error_budget"]
    assert budget["dominant"] == "geometry"
    assert budget["geometry_km"] == pytest.approx(
        est.height_max_km - est.height_min_km, rel=1e-12
    )
    # 四项全宽口径的实测值，逐项钉住（abs=5e-5 = 四位小数的舍入半宽）
    assert budget["geometry_km"] == pytest.approx(634.9181, abs=5e-5)
    assert budget["noise_km"] == pytest.approx(15.7048, abs=5e-5)
    assert budget["elevation_km"] == pytest.approx(14.9325, abs=5e-5)
    assert budget["projection_km"] == pytest.approx(0.1637, abs=5e-5)
    rest = budget["noise_km"] + budget["elevation_km"] + budget["projection_km"]
    assert rest == pytest.approx(30.8011, abs=1e-3)
    # 实测比值 20.61；断言 10 倍留一半余量。修正被高估的投影项让主导结论更强。
    assert budget["geometry_km"] / rest == pytest.approx(20.61, abs=0.01)
    assert budget["geometry_km"] > 10.0 * rest
    # 投影项来自 R397(c) 实测 +0.016%，不是 R303 作废的 +1.4%（后者折算 14.11 km）
    assert budget["projection_km"] < 1.0


def test_to_dict_of_a_nonconverged_estimate_is_still_json_safe():
    """不收敛时区间字段为 ``None``，仍须是合法 JSON（裸 ``NaN`` token 不是）。"""
    est = estimate_height(1e6, 45.0, site_lat_deg=SITE_LAT_B)
    text = json.dumps(est.to_dict(), allow_nan=False, ensure_ascii=False)
    assert "NaN" not in text
    assert json.loads(text)["converged"] is False


def test_config_carries_the_orbit_bracket(cfg):
    """``orbit.bracket_km``：brentq 的搜索区间，模块体内不留裸数字。"""
    assert cfg["orbit"]["bracket_km"] == [200.0, 40000.0]


def test_estimate_records_the_bracket_it_searched():
    est = estimate_height(RATE_B, ELEV_B_TRACKED, site_lat_deg=SITE_LAT_B)
    assert est.bracket_km == (200.0, 40000.0)


# --------------------------------------------------------------------------
# 5.7 TLE（裁决 117/118/123/314/410）
# --------------------------------------------------------------------------


def test_mean_motion_round_trip_covers_leo():
    """``height_from_mean_motion`` 的参考量（裁决 314，供报告与答辩）。

    实测 h(13.479)=1079.99、h(15.00)=566.90、h(14.50)=725.65、h(2.0)=20232.09 km。
    """
    expected = {13.479: 1079.99, 15.0: 566.90, 14.5: 725.65, 2.0: 20232.09}
    for n, want in expected.items():
        assert height_from_mean_motion(n) == pytest.approx(want, abs=0.01)
    with pytest.raises(ValueError, match="平均运动"):
        height_from_mean_motion(0.0)


@requires_tle
def test_cross_check_uses_the_measured_rate_not_a_round_trip():
    """交叉校验必须拿实测角速度去比 TLE，而不是把模型输出喂回自己的反函数。

    实测：一个完全丢掉斜距的错误模型（omega = sqrt(mu/(R+h))/h），往返相对误差
    0.00e+00，同样通过往返断言——往返测试对模型正确性完全无知。
    """
    ref = load_tle(TLE_PATH, 60385)
    est = estimate_height(RATE_B, ELEV_B_TRACKED, site_lat_deg=SITE_LAT_B)
    out = cross_check_tle(est, ref)
    assert out["agrees"] in (True, False)     # 判定有做出来，不断言判定为真
    assert out["reference_height_km"] == pytest.approx(ref.height_km, rel=1e-12)
    assert "km" in out["note"]


@requires_tle
def test_tle_mean_motion_gives_leo_height():
    """只判「抄录的行解析出来是个 LEO 量级的数」，不预判星座标称高度。

    裁决 410：两条断言**不是逐位等价，而是一宽一窄**。实测
    11.3 < n < 16.3 允许 h in [192.54, 2010.34]，比 200 < h < 2000 略宽，
    所以实际生效的是高度断言，``n`` 那条永不单独拒（反例 n=16.29 -> h=195.23，
    由高度断言拒掉）。这样取是为了不出现两条断言互相矛盾——用 12 < n < 17 就会：
    实测 h=1800 km 的 n=11.7387，过 h<2000 而不过 n>12。
    """
    tle = load_tle(TLE_PATH, 60385)
    assert 11.3 < tle.mean_motion_rev_day < 16.3
    h = height_from_mean_motion(tle.mean_motion_rev_day)
    assert 200.0 < h < 2000.0


@requires_tle
def test_cross_check_output_survives_json_dumps():
    """``bool(...)`` 那层包裹必须保留：实测 ``json.dumps(np.bool_(True))`` 抛
    ``TypeError``（numpy 2.2.6 的已知不对称，``np.float64`` 可序列化，
    ``np.bool_`` 与 ``np.int64`` 不行）。
    """
    ref = load_tle(TLE_PATH, 60385)
    est = estimate_height(RATE_B, ELEV_B_TRACKED, site_lat_deg=SITE_LAT_B)
    out = cross_check_tle(est, ref)
    assert type(out["agrees"]) is bool
    assert type(out["inside_band"]) is bool
    json.dumps(out, allow_nan=False, ensure_ascii=False)


@requires_tle
def test_cross_check_flags_a_reference_outside_the_band():
    fake = TLEReference(name="fake", norad_id=1, mean_motion_rev_day=2.0)
    assert fake.height_km > 20000.0          # 实测 20232.09 km
    est = estimate_height(RATE_B, ELEV_B_TRACKED, site_lat_deg=SITE_LAT_B)
    out = cross_check_tle(est, fake)
    assert out["agrees"] is False
    assert out["margin_km"] > 10000.0        # 实测 19383.73（带上界 848.3592）


def test_cross_check_accepts_a_reference_inside_the_band():
    """h(n=14.50)=725.65 落在实测宽带 [213.4411, 848.3592] 内（实测 True）。

    本条不挂 ``@requires_tle``：它用合成 ``TLEReference``，不读文件。带内
    ``margin_km`` 必须是 0，否则「带外距离」这个量没有零点。
    """
    inside = TLEReference(name="synthetic", norad_id=0, mean_motion_rev_day=14.5)
    assert inside.height_km == pytest.approx(725.65, abs=0.01)
    est = estimate_height(RATE_B, ELEV_B_TRACKED, site_lat_deg=SITE_LAT_B)
    out = cross_check_tle(est, inside)
    assert out["agrees"] is True
    assert out["inside_band"] is True
    assert out["margin_km"] == 0.0
    assert out["estimated_height_min_km"] == pytest.approx(213.4411, rel=1e-4)
    assert out["estimated_height_max_km"] == pytest.approx(848.3592, rel=1e-4)
    assert out["reference"] == "synthetic"
    # h(n=13.479)=1079.99 高出带上界 231.64 km（旧任务书 231.29）
    above = TLEReference(name="q7-nominal", norad_id=0, mean_motion_rev_day=13.479)
    out = cross_check_tle(est, above)
    assert out["agrees"] is False
    assert out["margin_km"] == pytest.approx(231.64, abs=0.01)


def test_cross_check_of_a_nonconverged_estimate_does_not_agree():
    """不收敛时没有带，判定只能是「不一致」，且不得抛。"""
    est = estimate_height(1e6, 45.0, site_lat_deg=SITE_LAT_B)
    out = cross_check_tle(est, TLEReference(name="x", norad_id=0, mean_motion_rev_day=14.5))
    assert out["agrees"] is False
    assert out["inside_band"] is False
    assert out["estimated_height_min_km"] is None
    json.dumps(out, allow_nan=False, ensure_ascii=False)


def test_load_tle_reports_malformed_mean_motion(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("NAME\n1 60385U x\n2 60385 ...\n", encoding="utf-8")
    with pytest.raises(ValueError, match="平均运动"):
        load_tle(p, 60385)


def test_load_tle_parses_the_mean_motion_columns(tmp_path):
    """实测真实格式行：ln[2:7] -> '60385'，ln[52:63] -> '13.5432109 '。
    TLE 第 2 行第 53-63 列是平均运动，0-based 切片 [52:63]。
    """
    line1 = "1 60385U 24123A   26200.50000000  .00001234  00000+0  12345-4 0  9990"
    line2 = "2 60385  89.5678 123.4567 0001234  12.3456 347.8901 13.5432109 123456"
    assert len(line2) == 69                      # TLE 原始列宽
    assert line2[2:7] == "60385"
    assert line2[52:63] == "13.5432109 "
    p = tmp_path / "tle.txt"
    p.write_text(
        "# 注释行必须被跳过\n\nQIANFAN-7\n%s\n%s\n" % (line1, line2), encoding="utf-8"
    )
    ref = load_tle(p, 60385)
    assert ref.name == "QIANFAN-7"
    assert ref.norad_id == 60385
    assert ref.mean_motion_rev_day == pytest.approx(13.5432109, rel=1e-12)
    assert ref.height_km == pytest.approx(height_from_mean_motion(13.5432109), rel=1e-12)
    with pytest.raises(KeyError, match="60394"):
        load_tle(p, 60394)


def test_the_shipped_tle_file_carries_only_the_header_comments():
    """``docs/reference/tle.txt`` 里**没有**真实两行根数是判过的合法状态
    （容器无网络，需人工从 celestrak 抄入）。这条钉住「不许伪造数据行凑绿」。

    抄入真实根数后本条会红——那时把它改成「已抄录」的正向断言，同时四条
    ``@requires_tle`` 自动放行。
    """
    text = TLE_PATH.read_text(encoding="utf-8")
    assert "celestrak" in text
    assert "60385" in text and "60394" in text
    data_lines = [
        ln for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    assert data_lines == []
    assert not _tle_available(60385)
