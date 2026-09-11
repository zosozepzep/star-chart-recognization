from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from src.astrometry.photometry import ZeroPoint
from src.target.dual_frame import Track
from src.target.trajectory import (
    Trajectory,
    angular_rates,
    build_trajectory,
    offset_to_radec,
)

# Truth-derived plate scale (TruthReport.plate_scale_arcsec_px, measured 6.179047).
# Rounded to 6.179 throughout so the closed-form expectations below stay readable;
# the module itself never sees a dataset path, only this float.
SCALE = 6.179

# Fixture epoch. Any epoch near the real data works; it is pinned so the
# ``t.unix`` degradation test below is reproducible (unix ~1.75e9, float64
# resolution ~2.4e-07 s there).
EPOCH = Time("2026-07-22T17:26:44.267", format="isot", scale="utc")


def make_track(n=20, step=(125.73, -5.0), start=(100.0, 200.0), flux=1000.0) -> Track:
    """20 点匀速直线，步长模长实测 125.829380115 px（含 dy）。

    ``xy_det`` 与 ``xy_sky`` 刻意取同一个数组：本文件测的是时序表与角速度，
    两系之别由 ``tests/target/test_dual_frame.py`` 守。
    """
    xy = np.array([start], dtype=np.float64) + np.arange(n)[:, None] * np.asarray(
        step, dtype=np.float64
    )
    return Track(
        frames=list(range(n)),
        xy_det=xy.copy(),
        xy_sky=xy.copy(),
        flux=np.full(n, float(flux)),
        peak=np.full(n, 300.0),
        elongation=np.full(n, 1.2),
    )


class FakeSeq:
    """最小序列替身：只有 ``dataset_id`` 与 ``times()``。

    刻意**不**定义 ``__len__``——帧号上界守卫必须写 ``len(sequence.times())``
    而不是 ``len(sequence)``（裁决 304）。写成后者时本类实测抛
    ``TypeError: object of type 'FakeSeq' has no len()``，本文件所有传 FakeSeq
    的测试会当场炸。不要为迁就守卫给本类加 ``__len__``。
    """

    dataset_id = "fixture"

    def __init__(self, n=20, cadence_s=1.008):
        self.n = n
        self.cadence_s = cadence_s

    def times(self) -> Time:
        return EPOCH + np.arange(self.n) * self.cadence_s * u.s


def rotate_flip(xy: np.ndarray, rotation_deg: float, flip: bool) -> np.ndarray:
    """旋转（可选先做 x 翻转）——正交变换，严格保模长。"""
    a = np.deg2rad(rotation_deg)
    m = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float64)
    if flip:
        m = m @ np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    return np.asarray(xy, dtype=np.float64) @ m.T


# --------------------------------------------------------------------------
# 精确值（rel=1e-9，裁决 108/302）
# --------------------------------------------------------------------------


def test_trajectory_reports_stable_rate():
    trj = build_trajectory(make_track(), FakeSeq(), SCALE)
    assert len(trj) == 20
    assert trj.source == "fixture"
    assert trj.scale_arcsec_px == SCALE
    # 精确值：|(125.73, -5.0)| = 125.829380115 px，含 dy。只算 dx 会得到 770.719911，
    # 相对差 7.9e-4——rel=0.02 分辨不出这个实现错误，所以钉到 1e-9。
    assert trj.mean_rate_arcsec_s == pytest.approx(771.329106873, rel=1e-9)
    # 完美匀速直线，散度应为浮点噪声级（实测 9.66e-13），不是"小于均值的 5%"。
    assert trj.rate_std_arcsec_s < 1e-6
    # 只算 dx 的那个错误实现给出的值，写出来供读者对照上面的 7.9e-4。
    assert trj.mean_rate_arcsec_s != pytest.approx(770.719911, rel=1e-9)


def test_total_arc_requires_the_scale_to_be_passed():
    """scale 必须走构造参数：默认 1.0 时 total_arc 会少乘一个像素比例尺，无报错。"""
    pts = build_trajectory(make_track(), FakeSeq(), SCALE).points
    bare = Trajectory(points=pts, source="x")            # 没传 scale
    withs = Trajectory(points=pts, source="x", scale_arcsec_px=SCALE)
    assert withs.total_arc_arcsec == pytest.approx(bare.total_arc_arcsec * SCALE, rel=1e-12)
    assert withs.total_arc_arcsec == pytest.approx(14772.495055, rel=1e-9)


def test_duration_uses_absolute_unix_seconds_not_the_isot_string():
    """``isot`` 默认 3 位小数，节拍 1.0075 s 时往返实测损失 5.0e-04 s。

    1.008 / 1.009 / 0.030 s 的节拍下损失恰为 0，所以只有非三位小数的节拍能
    钉住这条。``time_isot`` 保留供 ``to_records()`` 与中文报告，不参与算术。
    """
    trj = build_trajectory(make_track(), FakeSeq(cadence_s=1.0075), SCALE)
    assert trj.duration_s == pytest.approx(19 * 1.0075, abs=1e-7)
    # 经 isot 字符串往返的那条错路：19 步各被截到 3 位小数。
    isot = Time([p.time_isot for p in trj.points], format="isot", scale="utc")
    round_tripped = float((isot[-1] - isot[0]).sec)
    assert abs(round_tripped - 19 * 1.0075) == pytest.approx(5.0e-04, rel=0.05)


def test_relative_time_base_keeps_the_1e_9_consistency():
    """``(t - t[0]).sec`` 与 ``t.unix`` 两条时基在本夹具上分得开。

    ``t.unix`` 量级 1.75e9，float64 分辨率约 2.4e-07 s，20 点夹具的步长因此在
    1.007999897..1.008000135 之间抖动。实测 mean 相对误差 2.59e-09（超 rel=1e-9）、
    std 9.008e-05（超 1e-6），两条断言都会红。相对秒走 Time 内部的 jd1/jd2
    双精度对，不做绝对历元的大数相减，实测 mean 相对误差 4.02e-14、std 3.41e-09。

    这条夹具**不可删**：在真实 dataset B 数据上两种时基给出的 mean 逐位相同，
    所以数据集测试抓不到换时基这个错误。
    """
    seq = FakeSeq()
    t = seq.times()
    exact = 771.329106873

    rel_sec = angular_rates(make_track().xy_sky, (t - t[0]).sec, SCALE)
    assert np.nanmean(rel_sec[1:]) == pytest.approx(exact, rel=1e-9)
    assert np.nanstd(rel_sec[1:]) < 1e-6

    unix = angular_rates(make_track().xy_sky, t.unix, SCALE)
    assert abs(np.nanmean(unix[1:]) / exact - 1.0) > 1e-9
    assert np.nanstd(unix[1:]) > 1e-6


def test_mean_rate_excludes_the_forward_filled_first_point():
    """统计量取的是 n-1 个**独立**步速率，不含 ``rate[0]`` 那个前向填充值。

    ``angular_rates`` 把 ``rate[0]`` 填成 ``rate[1]``，好让速率列与点列同长；
    把它算进均值等于给第一步双倍权重。匀速夹具下两者相同，所以要一个变速夹具
    才分得开：20 点、19 步，前 10 步各 100 px、后 9 步各 200 px。独立均值是
    ``(10*100 + 9*200) / 19 * SCALE / 1.008``，含填充值的 20 项均值多算了一份
    第一步。这条同时钉住报告里「54 帧 53 对」那个口径——数据集 B 上两者相差
    770.1507 vs 769.8910 ″/s。
    """
    n = 20
    dx = np.concatenate([[0.0], np.where(np.arange(1, n) <= 10, 100.0, 200.0)])
    xy = np.column_stack((np.cumsum(dx), np.zeros(n)))
    trk = Track(
        frames=list(range(n)),
        xy_det=xy.copy(),
        xy_sky=xy.copy(),
        flux=np.full(n, 1000.0),
        peak=np.full(n, 300.0),
        elongation=np.full(n, 1.2),
    )
    trj = build_trajectory(trk, FakeSeq(), SCALE)
    rates = np.array([p.rate_arcsec_s for p in trj.points])
    assert rates[0] == pytest.approx(rates[1])          # 前向填充
    independent = (10 * 100.0 + 9 * 200.0) / 19.0 * SCALE / 1.008
    assert trj.mean_rate_arcsec_s == pytest.approx(independent, rel=1e-9)
    assert trj.mean_rate_arcsec_s != pytest.approx(float(np.mean(rates)), rel=1e-9)


# --------------------------------------------------------------------------
# 裁决 395：角速度只用得到一个标量比例尺
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rotation_deg, flip", [(0.0, False), (0.0, True), (34.0, False), (34.0, True),
                           (-20.0, False), (-20.0, True)]
)
def test_angular_rates_are_invariant_under_rotation_and_flip(rotation_deg, flip):
    """旋转是共形映射、翻转是正交变换，两者严格保模长。

    这就是 ``angular_rates`` 不需要 ``Orientation`` 形参的理由：定向信息在弧长与
    角速度上完全约掉。控制器实测经 ``pixel_to_offset`` 与标量直乘的最大差为
    1.137e-13，六种取值下 ``total_arc`` 都是 14772.495055。
    """
    seq = FakeSeq()
    t = seq.times()
    times = (t - t[0]).sec
    xy = make_track().xy_sky
    base = angular_rates(xy, times, SCALE)
    moved = angular_rates(rotate_flip(xy, rotation_deg, flip), times, SCALE)
    assert np.nanmax(np.abs(moved - base)) < 1e-9
    steps = np.hypot(*np.diff(rotate_flip(xy, rotation_deg, flip), axis=0).T)
    assert float(steps.sum()) * SCALE == pytest.approx(14772.495055, rel=1e-9)


# --------------------------------------------------------------------------
# 退化输入
# --------------------------------------------------------------------------


def test_angular_rates_returns_all_nan_for_zero_time_steps():
    """实测 ``angular_rates(xy, zeros(3), 6.179)`` → ``[nan, nan, nan]``。

    不切片成 ``[1:]``：否则删掉 ``rate[0] = rate[1]`` 这行前向填充，测试照样通过。
    """
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    rate = angular_rates(xy, np.zeros(3), SCALE)
    assert np.isnan(rate).all()


def test_angular_rates_guards_reversed_timestamps_with_a_strict_inequality():
    """守卫必须写 ``dt > 0.0``，不是 ``dt != 0.0``——后者产出负角速度。

    ``times=[0, 5, 2]``、步长 100 px：正确实现给 ``[123.5800, 123.5800, nan]``，
    而 ``dt != 0.0`` 实测给 ``[nan, 123.5800, -205.9667]``——第三项是一个**负**
    角速度，它会静默拉低均值。角速度不可为负。
    """
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    rate = angular_rates(xy, [0.0, 5.0, 2.0], SCALE)
    assert rate[1] == pytest.approx(100.0 * SCALE / 5.0)     # = 123.5800
    assert rate[0] == pytest.approx(rate[1])
    assert np.isnan(rate[2])
    assert np.nanmin(rate) > 0.0


@pytest.mark.parametrize("n", [0, 1])
def test_angular_rates_short_inputs(n):
    """``n=0`` → 空数组；``n=1`` → ``[nan]``。"""
    xy = np.zeros((n, 2))
    rate = angular_rates(xy, np.zeros(n), SCALE)
    assert rate.shape == (n,)
    assert np.isnan(rate).all()          # 空数组上 .all() 为 True，故也断言 shape


def test_trajectory_stats_are_nan_without_warnings_when_every_rate_is_nan():
    """全 nan 时 ``nanmean``/``nanstd`` 实测各打一条 ``RuntimeWarning``（共两条）。

    守卫写 ``if r.size and np.isfinite(r).any()``，否则返回 ``float("nan")``。
    """
    trk = make_track(n=3)
    seq = FakeSeq(n=3, cadence_s=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trj = build_trajectory(trk, seq, SCALE)
        assert np.isnan(trj.mean_rate_arcsec_s)
        assert np.isnan(trj.rate_std_arcsec_s)


def test_single_point_trajectory_has_nan_rate_and_zero_duration():
    trj = build_trajectory(make_track(n=1), FakeSeq(n=1), SCALE)
    assert len(trj) == 1
    assert np.isnan(trj.points[0].rate_arcsec_s)
    assert np.isnan(trj.mean_rate_arcsec_s)
    assert trj.duration_s == 0.0
    assert trj.total_arc_arcsec == 0.0


def test_build_trajectory_rejects_frames_beyond_the_sequence():
    """帧号上界守卫用 ``len(sequence.times())``，而 FakeSeq 没有 ``__len__``。"""
    trk = make_track(n=20)
    with pytest.raises(ValueError, match="帧号超出序列长度"):
        build_trajectory(trk, FakeSeq(n=10), SCALE)


def test_build_trajectory_rejects_an_empty_track():
    trk = make_track(n=0)
    with pytest.raises(ValueError, match="轨迹为空"):
        build_trajectory(trk, FakeSeq(), SCALE)


# --------------------------------------------------------------------------
# 裁决 86/105：星等必须用零点自带的曝光
# --------------------------------------------------------------------------


def test_trajectory_mag_uses_zero_point_exposure():
    """裁决 86：零点带曝光拟合时，星等必须同样做曝光归一，否则实测差 3.807 mag。"""
    zp = ZeroPoint(15.335, 0.218, 54, "truth", exposure_s=0.030)
    trj = build_trajectory(make_track(), FakeSeq(), SCALE, zero_point=zp)
    expected = 15.335 - 2.5 * np.log10(1000.0 / 0.030)
    assert trj.points[0].mag == pytest.approx(expected, abs=1e-6)
    # 不做曝光归一会得到 7.835，差 3.807 mag——没有这一行，读者看不出漏了多少
    assert abs(trj.points[0].mag - (15.335 - 2.5 * np.log10(1000.0))) > 3.0


def test_trajectory_mag_without_exposure_normalisation():
    """只覆盖 ``exposure_s=None`` 那条不归一的分支（零点未记曝光时的合法用法）。"""
    zp = ZeroPoint(15.335, 0.218, 54, "truth")
    trj = build_trajectory(make_track(), FakeSeq(), SCALE, zero_point=zp)
    assert zp.exposure_s is None
    assert trj.points[0].mag == pytest.approx(15.335 - 2.5 * np.log10(1000.0), abs=1e-6)


def test_trajectory_mag_is_none_without_a_zero_point():
    trj = build_trajectory(make_track(), FakeSeq(), SCALE)
    assert all(p.mag is None for p in trj.points)


# --------------------------------------------------------------------------
# MVP 边界：三个 None 字段
# --------------------------------------------------------------------------


def test_pa_and_radec_are_none_in_mvp():
    """MVP 无绝对定向来源，三个字段必须报 ``None``。

    dataset A 首帧 FITS 头 28 行卡片里没有赤经赤纬、没有焦距、没有像元尺寸、
    没有 WCS，``ra/dec`` 需要盲板求解、``pa_deg`` 需要视场旋转与翻转标定。
    实测像素系轨迹方向 ``atan2(ox, oy)`` 均值 95.6670°（std 0.7767）而真值位置角
    （自北经东）均值 321.9317°（std 1.4255），相差 226.2648°——那 226° 整个落在
    未解的翻转+旋转里。这条测试防的是将来有人塞一个像素系方向角进 ``pa_deg``。
    """
    trj = build_trajectory(make_track(), FakeSeq(), SCALE)
    for p in trj.points:
        assert p.ra_deg is None
        assert p.dec_deg is None
        assert p.pa_deg is None


def test_to_records_is_json_safe_with_builtin_scalars():
    """所有数值字段必须是内建 ``float``/``int``，``None`` 原样保留。

    用 ``type(x) is float`` 而不是 ``isinstance``：``np.float64`` 是 ``float`` 的
    **子类**会混过去，而 ``np.float32`` / ``np.int64`` 实测都抛
    ``TypeError: Object of type float32 is not JSON serializable``。
    """
    zp = ZeroPoint(15.335, 0.218, 54, "truth", exposure_s=0.030)
    records = build_trajectory(make_track(), FakeSeq(), SCALE, zero_point=zp).to_records()
    assert len(records) == 20

    text = json.dumps(records, allow_nan=False, ensure_ascii=False)
    assert len(json.loads(text)) == 20

    r = records[0]
    assert type(r["frame"]) is int
    assert type(r["time_utc"]) is str
    for key in ("time_unix", "x_sky", "y_sky", "x_det", "y_det", "rate_arcsec_s", "flux", "mag"):
        assert type(r[key]) is float, f"{key} 不是内建 float: {type(r[key])}"
    for key in ("ra_deg", "dec_deg", "pa_deg"):
        assert r[key] is None


def test_to_records_coerces_non_finite_rate_to_none():
    """裁决 50 同类隐患：裸 ``NaN`` token 不是合法 JSON，严格解析器会拒收。"""
    records = build_trajectory(make_track(n=3), FakeSeq(n=3, cadence_s=0.0), SCALE).to_records()
    assert all(r["rate_arcsec_s"] is None for r in records)
    text = json.dumps(records, allow_nan=False, ensure_ascii=False)
    assert "NaN" not in text


# --------------------------------------------------------------------------
# 严格 TAN 反投影（本轮不被 build_trajectory 调用，纯函数）
# --------------------------------------------------------------------------


def test_offset_to_radec_is_gnomonic_not_linear():
    """线性近似 ``ra0 + Δx/cos(dec0)`` 在本项目真实跨度上实测差 1189.4823″ = 192.50 px。

    偏移量取自真值首末两行经严格 TAN 正投影得到的切平面坐标
    （切点 = 真值第 0 行 (209.23627, 12.35454)，端点大圆跨度 11.646240°）：
    ``(-25614.6558, +33930.7378)`` 角秒。视场本身 4096 px × 6.179 ″/px =
    7.0303 度 = 421.82 角分——注意不是「7 角分」，那个说法错 60 倍。
    """
    ra0, dec0 = 209.0, 12.0
    off_x, off_y = -25614.6558, 33930.7378
    ra_tan, dec_tan = offset_to_radec(ra0, dec0, off_x, off_y)
    ra_lin = ra0 + (off_x / 3600.0) / np.cos(np.deg2rad(dec0))
    dec_lin = dec0 + off_y / 3600.0

    a1, d1, a2, d2 = map(np.deg2rad, (ra_tan, dec_tan, ra_lin, dec_lin))
    sep_arcsec = np.degrees(
        2.0 * np.arcsin(
            np.sqrt(np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((a2 - a1) / 2) ** 2)
        )
    ) * 3600.0
    assert sep_arcsec > 1000.0
    assert sep_arcsec / 6.179 > 150.0        # px 量级，实测 192.50 px


def test_offset_to_radec_origin_round_trips():
    """零偏移回到切点，但**不是逐位精确**。

    实测 ``offset_to_radec(209, 12, 0, 0)`` = ``(209.000000000000000,
    12.000000000000002)``——``dec`` 差 2 ulp，来自 ``arctan`` / ``hypot`` 的舍入。
    ``abs=1e-12`` 有约 500× 余量。名字叫 ``..._round_trips`` 而不是
    ``..._is_exact``，因为它确实不精确。
    """
    ra, dec = offset_to_radec(209.0, 12.0, 0.0, 0.0)
    assert ra == pytest.approx(209.0, abs=1e-12)
    assert dec == pytest.approx(12.0, abs=1e-12)
    assert float(dec) != 12.0        # 2 ulp，不是逐位相等


def test_offset_to_radec_wraps_right_ascension_into_zero_360():
    """切点贴近 0h 时赤经必须落在 [0, 360)，不得给出负值。"""
    ra, _ = offset_to_radec(0.5, 12.0, -7200.0, 0.0)
    assert 0.0 <= float(ra) < 360.0
    assert float(ra) > 350.0


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------


def test_config_carries_the_frame_centre(cfg):
    """``geometry.center_px``：4096 px 画面中心，0-based，x=列 y=行。

    本轮 ``build_trajectory`` 不用它（无 anchor），但它是 Task 26 出图与国赛
    Task 18 盲板求解的共用常量，模块体内不留裸数字。
    """
    assert cfg["geometry"]["center_px"] == [2047.5, 2047.5]


# --------------------------------------------------------------------------
# 数据集测试（裁决 397）
# --------------------------------------------------------------------------


def test_dataset_b_rate_matches_truth(dataset_b_dir):
    """真值比对：均匀比例尺角速度 770.15 ″/s，与配上的 53 对真值 770.02 相对差 1.64e-4。

    **rel=0.002 是实测定的，不是随手取的。** 控制器实测的杀伤表（基准 = 配上那
    54 行真值的 53 个相邻对均值 770.0244 ″/s，大圆定义）：

    | 缺陷 | 偏移后 mean | 相对 | rel=0.002 |
    |---|---|---|---|
    | 正确实现 | 770.1507 | 0.000164 | 通过（余量 12.2×） |
    | 丢掉 dy 分量 | 766.3233 | 0.004806 | **拒** |
    | 丢掉 dx 分量 | 75.9733 | 0.901337 | **拒** |
    | 用 xy_det 代 xy_sky | 11.3302 | 0.985286 | **拒** |
    | 比例尺错 1% | 777.8523 | 0.010166 | **拒** |
    | 比例尺错 0.5% | 774.0015 | 0.005165 | **拒** |
    | 比例尺错 0.2% | 771.6910 | 0.002164 | **拒** |
    | 节拍硬编码 1.008 | 770.7406 | 0.000930 | 通过 |
    | 帧号错 ±1 | 770.1506 / 770.1933 | 0.000164 / 0.000219 | 通过 |

    **这条测试抓不到**：节拍硬编码（偏 0.72 ″/s）、``times()[frames]`` 错 1 帧
    （偏 0.13–0.17 ″/s）、``t.unix`` 与相对秒之别（真实数据上逐位相同）。
    比例尺本身由真值定标定出，不在这里守。

    **两个 54 不是同一个 54（裁决 400）。** 真值文件共 **55 行**；配上轨迹的是
    **54 行**，相邻对数 **53**，其均值 **770.0244 ″/s**。全 55 行 54 对的均值是
    **769.7526 ″/s**，两者差 0.272 ″/s。本测试与「配上的那 54 行」比。

    ``rate_std < 0.02 * mean`` 的依据：实测真实 ``xy_sky`` 基线 ``std/mean`` =
    0.007672，叠加质心噪声 400 次后 σ=0.5 px 档的 max 为 0.010931（余量 1.8×），
    而实测质心抖动最差约 0.18 px。
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
    assert trk.length == 54 and trk.start == 17 and trk.end == 70

    truth = load_truth(seq.truth_path)
    assert len(truth) == 55                      # 全表 55 行
    mf, mr = match_frames(truth, seq, trk.frames)
    assert len(mr) == 54                         # 配上轨迹的 54 行
    assert trk.frames[0] == mf[0]                # 帧对齐，防真值行错配

    ra, dec = truth.ra_deg[mr], truth.dec_deg[mr]
    t = truth.time[mr]
    a1, d1, a2, d2 = map(np.deg2rad, (ra[:-1], dec[:-1], ra[1:], dec[1:]))
    step_arcsec = np.degrees(
        2.0 * np.arcsin(
            np.sqrt(np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((a2 - a1) / 2) ** 2)
        )
    ) * 3600.0
    truth_rate = step_arcsec / np.diff((t - t[0]).sec)
    assert len(truth_rate) == 53                 # 53 对
    truth_rate_matched = float(truth_rate.mean())
    assert truth_rate_matched == pytest.approx(770.0244, abs=1e-3)

    trj = build_trajectory(trk, seq, SCALE)
    assert len(trj) == 54
    assert trj.source == seq.dataset_id
    assert trj.mean_rate_arcsec_s == pytest.approx(truth_rate_matched, rel=0.002)
    assert trj.rate_std_arcsec_s < 0.02 * trj.mean_rate_arcsec_s
    # 全 55 行 54 对的 769.7526 不是本测试的基准；差 0.272 ″/s。
    assert abs(truth_rate_matched - 769.7526) == pytest.approx(0.272, abs=1e-3)

    json.dumps(trj.to_records(), allow_nan=False, ensure_ascii=False)
