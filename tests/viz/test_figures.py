"""8 张成果图的测试。

**本模块最重要的一条是字体守卫**：只检查「字体名非空且在 ttflist 里」会放过
``DejaVu Sans``——它两条都过，而它对本项目实际要画的汉字一个字形都没有，
于是每张图的每个汉字都是空心方框而全部断言绿。所以守卫必须是**字形级**的
（``FT2Font.get_char_index(ord(ch)) != 0``），且检查的字符集必须是这 8 张图
**实际会画的那一套**（``FIGURE_CJK``），不是一个抽象样本。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.viz.figures import (
    FIGURE_CJK,
    FontUnavailable,
    glyphs_missing,
    plot_detections,
    plot_dual_frame_verdict,
    plot_height_interval,
    plot_hotpixels,
    plot_magnitude_histogram,
    plot_registration_residuals,
    plot_threshold_curve,
    plot_trajectory,
    setup_matplotlib,
)

# ---------------------------------------------------------------------------
# 尺寸下界。**上界侧的配方**（容器内，matplotlib 3.10.9，stock rcParams、
# 未调用 setup_matplotlib、Agg、`savefig(dpi=150)`，见 probe/t26-r451-fix.txt）：
#
#     fig = plt.figure(figsize=FS)                    # 纯空白
#     fig, ax = plt.subplots(figsize=FS)              # 空坐标轴
#
#   figsize       纯空白   空坐标轴
#  10.0x8.0         9930     18940     <- 本模块用
#   9.0x6.0         6927     15820     <- 本模块用
#  12.0x6.0         8502     17523     <- 本模块用
#  12.0x8.0        11288     20396        本模块未用，取它是为了留余量
#
# 本模块实际只用 10.0x8.0 / 9.0x6.0 / 12.0x6.0 三档（`grep figsize figures.py`），
# 空坐标轴上界因此是 18940；`MAX_EMPTY_AXES_BYTES` 仍取 20396 是为了留一档余量。
#
# **下界侧不能用合成的「薄真图」锚**（R451）：所谓「薄真图」不是一个良定义的量，
# 它随画法连续变化，且能低到空坐标轴以下 —— 实测同为「真图」的
# `ax.plot([1],[1.0],marker="o")` 在 9.0x6.0 只有 **17838** 字节，比 20396 还小；
# 而 `xlabel` 从 `"x"` 换成 `"frame"` 一处就让 9.0x6.0 从 41669 变 42567（差 898）。
# 所以这里改锚到**本模块真正画出来的图**：跑一遍本文件的 39 条测试、
# 统计它写出的全部 82 张 PNG（probe/t26-real-figure-sizes.txt），
# 最薄的一张是 **61657** 字节，即 `plot_magnitude_histogram` 的
# 全 NaN 退化分支（见 test_magnitude_histogram_with_all_nan）；
# 最大 309008（test_detections）。
#
# 于是可分离窗口是 **(20396, 61657)**，取 **24000**。注意 24000 离下界只有
# 3600 的余量、离上界有 37657 —— 见下面那条测试的 docstring：这个不对称是
# 有意的，也正是 R449 说它判别力为零的原因。
#
# 这条下界只挡「一张图基本没画东西」。它**挡不住**「画了中文标签但没画数据」的
# 退化图：实测一张空坐标轴只要带上中文标题与轴标签就有 46983~59775 字节，
# 远在任何合理下界之上。真正的防线是各图的 probe 口径断言。
MIN_REAL_FIGURE_BYTES = 24000   # 从 default.yaml 的 viz.min_real_figure_bytes 读
MAX_EMPTY_AXES_BYTES = 20396    # 12.0x8.0 空坐标轴（本模块三档的上界是 18940）
MIN_THIN_REAL_BYTES = 61657     # 本模块 82 张 PNG 里最薄的一张（全 NaN 星等直方图）
# ---------------------------------------------------------------------------


def test_the_byte_floor_actually_separates_blank_from_real():
    """把上面那张表的载荷结论固化：下界必须**高于**任何空坐标轴图，
    否则一张画空了的图也能过 `.st_size >` 断言。16000 过不了这一条
    （10x8 的空坐标轴图就有 18940 字节）。

    **这条断言的判别力是零，登记在案（R449）**：本模块最薄的真图 61657 字节
    是下界的 2.57 倍，所以 20396 到 61657 之间**任何**取值都不会改变
    本文件任何一条判决 —— 实测把 8 处 `st_size >` 全改成 5000 仍然 39 条全过。
    留着它的理由只有一个：它挡的那个失败模式（`.exists()` 过了但图是空的）
    在图 6 的未收敛分支上真实发生过。**不要把它当测试，它是烟雾报警器。**
    """
    from src.config import load_config

    assert load_config()["viz"]["min_real_figure_bytes"] == MIN_REAL_FIGURE_BYTES
    assert MIN_REAL_FIGURE_BYTES > MAX_EMPTY_AXES_BYTES, "下界必须高于任何空坐标轴图"
    assert MIN_REAL_FIGURE_BYTES > 18940, "下界必须高于 10x8 的空坐标轴图"
    assert MIN_REAL_FIGURE_BYTES < MIN_THIN_REAL_BYTES, "下界必须低于最薄的真图"


# ===== 字体：本任务最重要的四条 =====

def test_selected_font_covers_every_character_the_figures_draw():
    """**本任务最重要的测试。**

    断言的是「选中字体覆盖本项目实际要画的字符集」，不是「字体名非空」。
    只查名字的旧式守卫实测会放过 DejaVu Sans，而它缺失 FIGURE_CJK 的全部汉字。
    字形索引 0 = 该字符在该字体里不存在，这是唯一可靠的判据。
    """
    name = setup_matplotlib()
    assert isinstance(name, str) and name
    missing = glyphs_missing(name, FIGURE_CJK)
    assert missing == [], f"{name} 缺失字形：{''.join(missing)}"


def test_the_name_only_check_would_have_passed_dejavu():
    """把「旧守卫为什么不够」固化成断言，防止有人把守卫改回去。

    只查汉字子集：`FIGURE_CJK` 里的 `±σ″` 三个符号 DejaVu Sans 实测**有**字形
    （正是这一点让名字检查看起来「没问题」），断言「全部字形缺失」会假红。
    载荷是**汉字一个都没有**。
    """
    from matplotlib import font_manager as fm

    han = "".join(ch for ch in FIGURE_CJK if ord(ch) > 0x2FFF)
    assert len(han) >= 60
    assert "DejaVu Sans" in {f.name for f in fm.fontManager.ttflist}   # 名字检查通过
    assert len(glyphs_missing("DejaVu Sans", han)) == len(han)         # 汉字字形全灭


def test_dejavu_sans_is_not_in_the_candidate_list():
    """DejaVu Sans 必须从候选表里删掉。它在表里的唯一作用是让选择逻辑
    「成功」选中一个画不出汉字的字体。回落由 matplotlib 兜底。
    """
    from src.config import load_config

    assert "DejaVu Sans" not in load_config()["viz"]["font_candidates"]


def test_setup_matplotlib_raises_when_no_candidate_has_glyphs():
    """候选全不可用时必须抛 FontUnavailable，中文消息，且列出尝试过的
    每个名字及其被拒原因（否则排查要靠猜）。
    """
    with pytest.raises(FontUnavailable) as exc:
        setup_matplotlib(candidates=("DejaVu Sans", "No Such Font 12345"))
    msg = str(exc.value)
    assert any("一" <= ch <= "鿿" for ch in msg)
    assert "DejaVu Sans" in msg and "No Such Font 12345" in msg


def test_minus_sign_is_configured_away():
    """实测 SimHei 的 U+2212 MINUS SIGN 字形索引为 **0**（ASCII 连字符是 16）。
    负刻度默认用 U+2212，所以 `axes.unicode_minus = False` 是载荷代码，
    不是风格偏好。
    """
    import matplotlib.pyplot as plt

    setup_matplotlib()
    assert plt.rcParams["axes.unicode_minus"] is False
    assert glyphs_missing(plt.rcParams["font.sans-serif"][0], "-") == []


def test_superscript_two_is_never_drawn():
    """实测 `²` U+00B2 在 SimHei 里字形索引也是 **0**。而天光面亮度的单位就是
    mag/arcsec²，第 8 张图的标签必然用到它，没有第二条路绕开。

    所以面积单位一律写 ASCII 的 `mag/arcsec^2`，`FIGURE_CJK` 里不得出现 `²`，
    且选中字体必须覆盖 `^` 与 `2`。
    """
    name = setup_matplotlib()
    assert "²" not in FIGURE_CJK, "面积单位不得写 arcsec²，改 arcsec^2"
    assert glyphs_missing(name, "mag/arcsec^2") == []


def test_arcsec_and_greek_glyphs_exist():
    """轴标签用 ″(U+2033)、σ、Δ、±。SimHei 下实测索引 269/183/145/102，
    都存在——但换字体后可能不存在，所以钉住。
    """
    assert glyphs_missing(setup_matplotlib(), "″σΔ±") == []


def test_all_figures_use_no_gui_backend():
    import matplotlib

    setup_matplotlib()
    assert matplotlib.get_backend().lower() == "agg"


# ===== 图 1：探测源叠加 =====

def _table(n, seed, size=256):
    from src.detect.segmentation import SourceTable

    rng = np.random.default_rng(seed)
    return SourceTable(
        frame=30,
        x=rng.uniform(10, size - 10, n), y=rng.uniform(10, size - 10, n),
        flux=rng.uniform(200, 5000, n), peak=rng.uniform(50, 900, n),
        elongation=np.full(n, 1.20), npix=np.full(n, 9, dtype=np.int64),
    )


def test_detections(tmp_path):
    from src.calib.hotpixel import HotCluster

    img = np.random.default_rng(4).normal(6.45, 3.8, size=(256, 256))
    hot = [HotCluster(cx=100.0, cy=100.0, npix=4, median_value=1200.0, flat_ratio=0.01)]
    p = plot_detections(img, _table(30, 4), tmp_path / "det.png",
                        target_xy=(128.0, 128.0), hot_clusters=hot)
    assert p.exists() and p.stat().st_size > MIN_REAL_FIGURE_BYTES


def test_detections_without_optional_overlays(tmp_path):
    """`target_xy=None` 且 `hot_clusters=None` 是常态（未锁定目标的帧）。"""
    img = np.random.default_rng(5).normal(6.45, 3.8, size=(128, 128))
    assert plot_detections(img, _table(8, 5, 128), tmp_path / "d2.png").exists()


def test_detections_with_an_empty_table(tmp_path):
    """空源表必须画出图（图上写明 0 个），不能崩。
    `np.percentile` 在空数组上抛 IndexError，`len(table)==0` 的图例也要成立。
    """
    from src.detect.segmentation import SourceTable

    img = np.random.default_rng(6).normal(6.45, 3.8, size=(64, 64))
    t = SourceTable(frame=3, x=np.zeros(0), y=np.zeros(0), flux=np.zeros(0),
                    peak=np.zeros(0), elongation=np.zeros(0),
                    npix=np.zeros(0, dtype=np.int64))
    info = plot_detections(img, t, tmp_path / "d3.png", probe=True)
    assert info["path"].exists() and info["n_sources"] == 0
    assert any("0 个" in text for text in info["all_labels"]), "空源表必须在图上写明 0 个"


# ===== 图 2：可复现性曲线 =====

def _curve():
    """`RepeatabilityPoint` 实测有 **6** 个字段（`n_sigma`、`n_detected_mean`、
    `n_reproducible`、`n_reference`、`reproducibility`、`purity`），全部无默认值。
    `reproducibility` 与 `purity` 分子相同、分母不同，夹具不让两者相等。
    """
    from src.validate.repeatability import RepeatabilityCurve, RepeatabilityPoint

    rows = [(2.0, 2284.0, 180, 2284, 0.0788, 0.90),
            (3.0, 169.0, 140, 169, 0.8284, 0.93),
            (4.0, 116.0, 104, 116, 0.8966, 0.95),
            (5.0, 95.0, 86, 95, 0.9053, 0.96)]
    pts = [RepeatabilityPoint(n_sigma=s, n_detected_mean=d, n_reproducible=r,
                              n_reference=ref, reproducibility=rep, purity=pur)
           for s, d, r, ref, rep, pur in rows]
    return RepeatabilityCurve(points=pts, frames=[16, 20, 24, 28, 32, 36],
                              match_radius_px=3.0)


def test_threshold_curve(tmp_path):
    p = plot_threshold_curve(_curve(), tmp_path / "curve.png",
                             title="数据集 B 阈值-复现率")
    assert p.exists() and p.stat().st_size > MIN_REAL_FIGURE_BYTES


def test_threshold_curve_with_a_single_point(tmp_path):
    """一个点的曲线（只扫了一个 sigma）必须画出图，不能因为 `np.diff` 或
    `set_xlim(min, max)` 退化而崩。该点复现率 0.0788 低于门限，
    `recommended()` 会抛 ValueError，此时图上写明「无达标档位」。
    """
    c = _curve()
    one = type(c)(points=c.points[:1], frames=c.frames,
                  match_radius_px=c.match_radius_px)
    info = plot_threshold_curve(one, tmp_path / "c1.png", probe=True)
    assert info["path"].exists() and info["marked_sigma"] is None
    assert any("无达标档位" in text for text in info["all_labels"])


def test_threshold_curve_marks_the_recommended_threshold(tmp_path):
    """曲线图必须标出 `curve.recommended()` 选中的那个 sigma——
    「约 10² 颗」这个结论的全部依据就是这一个点，图上不标等于图没讲完。
    """
    c = _curve()
    info = plot_threshold_curve(c, tmp_path / "c2.png", probe=True)
    assert info["marked_sigma"] == pytest.approx(c.recommended().n_sigma)


# ===== 图 3：配准残差分布 =====

def _registration(n_pairs=53, rms=(0.31, 0.42)):
    """`RegistrationResult` 实测字段：`reference`、`frames`、`matrices`、`pairs`；
    `PairSolution` 是 7 个无默认值字段。`report()` 给 `rms_max_px` /
    `rms_median_px` / `inliers_min|max|median`。
    """
    from src.register.solver import PairSolution, RegistrationResult

    rng = np.random.default_rng(12)
    pairs, mats = [], {}
    for k in range(n_pairs):
        f = 16 + k
        pairs.append(PairSolution(
            frame_from=f, frame_to=f + 1, matrix=np.eye(3),
            n_inliers=int(rng.integers(80, 140)),
            rms_px=float(rng.uniform(*rms)),
            rotation_deg=float(rng.normal(0.0, 0.02)),
            shift_px=(float(rng.normal(-80.0, 1.0)), float(rng.normal(0.3, 0.1))),
        ))
        mats[f] = np.eye(3)
    mats[16 + n_pairs] = np.eye(3)
    return RegistrationResult(reference=16, frames=sorted(mats),
                              matrices=mats, pairs=pairs)


def test_registration_residuals(tmp_path):
    p = plot_registration_residuals(_registration(), tmp_path / "reg.png")
    assert p.exists() and p.stat().st_size > MIN_REAL_FIGURE_BYTES


def test_registration_residuals_shows_both_rms_and_inliers(tmp_path):
    """残差分布图必须同时给残差与内点数：单看 RMS 小说明不了配准好——
    内点数塌到 3 个时 RMS 必然很小。两个量一起看才是证据。
    """
    info = plot_registration_residuals(_registration(), tmp_path / "r2.png", probe=True)
    assert info["n_axes"] >= 2
    assert info["rms_median_px"] == pytest.approx(
        _registration().report()["rms_median_px"], rel=1e-12)
    text = " ".join(info["all_labels"])
    assert "内点" in text, "只画残差不画内点等于把 R402 那条论证画没了"


def test_registration_residuals_with_no_pairs(tmp_path):
    """0 对（单帧数据集）时 `report()` 的 rms 是 nan、inliers 是 0。
    必须画出一张写明「无配准对」的图，不能崩。
    """
    from src.register.solver import RegistrationResult

    res = RegistrationResult(reference=16, frames=[16],
                             matrices={16: np.eye(3)}, pairs=[])
    info = plot_registration_residuals(res, tmp_path / "r0.png", probe=True)
    assert info["path"].exists() and info["n_pairs"] == 0
    assert any("无配准对" in text for text in info["all_labels"])


# ===== 图 4：双坐标系目标判决（本项目主亮点）=====

def _verdict(n=20, v_det=0.35, v_sky=125.7, is_target=True):
    """`Track` 字段：`frames`、`xy_det`、`xy_sky`、`flux`、`peak`、`elongation`、
    `lock_span_px=1.0`。点集统一 `(N,2)` float64 `[x, y]`。
    `classify` 实测**会写** `track.lock_span_px`，所以走 `classify` 造 Verdict
    比手搓字段更接近真实路径。
    """
    from src.target.dual_frame import Track, classify

    i = np.arange(n, dtype=np.float64)
    det = np.column_stack([2271.6 + v_det * i, 1958.8 + 0.1 * i])
    sky = np.column_stack([2271.6 + v_sky * i, 1958.8 - 5.0 * i])
    trk = Track(frames=[17 + int(k) for k in i], xy_det=det, xy_sky=sky,
                flux=np.full(n, 1000.0), peak=np.full(n, 480.0),
                elongation=np.full(n, 1.20))
    return classify(trk) if is_target else classify(trk, v_sky_min=1e9)


def test_dual_frame_verdict(tmp_path):
    p = plot_dual_frame_verdict(_verdict(), tmp_path / "dual.png")
    assert p.exists() and p.stat().st_size > MIN_REAL_FIGURE_BYTES


def test_dual_frame_verdict_draws_both_frames_not_just_one(tmp_path):
    """**这张图的全部意义是两条轨迹并列**：恒星系里目标在动、机架系里恒星不动。
    只画一条等于把主亮点画没了，而 `.exists()` 对此完全免疫。
    所以查两条线的数据范围：det 的位移必须远小于 sky 的。
    """
    v = _verdict()
    info = plot_dual_frame_verdict(v, tmp_path / "d2.png", probe=True)
    assert info["n_tracks_drawn"] == 2
    assert info["drift_det_px"] == pytest.approx(v.track.drift_det_px, rel=1e-12)
    assert info["drift_sky_px"] == pytest.approx(v.track.drift_sky_px, rel=1e-12)
    assert info["drift_sky_px"] > 100.0 * info["drift_det_px"]


def test_dual_frame_verdict_prints_the_contrast_and_the_reasons(tmp_path):
    """判决图必须写出对比度与判决理由。`Verdict.reasons` 在 is_target=False
    时非空（中文），图上不写理由，答辩现场就没法解释为什么判否。
    """
    info = plot_dual_frame_verdict(_verdict(is_target=False), tmp_path / "d3.png",
                                   probe=True)
    assert info["is_target"] is False
    assert info["n_reasons"] >= 1
    text = " ".join(info["all_labels"])
    assert "对比度" in text
    for reason in _verdict(is_target=False).reasons:
        assert reason in text, "判决理由必须逐条画在图上"


def test_dual_frame_verdict_survives_infinite_contrast(tmp_path):
    """`Verdict.to_dict()` 的 docstring 点明：v_det 恰为 0 时 contrast 是 `inf`。
    图上格式化 `inf` 会写出「inf」，可接受；但 `set_xlim` 之类吃到 inf 会崩。

    注意 `v_det_px` 走 `_step_median`，是 **2-D 步长**的中位数：
    `_verdict(v_det=0.0)` 的 y 仍以 0.1 px/帧 变化，步长中位数是 0.1、不是 0。
    要造出 `inf` 必须让探测器系**完全静止**（x、y 都不动），
    而那同时会触发 `pixel_locked`，于是 `is_target=False`。
    这两件事一起发生是实现的真实行为，测试按这个写。
    """
    from src.target.dual_frame import Track, classify

    n = 20
    i = np.arange(n, dtype=np.float64)
    det = np.column_stack([np.full(n, 2271.6), np.full(n, 1958.8)])   # 完全静止
    sky = np.column_stack([2271.6 + 125.7 * i, 1958.8 - 5.0 * i])
    v = classify(Track(frames=[17 + int(k) for k in i], xy_det=det, xy_sky=sky,
                       flux=np.full(n, 1000.0), peak=np.full(n, 480.0),
                       elongation=np.full(n, 1.20)))
    assert not np.isfinite(v.contrast) and v.pixel_locked and v.is_target is False
    assert plot_dual_frame_verdict(v, tmp_path / "d4.png").exists()


# ===== 图 5：目标轨迹 + 角速度 =====

def _trajectory(n=20, scale=6.179):
    """`TrajectoryPoint` 的 `time_unix` 无默认值；`scale_arcsec_px` 必须走
    **构造参数**——构造后赋值会让 `total_arc_arcsec` 悄悄按 1.0 ″/px 算。
    `ra_deg`/`dec_deg`/`pa_deg` 在 MVP 一律 `None`，**图上不得出现赤经赤纬
    与位置角**。

    时刻用 `Time + TimeDelta` 递推，不用 `"...:%09.6f" % (44.0 + i)` 拼字符串：
    后者在 i>=16 时写出 60.000000~63.000000 秒，erfa 会报
    `"dtf2d" yielded 1 of "time is after end of day"` 并把 17:26:63 解释成
    17:27:03。测试不该带着可消除的警告跑。
    """
    from astropy.time import Time, TimeDelta

    from src.target.trajectory import Trajectory, TrajectoryPoint

    t0 = Time("2026-07-21T17:26:44.000000", format="isot", scale="utc")
    pts = []
    for i in range(n):
        t = t0 + TimeDelta(float(i), format="sec")
        pts.append(TrajectoryPoint(
            frame=17 + i, time_isot=t.isot,
            time_unix=float(t.unix),
            x_det=2271.6 + 0.35 * i, y_det=1958.8 + 0.1 * i,
            x_sky=2271.6 + 125.73 * i, y_sky=1958.8 - 5.0 * i,
            ra_deg=None, dec_deg=None,
            rate_arcsec_s=770.15 + 0.3 * ((-1) ** i), pa_deg=None,
            flux=1000.0, mag=8.4,
        ))
    return Trajectory(points=pts, source="test", scale_arcsec_px=scale)


def test_trajectory(tmp_path):
    p = plot_trajectory(_trajectory(), tmp_path / "trj.png")
    assert p.exists() and p.stat().st_size > MIN_REAL_FIGURE_BYTES


def test_trajectory_never_labels_an_axis_ra_or_dec(tmp_path):
    """口径守卫：图 5 的位置轴是**恒星系像素**，不是赤经赤纬；MVP 不报位置角。
    夹具里 ra/dec/pa 全是 None，任何把它们画上去的实现都会先崩或画出 None。
    这里直接断言标签文字里没有这三个词。
    """
    info = plot_trajectory(_trajectory(), tmp_path / "t2.png", probe=True)
    text = " ".join(info["all_labels"])
    for banned in ("赤经", "赤纬", "位置角", "RA", "Dec"):
        assert banned not in text, f"图 5 不得出现「{banned}」"
    assert "像素" in text or "px" in text


def test_trajectory_rate_uses_the_n_minus_one_convention(tmp_path):
    """`Trajectory.mean_rate_arcsec_s` 实测走 `_independent_rates`，
    **排除 points[0]**（53 对 vs 54 点差 0.26 ″/s）。图上标的均值必须与那个
    属性一致，不能自己按 54 点重算。
    """
    trj = _trajectory()
    info = plot_trajectory(trj, tmp_path / "t3.png", probe=True)
    assert info["mean_rate_arcsec_s"] == pytest.approx(
        trj.mean_rate_arcsec_s, rel=1e-12)
    assert len(trj) == 20
    assert len(trj._independent_rates) == 19      # n-1，不是 n


# ===== 图 6：高度反演区间 =====

def _orbit(hmin=213.44, hmax=848.36):
    """`OrbitEstimate` 实测：`rate_arcsec_s`、`elevation_deg`、`bracket_km`、
    `converged` 四个无默认值，其余全部 `None` 默认。`height_km` 是**区间中点**，
    只作展示。
    """
    from src.analysis.orbit import OrbitEstimate

    return OrbitEstimate(
        rate_arcsec_s=770.15, elevation_deg=41.7,
        bracket_km=(200.0, 40000.0), converged=True,
        height_min_km=hmin, height_max_km=hmax,
        height_km=0.5 * (hmin + hmax),
        period_min_min=88.7, period_max_min=101.9,
        note="自相容区间，未与公开根数比对",
    )


def test_height_interval(tmp_path):
    p = plot_height_interval(_orbit(), tmp_path / "h.png")
    assert p.exists() and p.stat().st_size > MIN_REAL_FIGURE_BYTES


def test_height_interval_draws_the_band_not_a_single_value(tmp_path):
    """**口径守卫**：高度只报自相容区间 `[213.44, 848.36] km`。把中点 530.9
    画成一个「测得的高度」是答辩现场的原理性质疑。所以图上必须有带宽，
    且标注的两个端点必须逐字等于 `height_min_km` / `height_max_km`。
    """
    est = _orbit()
    info = plot_height_interval(est, tmp_path / "h2.png", probe=True)
    assert info["band_low_km"] == pytest.approx(213.44, abs=1e-9)
    assert info["band_high_km"] == pytest.approx(848.36, abs=1e-9)
    assert info["band_width_km"] == pytest.approx(est.band_width_km, rel=1e-12)
    text = " ".join(info["all_labels"])
    assert "213.44" in text and "848.36" in text, "两个端点必须逐字标在图上"
    assert "区间" in text


def test_height_interval_never_claims_tle_agreement(tmp_path):
    """**口径守卫**：TLE 四条测试是 SKIPPED，真实根数没抄进
    `docs/reference/tle.txt`。图上不得出现「吻合」「TLE」「两行根数」这类字样。
    """
    info = plot_height_interval(_orbit(), tmp_path / "h3.png", probe=True)
    text = " ".join(info["all_labels"])
    for banned in ("TLE", "两行根数", "吻合", "公开根数"):
        assert banned not in text, f"图 6 不得出现「{banned}」"


def test_height_interval_when_inversion_did_not_converge(tmp_path):
    """`converged=False` 时 height_min/max 都是 `None`。必须画一张写明
    「未收敛」的图，而不是 TypeError。
    """
    from src.analysis.orbit import OrbitEstimate

    est = OrbitEstimate(rate_arcsec_s=12.0, elevation_deg=41.7,
                        bracket_km=(200.0, 40000.0), converged=False,
                        note="角速度低于圆轨道下界，无解")
    assert est.band_width_km is None
    info = plot_height_interval(est, tmp_path / "h4.png", probe=True)
    assert info["path"].exists() and info["band_width_km"] is None
    assert any("未收敛" in text for text in info["all_labels"])


# ===== 图 7：热像素 / 传感器缺陷 =====

def _hotmap(xy=((244, 3174), (312, 3206), (327, 3210), (2192, 3223))):
    """`plot_hotpixels` 只读 `.clusters`，从不碰 `.mask`（4096² bool 掩膜实测
    16.0 MiB），所以用 1×1 占位。
    """
    from src.calib.hotpixel import HotCluster, HotPixelMap

    cl = [HotCluster(cx=float(x), cy=float(y), npix=4,
                     median_value=1200.0, flat_ratio=0.01) for x, y in xy]
    return HotPixelMap(mask=np.zeros((1, 1), dtype=bool), clusters=cl)


def _sensor_report():
    """`SensorReport` 实测字段（Task 22 交付）。
    交付口径数字：4 个缺陷簇 + 每历元 1 个第 0 行伪影。
    """
    from src.analysis.sensor_health import SensorReport

    return SensorReport(
        dataset_id="A", epoch_isot="2026-03-09T14:37:11.741400",
        n_clusters=4,
        clusters=[{"cx": 244.1, "cy": 3173.6, "npix": 4,
                   "median_value": 304.0, "flat_ratio": 0.01}],
        background_median=5.0, background_rms=3.5269, n_saturated=0,
        fwhm_median_px=3.52, n_metadata_artifacts=1, metadata_rows=1,
        background_frame_index=0, background_exposure_ms=40.0,
    )


def test_hotpixels(tmp_path):
    p = plot_hotpixels(_hotmap(), tmp_path / "hot.png", shape=(4096, 4096))
    assert p.exists() and p.stat().st_size > MIN_REAL_FIGURE_BYTES


def test_hotpixels_with_no_clusters(tmp_path):
    """干净探测器（0 簇）必须画出图。实现里按像素数算 marker 尺寸的那一步
    在空列表上会让 scatter 收到空 sizes。
    """
    info = plot_hotpixels(_hotmap(xy=()), tmp_path / "h0.png", shape=(4096, 4096),
                          probe=True)
    assert info["path"].exists() and info["n_clusters"] == 0
    assert any("0 个" in text for text in info["all_labels"])


def test_hotpixels_reports_the_metadata_exclusion_instead_of_hiding_it(tmp_path):
    """**口径守卫**：第 0 行伪影的剔除依据是**量级**，不是曝光响应，
    而且剔除数量必须如实报出、不隐藏（`n_metadata_artifacts`）。
    带 `report=` 时图上必须写出这个数。
    """
    rep = _sensor_report()
    info = plot_hotpixels(_hotmap(), tmp_path / "h5.png", shape=(4096, 4096),
                          report=rep, probe=True)
    assert info["n_metadata_artifacts"] == rep.n_metadata_artifacts == 1
    text = " ".join(info["all_labels"])
    assert "不随曝光变化" not in text, "不得用「不随曝光变化」作剔除理由"
    assert "量级" in text, "剔除依据必须写明是数值量级"


# ===== 图 8：视宁度 + 天光 + 极限星等 =====

def test_magnitude_histogram(tmp_path):
    mags = np.random.default_rng(9).normal(11.0, 1.5, 300)
    p = plot_magnitude_histogram(mags, tmp_path / "mag.png", limiting=13.8)
    assert p.exists() and p.stat().st_size > MIN_REAL_FIGURE_BYTES


def test_magnitude_histogram_with_all_nan(tmp_path):
    """无零点时全部星等是 nan。空数组喂给 `ax.hist` 必须不崩。"""
    assert plot_magnitude_histogram(np.full(50, np.nan), tmp_path / "m0.png").exists()


def test_magnitude_histogram_calls_the_limit_a_percentile_not_the_faintest(tmp_path):
    """**口径守卫**：极限星等是**经验暗端百分位**，不是「最暗可复现源」。
    图例文字必须体现百分位口径。
    """
    mags = np.random.default_rng(9).normal(11.0, 1.5, 300)
    info = plot_magnitude_histogram(mags, tmp_path / "m2.png",
                                    limiting=13.8, probe=True)
    text = " ".join(info["all_labels"])
    assert "百分位" in text
    assert "最暗可复现源" not in text


def test_magnitude_histogram_with_seeing_and_sky(tmp_path):
    """图 8 要三样东西并列。两个入口在 `src/analysis/sensor_health.py` 末尾
    （`SeeingEstimate` / `sky_brightness`）。

    **口径守卫**：天光面亮度与 `fwhm_arcsec` 都是**真值定标量**，只有 `fwhm_px`
    与 `elongation_median` 可声称独立；天光亮度只作**条件命题**、**不报小数**
    （整数 ADU 量化地板 0.18 mag，零点散度 0.22 mag）。

    **为什么夹具传的是 17.344337850113781 而不是 17.3**：传预先舍入过的 `17.3`
    再断言 `"17.3" in text` 方向是反的——它要求图上画出一位小数，而口径要的是
    **不画小数**；而且两位小数实现渲染出 `17.30`，`"17.3" in "17.30"` 为 True、
    `"17.34" in "17.30"` 为 False，两条断言全绿而实现违规。传原值后
    `%.1f` → `17.3`、`%.2f` → `17.34`，`"17.3" not in text` 对**任何**小数格式
    都红，只有整数格式绿。**不要把原值改回 17.3。**
    """
    from src.analysis.sensor_health import SeeingEstimate

    see = SeeingEstimate(n_stars=137, n_rejected=0, fwhm_px=3.52,
                         fwhm_arcsec=21.75, elongation_median=1.20,
                         note="真值定标的板比例 6.179 ″/px")
    mags = np.random.default_rng(9).normal(11.0, 1.5, 300)
    # 传**实测原值**（B 历元第 30 帧），不传预先舍入过的 17.3。
    info = plot_magnitude_histogram(mags, tmp_path / "m3.png", limiting=13.8,
                                    seeing=see, sky_mag_arcsec2=17.344337850113781,
                                    probe=True)
    text = " ".join(info["all_labels"])
    assert "arcsec^2" in text and "²" not in text
    # 天光亮度不报小数。整数量化地板 0.18 mag、零点散度 0.22 mag，
    # 小数位没有信息。下面三条各自独立：
    assert "17" in text                                  # 数字确实画出来了
    assert "17.3" not in text and "17.34" not in text     # 任何小数位都红
    assert "独立验证" not in text


# ===== 通用 =====

def test_every_figure_closes_its_figure(tmp_path):
    """`_save` 里的 `plt.close(fig)` 需要自己的测试。一次出 8 张图，
    漏 close 会撞上 matplotlib 的 `More than 20 figures` 警告并吃满内存。
    """
    import matplotlib.pyplot as plt

    plt.close("all")
    img = np.random.default_rng(11).normal(6.45, 3.8, size=(64, 64))
    t = _table(1, 11, 64)
    for i in range(25):
        plot_detections(img, t, tmp_path / f"c{i}.png")
    assert len(plt.get_fignums()) == 0, f"泄漏了 {len(plt.get_fignums())} 个 figure"


def _all_drawn_labels(tmp_path) -> list[str]:
    """把**八张图**（连同它们的退化分支）实际画出的每一处文字收集起来。

    这是 `test_figure_cjk_is_actually_what_the_figures_draw` 的强版本所需的
    输入：守卫查的字符集必须是画出来的那一套，否则查的是一套字、画的是另一套。
    """
    from src.analysis.orbit import OrbitEstimate
    from src.detect.segmentation import SourceTable
    from src.analysis.sensor_health import SeeingEstimate
    from src.register.solver import RegistrationResult

    labels: list[str] = []
    img = np.random.default_rng(21).normal(6.45, 3.8, size=(64, 64))

    labels += plot_detections(img, _table(6, 21, 64), tmp_path / "a1.png",
                              target_xy=(30.0, 30.0),
                              hot_clusters=list(_hotmap().clusters),
                              probe=True)["all_labels"]
    empty_table = SourceTable(frame=3, x=np.zeros(0), y=np.zeros(0),
                              flux=np.zeros(0), peak=np.zeros(0),
                              elongation=np.zeros(0), npix=np.zeros(0, dtype=np.int64))
    labels += plot_detections(img, empty_table, tmp_path / "a2.png",
                              probe=True)["all_labels"]

    curve = _curve()
    labels += plot_threshold_curve(curve, tmp_path / "a3.png",
                                   title="数据集 B 阈值-复现率曲线",
                                   probe=True)["all_labels"]
    labels += plot_threshold_curve(
        type(curve)(points=curve.points[:1], frames=curve.frames,
                    match_radius_px=curve.match_radius_px),
        tmp_path / "a4.png", probe=True)["all_labels"]

    labels += plot_registration_residuals(_registration(), tmp_path / "a5.png",
                                          probe=True)["all_labels"]
    labels += plot_registration_residuals(
        RegistrationResult(reference=16, frames=[16], matrices={16: np.eye(3)},
                           pairs=[]),
        tmp_path / "a6.png", probe=True)["all_labels"]

    labels += plot_dual_frame_verdict(_verdict(), tmp_path / "a7.png",
                                      probe=True)["all_labels"]
    labels += plot_dual_frame_verdict(_verdict(is_target=False), tmp_path / "a8.png",
                                      probe=True)["all_labels"]

    labels += plot_trajectory(_trajectory(), tmp_path / "a9.png",
                              probe=True)["all_labels"]

    labels += plot_height_interval(_orbit(), tmp_path / "a10.png",
                                   probe=True)["all_labels"]
    labels += plot_height_interval(
        OrbitEstimate(rate_arcsec_s=12.0, elevation_deg=41.7,
                      bracket_km=(200.0, 40000.0), converged=False),
        tmp_path / "a11.png", probe=True)["all_labels"]

    labels += plot_hotpixels(_hotmap(), tmp_path / "a12.png", shape=(4096, 4096),
                             report=_sensor_report(), probe=True)["all_labels"]
    labels += plot_hotpixels(_hotmap(xy=()), tmp_path / "a13.png", shape=(4096, 4096),
                             probe=True)["all_labels"]

    mags = np.random.default_rng(9).normal(11.0, 1.5, 300)
    labels += plot_magnitude_histogram(
        mags, tmp_path / "a14.png", limiting=13.8,
        seeing=SeeingEstimate(n_stars=137, n_rejected=0, fwhm_px=3.52,
                              fwhm_arcsec=21.75, elongation_median=1.20),
        sky_mag_arcsec2=17.344337850113781, probe=True)["all_labels"]
    labels += plot_magnitude_histogram(np.full(50, np.nan), tmp_path / "a15.png",
                                       probe=True)["all_labels"]
    return labels


def test_figure_cjk_is_actually_what_the_figures_draw(tmp_path):
    """**强版本**：`FIGURE_CJK` 不是一个手挑的样本，它必须**覆盖本模块所有
    标题、轴标签、图例、文本块里出现的每一个非 ASCII 字符**。
    否则守卫查的是一套字、画的是另一套。

    ASCII 字符的覆盖由 `test_superscript_two_is_never_drawn`（`mag/arcsec^2`）
    与 `test_minus_sign_is_configured_away`（`-`）单独钉住。
    """
    drawn = {ch for text in _all_drawn_labels(tmp_path) for ch in text
             if ord(ch) > 0x7F}
    assert drawn, "八张图一个非 ASCII 字符都没画，说明标签根本没上图"
    missing = sorted(drawn - set(FIGURE_CJK))
    assert missing == [], f"FIGURE_CJK 漏掉了图上实际画出的字符：{''.join(missing)}"

    assert len(FIGURE_CJK) >= 60
    assert "²" not in FIGURE_CJK
    for ch in FIGURE_CJK:
        assert ord(ch) > 0x2FFF or ch in "″σΔ±^2-", f"FIGURE_CJK 混进了 {ch!r}"


def test_the_selected_font_covers_every_character_actually_drawn(tmp_path):
    """收口：把上一条收集到的字符直接喂给选中字体做字形级检查。

    `FIGURE_CJK` 与实际标签的一致性由上一条保证，这一条保证的是
    「实际画出的字符**在选中字体里都有字形**」——两条合起来才排除方框图。
    ASCII 也一并检查：`^`、`2`、`-` 都在里面。

    `\\n` 要排除：多行文本块里的换行符在**任何**字体里字形索引都是 0，
    它不是要渲染的字符。
    """
    name = setup_matplotlib()
    drawn = "".join(sorted({ch for text in _all_drawn_labels(tmp_path)
                            for ch in text if not ch.isspace()}))
    missing = glyphs_missing(name, drawn)
    assert missing == [], f"{name} 缺失图上实际画出的字形：{''.join(missing)}"
