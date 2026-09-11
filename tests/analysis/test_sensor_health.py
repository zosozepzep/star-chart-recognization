from __future__ import annotations

import json

import numpy as np
import pytest

from src.analysis.sensor_health import SensorReport, build_report, compare_epochs
from src.calib.hotpixel import HotCluster, HotPixelMap
from src.config import load_config

# ---- 实测常量：来自 probe/t22-recheck.py..t22-recheck4.py，勿改（R417） ----
N_DEFECT_CLUSTERS = 4          # 剔除 (0,0) 读出伪影后的真坏点簇数，两历元一致
N_METADATA_ARTIFACTS = 1       # 每历元 1 个落在第 0 行的读出伪影
MAX_SEPARATION_PX = 0.3353     # 交付口径 4 对的最大分离；5 对口径下同值
N_BIT_IDENTICAL = 2            # R420：交付口径是 2 of 4，不是 5 对口径的 3 of 5
MEAN_SEPARATION_PX = 0.1166    # 4 对口径；5 对口径是 0.0933，不得混用
EPOCH_GAP_DAYS = 134.158023
KNOWN_DEFECTS = [(244.0, 3174.0), (312.0, 3206.0), (327.0, 3210.0), (2192.0, 3223.0)]
# 帧预算从配置读入，不在测试里硬编码——它是 R138 的载荷项
REPORT_FRAMES = int(load_config()["sensor_health"]["report_frames"])


def fake_map(centers):
    clusters = [
        HotCluster(cx=float(x), cy=float(y), npix=4, median_value=1200.0, flat_ratio=0.01)
        for x, y in centers
    ]
    return HotPixelMap(mask=np.zeros((16, 16), dtype=bool), clusters=clusters)


STATS = {"median": 6.45, "std": 3.80, "min": 0.0, "max": 4000.0, "n_saturated": 0}


def report(tag, centers, **kw):
    kw.setdefault("epoch_isot", "2026-03-09T00:00:00")
    return build_report(tag, fake_map(centers), STATS, **kw)


# ---------- build_report ----------

def test_build_report_captures_counts():
    rep = report("A", [(244, 3174), (312, 3206)])
    assert rep.n_clusters == 2
    assert rep.background_median == pytest.approx(6.45)
    assert rep.background_rms == pytest.approx(3.80)
    assert rep.n_saturated == 0


def test_report_to_dict_is_json_ready():
    rep = report("A", [(244, 3174)])
    d = json.loads(json.dumps(rep.to_dict(), allow_nan=False))
    assert d["clusters"][0]["cx"] == pytest.approx(244.0)


def test_report_to_dict_rejects_non_finite_without_allow_nan():
    """R133 的先例：json.dumps 默认会写出裸 NaN，严格消费方会崩。"""
    rep = build_report(
        "A", fake_map([(1, 1)]),
        {"median": float("nan"), "std": 3.8, "min": 0.0, "max": 1.0, "n_saturated": 0},
        epoch_isot="t1",
    )
    d = rep.to_dict()
    assert d["background_median_adu"] is None
    json.loads(json.dumps(d, allow_nan=False))


def test_metadata_row_clusters_are_excluded_but_counted():
    """R418：第 0 行的读出伪影必须从坏点普查中剔除，且剔除数量要如实报出。"""
    rep = report("A", [(0, 0), (7, 0), (2192, 3223)], metadata_rows=1)
    assert rep.n_clusters == 1
    assert rep.n_metadata_artifacts == 2
    assert rep.metadata_rows == 1
    assert [c["cx"] for c in rep.clusters] == [pytest.approx(2192.0)]


def test_metadata_rows_zero_keeps_everything():
    rep = report("A", [(0, 0), (2192, 3223)], metadata_rows=0)
    assert rep.n_clusters == 2
    assert rep.n_metadata_artifacts == 0


def test_metadata_rows_boundary_is_row_index_not_count():
    """cy < metadata_rows 才剔。cy=1.0 在 metadata_rows=1 下必须**保留**——
    差一错会把第 1 行也吃掉，而实测第 1 行有 0 个异常像素，白丢一行感光区。"""
    rep = report("A", [(5, 0.999), (5, 1.0), (5, 1.001)], metadata_rows=1)
    assert rep.n_clusters == 2
    assert rep.n_metadata_artifacts == 1
    assert [pytest.approx(c["cy"]) for c in rep.clusters] == [1.0, 1.001]


def test_metadata_rows_rejects_negative():
    with pytest.raises(ValueError, match="metadata_rows"):
        report("A", [(1, 1)], metadata_rows=-1)


def test_background_provenance_is_recorded():
    """R422/2.7：背景中位数强依赖取哪一帧（实测 B 帧0=11.0 vs 帧19=6.0 ADU，
    差 83.33%），不记录来源的背景数字没有意义。"""
    rep = report("A", [(1, 1)], background_frame_index=19, background_exposure_ms=30.0)
    d = rep.to_dict()
    assert d["background_frame_index"] == 19
    assert d["background_exposure_ms"] == pytest.approx(30.0)


# ---------- compare_epochs ----------

def test_compare_epochs_finds_common_defects():
    a = report("A", [(244, 3174), (312, 3206), (900, 900)])
    b = report("B", [(244.4, 3174.2), (312, 3206), (1500, 1500)], epoch_isot="2026-07-21T00:00:00")
    res = compare_epochs(a, b)
    assert res.n_common == 2
    # 按分离最大值断言，而不是按 common[0]——匹配顺序不该是契约的一部分
    assert res.max_separation_px < 1.0
    assert len(res.only_in_a) == 1 and len(res.only_in_b) == 1
    assert res.persistence == pytest.approx(2 / 3, abs=1e-6)
    assert res.n_a == 3 and res.n_b == 3


def test_compare_epochs_respects_match_radius():
    a = report("A", [(100, 100)])
    b = report("B", [(105, 100)], epoch_isot="t2")
    assert compare_epochs(a, b, match_radius_px=3.0).n_common == 0
    assert compare_epochs(a, b, match_radius_px=8.0).n_common == 1


def test_match_radius_is_exclusive_at_the_boundary():
    """d 恰好等于半径时不算匹配（严格小于），把边界语义钉死。"""
    a = report("A", [(100, 100)])
    b = report("B", [(105, 100)], epoch_isot="t2")
    assert compare_epochs(a, b, match_radius_px=5.0).n_common == 0
    assert compare_epochs(a, b, match_radius_px=5.0001).n_common == 1


def test_match_radius_rejects_non_positive():
    a, b = report("A", [(1, 1)]), report("B", [(1, 1)], epoch_isot="t2")
    with pytest.raises(ValueError, match="match_radius_px"):
        compare_epochs(a, b, match_radius_px=0.0)


def test_matching_picks_the_nearest_pair_not_the_first_seen():
    """R140：贪心「先到先得」会配出更差的一对。

    实测反例：A=[(100,100), (101,100)]、B=[(101.2,100)]，距离 1.2 与 0.2。
    贪心按遍历顺序把 B 配给 A[0]（d=1.2）；正确做法按距离升序分配，
    配给 A[1]（d=0.2）。
    """
    a = report("A", [(100, 100), (101, 100)])
    b = report("B", [(101.2, 100)], epoch_isot="t2")
    res = compare_epochs(a, b, match_radius_px=3.0)
    assert res.n_common == 1
    assert res.common[0]["separation_px"] == pytest.approx(0.2, abs=1e-6)
    assert res.common[0]["xy_a"] == pytest.approx([101.0, 100.0])
    # 被挤掉的那个 A 簇必须落在 only_in_a，不能凭空消失
    assert len(res.only_in_a) == 1
    assert res.only_in_a[0]["cx"] == pytest.approx(100.0)


def test_matching_is_symmetric_under_input_order():
    """两侧各自逆序，配对结果不变——顺序无关是 R140 的可观测后果。"""
    fwd = compare_epochs(
        report("A", [(100, 100), (101, 100)]),
        report("B", [(101.2, 100)], epoch_isot="t2"),
        match_radius_px=3.0,
    )
    rev = compare_epochs(
        report("A", [(101, 100), (100, 100)]),
        report("B", [(101.2, 100)], epoch_isot="t2"),
        match_radius_px=3.0,
    )
    assert fwd.n_common == rev.n_common == 1
    assert fwd.common[0]["separation_px"] == pytest.approx(rev.common[0]["separation_px"])


def test_common_carries_both_npix_because_they_differ_across_epochs():
    """R419：实测 4 对里有 2 对 npix 是 9 vs 7（−22%）。位置精度与簇范围是
    两件事，一列 npix 印不下，所以 common 必须两侧都带。"""
    a = report("A", [(244.111, 3173.556)])
    b = report("B", [(244.143, 3173.429)], epoch_isot="t2")
    a.clusters[0]["npix"] = 9
    b.clusters[0]["npix"] = 7
    res = compare_epochs(a, b)
    assert res.n_common == 1
    assert res.common[0]["npix_a"] == 9
    assert res.common[0]["npix_b"] == 7
    assert res.common[0]["separation_px"] == pytest.approx(0.1310, abs=5e-5)


def test_every_cluster_is_accounted_for_exactly_once():
    """守恒律：n_common + len(only_in_x) 必须等于该侧簇总数，两侧都成立。"""
    a = report("A", [(0, 500), (100, 100), (101, 100), (900, 900)])
    b = report("B", [(101.2, 100), (900.1, 900.0), (3000, 3000)], epoch_isot="t2")
    res = compare_epochs(a, b, match_radius_px=3.0)
    assert res.n_common + len(res.only_in_a) == res.n_a == 4
    assert res.n_common + len(res.only_in_b) == res.n_b == 3


def test_compare_epochs_handles_empty_side():
    a = report("A", [])
    b = report("B", [(100, 100)], epoch_isot="t2")
    res = compare_epochs(a, b)
    assert res.n_common == 0
    assert res.persistence == 0.0
    assert res.jaccard == 0.0
    assert res.max_separation_px is None
    assert len(res.only_in_b) == 1


def test_compare_epochs_handles_both_sides_empty():
    res = compare_epochs(report("A", []), report("B", [], epoch_isot="t2"))
    assert res.n_common == 0 and res.persistence == 0.0
    assert res.only_in_a == [] and res.only_in_b == []
    json.loads(json.dumps(res.to_dict(), allow_nan=False))


def test_persistence_reports_both_counts_so_min_cannot_mislead():
    """R143：persistence = common/min(n_a,n_b)。3 对 100 且 3 个全配上时它读
    1.0000，即「100% 坏点持久」而 97 个 B 簇无解释。jaccard 与 n_a/n_b 让它可审计。"""
    a = report("A", [(100, 100), (200, 200), (300, 300)])
    b = report("B", [(100, 100), (200, 200), (300, 300)] + [(1000 + 10 * i, 50) for i in range(97)],
               epoch_isot="t2")
    res = compare_epochs(a, b)
    assert res.n_common == 3
    assert res.persistence == pytest.approx(1.0)      # min() 口径
    assert res.jaccard == pytest.approx(3 / 100)      # 并集口径，不掩盖盈余
    assert res.n_a == 3 and res.n_b == 100


def test_compare_to_dict_is_json_ready():
    a = report("A", [(244, 3174)])
    b = report("B", [(244, 3174)], epoch_isot="t2")
    d = json.loads(json.dumps(compare_epochs(a, b).to_dict(), allow_nan=False))
    assert d["n_a"] == 1 and d["n_b"] == 1
    assert d["jaccard"] == pytest.approx(1.0)


# ---------- 配置 ----------

def test_config_section_is_present_and_typed():
    cfg = load_config()["sensor_health"]
    assert cfg["report_frames"] == 20
    assert cfg["metadata_rows"] == 1
    assert cfg["match_radius_px"] == pytest.approx(3.0)
    assert cfg["min_epoch_gap_days"] == pytest.approx(100.0)


def test_min_epoch_gap_is_below_the_measured_gap_or_the_claim_is_vacuous():
    """min_epoch_gap_days 是「构成跨历元证据」的门槛。它必须低于实测间隔
    134.158023 天，否则本项目自己的两个历元都不合格；也不能低到把同一晚
    的两组曝光算成两个历元。"""
    gap = float(load_config()["sensor_health"]["min_epoch_gap_days"])
    assert 1.0 < gap < EPOCH_GAP_DAYS


# ---------- 跨数据集（真实数据） ----------

def test_cross_dataset_defects_persist(dataset_a_dir, dataset_b_dir):
    """相隔 134.158 天的两次观测，四个坏点簇 4/4 复现，最大分离 0.3353 px。

    R420：这里的每个数都是**交付口径**（已剔除第 0 行的 4 簇）。含伪影的
    5 簇口径给 mean=0.0933、median=0.0000、「3 对逐位相同」——那组数不得
    出现在任何持久性主张里，因为其中一对就是被本函数剔掉的伪影。
    """
    from astropy.time import Time

    from src.calib.background import background_stats
    from src.calib.hotpixel import build_from_sequence
    from src.dataio.fits_loader import FrameSequence

    reports = []
    for d in (dataset_a_dir, dataset_b_dir):
        seq = FrameSequence.from_directory(d)
        frames = list(range(min(REPORT_FRAMES, len(seq))))
        hpm = build_from_sequence(seq, frames)
        last = frames[-1]
        reports.append(
            build_report(
                seq.dataset_id,
                hpm,
                background_stats(seq.image(last)),
                epoch_isot=str(seq.headers[0].date_obs.isot),
                background_frame_index=last,
                background_exposure_ms=float(seq.headers[last].exposure_ms),
            )
        )

    rep_a, rep_b = reports
    # 每历元：4 个真坏点簇 + 1 个被剔除并计数的读出伪影
    assert rep_a.n_clusters == N_DEFECT_CLUSTERS
    assert rep_b.n_clusters == N_DEFECT_CLUSTERS
    assert rep_a.n_metadata_artifacts == N_METADATA_ARTIFACTS
    assert rep_b.n_metadata_artifacts == N_METADATA_ARTIFACTS
    # 两个夹具必须真的是两个历元——否则本测试用同一目录也会全绿（R139）
    gap = (Time(rep_b.epoch_isot) - Time(rep_a.epoch_isot)).to_value("day")
    assert gap == pytest.approx(EPOCH_GAP_DAYS, abs=1e-4)
    assert gap > float(load_config()["sensor_health"]["min_epoch_gap_days"])

    res = compare_epochs(rep_a, rep_b)
    assert res.n_common == N_DEFECT_CLUSTERS
    assert res.only_in_a == [] and res.only_in_b == []
    assert res.persistence == pytest.approx(1.0)
    assert res.jaccard == pytest.approx(1.0)
    # 实测 0.3353 px。abs=5e-5 是四位钉值自身的舍入余量，不是随手给的松量
    assert res.max_separation_px == pytest.approx(MAX_SEPARATION_PX, abs=5e-5)

    seps = sorted(c["separation_px"] for c in res.common)
    assert len(seps) == N_DEFECT_CLUSTERS
    assert sum(1 for s in seps if s == 0.0) == N_BIT_IDENTICAL
    assert float(np.mean(seps)) == pytest.approx(MEAN_SEPARATION_PX, abs=5e-5)
    assert seps[2] == pytest.approx(0.1310, abs=5e-5)

    # npix 在两历元不同（R419）：两对相同、两对 9 vs 7
    npix_pairs = sorted((c["npix_a"], c["npix_b"]) for c in res.common)
    assert npix_pairs == [(1, 1), (2, 2), (9, 7), (9, 7)]

    found = np.array([c["xy_b"] for c in res.common], dtype=float)
    assert found.shape == (N_DEFECT_CLUSTERS, 2)   # 空 common 会让下面的切片抛 IndexError
    for kx, ky in KNOWN_DEFECTS:
        assert np.min(np.hypot(found[:, 0] - kx, found[:, 1] - ky)) < 1.0


def test_metadata_artifact_is_the_frame_maximum_not_merely_exposure_stable(dataset_b_dir):
    """R418：剔除第 0 行的依据是**量级**，不是曝光响应。

    「不随曝光变化」在历元 A 反号重叠（(0,0) 是 −0.3723，真缺陷 (2192,3223)
    是 −0.3932，(327,3210) 只有 −0.0667），不能区分伪影与真缺陷。能区分的是：
    px(0,0) 恰好是全帧最大值，且是同帧中部行中位数的四位数倍。
    """
    from src.dataio.fits_loader import FrameSequence

    seq = FrameSequence.from_directory(dataset_b_dir)
    img = np.asarray(seq.image(0), dtype=np.float64)
    assert img[0, 0] == img.max()                  # 伪影就是全帧最大值
    ref = float(np.median(img[2000]))
    assert img[0, 0] / ref > 1000.0                # 实测 2452 倍
    assert img[3223, 2192] / ref < 1000.0          # 最亮真缺陷只有 236 倍
    # 第 0 行的中位数与第 1 行相等，所以「整行都是状态字」是错的说法
    assert np.median(img[0]) == np.median(img[1])
    # 剔除代价为零：第 1、2 行没有任何异常像素
    hi = ref + 10.0 * float(img[2000].std())
    assert int(np.count_nonzero(img[0] > hi)) == 3
    assert int(np.count_nonzero(img[1] > hi)) == 0
    assert int(np.count_nonzero(img[2] > hi)) == 0


def test_cross_dataset_cluster_census_is_stable_in_frame_budget(dataset_a_dir):
    """R138/R421：簇数对帧预算是悬崖式依赖，20 帧起才收敛。

    交付口径实测（历元 A，已剔除第 0 行）：5→115、10→8、15→6、20→4、30→4。
    钉的是悬崖的**形状**，不是一个耦合 N_DEFECT_CLUSTERS 的下界——
    `len(census(5)) > 10*N` 那种写法在 N 变动时会倒过来失败，而那与悬崖无关。
    """
    from src.calib.hotpixel import build_from_sequence
    from src.dataio.fits_loader import FrameSequence

    seq = FrameSequence.from_directory(dataset_a_dir)
    assert len(seq) >= 30, "本测试需要至少 30 帧才能证明平台区"

    def census(nf):
        hpm = build_from_sequence(seq, list(range(nf)))
        rep = build_report("A", hpm, {"median": 5.0, "std": 3.4, "min": 0.0, "max": 1.0,
                                      "n_saturated": 0}, epoch_isot="t")
        return sorted((round(c["cx"], 1), round(c["cy"], 1)) for c in rep.clusters)

    at_20, at_30 = census(20), census(30)
    assert len(at_20) == N_DEFECT_CLUSTERS
    assert at_20 == at_30, "20 与 30 帧应给出同一批簇，否则 REPORT_FRAMES 仍在悬崖上"
    assert at_20 == [(244.1, 3173.6), (312.0, 3206.5), (327.0, 3210.1), (2192.0, 3223.0)]
    # 悬崖的形状：5 帧时几乎全是星点与噪声，10 帧已降到个位数
    assert len(census(5)) == 115
    assert len(census(10)) == 8
    assert len(census(15)) == 6


# ---- 观测条件量化（派生数据集 5）----
# 实测常量：来自 probe/t23-recheck2.py..t23-recheck4.py，勿改（R423, R427-429）

SCALE_ARCSEC_PX = 6.179          # 真值定出的板比例（R426：这不是独立观测量）
SKY_MU_MIN, SKY_MU_MAX = 15.0, 23.0
ZP_TRUTH_B = 15.335125           # 数据集 B 真值零点，台账 :9235
ZP_TRUTH_B_STD = 0.217907
# 真实数据实测（第 30 帧）。R402：余量小才有判别力，不得放宽。
FWHM_PX_B, FWHM_PX_A = 3.403649, 3.409136
ELONG_B, ELONG_A = 1.195896, 1.202717
SKY_MU_B, SKY_MU_A = 17.344338, 17.542291


def _fit(sigma=1.6, ok=True, sigma_y=None):
    from src.detect.psf import FWHM_PER_SIGMA, GaussianFit

    sy = sigma if sigma_y is None else sigma_y
    return GaussianFit(
        x=10.0, y=10.0, amplitude=900.0,
        sigma_x=sigma, sigma_y=sy, theta_deg=0.0, background=6.0,
        fwhm_px=FWHM_PER_SIGMA * float(np.sqrt(sigma * sy)),
        residual_ratio=0.05 if ok else 0.9,
        success=ok,
    )


def _derive_zp(exposure_s):
    """从同一批数据派生零点。

    R427：**不要**手写 21.0546 / 17.2474 这两个四位字面量再断 ``abs=1e-6``——
    实测那样两条路径差 3.14e-06，测试必然红。从函数派生时实测差 3.55e-15。
    """
    from src.astrometry.photometry import zero_point_from_truth

    return zero_point_from_truth(
        np.full(12, 5000.0), np.full(12, 8.0), exposure_s=exposure_s
    )


# ---- estimate_seeing ----

def test_estimate_seeing_converts_px_to_arcsec():
    from src.analysis.sensor_health import estimate_seeing
    from src.detect.psf import FWHM_PER_SIGMA

    est = estimate_seeing([_fit(1.6) for _ in range(20)], SCALE_ARCSEC_PX)
    assert est.n_stars == 20
    assert est.n_rejected == 0
    assert est.fwhm_px == pytest.approx(1.6 * FWHM_PER_SIGMA, rel=1e-12)
    assert est.fwhm_px == pytest.approx(3.767712, abs=1e-6)
    assert est.fwhm_arcsec == pytest.approx(est.fwhm_px * SCALE_ARCSEC_PX, rel=1e-12)
    assert est.elongation_median == pytest.approx(1.0, rel=1e-12)


def test_estimate_seeing_uses_median_not_mean():
    """R149：全部 sigma 相同时 median 与 mean 不可区分，旧测试测不出。

    实测 sigma=[1.2,1.4,1.6,1.8,9.9] -> median 1.6000 / mean 3.1800，
    FWHM 3.767712 vs 7.488328，差 3.720616 px。一个失控的拟合不该拖动视宁度。
    """
    from src.analysis.sensor_health import estimate_seeing

    est = estimate_seeing([_fit(s) for s in (1.2, 1.4, 1.6, 1.8, 9.9)], SCALE_ARCSEC_PX)
    assert est.n_stars == 5
    assert est.fwhm_px == pytest.approx(3.767712, abs=1e-6)
    # 均值会给 7.488328，差 3.720616：钉死这个差，比 != 更有判别力
    assert abs(est.fwhm_px - 7.488328) == pytest.approx(3.720616, abs=1e-6)


def test_estimate_seeing_ignores_failed_fits():
    from src.analysis.sensor_health import estimate_seeing
    from src.detect.psf import FWHM_PER_SIGMA

    est = estimate_seeing([_fit(2.0), _fit(9.0, ok=False), _fit(2.0)], SCALE_ARCSEC_PX)
    assert est.n_stars == 2
    assert est.n_rejected == 1
    assert est.fwhm_px == pytest.approx(2.0 * FWHM_PER_SIGMA, rel=1e-12)


@pytest.mark.parametrize(
    "sigma_x,sigma_y",
    [(0.0, 0.0), (-1.6, -1.6), (-1.6, 1.6), (1e6, 1e6)],
)
def test_estimate_seeing_rejects_unphysical_sigma(sigma_x, sigma_y):
    """R150：sigma<=0 与荒谬大值必须被剔除，而不是算进中位数。

    实测四种穿透：sigma=0 -> fwhm 0.0000；两轴同负 -> 乘积为正、负号被 sqrt
    吞掉得 3.767712；一正一负 -> sqrt(负)=nan，np.median 把结果整体传染成 nan
    而 n_stars 照旧计数，随后 to_dict() 里的 nan 让 json.dumps 写出裸 NaN；
    sigma=1e6 -> fwhm 2354820.0450。四种都能过 isfinite。
    """
    from src.analysis.sensor_health import estimate_seeing

    est = estimate_seeing(
        [_fit(1.6), _fit(sigma_x, sigma_y=sigma_y), _fit(1.6)], SCALE_ARCSEC_PX
    )
    assert est.n_stars == 2
    assert est.n_rejected == 1
    assert est.fwhm_px == pytest.approx(3.767712, abs=1e-6)
    json.loads(json.dumps(est.to_dict(), allow_nan=False))


def test_estimate_seeing_reports_elongation_as_the_readability_precondition():
    """R428：拖长比是「这个 FWHM 能不能当星像宽度读」的**前提条件**。

    合成 10:1 拖长（sigma 1.6 / 16.0）实测给 fwhm 11.914552 px、73.6200 arcsec
    ——这个数不是视宁度。**注意这是构造值，不是实测**：真实数据上拖长比
    中位数只有 1.20（见 test_dataset_*），前提是成立的。旧任务书把
    11.914552 写成 11.9151 并称之为「实测」，两处都错（R427/R428）。
    """
    from src.analysis.sensor_health import estimate_seeing

    est = estimate_seeing([_fit(1.6, sigma_y=16.0) for _ in range(10)], SCALE_ARCSEC_PX)
    assert est.elongation_median == pytest.approx(10.0, rel=1e-12)
    assert est.fwhm_px == pytest.approx(11.914552, abs=1e-6)
    assert est.fwhm_arcsec == pytest.approx(73.6200, abs=1e-4)
    assert "拖长" in est.note


def test_estimate_seeing_elongation_is_major_over_minor_not_the_reciprocal():
    """拖长比必须 >= 1。写成 min/max 会给 0.1 而不是 10.0，且沿哪个轴拖长
    不该改变这个数——两个方向必须给出同一个值。"""
    from src.analysis.sensor_health import estimate_seeing

    tall = estimate_seeing([_fit(1.6, sigma_y=16.0)], SCALE_ARCSEC_PX)
    wide = estimate_seeing([_fit(16.0, sigma_y=1.6)], SCALE_ARCSEC_PX)
    assert tall.elongation_median == pytest.approx(10.0, rel=1e-12)
    assert wide.elongation_median == pytest.approx(tall.elongation_median, rel=1e-12)
    assert tall.fwhm_px == pytest.approx(wide.fwhm_px, rel=1e-12)


def test_estimate_seeing_without_any_fit():
    from src.analysis.sensor_health import estimate_seeing

    est = estimate_seeing([], SCALE_ARCSEC_PX)
    assert est.n_stars == 0
    assert est.fwhm_px is None and est.fwhm_arcsec is None
    assert est.elongation_median is None
    assert "无" in est.note
    json.loads(json.dumps(est.to_dict(), allow_nan=False))


def test_seeing_to_dict_is_json_ready_when_every_fit_is_rejected():
    """全部被剔的路径也要能序列化——消费方拿到 null 而不是 NaN。"""
    from src.analysis.sensor_health import estimate_seeing

    est = estimate_seeing([_fit(1e6), _fit(0.0)], SCALE_ARCSEC_PX)
    assert est.n_stars == 0 and est.n_rejected == 2
    assert json.loads(json.dumps(est.to_dict(), allow_nan=False))["fwhm_px"] is None


# ---- sky_brightness ----

def test_sky_brightness_takes_exposure_from_the_zero_point():
    """R424：曝光由 zp.exposure_s 携带。两种自洽约定必须给出同一个 mu。

    实测同一批数据两次派生：exposure_s=0.030 -> ZP 21.0546、
    exposure_s=None -> ZP 17.2474，差 3.807197 = -2.5*log10(0.030)。
    两条路径的 mu 之差实测 3.55e-15（从函数派生；手写四位字面量会是 3.14e-06，
    见 R427，所以这里**必须**用 _derive_zp）。
    """
    from src.analysis.sensor_health import sky_brightness

    zp_abs, zp_plain = _derive_zp(0.030), _derive_zp(None)
    assert zp_abs.exposure_s == pytest.approx(0.030)
    assert zp_plain.exposure_s is None
    assert (zp_abs.value - zp_plain.value) == pytest.approx(3.807197, abs=1e-6)

    kw = dict(scale_arcsec_px=SCALE_ARCSEC_PX)
    mu_abs = sky_brightness(6.45, zero_point=zp_abs, **kw)
    mu_plain = sky_brightness(6.45, zero_point=zp_plain, **kw)
    assert mu_abs == pytest.approx(mu_plain, abs=1e-9)
    assert mu_abs == pytest.approx(19.1781, abs=0.001)
    assert SKY_MU_MIN < mu_abs < SKY_MU_MAX


def test_sky_brightness_refuses_a_caller_exposure_that_contradicts_the_zero_point():
    """R424：不一致**就是** 3.807197 mag 的错，而它可检测，所以必须抛而不是静默挑一边。

    实测：用 exposure_s=None 的零点再除一次 t，19.178117 -> 15.370920。
    与 calibrate_table（photometry.py:158）已确立的先例同形。
    """
    from src.analysis.sensor_health import sky_brightness

    kw = dict(scale_arcsec_px=SCALE_ARCSEC_PX)
    with pytest.raises(ValueError, match="曝光"):
        sky_brightness(6.45, zero_point=_derive_zp(None), exposure_s=0.030, **kw)
    with pytest.raises(ValueError, match="曝光"):
        sky_brightness(6.45, zero_point=_derive_zp(0.030), exposure_s=0.080, **kw)
    # 复述一个一致的值是允许的，且不改变结果
    mu_quiet = sky_brightness(6.45, zero_point=_derive_zp(0.030), **kw)
    mu_echo = sky_brightness(6.45, zero_point=_derive_zp(0.030), exposure_s=0.030, **kw)
    assert mu_echo == pytest.approx(mu_quiet, abs=1e-12)


def test_sky_brightness_grows_fainter_with_longer_exposure():
    """R425 载荷断言 (a)：同样的 ADU 读数在更长曝光下意味着更暗的天空。

    实测 0.030 -> 19.178117、0.080 -> 20.243039，delta +1.064922
    （= 2.5*log10(0.080/0.030)，实测 1.064921830680703）。
    把 b/t 写成 b*t 的变异体给出 delta −1.064922（递减），被这一条抓到。
    物理带抓不到它——带的判别力随输入变化，在 b=1 处归零（R425）。

    **两个零点必须共用同一个 value、只换 exposure_s 字段**，不能各自
    `_derive_zp`。任务书这一条写成 `_derive_zp(0.030)` vs `_derive_zp(0.080)`，
    但那两次派生用的是同一批 flux/truth_mag，于是
    ``ZP(t) = ZP(None) − 2.5log10(t)`` 把曝光项吸进了零点，随后 ``b/t``
    又贡献 ``+2.5log10(t)``，两者**精确抵消**：实测 delta = 0.0（逐位为零），
    `mu_long > mu_short` 对正确实现就是假的。那个抵消正是
    `takes_exposure_from_the_zero_point` 断言的同一事实。
    「同一台设备两个曝光」的正确模型是一个**流量率**零点配两个 exposure_s
    ——那才给出任务书钉的 +1.064922，且 b*t 变异体在此路径上给 −1.064922。
    """
    from src.analysis.sensor_health import sky_brightness
    from src.astrometry.photometry import ZeroPoint

    rate_zp = _derive_zp(0.030)          # 流量率零点，不手写字面量（R427）
    kw = dict(scale_arcsec_px=SCALE_ARCSEC_PX)
    zp_short = ZeroPoint(rate_zp.value, rate_zp.std, rate_zp.n_points, "truth", exposure_s=0.030)
    zp_long = ZeroPoint(rate_zp.value, rate_zp.std, rate_zp.n_points, "truth", exposure_s=0.080)
    mu_short = sky_brightness(6.45, zero_point=zp_short, **kw)
    mu_long = sky_brightness(6.45, zero_point=zp_long, **kw)
    assert mu_long > mu_short
    assert (mu_long - mu_short) == pytest.approx(1.064922, abs=1e-6)
    assert (mu_long - mu_short) == pytest.approx(2.5 * np.log10(0.080 / 0.030), abs=1e-12)
    # 同一批数据各自重拟零点则曝光项精确抵消——这是不变性，不是单调性
    assert sky_brightness(6.45, zero_point=_derive_zp(0.080), **kw) == pytest.approx(
        sky_brightness(6.45, zero_point=_derive_zp(0.030), **kw), abs=1e-9
    )


def test_sky_brightness_grows_brighter_with_more_background():
    """R425 载荷断言 (b)：背景越高天空越亮（mu 越小），且亮 10 倍恰好差 2.5 mag。

    这一条抓符号翻转。**不要**靠物理带抓它：b=1 ADU 时 ``2.5*log10(1)`` 恰好
    为 0，符号翻转与正确实现给出**同一个数**，任何带都失效（R425）。
    那个塌缩值等于 ``ZP + 2.5*log10(6.179²)`` = ``ZP + 3.954590976072891``，
    所以它**随零点走**，不是一个常数：本测试用的合成夹具
    ``_derive_zp(None)``（ZP=17.247425010840050）给 **21.202015986912940**；
    真值零点 ``ZP_TRUTH_B``=15.335125 给 **19.289715976072891**（见
    probe/t26-r444-zp.txt）。R444 记的 19.2897 走的是后者，
    **不是本测试这条路径**，别把它当成这里会看到的数。

    连带纠正 R444 的一句话：它写「b=100 时连正确实现都被带拒」。
    这只在真值零点下成立（14.289715976072891 落在 15<mu<23 带外）；
    本测试的合成夹具下 b=100 给 16.202015986912940，**在带内**。
    带的判别力归零这个结论不依赖 b=100，它只依赖 b=1 处的恒等式。
    旧任务书钉的 17.2115 在十种读法下都复现不出，不得引用。

    b=20.0 的钉值是 **17.949441**，不是任务书写的 17.949416：后者用四位字面量
    ``ZP=17.2474`` 算出（17.949415986912939），正是 R427 自己禁止的读法；
    从 ``zero_point_from_truth`` 派生的 ``ZP=17.247425010840050``
    实测给 **17.949440997752987**，两者差 2.5e-05，远超 abs=1e-6。
    """
    from src.analysis.sensor_health import sky_brightness

    kw = dict(zero_point=_derive_zp(None), scale_arcsec_px=SCALE_ARCSEC_PX)
    assert sky_brightness(20.0, **kw) < sky_brightness(6.45, **kw)
    assert sky_brightness(20.0, **kw) == pytest.approx(17.949441, abs=1e-6)
    delta = sky_brightness(6.45, **kw) - sky_brightness(64.5, **kw)
    assert delta == pytest.approx(2.5, abs=1e-9)


def test_sky_brightness_area_normalisation_is_per_arcsec2():
    """R425 载荷断言 (c)：面积项恰好 2.5*log10(6.179²) = 3.954591 mag，且是**加**的。

    per-arcsec² 的流量比 per-px 小 38.1800 倍，所以星等更暗。
    丢掉 /s² 在这个算例上给 15.223526（任务书写的 9.5039 是
    「13.4585 − 3.954591」，而 13.4585 本身是零点错配的产物，不是本函数的输出）。
    """
    from src.analysis.sensor_health import sky_brightness

    zp = _derive_zp(None)
    mu = sky_brightness(6.45, zero_point=zp, scale_arcsec_px=SCALE_ARCSEC_PX)
    mu_unit = sky_brightness(6.45, zero_point=zp, scale_arcsec_px=1.0)
    assert (mu - mu_unit) == pytest.approx(3.954591, abs=1e-6)
    assert (mu - mu_unit) == pytest.approx(2.5 * np.log10(SCALE_ARCSEC_PX ** 2), abs=1e-12)
    assert mu > mu_unit          # 方向：per-arcsec² 更暗


def test_sky_brightness_is_none_without_zero_point():
    from src.analysis.sensor_health import sky_brightness

    assert sky_brightness(6.45, zero_point=None, scale_arcsec_px=SCALE_ARCSEC_PX) is None


@pytest.mark.parametrize("bkg,s", [(0.0, 6.179), (-1.0, 6.179), (6.45, 0.0), (6.45, -1.0)])
def test_sky_brightness_is_none_for_nonpositive_inputs(bkg, s):
    """非正输入返回 None，不留 nan，也不留 RuntimeWarning。"""
    from src.analysis.sensor_health import sky_brightness

    assert sky_brightness(bkg, zero_point=_derive_zp(None), scale_arcsec_px=s) is None


def test_sky_brightness_none_survives_json():
    from src.analysis.sensor_health import sky_brightness

    payload = {"sky_mu": sky_brightness(0.0, zero_point=_derive_zp(None),
                                        scale_arcsec_px=SCALE_ARCSEC_PX)}
    assert json.loads(json.dumps(payload, allow_nan=False))["sky_mu"] is None


# ---- 真实数据（R428：断言余量必须小，实测值见上表） ----

def _measure(directory, frame=30):
    from src.analysis.sensor_health import estimate_seeing
    from src.calib.background import model_background
    from src.dataio.fits_loader import FrameSequence
    from src.detect.psf import fit_table
    from src.detect.segmentation import detect_sources_in_frame

    seq = FrameSequence.from_directory(directory)
    img = seq.image(frame)
    model = model_background(img)
    table = detect_sources_in_frame(img, model, n_sigma=5.0, frame=frame).brightest(80)
    _, fits = fit_table(model.subtract(img), table)
    return estimate_seeing(fits, SCALE_ARCSEC_PX), fits, img


def test_dataset_b_seeing_is_about_three_and_a_half_pixels(dataset_b_dir):
    """实测 fwhm_px 3.403649、拖长中位 1.195896、78 个拟合全部通过 sigma 过滤。

    R402：旧任务书的 2.0<x<8.0 在实测 3.40 上余量 1.4/4.6，太松，
    收紧到 ±0.2。1.0<=elong<20.0 同理收到 1.15..1.30。
    """
    est, fits, _ = _measure(dataset_b_dir)
    assert est.n_stars == 78
    assert 3.2 < est.fwhm_px < 3.6
    assert est.fwhm_px == pytest.approx(FWHM_PX_B, abs=0.01)
    assert est.fwhm_arcsec == pytest.approx(21.031149, abs=0.1)
    assert 1.15 < est.elongation_median < 1.30
    assert est.elongation_median == pytest.approx(ELONG_B, abs=0.01)
    json.loads(json.dumps(est.to_dict(), allow_nan=False))


def test_sigma_filter_rejects_nothing_on_real_data(dataset_b_dir):
    """R428：σ 过滤器 [0.3, 20.0] 在真实数据上实测剔 0 个（σ 跨度 0.883..2.063）。

    它防的是穿透，不是常态——所以报告与 docstring 不得暗示它筛掉了什么。
    这条断言比 `n_stars >= 20` 更有判别力：过滤器一旦意外变严就立刻红。
    """
    est, fits, _ = _measure(dataset_b_dir)
    assert est.n_rejected == 0
    assert est.n_stars == sum(1 for f in fits if f.success)


def test_seeing_agrees_across_two_epochs(dataset_a_dir, dataset_b_dir):
    """R428：fwhm_px 是**纯像素量、不含真值**，所以这一条可以当独立自查主张。

    实测跨 134.158023 天、不同曝光（30 vs 40 ms）、不同目标：
    3.403649 vs 3.409136，差 0.005487 px（相对 0.16%）；拖长比差 0.006821。
    """
    est_b, _, _ = _measure(dataset_b_dir)
    est_a, _, _ = _measure(dataset_a_dir)
    assert est_a.fwhm_px == pytest.approx(FWHM_PX_A, abs=0.01)
    assert abs(est_b.fwhm_px - est_a.fwhm_px) < 0.05
    assert abs(est_b.elongation_median - est_a.elongation_median) < 0.05


def test_dataset_b_sky_brightness_is_a_conditional_statement(dataset_b_dir):
    """R426/R429：这个数的全部信息量在零点里，而零点是真值定出的。

    给定真值零点 15.335125（散度 0.217907）与真值比例尺 6.179，
    实测数据集 B 第 30 帧全帧中位 6.0 ADU -> mu = 17.344338 mag/arcsec²，
    落在城市微光夜空（~17）边缘。**这是条件命题，不是独立观测量。**
    背景中位是整数 ADU，1 ADU 在 6.0 上就是 0.18 mag，所以小数位无意义。
    """
    from src.analysis.sensor_health import sky_brightness
    from src.astrometry.photometry import ZeroPoint

    _, _, img = _measure(dataset_b_dir)
    bkg = float(np.median(np.asarray(img, dtype=np.float64)))
    assert bkg == 6.0                       # 整数量化，实测
    zp = ZeroPoint(ZP_TRUTH_B, ZP_TRUTH_B_STD, 55, "truth", exposure_s=None)
    mu = sky_brightness(bkg, zero_point=zp, scale_arcsec_px=SCALE_ARCSEC_PX)
    assert mu == pytest.approx(SKY_MU_B, abs=1e-4)
    assert SKY_MU_MIN < mu < SKY_MU_MAX
    # 零点是唯一的信息来源：平移零点 1 mag，mu 平移 1 mag
    zp2 = ZeroPoint(ZP_TRUTH_B + 1.0, ZP_TRUTH_B_STD, 55, "truth", exposure_s=None)
    mu2 = sky_brightness(bkg, zero_point=zp2, scale_arcsec_px=SCALE_ARCSEC_PX)
    assert (mu2 - mu) == pytest.approx(1.0, abs=1e-9)
