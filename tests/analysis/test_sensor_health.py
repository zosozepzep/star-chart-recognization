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
