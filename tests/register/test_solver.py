from __future__ import annotations

import numpy as np
import pytest

from src.detect.segmentation import SourceTable
from src.register.solver import (
    RegistrationError,
    register_sequence,
    solve_pair,
    vote_correspondences,
)
from src.register.transform import apply_transform, decompose, similarity_matrix

CENTER = (2047.5, 2047.5)
VOTE_KW = dict(
    center=CENTER,
    rot_range_deg=1.5,
    rot_step_deg=0.05,
    shift_max_px=400.0,
    vote_bin_px=4.0,
    match_radius_px=5.0,
)


def as_table(xy, frame=0):
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    return SourceTable(
        frame=frame,
        x=xy[:, 0].copy(),
        y=xy[:, 1].copy(),
        flux=np.full(n, 1000.0),
        peak=np.full(n, 200.0),
        elongation=np.ones(n),
        npix=np.full(n, 9, dtype=np.int64),
    )


def synth_pair(n=90, rot=0.1, dx=-126.0, dy=5.0, seed=0, drop=0.0, noise=0.0):
    rng = np.random.default_rng(seed)
    src = rng.uniform(200.0, 3900.0, size=(n, 2))
    M = similarity_matrix(rotation_deg=rot, tx=dx, ty=dy, center=CENTER)
    dst = apply_transform(M, src)
    if noise:
        dst = dst + rng.normal(0.0, noise, size=dst.shape)
    if drop:
        keep = rng.random(n) > drop
        dst = dst[keep]
    return src, dst, M


def test_vote_finds_correspondences():
    src, dst, _ = synth_pair()
    a, b, n = vote_correspondences(src, dst, **VOTE_KW)
    assert n >= 80
    assert a.shape == b.shape == (n, 2)


def test_vote_survives_partial_overlap_and_noise():
    src, dst, _ = synth_pair(n=120, drop=0.35, noise=0.4, seed=3)
    _, _, n = vote_correspondences(src, dst, **VOTE_KW)
    assert n >= 50


def test_vote_returns_zero_on_unrelated_fields():
    rng = np.random.default_rng(8)
    src = rng.uniform(0.0, 4096.0, size=(60, 2))
    dst = rng.uniform(0.0, 4096.0, size=(60, 2))
    _, _, n = vote_correspondences(src, dst, **VOTE_KW)
    assert n < 20


def test_vote_returns_empty_shapes_below_min_points():
    """< 3 points on either side must early-return (0, 2)-shaped arrays, not just n=0.

    A caller doing ``a.shape == b.shape == (n, 2)`` (as
    ``test_vote_finds_correspondences`` does) depends on the shape, not just the count.
    """
    rng = np.random.default_rng(9)
    src = rng.uniform(0.0, 4096.0, size=(2, 2))
    dst = rng.uniform(0.0, 4096.0, size=(2, 2))
    a, b, n = vote_correspondences(src, dst, **VOTE_KW)
    assert n == 0
    assert a.shape == (0, 2)
    assert b.shape == (0, 2)

    src3 = rng.uniform(0.0, 4096.0, size=(3, 2))
    dst3 = rng.uniform(0.0, 4096.0, size=(3, 2))
    a3, b3, n3 = vote_correspondences(src3, dst3, **VOTE_KW)
    assert n3 == 0
    assert a3.shape == (0, 2)
    assert b3.shape == (0, 2)


def test_solve_pair_recovers_transform():
    src, dst, M = synth_pair(rot=0.12, dx=-125.7, dy=4.2, seed=5)
    sol = solve_pair(as_table(src, 0), as_table(dst, 1), min_inliers=30, **VOTE_KW)
    assert sol.frame_from == 0 and sol.frame_to == 1
    assert sol.n_inliers >= 70
    assert sol.rms_px < 0.2
    # matrix 把 frame_to 坐标映回 frame_from 坐标，故为 M 的逆
    back = apply_transform(sol.matrix, dst)
    d = decompose(sol.matrix)
    assert d["rotation_deg"] == pytest.approx(-0.12, abs=0.02)
    assert np.abs(back - apply_transform(np.linalg.inv(M), dst)).max() < 0.5


def test_solve_pair_raises_when_inliers_insufficient():
    rng = np.random.default_rng(12)
    src = rng.uniform(0.0, 4096.0, size=(40, 2))
    dst = rng.uniform(0.0, 4096.0, size=(40, 2))
    with pytest.raises(RegistrationError):
        solve_pair(as_table(src, 0), as_table(dst, 1), min_inliers=30, **VOTE_KW)


def test_solve_pair_needs_rotation_scan_at_large_rotation():
    """Rotation scan sweep (n=120, seed=7): full 61-step scan vs collapsed rot=0 only.

    | true rotation | full scan | rot=0 only |
    |---|---|---|
    | 0.10 deg | 120 | 119 |
    | 0.12 deg | 120 | 112 |
    | 0.20 deg | 120 | 42 |
    | 0.30 deg | 120 | 21 (below min_inliers=30) |
    | 0.60 deg | 120 | 11 |
    | 1.50 deg | 120 | 2 |

    At rot=0.6 deg the corner displacement (30.32 px true corner, 21.55 px single-axis)
    is far outside match_radius_px=5.0 with no rotation search, so the full scan is
    load-bearing here, unlike the brief's rot=0.1/0.12 fixtures.
    """
    src, dst, _ = synth_pair(n=120, rot=0.6, dx=-126.0, dy=5.0, seed=7)
    sol = solve_pair(as_table(src, 0), as_table(dst, 1), min_inliers=30, **VOTE_KW)
    assert sol.n_inliers >= 110
    assert sol.rms_px < 1e-6
    d = decompose(sol.matrix)
    assert d["rotation_deg"] == pytest.approx(-0.60, abs=0.02)


def test_solve_pair_rms_reflects_injected_noise():
    """rms_px must be a real residual, not a hardcoded constant.

    Measured (n=120, rot=0.6, seed=7): sigma=0.0 -> rms 0.00000; sigma=0.3 -> rms
    ~0.396 (not 0.3, because the residual is the 2-D radial distance of per-axis
    Gaussian noise, not the per-axis sigma itself); sigma=0.4 -> 0.528; sigma=0.8 ->
    1.057. Two-sided bound: the lower bound catches a hardcoded zero, the upper bound
    catches a residual computed in the wrong direction or against the wrong point set.
    """
    src, dst, _ = synth_pair(n=120, rot=0.6, dx=-126.0, dy=5.0, seed=7, noise=0.3)
    sol = solve_pair(as_table(src, 0), as_table(dst, 1), min_inliers=30, **VOTE_KW)
    assert 0.30 < sol.rms_px < 0.50


def test_register_sequence_chains_transforms():
    """构造 10 帧、每帧移动 126 px 的合成序列，累计位移必须线性累加。"""
    rng = np.random.default_rng(21)
    base = rng.uniform(-2000.0, 6000.0, size=(400, 2))
    frames = list(range(10))
    detections = {}
    for i in frames:
        M = similarity_matrix(rotation_deg=-0.1 * i, tx=126.0 * i, ty=-4.0 * i, center=CENTER)
        xy = apply_transform(M, base)
        inside = (xy[:, 0] > 0) & (xy[:, 0] < 4096) & (xy[:, 1] > 0) & (xy[:, 1] < 4096)
        detections[i] = as_table(xy[inside], frame=i)

    result = register_sequence(detections, frames, reference=0)
    assert result.reference == 0
    assert np.allclose(result.matrices[0], np.eye(3))
    # frame i 的点映回参考系应与 frame 0 的对应点重合
    #
    # 边界必须是 1e-6 而不是原 brief 的 1.0：链式累积实测误差是 1.3e-10 px
    # （f1-f9 全部 6.7e-11 - 1.4e-10），而把 `matrices[cur] @ sol.matrix` 写成
    # `sol.matrix @ matrices[cur]`（Step 5 变异 2）只让 f9 偏离 0.046 px、f3 偏离
    # 0.0015 px——在 1.0 下这个变异完全不被察觉。本合成序列每帧只转 0.1 度，
    # 矩阵乘法不可交换的那部分（旋转与平移的耦合项）因此很小，靠几何放大不现实，
    # 只能靠边界。1e-6 对实测值仍有 7700x 余量，对变异有 1.5e3x（f3）的判别力。
    for i in [3, 6, 9]:
        M = similarity_matrix(rotation_deg=-0.1 * i, tx=126.0 * i, ty=-4.0 * i, center=CENTER)
        xy = apply_transform(M, base[:20])
        sky = result.to_sky(i, xy)
        assert np.abs(sky - base[:20]).max() < 1e-6


def test_to_detector_is_inverse_of_to_sky():
    """Algebraic-identity form: allclose(to_detector(to_sky(x))) holds for ANY invertible
    matrix (measured with a garbage matrix unrelated to registration: allclose True, max
    err 4.55e-13). This test alone cannot witness a wrong `matrices` — see the
    ground-truth form in test_to_sky_recovers_ground_truth_positions below, which is the
    one that has teeth (garbage matrix there is off by 9.56e+03 px).
    """
    rng = np.random.default_rng(22)
    base = rng.uniform(0.0, 4096.0, size=(300, 2))
    frames = list(range(5))
    detections = {
        i: as_table(
            apply_transform(
                similarity_matrix(tx=100.0 * i, center=CENTER), base
            ),
            frame=i,
        )
        for i in frames
    }
    result = register_sequence(detections, frames, reference=0)
    xy = np.array([[1000.0, 1000.0], [2500.0, 3000.0]])
    assert np.allclose(result.to_detector(3, result.to_sky(3, xy)), xy, atol=1e-6)


def test_to_sky_recovers_ground_truth_positions():
    """Ground-truth form (Ruling 43): unlike the algebraic-identity roundtrip test above,
    this pins that `matrices` is actually correct, not just self-consistent. A garbage
    matrix fails this by 9.56e+03 px (measured), while it passes the roundtrip test.
    """
    rng = np.random.default_rng(22)
    base = rng.uniform(0.0, 4096.0, size=(300, 2))
    frames = list(range(5))
    detections = {
        i: as_table(
            apply_transform(
                similarity_matrix(tx=100.0 * i, center=CENTER), base
            ),
            frame=i,
        )
        for i in frames
    }
    result = register_sequence(detections, frames, reference=0)
    detector_xy = apply_transform(similarity_matrix(tx=100.0 * 3, center=CENTER), base[:20])
    sky = result.to_sky(3, detector_xy)
    assert np.abs(sky - base[:20]).max() < 1e-6


def test_report_contains_inlier_and_rms_summary():
    rng = np.random.default_rng(23)
    base = rng.uniform(0.0, 4096.0, size=(300, 2))
    frames = list(range(4))
    detections = {
        i: as_table(apply_transform(similarity_matrix(tx=80.0 * i, center=CENTER), base), i)
        for i in frames
    }
    rep = register_sequence(detections, frames, reference=0).report()
    assert rep["n_pairs"] == 3
    assert rep["inliers_min"] >= 290
    assert rep["rms_max_px"] < 0.5
    assert "cumulative" in rep
    assert np.isfinite(rep["inliers_median"])
    assert np.isfinite(rep["rms_median_px"])
    assert len(rep["pairs"]) == rep["n_pairs"]
    for pair_rec, (f_from, f_to) in zip(rep["pairs"], zip(frames, frames[1:])):
        assert pair_rec["from"] == f_from
        assert pair_rec["to"] == f_to
    # matrices[frame] maps detector -> reference, i.e. the inverse of the physical
    # field motion, so decompose(...)["tx"] carries the OPPOSITE sign to the field
    # drift. This sequence drifts tx=+80 px/frame; the cumulative tx over 3 pairs to
    # frame 3 measures -240.000. The dataset test's cumulative tx=+6495 (positive) is
    # the same convention applied to a field drifting the other way.
    assert rep["cumulative"]["tx"] == pytest.approx(-240.0, abs=0.01)


def test_register_sequence_requires_at_least_two_frames():
    with pytest.raises(ValueError, match="配准至少需要 2 帧"):
        register_sequence({0: as_table(np.zeros((5, 2)))}, [0])


def test_register_sequence_requires_reference_be_first_frame():
    """This guard is load-bearing: `matrices` is seeded only at `ref`, so a mid-sequence
    reference would silently produce a KeyError deep in the chain loop.
    """
    rng = np.random.default_rng(24)
    base = rng.uniform(0.0, 4096.0, size=(50, 2))
    frames = [0, 1, 2]
    detections = {
        i: as_table(apply_transform(similarity_matrix(tx=10.0 * i, center=CENTER), base), i)
        for i in frames
    }
    with pytest.raises(ValueError, match="当前实现要求参考帧为 frames 的首帧"):
        register_sequence(detections, frames, reference=1)


def test_register_sequence_config_min_inliers_is_plumbed():
    """config={'min_inliers': 500} on a fixture that otherwise solves with ~300 inliers
    must raise RegistrationError -- proves the value reaches solve_pair rather than
    being silently dropped in favour of the 30 default.
    """
    rng = np.random.default_rng(25)
    base = rng.uniform(0.0, 4096.0, size=(300, 2))
    frames = list(range(3))
    detections = {
        i: as_table(apply_transform(similarity_matrix(tx=50.0 * i, center=CENTER), base), i)
        for i in frames
    }
    with pytest.raises(RegistrationError):
        register_sequence(detections, frames, reference=0, config={"min_inliers": 500})


def test_register_sequence_config_rot_range_deg_is_plumbed():
    """config={'rot_range_deg': 0.0} on the rot=0.6 fixture must also fail, proving the
    vote kwargs are plumbed through config and not just the popped min_inliers key.
    """
    src, dst, _ = synth_pair(n=120, rot=0.6, dx=-126.0, dy=5.0, seed=7)
    detections = {0: as_table(src, 0), 1: as_table(dst, 1)}
    with pytest.raises(RegistrationError):
        register_sequence(
            detections, [0, 1], reference=0, config={"rot_range_deg": 0.0}
        )


def test_dataset_b_registration_matches_measured_baseline(dataset_b_dir):
    """复现 spec 基线：内点 74-93，累计 f70->f16 尺度 0.99822 / 旋转 -3.325 度。"""
    from src.calib.segment import tracking_segment
    from src.dataio.fits_loader import FrameSequence
    from src.detect.segmentation import detect_sequence
    from src.calib.hotpixel import build_from_sequence

    seq = FrameSequence.from_directory(dataset_b_dir)
    seg = tracking_segment(seq.headers)
    frames = list(range(seg.start, seg.end + 1))
    hpm = build_from_sequence(seq, frames[:30])
    dets = detect_sequence(seq, frames, n_sigma=4.0, npixels=5, hot_clusters=hpm.clusters)

    result = register_sequence(dets, frames, reference=seg.start)
    rep = result.report()
    pairs = rep["pairs"]
    assert len(pairs) == 54
    assert pairs[0]["from"] == seg.start
    # 段首 pair (f16->f17) 跨机架整定过渡：ΔAZ 0.11899 deg/frame 仅为段内均值
    # 0.221442 的 53.7%，帧间映射带真实切变，4 参数相似变换吸收不了
    # （同点集 6 参数仿射 rms 0.6618，比值 0.357）。实测 1.8567。
    assert pairs[0]["rms_px"] < 2.0
    # 其余 53 个 pair 全部健康：实测 0.6229-0.8322
    assert max(q["rms_px"] for q in pairs[1:]) < 1.0
    assert rep["rms_median_px"] < 1.0            # 实测 0.7363
    assert rep["rms_max_px"] == pytest.approx(pairs[0]["rms_px"])
    assert rep["inliers_min"] >= 50              # 实测 74
    cum = decompose(result.matrices[seg.end])
    assert cum["scale"] == pytest.approx(0.99822, abs=0.004)
    assert cum["rotation_deg"] == pytest.approx(-3.325, abs=0.3)
    assert cum["tx"] == pytest.approx(6495.0, rel=0.05)
