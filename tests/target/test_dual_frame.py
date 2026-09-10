from __future__ import annotations

import json

import numpy as np
import pytest

from src.detect.segmentation import SourceTable
from src.register.solver import register_sequence
from src.register.transform import apply_transform, similarity_matrix
from src.target.dual_frame import Track, build_tracks, classify, find_targets

CENTER = (2047.5, 2047.5)


def as_table(xy, frame, flux=None):
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    n = len(xy)
    return SourceTable(
        frame=frame,
        x=xy[:, 0].copy(),
        y=xy[:, 1].copy(),
        flux=np.full(n, 1000.0) if flux is None else np.asarray(flux, dtype=np.float64),
        peak=np.full(n, 300.0),
        elongation=np.full(n, 1.2),
        npix=np.full(n, 9, dtype=np.int64),
    )


def make_track(
    xy_det,
    xy_sky,
    *,
    frames=None,
    peak=None,
    flux=None,
    elongation=None,
    lock_span_px=None,
) -> Track:
    """Build a Track directly, bypassing build_tracks.

    Hand construction is deliberate for the criterion-shape tests: the point
    is to pin what ``pixel_locked``/``classify`` do with a given geometry, not
    to re-exercise the linker.

    ``lock_span_px=None`` deliberately falls through to the dataclass default
    instead of restating 1.0 here, so that raising the production default is a
    mutation these tests can catch.
    """
    xy_det = np.asarray(xy_det, dtype=np.float64).reshape(-1, 2)
    xy_sky = np.asarray(xy_sky, dtype=np.float64).reshape(-1, 2)
    n = len(xy_det)
    kwargs = {} if lock_span_px is None else {"lock_span_px": lock_span_px}
    return Track(
        frames=list(range(n)) if frames is None else list(frames),
        xy_det=xy_det,
        xy_sky=xy_sky,
        flux=np.full(n, 1000.0) if flux is None else np.asarray(flux, dtype=np.float64),
        peak=np.full(n, 900.0) if peak is None else np.asarray(peak, dtype=np.float64),
        elongation=(
            np.full(n, 1.2) if elongation is None else np.asarray(elongation, dtype=np.float64)
        ),
        **kwargs,
    )


def static_det_moving_sky(n_frames: int, *, step_det: float = 0.0, step_sky: float = 126.0):
    """Return (xy_det, xy_sky) for a source drifting ``step_det`` px/frame in x.

    The sky track always drifts ``step_sky`` px/frame, so ``drift_sky_px`` is
    ``step_sky * (n_frames - 1)``: 1134 / 3654 / 6678 px at 10 / 30 / 54 frames.
    """
    i = np.arange(n_frames, dtype=np.float64)
    xy_det = np.column_stack((244.0 + step_det * i, 3174.0 + np.zeros_like(i)))
    xy_sky = np.column_stack((244.0 + step_sky * i, 3174.0 + np.zeros_like(i)))
    return xy_det, xy_sky


def _build_scene(*, second_target: bool):
    """30 帧：星场每帧移动 126 px，探测器系近静止的目标，3 个热像素残点。

    目标在探测器系每帧漂移 1.8 px（与数据集 B 实测一致），恒星在探测器系
    每帧漂移 126 px。热像素残点在探测器系严格不动。

    ``second_target=True`` 时另加两个源，使轨迹长度不再全部相同——
    `synthetic_scene` 的四条轨迹实测**全都是长度 30**，在那上面断言排序会退化成
    恒真式。加的两个源是：第 10 帧进场的第二个目标（长度 20），以及 f0–f14 只存在
    15 帧就消失的静止点。后者是必需的：只有第二个目标时播种顺序本身就恰好是
    降序（f0 先给出四条长度 30，f10 再给出长度 20），删掉 `tracks.sort` 也测不出
    来；加上这个 f0 播种、长度 15 的源后，播种顺序变成
    ``[30, 30, 30, 30, 15, 20]``，不再是降序。
    """
    rng = np.random.default_rng(31)
    frames = list(range(30))
    base_stars = rng.uniform(-1500.0, 5500.0, size=(500, 2))
    locked = np.array([[244.0, 3174.0], [312.0, 3206.0], [327.0, 3210.0]])
    detections = {}
    for i in frames:
        M = similarity_matrix(rotation_deg=-0.1 * i, tx=126.0 * i, ty=-4.0 * i, center=CENTER)
        stars = apply_transform(M, base_stars)
        inside = (
            (stars[:, 0] > 5) & (stars[:, 0] < 4090) & (stars[:, 1] > 5) & (stars[:, 1] < 4090)
        )
        target = np.array([[2271.6 + 1.81 * i, 1958.8 + 0.33 * i]])
        xy = np.vstack([stars[inside], target, locked])
        flux = np.concatenate(
            [np.full(inside.sum(), 1500.0), [264.0 + 38.0 * i], np.full(3, 900.0)]
        )
        if second_target:
            if i <= 14:  # short-lived static point, seeded at f0, length 15
                xy = np.vstack([xy, [[700.0, 500.0]]])
                flux = np.concatenate([flux, [900.0]])
            if i >= 10:  # second target, enters at f10, length 20
                xy = np.vstack([xy, [[1500.0 + 1.7 * (i - 10), 2600.0 + 0.4 * (i - 10)]]])
                flux = np.concatenate([flux, [1200.0]])
        detections[i] = as_table(xy, i, flux)
    # frames is ascending and contiguous — register_sequence does not check this
    # (Ruling 335 Minor 4) and would silently produce garbage otherwise.
    registration = register_sequence(detections, frames, reference=0)
    return frames, detections, registration


@pytest.fixture()
def synthetic_scene():
    """One moving target (length 30) plus three static hot pixels."""
    return _build_scene(second_target=False)


@pytest.fixture()
def two_target_scene():
    """Same scene plus a second target at f10 and a short-lived point at f0-f14.

    Measured: ``build_tracks`` returns lengths ``[30, 30, 30, 30, 20, 15]`` from a
    seeding order of ``[30, 30, 30, 30, 15, 20]``, and ``find_targets`` returns 2
    hits in order ``[30, 20]``. A separate fixture is used deliberately — several
    tests assert ``== 4`` tracks and exactly 1 hit against ``synthetic_scene``,
    and those assertions are correct as they stand.
    """
    return _build_scene(second_target=True)




def test_build_tracks_finds_stationary_track(synthetic_scene):
    """Exactly four long tracks, split 3 locked / 1 not (Ruling 51).

    The four are: the moving target (1.81, 0.33 px/frame) and the three planted
    static points at (244, 3174), (312, 3206), (327, 3210). ``>= 4`` would pass
    silently on a spurious fifth fragment, which is the failure mode
    ``build_tracks`` actually has — the ``claimed`` set is what prevents it.
    """
    frames, detections, reg = synthetic_scene
    tracks = build_tracks(detections, frames, reg, radius=8.0, min_frames=10)
    long_tracks = [t for t in tracks if t.length >= 25]
    assert len(long_tracks) == 4
    locked = [t for t in long_tracks if t.pixel_locked]
    assert len(locked) == 3
    assert len([t for t in long_tracks if not t.pixel_locked]) == 1


def test_target_track_has_expected_kinematics(synthetic_scene):
    frames, detections, reg = synthetic_scene
    tracks = build_tracks(detections, frames, reg, radius=8.0, min_frames=10)
    target = max(
        (t for t in tracks if not t.pixel_locked), key=lambda t: t.length, default=None
    )
    assert target is not None
    assert target.length >= 25
    assert target.v_det_px == pytest.approx(1.84, abs=0.3)
    assert target.v_sky_px == pytest.approx(126.0, rel=0.1)


def test_classify_accepts_target(synthetic_scene):
    frames, detections, reg = synthetic_scene
    tracks = build_tracks(detections, frames, reg, radius=8.0, min_frames=10)
    target = max((t for t in tracks if not t.pixel_locked), key=lambda t: t.length)
    verdict = classify(target)
    assert verdict.is_target
    assert verdict.contrast > 20.0
    assert not verdict.pixel_locked


def test_classify_rejects_pixel_locked_source(synthetic_scene):
    frames, detections, reg = synthetic_scene
    tracks = build_tracks(detections, frames, reg, radius=8.0, min_frames=10)
    locked = [t for t in tracks if t.pixel_locked]
    assert locked, "合成场景应产生像素锁定轨迹"
    for t in locked:
        v = classify(t)
        assert not v.is_target
        assert "像素锁定" in " ".join(v.reasons)


def test_classify_rejects_star(synthetic_scene):
    """恒星在探测器系高速移动，v_det 远超阈值。"""
    frames, detections, reg = synthetic_scene
    fake_star = build_tracks(
        {i: as_table([[100.0 + 126.0 * i, 100.0]], i) for i in frames[:15]},
        frames[:15],
        reg,
        radius=200.0,
        min_frames=10,
    )
    assert fake_star
    v = classify(fake_star[0])
    assert not v.is_target


def test_find_targets_returns_exactly_one(synthetic_scene):
    frames, detections, reg = synthetic_scene
    verdicts = find_targets(detections, frames, reg)
    assert len(verdicts) == 1
    assert verdicts[0].is_target


def test_short_tracks_are_discarded(synthetic_scene):
    frames, detections, reg = synthetic_scene
    tracks = build_tracks(detections, frames, reg, radius=8.0, min_frames=25)
    assert all(t.length >= 25 for t in tracks)


def test_seeding_from_every_frame_matters():
    """只从首帧播种会漏掉中途进场的目标——这是本项目踩过的坑。"""
    frames = list(range(20))
    rng = np.random.default_rng(41)
    base = rng.uniform(-1000.0, 5000.0, size=(400, 2))
    detections = {}
    for i in frames:
        M = similarity_matrix(tx=126.0 * i, center=CENTER)
        stars = apply_transform(M, base)
        inside = (stars[:, 0] > 5) & (stars[:, 0] < 4090)
        xy = stars[inside]
        if i >= 8:  # 目标第 8 帧才出现
            xy = np.vstack([xy, [[2000.0 + 1.8 * (i - 8), 1500.0]]])
        detections[i] = as_table(xy, i)
    reg = register_sequence(detections, frames, reference=0)
    verdicts = find_targets(detections, frames, reg, config={"min_track_frames": 10})
    assert len(verdicts) == 1
    assert verdicts[0].track.start == 8


# --------------------------------------------------------------------------
# Ruling 49: pin the criterion's shape, not just one passing case.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_frames, drift_sky", [(10, 1134.0), (30, 3654.0), (54, 6678.0)])
def test_pixel_lock_is_sequence_length_independent(n_frames, drift_sky):
    """A static source reads locked and a moving one does not, at every length.

    Under the scaled form the threshold was ``0.05 * drift_sky`` = 56.7 / 182.7
    / 333.9 px at these three lengths, so the *moving* source (16.2 / 52.2 /
    95.4 px of detector span) read locked too. This is the test that makes
    reintroducing any drift_sky scaling impossible.
    """
    static_det, static_sky = static_det_moving_sky(n_frames, step_det=0.0)
    static = make_track(static_det, static_sky)
    assert static.drift_sky_px == pytest.approx(drift_sky)
    assert static.pixel_locked is True

    moving_det, moving_sky = static_det_moving_sky(n_frames, step_det=1.8)
    moving = make_track(moving_det, moving_sky)
    assert moving.drift_sky_px == pytest.approx(drift_sky)
    assert moving.pixel_locked is False


@pytest.mark.parametrize("span, norm", [(0.90, 1.2728), (0.80, 1.1314)])
def test_diagonal_jitter_is_not_pixel_locked(span, norm):
    """Equal jitter on both axes: each axis under 1.0 px, the norm above it.

    ``np.hypot(*span) >= max(span)`` always, so ``norm < thr`` implies
    ``np.all(span < thr)`` — the norm is the strictly *tighter* guard and marks
    strictly fewer sources locked. These two points are the only shape where the
    two forms disagree, and they are exactly the shape of a hot pixel jittering
    equally on both axes. The tighter guard is the safe direction because a
    missed hot pixel has ``v_det == 0`` and is therefore reported as the target
    at infinite contrast. This assertion is the one that dies if someone
    substitutes ``np.all(span < thr)``.
    """
    xy_det = np.array([[244.0, 3174.0], [244.0 + span, 3174.0 + span]])
    xy_sky = np.array([[244.0, 3174.0], [370.0, 3174.0]])
    track = make_track(xy_det, xy_sky)
    measured = np.abs(xy_det.max(axis=0) - xy_det.min(axis=0))
    assert np.hypot(*measured) == pytest.approx(norm, abs=1e-4)
    assert np.all(measured < 1.0), "each axis must be under the threshold for this to bite"
    assert track.pixel_locked is False


def test_one_axis_motion_is_not_pixel_locked():
    """Plain regression on the one-axis case: span (19.80, 0.00), 12 frames.

    Per Ruling 257 this does NOT distinguish the norm from ``np.all`` — measured,
    ``np.all([19.80, 0.00] < 1.0)`` is False too, so both forms read False here.
    Kept because it is the slowest real target span in this task and the number
    the 1.0 px threshold is justified against (19.8x of margin).
    """
    n_frames = 12
    i = np.arange(n_frames, dtype=np.float64)
    xy_det = np.column_stack((2000.0 + 1.8 * i, np.full(n_frames, 1500.0)))
    xy_sky = np.column_stack((2000.0 + 126.0 * i, np.full(n_frames, 1500.0)))
    track = make_track(xy_det, xy_sky)
    span = np.abs(xy_det.max(axis=0) - xy_det.min(axis=0))
    assert span == pytest.approx([19.80, 0.00])
    assert np.hypot(*span) == pytest.approx(19.80)
    assert track.pixel_locked is False


def test_jittering_hot_pixel_at_half_pixel_boundary_is_locked():
    """A hot pixel straddling x = 244.50 flips its rounded integer every frame.

    The spec's literal 「整数像素位置在全序列不变」 therefore reads False here, and
    letting this source through would report a sensor defect as the target at
    infinite contrast (its ``v_det`` is ~0.02–0.25 px/frame, well under
    ``v_det_max=8.0``, and its ``v_sky`` ~126 clears ``v_sky_min=30``). That is
    why the implementation uses a span threshold instead of ``np.round``
    equality. Measured, the rounded x takes both 244 and 245 at every jitter
    level down to sigma = 0.007 px.

    The span norm itself is a noise realisation (~0.42 at sigma=0.08 for this
    seed, drifting with the RNG), so only the boolean and a wide bound are
    asserted.
    """
    rng = np.random.default_rng(7)
    n_frames = 30
    x = 244.50 + rng.normal(0.0, 0.08, size=n_frames)
    y = 3174.50 + rng.normal(0.0, 0.08, size=n_frames)
    xy_det = np.column_stack((x, y))
    i = np.arange(n_frames, dtype=np.float64)
    xy_sky = np.column_stack((244.5 + 126.0 * i, np.full(n_frames, 3174.5)))
    track = make_track(xy_det, xy_sky)

    assert set(np.round(x).astype(np.int64)) == {244, 245}
    span_norm = float(np.hypot(*np.abs(xy_det.max(axis=0) - xy_det.min(axis=0))))
    assert 0.0 < span_norm < 1.0
    assert track.pixel_locked is True


def test_exact_hot_pixel_is_locked():
    """The closed-form case: zero detector span, norm exactly 0.000."""
    n_frames = 30
    xy_det = np.tile([244.0, 3174.0], (n_frames, 1)).astype(np.float64)
    i = np.arange(n_frames, dtype=np.float64)
    xy_sky = np.column_stack((244.0 + 126.0 * i, np.full(n_frames, 3174.0)))
    track = make_track(xy_det, xy_sky)
    assert np.hypot(*np.abs(xy_det.max(axis=0) - xy_det.min(axis=0))) == 0.0
    assert track.pixel_locked is True


# --------------------------------------------------------------------------
# Ruling 50: contrast = inf for a zero-v_det track, and JSON safety.
# --------------------------------------------------------------------------


def test_static_track_has_infinite_contrast_but_is_rejected():
    """A hot pixel scores the highest possible contrast; only the lock guard stops it."""
    n_frames = 30
    xy_det = np.tile([244.0, 3174.0], (n_frames, 1)).astype(np.float64)
    i = np.arange(n_frames, dtype=np.float64)
    xy_sky = np.column_stack((244.0 + 126.0 * i, np.full(n_frames, 3174.0)))
    verdict = classify(make_track(xy_det, xy_sky))

    assert verdict.v_det_px == 0.0
    assert verdict.contrast == float("inf")
    assert not verdict.is_target
    assert verdict.pixel_locked
    assert "像素锁定" in " ".join(verdict.reasons)


def test_to_dict_is_valid_json_for_infinite_contrast():
    """``json.dumps`` renders bare ``Infinity``, which is not valid JSON.

    ``to_dict`` coerces a non-finite ``stationarity_contrast`` to ``None`` so the
    Task 30 report parses under any strict JSON reader.
    """
    n_frames = 30
    xy_det = np.tile([244.0, 3174.0], (n_frames, 1)).astype(np.float64)
    i = np.arange(n_frames, dtype=np.float64)
    xy_sky = np.column_stack((244.0 + 126.0 * i, np.full(n_frames, 3174.0)))
    payload = classify(make_track(xy_det, xy_sky)).to_dict()

    assert payload["stationarity_contrast"] is None
    text = json.dumps(payload, allow_nan=False)
    assert "Infinity" not in text
    assert json.loads(text)["stationarity_contrast"] is None


def test_to_dict_keeps_finite_contrast_and_reports_peak_flatness():
    n_frames = 12
    i = np.arange(n_frames, dtype=np.float64)
    xy_det = np.column_stack((2000.0 + 1.8 * i, np.full(n_frames, 1500.0)))
    xy_sky = np.column_stack((2000.0 + 126.0 * i, np.full(n_frames, 1500.0)))
    payload = classify(make_track(xy_det, xy_sky, peak=np.full(n_frames, 900.0))).to_dict()

    assert payload["is_target"] is True
    assert payload["stationarity_contrast"] == pytest.approx(70.0, rel=0.05)
    assert payload["peak_flat_ratio"] == pytest.approx(0.0)
    json.dumps(payload, allow_nan=False)


# --------------------------------------------------------------------------
# Ruling 48a / 261: peak flatness is reported evidence, never a veto.
# --------------------------------------------------------------------------


def test_peak_flat_ratio_closed_form_values():
    """Closed-form peak arrays, including the two escapes that veto would cause.

    A hot pixel beside a bleeding bright star reads 0.6667 and one the moving
    target crosses once reads 0.5556 — both far above the spec's 0.05, so
    AND-ing peak flatness into ``pixel_locked`` would mark both unlocked and
    report them as the target. Hand-built arrays are used because
    ``synthetic_scene``'s ``as_table`` sets ``peak = 300.0`` everywhere, making
    every end-to-end peak_flat_ratio exactly 0.0 for target and hot pixels
    alike — vacuous.
    """
    xy_det = np.tile([244.0, 3174.0], (30, 1)).astype(np.float64)
    xy_sky = np.column_stack((244.0 + 126.0 * np.arange(30.0), np.full(30, 3174.0)))

    def flat(peak):
        return make_track(xy_det, xy_sky, peak=peak).peak_flat_ratio

    assert flat(np.full(30, 900.0)) == pytest.approx(0.0000, abs=1e-4)
    assert flat(900.0 + np.where(np.arange(30) % 2 == 0, 1.0, -1.0)) == pytest.approx(
        0.0022, abs=1e-4
    )
    assert flat(np.linspace(900.0, 918.0, 30)) == pytest.approx(0.0198, abs=1e-4)

    crossing = np.full(30, 900.0)
    crossing[15] = 1400.0
    assert flat(crossing) == pytest.approx(0.5556, abs=1e-4)

    bleeding = np.concatenate([np.full(15, 900.0), np.full(15, 1800.0)])
    assert flat(bleeding) == pytest.approx(0.6667, abs=1e-4)


def test_peak_flat_ratio_guards_zero_median():
    """An all-zero peak array must not divide by zero (Ruling 261)."""
    xy_det = np.tile([244.0, 3174.0], (10, 1)).astype(np.float64)
    xy_sky = np.column_stack((244.0 + 126.0 * np.arange(10.0), np.full(10, 3174.0)))
    track = make_track(xy_det, xy_sky, peak=np.zeros(10))
    assert track.peak_flat_ratio == 0.0


def test_peak_flatness_never_vetoes_the_lock(synthetic_scene):
    """A locked hot pixel stays locked no matter how its peak swings."""
    frames, detections, reg = synthetic_scene
    tracks = build_tracks(detections, frames, reg, radius=8.0, min_frames=10)
    locked = [t for t in tracks if t.pixel_locked]
    assert locked
    t = locked[0]
    t.peak = np.concatenate(
        [np.full(t.length // 2, 900.0), np.full(t.length - t.length // 2, 1800.0)]
    )
    assert t.peak_flat_ratio > 0.5
    assert t.pixel_locked is True
    assert not classify(t).is_target


def test_build_tracks_populates_peak_from_the_source_table(synthetic_scene):
    frames, detections, reg = synthetic_scene
    tracks = build_tracks(detections, frames, reg, radius=8.0, min_frames=10)
    assert tracks
    for t in tracks:
        assert t.peak.shape == (t.length,)
        assert t.peak.dtype == np.float64
        assert np.all(t.peak == 300.0)  # as_table plants a constant peak


# --------------------------------------------------------------------------
# Ruling 52: pin the claimed set and the greedy-linking contract.
# --------------------------------------------------------------------------


def test_no_source_is_claimed_by_two_tracks(synthetic_scene):
    """``claimed`` is what stops every frame's reseed from making fragments.

    Without it the fixture's four tracks become many overlapping ones.
    """
    frames, detections, reg = synthetic_scene
    tracks = build_tracks(detections, frames, reg, radius=8.0, min_frames=10)
    assert tracks

    seen: dict[tuple[int, int], int] = {}
    for n_track, t in enumerate(tracks):
        for n, f in enumerate(t.frames):
            table = detections[f]
            # xy_det is copied verbatim out of the table, so exact equality holds.
            hits = np.flatnonzero(
                (table.x == t.xy_det[n, 0]) & (table.y == t.xy_det[n, 1])
            )
            assert len(hits) >= 1
            key = (f, int(hits[0]))
            assert key not in seen, (
                f"源 {key} 同时出现在轨迹 {seen.get(key)} 和 {n_track}"
            )
            seen[key] = n_track


def test_build_tracks_returns_descending_length_order(two_target_scene):
    """``find_targets`` relies on this ordering for its output order.

    Uses the two-target fixture on purpose: ``synthetic_scene``'s four tracks are
    all exactly length 30, so ``lengths == sorted(lengths, reverse=True)`` holds
    for ascending order too and pins nothing. Here the lengths are
    ``[30, 30, 30, 30, 20, 15]`` from a seeding order of
    ``[30, 30, 30, 30, 15, 20]`` — so this fails both if the sort is reversed and
    if it is removed entirely.
    """
    frames, detections, reg = two_target_scene
    tracks = build_tracks(detections, frames, reg, radius=8.0, min_frames=10)
    lengths = [t.length for t in tracks]
    assert lengths == [30, 30, 30, 30, 20, 15]
    assert lengths != sorted(lengths), "fixture must not be length-degenerate"
    assert lengths == sorted(lengths, reverse=True)


def test_find_targets_orders_hits_by_descending_length(two_target_scene):
    """``hits.sort`` is unpinned on a one-hit fixture; this one yields two.

    Note that *removing* ``hits.sort`` is an equivalent mutant rather than a test
    gap: ``hits`` is a filter of ``tracks``, ``build_tracks`` already returns them
    descending, and a subsequence of a descending list is descending. Reversing
    the sort is not equivalent and this test catches it.
    """
    frames, detections, reg = two_target_scene
    verdicts = find_targets(detections, frames, reg)
    assert [v.track.length for v in verdicts] == [30, 20]
    assert [v.track.start for v in verdicts] == [0, 10]
    assert all(v.is_target for v in verdicts)



# --------------------------------------------------------------------------
# The length-1 guard in _step_median.
# --------------------------------------------------------------------------


def test_single_frame_track_has_zero_speeds_and_stays_json_safe():
    """``_step_median``'s ``len(xy) < 2`` guard is reachable from the public API.

    Without it ``np.median(np.diff(one_row))`` is ``nan`` (plus two
    ``RuntimeWarning``s), and that ``nan`` makes **both** speed criteria silently
    fail (``nan > 8.0`` is False, ``nan < 30.0`` is False), so ``classify``
    appends no speed reason at all. ``to_dict``'s ``v_det_px_per_frame`` /
    ``v_sky_px_per_frame`` have no ``isfinite`` coercion — unlike
    ``stationarity_contrast`` — so ``json.dumps(..., allow_nan=False)`` raises
    ``ValueError``.

    Length-1 tracks are constructible through the public API: measured,
    ``find_targets(..., config={"min_track_frames": 1})`` builds 3984 of them on
    the synthetic scene.

    The hazard is conditional, not realised: a length-1 track's ``max`` and
    ``min`` are the same row, so its span is exactly (0, 0) and its norm exactly
    0, hence it is always ``pixel_locked`` for any ``lock_span_px > 0`` and
    ``find_targets`` — which returns hits only — never emits it. Reaching the
    report needs a consumer that serialises *rejected* verdicts too, and how
    Task 30 consumes this dict is not yet decided.
    """
    track = make_track([[244.0, 3174.0]], [[900.0, 3174.0]], frames=[5])
    assert track.length == 1
    assert track.v_det_px == 0.0
    assert track.v_sky_px == 0.0

    verdict = classify(track)
    assert verdict.v_det_px == 0.0
    assert verdict.v_sky_px == 0.0
    assert not verdict.is_target
    assert verdict.pixel_locked  # span is exactly (0, 0)

    payload = verdict.to_dict()
    assert payload["v_det_px_per_frame"] == 0.0
    assert payload["v_sky_px_per_frame"] == 0.0
    json.dumps(payload, allow_nan=False)


# --------------------------------------------------------------------------
# find_targets must actually read all five config keys (Ruling 264's hole).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, value",
    [
        ("v_det_max_px", 0.5),
        ("v_sky_min_px", 1e9),
        ("lock_span_px", 1e9),
        ("track_radius_px", 0.001),
        ("min_track_frames", 999),
    ],
)
def test_find_targets_reads_each_config_key(synthetic_scene, key, value):
    """Each key, on its own, must be able to drive the one hit to zero.

    ``test_config_default_matches_the_code_default`` pins that the YAML carries
    the right key names; nothing pinned that ``find_targets`` reads them. Ruling
    264 predicted exactly this hole — misspelling any of the five silently falls
    back to the default and every test that calls ``classify`` directly still
    passes. Parametrised rather than bundled into one test on purpose: a single
    test asserting all five at once would still pass if only some keys were
    fixed.
    """
    frames, detections, reg = synthetic_scene
    assert len(find_targets(detections, frames, reg)) == 1
    assert find_targets(detections, frames, reg, config={key: value}) == []


# --------------------------------------------------------------------------
# Ruling 53: negative tests at the two speed thresholds.
# --------------------------------------------------------------------------


def speed_track(v_det: float, v_sky: float, n_frames: int = 12) -> Track:
    i = np.arange(n_frames, dtype=np.float64)
    xy_det = np.column_stack((2000.0 + v_det * i, np.full(n_frames, 1500.0)))
    xy_sky = np.column_stack((2000.0 + v_sky * i, np.full(n_frames, 1500.0)))
    return make_track(xy_det, xy_sky)


def test_classify_rejects_a_track_above_v_det_max():
    track = speed_track(10.0, 126.0)
    assert track.v_det_px == pytest.approx(10.0)
    verdict = classify(track)
    assert not verdict.is_target
    assert any("探测器系速度" in r for r in verdict.reasons)


def test_classify_rejects_a_track_below_v_sky_min():
    track = speed_track(1.8, 20.0)
    assert track.v_sky_px == pytest.approx(20.0)
    verdict = classify(track)
    assert not verdict.is_target
    assert any("天球系速度" in r for r in verdict.reasons)


def test_classify_collects_both_speed_reasons_without_short_circuiting():
    verdict = classify(speed_track(10.0, 20.0))
    assert not verdict.is_target
    assert not verdict.pixel_locked
    assert len(verdict.reasons) == 2
    assert any("探测器系速度" in r for r in verdict.reasons)
    assert any("天球系速度" in r for r in verdict.reasons)


def test_config_default_matches_the_code_default(cfg):
    """`target.lock_span_px` is the renamed key; the old `lock_ratio` must be gone."""
    target = cfg["target"]
    assert "lock_ratio" not in target
    assert target["lock_span_px"] == 1.0
    assert target["v_det_max_px"] == 8.0
    assert target["v_sky_min_px"] == 30.0
    assert Track.lock_span_px == 1.0


def test_dataset_b_recovers_the_known_target(dataset_b_dir):
    """端到端复现 spec 基线：f17-f70 共 54 帧，探测器系漂移 95.8 px。"""
    from src.calib.hotpixel import build_from_sequence
    from src.calib.segment import tracking_segment
    from src.dataio.fits_loader import FrameSequence
    from src.detect.segmentation import detect_sequence

    seq = FrameSequence.from_directory(dataset_b_dir)
    seg = tracking_segment(seq.headers)
    frames = list(range(seg.start, seg.end + 1))
    hpm = build_from_sequence(seq, frames[:30])
    dets = detect_sequence(seq, frames, n_sigma=4.0, hot_clusters=hpm.clusters)
    reg = register_sequence(dets, frames, reference=seg.start)

    verdicts = find_targets(dets, frames, reg)
    assert len(verdicts) == 1
    trk = verdicts[0].track
    assert trk.length >= 50
    assert trk.start <= 18 and trk.end >= 69
    assert trk.xy_det[0] == pytest.approx([2271.6, 1958.8], abs=3.0)
    assert trk.xy_det[-1] == pytest.approx([2365.8, 1976.3], abs=3.0)
    assert trk.drift_det_px == pytest.approx(95.8, rel=0.1)
    assert verdicts[0].contrast > 20.0
