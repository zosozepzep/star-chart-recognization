from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from src.validate.truth import (
    TruthReport,
    TruthTable,
    compare,
    load_truth,
    match_frames,
    write_report,
)

SAMPLE = """2026 07 21 17 26 44.2670 209.23627 012.35454 008.44
2026 07 21 17 26 45.2750 209.10971 012.52617 008.24
2026 07 21 17 26 46.2830 208.98229 012.69737 008.31
"""

REPO_ROOT = Path(__file__).resolve().parents[2]
JUDGEMENT_PACKAGES = ("detect", "register", "target")
STUB_T0 = "2026-07-21T17:26:44.000"


class StubSequence:
    """Minimal stand-in for ``FrameSequence``.

    ``match_frames`` and ``compare`` touch only ``dataset_id`` and ``times()``,
    so a stub suffices — and it is the only way to build a frame clock that is
    *offset* from the truth clock, which dataset B's is not (see
    ``test_match_frames_real_dat_clock_is_exact``).
    """

    def __init__(self, n: int, cadence_s: float = 1.0, dataset_id: str = "stub"):
        self.dataset_id = dataset_id
        self._times = Time(STUB_T0, scale="utc") + TimeDelta(
            np.arange(n, dtype=np.float64) * cadence_s, format="sec"
        )

    def __len__(self) -> int:
        return len(self._times)

    def times(self) -> Time:
        return self._times


class StubTrack:
    """Minimal stand-in for ``Track`` — ``compare`` reads only these three fields."""

    def __init__(self, frames, xy_sky, flux):
        self.frames = list(frames)
        self.xy_sky = np.asarray(xy_sky, dtype=np.float64).reshape(-1, 2)
        self.flux = np.asarray(flux, dtype=np.float64)


def truth_at_offsets(offsets_s, mags=None) -> TruthTable:
    """A TruthTable whose times are ``STUB_T0 + offsets_s``, on a moving target.

    Every row gets a *distinct* RA/Dec on purpose (Ruling 345): duplicating the
    coordinates alongside the times makes the angular step go to zero in the
    same position as the pixel step, so their ratio agrees by construction and
    the fixture proves nothing.
    """
    n = len(offsets_s)
    return TruthTable(
        path=Path("synthetic.DAT"),
        time=Time(STUB_T0, scale="utc")
        + TimeDelta(np.asarray(offsets_s, dtype=np.float64), format="sec"),
        ra_deg=209.0 - np.arange(n, dtype=np.float64) * 0.1,
        dec_deg=12.0 + np.arange(n, dtype=np.float64) * 0.1,
        mag=np.full(n, 8.0) if mags is None else np.asarray(mags, dtype=np.float64),
    )


def straight_track(n: int = 6) -> StubTrack:
    """A track moving 100 px per frame in x, constant flux."""
    return StubTrack(
        frames=list(range(n)),
        xy_sky=np.stack([np.arange(n, dtype=np.float64) * 100.0, np.zeros(n)], axis=1),
        flux=np.full(n, 1000.0),
    )


@pytest.fixture()
def sample_dat(tmp_path):
    p = tmp_path / "20260721172644_6002_060385_0003.DAT"
    p.write_text(SAMPLE, encoding="ascii")
    return p


def test_load_truth_parses_columns(sample_dat):
    t = load_truth(sample_dat)
    assert len(t) == 3
    assert t.ra_deg[0] == pytest.approx(209.23627)
    assert t.dec_deg[0] == pytest.approx(12.35454)
    assert t.mag[0] == pytest.approx(8.44)


def test_load_truth_parses_utc_times(sample_dat):
    t = load_truth(sample_dat)
    assert t.time.scale == "utc"
    assert t.time[0].isot.startswith("2026-07-21T17:26:44.267")
    dt = np.diff(t.time.unix)
    assert dt == pytest.approx([1.008, 1.008], abs=0.01)


def test_load_truth_marks_zero_magnitude_as_missing(tmp_path):
    """Driven by a synthetic ``.DAT`` on purpose (Ruling 268).

    No row of the real file has ``mag <= 0`` (measured: 55/55 finite, range
    7.190-8.810), so against the real file this test would pass vacuously and
    would not notice the loss of the ``mag[mag <= 0] = nan`` line.
    """
    p = tmp_path / "z.DAT"
    p.write_text("2026 07 21 17 26 44.2670 209.23627 012.35454 000\n", encoding="ascii")
    t = load_truth(p)
    assert np.isnan(t.mag[0])


def test_load_truth_rejects_wrong_column_count(tmp_path):
    p = tmp_path / "bad.DAT"
    p.write_text("2026 07 21 17 26 44.2670 209.23627\n", encoding="ascii")
    with pytest.raises(ValueError):
        load_truth(p)


def test_load_truth_rejects_empty_file_with_honest_message(tmp_path):
    """Ruling 352：空文件的诊断必须说"空"，不能说"实际 1 列"。

    ``np.loadtxt(..., ndmin=2)`` gives an empty file ``shape (0, 1)`` (measured),
    so without the ``raw.size == 0`` branch the column guard reports "实际 1 列"
    for a file that has no columns at all. The assertion below pins the *message*,
    not merely the exception type — the pre-fix code raised ``ValueError`` too.
    """
    p = tmp_path / "empty.DAT"
    p.write_text("", encoding="ascii")
    with pytest.raises(ValueError, match="为空或不含数据行"):
        load_truth(p)

    # A whitespace-only file takes the same path (measured shape (0, 1)).
    q = tmp_path / "blank.DAT"
    q.write_text("\n   \n\n", encoding="ascii")
    with pytest.raises(ValueError, match="为空或不含数据行"):
        load_truth(q)


def test_real_dat_has_55_rows(dataset_b_dir):
    from src.dataio.fits_loader import FrameSequence

    seq = FrameSequence.from_directory(dataset_b_dir)
    t = load_truth(seq.truth_path)
    assert len(t) == 55
    assert np.isfinite(t.mag).all()
    assert t.mag.min() > 7.0 and t.mag.max() < 10.0


def test_match_frames_maps_dat_to_f16_f70(dataset_b_dir):
    """真值 55 行必须精确对应跟踪段 f16-f70，时间误差应为 0。"""
    from src.calib.segment import tracking_segment
    from src.dataio.fits_loader import FrameSequence

    seq = FrameSequence.from_directory(dataset_b_dir)
    truth = load_truth(seq.truth_path)
    seg = tracking_segment(seq.headers)
    frames, rows = match_frames(truth, seq, list(range(len(seq))))
    assert frames.tolist() == list(range(seg.start, seg.end + 1))
    assert rows.tolist() == list(range(55))
    # Ruling 55: no two truth rows may claim the same frame. This pins that
    # dataset B is clean, which is what makes compare's duplicate-frame guard
    # never firing here a measured fact rather than an untested assumption.
    assert len(set(frames.tolist())) == len(frames)
    dt = np.abs(seq.times().unix[frames] - truth.time.unix[rows])
    assert dt.max() < 1e-3


def test_match_frames_real_dat_clock_is_exact(dataset_b_dir):
    """真值时钟与帧时钟逐行相等（作为时刻，非文本）：dt 恰为 0，不是"很小"。

    Because ``dt.max()`` is exactly 0 on this file, no value of ``max_dt_s``
    filters anything here — this file cannot exercise the tolerance at all,
    which is why ``test_match_frames_respects_tolerance_synthetic`` exists.
    """
    from src.dataio.fits_loader import FrameSequence

    seq = FrameSequence.from_directory(dataset_b_dir)
    truth = load_truth(seq.truth_path)
    frames, rows = match_frames(truth, seq, list(range(len(seq))))
    assert len(frames) == 55
    dt = np.abs(seq.times().unix[frames] - truth.time.unix[rows])
    assert dt.max() == 0.0


def test_match_frames_respects_tolerance_synthetic():
    """Ruling 343: offsets 0.0 / 0.2 / 0.45 s against a 1.0 s-cadence stub.

    Every offset is under half a cadence, so ``argmin`` snaps each row to *its
    own* frame — asserted row by row below, not merely by the final count. An
    offset at or above half a cadence is absorbed by the neighbouring frame and
    never reaches the tolerance branch: 0.8 s snaps to the *next* frame with
    dt = 0.2, which is why Ruling 54's prescribed 0.0/0.2/0.8 fixture measured
    3 of 3 at ``max_dt_s=0.5`` instead of the intended 2 of 3.

    Expected counts 3 / 2 / 1 at tolerances 0.5 / 0.3 / 0.05. A ``match_frames``
    that ignores ``max_dt_s`` returns 3 at every tier, so 0.3 and 0.05 go red.
    """
    seq = StubSequence(6, cadence_s=1.0)
    truth = truth_at_offsets([0.0, 1.2, 2.45])

    frame_unix = seq.times().unix
    for row, (want_frame, want_dt) in enumerate([(0, 0.0), (1, 0.2), (2, 0.45)]):
        k = int(np.argmin(np.abs(frame_unix - truth.time.unix[row])))
        assert k == want_frame, f"row {row} snapped to frame {k}, expected {want_frame}"
        assert abs(frame_unix[k] - truth.time.unix[row]) == pytest.approx(
            want_dt, abs=1e-6
        )

    expected = {
        0.5: ([0, 1, 2], [0, 1, 2]),
        0.3: ([0, 1], [0, 1]),
        0.05: ([0], [0]),
    }
    for tol, (want_frames, want_rows) in expected.items():
        frames, rows = match_frames(truth, seq, list(range(6)), max_dt_s=tol)
        assert frames.tolist() == want_frames, f"max_dt_s={tol}"
        assert rows.tolist() == want_rows, f"max_dt_s={tol}"


def test_compare_rejects_duplicate_frames():
    """Ruling 344：两行真值撞到同一帧时必须报错，不得静默给出偏置的比例尺。

    The harm is *not* the "three numbers disagree" one Ruling 344 first recorded.
    Since Ruling 348 put the ``good`` mask on all three means, the zero pixel
    step a duplicate frame produces is dropped from all of them together, and
    ``plate_scale`` measurably agrees with ``angular_step / pixel_step`` again
    (measured 0.0% apart on this very fixture, versus 33% pre-fix). The reason to
    refuse is the underlying one: duplicate frames mean the truth clock is offset
    from the frame clock, so the angular and pixel displacements of a step no
    longer span the same time interval and the scale is silently biased.
    """
    seq = StubSequence(6, cadence_s=1.0)
    truth = truth_at_offsets([0.0, 0.1, 1.0, 2.0])  # rows 0 and 1 both snap to frame 0
    track = straight_track()

    frames, _ = match_frames(truth, seq, track.frames)
    assert frames.tolist() == [0, 0, 1, 2], "fixture must actually produce a duplicate"

    with pytest.raises(ValueError, match="重复"):
        compare(track, truth, seq, None)


def test_compare_masks_all_three_means_consistently():
    """Ruling 348：一个零像素步长必须从三个均值里一起剔除，否则三数字互不自洽。

    Distinct frames [0,1,2,3] on purpose — Ruling 344's duplicate-frame guard
    must *not* fire, because this is the path it cannot cover. The target sits
    still between f0 and f1, so the pixel steps are ``[0.0, 100.0, 100.0]``.

    Both expected values are analytic, not observed:

    * ``pixel_step_px == 100.0`` exactly — the two non-degenerate steps of the
      fixture's own ``xy_sky`` are 100 px each, by construction.
    * self-consistency ``plate_scale == angular_step / pixel_step`` holds by
      construction once the three statistics share one mask; the only residual
      is "mean of ratios vs ratio of means", which is 0 here because every
      surviving pixel step is identical.

    Teeth: mutating ``good = pixel > 1e-6`` to ``> -1.0`` keeps the zero step, so
    ``pixel_step_px`` becomes 66.6667 and ``plate_scale`` becomes ``inf``
    (measured).
    """
    seq = StubSequence(6, cadence_s=1.0)
    truth = truth_at_offsets([0.0, 1.0, 2.0, 3.0])
    track = StubTrack(
        frames=[0, 1, 2, 3],
        xy_sky=[[0.0, 0.0], [0.0, 0.0], [100.0, 0.0], [200.0, 0.0]],
        flux=[1000.0, 1100.0, 1200.0, 1300.0],
    )

    frames, _ = match_frames(truth, seq, track.frames)
    assert len(set(frames.tolist())) == len(frames), "fixture must have distinct frames"

    rep = compare(track, truth, seq)
    assert rep.available is True
    assert rep.pixel_step_px == pytest.approx(100.0, rel=1e-12)
    assert rep.plate_scale_arcsec_px == pytest.approx(
        rep.angular_step_arcsec / rep.pixel_step_px, rel=1e-3
    )


def test_compare_refuses_a_fully_stationary_target():
    """Ruling 346/348：像素步长全为零时不得报 nan，必须 available=False。

    ``np.mean([])`` is nan (measured, with ``RuntimeWarning: Mean of empty
    slice``), and nan renders as a bare ``NaN`` token that is not valid JSON —
    so the pre-fix code shipped a corrupt deliverable with ``available=True``.
    """
    seq = StubSequence(6, cadence_s=1.0)
    truth = truth_at_offsets([0.0, 1.0, 2.0, 3.0])
    track = StubTrack(frames=[0, 1, 2, 3], xy_sky=[[5.0, 5.0]] * 4, flux=[1000.0] * 4)

    rep = compare(track, truth, seq)
    assert rep.available is False
    assert rep.n_matched == 4
    assert rep.plate_scale_arcsec_px is None
    assert "像素步长全为零" in rep.note


@pytest.mark.parametrize(
    "offsets_s, want_available, want_n_matched",
    [
        ([0.0], False, 0),
        ([0.0, 1.0], False, 0),
        ([0.0, 1.0, 2.0], True, 3),
    ],
)
def test_compare_needs_at_least_three_matched_points(
    offsets_s, want_available, want_n_matched
):
    """Ruling 348：`len(frames) < 3` 的边界精确在 3，两侧都测。

    Without this the mutation ``if len(frames) < 3:`` -> ``if False:`` survives:
    every other ``compare`` test matches >= 4 rows or is stopped earlier by the
    duplicate-frame guard.

    Note the mutant's measured behaviour changed with Ruling 348's mask fix, so
    the two rejecting tiers fail for *different* reasons and both are needed.
    Pre-fix the mutant returned ``plate_scale = nan`` with ``available=True`` at
    1 matched point; now the stationary-target guard catches that tier first
    (1 point means 0 pixel steps, so ``good.any()`` is False) and reports
    ``n_matched=1``, which is why ``want_n_matched == 0`` is asserted rather than
    only ``available``. The 2-point tier is the one that still reaches the
    arithmetic: a single non-degenerate step gets through and the mutant ships
    ``available=True`` with a plate scale derived from that one step (measured
    5.035849 on this fixture).

    ``want_n_matched == 0`` on the rejecting tiers pins that the *minimum-points*
    early return is the one that fired — it leaves ``n_matched`` at its dataclass
    default, whereas the stationary-target return reports the real count.
    """
    seq = StubSequence(6, cadence_s=1.0)
    rep = compare(straight_track(), truth_at_offsets(offsets_s), seq)

    assert rep.available is want_available
    assert rep.n_matched == want_n_matched
    if want_available:
        assert np.isfinite(rep.plate_scale_arcsec_px)
    else:
        assert rep.plate_scale_arcsec_px is None
        assert "不足以定量比对" in rep.note


def test_angular_step_places_cos_factor_on_ra():
    """Ruling 347：cos(dec) 属于赤经差。dec 恒为 60 时纯合成地区分对错。

    180.0 是**解析值**：0.1 deg/step of pure RA motion at dec == 60 gives
    ``0.1 * cos(60 deg) * 3600 = 180.000000`` arcsec/step, and ``ddec == 0`` so
    the hypot is that alone. It is derived, not back-filled from an observation.

    On dataset B this physical error is invisible: dec spans 12.526-21.527, so
    cos(dec) in [0.9302, 0.9762] and the baseline test survives the mutation with
    0.000443 of margin left. Here the mutation ``hypot(dra, ddec*cos(dec))``
    yields 360.0 (dec is constant, so its cos factor multiplies a zero and the
    un-compressed RA difference is reported in full) — a factor of 2 off.
    """
    seq = StubSequence(6, cadence_s=1.0)
    truth = TruthTable(
        path=Path("synthetic.DAT"),
        time=Time(STUB_T0, scale="utc") + TimeDelta(np.arange(4.0), format="sec"),
        ra_deg=209.0 - np.arange(4, dtype=np.float64) * 0.1,
        dec_deg=np.full(4, 60.0),
        mag=np.array([8.0, 8.2, 8.4, 8.6]),
    )

    rep = compare(straight_track(), truth, seq)
    assert rep.angular_step_arcsec == pytest.approx(180.0, rel=1e-3)


def test_compare_reports_none_correlation_when_variance_is_zero(tmp_path):
    """Ruling 346：星等或 flux 恒定时相关系数无定义，报 None 而不是 nan。

    ``truth_at_offsets`` defaults to mag == 8.0 for every row, so this shape is
    one line away from the tests that already exist. ``np.corrcoef`` divides by a
    zero standard deviation and returns nan (measured); nan then renders as a
    bare ``NaN`` token in the written report. ``zero_point`` and its scatter need
    no variance and stay finite.
    """
    seq = StubSequence(6, cadence_s=1.0)
    truth = truth_at_offsets([0.0, 1.0, 2.0, 3.0])  # mag constant 8.0
    rep = compare(straight_track(), truth, seq)

    assert rep.mag_correlation is None
    assert np.isfinite(rep.zero_point)
    assert np.isfinite(rep.zero_point_std)
    # allow_nan=False on the bytes that write_report actually writes.
    path = write_report(rep, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["mag_correlation"] is None
    assert "NaN" not in path.read_text(encoding="utf-8")

    # Constant flux is the mirror case: the truth magnitudes vary, the
    # instrumental ones do not.
    varying = truth_at_offsets([0.0, 1.0, 2.0, 3.0], mags=[8.0, 8.2, 8.4, 8.6])
    rep2 = compare(straight_track(), varying, seq)  # straight_track flux is constant
    assert rep2.mag_correlation is None
    assert np.isfinite(rep2.zero_point)


def test_write_report_refuses_to_write_non_json_nan(tmp_path):
    """Ruling 346：`nan` 渲染成裸 `NaN` token 不是合法 JSON，必须响亮抛错。

    ``json.dumps`` defaults to ``allow_nan=True``; ``jq`` / ``JSON.parse`` / Go's
    ``encoding/json`` all reject the result, while Python's own ``json.loads``
    accepts it — which is exactly why ``test_write_report_creates_json`` could
    not catch this. This test pins the loud-failure behaviour itself, on the
    dataclass directly, so it survives any future change to ``compare``'s guards.
    """
    rep = TruthReport(
        dataset_id="demo",
        available=True,
        n_matched=10,
        plate_scale_arcsec_px=float("nan"),
        note="人造的 nan",
    )
    with pytest.raises(ValueError):
        write_report(rep, tmp_path)
    assert not (tmp_path / "truth_report.json").exists(), "抛错时不得留下损坏的文件"


def test_compare_without_magnitudes_degrades_to_geometry_only():
    """Ruling 58：星等全缺测时降级为纯几何比对，而不是抛错或把 nan 报出去。"""
    seq = StubSequence(6, cadence_s=1.0)
    truth = truth_at_offsets([0.0, 1.0, 2.0, 3.0], mags=[np.nan] * 4)
    rep = compare(straight_track(), truth, seq, None)

    assert rep.available is True
    assert rep.n_matched == 4
    assert rep.mag_correlation is None
    assert rep.zero_point is None
    assert rep.zero_point_std is None
    assert np.isfinite(rep.plate_scale_arcsec_px)
    data = json.loads(json.dumps(rep.to_dict(), allow_nan=False))
    assert data["mag_correlation"] is None
    assert data["zero_point"] is None
    assert data["zero_point_std"] is None


def test_judgement_chain_does_not_import_truth():
    """真值隔离：detect/register/target 三个包不得引用本模块。

    判决链一旦引用真值就变成"用真值算出真值"，答辩中是致命的原理性质疑。
    Task 13 的 ``tests/test_architecture.py`` 会用完整导入图再查一遍；本测试
    只守这一个方向，因为它是整条判决链唯一不可回退的约束。
    """
    offenders: list[str] = []
    for package in JUDGEMENT_PACKAGES:
        for py in sorted((REPO_ROOT / "src" / package).rglob("*.py")):
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if "validate.truth" in name or name in {"src.validate", "validate"}:
                        rel = py.relative_to(REPO_ROOT).as_posix()
                        offenders.append(f"{rel}:{node.lineno} imports {name}")

    assert offenders == [], "真值隔离被破坏：" + "; ".join(offenders)


def test_compare_reproduces_measured_baseline(dataset_b_dir):
    """比例尺 6.179 +- 0.013 ''/px，星等相关 r=0.949，零点 15.335 +- 0.218。

    Slowest test in the suite (~103 s; ``detect_sequence`` over 55 frames is
    ~91 s of it). It is the project's only end-to-end quantitative gate, so it
    runs unskipped and at full frame count.
    """
    from src.calib.hotpixel import build_from_sequence
    from src.calib.segment import tracking_segment
    from src.dataio.fits_loader import FrameSequence
    from src.detect.segmentation import detect_sequence
    from src.register.solver import register_sequence
    from src.target.dual_frame import find_targets

    seq = FrameSequence.from_directory(dataset_b_dir)
    seg = tracking_segment(seq.headers)
    frames = list(range(seg.start, seg.end + 1))
    hpm = build_from_sequence(seq, frames[:30])
    dets = detect_sequence(seq, frames, n_sigma=4.0, hot_clusters=hpm.clusters)
    reg = register_sequence(dets, frames, reference=seg.start)
    track = find_targets(dets, frames, reg)[0].track

    rep = compare(track, load_truth(seq.truth_path), seq, reg)
    assert rep.available
    assert rep.n_matched >= 50
    assert rep.dt_max_s < 1e-3
    assert rep.plate_scale_arcsec_px == pytest.approx(6.179, abs=0.05)
    assert rep.plate_scale_std < 0.05
    assert rep.angular_step_arcsec == pytest.approx(776.9, rel=0.02)
    assert rep.pixel_step_px == pytest.approx(125.73, rel=0.02)
    assert rep.mag_correlation == pytest.approx(0.949, abs=0.03)
    assert rep.zero_point == pytest.approx(15.335, abs=0.1)
    assert rep.zero_point_std == pytest.approx(0.218, abs=0.1)


def test_write_report_creates_json(tmp_path):
    rep = TruthReport(
        dataset_id="demo",
        available=True,
        n_matched=55,
        dt_max_s=0.0,
        plate_scale_arcsec_px=6.179,
        plate_scale_std=0.013,
        angular_step_arcsec=776.9,
        pixel_step_px=125.73,
        mag_correlation=0.949,
        zero_point=15.335,
        zero_point_std=0.218,
        note="",
    )
    path = write_report(rep, tmp_path)
    assert path.name == "truth_report.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["available"] is True
    assert data["plate_scale_arcsec_px"] == pytest.approx(6.179)


def test_write_report_when_no_truth_records_reason(tmp_path):
    rep = TruthReport(dataset_id="A", available=False, note="数据集无 .DAT，无真值可比对")
    path = write_report(rep, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["available"] is False
    assert "无真值" in data["note"]
    assert data["plate_scale_arcsec_px"] is None
