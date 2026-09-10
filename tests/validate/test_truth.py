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
    """Ruling 344：两行真值撞到同一帧时必须报错，不得静默给出三个互不自洽的数。

    ``good = pixel > 1e-6`` protects ``plate_scale_arcsec_px`` only; the other
    two means keep the zero step, so the report would ship
    ``plate_scale != angular_step / pixel_step`` (measured 33% apart) with
    ``available=True`` and no note.
    """
    seq = StubSequence(6, cadence_s=1.0)
    truth = truth_at_offsets([0.0, 0.1, 1.0, 2.0])  # rows 0 and 1 both snap to frame 0
    track = straight_track()

    frames, _ = match_frames(truth, seq, track.frames)
    assert frames.tolist() == [0, 0, 1, 2], "fixture must actually produce a duplicate"

    with pytest.raises(ValueError, match="重复"):
        compare(track, truth, seq, None)


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
