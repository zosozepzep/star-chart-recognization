import argparse
import csv
import json
from types import SimpleNamespace

import numpy as np

from src import cli
from src.detect.segmentation import SourceTable
from src.pipeline import DetectionRun
from src.register.solver import RegistrationResult


def test_no_targets_or_stars_exports_empty_tables_without_reading_truth(tmp_path, monkeypatch):
    frames = list(range(6))
    registration = RegistrationResult(0, frames, {f: np.eye(3) for f in frames})
    run = DetectionRun(frames, 0, {}, {}, {f: SourceTable.empty(f) for f in frames}, 0, registration, [])
    seq = SimpleNamespace(dataset_id="empty-scene", headers=[SimpleNamespace(exposure_s=1, path=tmp_path / "frame.fits")])
    class Sequence:
        headers = seq.headers
        dataset_id = seq.dataset_id
        def __len__(self):
            return 6
    monkeypatch.setattr(cli.FrameSequence, "from_directory", lambda _: Sequence())
    monkeypatch.setattr(cli, "analyze_sequence", lambda *a, **kw: run)
    def forbidden(*a):
        raise AssertionError("Default inference must not load truth")
    monkeypatch.setattr(cli, "calibrate_after_detection", forbidden)
    out = tmp_path / "results"
    args = argparse.Namespace(config=None, input=tmp_path, output=out, no_plot=True,
                              reference_frame=0, frames=None, calibrate_from_truth=False)
    cli.run_analysis(args)
    result = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert result["target_count"] == 0
    assert result["star_catalog"]["faintest_confirmed_star"] is None
    assert result["star_catalog"]["calibration"] is None
    assert result["registration"]["rms_max_px"] is None
    for name in ["stars.csv", "targets.csv"]:
        with (out / name).open(encoding="utf-8-sig") as fh:
            assert list(csv.DictReader(fh)) == []


def test_existing_results_are_not_overwritten(tmp_path):
    import pytest
    existing = tmp_path / "report.json"
    existing.write_text("preserve", encoding="utf-8")
    args = argparse.Namespace(config=None, output=tmp_path)
    with pytest.raises(ValueError, match="非空"):
        cli.run_analysis(args)
    assert existing.read_text(encoding="utf-8") == "preserve"
