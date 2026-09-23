from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.star_catalog import confirmation_hits, star_catalog
from src.astrometry.photometry import ZeroPoint
from src.detect.segmentation import SourceTable
from src.pipeline import DetectionRun
from src.register.solver import RegistrationResult


def table(frame, xy, flux):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    n = len(xy)
    return SourceTable(frame, xy[:, 0], xy[:, 1], np.asarray(flux), np.ones(n), np.ones(n), np.ones(n, dtype=int))


def test_faintest_is_a_confirmed_row_not_noise_or_percentile():
    tables = {f: table(f, [[10, 10], [20, 20]], [100, 10]) for f in range(6)}
    tables[2] = table(2, [[10, 10], [20, 20], [30, 30], [40, 40]], [100, 10, 0.1, -10])
    reg = RegistrationResult(0, list(tables), {f: np.eye(3) for f in tables})
    report = star_catalog(tables, reg, 2, min_hits=4, radius=1, exposure_s=1,
                          zero_point=ZeroPoint(20, 0.2, 10, exposure_s=1))
    faintest = report["faintest_confirmed_star"]
    assert report["n_confirmed_stars"] == 2
    assert faintest["x"] == 20
    assert faintest["aperture_flux_adu"] == 10
    assert faintest["hits"] == 6
    assert faintest["mag"] == pytest.approx(17.5)


def test_one_detection_cannot_confirm_two_reference_sources():
    positions = {0: [[0, 0], [0.5, 0]], 1: [[0.1, 0]], 2: [[0.1, 0]]}
    np.testing.assert_array_equal(confirmation_hits(positions, 0, 1), [3, 1])


def test_matches_are_in_registered_sky_coordinates():
    tables = {0: table(0, [[10, 10]], [10]), 1: table(1, [[110, 10]], [10])}
    shift = np.eye(3)
    shift[0, 2] = -100
    reg = RegistrationResult(0, [0, 1], {0: np.eye(3), 1: shift})
    result = star_catalog(tables, reg, 0, min_hits=2, radius=1, exposure_s=.5)
    assert result["n_confirmed_stars"] == 1
    assert result["faintest_confirmed_star"]["mag"] is None
    assert result["faintest_confirmed_star"]["instrumental_mag_rate"] == pytest.approx(-2.5 * np.log10(20))


def test_empty_or_unconfirmed_catalog_has_no_faintest_star():
    for positions in ({0: [], 1: []}, {0: [[10, 10]], 1: []}):
        tables = {f: table(f, xy, [10] * len(xy)) for f, xy in positions.items()}
        reg = RegistrationResult(0, [0, 1], {f: np.eye(3) for f in tables})
        result = star_catalog(tables, reg, 0, min_hits=2, radius=1, exposure_s=1)
        assert result["faintest_confirmed_star"] is None


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf")])
def test_invalid_matching_radius_is_rejected(radius):
    with pytest.raises(ValueError):
        confirmation_hits({0: []}, 0, radius)


def test_calibration_joins_by_frame_and_source_not_prefix_slice():
    detections = {f: table(f, [[20, 20], [10, 10]], [1, 1]) for f in [10, 11, 12]}
    run = DetectionRun([10, 11, 12], 11, detections,
                       {10: np.array([999, 10]), 11: np.array([999, 20]), 12: np.array([999, 30])},
                       {}, 0, None, [])
    track = SimpleNamespace(frames=[10, 11, 12], xy_det=np.array([[10, 10]] * 3))
    np.testing.assert_array_equal(run.target_flux_rates(track, [12, 10]), [30, 10])
