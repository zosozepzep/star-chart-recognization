from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import load_config
from src.target.trail_refinement import fit_trail_center, refine_and_relink, trail_density
from src.validate.injection import DEFAULT_PROTOCOL, trail_patch, ImageSequence
from src.validate.trail_experiment import FIT_CONFIG
from tests.validate.test_benchmark import verdict


@pytest.mark.parametrize('velocity,sigma', [([12., 0.], 1.15), ([-5., 9.], 1.8), ([0., 0.], 1.3)])
def test_analytic_model_has_correct_flux_center_and_variance(velocity, sigma):
    yy, xx = np.indices((101, 101))
    center = np.array([50.2, 49.7])
    density = trail_density(xx, yy, center, velocity, 1.5, sigma)
    assert density.sum() == pytest.approx(1., abs=1e-6)
    np.testing.assert_allclose([(xx*density).sum(), (yy*density).sum()], center, atol=1e-5)
    points = np.column_stack((xx.ravel(), yy.ravel()))-center
    cov = (points.T*density.ravel())@points
    np.testing.assert_allclose(cov, sigma**2*np.eye(2)+np.outer(velocity, velocity)*1.5**2/12, atol=1e-4)


@pytest.mark.parametrize('velocity,offset', [([12., 0.], [5., 0.]), ([-8., 9.], [-3., 3.])])
def test_recovers_full_exposure_center_from_biased_fragment(velocity, offset):
    protocol = {**load_config(DEFAULT_PROTOCOL), 'psf_sigma_px': 1.7}
    center = np.array([100.3, 98.8])
    image = 100 + np.random.default_rng(42).normal(0, 3, (200, 200))
    region, patch = trail_patch(image.shape, center, velocity, 1.5, 5400., protocol)
    image[region] += patch
    original = image.copy()
    result = fit_trail_center(image, center+offset, velocity, 1.5, load_config(FIT_CONFIG), saturation_adu=30000)
    assert result['accepted'], result
    np.testing.assert_allclose(result['fitted_xy'], center, atol=.3)
    assert result['sigma_px'] == pytest.approx(1.7, abs=.15)
    np.testing.assert_array_equal(image, original)


def test_bad_fit_edge_unresolved_and_invalid_pixels_are_reported():
    config = load_config(FIT_CONFIG)
    image = np.full((160, 160), 100.)
    kwargs = dict(exposure_s=1.5, config=config, saturation_adu=30000)
    assert fit_trail_center(image, [5, 5], [12, 0], **kwargs)['reason'] == 'image_edge'
    assert fit_trail_center(image, [80, 80], [2, 0], **kwargs)['reason'] == 'unresolved_trail'
    result = fit_trail_center(image, [80, 80], [12, 0], **kwargs)
    assert not result['accepted']
    image[50:110] = np.nan
    assert fit_trail_center(image, [80, 80], [12, 0], **kwargs)['reason'] == 'masked_pixels'
    with pytest.raises(ValueError):
        fit_trail_center(image, [80, 80], [np.nan, 0], **kwargs)


def test_stationary_blend_mask_and_bad_pixel_do_not_pull_center():
    config = load_config(FIT_CONFIG)
    center, velocity = np.array([100.2, 100.4]), [12., 0.]
    yy, xx = np.indices((200, 200))
    image = 100 + np.random.default_rng(4).normal(0, 3, (200, 200))
    image += 5400*trail_density(xx, yy, center, velocity, 1.5, 1.4)
    image += 30000*trail_density(xx, yy, [100, 109], [0, 0], 1.5, 1.)
    image[97, 96] = -32768
    result = fit_trail_center(image, center+[3, 0], velocity, 1.5, config,
                              saturation_adu=30000, stationary_xy=[[100, 109]])
    assert result['accepted'], result
    np.testing.assert_allclose(result['fitted_xy'], center, atol=.3)


def test_relink_uses_measured_velocity_preserves_failures_and_never_fills_frames(monkeypatch):
    from src.target import trail_refinement as module
    from src.detect.segmentation import SourceTable
    from astropy.time import Time, TimeDelta
    frames = [0, 1, 3, 4]
    track = verdict(frames, np.array([[50., 50.], [60., 50.], [80., 50.], [90., 50.]]))
    tables = {f: SourceTable(f, [50+10*f], [50.], [100.], [60.], [2.], [5]) for f in range(5)}
    initial = {f: t.xy.copy() for f, t in tables.items()}
    headers = [SimpleNamespace(date_obs=Time('2026-01-01')+TimeDelta(2*f, format='sec'), exposure_s=1.5) for f in range(5)]
    sequence = ImageSequence(headers, [np.full((160, 160), 100.)]*5)
    run = SimpleNamespace(detections=tables, frames=list(range(5)), registration=None)
    monkeypatch.setattr(module, 'stationary_positions', lambda *a: {f: [] for f in range(5)})
    calls = []

    def fit(image, xy, velocity, *args, **kwargs):
        np.testing.assert_allclose(velocity, [5., 0.], atol=1e-6)
        calls.append(xy.copy())
        return dict(accepted=len(calls) != 2, fitted_xy=(xy+[1, 0]).tolist(), reason='test')

    def link(new, *args):
        for f in (1, 2):
            np.testing.assert_array_equal(new[f].xy, initial[f])
        for f in (0, 3, 4):
            np.testing.assert_array_equal(new[f].xy, initial[f]+[1, 0])
        return [], {}

    monkeypatch.setattr(module, 'fit_trail_center', fit)
    monkeypatch.setattr(module, 'find_space_targets', link)
    monkeypatch.setattr(module, 'check_tracks', lambda *a: ([], []))
    config = load_config()
    unchanged = deepcopy(config)
    _, evidence = refine_and_relink(sequence, run, [track], config, load_config(FIT_CONFIG))
    assert len(calls) == 4 and len(evidence['fits']) == 4 and config == unchanged
    for f in tables:
        np.testing.assert_array_equal(tables[f].xy, initial[f])


def test_real_pipeline_retains_targets_and_rejects_untrailed_flashes():
    from src.validate.benchmark import SyntheticSequence, evaluate_tracks
    from src.validate.trail_experiment import compare_methods
    generated = SyntheticSequence(41, 'clean')
    sequence = ImageSequence(generated.headers, [generated.image(f) for f in range(len(generated))])
    results, _ = compare_methods(sequence, load_config(), load_config(FIT_CONFIG))
    assert not hasattr(sequence, 'truth')
    for result in results.values():
        score = evaluate_tracks(result['predictions'], generated.truth)
        assert score['tp'] == 2 and score['fp'] == 0
    diagnostics = results['refined40']['diagnostics']
    assert any(f['accepted'] for f in diagnostics['fits'])
    assert diagnostics['rejected_trails']
