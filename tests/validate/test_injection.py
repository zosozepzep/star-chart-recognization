from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time, TimeDelta

from src.config import load_config
from src.space_pipeline import analyze_space_sequence
from src.validate.benchmark import SyntheticSequence
from src.validate.injection import (
    DEFAULT_PROTOCOL, ImageSequence, aggregate_cases, crop_bounds, inject_target,
    run_injection, score_injection, select_threshold, trail_patch, validate_protocol,
)
from tests.validate.test_benchmark import verdict


def test_protocol_rejects_leakage_nonfinite_grid_and_missing_baseline():
    protocol, config = load_config(DEFAULT_PROTOCOL), load_config()
    validate_protocol(protocol, config)
    for key, value in [('holdout_seeds', [41]), ('fluxes_adu', [float('nan')]),
                       ('peak_thresholds_adu', [40, 60]), ('crop_size', 768.5),
                       ('speeds_px_s', [0]), ('tuning_seeds', [41, 41])]:
        invalid = {**protocol, key: value}
        with pytest.raises(ValueError):
            validate_protocol(invalid, config)


def test_spatial_holdout_excludes_tuning_pixels_and_telemetry():
    for seed in range(20):
        tune = crop_bounds((4096, 4096), 768, 'tuning', seed, 1)
        hold = crop_bounds((4096, 4096), 768, 'holdout', seed, 1)
        assert tune[0] + tune[2] <= 2048 <= hold[0]
        assert tune[1] >= 1 and hold[1] >= 1
        assert tune == crop_bounds((4096, 4096), 768, 'tuning', seed, 1)
    with pytest.raises(ValueError):
        crop_bounds((512, 512), 768, 'tuning', 41, 1)


@pytest.mark.parametrize('velocity', [(0., 0.), (9., 1.4), (-5., 3.)])
def test_render_preserves_flux_centroid_and_exposure_second_moment(velocity):
    protocol = load_config(DEFAULT_PROTOCOL)
    position, exposure, flux = np.array([80.3, 81.7]), 1.5, 1800.
    region, patch = trail_patch((160, 160), position, velocity, exposure, flux, protocol)
    yy, xx = np.mgrid[region[0], region[1]]
    center = np.array([(xx*patch).sum(), (yy*patch).sum()])/flux
    np.testing.assert_allclose(patch.sum(), flux, atol=1e-9)
    np.testing.assert_allclose(center, position, atol=1e-6)
    points = np.column_stack((xx.ravel(), yy.ravel())) - center
    covariance = (points.T * patch.ravel()) @ points / flux
    eigenvalues = np.linalg.eigvalsh(covariance)
    expected = (np.linalg.norm(velocity)*exposure)**2 / 12
    assert abs((eigenvalues[1]-eigenvalues[0])-expected) < .03


def test_injection_preserves_bad_pixels_and_input_and_is_repeatable():
    base = SyntheticSequence(41, 'clean')
    background = ImageSequence(base.headers, [np.full((512, 512), 100.) for _ in base.headers])
    protocol, config = load_config(DEFAULT_PROTOCOL), load_config()
    injected, truth, _ = inject_target(background, 41, 1800., 6., protocol, config)
    x, y = np.rint(truth[0]['xy'][0]).astype(int)
    background.images[0][y, x] = -32768
    background.images[0][y, x+1] = 31000
    background.images[0][y, x+2] = np.nan
    original = background.image(0).copy()
    a, truth_a, audit = inject_target(background, 41, 1800., 6., protocol, config)
    b, truth_b, _ = inject_target(background, 41, 1800., 6., protocol, config)
    np.testing.assert_array_equal(background.image(0), original)
    np.testing.assert_array_equal(a.image(0), b.image(0))
    np.testing.assert_array_equal(truth_a[0]['xy'], truth_b[0]['xy'])
    assert a.image(0)[y, x] == -32768 and a.image(0)[y, x+1] == 31000
    assert np.isnan(a.image(0)[y, x+2])
    assert 0 < audit[0]['signal_masked_fraction'] < 1
    assert audit[0]['added_flux_adu'] < 1800
    assert not hasattr(a, 'truth')
    assert audit[1]['added_flux_adu'] == pytest.approx(1800)


def test_scoring_separates_recovery_controls_duplicates_and_unknown_candidates():
    frames = list(range(7))
    xy = np.column_stack((np.arange(7)*9., np.arange(7)*1.4))
    truth = [dict(id=1, xy=xy, visible_frames=frames)]
    control = verdict(frames, xy+50)
    result = score_injection([verdict(frames, xy), verdict(frames, xy), control,
                              verdict(frames, xy+100)], truth, [control], 7, 3.)
    assert result['recovered_count'] == 1 and result['missed_count'] == 0
    assert result['matched_control_count'] == 1 and result['new_unmatched_count'] == 2
    assert result['unmatched_prediction_count'] == 3
    assert 'precision' not in result and 'fp' not in result
    # A pre-existing path cannot masquerade as a successful injection.
    ambiguous = score_injection([verdict(frames, xy)], truth, [verdict(frames, xy)], 7, 3.)
    assert ambiguous['recovered_count'] == 0 and ambiguous['missed_count'] == 1
    assert ambiguous['ambiguous_truth_ids'] == [1]
    assert ambiguous['localization_rmse_px'] is None


def test_selection_cannot_use_holdout_and_keeps_baseline_on_tie():
    rows = [dict(split='tuning', threshold_adu=t, recovered_count=2, new_unmatched_count=0)
            for t in (40, 60, 80)]
    assert select_threshold(rows, 80)['selected_threshold_adu'] == 80
    rows.append(dict(split='holdout', threshold_adu=40, recovered_count=1000, new_unmatched_count=0))
    assert select_threshold(rows, 80)['selected_threshold_adu'] == 80
    rows[1]['recovered_count'] = 3
    assert select_threshold(rows, 80)['selected_threshold_adu'] == 60


def test_bins_keep_misses_in_denominator_and_do_not_pool_speeds_or_splits():
    common = dict(split='holdout', threshold_adu=80., flux_adu=600., speed_px_s=2.,
                  truth_count=1, recovered_count=1, new_unmatched_count=0,
                  ambiguous_truth_ids=[], association_and_check_seconds=.1)
    rows = [common, {**common, 'recovered_count': 0, 'ambiguous_truth_ids': [1]},
            {**common, 'speed_px_s': 12., 'recovered_count': 0},
            {**common, 'split': 'tuning'}]
    bins = aggregate_cases(rows)
    assert len(bins) == 3
    slow = next(b for b in bins if b['split'] == 'holdout' and b['speed_px_s'] == 2.)
    assert slow['truth_count'] == 2 and slow['recovery_rate'] == .5
    assert slow['ambiguous_truth_count'] == 1


def test_real_pipeline_recovers_an_injection_without_truth_access():
    base = SyntheticSequence(41, 'clean')
    protocol, config = load_config(DEFAULT_PROTOCOL), load_config()
    background = ImageSequence(base.headers, [base.image(f) for f in range(len(base))])
    controls = analyze_space_sequence(background, config).targets
    injected, truth, _ = inject_target(background, 809, 12000., 6., protocol, config)
    result = analyze_space_sequence(injected, config)
    score = score_injection(result.targets, truth, controls, len(base), 3.)
    assert score['recovered_count'] == 1
    assert score['localization_rmse_px'] < 1


def test_runner_freezes_selection_before_holdout_and_preserves_outputs(tmp_path, monkeypatch):
    from src.validate import injection
    headers = []
    for f in range(6):
        path = tmp_path/f'{f}.fits'
        fits.PrimaryHDU(np.full((512, 768), 100, dtype=np.int16)).writeto(path)
        headers.append(SimpleNamespace(path=path, exposure_s=1.5,
                       date_obs=Time('2026-03-30T16:32:05') + TimeDelta(f*1.5, format='sec')))
    sequence = ImageSequence(headers, [])
    sequence.observation_mode = 'space'
    monkeypatch.setattr(injection.FrameSequence, 'from_directory', lambda _: sequence)
    protocol = {**load_config(DEFAULT_PROTOCOL), 'crop_size': 256,
                'tuning_seeds': [41], 'holdout_seeds': [137], 'fluxes_adu': [1800.],
                'speeds_px_s': [2.], 'peak_thresholds_adu': [80.]}
    config, output = load_config(), tmp_path/'result'
    unchanged = deepcopy(config)
    calls = []

    def scan(seq, effective, thresholds):
        assert not hasattr(seq, 'truth')
        assert effective['space']['metadata_rows'] == 0
        assert (output/'protocol.json').is_file()
        if len(calls) >= 2:
            assert (output/'selection.json').is_file()
        calls.append(len(seq))
        return SimpleNamespace(diagnostics={'per_frame': []}), {
            80.: dict(predictions=[], candidate_counts={}, rejected_count=0,
                      association_and_check_seconds=.01)}, .1

    monkeypatch.setattr(injection, 'scan_thresholds', scan)
    report = run_injection(tmp_path, output, protocol=protocol, config=config)
    assert len(calls) == 4 and config == unchanged
    assert report['selection']['selected_threshold_adu'] == 80
    assert len(report['cases']) == 2 and all(r['missed_count'] == 1 for r in report['cases'])
    assert len(report['input_files']) == 6
    assert len((output/'cases.jsonl').read_text().splitlines()) == 2
    for name in ('injection.json', 'recovery.csv', 'report.md', 'recovery.png'):
        assert (output/name).stat().st_size > 0
    assert json.loads((output/'injection.json').read_text())['total_seconds'] > 0
    with pytest.raises(ValueError, match='empty'):
        run_injection(tmp_path, output, protocol=protocol)

    def fail(*args):
        raise RuntimeError('registration failure')

    monkeypatch.setattr(injection, 'scan_thresholds', fail)
    failed = tmp_path/'failed'
    with pytest.raises(RuntimeError):
        run_injection(tmp_path, failed, protocol=protocol)
    assert (failed/'protocol.json').exists() and (failed/'failure.json').exists()
    assert not (failed/'injection.json').exists()
