from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import load_config
from src.space_pipeline import analyze_space_sequence
from src.validate.benchmark import SyntheticSequence
from src.validate.injection import DEFAULT_PROTOCOL, ImageSequence, file_hash, inject_target
from src.validate.injection_review import (
    audit_track, component_evidence, inspect_frames, loss_stage, replay_predictions,
    run_review, stage_counts, unassigned_indices, verify_baseline,
)
from tests.validate.test_benchmark import verdict


def test_component_attribution_uses_label_not_the_pixel_under_a_centroid():
    labels = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    signal = np.array([[2., 2, 2], [2, 10, 2], [2, 2, 2]])
    result = component_evidence(labels, signal+3, signal, 1)
    assert result['segment_injected_signal_adu'] == 16
    assert result['segment_positive_residual_adu'] == 40
    assert result['injected_to_positive_residual_ratio'] == .4


def test_stage_predicate_respects_equality_and_verifies_production_counts():
    motion = load_config()['space']['motion']
    source = dict(error_px=3., stationary_hits=3, npixels=2000, peak_adu=80., index=0)
    sources = [source, {**source, 'stationary_hits': 4, 'index': 1},
               {**source, 'peak_adu': 79.999, 'index': 2}, {**source, 'error_px': 3.01, 'index': 3}]
    frame = dict(frame=0, raw_near_count=4, sources=sources)
    totals, flags = stage_counts([frame], motion, {0: 2}, 3.)
    assert totals == dict(raw=1, valid=1, moving=1, area=1, eligible=1)
    assert flags[0]['eligible_source_indices'] == [0]
    with pytest.raises(ValueError, match='predicate'):
        stage_counts([frame], motion, {0: 1}, 3.)


def test_loss_stage_never_conflates_trail_rejection_and_localization():
    totals = dict(raw=15, valid=15, moving=15, area=15, eligible=15)
    assert loss_stage(totals, 0, 0) == 'association_or_localization'
    assert loss_stage(totals, 0, 1) == 'trail_check'
    assert loss_stage(totals, 1, 1) == 'recovered'
    assert loss_stage(totals, 0, 0, ambiguous=True) == 'preexisting_ambiguity'
    for key, stage in [('raw', 'detection_or_centroid'), ('valid', 'mask_or_edge'),
                       ('moving', 'stationary_filter'), ('area', 'area_filter'), ('eligible', 'peak_filter')]:
        assert loss_stage({**totals, key: 3}, 0, 0) == stage


def test_unassigned_indices_do_not_relabel_controls_or_duplicate_recovery():
    xy = np.column_stack((np.arange(7)*9., np.arange(7)*1.4))
    frames = list(range(7))
    truth = [dict(id=1, xy=xy, visible_frames=frames)]
    predictions = [verdict(frames, xy+50), verdict(frames, xy), verdict(frames, xy), verdict(frames, xy+100)]
    unknown, scored = unassigned_indices(predictions, truth, [predictions[0]], 7, 3.)
    assert len(unknown) == 2 and 0 not in unknown and 3 in unknown
    assert scored['recovered_count'] == 1 and scored['matched_control_count'] == 1


def test_replay_rejects_position_or_frame_drift():
    frames, xy = list(range(4)), np.column_stack((np.arange(4)*9., np.zeros(4)))
    saved = [dict(frames=frames, xy=xy.tolist())]
    replay_predictions([verdict(frames, xy)], saved)
    with pytest.raises(ValueError, match='coordinates'):
        replay_predictions([verdict(frames, xy+.001)], saved)
    with pytest.raises(ValueError, match='count'):
        replay_predictions([], saved)


def test_input_and_source_checks_fail_before_overwriting_any_artifact(tmp_path):
    source, image = tmp_path/'source.py', tmp_path/'test.fits'
    source.write_text('original', encoding='utf-8')
    image.write_bytes(b'fits bytes')
    report = dict(kind='real-background-injection', source_sha256={'source.py': file_hash(source)},
                  input_files=[dict(name='test.fits', sha256=file_hash(image))])
    verify_baseline(report, tmp_path, tmp_path)
    source.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='source mismatch'):
        verify_baseline(report, tmp_path, tmp_path)
    source.write_text('original', encoding='utf-8')
    image.write_bytes(b'changed')
    with pytest.raises(ValueError, match='FITS mismatch'):
        verify_baseline(report, tmp_path, tmp_path)
    output = tmp_path/'existing'
    output.mkdir()
    (output/'report.json').write_text('preserve')
    with pytest.raises(ValueError, match='empty'):
        run_review(tmp_path/'missing.json', tmp_path, output)
    assert (output/'report.json').read_text() == 'preserve'


def test_track_audit_explains_the_unchanged_match_boundary():
    truth = np.column_stack((np.arange(5)*9., np.zeros(5)))
    xy = truth + np.array([[0, 0], [1, 0], [4, 0], [5, 0], [6, 0]])
    prediction = verdict(list(range(5)), xy)
    prediction.trail_evidence = dict(consistent_fraction=1.)
    frame_evidence = [dict(sources=[dict(xy=point.tolist(), error_px=float(np.linalg.norm(point-truth[f])))])
                      for f, point in enumerate(xy)]
    result = audit_track(prediction, truth, frame_evidence, 3.)
    assert result['within_radius_count'] == 2 and result['within_radius_fraction'] == .4
    assert result['trail_consistent_fraction'] == 1
    assert all(o['cross_error_px'] == 0 for o in result['observations'])
    assert result['observations'][-1]['along_error_px'] == 6


def test_rebuilt_segmentation_and_candidate_audit_match_real_pipeline():
    from src.target.space_motion import find_space_targets
    base = SyntheticSequence(41, 'clean')
    background = ImageSequence(base.headers, [base.image(f) for f in range(len(base))])
    config, protocol = load_config(), load_config(DEFAULT_PROTOCOL)
    original_config = deepcopy(config)
    before = background.image(0).copy()
    control_run = analyze_space_sequence(background, config)
    injected, truth, _ = inject_target(background, 809, 12000., 6., protocol, config)
    run = analyze_space_sequence(injected, config)
    evidence, windows = inspect_frames(background, injected, run, control_run, truth[0]['xy'], config, 3.)
    _, counts = find_space_targets(run.detections, run.frames, run.registration, injected.times().unix,
                                   config['space']['motion'])
    totals, _ = stage_counts(evidence, config['space']['motion'], counts, 3.)
    assert totals['eligible'] >= 4
    assert len(windows) == len(base)
    assert all(0 <= e['signal_in_segments_fraction'] <= 1+1e-12 for e in evidence)
    np.testing.assert_array_equal(before, background.image(0))
    assert config == original_config
