import numpy as np

from src.config import load_config
from src.register.solver import RegistrationResult
from src.target.dual_frame import Track
from src.target.space_motion import MotionVerdict, find_space_targets
from src.target.trail_check import check_tracks
from src.space_pipeline import analyze_space_sequence
from src.validate.benchmark import SyntheticSequence, evaluate_tracks


def verdict(frames, points):
    points = np.asarray(points, dtype=float)
    n = len(frames)
    track = Track(frames, points, points, np.ones(n), np.ones(n), np.ones(n))
    return MotionVerdict(track, np.array([9., 1.4]), 0., float(n-1))


def test_one_to_one_scoring_counts_duplicate_and_wrong_tracks_as_false_positives():
    points = np.column_stack((np.arange(7)*9., np.arange(7)*1.4))
    truth = [dict(id=1, xy=points, visible_frames=list(range(7)))]
    predictions = [verdict(list(range(7)), points), verdict(list(range(7)), points),
                   verdict(list(range(7)), points+30)]
    result = evaluate_tracks(predictions, truth)
    assert (result['tp'], result['fp'], result['fn']) == (1, 2, 0)
    assert result['precision'] == 1/3
    assert result['localization_rmse_px'] == 0
    assert evaluate_tracks([], truth)['fn'] == 1
    assert evaluate_tracks([], [])['precision'] is None


def test_temporal_evidence_reports_real_gaps_not_probability():
    frames = [0, 1, 4, 5]
    result = verdict(frames, [[f*9, f*1.4] for f in frames]).to_dict()
    assert result['temporal_evidence'] == dict(observed_frames=4, span_frames=6,
        missing_frames=2, coverage_fraction=4/6, max_consecutive_missing=2)
    assert 'confidence' not in result


def test_synthetic_image_pipeline_and_independent_trail_negative_control():
    seq = SyntheticSequence(41, 'clean')
    config = load_config()
    run = analyze_space_sequence(seq, config, reference_frame=6)
    full = evaluate_tracks(run.targets, seq.truth)
    before, _ = find_space_targets(run.detections, run.frames, run.registration, seq.times().unix,
                                    config['space']['motion'])
    unfiltered = evaluate_tracks(before, seq.truth)
    accepted, rejected = check_tracks(seq, before, config['space']['trail_check'])
    assert full['tp'] == 2 and full['fp'] == 0
    assert unfiltered['fp'] >= 1
    assert rejected and all(r['is_target'] is False for r in rejected)
    assert all(r['trail_evidence']['consistent_fraction'] < .5 for r in rejected)


def test_seeded_frames_are_repeatable_and_truth_is_not_passed_to_inference():
    a, b = SyntheticSequence(137, 'gaps'), SyntheticSequence(137, 'gaps')
    np.testing.assert_array_equal(a.image(4), b.image(4))
    config = load_config()
    # Detection consumes only image(), times(), headers and length.
    class BlindSequence:
        headers = a.headers
        def __len__(self): return len(a)
        def image(self, frame): return a.image(frame)
        def times(self): return a.times()
    run = analyze_space_sequence(BlindSequence(), config, reference_frame=6)
    assert evaluate_tracks(run.targets, a.truth)['tp'] == 2
