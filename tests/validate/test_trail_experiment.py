import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import load_config
from src.validate.injection import DEFAULT_PROTOCOL, write_json
from src.validate.trail_experiment import new_crops, overlaps, extra_metrics, run_experiment, VARIANTS
from tests.validate.test_benchmark import verdict


def test_new_crops_are_deterministic_and_disjoint_from_all_used_regions():
    old = [dict(seed=41, bounds=[0, 1, 768, 768]), dict(seed=137, bounds=[2000, 1500, 768, 768])]
    crops = new_crops((4096, 4096), 768, old, 1)
    assert crops == new_crops((4096, 4096), 768, old, 1)
    seen = [c['bounds'] for c in old]
    for crop in crops:
        assert crop['bounds'][1] >= 1
        assert not any(overlaps(crop['bounds'], b) for b in seen)
        seen.append(crop['bounds'])
    with pytest.raises(ValueError):
        new_crops((4096, 4096), 768, old, 1, seeds=[41])
    with pytest.raises(ValueError, match='reserve'):
        new_crops((769, 769), 768, [dict(seed=1, bounds=[0, 1, 768, 768])], 1)


def test_duplicate_and_all_observation_metrics_do_not_hide_outliers():
    xy = np.column_stack((np.arange(5)*10, np.zeros(5))).astype(float)
    biased = xy.copy()
    biased[-1] += [10, 0]
    truth = [dict(id=1, xy=xy, visible_frames=list(range(5)))]
    result = extra_metrics([verdict(range(5), biased), verdict(range(5), xy)], truth,
                            dict(matches=[dict(prediction_index=0)]), 3.)
    assert result['duplicate_eligible_count'] == 1
    assert result['matched_track_squared_error_sum'] == 100
    assert result['matched_track_observations'] == 5


def test_runner_freezes_source_before_new_pixels_and_preserves_failures(tmp_path, monkeypatch):
    from src.validate import trail_experiment as module
    protocol = {**load_config(DEFAULT_PROTOCOL), 'tuning_seeds': [41], 'fluxes_adu': [1800.], 'speeds_px_s': [2.]}
    baseline = dict(kind='real-background-injection', effective_config=load_config(), production_config=load_config(),
                    protocol=protocol, image_shape=[4096, 4096], input_files=[], environment={},
                    crops=[dict(seed=41, split='tuning', bounds=[0, 1, 768, 768])],
                    controls=[dict(seed=41, threshold_adu=t, predictions=[]) for t in (40, 80)],
                    cases=[dict(seed=41, flux_adu=1800., speed_px_s=2., threshold_adu=t, predictions=[],
                                recovered_count=0, new_unmatched_count=0) for t in (40, 80)])
    path = tmp_path/'baseline.json'
    write_json(path, baseline)
    monkeypatch.setattr(module, 'verify_baseline', lambda *args: None)
    monkeypatch.setattr(module, 'source_hashes', lambda: {'test': 'frozen'})

    class Sequence:
        headers = []
        def __len__(self):
            return 6

    monkeypatch.setattr(module.FrameSequence, 'from_directory', lambda _: Sequence())
    opened = []
    monkeypatch.setattr(module, 'load_crop', lambda _, bounds: opened.append(bounds) or object())
    monkeypatch.setattr(module, 'compare_methods', lambda *a: ({v: dict(predictions=[], seconds=.1, diagnostics={}) for v in VARIANTS}, .2))
    monkeypatch.setattr(module, 'inject_target', lambda *a: (object(), [dict(id=1, xy=np.zeros((6, 2)), visible_frames=list(range(6)))], []))
    output = tmp_path/'dev'
    run_experiment(tmp_path, output, path)
    assert len(opened) == 1 and (output/'receipt.json').exists()
    with pytest.raises(ValueError, match='empty'):
        run_experiment(tmp_path, output, path)
    development = output/'experiment.json'
    run_experiment(tmp_path, tmp_path/'val', path, phase='validation', development_path=development)
    assert len(opened) == 4
    monkeypatch.setattr(module, 'source_hashes', lambda: {'test': 'changed'})
    with pytest.raises(ValueError, match='frozen'):
        run_experiment(tmp_path, tmp_path/'stale', path, phase='validation', development_path=development)
    assert len(opened) == 4
    monkeypatch.setattr(module, 'source_hashes', lambda: {'test': 'frozen'})
    def fail(*args):
        raise RuntimeError('fit failed')
    monkeypatch.setattr(module, 'compare_methods', fail)
    with pytest.raises(RuntimeError):
        run_experiment(tmp_path, tmp_path/'failed', path)
    assert (tmp_path/'failed/failure.json').exists()
    assert not (tmp_path/'failed/experiment.json').exists()
