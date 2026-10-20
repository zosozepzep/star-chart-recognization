"""Exercise real pools, inference equivalence, bounded input and cleanup."""
from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import threading
import time

import numpy as np
import pytest
from astropy.io import fits
from threadpoolctl import threadpool_info

from src.config import load_config
from src.dataio.fits_loader import FrameSequence
from src.parallel import execution_options, execution_plan, frame_input, map_frames, read_frame
from src.pipeline import analyze_sequence
from src.space_pipeline import analyze_space_sequence


def worker_identity(index):
    time.sleep(.08)
    return index, os.getpid(), threading.get_ident(), threadpool_info()


@pytest.mark.parametrize('executor', ['process', 'thread'])
def test_real_parallel_workers_preserve_order_and_limit_native_threads(executor):
    before = threadpool_info()
    plan = execution_plan({'execution': {'workers': 2, 'executor': executor}}, 8)
    results = list(map_frames(worker_identity, range(8), plan))
    assert [r[0] for r in results] == list(range(8))
    identities = {(r[1], r[2]) for r in results}
    assert len(identities) == 2
    if executor == 'process':
        assert all(r[1] != os.getpid() for r in results)
    else:
        assert all(r[1] == os.getpid() for r in results)
    assert all(pool['num_threads'] == 1 for r in results for pool in r[3])
    assert threadpool_info() == before


def test_submission_is_bounded_and_out_of_order_completions_are_reordered():
    started, release = threading.Event(), threading.Event()
    submitted, answers, errors = [], [], []
    def jobs():
        for n in range(10):
            submitted.append(n)
            yield n
    def work(n):
        if n == 0:
            assert release.wait(5)
        if n == 1:
            started.set()
        return n
    def consume():
        try:
            answers.extend(map_frames(work, jobs(), {'workers': 2, 'executor': 'thread'}))
        except Exception as exc:
            errors.append(exc)
    consumer = threading.Thread(target=consume)
    consumer.start()
    try:
        assert started.wait(5)
        assert submitted == [0, 1]
    finally:
        release.set()
        consumer.join(10)
    assert not consumer.is_alive() and not errors
    assert answers == list(range(10))


@pytest.mark.parametrize('executor', ['process', 'thread'])
def test_worker_failure_propagates_and_pool_is_joined(tmp_path, executor):
    old_children = {p.pid for p in multiprocessing.active_children()}
    plan = execution_plan({'execution': {'workers': 2, 'executor': executor}}, 3)
    with pytest.raises(FileNotFoundError, match='missing-frame'):
        list(map_frames(read_frame, [tmp_path/'missing-frame.fits'] * 3, plan))
    assert {p.pid for p in multiprocessing.active_children()} == old_children


@pytest.mark.parametrize('workers', [-1, True, 2.5, '4', None])
def test_bad_worker_counts_rejected(workers):
    with pytest.raises(ValueError, match='workers'):
        execution_options({'execution': {'workers': workers}})


def test_old_config_auto_and_short_sequence(monkeypatch):
    assert execution_plan({}, 15)['executor'] == 'serial'
    monkeypatch.setattr('src.parallel.cpu_capacity', lambda: 16)
    assert execution_plan({'execution': {'workers': 0}}, 15)['workers'] == 4
    assert execution_plan({'execution': {'workers': 8}}, 2)['workers'] == 2
    monkeypatch.setattr('src.parallel.cpu_capacity', lambda: 1)
    assert execution_plan({'execution': {'workers': 0}}, 15)['executor'] == 'serial'
    with pytest.raises(ValueError, match='executor'):
        execution_options({'execution': {'executor': 'invalid'}})
    with pytest.raises(ValueError, match='映射'):
        execution_options({'execution': []})


def assert_same_run(serial, parallel):
    assert serial.frames == parallel.frames
    assert serial.reference_frame == parallel.reference_frame
    assert serial.raw_star_count == parallel.raw_star_count
    assert serial.diagnostics == parallel.diagnostics
    for name in ('detections', 'stars'):
        a, b = getattr(serial, name), getattr(parallel, name)
        assert list(a) == list(b)
        for f in a:
            for column in ('x', 'y', 'flux', 'peak', 'elongation', 'npix'):
                np.testing.assert_array_equal(getattr(a[f], column), getattr(b[f], column))
    for f in serial.frames:
        np.testing.assert_array_equal(serial.aperture_rates[f], parallel.aperture_rates[f])
    assert serial.registration.report() == parallel.registration.report()
    assert [v.to_dict() for v in serial.targets] == [v.to_dict() for v in parallel.targets]


@pytest.mark.parametrize('executor', ['process', 'thread'])
def test_space_pipeline_matches_serial_without_pickling_sequence_or_truth(executor):
    from src.validate.benchmark import SyntheticSequence
    generated = SyntheticSequence(137, 'gaps')
    class BlindSequence:
        headers = generated.headers
        def __len__(self): return len(generated)
        def image(self, f): return generated.image(f)
        def times(self): return generated.times()
        def __getstate__(self): raise AssertionError('Never pickle a sequence')
        @property
        def truth(self): raise AssertionError('Never read truth')
    sequence = BlindSequence()
    config = load_config()
    config['background']['backend'] = 'cpu'
    config['execution'] = dict(workers=1, executor=executor)
    serial = analyze_space_sequence(sequence, config, reference_frame=6)
    assert len(serial.targets) == 2
    config['execution']['workers'] = 2
    before = deepcopy(config)
    parallel = analyze_space_sequence(sequence, config, reference_frame=6)
    assert config == before
    assert parallel.execution['workers'] == 2
    assert_same_run(serial, parallel)


@pytest.fixture
def ground_sequence(tmp_path):
    rng = np.random.default_rng(725)
    stars = rng.uniform(60, 380, (55, 2))
    amplitudes = rng.uniform(200, 700, len(stars))
    yy, xx = np.indices((512, 512))
    for f in range(6):
        pixels = rng.normal(100, 3, xx.shape)
        for (x, y), amp in zip(stars + [4*f, 0], amplitudes):
            pixels += amp * np.exp(-((xx-x)**2 + (yy-y)**2) / (2*1.2**2))
        pixels += 500 * np.exp(-((xx-240-1.5*f)**2 + (yy-430)**2) / (2*1.2**2))
        header = fits.Header({'DATE-OBS': f'2026-03-30T16:32:{f:02d}', 'EXPOSURE': '0000 01000',
                              'AZIMUTH': 270., 'ELEVATIO': 30.,
                              'SITELONG': 120., 'SITELATI': 30., 'SITEALTI': 100.})
        fits.PrimaryHDU(pixels.astype(np.int16), header).writeto(tmp_path/f'{f}.fits')
    return FrameSequence.from_directory(tmp_path)


@pytest.mark.parametrize('executor', ['process', 'thread'])
def test_ground_fits_pipeline_matches_serial(ground_sequence, executor):
    assert isinstance(frame_input(ground_sequence, 0), Path)
    config = load_config()
    config['background']['backend'] = 'cpu'
    config['execution'] = dict(workers=1, executor=executor)
    config['target'].update(min_track_frames=4, v_sky_min_px=2.)
    serial = analyze_sequence(ground_sequence, config, reference_frame=2, frame_range=(0, 5))
    assert serial.raw_star_count > 30
    assert len(serial.targets) == 1
    config['execution']['workers'] = 2
    parallel = analyze_sequence(ground_sequence, config, reference_frame=2, frame_range=(0, 5))
    assert_same_run(serial, parallel)
