import struct

import numpy as np
import pytest
from astropy.io import fits

from src.config import load_config
from src.dataio.space_metadata import decode_auxiliary
from src.dataio.fits_loader import FrameSequence
from src.detect.segmentation import SourceTable
from src.register.solver import RegistrationResult
from src.register.space import register_star_field
from src.register.transform import apply_transform, similarity_matrix
from src.space_pipeline import prepare_space_image
from src.target.space_motion import find_space_targets
from src.target.trail_check import measure_trail


def table(f, xy):
    xy = np.asarray(xy).reshape(-1, 2)
    n = len(xy)
    return SourceTable(f, xy[:, 0], xy[:, 1], np.arange(n) * 10 + 1000,
                       np.full(n, 200.), np.full(n, 1.2), np.full(n, 10))


def test_official_auxiliary_byte_order_and_invalid_payload():
    values = [270., 0., 129., -2., 7e6, 0., 0., 0., 7500., 0., 0., 0., 0., 1.] + [0.] * 12
    raw = np.frombuffer(struct.pack('<26d', *values), dtype=np.uint16).byteswap().tobytes()
    result = decode_auxiliary(raw)
    assert result['ra_deg'] == 129
    assert result['j2000_x'] == 7e6
    assert result['q4'] == 1
    with pytest.raises(ValueError):
        decode_auxiliary(bytes(208))
    with pytest.raises(ValueError):
        decode_auxiliary(raw[:200])


def test_fits_payload_selects_space_mode_and_mixed_sequence_is_rejected(tmp_path):
    values = [270., 0., 129., -2., 7e6, 0., 0., 0., 7500., 0., 0., 0., 0., 1.] + [0.] * 12
    raw = np.frombuffer(struct.pack('<26d', *values), dtype=np.uint16).byteswap().tobytes()
    pixels = np.full((128, 128), 20, dtype=np.int16)
    pixels[0, :104] = np.frombuffer(raw, dtype='>i2')
    header = fits.Header({'DATE-OBS': '2026-03-30T16:32:05.4132', 'EXPOSURE': '0000 01500',
                          'AZIMUTH': '0270.00000 0000.00000', 'ELEVATIO': '0000.00000 0000.00000',
                          'SITELONG': 0., 'SITELATI': 0., 'SITEALTI': 0.})
    fits.PrimaryHDU(pixels, header).writeto(tmp_path/'space.fits')
    sequence = FrameSequence.from_directory(tmp_path)
    assert sequence.observation_mode == 'space'
    assert sequence.headers[0].space_aux['ra_deg'] == 129
    assert sequence.headers[0].exposure_s == 1.5
    np.testing.assert_array_equal(sequence.image(0), pixels)
    header['DATE-OBS'] = '2026-03-30T16:32:06.8862'
    fits.PrimaryHDU(np.full_like(pixels, 20), header).writeto(tmp_path/'bad.fits')
    with pytest.raises(ValueError, match='混合'):
        _ = FrameSequence.from_directory(tmp_path).observation_mode


def test_space_mask_preserves_coordinates_and_does_not_wrap_negative_pixels():
    original = np.full((40, 40), 20.)
    original[0] = 32000
    original[20, 20] = -32768
    original[30, 30] = 30500
    original[10, 10] = 1000
    image, mask = prepare_space_image(original, load_config()['space'])
    assert image.shape == original.shape
    assert image[20, 20] == 20 and image[30, 30] == 20
    assert mask[0].all() and mask[20, 20]
    assert original[20, 20] == -32768
    assert image[10, 10] == 1000  # No crop / coordinate offset.


def test_two_moving_targets_with_gap_irregular_times_and_random_transients():
    rng = np.random.default_rng(41)
    frames = list(range(9))
    times = np.array([0., 1., 2.6, 3.7, 5., 6.7, 8., 9.5, 11.])
    fixed = rng.uniform(300, 900, (45, 2))
    tables = {}
    for f in frames:
        points = list(fixed + rng.normal(0, .03, fixed.shape))
        if f != 4:
            points.append(np.array([50., 80.]) + times[f] * np.array([5., 2.]))
        points.append(np.array([150., 230.]) + times[f] * np.array([-3., -4.]))
        points.extend(rng.uniform(1000, 1800, (8, 2)))
        tables[f] = table(f, points)
    reg = RegistrationResult(0, frames, {f: np.eye(3) for f in frames})
    verdicts, _ = find_space_targets(tables, frames, reg, times, load_config()['space']['motion'])
    assert len(verdicts) == 2
    assert sorted(v.track.length for v in verdicts) == [8, 9]
    velocities = sorted(tuple(np.round(v.velocity, 5)) for v in verdicts)
    assert velocities == [(-3., -4.), (5., 2.)]
    nodes = [(f, tuple(xy)) for v in verdicts for f, xy in zip(v.track.frames, v.track.xy_det)]
    assert len(nodes) == len(set(nodes))


def test_stationary_stars_and_single_frame_events_are_not_motion():
    rng = np.random.default_rng(4)
    fixed = rng.uniform(0, 3000, (80, 2))
    tables = {f: table(f, np.vstack([fixed, [[f*350+10, 4000]]]) if f % 2 == 0 else fixed)
              for f in range(6)}
    reg = RegistrationResult(0, list(tables), {f: np.eye(3) for f in tables})
    result, _ = find_space_targets(tables, list(tables), reg, np.arange(6), load_config()['space']['motion'])
    assert result == []


def test_dense_registration_recovers_transform_with_outliers():
    rng = np.random.default_rng(7)
    stars = rng.uniform(100, 3900, (160, 2))
    transform = similarity_matrix(rotation_deg=.1, tx=12, ty=-7)
    shifted = apply_transform(transform, stars) + rng.normal(0, .04, stars.shape)
    tables = {0: table(0, stars), 1: table(1, np.vstack([shifted, rng.uniform(0, 4096, (20, 2))]))}
    reg = register_star_field(tables, [0, 1], load_config()['space']['registration'])
    assert np.max(np.linalg.norm(reg.to_sky(1, shifted) - stars, axis=1)) < .2
    assert reg.pairs[0].rms_px < .1


def test_short_round_flash_cannot_support_fast_motion_but_resolved_trail_can():
    yy, xx = np.indices((160, 160))
    point = 20 + 200 * np.exp(-((xx-80)**2+(yy-80)**2)/2)
    config = load_config()['space']['trail_check']
    assert not measure_trail(point, [80, 80], np.array([30., 0.]), 1.5, config)['consistent']
    trail = np.full((160, 160), 20.)
    for x in np.linspace(57.5, 102.5, 80):
        trail += 80 * np.exp(-((xx-x)**2+(yy-80)**2)/2)
    assert measure_trail(trail, [80, 80], np.array([30., 0.]), 1.5, config)['consistent']
    assert not measure_trail(trail, [80, 80], np.array([0., 30.]), 1.5, config)['consistent']
