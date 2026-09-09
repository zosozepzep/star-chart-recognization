from __future__ import annotations

import numpy as np
import pytest

from src.dataio.fits_loader import (
    FrameSequence,
    load_image,
    parse_exposure_ms,
    parse_pointing,
    read_header,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0270.28456 0000.00000", 270.28456),
        ("0015.24484 0000.00000", 15.24484),
        ("0088.74145 0000.00000", 88.74145),
        (270.28456, 270.28456),
    ],
)
def test_parse_pointing(raw, expected):
    assert parse_pointing(raw) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0000          00080", 80.0),
        ("0000          00030", 30.0),
        ("0000          00040", 40.0),
        (30, 30.0),
    ],
)
def test_parse_exposure_ms(raw, expected):
    assert parse_exposure_ms(raw) == pytest.approx(expected)


def test_parse_pointing_rejects_garbage():
    with pytest.raises(ValueError):
        parse_pointing("not a number")


def test_read_header_dataset_b_frame0(dataset_b_dir):
    path = sorted(dataset_b_dir.glob("*.fits"))[0]
    hdr = read_header(path, 0)
    assert hdr.index == 0
    assert hdr.azimuth_deg == pytest.approx(270.28456, abs=1e-4)
    assert hdr.elevation_deg == pytest.approx(15.24484, abs=1e-4)
    assert hdr.exposure_ms == pytest.approx(80.0)
    assert hdr.site_lon_deg == pytest.approx(88.34380, abs=1e-5)
    assert hdr.site_lat_deg == pytest.approx(43.31200, abs=1e-5)
    assert hdr.site_alt_m == pytest.approx(1801.0)
    assert hdr.date_obs.isot.startswith("2026-07-21T17:26:28")


def test_load_image_shape_and_dtype(dataset_b_dir):
    path = sorted(dataset_b_dir.glob("*.fits"))[0]
    img = load_image(path)
    assert img.shape == (4096, 4096)
    assert img.dtype == np.float64
    assert np.isfinite(img).all()
    assert img.min() >= 0.0


def test_sequence_dataset_b(dataset_b_dir):
    seq = FrameSequence.from_directory(dataset_b_dir)
    assert seq.dataset_id == "60385_20260722_1485695076008039_PIC_POS"
    assert len(seq) == 80
    assert seq.truth_path is not None and seq.truth_path.suffix.upper() == ".DAT"
    assert seq.cadence_s() == pytest.approx(1.009, abs=0.02)
    # 曝光在 f08 处从 80 ms 降到 30 ms
    assert seq.headers[0].exposure_ms == pytest.approx(80.0)
    assert seq.headers[8].exposure_ms == pytest.approx(30.0)
    # 帧按时间严格递增
    dt = np.diff(seq.times().unix)
    assert (dt > 0).all()


def test_sequence_dataset_a_has_no_truth(dataset_a_dir):
    seq = FrameSequence.from_directory(dataset_a_dir)
    assert len(seq) == 36
    assert seq.truth_path is None
    assert seq.headers[0].site_alt_m == pytest.approx(1798.0)


def test_sequence_image_matches_direct_load(dataset_b_dir):
    seq = FrameSequence.from_directory(dataset_b_dir)
    direct = load_image(seq.headers[3].path)
    assert np.array_equal(seq.image(3), direct)
