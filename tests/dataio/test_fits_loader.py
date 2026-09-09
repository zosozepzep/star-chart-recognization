from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from src.dataio.fits_loader import (
    FrameSequence,
    load_image,
    parse_exposure_ms,
    parse_pointing,
    read_header,
)


def _write_fits(path, date_obs: str, fill: int) -> None:
    """写一个最小可用的合成 FITS：头部字段齐全，像素为常量便于区分。"""
    data = np.full((4, 4), fill, dtype=np.int16)
    hdr = fits.Header()
    hdr["DATE-OBS"] = date_obs
    hdr["AZIMUTH"] = "0270.28456 0000.00000"
    hdr["ELEVATIO"] = "0015.24484 0000.00000"
    hdr["EXPOSURE"] = "0000          00080"
    hdr["SITELONG"] = 88.34380
    hdr["SITELATI"] = 43.31200
    hdr["SITEALTI"] = 1801.0
    fits.PrimaryHDU(data=data, header=hdr).writeto(path)


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
    # 期望路径独立于 seq 推导，否则断言退化为 load_image(p) == load_image(p)
    expected_path = sorted(dataset_b_dir.glob("*.fits"))[3]
    direct = load_image(expected_path)
    assert np.array_equal(seq.image(3), direct)
    # 若 image() 忽略 index（常量索引 bug），下面这条会失败
    assert not np.array_equal(seq.image(3), seq.image(4))


def test_from_directory_reorders_and_reindexes(tmp_path):
    """文件名次序与时间次序不同时，序列必须按时间重排并重编 index。

    用三帧而非两帧：两元素置换 [1, 0] 是自身的逆，无法区分 gather 与 scatter 实现。
    这里文件名序 a,b,c 对应时间序 c,a,b（置换非自逆），因此把重排方向写反会失败。
    """
    _write_fits(tmp_path / "a.fits", "2026-07-21T17:26:30.000", fill=10)
    _write_fits(tmp_path / "b.fits", "2026-07-21T17:26:31.000", fill=20)
    _write_fits(tmp_path / "c.fits", "2026-07-21T17:26:29.000", fill=30)

    seq = FrameSequence.from_directory(tmp_path)
    assert [h.path.name for h in seq.headers] == ["c.fits", "a.fits", "b.fits"]
    assert [h.index for h in seq.headers] == [0, 1, 2]
    assert seq.image(0)[0, 0] == pytest.approx(30.0)
    assert seq.image(1)[0, 0] == pytest.approx(10.0)
    assert seq.image(2)[0, 0] == pytest.approx(20.0)
    assert seq.cadence_s() == pytest.approx(1.0)


def test_from_directory_raises_without_fits(tmp_path):
    with pytest.raises(FileNotFoundError):
        FrameSequence.from_directory(tmp_path)


def test_from_directory_raises_on_multiple_truth_files(tmp_path):
    _write_fits(tmp_path / "f0.fits", "2026-07-21T17:26:29.000", fill=10)
    (tmp_path / "one.DAT").write_text("")
    (tmp_path / "two.DAT").write_text("")
    with pytest.raises(ValueError, match="多个真值文件"):
        FrameSequence.from_directory(tmp_path)


def test_cadence_s_raises_on_single_frame(tmp_path):
    """单帧序列的帧间隔是显式错误，不是 nan。"""
    _write_fits(tmp_path / "only.fits", "2026-07-21T17:26:29.000", fill=10)
    seq = FrameSequence.from_directory(tmp_path)
    assert len(seq) == 1
    with pytest.raises(ValueError, match="少于 2 帧"):
        seq.cadence_s()
