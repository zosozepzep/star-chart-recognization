import csv
import numpy as np
import pytest
from astropy.wcs import WCS

from src.astrometry.catalog_match import calibrate_catalog
from src.config import load_config
from src.detect.segmentation import SourceTable


def test_offline_catalog_recovers_blind_plate_and_zero_point_with_photometric_outliers(tmp_path):
    rng = np.random.default_rng(104)
    n = 100
    xy = rng.uniform(300, 3800, (n, 2))
    magnitudes = np.linspace(7, 12, n)
    scale, angle = 8.4, .4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    tangent = (xy-2048) @ rotation * scale
    wcs = WCS(naxis=2)
    wcs.wcs.crval = [129, -2]
    wcs.wcs.crpix = [1, 1]
    wcs.wcs.cdelt = [1/3600, 1/3600]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    sky = wcs.all_pix2world(tangent, 0)
    rate = 10**((20-magnitudes)/2.5)
    rate[60:64] *= 3  # Variables/blends may spoil flux even with correct position.
    noisy = xy+rng.normal(0, .02, xy.shape)
    table = SourceTable(0, *noisy.T, rate*1.5, np.ones(n)*1000, np.ones(n)*1.2, np.ones(n, dtype=int)*10)
    path = tmp_path/'catalog.csv'
    with path.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['source_id', 'ra_deg', 'dec_deg', 'gmag'])
        writer.writerows((str(1000000000000000000+i), *sky[i], magnitudes[i]) for i in range(n))
    zp, report = calibrate_catalog(table, 1.5, path, [129, -2], load_config()['catalog_match'])
    assert zp.value == pytest.approx(20, abs=.01)
    assert zp.n_points == 96
    assert report['matched_stars'] == 100
    assert report['astrometric_rms_arcsec'] < .4
    assert report['approximate_scale_arcsec_px'] == pytest.approx(scale, rel=.003)
    assert len(report['catalog_sha256']) == 64
    assert all(isinstance(r['source_id'], str) for r in report['calibration_pairs'])


def test_invalid_exposure_rejected_before_catalog_io(tmp_path):
    with pytest.raises(ValueError, match='曝光'):
        calibrate_catalog(SourceTable.empty(), 0, tmp_path/'missing.csv', [0, 0], load_config()['catalog_match'])
