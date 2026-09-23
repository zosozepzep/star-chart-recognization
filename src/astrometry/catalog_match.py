"""Offline Gaia-reference calibration after independent image detection.

Triangle matching supplies a blind scale/orientation initialization. Unique
catalog matches then constrain a quadratic tangent-plane plate fit. The G-band
zero point is approximate for a camera with an unspecified spectral response.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from astropy.wcs import WCS

from src.astrometry.photometry import ZeroPoint


def triangle_features(xy, neighbors):
    triangles = set()
    _, indices = cKDTree(xy).query(xy, k=min(neighbors, len(xy)))
    for i, near in enumerate(indices):
        for a, b in itertools.combinations(near[1:], 2):
            triangles.add(tuple(sorted((i, int(a), int(b)))))
    tri = np.array(sorted(triangles), dtype=int)
    if not len(tri):
        raise ValueError('星点不足以建立三角形匹配')
    xyz = xy[tri]
    lengths = np.stack([np.linalg.norm(xyz[:, 1]-xyz[:, 2], axis=1),
                        np.linalg.norm(xyz[:, 0]-xyz[:, 2], axis=1),
                        np.linalg.norm(xyz[:, 0]-xyz[:, 1], axis=1)], axis=1)
    order = np.argsort(lengths, axis=1)
    lengths = np.take_along_axis(lengths, order, axis=1)
    ratios = lengths[:, :2] / np.maximum(lengths[:, 2:], 1e-12)
    good = (ratios[:, 0] > .2) & (ratios.sum(axis=1) > 1.05)
    return ratios[good], np.take_along_axis(tri, order, axis=1)[good]


def initial_plate(points, catalog, config):
    af, ai = triangle_features(points, 8)
    bf, bi = triangle_features(catalog[:config['seed_catalog_stars']], 10)
    matches = cKDTree(bf).query_ball_point(af, config['triangle_tolerance'])
    tree = cKDTree(catalog)
    best, matrix = 0, None
    for i, near in enumerate(matches):
        for j in near:
            try:
                candidate = np.linalg.solve(np.column_stack((points[ai[i]], np.ones(3))), catalog[bi[j]])
            except np.linalg.LinAlgError:
                continue
            singular = np.linalg.svd(candidate[:2], compute_uv=False)
            if singular[1] <= 0 or singular[0]/singular[1] > 1.05:
                continue
            if not config['min_scale_arcsec_px'] <= singular.mean() <= config['max_scale_arcsec_px']:
                continue
            projected = np.column_stack((points, np.ones(len(points)))) @ candidate
            distances, idx = tree.query(projected)
            count = len(set(idx[distances < config['initial_radius_px'] * singular.mean()]))
            if count > best:
                best, matrix = count, candidate
        if best >= config['early_stop_matches']:
            break
    if best < config['min_initial_matches']:
        raise ValueError(f'星表初始匹配不足：{best}，保留未定标结果')
    return matrix


def plate_design(xy, center, normalization):
    x, y = ((np.asarray(xy)-center)/normalization).T
    return np.column_stack((np.ones(len(x)), x, y, x*x, x*y, y*y))


def calibrate_catalog(table, exposure_s, path, center_radec, config):
    if not np.isfinite(exposure_s) or exposure_s <= 0:
        raise ValueError('星表测光曝光时间必须为有限正数')
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as fh:
        rows = list(csv.DictReader(fh))
    values = np.array([[float(r[k]) for k in ('ra_deg', 'dec_deg', 'gmag')] for r in rows])
    if len(values) < config['min_matches'] or not np.isfinite(values).all():
        raise ValueError('星表行数不足或含无效坐标/星等')
    if np.any((values[:, 0] < 0) | (values[:, 0] >= 360) | (abs(values[:, 1]) > 90)):
        raise ValueError('星表赤经或赤纬超出范围')
    order = np.argsort(values[:, 2], kind='stable')
    values, rows = values[order], [rows[i] for i in order]
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1, 1]
    wcs.wcs.crval = center_radec
    wcs.wcs.cdelt = [1/3600, 1/3600]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    sky = wcs.all_world2pix(values[:, :2], 0)  # tangent-plane arcseconds
    compact = table.select((table.npix <= 200) & (table.elongation < 1.6) &
                           (table.flux > 0) & np.isfinite(table.flux)).brightest(config['max_image_stars'])
    if len(compact) < config['min_matches']:
        raise ValueError('未饱和、紧致星点不足以定标')
    xy = compact.xy
    matrix = initial_plate(xy[:config['seed_image_stars']], sky, config)
    scale = float(np.mean(np.linalg.svd(matrix[:2], compute_uv=False)))
    projected = np.column_stack((xy, np.ones(len(xy)))) @ matrix
    center, normalization = np.mean(xy, axis=0), max(float(np.ptp(xy, axis=0).max()/2), 1.)
    design = plate_design(xy, center, normalization)
    tree = cKDTree(sky)
    limit = config['initial_radius_px'] * scale
    for _ in range(8):
        distances, idx = tree.query(projected)
        _, reverse = cKDTree(projected).query(sky)
        good = (distances < limit) & (reverse[idx] == np.arange(len(xy)))
        if good.sum() < config['min_matches']:
            raise ValueError(f'星表一对一匹配不足：{good.sum()}')
        coefficients, _, rank, _ = np.linalg.lstsq(design[good], sky[idx[good]], rcond=None)
        if rank < 6:
            raise ValueError('星表匹配几何退化')
        projected = design @ coefficients
        residual = np.linalg.norm(projected[good] - sky[idx[good]], axis=1)
        median = np.median(residual)
        mad = 1.4826*np.median(np.abs(residual-median))
        limit = min(config['initial_radius_px']*scale, max(config['clip_floor_arcsec'], median+3*mad))
    astrometric_rms = float(np.sqrt(np.mean(residual**2)))
    if astrometric_rms > config['max_astrometric_rms_arcsec']:
        raise ValueError(f'星表位置拟合 RMS {astrometric_rms:.2f} arcsec 超限')
    rates = compact.flux[good] / exposure_s
    delta = values[idx[good], 2] + 2.5*np.log10(rates)
    keep = np.ones(len(delta), dtype=bool)
    for _ in range(5):
        median = np.median(delta[keep])
        mad = 1.4826*np.median(np.abs(delta[keep]-median))
        keep = np.abs(delta-median) <= max(.05, 3*mad)
        if keep.sum() < config['min_matches']:
            raise ValueError('星表测光一致配对不足')
    zp = ZeroPoint(float(np.median(delta[keep])), float(np.std(delta[keep])), int(keep.sum()),
                   source='Gaia DR3 G reference; camera passband unspecified', exposure_s=1.)
    if zp.std > config['max_photometric_scatter_mag']:
        raise ValueError(f'星表零点散度 {zp.std:.2f} mag 超限')
    source_indices = np.flatnonzero(good)
    pairs = [dict(source_id=rows[idx[i]]['source_id'], x=float(xy[i, 0]), y=float(xy[i, 1]),
                  gmag=float(values[idx[i], 2]), aperture_rate=float(compact.flux[i]/exposure_s),
                  zero_point=float(delta[j]), photometry_used=bool(keep[j]))
             for j, i in enumerate(source_indices)]
    info = dict(catalog_file=path.name, catalog_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                matched_stars=int(good.sum()), astrometric_rms_arcsec=astrometric_rms,
                approximate_scale_arcsec_px=scale, center_radec_deg=list(center_radec),
                plate_center_px=center.tolist(), plate_normalization_px=normalization,
                plate_coefficients=coefficients.tolist(), calibration_pairs=pairs,
                note='Gaia DR3 ICRS epoch 2016 positions; unpropagated proper motion and unknown camera passband are systematic limits. G-reference magnitudes are approximate, not standard V magnitudes.')
    return zp, info
