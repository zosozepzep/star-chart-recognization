"""Official space-image path: telemetry masking, dense-field registration, motion."""
from __future__ import annotations

import logging
import numpy as np
from scipy.ndimage import maximum_filter

from src.calib.background import model_background
from src.detect.segmentation import detect_sources_in_frame
from src.detect.centroid import aperture_flux
from src.register.space import register_star_field
from src.target.space_motion import find_space_targets
from src.target.trail_check import check_tracks
from src.parallel import execution_plan, frame_input, map_frames, read_frame

logger = logging.getLogger(__name__)


def prepare_space_image(image, config):
    image = np.array(image, dtype=float, copy=True)
    invalid = ~np.isfinite(image) | (image < 0) | (image >= config['saturation_adu'])
    invalid[:config['metadata_rows']] = True
    median = np.median(image[~invalid])
    image[invalid] = median
    excluded = maximum_filter(invalid, size=2 * int(config['invalid_margin_px']) + 1)
    return image, excluded


def valid_sources(table, excluded, radius):
    ny, nx = excluded.shape
    x, y = np.rint(table.x).astype(int), np.rint(table.y).astype(int)
    good = ((table.x >= radius) & (table.x < nx - radius) &
            (table.y >= radius) & (table.y < ny - radius))
    good &= ~excluded[np.clip(y, 0, ny - 1), np.clip(x, 0, nx - 1)]
    return table.select(good)


def _analyze_space_frame(job):
    from src.pipeline import _with_flux
    source, f, exposure, verify, config = job
    conf, dc = config['space'], config['detect']
    radius = dc['aperture_radius_px']
    image, excluded = prepare_space_image(read_frame(source), conf)
    model = model_background(image, **config['background'])
    table = detect_sources_in_frame(image, model, n_sigma=dc['search_n_sigma'], npixels=dc['npixels'], frame=f)
    table = valid_sources(table, excluded, radius)
    residual = model.subtract(image)
    rates = aperture_flux(residual, table.xy, radius=radius) / exposure
    stars, raw_count = None, 0
    if verify:
        report = detect_sources_in_frame(image, model, n_sigma=dc['report_n_sigma'], npixels=dc['npixels'], frame=f)
        report = valid_sources(report, excluded, radius)
        raw_count = len(report)
        # Elongated transients are not point-like star candidates.
        report = report.select(report.elongation <= conf['star_max_elongation'])
        stars = _with_flux(report, aperture_flux(residual, report.xy, radius=radius))
    diagnostics = dict(frame=f, search_sources=len(table), excluded_pixels=int(excluded.sum()),
                       background_adu=float(np.median(model.background)),
                       noise_rms_adu=float(np.median(model.rms)))
    return f, table, rates, stars, raw_count, diagnostics


def analyze_space_sequence(sequence, config, *, reference_frame=None, frame_range=None):
    from src.pipeline import DetectionRun
    conf = config['space']
    start, end = (0, len(sequence) - 1) if frame_range is None else frame_range
    if not 0 <= start < end < len(sequence):
        raise ValueError('天基帧范围应满足 0 <= START < END < 序列长度')
    frames = list(range(start, end + 1))
    reference = frames[len(frames) // 2] if reference_frame is None else reference_frame
    if reference not in frames:
        raise ValueError('指定图像不在分析帧范围中')
    budget = min(int(config['repeatability']['frames']), len(frames))
    if budget < config['star_report']['min_hits'] or len(frames) < conf['motion']['min_track_frames']:
        raise ValueError('天基序列不足以执行恒星核验与运动轨迹确认')
    exposures = np.array([sequence.headers[f].exposure_s for f in frames])
    if not np.isfinite(exposures).all() or np.any(exposures <= 0):
        raise ValueError('曝光时间必须为有限正数')
    times = sequence.times().unix
    if np.any(np.diff(times[frames]) <= 0):
        raise ValueError('观测时间必须严格递增')
    ws = max(0, min(frames.index(reference) - budget // 2, len(frames) - budget))
    verification = frames[ws:ws + budget]
    tables, rates, stars, diagnostics = {}, {}, {}, []
    raw_count = 0
    plan = execution_plan(config, len(frames))
    logger.info('CPU 执行：%s，%d 个工作单元', plan['executor'], plan['workers'])
    jobs = ((frame_input(sequence, f), f, sequence.headers[f].exposure_s,
             f in verification, config) for f in frames)
    for n, (f, table, flux_rates, frame_stars, count, diagnostic) in enumerate(
            map_frames(_analyze_space_frame, jobs, plan), 1):
        tables[f] = table
        rates[f] = flux_rates
        if frame_stars is not None:
            stars[f] = frame_stars
            if f == reference:
                raw_count = count
        diagnostics.append(diagnostic)
        logger.info('天基分析 %d/%d：f%d，%d 个搜索源', n, len(frames), f, len(table))
    logger.info('阶段：恒星背景配准')
    registration = register_star_field(tables, frames, conf['registration'])
    logger.info('阶段：多帧运动关联')
    targets, candidate_counts = find_space_targets(tables, frames, registration, times, conf['motion'])
    logger.info('阶段：曝光拖尾核验（%d 条初步轨迹）', len(targets))
    targets, rejected = check_tracks(sequence, targets, conf['trail_check'])
    return DetectionRun(frames, reference, tables, rates, stars, raw_count, registration, targets,
                        dict(mode='space', per_frame=diagnostics, motion_candidates=candidate_counts,
                             metadata_rows_excluded=conf['metadata_rows'],
                             rejected_motion_hypotheses=rejected,
                             pixel_policy='signed FITS; negative/saturated pixels masked, no unsigned reinterpretation',
                             hotpixel_policy='no temporal flatness rejection in a stationary star field'),
                        execution=plan)
