"""Experimental pixel-based exposure-trail localization, never given truth.

Track velocities initialize a fixed-length/direction exposure model. Only
existing observations are fitted; failed fits retain their measured positions.
Production entry points do not import this module.
"""
from __future__ import annotations

from collections import Counter
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.special import erf

from src.detect.segmentation import SourceTable
from src.target.space_motion import find_space_targets, fit_motion
from src.target.trail_check import check_tracks


def trail_density(x, y, center, velocity, exposure_s, sigma):
    """Unit-flux circular Gaussian convolved with a uniform exposure segment.

    Evaluate the continuous analytic model at pixel centers. No injection
    renderer, injected PSF width, brightness or position is used here.
    """
    velocity = np.asarray(velocity, dtype=float)
    speed = float(np.linalg.norm(velocity))
    dx, dy = x - center[0], y - center[1]
    length = speed * exposure_s
    if length < 1e-5:
        return np.exp(-(dx*dx + dy*dy)/(2*sigma*sigma))/(2*np.pi*sigma*sigma)
    ux, uy = velocity / speed
    along, across = dx*ux + dy*uy, -dx*uy + dy*ux
    scale = np.sqrt(2)*sigma
    ends = erf((along+length/2)/scale) - erf((along-length/2)/scale)
    return np.exp(-across*across/(2*sigma*sigma))*ends/(2*length*np.sqrt(2*np.pi)*sigma)


def fit_trail_center(image, position, velocity, exposure_s, config, *,
                     saturation_adu, metadata_rows=0, stationary_xy=()):
    """Fit center, flux, PSF width and a background plane on original pixels.

    Known stationary detections and invalid pixels are masked. Signal support,
    residual, signal-to-noise and boundary checks reject unreliable fits. SNR
    is a template/noise diagnostic, not a calibrated probability.
    """
    position, velocity = np.asarray(position, float), np.asarray(velocity, float)
    if (position.shape != (2,) or velocity.shape != (2,)
            or not np.isfinite(np.r_[position, velocity, exposure_s]).all()
            or exposure_s <= 0):
        raise ValueError('Finite xy, velocity and positive exposure required')
    length = float(np.linalg.norm(velocity)*exposure_s)
    evidence = dict(original_xy=position.tolist(), velocity_det_px_s=velocity.tolist(),
                    expected_length_px=length, accepted=False)
    if length < config['min_length_px']:
        return {**evidence, 'reason': 'unresolved_trail'}
    shift = min(config['max_shift_px'], length/2 + config['sigma_initial_px'])
    half = int(np.ceil(length/2 + shift + config['window_padding_px']))
    if half > config['max_half_window_px']:
        return {**evidence, 'reason': 'window_limit'}
    px, py = np.rint(position).astype(int)
    x0, x1, y0, y1 = px-half, px+half+1, py-half, py+half+1
    if x0 < 0 or y0 < metadata_rows or x1 > image.shape[1] or y1 > image.shape[0]:
        return {**evidence, 'reason': 'image_edge'}
    patch = np.asarray(image[y0:y1, x0:x1], float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    invalid = ~np.isfinite(patch) | (patch < 0) | (patch >= saturation_adu)
    valid = ~maximum_filter(invalid, size=3)
    for sx, sy in stationary_xy:
        if x0-4 <= sx <= x1+4 and y0-4 <= sy <= y1+4:
            valid &= (xx-sx)**2 + (yy-sy)**2 > config['stationary_mask_radius_px']**2
    evidence['valid_fraction'] = float(valid.mean())
    if valid.mean() < config['min_valid_fraction']:
        return {**evidence, 'reason': 'masked_pixels'}
    z, x, y = patch[valid], xx[valid], yy[valid]
    median = float(np.median(z))
    noise = max(1., float(1.4826*np.median(np.abs(z-median))))
    bx, by = (x-position[0])/half, (y-position[1])/half
    initial = trail_density(x, y, position, velocity, exposure_s, config['sigma_initial_px'])
    flux = max(1., float(initial @ (z-median) / max(initial @ initial, 1e-12)))
    # Flux is parameterized in units of the measured initialization for scaling.
    p0 = [0., 0., 1., config['sigma_initial_px'], median, 0., 0.]
    lower = [-shift, -shift, 0., config['sigma_min_px'], -np.inf, -np.inf, -np.inf]
    upper = [shift, shift, np.inf, config['sigma_max_px'], np.inf, np.inf, np.inf]

    def residual(p):
        model = flux*p[2]*trail_density(x, y, position+p[:2], velocity, exposure_s, p[3])
        return (model + p[4] + p[5]*bx + p[6]*by - z)/noise

    result = least_squares(residual, p0, bounds=(lower, upper), loss='soft_l1',
                           f_scale=config['robust_scale'], max_nfev=config['max_nfev'],
                           x_scale='jac')
    p = result.x
    density = trail_density(x, y, position+p[:2], velocity, exposure_s, p[3])
    # Remove the background plane's subspace when estimating template SNR.
    plane = np.column_stack((np.ones(len(x)), bx, by))
    orthogonal = density - plane @ np.linalg.lstsq(plane, density, rcond=None)[0]
    snr = float(flux*p[2]*np.linalg.norm(orthogonal)/noise)
    support = float(density.sum())
    rms = float(np.sqrt(np.mean(result.fun**2)))
    margin = config['bound_margin']
    at_bound = (np.any(np.abs(p[:2]) >= shift*(1-margin))
                or p[3] <= config['sigma_min_px']*(1+margin)
                or p[3] >= config['sigma_max_px']*(1-margin))
    reason = ('optimizer_failed' if not result.success else
              'parameter_bound' if at_bound else
              'masked_signal' if support < config['min_signal_support'] else
              'low_snr' if snr < config['min_snr'] else
              'model_mismatch' if rms > config['max_residual_rms_noise'] else 'accepted')
    return dict(evidence, reason=reason, fitted_xy=(position+p[:2]).tolist(),
                shift_px=float(np.linalg.norm(p[:2])), sigma_px=float(p[3]),
                fitted_flux_adu=float(flux*p[2]), noise_adu=noise, snr=snr,
                signal_support=support, residual_rms_noise=rms, nfev=int(result.nfev),
                accepted=reason == 'accepted')


def stationary_positions(run, motion):
    sky = {f: run.registration.to_sky(f, run.detections[f].xy) for f in run.frames}
    trees = {f: cKDTree(xy) for f, xy in sky.items()}
    result = {}
    for f in run.frames:
        hits = np.ones(len(sky[f]), dtype=int)
        for g in run.frames:
            if g != f:
                hits += trees[g].query(sky[f])[0] < motion['stationary_radius_px']
        result[f] = run.detections[f].xy[hits >= motion['stationary_min_hits']]
    return result


def refine_and_relink(sequence, run, hypotheses, config, fit_config):
    """Relocate raw track observations, then run unchanged linking/trail checks.

    Untracked detections and failed fits remain unchanged. No gap filling or
    forced photometry is performed. Registration and original peak/area/flux
    measurements are retained; fitted fluxes are diagnostic fields only.
    """
    motion = config['space']['motion']
    stationary = stationary_positions(run, motion) if hypotheses else {}
    times = sequence.times().unix
    proposals = {f: run.detections[f].xy.copy() for f in run.frames}
    evidence = []
    jobs = {}
    for k, verdict in enumerate(hypotheses):
        track = verdict.track
        velocity = fit_motion(times[track.frames], track.xy_det)[0][1]
        for f, xy in zip(track.frames, track.xy_det):
            distances = np.linalg.norm(run.detections[f].xy-xy, axis=1)
            idx = int(np.argmin(distances))
            if distances[idx] > 1e-7:
                raise ValueError('Track observation missing from source table')
            jobs.setdefault(f, []).append((k, idx, xy, velocity))
    for f, frame_jobs in sorted(jobs.items()):
        image = sequence.image(f)
        for k, idx, xy, velocity in frame_jobs:
            result = fit_trail_center(image, xy, velocity, sequence.headers[f].exposure_s,
                                      fit_config, saturation_adu=config['space']['saturation_adu'],
                                      metadata_rows=config['space']['metadata_rows'],
                                      stationary_xy=stationary[f])
            evidence.append(dict(frame=f, source_index=idx, hypothesis_index=k, **result))
            if result['accepted']:
                proposals[f][idx] = result['fitted_xy']
    tables = {f: SourceTable(f, proposals[f][:, 0], proposals[f][:, 1],
                             t.flux.copy(), t.peak.copy(), t.elongation.copy(), t.npix.copy())
              for f, t in run.detections.items()}
    targets, counts = find_space_targets(tables, run.frames, run.registration, times, motion)
    targets, rejected = check_tracks(sequence, targets, config['space']['trail_check'])
    return targets, dict(fits=evidence, fit_reasons=dict(Counter(r['reason'] for r in evidence)),
                         candidate_counts=counts, rejected_trails=rejected)
