"""Bounded-cost registration for dense, overlapping space-based star fields."""
import numpy as np
from scipy.spatial import cKDTree

from src.register.solver import RegistrationError, RegistrationResult, PairSolution, solve_pair
from src.register.transform import apply_transform, fit_similarity, decompose


def register_star_field(tables, frames, config):
    frames = list(frames)
    cap = int(config['max_sources'])
    compact = {f: t.select((t.elongation < config['max_elongation']) &
                          (t.npix <= config['max_npixels'])).brightest(cap)
               for f, t in tables.items()}
    ref = frames[0]
    a = compact[ref].xy
    matrices, pairs = {ref: np.eye(3)}, []
    for f in frames[1:]:
        # Direct fits avoid drift accumulation over this short overlapping field.
        initial = solve_pair(compact[ref], compact[f], min_inliers=config['min_inliers'])
        b = compact[f].xy
        matrix = initial.matrix
        for _ in range(5):
            projected = apply_transform(matrix, b)
            dist, idx = cKDTree(a).query(projected)
            _, reverse = cKDTree(projected).query(a)
            good = (dist < config['match_radius_px']) & (reverse[idx] == np.arange(len(b)))
            if good.sum() < config['min_inliers']:
                raise RegistrationError(f'天基 f{f} 配准内点不足：{good.sum()}')
            median = np.median(dist[good])
            scatter = 1.4826 * np.median(np.abs(dist[good] - median))
            good &= dist <= max(config['clip_floor_px'], median + 3 * scatter)
            if good.sum() < config['min_inliers']:
                raise RegistrationError(f'天基 f{f} 配准剪裁后内点不足')
            matrix = fit_similarity(b[good], a[idx[good]])
        residual = apply_transform(matrix, b[good]) - a[idx[good]]
        rms = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
        if rms > config['max_rms_px']:
            raise RegistrationError(f'天基 f{f} 配准 RMS {rms:.3f} 超限')
        dec = decompose(matrix)
        matrices[f] = matrix
        pairs.append(PairSolution(ref, f, matrix, int(good.sum()), rms,
                                  dec['rotation_deg'], (dec['tx'], dec['ty'])))
    return RegistrationResult(ref, frames, matrices, pairs)
