"""Motion relative to registered stars, with time-based prediction and gaps.

Stationary-source removal is followed by deterministic two-point hypotheses.
Accepted tracks need independent detections in several images and a bounded
linear-fit residual. No target locations or expected target counts are supplied.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

from src.target.dual_frame import Track


@dataclass
class MotionVerdict:
    track: Track
    velocity: np.ndarray
    rms_px: float
    time_span_s: float
    trail_evidence: dict | None = None

    def to_dict(self):
        return dict(is_target=True, method='registered linear motion with gap tolerance',
                    frames=[self.track.start, self.track.end], n_frames=self.track.length,
                    velocity_sky_px_per_s=self.velocity.tolist(),
                    speed_sky_px_per_s=float(np.linalg.norm(self.velocity)),
                    fit_rms_px=self.rms_px, time_span_s=self.time_span_s,
                    xy_start=self.track.xy_det[0].tolist(), xy_end=self.track.xy_det[-1].tolist(),
                    reasons=['相对恒星背景运动，多帧独立检出，时间序列线性拟合通过'],
                    identity='unidentified moving-source candidate', trail_evidence=self.trail_evidence)


def fit_motion(times, xy):
    t = np.asarray(times, dtype=float)
    design = np.column_stack((np.ones(len(t)), t - t[0]))
    coefficients = np.linalg.lstsq(design, xy, rcond=None)[0]
    residual = np.linalg.norm(design @ coefficients - xy, axis=1)
    return coefficients, residual


def find_space_targets(tables, frames, registration, times, config):
    frames = list(frames)
    stamps = {f: float(times[f]) for f in frames}
    if any(stamps[b] <= stamps[a] for a, b in zip(frames, frames[1:])):
        raise ValueError('天基运动分析要求严格递增的观测时间')
    xy = {f: registration.to_sky(f, tables[f].xy) for f in frames}
    trees = {f: cKDTree(a) for f, a in xy.items()}
    candidates, hits_by_frame = {}, {}
    for f in frames:
        hits = np.ones(len(xy[f]), dtype=int)
        for g in frames:
            if g != f:
                distance, _ = trees[g].query(xy[f])
                hits += distance < config['stationary_radius_px']
        hits_by_frame[f] = hits
        t = tables[f]
        candidates[f] = np.flatnonzero((hits < config['stationary_min_hits']) &
                                       (t.peak >= config['min_peak_adu']) &
                                       (t.npix <= config['max_npixels']))
    positions = {f: xy[f][idx] for f, idx in candidates.items()}
    ctrees = {f: cKDTree(a) for f, a in positions.items()}
    hypotheses = {}
    radius = config['link_radius_px']
    min_frames = config['min_track_frames']
    max_gap = config['max_gap_frames']
    for si, f in enumerate(frames):
        if len(frames) - si < min_frames:
            break
        for k, point in enumerate(positions[f]):
            for sj in range(si + 1, min(si + max_gap + 2, len(frames))):
                g = frames[sj]
                dt = stamps[g] - stamps[f]
                neighbors = ctrees[g].query_ball_point(point, config['max_speed_px_s'] * dt)
                for j in neighbors:
                    velocity = (positions[g][j] - point) / dt
                    if np.linalg.norm(velocity) < config['min_speed_px_s']:
                        continue
                    path = [(f, k), (g, j)]
                    intercept, t0 = point, stamps[f]
                    missing = 0
                    for h in frames[sj + 1:]:
                        predicted = intercept + velocity * (stamps[h] - t0)
                        dist, n = ctrees[h].query(predicted)
                        if dist > radius:
                            missing += 1
                            if missing > max_gap:
                                break
                            continue
                        path.append((h, int(n)))
                        missing = 0
                        coefficients, _ = fit_motion([stamps[u] for u, _ in path],
                                                      np.array([positions[u][v] for u, v in path]))
                        intercept, velocity = coefficients
                    if len(path) < min_frames:
                        continue
                    path_xy = np.array([positions[u][v] for u, v in path])
                    coeff, residual = fit_motion([stamps[u] for u, _ in path], path_xy)
                    rms = float(np.sqrt(np.mean(residual ** 2)))
                    speed = float(np.linalg.norm(coeff[1]))
                    if (rms > config['max_rms_px'] or residual.max() > radius or
                            not config['min_speed_px_s'] <= speed <= config['max_speed_px_s']):
                        continue
                    key = tuple((u, int(candidates[u][v])) for u, v in path)
                    hypotheses[key] = (coeff[1], rms)
    claimed, verdicts = set(), []
    for path, (velocity, rms) in sorted(hypotheses.items(), key=lambda pair: (-len(pair[0]), pair[1][1], pair[0])):
        if any(node in claimed for node in path):
            continue
        claimed.update(path)
        fs = [f for f, _ in path]
        det = np.array([tables[f].xy[k] for f, k in path])
        sky = np.array([xy[f][k] for f, k in path])
        track = Track(fs, det, sky,
                      np.array([tables[f].flux[k] for f, k in path]),
                      np.array([tables[f].peak[k] for f, k in path]),
                      np.array([tables[f].elongation[k] for f, k in path]))
        verdicts.append(MotionVerdict(track, velocity, rms, stamps[fs[-1]] - stamps[fs[0]]))
    return verdicts, {f: len(candidates[f]) for f in frames}
