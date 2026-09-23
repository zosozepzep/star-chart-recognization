"""Check whether intra-exposure trail shape supports an inter-frame velocity.

For a uniform linear trail blurred by a circular PSF, the difference of the
principal second moments is L²/12. This is a broad consistency test, not a
precision trail-length estimator; blends and variable brightness affect it.
"""
import numpy as np
from scipy.ndimage import label


def measure_trail(image, xy, velocity, exposure_s, config):
    expected = float(np.linalg.norm(velocity) * exposure_s)
    half = min(config['max_half_window_px'], max(12, int(expected / 2 + 10)))
    x, y = np.rint(xy).astype(int)
    y0, y1 = max(1, y-half), min(image.shape[0], y+half+1)
    x0, x1 = max(0, x-half), min(image.shape[1], x+half+1)
    a = np.asarray(image[y0:y1, x0:x1], dtype=float)
    median = np.median(a)
    noise = max(1., 1.4826 * np.median(np.abs(a - median)))
    residual = a - median
    labels, _ = label(residual > config['threshold_sigma'] * noise, structure=np.ones((3, 3)))
    yy, xx = np.indices(a.shape)
    near = (xx+x0-xy[0])**2 + (yy+y0-xy[1])**2 <= 4**2
    choices = np.unique(labels[near & (labels > 0)])
    if not len(choices):
        return dict(expected_length_px=expected, measured_length_px=None, direction_cosine=None, consistent=False)
    chosen = max(choices, key=lambda k: float(residual[(labels == k) & near].sum()))
    keep = labels == chosen
    weights = np.maximum(residual[keep], 0)
    points = np.column_stack((xx[keep], yy[keep]))
    center = np.average(points, axis=0, weights=weights)
    covariance = ((points-center).T * weights) @ (points-center) / weights.sum()
    values, vectors = np.linalg.eigh(covariance)
    length = float(np.sqrt(12 * max(values[1] - values[0], 0)))
    alignment = float(abs(vectors[:, 1] @ velocity) / max(np.linalg.norm(velocity), 1e-12))
    # Point-like slow objects need no resolved streak. Faster candidates must
    # have the elongation implied by their exposure and speed.
    consistent = (expected < config['resolved_length_px'] or
                  (length >= config['min_length_ratio'] * expected and
                   length <= config['max_length_ratio'] * expected + 3 and
                   alignment >= config['min_direction_cosine']))
    return dict(expected_length_px=expected, measured_length_px=length,
                direction_cosine=alignment, consistent=bool(consistent))


def check_tracks(sequence, targets, config):
    measurements = [[] for _ in targets]
    for f in sorted({f for v in targets for f in v.track.frames}):
        image = sequence.image(f)
        for k, verdict in enumerate(targets):
            track = verdict.track
            if f not in track.frames:
                continue
            i = track.frames.index(f)
            left, right = max(0, i-1), min(len(track.frames)-1, i+1)
            dt = (sequence.headers[track.frames[right]].date_obs - sequence.headers[track.frames[left]].date_obs).sec
            velocity = (track.xy_det[right] - track.xy_det[left]) / dt
            measurements[k].append(dict(frame=f, **measure_trail(image, track.xy_det[i], velocity,
                                                                sequence.headers[f].exposure_s, config)))
    accepted, rejected = [], []
    for verdict, rows in zip(targets, measurements):
        fraction = sum(r['consistent'] for r in rows) / len(rows)
        verdict.trail_evidence = dict(consistent_fraction=fraction, frames=rows)
        if fraction >= config['min_consistent_fraction']:
            accepted.append(verdict)
        else:
            rejected.append(verdict.to_dict())
    return accepted, rejected
