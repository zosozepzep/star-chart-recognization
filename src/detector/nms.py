from __future__ import annotations

from scipy.spatial import cKDTree


def spatial_nms(detections, radius=3.0):
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d.score, reverse=True)

    accepted = []
    accepted_points = []
    tree = None

    for det in detections:
        point = [det.x, det.y]

        if tree is not None:
            idx = tree.query_ball_point(point, radius)
            if idx:
                continue

        accepted.append(det)
        accepted_points.append(point)
        tree = cKDTree(accepted_points)

    return accepted