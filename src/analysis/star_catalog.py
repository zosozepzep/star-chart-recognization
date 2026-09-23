"""Reference-frame star candidates confirmed by one-to-one matches in sky pixels.

The faintest candidate is an actual row, not a magnitude percentile. Counts are
conservative detections at the stated threshold, not a claim of completeness.
This module never reads truth files. Fluxes must use the same aperture as the ZP.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial import cKDTree

from src.astrometry.photometry import instrumental_mag
from src.pointset import as_xy


def confirmation_hits(positions, reference: int, radius: float) -> np.ndarray:
    """Count at most one mutual-nearest detection per frame for each reference row."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("匹配半径必须是有限正数")
    arrays = {f: as_xy(xy) for f, xy in positions.items()}
    ref = arrays[reference]
    if any(not np.isfinite(xy).all() for xy in arrays.values()):
        raise ValueError("恒星坐标不能包含 NaN 或 Infinity")
    hits = np.ones(len(ref), dtype=int)
    if not len(ref):
        return hits
    ref_tree = cKDTree(ref)
    for frame, xy in arrays.items():
        if frame == reference or not len(xy):
            continue
        distance, nearest = cKDTree(xy).query(ref)
        _, reverse = ref_tree.query(xy)
        hits += (distance <= radius) & (reverse[nearest] == np.arange(len(ref)))
    return hits


def star_catalog(tables, registration, reference: int, *, min_hits: int,
                 radius: float, exposure_s: float, zero_point=None) -> dict:
    """Select the faintest positive-flux confirmed source in the designated image.

The supplied tables have background-subtracted fixed-aperture ADU fluxes. ZP,
when supplied, must calibrate ADU/s (exposure_s=1). Missing calibration produces
null magnitudes, never an assumed absolute zero point.
"""
    if isinstance(min_hits, bool) or not isinstance(min_hits, int) or not 2 <= min_hits <= len(tables):
        raise ValueError("确认帧数必须为 2 到核验帧数之间的整数")
    if not math.isfinite(exposure_s) or exposure_s <= 0:
        raise ValueError("曝光时间必须是有限正数")
    if zero_point is not None and (zero_point.exposure_s != 1.0 or not math.isfinite(zero_point.value)):
        raise ValueError("星等零点必须按 ADU/s 定标（exposure_s=1）且数值有限")
    # Invalid measurements cannot confirm a star in another frame.
    clean = {f: t.select(np.isfinite(t.flux) & (t.flux > 0)) for f, t in tables.items()}
    positions = {f: registration.to_sky(f, t.xy) for f, t in clean.items()}
    hits = confirmation_hits(positions, reference, radius)
    table = clean[reference]
    mags = instrumental_mag(table.flux, exposure_s=exposure_s)
    records = []
    for i, row in enumerate(table.to_records()):
        records.append({
            "source_id": i + 1, "frame": reference,
            "x": row["x"], "y": row["y"], "aperture_flux_adu": row["flux"],
            "instrumental_mag_rate": float(mags[i]),
            "mag": float(mags[i] + zero_point.value) if zero_point is not None else None,
            "hits": int(hits[i]), "confirmed": bool(hits[i] >= min_hits),
        })
    confirmed = [r for r in records if r["confirmed"]]
    faintest = min(confirmed, key=lambda r: r["aperture_flux_adu"]) if confirmed else None
    return {
        "reference_frame": reference, "verification_frames": sorted(tables),
        "min_hits": min_hits, "match_radius_px": radius,
        "n_positive_aperture_sources": len(records), "n_confirmed_stars": len(confirmed),
        "faintest_confirmed_star": faintest,
        "calibration": zero_point.to_dict() if zero_point is not None else None,
        "magnitude_system": "truth-referenced; passband unspecified" if zero_point is not None else "instrumental only",
        "note": "指定图像中通过跨帧核验的恒星候选；不代表全部恒星或探测完备性极限。",
        "sources": records,
    }
