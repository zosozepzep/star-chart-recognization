"""Config-driven inference. No imports from truth/validation calibration code."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import numpy as np

from src.calib.background import model_background
from src.calib.hotpixel import build_from_sequence
from src.calib.segment import tracking_segment
from src.detect.centroid import aperture_flux
from src.detect.segmentation import SourceTable, detect_sources_in_frame
from src.register.solver import register_sequence
from src.target.dual_frame import find_targets

logger = logging.getLogger(__name__)


@dataclass
class DetectionRun:
    frames: list
    reference_frame: int
    detections: dict
    aperture_rates: dict
    stars: dict
    raw_star_count: int
    registration: object
    targets: list
    diagnostics: dict | None = None

    def target_flux_rates(self, track, frames) -> np.ndarray:
        """Join calibration rows by frame id and actual source coordinates."""
        positions = dict(zip(track.frames, track.xy_det))
        rates = []
        for frame in frames:
            xy = positions[int(frame)]
            distance = np.linalg.norm(self.detections[int(frame)].xy - xy, axis=1)
            index = int(np.argmin(distance))
            if distance[index] > 1e-6:
                raise ValueError("目标测光坐标与探测源表不一致")
            rates.append(self.aperture_rates[int(frame)][index])
        return np.asarray(rates)


def _with_flux(table, flux):
    return SourceTable(frame=table.frame, x=table.x, y=table.y, flux=flux,
                       peak=table.peak, elongation=table.elongation, npix=table.npix)


def analyze_sequence(sequence, config, *, reference_frame=None, frame_range=None) -> DetectionRun:
    if getattr(sequence, 'observation_mode', 'ground') == 'space':
        from src.space_pipeline import analyze_space_sequence
        return analyze_space_sequence(sequence, config, reference_frame=reference_frame, frame_range=frame_range)
    segment_cfg = config["segment"]
    if frame_range is None:
        segment = tracking_segment(sequence.headers, window=segment_cfg["window"],
                                   std_threshold=segment_cfg["std_threshold_deg"],
                                   min_length=segment_cfg["min_length"])
        frames = segment.frames
    else:
        start, end = frame_range
        if not 0 <= start < end < len(sequence):
            raise ValueError("帧范围应满足 0 <= START < END < 序列长度（含 END）")
        frames = list(range(start, end + 1))
    reference = frames[len(frames) // 2] if reference_frame is None else reference_frame
    if reference not in frames:
        raise ValueError("指定图像不在分析帧范围中")
    budget = min(int(config["repeatability"]["frames"]), len(frames))
    min_hits = int(config["star_report"]["min_hits"])
    if not 2 <= min_hits <= budget:
        raise ValueError("序列不足以执行配置中的恒星跨帧确认")
    # Use a local contiguous window around the chosen image to limit field roll-out.
    start = max(0, min(frames.index(reference) - budget // 2, len(frames) - budget))
    verification = set(frames[start:start + budget])
    exposures = [sequence.headers[f].exposure_s for f in frames]
    if any(not np.isfinite(e) or e <= 0 for e in exposures):
        raise ValueError("所有分析帧都必须有有效的正曝光时间")
    hot = dict(config["hotpixel"])
    reject_radius = hot.pop("reject_radius_px")
    hpm = build_from_sequence(sequence, frames, **hot)
    det_cfg = config["detect"]
    radius = float(det_cfg["aperture_radius_px"])
    detections, rates, stars = {}, {}, {}
    raw_count = 0
    for step, frame in enumerate(frames, 1):
        image = sequence.image(frame)
        model = model_background(image, **config["background"])
        kw = dict(npixels=det_cfg["npixels"], hot_clusters=hpm.clusters,
                  reject_radius=reject_radius, frame=frame)
        detected = detect_sources_in_frame(image, model, n_sigma=det_cfg["search_n_sigma"], **kw)
        detections[frame] = detected
        residual = model.subtract(image)
        rates[frame] = aperture_flux(residual, detected.xy, radius=radius) / sequence.headers[frame].exposure_s
        if frame in verification:
            table = detect_sources_in_frame(image, model, n_sigma=det_cfg["report_n_sigma"], **kw)
            if frame == reference:
                raw_count = len(table)
            # Truncated apertures make edge stars appear artificially faint.
            ny, nx = image.shape
            valid = ((table.x >= radius) & (table.x <= nx - 1 - radius) &
                     (table.y >= radius) & (table.y <= ny - 1 - radius))
            table = table.select(valid)
            stars[frame] = _with_flux(table, aperture_flux(residual, table.xy, radius=radius))
        logger.info("分析进度 %d/%d：f%d，%d 个搜索源", step, len(frames), frame, len(detected))
    registration = register_sequence(detections, frames, config=config["register"])
    targets = find_targets(detections, frames, registration, config=config["target"])
    return DetectionRun(frames, reference, detections, rates, stars, raw_count, registration, targets)
