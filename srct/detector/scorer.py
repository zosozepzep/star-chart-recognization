from __future__ import annotations

import numpy as np

from detector.feature_extractor import SourceDetection


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class SourceScorer:
    def __init__(self, acceptance_threshold: float = 0.65):
        self.acceptance_threshold = acceptance_threshold

    def score_sources(self, detections):
        for det in detections:
            det.score = self.score_single(det)
        return detections

    def score_single(self, det: SourceDetection) -> float:
        snr_score = self._snr_score(det.snr)
        psf_score = self._psf_score(det)
        shape_score = self._shape_score(det)
        vote_score = self._vote_score(det)
        edge_score = self._edge_score(det)
        flag_score = self._flag_score(det)

        score = (
            0.30 * snr_score +
            0.25 * psf_score +
            0.15 * shape_score +
            0.15 * vote_score +
            0.10 * edge_score +
            0.05 * flag_score
        )

        return float(np.clip(score, 0.0, 1.0))

    def accepted(self, det: SourceDetection) -> bool:
        return det.score >= self.acceptance_threshold

    def _snr_score(self, snr):
        return sigmoid((snr - 3.0) / 2.0)

    def _psf_score(self, det):
        if not det.fit_success:
            return 0.2

        residual_term = np.exp(-5.0 * det.residual_ratio)

        if np.isfinite(det.elongation):
            elong_term = np.exp(-abs(det.elongation - 1.0))
        else:
            elong_term = 0.0

        return 0.7 * residual_term + 0.3 * elong_term

    def _shape_score(self, det):
        if not np.isfinite(det.ellipticity):
            return 0.5
        return np.exp(-3.0 * det.ellipticity)

    def _vote_score(self, det):
        n = len(set(det.engine_votes))
        return min(1.0, 0.4 + 0.3 * n)

    def _edge_score(self, det):
        if det.flags.get("edge_truncated", False):
            return 0.2
        return 1.0

    def _flag_score(self, det):
        penalty_flags = [
            "feature_extraction_failed",
            "psf_fit_failed",
        ]
        if any(det.flags.get(k, False) for k in penalty_flags):
            return 0.2
        return 1.0