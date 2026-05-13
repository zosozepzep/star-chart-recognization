from __future__ import annotations
from photutils.detection import DAOStarFinder
from photutils.segmentation import detect_sources, SourceCatalog  # <-- 新增 SourceCatalog
from typing import List
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.segmentation import detect_sources
from detector.feature_extractor import SourceDetection
import numpy as np

class CandidateGenerator:
    def __init__(
        self,
        dao_fwhm: float = 3.0,
        dao_threshold_sigma: float = 3.0,
        seg_threshold_sigma: float = 2.5,
        npixels: int = 5,
        merge_radius: float = 3.0,
    ):
        self.dao_fwhm = dao_fwhm
        self.dao_threshold_sigma = dao_threshold_sigma
        self.seg_threshold_sigma = seg_threshold_sigma
        self.npixels = npixels
        self.merge_radius = merge_radius

    def generate(self, data: np.ndarray) -> List[SourceDetection]:
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        data_sub = data - median

        candidates = []
        candidates.extend(self._run_dao(data_sub, std))
        candidates.extend(self._run_segmentation(data_sub, std))

        return self._deduplicate(candidates)

    def _run_dao(self, data_sub, std):
        dao = DAOStarFinder(
            fwhm=self.dao_fwhm,
            threshold=self.dao_threshold_sigma * std,
        )

        table = dao(data_sub)
        if table is None:
            return []

        results = []
        for row in table:
            results.append(
                SourceDetection(
                    x=float(row["xcentroid"]),
                    y=float(row["ycentroid"]),
                    engine="DAO",
                    engine_votes=["DAO"],
                )
            )
        return results

    def _run_segmentation(self, data_sub, std):
        threshold = self.seg_threshold_sigma * std

        # 1. 获取像素级分割图 (只包含区域标签)
        segm = detect_sources(
            data_sub,
            threshold=threshold,
            npixels=self.npixels,
        )

        if segm is None:
            return []

        # 2. 核心修复：生成物理特征星表 (计算质心、面积等)
        cat = SourceCatalog(data_sub, segm)

        results = []
        # 直接遍历星表中的每一个物理对象
        for obj in cat:
            results.append(
                SourceDetection(
                    x=float(obj.xcentroid),
                    y=float(obj.ycentroid),
                    engine="SEG",
                    engine_votes=["SEG"],
                )
            )
        return results

    def _deduplicate(self, candidates):
        if not candidates:
            return []

        merged = []

        for cand in candidates:
            found = False

            for existing in merged:
                dist = np.hypot(cand.x - existing.x, cand.y - existing.y)

                if dist < self.merge_radius:
                    existing.engine_votes = list(
                        set(existing.engine_votes + cand.engine_votes)
                    )
                    found = True
                    break

            if not found:
                merged.append(cand)

        return merged