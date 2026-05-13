from __future__ import annotations
import numpy as np
from photutils.detection import DAOStarFinder
from photutils.segmentation import detect_sources
from detector.feature_extractor import SourceDetection  # 预留给后续转换
from scipy.spatial import cKDTree
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

    def generate(self, data_sub: np.ndarray, global_std: float) -> list:
        all_candidates = []
        
        # ---------------------------------------------------------
        # 引擎 A: DAOStarFinder (针对点状星)
        # ---------------------------------------------------------
        dao_thresh = self.dao_threshold_sigma * global_std
        
        # 修复：使用类的属性 self.dao_fwhm
        dao = DAOStarFinder(fwhm=self.dao_fwhm, threshold=dao_thresh)
        dao_tbl = dao.find_stars(data_sub)
        
        if dao_tbl is not None:
            for row in dao_tbl:
                all_candidates.append({
                    'x': row['xcentroid'],
                    'y': row['ycentroid'],
                    'engine_votes': ['DAO']  # 修复：改为列表，键名为 engine_votes
                })

        # ---------------------------------------------------------
        # 引擎 B: Segmentation (针对弥散/拖尾碎片)
        # ---------------------------------------------------------
        seg_thresh = self.seg_threshold_sigma * global_std
        
        seg_img = detect_sources(data_sub, threshold=seg_thresh, npixels=self.npixels)
        
        if seg_img is not None:
            for obj in seg_img.labels:
                slices = seg_img.slices[obj - 1]
                y_center = (slices[0].start + slices[0].stop) / 2.0
                x_center = (slices[1].start + slices[1].stop) / 2.0
                all_candidates.append({
                    'x': x_center,
                    'y': y_center,
                    'engine_votes': ['SEG']  # 修复：改为列表，键名为 engine_votes
                })

# ---------------------------------------------------------
        # 引擎融合与去重
        # ---------------------------------------------------------
        merged_dicts = self._deduplicate(all_candidates)
        
        # 🌟 关键修复：将原生字典转换为强类型的 SourceDetection 对象
        final_candidates = []
        for cand in merged_dicts:
            # 实例化数据契约对象
            det = SourceDetection(x=cand['x'], y=cand['y'])
            # 注入选票信息
            det.engine_votes = cand['engine_votes']
            final_candidates.append(det)
            
        return final_candidates

    def _deduplicate(self, candidates: list) -> list:
        if not candidates:
            return []
        
        # 提取所有坐标
        coords = np.array([[c['x'], c['y']] for c in candidates])
        # 建立空间索引
        tree = cKDTree(coords)
        
        # 找出所有距离在 merge_radius 以内的点对
        # 这会返回一个包含索引对的集合，例如 {(1, 5), (10, 12)}
        pairs = tree.query_pairs(r=self.merge_radius)
        
        # 简单的聚类逻辑：将靠得太近的点合并
        # 为了快速实战，我们直接保留索引较小的那个点
        to_skip = set()
        for i, j in pairs:
            to_skip.add(max(i, j))
            # 将选票合并到保留的点上
            candidates[min(i, j)]['engine_votes'] = list(
                set(candidates[min(i, j)]['engine_votes'] + candidates[max(i, j)]['engine_votes'])
            )
        
        return [c for i, c in enumerate(candidates) if i not in to_skip]