# src/tracking/physical_validator.py
import numpy as np
from typing import List
from detector.feature_extractor import Track, TrackState

class TrackValidator:
    def __init__(self, min_hits: int = 5, min_r2: float = 0.85, min_score: float = 0.60):
        """
        物理一致性终审法庭
        :param min_hits: 成为真实轨迹的最低命中帧数要求
        :param min_r2: 最低线性度要求
        :param min_score: 综合轨迹得分及格线
        """
        self.min_hits = min_hits
        self.min_r2 = min_r2
        self.min_score = min_score

    def validate_tracks(self, tracks: List[Track]) -> List[Track]:
        """对所有 CONFIRMED 轨迹进行联合审查"""
        valid_tracks = []
        for track in tracks:
            # 1. 基础的 Persistence (持久度) 审查
            if track.hits < self.min_hits:
                track.state = TrackState.LOST
                continue
                
            # 2. 计算物理与统计指标
            self._compute_kinematic_metrics(track)
            self._compute_photometric_metrics(track)
            
            # 3. 综合打分
            final_score = self._compute_track_score(track)
            
            # 4. 判决
            if final_score >= self.min_score and track.linearity_r2 >= self.min_r2:
                valid_tracks.append(track)
            else:
                track.state = TrackState.LOST
                
        return valid_tracks

    def _compute_kinematic_metrics(self, track: Track):
        """计算线性度 (Linearity R^2) 与速度一致性"""
        t = np.array(track.frame_indices)
        x = np.array([d.x for d in track.detections])
        y = np.array([d.y for d in track.detections])
        
        # 如果点太少，无法进行有意义的统计
        if len(t) < 3:
            track.linearity_r2 = 0.0
            return

        # --- 线性度 R^2 计算 (使用对时间的线性回归) ---
        # 真实目标应该是 x(t) 和 y(t) 都高度符合线性
        # 计算 Pearson 相关系数的平方
        r_matrix_x = np.corrcoef(t, x)
        r_matrix_y = np.corrcoef(t, y)
        r2_x = r_matrix_x[0, 1]**2 if not np.isnan(r_matrix_x[0, 1]) else 1.0
        r2_y = r_matrix_y[0, 1]**2 if not np.isnan(r_matrix_y[0, 1]) else 1.0
        
        # 考虑到如果目标是垂直或水平运动，某一轴的方差可能为 0
        # 我们取两轴 R^2 的加权平均 (基于位移跨度)
        span_x = np.max(x) - np.min(x)
        span_y = np.max(y) - np.min(y)
        
        if span_x + span_y > 1e-5:
            track.linearity_r2 = (r2_x * span_x + r2_y * span_y) / (span_x + span_y)
        else:
            track.linearity_r2 = 1.0 # 没移动 (静态恒星)
            
        # --- 速度一致性计算 ---
        # 计算逐帧瞬时速度的方差
        dt = np.diff(t)
        vx_inst = np.diff(x) / dt
        vy_inst = np.diff(y) / dt
        track.vel_variance = np.var(vx_inst) + np.var(vy_inst)
        
        speed = np.hypot(track.velocity_x, track.velocity_y)
        # 如果速度小于 0.5 像素/帧，认定为恒星，直接赋予最高线性度免检
        if speed < 0.5:
            track.linearity_r2 = 1.0
            track.is_static = True # 打上标签，方便后期区分

    def _compute_photometric_metrics(self, track: Track):
        """计算光度一致性 (Photometric Consistency)"""
        fluxes = np.array([d.flux for d in track.detections])
        
        # 计算变异系数 (Coefficient of Variation, CV) = 标准差 / 平均值
        # 变异系数越小，光度越一致
        mean_flux = np.mean(fluxes)
        if mean_flux > 0:
            track.flux_cv = np.std(fluxes) / mean_flux
        else:
            track.flux_cv = 1.0

    def _compute_track_score(self, track: Track) -> float:
        """合成最终的 Track Score"""
        # 1. 存在性得分: 出现帧数越多越好，上限 1.0
        persistence_score = min(1.0, track.hits / 10.0)
        
        # 2. 线性度得分: 直接使用 R^2
        linearity_score = track.linearity_r2
        
        # 3. 速度一致性得分: 方差越小得分越高 (使用高斯惩罚)
        vel_consistency_score = np.exp(-track.vel_variance)
        
        # 4. 光度一致性得分: CV 越小得分越高
        photo_consistency_score = np.exp(-2.0 * getattr(track, 'flux_cv', 1.0))
        
        # 5. 加权融合 (权重可以写在 YAML 配置文件里)
        track.track_score = (
            0.4 * linearity_score +
            0.3 * persistence_score +
            0.15 * vel_consistency_score +
            0.15 * photo_consistency_score
        )
        return float(track.track_score)