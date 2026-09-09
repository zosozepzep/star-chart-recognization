# src/tracking/associator.py
import logging
import numpy as np
from scipy.spatial import cKDTree
from typing import List
# 导入我们在上一节刚刚合并好的强类型数据契约
from detector.feature_extractor import SourceDetection, Track, TrackState
#导入卡尔曼滤波器模块
from tracking.kalman import ConstantVelocityKalmanFilter
logger = logging.getLogger("Associator")

class KDTreeAssociator:
    def __init__(self, match_radius: float = 3.0, max_lost_frames: int = 3, min_hits_to_confirm: int = 3):
        """
        基于空间索引与运动学预测的高性能目标关联引擎
        
        :param match_radius: 允许的最大物理偏离半径 (像素)
        :param max_lost_frames: 容忍目标由于闪烁或遮挡消失的最大帧数
        :param min_hits_to_confirm: 连续命中几帧后，确认为真实物理目标
        """
        self.match_radius = match_radius
        self.max_lost_frames = max_lost_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        
        self.active_tracks: List[Track] = []
        self._next_track_id = 1

    def update(self, current_detections: List[SourceDetection], frame_index: int):
        """执行单帧时序更新：预测 (Predict) -> 匹配 (Match) -> 更新状态机 (Update)"""
        
        # 1. 应对全黑帧或无目标帧的防御逻辑
        if not current_detections:
            self._increment_lost_counters()
            return

        # 2. 构建当前帧观测点的 KD-Tree 空间索引
        obs_coords = np.array([[d.x, d.y] for d in current_detections])
        kdtree = cKDTree(obs_coords)
        
        matched_obs_indices = set()

        # 3. 对现有轨迹进行前向物理预测与关联
        for track in self.active_tracks:
            if track.state == TrackState.LOST:
                continue

            # 物理预测：根据上一帧的位置和速度，推算当前帧它应该在哪
            pred_x, pred_y = track.predict_position(frame_index)
            
            # 空间查询：在预测位置半径 R 内寻找最匹配的观测点
            distances, indices = kdtree.query([pred_x, pred_y], k=1, distance_upper_bound=self.match_radius)
            
            if distances != float('inf') and indices not in matched_obs_indices:
                # 命中目标：提取最新的物理观测特征
                matched_obs_idx = indices
                matched_det = current_detections[matched_obs_idx]
                
                self._update_track(track, matched_det, frame_index)
                matched_obs_indices.add(matched_obs_idx)
            else:
                # 未命中：增加丢失计数，但不立刻抹杀 (容忍闪烁)
                track.time_since_update += 1
                if track.time_since_update > self.max_lost_frames:
                    track.state = TrackState.LOST

        # 4. 孕育新生命：为未被认领的游离孤立点创建全新的轨迹
        new_tracks_count = 0
        for i, det in enumerate(current_detections):
            if i not in matched_obs_indices:
                self._create_new_track(det, frame_index)
                new_tracks_count += 1
                
        # 5. 垃圾回收：清理彻底丢失的噪点轨迹
        self.active_tracks = [t for t in self.active_tracks if t.state != TrackState.LOST]

    def _create_new_track(self, det: SourceDetection, frame_index: int):
        """孕育初始生命周期，初始化卡尔曼矩阵"""
        new_track = Track(
            track_id=self._next_track_id,
            detections=[det],
            frame_indices=[frame_index],
        )
        # 赋予初始状态
        new_track.kf = ConstantVelocityKalmanFilter(init_x=det.x, init_y=det.y)
        
        self.active_tracks.append(new_track)
        self._next_track_id += 1

    def _update_track(self, track: Track, det: SourceDetection, frame_index: int):
        """使用观测值进行矩阵融合更新"""
        # 1. 必须先执行一次矩阵推演 (Predict) 才能进行融合
        track.kf.predict()
        
        # 2. 执行最优观测融合 (Update)，传入目标的信噪比！
        track.kf.update(meas_x=det.x, meas_y=det.y, snr=det.snr)

        # 追加历史记录
        track.detections.append(det)
        track.frame_indices.append(frame_index)
        track.time_since_update = 0
        track.hits += 1

        if track.state == TrackState.TENTATIVE and track.hits >= self.min_hits_to_confirm:
            track.state = TrackState.CONFIRMED
    def _increment_lost_counters(self):
        for track in self.active_tracks:
            track.time_since_update += 1
            if track.time_since_update > self.max_lost_frames:
                track.state = TrackState.LOST
        self.active_tracks = [t for t in self.active_tracks if t.state != TrackState.LOST]

    def get_confirmed_tracks(self) -> List[Track]:
        return [t for t in self.active_tracks if t.state == TrackState.CONFIRMED]