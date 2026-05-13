from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from enum import IntEnum
from tracking.kalman import ConstantVelocityKalmanFilter

@dataclass
class SourceDetection:
    # 基础信息
    x: float
    y: float
    engine: str = "unknown"
    engine_votes: List[str] = field(default_factory=list)

    # 光度特征
    peak: float = 0.0
    flux: float = 0.0
    local_background: float = 0.0
    local_rms: float = 0.0
    snr: float = 0.0

    # 形态特征
    fwhm_x: float = np.nan
    fwhm_y: float = np.nan
    fwhm: float = np.nan
    ellipticity: float = np.nan
    elongation: float = np.nan
    theta: float = np.nan

    # PSF 拟合特征
    residual_ratio: float = np.nan
    chi2: float = np.nan
    fit_success: bool = False

    # 评分
    score: float = 0.0

    # 标志位
    flags: Dict[str, bool] = field(default_factory=dict)

class TrackState(IntEnum):
    """轨迹状态机 (生命周期管理)"""
    TENTATIVE = 0  # 试探态：刚出现，可能是噪点闪烁
    CONFIRMED = 1  # 确认态：连续出现多帧，具备物理可信度
    LOST = 2       # 丢失态：连续多帧未匹配到，准备从内存中销毁

@dataclass
class Track:
    """
    空间目标物理轨迹 (Physical Target Track)
    记录目标在时域上的运动学状态与光度演化
    """
    track_id: int
    state: TrackState = TrackState.TENTATIVE
    vel_variance: float = 0.0
    flux_cv: float = 0.0
    track_score: float = 0.0
    # 观测历史 (History)
    # 这里的 List[SourceDetection] 直接使用了你上面定义的类
    detections: List[SourceDetection] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    
    # 运动学状态估计 (Kinematic State Estimation)
    # 单位: pixels / frame
    kf: Optional[ConstantVelocityKalmanFilter] = None
    
    # 物理一致性校验参数
    linearity_r2: float = 0.0   # 轨迹的线性度拟合优度 (Phase 3 会用到)
    
    # 轨迹健康度指标 (Health Metrics)
    time_since_update: int = 0  # 距离上次观测到该目标过了几帧
    hits: int = 1               # 总命中帧数
    
    @property
    def last_detection(self) -> SourceDetection:
        """获取最新一帧的物理特征"""
        return self.detections[-1]

    @property
    def last_frame_index(self) -> int:
        """获取最后一次观测到的帧号"""
        return self.frame_indices[-1]

    def predict_position(self, current_frame_index: int) -> tuple[float, float]:
        """直接调用卡尔曼矩阵进行物理预测"""
        # 如果卡尔曼滤波器还没初始化（防御性编程）
        if self.kf is None:
            return float(self.last_detection.x), float(self.last_detection.y)
            
        dt = current_frame_index - self.last_frame_index
        # kf.x 状态向量: [x位置, y位置, x速度, y速度]
        # 根据矩阵当前的最优估计位置和速度，向未来推演 dt 帧
        pred_x = self.kf.x[0] + self.kf.x[2] * dt
        pred_y = self.kf.x[1] + self.kf.x[3] * dt
        
        return float(pred_x), float(pred_y)

    # ==========================================
    # 动态属性 (Properties)：为了兼容外部调用
    # ==========================================
    @property
    def velocity_x(self) -> float:
        """从卡尔曼状态矩阵中安全提取 X 轴速度"""
        return float(self.kf.x[2]) if self.kf else 0.0

    @property
    def velocity_y(self) -> float:
        """从卡尔曼状态矩阵中安全提取 Y 轴速度"""
        return float(self.kf.x[3]) if self.kf else 0.0

class FeatureExtractor:
    def __init__(self, patch_size: int = 15, aperture_radius: float = 3.0):
        self.patch_size = patch_size
        self.aperture_radius = aperture_radius

    def extract(self, data: np.ndarray, candidates: List[SourceDetection]) -> List[SourceDetection]:
        results = []
        for det in candidates:
            try:
                self._extract_single(data, det)
                results.append(det)
            except Exception as e:
                det.flags["feature_extraction_failed"] = True
                det.flags["error"] = str(e)
                results.append(det)
        return results

    def _extract_single(self, data: np.ndarray, det: SourceDetection):
        x, y = det.x, det.y

        patch = self._extract_patch(data, x, y)
        if patch is None:
            det.flags["edge_truncated"] = True
            return

        mean, median, std = sigma_clipped_stats(patch, sigma=3.0)

        det.local_background = float(median)
        det.local_rms = float(std)

        ix = int(round(x))
        iy = int(round(y))
        det.peak = float(data[iy, ix])

        positions = [(x, y)]

        aperture = CircularAperture(positions, r=self.aperture_radius)
        annulus = CircularAnnulus(positions, r_in=6, r_out=8)

        apers = [aperture, annulus]
        phot = aperture_photometry(data, apers)

        aperture_sum = float(phot["aperture_sum_0"][0])
        annulus_sum = float(phot["aperture_sum_1"][0])

        annulus_area = annulus.area
        aperture_area = aperture.area

        annulus_mean = annulus_sum / annulus_area
        bkg_total = annulus_mean * aperture_area

        det.flux = aperture_sum - bkg_total

        noise = np.sqrt(aperture_area * det.local_rms ** 2 + max(det.flux, 0))
        if noise > 0:
            det.snr = det.flux / noise
        # 在 FeatureExtractor 类中补齐：
    def _extract_patch(self, data: np.ndarray, x: float, y: float) -> Optional[np.ndarray]:
        half_size = self.patch_size // 2
        ix, iy = int(round(x)), int(round(y))
        
        y0, y1 = iy - half_size, iy + half_size + 1
        x0, x1 = ix - half_size, ix + half_size + 1
        
        # 边界检查
        if y0 < 0 or y1 > data.shape[0] or x0 < 0 or x1 > data.shape[1]:
            return None
            
        return data[y0:y1, x0:x1].astype(np.float64)