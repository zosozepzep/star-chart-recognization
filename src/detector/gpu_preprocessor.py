# src/detector/gpu_preprocessor.py
import logging
import numpy as np
from scipy.ndimage import zoom
from typing import Tuple

logger = logging.getLogger("CPUPreprocessor")

class HybridBackgroundEstimator:
    def __init__(self, box_size: int = 128, sigma_clip: float = 2.5, max_iters: int = 5):
        """
        CPU 背景建模器
        :param box_size: 局部背景统计网格大小
        :param sigma_clip: Sigma 剪切阈值
        :param max_iters: 最大迭代剔除次数
        """
        self.box_size = box_size
        self.sigma_clip = sigma_clip
        self.max_iters = max_iters
        logger.info(f"Initialized CPU Preprocessor (Box: {box_size}, Clip: {sigma_clip}σ)")

    def estimate(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行极速背景建模
        返回: (background_map, rms_map)
        """
        h, w = data.shape
        
        # ==========================================
        # 阶段 1: CPU 端安全预处理 (Padding)
        # ==========================================
        # 计算需要填充的边缘像素，确保能被 box_size 完美整除
        pad_h = (self.box_size - h % self.box_size) % self.box_size
        pad_w = (self.box_size - w % self.box_size) % self.box_size
        
        # 使用镜像边缘填充 (Reflect)，防止边缘背景产生突变断层
        padded_data = np.pad(data, ((0, pad_h), (0, pad_w)), mode='reflect')
        new_h, new_w = padded_data.shape
        
        # ==========================================
        # 阶段 2: CPU 端局部网格统计
        # ==========================================
        ny = new_h // self.box_size
        nx = new_w // self.box_size
        grid = padded_data.astype(np.float64, copy=False).reshape(
            ny, self.box_size, nx, self.box_size
        ).swapaxes(1, 2)
        
        for i in range(self.max_iters):
            median = np.median(grid, axis=(2, 3), keepdims=True)
            std = np.std(grid, axis=(2, 3), keepdims=True)
            
            mask = np.abs(grid - median) > (self.sigma_clip * std)
            
            grid = np.where(mask, median, grid)
            
        bkg_grid_cpu = np.median(grid, axis=(2, 3))
        rms_grid_cpu = np.std(grid, axis=(2, 3))

        # ==========================================
        # 阶段 3: CPU 端高精度三次样条插值 (Bicubic Zoom)
        # ==========================================
        # 计算插值放大倍率
        zoom_y = new_h / ny
        zoom_x = new_w / nx
        
        # 使用 scipy.ndimage.zoom 进行三次样条插值 (order=3)，完美复刻 Astropy 的 BkgZoomInterpolator
        bkg_map_padded = zoom(bkg_grid_cpu, (zoom_y, zoom_x), order=3)
        rms_map_padded = zoom(rms_grid_cpu, (zoom_y, zoom_x), order=3)
        
        # 裁剪掉我们在阶段 1 添加的 Padding 边缘，还原真实尺寸
        bkg_map = bkg_map_padded[:h, :w]
        rms_map = rms_map_padded[:h, :w]
        
        return bkg_map, rms_map
