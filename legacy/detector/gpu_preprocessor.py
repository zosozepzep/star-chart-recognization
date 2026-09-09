# src/detector/gpu_preprocessor.py
import logging
import numpy as np
import cupy as cp
from scipy.ndimage import zoom
from typing import Tuple

logger = logging.getLogger("GPUPreprocessor")

class HybridBackgroundEstimator:
    def __init__(self, box_size: int = 128, sigma_clip: float = 2.5, max_iters: int = 5):
        """
        CPU-GPU 混合架构背景建模器
        :param box_size: 局部背景统计网格大小
        :param sigma_clip: Sigma 剪切阈值
        :param max_iters: 最大迭代剔除次数
        """
        self.box_size = box_size
        self.sigma_clip = sigma_clip
        self.max_iters = max_iters
        logger.info(f"Initialized Hybrid GPU Preprocessor (Box: {box_size}, Clip: {sigma_clip}σ)")

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
        # 阶段 2: GPU 端极速并行统计 (CuPy)
        # ==========================================
        # 1. HtoD: 将双精度数据搬运至显存 (严格使用 float64 保证精度)
        d_data = cp.array(padded_data, dtype=cp.float64)
        
        # 2. 内存重构：无损切分为 (n_boxes_y, n_boxes_x, box_size, box_size)
        ny = new_h // self.box_size
        nx = new_w // self.box_size
        grid = d_data.reshape(ny, self.box_size, nx, self.box_size).swapaxes(1, 2)
        
        # 3. 极速 Sigma-Clipping 迭代
        for i in range(self.max_iters):
            # 在 box 内部并行计算中位数和标准差
            median = cp.median(grid, axis=(2, 3), keepdims=True)
            std = cp.std(grid, axis=(2, 3), keepdims=True)
            
            # 生成离群值掩模 (星点或宇宙射线)
            mask = cp.abs(grid - median) > (self.sigma_clip * std)
            
            # 核心提速技巧：直接用当前迭代的中位数替换离群值，保持 Tensor 形状不变！
            grid = cp.where(mask, median, grid)
            
        # 4. 提取最终的低分辨率背景网格与 RMS 网格
        bkg_grid_gpu = cp.median(grid, axis=(2, 3))
        rms_grid_gpu = cp.std(grid, axis=(2, 3))
        
        # 5. DtoH: 将极小尺寸的网格 (ny, nx) 传回系统内存
        bkg_grid_cpu = cp.asnumpy(bkg_grid_gpu)
        rms_grid_cpu = cp.asnumpy(rms_grid_gpu)
        
        # 主动释放显存池 (适合在 Docker/WSL2 环境下严格控制显存占用)
        cp.get_default_memory_pool().free_all_blocks()

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