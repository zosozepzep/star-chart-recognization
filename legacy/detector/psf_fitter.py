from __future__ import annotations
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import curve_fit
from detector.feature_extractor import SourceDetection
logger = logging.getLogger("PSFFitter_MultiCore")

# ==========================================
# 全局纯函数：多进程计算核心
# 放在顶层，确保 Python 可以极速序列化 (Pickle)
# ==========================================
def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, background):
    x, y = coords
    model = amplitude * np.exp(
        -(((x - x0) ** 2) / (2 * sigma_x ** 2)
          + ((y - y0) ** 2) / (2 * sigma_y ** 2))
    ) + background
    return model.ravel()

def _worker_fit_gaussian(task_data):
    """
    独立的工作节点：只负责接收切片并执行极限数学拟合。
    传入: (idx, patch, x0_global, y0_global)
    返回: (idx, success, xf, yf, amplitude, fwhm, fwhm_x, fwhm_y, elongation, ellipticity, chi2, residual_ratio, error_msg)
    """
    idx, patch, x0_global, y0_global = task_data
    ny, nx = patch.shape
    y, x = np.mgrid[:ny, :nx]

    amplitude0 = patch.max() - np.median(patch)
    background0 = np.median(patch)
    x0 = nx / 2
    y0 = ny / 2

    p0 = [amplitude0, x0, y0, 1.5, 1.5, background0]

    bounds = (
        [0, 0, 0, 0.3, 0.3, -np.inf],
        [np.inf, nx, ny, 10.0, 10.0, np.inf],
    )

    try:
        popt, _ = curve_fit(
            gaussian_2d,
            (x, y),
            patch.ravel(),
            p0=p0,
            bounds=bounds,
            maxfev=200,
        )

        amp, xf, yf, sx, sy, bg = popt

        # 形态学计算
        fwhm_x = 2.3548 * sx
        fwhm_y = 2.3548 * sy
        fwhm = np.sqrt(fwhm_x * fwhm_y)

        a = max(fwhm_x, fwhm_y)
        b = min(fwhm_x, fwhm_y)

        elongation = a / b if b > 0 else np.inf
        ellipticity = 1 - b / a if a > 0 else np.nan

        # 残差计算
        model = gaussian_2d((x, y), *popt).reshape(patch.shape)
        residual = patch - model

        chi2 = float(np.mean(residual ** 2))
        denom = np.sum(np.abs(patch)) + 1e-8
        residual_ratio = float(np.sum(np.abs(residual)) / denom)

        # 返回全部计算结果
        return (idx, True, xf, yf, amp, fwhm, fwhm_x, fwhm_y, elongation, ellipticity, chi2, residual_ratio, "")

    except Exception as e:
        # 失败时返回空位与错误信息
        return (idx, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, str(e))


class PSFFitter:
    def __init__(self, patch_size: int = 11, max_workers: int = None):
        self.patch_size = patch_size
        # 默认留一个核心给操作系统和其他模块
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Initialized Multi-core PSFFitter with {self.max_workers} workers.")

    def fit_sources(self, data: np.ndarray, detections: list) -> list:
        if not detections:
            return []

        # ==========================================
        # 阶段 1：Map - 主进程切分数据，打包任务
        # ==========================================
        tasks = []
        for i, det in enumerate(detections):
            patch, x0_global, y0_global = self._extract_patch(data, det.x, det.y)
            if patch is None:
                det.fit_success = False
                det.flags["psf_patch_failed"] = True
                continue
            
            # 只打包最基础的数值，极大降低 IPC 序列化开销
            tasks.append((i, patch, x0_global, y0_global))

        if not tasks:
            return detections

        # ==========================================
        # 阶段 2：Execute - 多核并发轰炸
        # ==========================================
        # 动态计算 chunksize，避免核心频繁索要任务导致上下文切换开销
        chunk_size = max(1, len(tasks) // (self.max_workers * 4))
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for res in executor.map(_worker_fit_gaussian, tasks, chunksize=chunk_size):
                results.append(res)

        # ==========================================
        # 阶段 3：Reduce - 主进程回收结果，更新对象
        # ==========================================
        success_count = 0
        
        # 🌟 关键修复：使用 zip 完美对齐任务和结果，彻底告别索引错位风险！
        for task, res in zip(tasks, results):
            (idx, success, xf, yf, amp, fwhm, fwhm_x, fwhm_y, 
             elongation, ellipticity, chi2, residual_ratio, error_msg) = res
            
            det = detections[idx]
            
            if success:
                # 直接从配对的 task 中解包出全局坐标基准
                _, _, x0_global, y0_global = task
                
                det.x = x0_global + xf
                det.y = y0_global + yf
                det.amplitude = amp
                det.fwhm = fwhm
                det.fwhm_x = fwhm_x
                det.fwhm_y = fwhm_y
                det.elongation = elongation
                det.ellipticity = ellipticity
                det.chi2 = chi2
                det.residual_ratio = residual_ratio
                det.fit_success = True
                success_count += 1
            else:
                det.fit_success = False
                det.flags["psf_fit_failed"] = True
                det.flags["psf_error"] = error_msg

        logger.info(f" -> Multi-core PSF Fitting: {success_count}/{len(detections)} successful.")
        
        return detections

    def _extract_patch(self, data, x, y):
        half_size = self.patch_size // 2
        ix, iy = int(round(x)), int(round(y))
        
        y0, y1 = iy - half_size, iy + half_size + 1
        x0, x1 = ix - half_size, ix + half_size + 1
        
        # 边界检查
        if y0 < 0 or y1 > data.shape[0] or x0 < 0 or x1 > data.shape[1]:
            return None, 0, 0
            
        patch = data[y0:y1, x0:x1].astype(np.float64)
        return patch, x0, y0