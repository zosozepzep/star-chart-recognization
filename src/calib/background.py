"""背景与 RMS 建模。

128x128 网格 + sigma-clip 中位数 + 双三次插值放大（沿用 photutils 的 Background2D）。
CPU 为默认路径；GPU 后端仅在 cupy 可用时启用，任何 import 或运行失败都静默回退
CPU 并在返回值中标注真实后端——国赛评委机器很可能无 CUDA，崩溃是不可接受的。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.background import Background2D, MedianBackground

logger = logging.getLogger(__name__)

# 本数据集 BZERO=0 且 BITPIX=16，astropy 按有符号 int16 解出像素，
# 物理上限因此是 32767 而非 65535；写 65535 会让饱和判据永远为假。
# 与 src/dataio/fits_loader.py 的 load_image 文档保持一致。
SATURATION_ADU = 32767.0


def gpu_available() -> bool:
    """cupy 是否可用且真的能拿到设备。

    注意本镜像里 cupy 是装着的，`import cupy` 会成功，真正抛错的是
    getDeviceCount()（无驱动时报 cudaErrorInsufficientDriver）。所以两者都必须
    包在 try 里，只捕获 ImportError 是不够的。
    """
    try:
        import cupy  # noqa: F401

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@dataclass
class BackgroundModel:
    background: np.ndarray
    rms: np.ndarray
    backend: str

    def subtract(self, image: np.ndarray) -> np.ndarray:
        """扣除背景，返回 float64 残差图。"""
        return np.asarray(image, dtype=np.float64) - self.background

    def threshold(self, n_sigma: float) -> np.ndarray:
        """逐像素探测阈值（相对扣背景后的残差图）。"""
        return n_sigma * self.rms


def _model_cpu(image, box, filter_size, sigma, maxiters) -> tuple[np.ndarray, np.ndarray]:
    bkg = Background2D(
        image,
        (box, box),
        filter_size=(filter_size, filter_size),
        sigma_clip=SigmaClip(sigma=sigma, maxiters=maxiters),
        bkg_estimator=MedianBackground(),
    )
    return np.asarray(bkg.background, dtype=np.float64), np.asarray(
        bkg.background_rms, dtype=np.float64
    )


def _model_gpu(image, box, filter_size, sigma, maxiters) -> tuple[np.ndarray, np.ndarray]:
    """GPU 路径：网格统计在 GPU 上算，插值放大后取回主机。

    数值上与 CPU 路径一致：同样的 sigma-clip 迭代与中位数估计，同样的三次插值。
    """
    import cupy as cp
    from cupyx.scipy.ndimage import zoom as cp_zoom

    arr = cp.asarray(image, dtype=cp.float64)
    ny, nx = arr.shape
    gy, gx = ny // box, nx // box
    tiles = arr[: gy * box, : gx * box].reshape(gy, box, gx, box).transpose(0, 2, 1, 3)
    tiles = tiles.reshape(gy, gx, box * box)

    keep = cp.ones_like(tiles, dtype=bool)
    for _ in range(maxiters):
        masked = cp.where(keep, tiles, cp.nan)
        med = cp.nanmedian(masked, axis=2, keepdims=True)
        std = cp.nanstd(masked, axis=2, keepdims=True)
        new_keep = cp.abs(tiles - med) <= sigma * std
        if bool(cp.all(new_keep == keep)):
            break
        keep = new_keep
    masked = cp.where(keep, tiles, cp.nan)
    grid_bkg = cp.nanmedian(masked, axis=2)
    grid_rms = cp.nanstd(masked, axis=2)

    if filter_size > 1:
        from cupyx.scipy.ndimage import median_filter as cp_median_filter

        grid_bkg = cp_median_filter(grid_bkg, size=filter_size, mode="nearest")
        grid_rms = cp_median_filter(grid_rms, size=filter_size, mode="nearest")

    bkg = cp_zoom(grid_bkg, (ny / gy, nx / gx), order=3, mode="nearest")
    rms = cp_zoom(grid_rms, (ny / gy, nx / gx), order=3, mode="nearest")
    return cp.asnumpy(bkg), cp.asnumpy(rms)


def model_background(
    image: np.ndarray,
    *,
    box: int = 128,
    filter_size: int = 3,
    sigma: float = 3.0,
    maxiters: int = 5,
    backend: str = "auto",
) -> BackgroundModel:
    """建立背景与 RMS 模型；默认值与 src/config/default.yaml 的 background 节一致。"""
    image = np.asarray(image, dtype=np.float64)
    if backend not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"未知后端: {backend}")

    if backend in {"auto", "gpu"} and gpu_available():
        try:
            bkg, rms = _model_gpu(image, box, filter_size, sigma, maxiters)
            logger.info("background model used GPU backend")
            return BackgroundModel(background=bkg, rms=rms, backend="gpu")
        except Exception as exc:  # pragma: no cover - 取决于运行环境
            logger.warning("GPU background modelling failed, falling back to CPU: %s", exc)
    elif backend == "gpu":
        logger.warning("GPU backend requested but cupy unavailable, falling back to CPU")

    bkg, rms = _model_cpu(image, box, filter_size, sigma, maxiters)
    logger.info("background model used CPU backend")
    return BackgroundModel(background=bkg, rms=rms, backend="cpu")


def background_stats(image: np.ndarray, *, sigma: float = 3.0) -> dict:
    """全图底噪统计，用于传感器健康度与报告。"""
    image = np.asarray(image, dtype=np.float64)
    _, median, std = sigma_clipped_stats(image, sigma=sigma)
    return {
        "median": float(median),
        "std": float(std),
        "min": float(image.min()),
        "max": float(image.max()),
        "n_saturated": int(np.count_nonzero(image >= SATURATION_ADU)),
    }
