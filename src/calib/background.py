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
        """逐像素探测阈值（相对扣背景后的残差图）。

        n_sigma 必须为正：负阈值会得到一整幅负数图，下游 `residual > threshold`
        会把几乎所有像素判为源，是无声的灾难，因此这里直接拒绝而不是取绝对值。
        """
        if not n_sigma > 0:
            raise ValueError(f"n_sigma 必须为正数: {n_sigma}")
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


def _model_grid(xp, zoom_fn, median_filter_fn, image, box, filter_size, sigma, maxiters):
    """与后端无关的网格统计 + 插值放大核心。

    xp 传 numpy 或 cupy，zoom_fn / median_filter_fn 传对应的 scipy.ndimage 或
    cupyx.scipy.ndimage 函数。_model_gpu 注入 cupy；测试注入 numpy/scipy 来在无卡
    机器上验证同一份代码（见 tests/calib/test_background.py::_model_gpu_emulated）。

    几何上刻意对齐 photutils 的 edge_method='pad'：网格行列数向上取整覆盖整幅图，
    放大时用**整数**倍率 box 再裁回原尺寸，而不是用 ny/gy 这种分数倍率——后者会让
    背景面在远端边缘产生最多半格的空间错位。
    """
    arr = xp.asarray(image, dtype=xp.float64)
    ny, nx = arr.shape
    # 向上取整：末尾不足一格的部分用边缘复制补齐，保证每个像素都参与某个格子的统计。
    gy = -(-ny // box)
    gx = -(-nx // box)
    pad_y = gy * box - ny
    pad_x = gx * box - nx
    if pad_y or pad_x:
        arr = xp.pad(arr, ((0, pad_y), (0, pad_x)), mode="edge")

    tiles = arr.reshape(gy, box, gx, box).transpose(0, 2, 1, 3).reshape(gy, gx, box * box)

    keep = xp.ones_like(tiles, dtype=bool)
    for _ in range(maxiters):
        masked = xp.where(keep, tiles, xp.nan)
        med = xp.nanmedian(masked, axis=2, keepdims=True)
        std = xp.nanstd(masked, axis=2, keepdims=True)
        new_keep = xp.abs(tiles - med) <= sigma * std
        if bool(xp.all(new_keep == keep)):
            break
        keep = new_keep
    masked = xp.where(keep, tiles, xp.nan)
    grid_bkg = xp.nanmedian(masked, axis=2)
    grid_rms = xp.nanstd(masked, axis=2)

    if filter_size > 1:
        grid_bkg = median_filter_fn(grid_bkg, size=filter_size, mode="nearest")
        grid_rms = median_filter_fn(grid_rms, size=filter_size, mode="nearest")

    out = []
    for grid in (grid_bkg, grid_rms):
        # order/mode/grid_mode 与 photutils 的 BkgZoomInterpolator 默认值逐项一致；
        # grid_mode=True 尤其关键，它是像素面积（而非像素中心）坐标约定。
        full = zoom_fn(grid, box, order=3, mode="reflect", grid_mode=True)
        full = full[:ny, :nx]
        # photutils 的插值器 clip=True，把结果夹回网格值域，这里一并对齐。
        out.append(xp.clip(full, grid.min(), grid.max()))
    return out[0], out[1]


def _model_gpu(image, box, filter_size, sigma, maxiters) -> tuple[np.ndarray, np.ndarray]:
    """GPU 路径：网格统计在 GPU 上算，插值放大后取回主机。

    与 CPU 路径**只是近似一致，不是逐位一致**。实测上界（512x512 合成场、box=64）：
    背景 max|diff| = 0.1959，RMS max|diff| = 0.0390（中位数分别只有 0.0068 和 0.0017，
    差异集中在图幅角点，argmax 落在 (511, 0)）。两处不可消除的差异来源：

    1. 估计量不同，这是主因。CPU 走 photutils 的 MedianBackground / StdBackgroundRMS，
       还带 exclude_percentile=10 与 filter_threshold 语义，sigma-clip 用 astropy 的
       SigmaClip；这里是在自己的 clip 掩膜上直接取 nanmedian / nanstd。把两边的
       **网格**（放大之前）直接对比，背景已差 0.1351、RMS 差 0.0319——也就是说
       0.1959 里的绝大部分在插值之前就产生了。
    2. 插值细节不同。已把 order=3 / mode='reflect' / grid_mode=True / clip 和整数倍率
       放大 + 裁剪对齐 photutils 的 BkgZoomInterpolator + edge_method='pad'，插值本身
       只贡献剩下的约 0.06。

    注意：上述上界是在**无 GPU 的机器上用 numpy/scipy 注入 _model_grid 推导**的
    （cupyx.scipy.ndimage 的 zoom / median_filter 与 scipy 对应函数签名和默认值一致），
    而非在真实显卡上跑出来的。真上卡后应重新标定，尤其要复查 cupy 的 nanmedian
    在含 NaN 掩膜时的舍入是否与 numpy 一致。
    """
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter as cp_median_filter
    from cupyx.scipy.ndimage import zoom as cp_zoom

    bkg, rms = _model_grid(
        cp, cp_zoom, cp_median_filter, image, box, filter_size, sigma, maxiters
    )
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
        if box > min(image.shape):
            # box 大于图幅时网格退化成 1x1：背景面变成一个常数（实测 ptp 恰为 0.0），
            # 三次插值无从谈起，搬一趟显存也换不来任何加速。显式挡在这里并记警告，
            # 而不是让它默默算完——默认 box=128 时，任何小于 128 px 的裁剪图
            # （例如 stack 任务的重叠区）都会走到这条路径，留下日志才好排查。
            # 注意补边之后它已**不再抛** ZeroDivisionError，所以靠下面的 try 是
            # 兜不住的：没有异常，只有一幅安静的常数背景。
            logger.warning(
                "box %d exceeds image extent %s, GPU path unavailable, using CPU",
                box,
                image.shape,
            )
        else:
            try:
                bkg, rms = _model_gpu(image, box, filter_size, sigma, maxiters)
                logger.info("background model used GPU backend")
                return BackgroundModel(background=bkg, rms=rms, backend="gpu")
            except Exception as exc:  # pragma: no cover - 取决于运行环境
                logger.warning(
                    "GPU background modelling failed, falling back to CPU: %s", exc
                )
    elif backend == "gpu":
        logger.warning("GPU backend requested but cupy unavailable, falling back to CPU")

    bkg, rms = _model_cpu(image, box, filter_size, sigma, maxiters)
    logger.info("background model used CPU backend")
    return BackgroundModel(background=bkg, rms=rms, backend="cpu")


def background_stats(image: np.ndarray, *, sigma: float = 3.0) -> dict:
    """全图底噪统计，用于传感器健康度与报告。

    对非有限像素免疫。两类坏值都要防，且防法不同：
      NaN —— `np.nanmin/nanmax` 就够，但 `image.min()` 会得到 nan，让动态范围整体失效；
      +inf —— nanmax **不会**忽略它（返回 inf），而 `inf >= 32767` 为真会被误报成饱和。
    所以统一只在 isfinite 的像素上做 min/max 与饱和计数：inf 是读出损坏，不是真饱和，
    把两种硬件故障混为一谈会误导健康度报告。全图无有限像素时返回 nan 而非抛错。
    """
    image = np.asarray(image, dtype=np.float64)
    _, median, std = sigma_clipped_stats(image, sigma=sigma)
    finite = np.isfinite(image)
    values = image[finite]
    return {
        "median": float(median),
        "std": float(std),
        "min": float(values.min()) if values.size else float("nan"),
        "max": float(values.max()) if values.size else float("nan"),
        "n_saturated": int(np.count_nonzero(values >= SATURATION_ADU)),
    }
