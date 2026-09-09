from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import median_filter, zoom

from src.calib.background import (
    _model_grid,
    background_stats,
    gpu_available,
    model_background,
)
from src.config import load_config
from src.dataio.fits_loader import FrameSequence

# 由 CPU 模拟实测得出（见下方 test_cpu_and_emulated_gpu_agree），留约 1.5-2 倍裕度。
# 背景实测 max|diff| 0.1959、median|diff| 0.0068；RMS 实测 0.0390 / 0.0017。
# 同时钉住中位数：max 单独一项挡不住插值约定退化（grid_mode=False 的 max 反而更小，
# 只有 median 会从 0.0068 恶化到 0.0388），因此两个量级都要卡。
GPU_BKG_ATOL = 0.30
GPU_RMS_ATOL = 0.08
GPU_BKG_MEDIAN_ATOL = 0.015
GPU_RMS_MEDIAN_ATOL = 0.005


def _model_gpu_emulated(image, box, filter_size=3, sigma=3.0, maxiters=5):
    """把 numpy/scipy 注入 _model_grid，在无卡机器上跑 GPU 路径的同一份数值代码。

    cupyx.scipy.ndimage 的 zoom / median_filter 与 scipy 对应函数签名、默认值一致，
    因此这是 _model_gpu 的有效代理：被测的是同一个 _model_grid 函数本体。
    """
    return _model_grid(np, zoom, median_filter, image, box, filter_size, sigma, maxiters)


@pytest.fixture()
def synthetic_field():
    rng = np.random.default_rng(20260908)
    ny = nx = 512
    yy, xx = np.mgrid[0:ny, 0:nx]
    ramp = 6.0 + 0.004 * xx + 0.002 * yy          # 已知的线性背景梯度
    noise = rng.normal(0.0, 3.8, size=(ny, nx))    # 与数据集 B 实测 RMS 一致
    img = ramp + noise
    for cx, cy, amp in [(120.0, 130.0, 900.0), (300.0, 410.0, 400.0)]:
        img += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.6 ** 2))
    return img, ramp


@pytest.fixture()
def ramped_noise_field():
    """噪声 sigma 沿 x 从 2 涨到 8 的场。

    synthetic_field 的噪声是空间均匀的（RMS 极差仅 0.10），在那张图上任何断言都
    分不出"逐像素 RMS 图"和"一个标量"——而后续每一步探测都是逐像素定阈值的。
    这张图专门用来钉住 RMS 的空间结构。
    """
    rng = np.random.default_rng(4242)
    ny = nx = 512
    yy, xx = np.mgrid[0:ny, 0:nx]
    ramp = 6.0 + 0.004 * xx + 0.002 * yy
    sigma_map = 2.0 + 6.0 * (xx / (nx - 1))
    img = ramp + rng.normal(0.0, 1.0, size=(ny, nx)) * sigma_map
    return img, sigma_map


def test_background_recovers_known_ramp(synthetic_field):
    img, ramp = synthetic_field
    model = model_background(img, box=64, backend="cpu")
    assert model.background.shape == img.shape
    # 星点只占极少像素，背景应逼近真实梯度
    assert np.median(np.abs(model.background - ramp)) < 0.5


def test_rms_recovers_known_noise(synthetic_field):
    img, _ = synthetic_field
    model = model_background(img, box=64, backend="cpu")
    assert np.median(model.rms) == pytest.approx(3.8, rel=0.15)


def test_subtract_and_threshold(synthetic_field):
    img, _ = synthetic_field
    model = model_background(img, box=64, backend="cpu")
    sub = model.subtract(img)
    assert np.median(sub) == pytest.approx(0.0, abs=0.3)
    # 全图中位数对"减掉一个标量常数"是盲的；逐象限再查一次，梯度必须真被扣掉。
    # 实测：正确实现四象限中位数极差 0.11，误减标量中位数则为 1.52。
    quadrants = [
        float(np.median(sub[a : a + 256, b : b + 256]))
        for a in (0, 256)
        for b in (0, 256)
    ]
    assert np.ptp(quadrants) < 0.3
    thr = model.threshold(5.0)
    assert thr.shape == img.shape
    assert np.all(thr > 0)
    assert np.median(thr) == pytest.approx(5.0 * np.median(model.rms), rel=0.05)


def test_backend_label_is_recorded(synthetic_field):
    img, _ = synthetic_field
    assert model_background(img, box=64, backend="cpu").backend == "cpu"


def test_auto_backend_never_raises(synthetic_field):
    """无 CUDA 的机器上 auto 必须静默回退 CPU，不得崩溃。"""
    img, _ = synthetic_field
    model = model_background(img, box=64, backend="auto")
    assert model.backend in {"cpu", "gpu"}


def test_explicit_gpu_request_falls_back_when_unavailable(synthetic_field):
    img, _ = synthetic_field
    model = model_background(img, box=64, backend="gpu")
    if not gpu_available():
        assert model.backend == "cpu"


def test_unknown_backend_rejected(synthetic_field):
    """拼错的后端名必须立即报错，而不是静默按 CPU 跑。"""
    img, _ = synthetic_field
    with pytest.raises(ValueError):
        model_background(img, box=64, backend="cuda")


@pytest.mark.skipif(not gpu_available(), reason="无可用 GPU")
def test_cpu_and_gpu_agree(synthetic_field):
    img, _ = synthetic_field
    cpu = model_background(img, box=64, backend="cpu")
    gpu = model_background(img, box=64, backend="gpu")
    # 不是 1e-6：两条路径的估计量与插值约定本就不同，实测上界见模块常量。
    assert np.allclose(cpu.background, gpu.background, atol=GPU_BKG_ATOL)
    assert np.allclose(cpu.rms, gpu.rms, atol=GPU_RMS_ATOL)
    assert np.median(np.abs(cpu.background - gpu.background)) < GPU_BKG_MEDIAN_ATOL
    assert np.median(np.abs(cpu.rms - gpu.rms)) < GPU_RMS_MEDIAN_ATOL


def test_cpu_and_emulated_gpu_agree(synthetic_field):
    """无卡机器上也要能守住 GPU 路径的数值：用 numpy/scipy 注入同一个 _model_grid。

    实测 max|diff|：背景 0.1959、RMS 0.0390；median|diff| 分别 0.0068 / 0.0017。
    max 容差取 0.30 / 0.08（约 1.5-2 倍裕度），median 容差取 0.015 / 0.005。
    两个量级都卡是有原因的：把 grid_mode 改回 False，max 反而降到 0.1557（挡不住），
    但 median 会恶化到 0.0388，只有 median 断言能抓住这种插值约定退化。
    """
    img, _ = synthetic_field
    cpu = model_background(img, box=64, backend="cpu")
    emu_bkg, emu_rms = _model_gpu_emulated(img, 64)
    assert emu_bkg.shape == img.shape
    assert emu_rms.shape == img.shape
    d_bkg = np.abs(cpu.background - emu_bkg)
    d_rms = np.abs(cpu.rms - emu_rms)
    assert np.max(d_bkg) < GPU_BKG_ATOL
    assert np.max(d_rms) < GPU_RMS_ATOL
    assert np.median(d_bkg) < GPU_BKG_MEDIAN_ATOL
    assert np.median(d_rms) < GPU_RMS_MEDIAN_ATOL


def test_emulated_gpu_handles_non_divisible_shape():
    """500x500 / box=64 不是整数格：末尾 52 行列必须仍参与统计，且不得空间错位。

    旧实现用 ny // box 截断、再按 ny/gy 这种分数倍率放大，背景面在远端边缘最多偏
    半格。用一条陡峭的线性斜面来量这个错位——斜面每像素涨 0.20/0.10 ADU，半格
    （32 px）的错位会放大成好几个 ADU 的偏差。实测 median|bkg − ramp|：
    补边实现 0.32，截断实现 7.96（差 24 倍）；末 40 行分别 0.49 与 11.04。
    """
    ny = nx = 500
    yy, xx = np.mgrid[0:ny, 0:nx]
    ramp = 6.0 + 0.20 * xx + 0.10 * yy
    rng = np.random.default_rng(3)
    img = ramp + rng.normal(0.0, 1.0, size=(ny, nx))
    bkg, rms = _model_gpu_emulated(img, 64)
    assert bkg.shape == (ny, nx)
    assert rms.shape == (ny, nx)
    assert np.all(np.isfinite(bkg)) and np.all(np.isfinite(rms))
    # 阈值取在 0.32 与 7.96 之间，两侧裕度都在 3 倍以上。
    assert float(np.median(np.abs(bkg - ramp))) < 1.5
    # 末 40 行正是被截掉的那一块：截断实现 11.04，补边实现 0.49。
    assert float(np.median(np.abs(bkg[-40:, :] - ramp[-40:, :]))) < 3.0
    assert float(np.median(np.abs(bkg[:, -40:] - ramp[:, -40:]))) < 12.0


def test_gpu_path_skipped_when_box_exceeds_image(caplog, monkeypatch):
    """box 大于图幅时必须显式记警告并回退，而不是默默算出一幅常数背景。

    本机没有 GPU，所以要把 gpu_available 打成 True 才能走到这个分支——否则会先被
    "cupy 不可用"的回退截住，这段守卫在无卡机器上永远测不到。
    补边之后这种情况**不再抛异常**（网格退化成 1x1，背景 ptp 实测 0.0），
    所以必须是显式守卫，靠 except 是兜不住的。
    """
    monkeypatch.setattr("src.calib.background.gpu_available", lambda: True)
    rng = np.random.default_rng(11)
    img = 6.0 + rng.normal(0.0, 3.8, size=(100, 100))
    with caplog.at_level("WARNING"):
        model = model_background(img, box=128, backend="gpu")
    assert model.backend == "cpu"
    assert model.background.shape == img.shape
    messages = [r.getMessage() for r in caplog.records]
    assert any("exceeds image extent" in m for m in messages), messages
    # 必须是显式守卫，不能是被 except 捞回来的异常。
    assert not any("GPU background modelling failed" in m for m in messages), messages


def test_rms_and_threshold_are_per_pixel(ramped_noise_field):
    """RMS 与阈值必须逐像素跟随噪声，而不是一个被摊平的标量。

    实测：左四分之一 RMS 中位数 2.83、右四分之一 7.00，比值 2.47
    （真值 sigma 分别 2.75 / 7.25）。摊平成标量的实现比值恒为 1.0。
    """
    img, sigma_map = ramped_noise_field
    model = model_background(img, box=64, backend="cpu")
    q = img.shape[1] // 4
    left = float(np.median(model.rms[:, :q]))
    right = float(np.median(model.rms[:, -q:]))
    assert left == pytest.approx(float(np.median(sigma_map[:, :q])), rel=0.15)
    assert right == pytest.approx(float(np.median(sigma_map[:, -q:])), rel=0.15)
    assert right / left > 2.0

    thr = model.threshold(5.0)
    thr_left = float(np.median(thr[:, :q]))
    thr_right = float(np.median(thr[:, -q:]))
    assert thr_right / thr_left > 2.0
    # 阈值必须逐像素等于 n_sigma * rms，而不只是"中位数对得上"。
    assert np.allclose(thr, 5.0 * model.rms)


def test_threshold_rejects_non_positive_n_sigma(synthetic_field):
    """负阈值会让下游把整幅图判成源，必须直接拒绝。"""
    img, _ = synthetic_field
    model = model_background(img, box=64, backend="cpu")
    with pytest.raises(ValueError):
        model.threshold(-1.0)
    with pytest.raises(ValueError):
        model.threshold(0.0)


def test_defaults_match_config():
    """函数默认值必须与 src/config/default.yaml 的 background 节一致。

    所有测试都显式传 box=64，没有任何用例会因默认值漂移而失败——这条专门补位。
    """
    cfg = load_config()["background"]
    defaults = {
        name: p.default
        for name, p in inspect.signature(model_background).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    for key in ("box", "filter_size", "sigma", "maxiters", "backend"):
        assert defaults[key] == cfg[key], f"默认值与配置不一致: {key}"


def test_background_stats_dataset_b_frame30(dataset_b_dir):
    seq = FrameSequence.from_directory(dataset_b_dir)
    stats = background_stats(seq.image(30))
    assert stats["median"] == pytest.approx(6.45, abs=1.0)
    assert stats["std"] == pytest.approx(3.80, abs=1.0)
    assert stats["n_saturated"] == 0
    assert stats["max"] < 32767
    # 实测该帧 min=0、max=26978；min 若被误写成 max 这条立刻失败。
    assert stats["min"] == pytest.approx(0.0, abs=1e-9)
    assert stats["max"] == pytest.approx(26978.0, abs=1.0)


def test_background_stats_is_nan_safe():
    """含 NaN 坏列与 +inf 坏点的帧不得把动态范围报成非有限值、也不得虚报饱和。

    两种坏值要分别防：NaN 会毒化 image.min()/max()；+inf 连 nanmax 都挡不住，
    且 `inf >= 32767` 为真，会被当成一个真饱和像素上报。
    """
    rng = np.random.default_rng(99)
    img = 6.0 + rng.normal(0.0, 3.8, size=(64, 64))
    img[:, 10] = np.nan          # 一整列 NaN 坏像素
    img[5, 5] = np.inf           # 读出损坏，不是真饱和
    img[6, 6] = -np.inf
    img[0, 0] = 40000.0          # 唯一的真饱和像素（> 32767）
    # astropy 会为非有限输入发一条 AstropyUserWarning，这是预期行为；就地断言掉，
    # 不去放宽 pytest.ini 的全局过滤器。
    with pytest.warns(AstropyUserWarning, match="invalid values"):
        stats = background_stats(img)
    assert np.isfinite(stats["min"])
    assert np.isfinite(stats["max"])
    assert stats["max"] == pytest.approx(40000.0)   # 不是 inf
    assert stats["n_saturated"] == 1                # inf 不算饱和
    assert np.isfinite(stats["median"])
    assert np.isfinite(stats["std"])


def test_background_stats_all_non_finite():
    """全是坏值时返回 nan 而不是抛错——健康度报告要能说"这帧没救了"。"""
    img = np.full((8, 8), np.nan)
    # 空切片求均值的 RuntimeWarning 是这条极端用例的固有产物，就地吸收。
    with pytest.warns(AstropyUserWarning), np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
            stats = background_stats(img)
    assert np.isnan(stats["min"])
    assert np.isnan(stats["max"])
    assert stats["n_saturated"] == 0
