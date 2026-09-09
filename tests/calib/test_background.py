from __future__ import annotations

import numpy as np
import pytest

from src.calib.background import (
    background_stats,
    gpu_available,
    model_background,
)
from src.dataio.fits_loader import FrameSequence


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
    assert np.allclose(cpu.background, gpu.background, atol=1e-6)
    assert np.allclose(cpu.rms, gpu.rms, atol=1e-6)


def test_background_stats_dataset_b_frame30(dataset_b_dir):
    seq = FrameSequence.from_directory(dataset_b_dir)
    stats = background_stats(seq.image(30))
    assert stats["median"] == pytest.approx(6.45, abs=1.0)
    assert stats["std"] == pytest.approx(3.80, abs=1.0)
    assert stats["n_saturated"] == 0
    assert stats["max"] < 32767
