from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_A = REPO_ROOT / "data" / "images" / "60394_20260309_1437117414637934_PIC"
DATASET_B = REPO_ROOT / "data" / "images" / "60385_20260722_1485695076008039_PIC_POS"


def _require(path: Path) -> Path:
    if not path.is_dir() or not list(path.glob("*.fits")):
        pytest.skip(f"数据集缺失: {path}")
    return path


@pytest.fixture(scope="session")
def dataset_a_dir() -> Path:
    """QIANFAN-16 / NORAD 60394，36 帧，无 .DAT 真值。"""
    return _require(DATASET_A)


@pytest.fixture(scope="session")
def dataset_b_dir() -> Path:
    """QIANFAN-7 / NORAD 60385，80 帧，含 .DAT 真值。主开发数据集。"""
    return _require(DATASET_B)


@pytest.fixture()
def cfg() -> dict:
    return load_config()
