from __future__ import annotations

import importlib

import pytest

from src.config import load_config

CORE_MODULES = ["numpy", "scipy", "astropy", "photutils", "matplotlib", "pandas", "yaml"]

PACKAGES = [
    "src.dataio",
    "src.calib",
    "src.detect",
    "src.register",
    "src.stack",
    "src.target",
    "src.astrometry",
    "src.analysis",
    "src.validate",
    "src.viz",
]


@pytest.mark.parametrize("name", CORE_MODULES)
def test_core_dependency_importable(name):
    assert importlib.import_module(name) is not None


@pytest.mark.parametrize("name", PACKAGES)
def test_project_package_importable(name):
    assert importlib.import_module(name) is not None


def test_no_stdlib_shadowing_io_package():
    """包必须叫 dataio；出现 src/io 会遮蔽标准库 io。"""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.io")


def test_default_config_has_required_sections():
    conf = load_config()
    for section in [
        "segment",
        "background",
        "hotpixel",
        "detect",
        "psf",
        "register",
        "stack",
        "target",
        "photometry",
        "repeatability",
        "orbit",
        "viz",
    ]:
        assert section in conf, f"缺少配置段: {section}"


def test_load_config_returns_independent_copy(monkeypatch):
    """即使底层 YAML 结果被复用（未来若加缓存），返回值也必须是独立深拷贝。

    直接调用两次 load_config 无法验证这一点——yaml.safe_load 每次都会构造全新对象图，
    所以去掉 deepcopy 该断言依然通过。这里把一个共享 dict 作为解析结果注入，
    让 deepcopy 成为唯一能保证隔离的环节。
    """
    import src.config as config_module

    shared = {"detect": {"report_n_sigma": 5.0}}
    monkeypatch.setattr(config_module.yaml, "safe_load", lambda fh: shared)

    a = load_config()
    b = load_config()
    a["detect"]["report_n_sigma"] = 99.0
    assert b["detect"]["report_n_sigma"] == 5.0
    assert shared["detect"]["report_n_sigma"] == 5.0


def test_detect_thresholds_are_two_tiered():
    conf = load_config()["detect"]
    assert conf["search_n_sigma"] < conf["report_n_sigma"], "搜索阈值必须低于报数阈值"
