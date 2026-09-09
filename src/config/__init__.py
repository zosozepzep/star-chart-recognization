"""集中式配置加载。所有阈值与窗口参数都必须来自这里，代码中不写魔数。"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """读取 YAML 配置并返回可安全修改的深拷贝。"""
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(target, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件根节点必须是映射: {target}")
    return copy.deepcopy(data)
