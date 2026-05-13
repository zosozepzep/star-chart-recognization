#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_sextractor.py

使用 SExtractor 对整个 data/images 目录进行批量检测，
输出统一格式结果，便于与自定义 detector.py 对比。

项目结构:
.
├── data/images/
├── output/sextractor_test/
│   ├── default.sex
│   ├── default.param
│   ├── default.conv
│   └── default.nnw
└── src/benchmark_sextractor.py
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt


# =========================
# 路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "images" / "60394_20260309_1437117414637934_PIC"

SEX_DIR = PROJECT_ROOT / "output" / "sextractor_test"

CAT_DIR = SEX_DIR / "catalogs"
JSON_DIR = SEX_DIR / "json"
PNG_DIR = SEX_DIR / "preview"

SUMMARY_CSV = SEX_DIR / "summary.csv"

for d in [CAT_DIR, JSON_DIR, PNG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =========================
# 检查 source-extractor
# =========================
def get_sextractor_command():
    for cmd in ["source-extractor", "sex"]:
        if shutil.which(cmd):
            return cmd
    raise RuntimeError("未找到 source-extractor 或 sex 命令。")


SEX_CMD = get_sextractor_command()


# =========================
# 检查配置文件
# =========================
SEX_CONFIG = SEX_DIR / "default.sex"
PARAM_FILE = SEX_DIR / "default.param"

if not SEX_CONFIG.exists():
    raise FileNotFoundError(f"缺少配置文件: {SEX_CONFIG}")

if not PARAM_FILE.exists():
    raise FileNotFoundError(f"缺少参数文件: {PARAM_FILE}")


# =========================
# 运行 SExtractor
# =========================
def run_sextractor(fits_path: Path):
    stem = fits_path.stem
    cat_path = CAT_DIR / f"{stem}.cat"

    conv_file = SEX_DIR / "default.conv"
    nnw_file = SEX_DIR / "default.nnw"

    cmd = [
        SEX_CMD,
        str(fits_path),
        "-c", str(SEX_CONFIG),
        "-PARAMETERS_NAME", str(PARAM_FILE),
        "-CATALOG_NAME", str(cat_path),
        "-CATALOG_TYPE", "ASCII_HEAD",
        "-CLASSIFY", "N",
    ]

    if conv_file.exists():
        cmd += ["-FILTER_NAME", str(conv_file)]

    if nnw_file.exists():
        cmd += ["-STARNNW_NAME", str(nnw_file)]

    print(" ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

    if not cat_path.exists():
        raise FileNotFoundError(f"SExtractor 未生成输出文件: {cat_path}")

    return cat_path


# =========================
# 读取星表
# =========================
def load_catalog(cat_path: Path) -> pd.DataFrame:
    table = Table.read(cat_path, format="ascii.sextractor")
    df = table.to_pandas()
    return df


# =========================
# 保存 JSON
# =========================
def save_json(df: pd.DataFrame, json_path: Path):
    records = df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


# =========================
# 绘制预览图
# =========================
def create_preview(fits_path: Path, df: pd.DataFrame, png_path: Path):
    data = fits.getdata(fits_path)

    if data.ndim > 2:
        data = data.squeeze()

    finite = np.isfinite(data)
    vmin = np.percentile(data[finite], 5)
    vmax = np.percentile(data[finite], 99)

    plt.figure(figsize=(10, 10))
    plt.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    if "X_IMAGE" in df.columns and "Y_IMAGE" in df.columns:
        plt.scatter(
            df["X_IMAGE"] - 1,
            df["Y_IMAGE"] - 1,
            s=20,
            facecolors="none",
            edgecolors="lime",
            linewidths=0.6,
        )

    plt.title(f"{fits_path.name}  ({len(df)} detections)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


# =========================
# 单文件处理
# =========================
def process_one(fits_path: Path):
    stem = fits_path.stem

    cat_path = run_sextractor(fits_path)
    df = load_catalog(cat_path)

    json_path = JSON_DIR / f"{stem}.json"
    csv_path = CAT_DIR / f"{stem}.csv"
    png_path = PNG_DIR / f"{stem}.png"

    df.to_csv(csv_path, index=False)
    save_json(df, json_path)
    create_preview(fits_path, df, png_path)

    result = {
        "file": fits_path.name,
        "n_sources": len(df),
        "catalog": str(cat_path.relative_to(PROJECT_ROOT)),
        "csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "json": str(json_path.relative_to(PROJECT_ROOT)),
        "preview": str(png_path.relative_to(PROJECT_ROOT)),
    }

    print(f"Detected {len(df)} sources in {fits_path.name}")

    return result


# =========================
# 主程序
# =========================
def main():
    fits_files = sorted(
        list(DATA_DIR.glob("*.fits")) +
        list(DATA_DIR.glob("*.fit")) +
        list(DATA_DIR.glob("*.fts"))
    )

    if not fits_files:
        print("No FITS files found.")
        return

    print(f"Found {len(fits_files)} FITS files.")

    summary = []

    for fits_file in fits_files:
        try:
            result = process_one(fits_file)
            summary.append(result)
        except Exception as e:
            print(f"Failed on {fits_file.name}: {e}")

    if summary:
        df_summary = pd.DataFrame(summary)
        df_summary.to_csv(SUMMARY_CSV, index=False)
        print(f"Summary saved to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()