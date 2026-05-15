from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits

from src.benchmark.cross_match import run_cross_match


def render_pipeline_only_diagnostic(fits_image_path, pipeline_only_df, output_image="diagnostic_plot.png"):
    """
    生成一张单帧诊断图。

    这部分属于可视化辅助，不参与 benchmark 指标计算。
    它的作用是把 pipeline-only 的点画在原始 FITS 图像上，
    方便人工观察这些点是真实暗星、噪声，还是亮星附近的伪目标。
    """
    # 读取 FITS 原图数据。
    # 后面所有红圈都会叠加在这张图上。
    data = fits.getdata(fits_image_path)

    # Matplotlib 的 LogNorm 要求图像值必须大于 0。
    # FITS 数据里可能存在 0 或负数，所以这里先复制一份显示用数据，
    # 再把所有非正数替换成一个很小的正数，避免画图时报错。
    display_data = data.copy()
    floor_val = np.percentile(display_data[display_data > 0], 1) if np.any(display_data > 0) else 1e-5
    display_data[display_data <= 0] = floor_val

    # 使用 1% 到 99.5% 分位数做显示范围。
    # 这样可以压住极亮点，让暗弱结构也能看得见。
    vmin = np.percentile(display_data, 1)
    vmax = np.percentile(display_data, 99.5)

    # 防御极端情况：
    # 如果图像几乎是常数，分位数可能相等，此时退回到 min/max。
    if vmin >= vmax:
        vmin = display_data.min()
        vmax = display_data.max()

    # 画原始 FITS 图像。
    # Greys_r 表示反色灰度图，origin="lower" 保持天文图像常见坐标方向。
    plt.figure(figsize=(12, 12))
    plt.imshow(display_data, cmap="Greys_r", origin="lower", norm=LogNorm(vmin=vmin, vmax=vmax))

    # pipeline_only_df 表示“我方检测到，但 SExtractor 没匹配上”的点。
    # 这里用红色空心圆标出来，方便观察这些点集中在哪里。
    if not pipeline_only_df.empty:
        plt.scatter(
            pipeline_only_df["X"],
            pipeline_only_df["Y"],
            s=40,
            facecolors="none",
            edgecolors="red",
            linewidth=1.5,
            label="Pipeline Only",
        )

    # 添加基本标题、坐标轴和图例。
    plt.title("Cross-Match Diagnostic: Pipeline vs SExtractor", fontsize=16)
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")
    plt.legend(loc="upper right")

    # 保存高分辨率图片，供人工查看。
    # close() 用来释放 matplotlib 当前图，避免批量画图时占内存。
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    plt.close()


def run_cross_match_diagnosis(
    pipeline_csv,
    sextractor_csv,
    fits_image_path=None,
    match_radius=2.0,
    output_dir="output/benchmark",
    output_image="diagnostic_plot.png",
):
    print("Starting single-frame cross-match benchmark...")
    result = run_cross_match(
        pipeline_catalog_path=pipeline_csv,
        sextractor_catalog_path=sextractor_csv,
        match_radius=match_radius,
        output_dir=output_dir,
    )

    for key, value in result["summary"].items():
        print(f" -> {key}: {value}")

    if fits_image_path:
        render_pipeline_only_diagnostic(
            fits_image_path=fits_image_path,
            pipeline_only_df=result["pipeline_only"],
            output_image=output_image,
        )
        print(f"Diagnostic image saved to {output_image}")

    print(f"Benchmark outputs saved to {Path(output_dir)}")
    return result

if __name__ == "__main__":
    # 这里保持原脚本风格：直接在文件里填写路径，然后运行。
    # 如果想换数据，只需要改下面这几个变量。
    PIPELINE_CSV = "output/new/all_raw_detections.csv"
    SEXTRACTOR_CSV = "output/benchmark/sextractor/catalogs/20260330163205413_9901.csv"
    FITS_IMAGE = None
    OUTPUT_DIR = "output/benchmark"
    MATCH_RADIUS = 2.0

    run_cross_match_diagnosis(
        pipeline_csv=PIPELINE_CSV,
        sextractor_csv=SEXTRACTOR_CSV,
        fits_image_path=FITS_IMAGE,
        match_radius=MATCH_RADIUS,
        output_dir=OUTPUT_DIR,
    )
