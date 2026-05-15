from pathlib import Path

from .cross_match import run_batch_cross_match
from .sextractor_runner import run_sextractor_on_directory


def run_detection_benchmark(
    image_dir,
    pipeline_catalog_path,
    output_dir="output/benchmark",
    config_dir=None,
    match_radius=2.0,
    max_frames=None,
    frame_names=None,
):
    """
    detection-level benchmark 总入口。

    这个函数负责把两个步骤串起来：
    1. 对输入 FITS 图像自动运行 SExtractor，生成 SE catalog
    2. 用我方 all_raw_detections.csv 和 SE catalog 做逐帧 cross-match

    注意：
    这里只评估单帧 detection，不评估多帧 tracking。

    max_frames / frame_names 用于快速验证流程。
    例如 max_frames=2 表示只跑前两帧。
    """
    output_dir = Path(output_dir)
    sextractor_output_dir = output_dir / "sextractor"
    cross_match_output_dir = output_dir / "cross_match"

    print("Running SExtractor benchmark baseline...")
    sextractor_result = run_sextractor_on_directory(
        image_dir=image_dir,
        output_dir=sextractor_output_dir,
        config_dir=config_dir,
        max_frames=max_frames,
        frame_names=frame_names,
    )

    print("Running cross-match benchmark...")
    cross_match_result = run_batch_cross_match(
        pipeline_catalog_path=pipeline_catalog_path,
        sextractor_catalog_dir=sextractor_result["catalog_dir"],
        output_dir=cross_match_output_dir,
        match_radius=match_radius,
        max_frames=max_frames,
        frame_names=frame_names,
    )

    return {
        "sextractor": sextractor_result,
        "cross_match": cross_match_result,
    }
