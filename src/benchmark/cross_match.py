import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# 我方 pipeline 导出的 Level 1 检测表中，坐标列固定叫 X / Y。
PIPELINE_X_COL = "X"
PIPELINE_Y_COL = "Y"

# SExtractor 不同配置可能导出不同名字的坐标列。
# X_IMAGE / Y_IMAGE 是普通质心坐标，XWIN_IMAGE / YWIN_IMAGE 是窗口化质心坐标。
# benchmark 不强行要求用户记住这些列名，而是在下面自动识别。
SEXTRACTOR_COORD_CANDIDATES = [
    ("X_IMAGE", "Y_IMAGE"),
    ("XWIN_IMAGE", "YWIN_IMAGE"),
]


def read_catalog(path):
    """
    读取星表文件。

    pipeline 的输出通常是逗号分隔 CSV。
    SExtractor 的输出有时是逗号分隔，有时是空格分隔。

    这里先按 CSV 读取，如果发现只读到 1 列，说明它很可能不是逗号分隔；
    这时再按“任意空白字符分隔”重新读取。

    支持普通 CSV，也支持 SExtractor 常见的空格分隔文本。
    """
    path = Path(path)

    # SExtractor 的 ASCII_HEAD .cat 文件比较特殊：
    # 前几行用 "# 1 X_IMAGE ..." 这种形式写列名，后面才是纯数据。
    # pandas 不能直接把这种文件读成带列名的表，所以这里单独处理。
    if path.suffix.lower() == ".cat":
        return read_sextractor_cat(path)

    try:
        df = pd.read_csv(path)
        if len(df.columns) > 1:
            return df
    except pd.errors.ParserError:
        pass

    return pd.read_csv(path, sep=r"\s+")


def read_sextractor_cat(path):
    """
    读取 SExtractor 的 ASCII_HEAD 格式 .cat 文件。

    文件头部通常长这样：
    #   1 NUMBER
    #   2 X_IMAGE
    #   3 Y_IMAGE

    真正的数据行没有 #，只是空格分隔的数字。
    这个函数会先从注释行里提取列名，再读取后面的数字表格。
    """
    path = Path(path)
    columns = []
    data_rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("#"):
                parts = stripped.split()
                if len(parts) >= 3 and parts[1].isdigit():
                    columns.append(parts[2])
                continue

            data_rows.append(stripped.split())

    if not columns:
        raise ValueError(f"Cannot find column names in SExtractor catalog: {path}")

    df = pd.DataFrame(data_rows, columns=columns)

    # SExtractor 输出基本都是数值列。
    # 这里逐列尝试转换成数字；如果未来出现非数值列，就保留原始字符串。
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass

    return df


def find_sextractor_xy_columns(df):
    """
    自动识别 SExtractor 输出里的坐标列。

    返回值是两个字符串：
    - x_col: SExtractor 星表里的 x 坐标列名
    - y_col: SExtractor 星表里的 y 坐标列名

    如果找不到坐标列，直接报错。因为没有坐标就无法做 cross-match。
    """
    for x_col, y_col in SEXTRACTOR_COORD_CANDIDATES:
        if x_col in df.columns and y_col in df.columns:
            return x_col, y_col

    raise ValueError(f"Cannot find SExtractor coordinate columns. Current columns: {df.columns.tolist()}")


def cross_match_catalogs(
    pipeline_df,
    sextractor_df,
    match_radius=2.0,
    pipeline_x_col=PIPELINE_X_COL,
    pipeline_y_col=PIPELINE_Y_COL,
    sextractor_x_col=None,
    sextractor_y_col=None,
):
    """
    对单帧检测结果做 cross-match。

    match_radius 的单位是像素。
    默认值 2.0 不是从数据里自动估计出来的，而是沿用原诊断脚本的经验值。
    规划文档里也提到 cross-match 可以允许 1~2 像素误差。

    这个参数的含义是：
    如果 pipeline 目标和 SExtractor 目标之间的距离 <= match_radius，
    就认为它们有可能是同一个真实目标。

    后续如果要做更严谨的 benchmark，可以尝试多个半径，
    例如 1.0 / 1.5 / 2.0 / 3.0，观察 precision 和 recall 对半径的敏感程度。

    返回一个普通 dict：
    - matched: 双方匹配上的目标
    - pipeline_only: 只被我方检测到的目标
    - sextractor_only: 只被 SExtractor 检测到的目标
    - summary: precision / recall / F1 等统计指标

    注意：这个函数只评估单帧 detection，不评估多帧 tracking。
    """
    # 先检查我方星表有没有坐标列。
    # 如果没有 X / Y，后面无法构造坐标矩阵，应该尽早报错。
    if pipeline_x_col not in pipeline_df.columns or pipeline_y_col not in pipeline_df.columns:
        raise ValueError(f"Pipeline catalog must contain {pipeline_x_col} and {pipeline_y_col}.")

    # 如果调用方没有手动指定 SExtractor 坐标列，就自动识别。
    # 如果调用方指定了坐标列，也要检查这些列确实存在。
    if sextractor_x_col is None or sextractor_y_col is None:
        sextractor_x_col, sextractor_y_col = find_sextractor_xy_columns(sextractor_df)
    elif sextractor_x_col not in sextractor_df.columns or sextractor_y_col not in sextractor_df.columns:
        raise ValueError(f"SExtractor catalog must contain {sextractor_x_col} and {sextractor_y_col}.")

    # reset_index 的目的：
    # DataFrame 经过筛选或读取后，原始行号不一定连续。
    # 这里把原始行号保存成 pipeline_index / sextractor_index，方便后续追溯匹配来自哪一行。
    pipeline_work = pipeline_df.reset_index(drop=False).rename(columns={"index": "pipeline_index"})
    sextractor_work = sextractor_df.reset_index(drop=False).rename(columns={"index": "sextractor_index"})

    # 把坐标列提取成 numpy 数组，形状是：
    # [[x1, y1],
    #  [x2, y2],
    #  ...]
    # KDTree 需要这种纯数值矩阵作为输入。
    pipeline_coords = pipeline_work[[pipeline_x_col, pipeline_y_col]].to_numpy(dtype=float)
    sextractor_coords = sextractor_work[[sextractor_x_col, sextractor_y_col]].to_numpy(dtype=float)

    # 防御空输入：
    # 如果其中一边没有任何目标，就不需要跑 KDTree，直接返回“全都未匹配”的结果。
    if len(pipeline_coords) == 0 or len(sextractor_coords) == 0:
        summary = build_summary(len(pipeline_coords), len(sextractor_coords), 0, np.array([]))
        return {
            "matched": pd.DataFrame(),
            "pipeline_only": pipeline_work.copy(),
            "sextractor_only": sextractor_work.copy(),
            "summary": summary,
        }

    # KDTree 是一种空间索引结构。
    # 直观理解：它能快速回答“某个点附近 r 像素内有哪些点”。
    # 如果不用 KDTree，就要两两比较所有点，目标多的时候会非常慢。
    tree = cKDTree(sextractor_coords)

    # nearby_indices[i] 表示：
    # 第 i 个 pipeline 目标，在 match_radius 半径内找到了哪些 SExtractor 目标。
    nearby_indices = tree.query_ball_point(pipeline_coords, r=match_radius)

    # candidates 保存所有“可能匹配”的点对：
    # (距离, pipeline 的行号, SExtractor 的行号)
    #
    # 这里不是只找最近的一个点，而是找半径内所有候选。
    # 这样可以避免一个 SExtractor 点被多个 pipeline 点抢占时，后面的点没有机会匹配第二近邻。
    candidates = []
    for pipeline_idx, candidate_indices in enumerate(nearby_indices):
        for sextractor_idx in candidate_indices:
            distance = np.linalg.norm(pipeline_coords[pipeline_idx] - sextractor_coords[sextractor_idx])
            candidates.append((float(distance), pipeline_idx, int(sextractor_idx)))

    # 从距离最近的候选开始处理。
    # 这是一个简单、稳定、容易解释的一对一匹配策略。
    candidates.sort(key=lambda item: item[0])

    # 这两个 set 用来记录已经被匹配过的目标。
    # 一旦某个 pipeline 或 SExtractor 目标已经进入 matched，就不能再参与第二次匹配。
    matched_pipeline_indices = set()
    matched_sextractor_indices = set()
    match_rows = []

    for distance, pipeline_idx, sextractor_idx in candidates:
        # 一个 pipeline 目标只能匹配一个 SExtractor 目标。
        if pipeline_idx in matched_pipeline_indices:
            continue

        # 一个 SExtractor 目标也只能匹配一个 pipeline 目标。
        if sextractor_idx in matched_sextractor_indices:
            continue

        matched_pipeline_indices.add(pipeline_idx)
        matched_sextractor_indices.add(sextractor_idx)

        # 把双方的原始字段都保留下来。
        # 加 prefix 是为了避免两边有同名字段，比如都叫 X 或 FLUX。
        pipeline_row = pipeline_work.iloc[pipeline_idx].add_prefix("pipeline_")
        sextractor_row = sextractor_work.iloc[sextractor_idx].add_prefix("sextractor_")

        # 一条 matched 记录包含：
        # - pipeline 目标的所有字段
        # - SExtractor 目标的所有字段
        # - 两个点之间的像素距离
        row = {}
        row.update(pipeline_row.to_dict())
        row.update(sextractor_row.to_dict())
        row["match_distance"] = distance
        match_rows.append(row)

    # matched: 双方都检测到的目标。
    matched = pd.DataFrame(match_rows)

    # pipeline_only: 我方检测到了，但 SExtractor 没匹配上。
    # 在以 SExtractor 为 baseline 的语境里，它通常可以先看作 false positive / orphan。
    pipeline_only = pipeline_work.drop(index=list(matched_pipeline_indices)).reset_index(drop=True)

    # sextractor_only: SExtractor 检测到了，但我方没匹配上。
    # 在以 SExtractor 为 baseline 的语境里，它通常可以先看作 missed / false negative。
    sextractor_only = sextractor_work.drop(index=list(matched_sextractor_indices)).reset_index(drop=True)

    # 匹配距离用于评估定位精度。
    # 距离越小，说明双方对同一目标的位置估计越接近。
    if matched.empty:
        distances = np.array([])
    else:
        distances = matched["match_distance"].to_numpy(dtype=float)

    # 根据匹配数量和距离统计，生成 benchmark 的摘要指标。
    summary = build_summary(
        pipeline_total=len(pipeline_work),
        sextractor_total=len(sextractor_work),
        matched_total=len(matched),
        distances=distances,
    )

    return {
        "matched": matched,
        "pipeline_only": pipeline_only,
        "sextractor_only": sextractor_only,
        "summary": summary,
    }


def build_summary(pipeline_total, sextractor_total, matched_total, distances):
    """
    根据 cross-match 结果计算指标。

    这里暂时把 SExtractor 当作 baseline：
    - precision = matched / pipeline_total
      我方检出的目标里，有多少能和 SExtractor 对上。

    - recall = matched / sextractor_total
      SExtractor 检出的目标里，有多少也被我方检出。

    - F1 是 precision 和 recall 的综合指标。
    """
    pipeline_only_total = pipeline_total - matched_total
    sextractor_only_total = sextractor_total - matched_total

    # 避免除以 0：
    # 如果某一边星表为空，对应指标直接给 0。
    precision = matched_total / pipeline_total if pipeline_total else 0.0
    recall = matched_total / sextractor_total if sextractor_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "pipeline_total": int(pipeline_total),
        "sextractor_total": int(sextractor_total),
        "matched_total": int(matched_total),
        "pipeline_only_total": int(pipeline_only_total),
        "sextractor_only_total": int(sextractor_only_total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "match_distance_mean": float(np.mean(distances)) if len(distances) else 0.0,
        "match_distance_median": float(np.median(distances)) if len(distances) else 0.0,
        "match_distance_max": float(np.max(distances)) if len(distances) else 0.0,
    }


def run_cross_match(pipeline_catalog_path, sextractor_catalog_path, match_radius=2.0, output_dir=None):
    """
    文件级入口。

    这个函数负责：
    1. 从磁盘读取两个 catalog 文件
    2. 调用 cross_match_catalogs 做匹配和指标计算
    3. 如果指定 output_dir，就把结果写到磁盘
    """
    pipeline_df = read_catalog(pipeline_catalog_path)
    sextractor_df = read_catalog(sextractor_catalog_path)

    result = cross_match_catalogs(
        pipeline_df=pipeline_df,
        sextractor_df=sextractor_df,
        match_radius=match_radius,
    )

    if output_dir is not None:
        write_cross_match_outputs(result, output_dir)

    return result


def write_cross_match_outputs(result, output_dir):
    """
    把 benchmark 结果写到输出目录。

    输出文件：
    - matched.csv: 双方匹配上的目标明细
    - pipeline_only.csv: 只被我方检测到的目标
    - sextractor_only.csv: 只被 SExtractor 检测到的目标
    - summary.csv: 一行摘要指标，方便用 Excel 打开
    - summary.json: 同样的摘要指标，方便程序继续读取
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result["matched"].to_csv(output_path / "matched.csv", index=False)
    result["pipeline_only"].to_csv(output_path / "pipeline_only.csv", index=False)
    result["sextractor_only"].to_csv(output_path / "sextractor_only.csv", index=False)

    pd.DataFrame([result["summary"]]).to_csv(output_path / "summary.csv", index=False)
    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2, ensure_ascii=False)


def run_batch_cross_match(
    pipeline_catalog_path,
    sextractor_catalog_dir,
    output_dir,
    match_radius=2.0,
    max_frames=None,
    frame_names=None,
):
    """
    对多帧 detection catalog 做批量 benchmark。

    我方 pipeline 的 all_raw_detections.csv 是所有帧混在一个文件里，
    用 Frame_Name 区分每一帧。

    SExtractor 是每一帧一个 catalog 文件。
    所以这里的逻辑是：
    1. 读取我方 all_raw_detections.csv
    2. 按 Frame_Name 分组
    3. 为每一帧找到同名的 SExtractor catalog
    4. 每一帧单独 cross-match
    5. 汇总所有帧的总指标

    max_frames / frame_names 是快速验证用的过滤参数。
    正式评测时不传，默认处理所有帧。
    """
    pipeline_df = read_catalog(pipeline_catalog_path)
    sextractor_catalog_dir = Path(sextractor_catalog_dir)
    output_dir = Path(output_dir)

    if "Frame_Name" not in pipeline_df.columns:
        raise ValueError("Pipeline catalog must contain Frame_Name for batch benchmark.")

    if frame_names is not None:
        allowed_names = set(str(name) for name in frame_names)
        pipeline_df = pipeline_df[pipeline_df["Frame_Name"].isin(allowed_names)]

    if max_frames is not None:
        selected_frames = list(pipeline_df["Frame_Name"].drop_duplicates())[:max_frames]
        pipeline_df = pipeline_df[pipeline_df["Frame_Name"].isin(selected_frames)]

    frame_output_dir = output_dir / "frames"
    frame_output_dir.mkdir(parents=True, exist_ok=True)

    frame_summaries = []
    all_matched = []
    all_pipeline_only = []
    all_sextractor_only = []

    for frame_name, frame_df in pipeline_df.groupby("Frame_Name"):
        frame_stem = Path(str(frame_name)).stem
        sextractor_catalog_path = find_catalog_for_frame(sextractor_catalog_dir, frame_stem)

        if sextractor_catalog_path is None:
            print(f"Skipping benchmark for {frame_name}: SExtractor catalog not found.")
            continue

        sextractor_df = read_catalog(sextractor_catalog_path)
        result = cross_match_catalogs(
            pipeline_df=frame_df,
            sextractor_df=sextractor_df,
            match_radius=match_radius,
        )

        current_output_dir = frame_output_dir / frame_stem
        write_cross_match_outputs(result, current_output_dir)

        summary = dict(result["summary"])
        summary["Frame_Name"] = frame_name
        summary["sextractor_catalog"] = str(sextractor_catalog_path)
        frame_summaries.append(summary)

        all_matched.append(add_frame_name(result["matched"], frame_name))
        all_pipeline_only.append(add_frame_name(result["pipeline_only"], frame_name))
        all_sextractor_only.append(add_frame_name(result["sextractor_only"], frame_name))

    if not frame_summaries:
        raise RuntimeError("No frame was benchmarked. Please check SExtractor catalog output.")

    matched = concat_tables(all_matched)
    pipeline_only = concat_tables(all_pipeline_only)
    sextractor_only = concat_tables(all_sextractor_only)

    # 批量 benchmark 的总结果采用“逐帧平均值”。
    #
    # 注意：这里不再把所有帧的 detection 数量简单相加后计算 precision / recall。
    # 因为当前 benchmark 是 detection-level，不是 tracking-level。
    # 每一帧图像应该先独立评估，然后再对各帧指标取平均。
    #
    # 这类平均方式通常叫 macro-average：
    # - 每一帧权重相同
    # - 不会让目标数量特别多的帧主导最终指标
    overall_summary = build_average_summary(frame_summaries)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame_summaries).to_csv(output_dir / "frame_summary.csv", index=False)
    pd.DataFrame([overall_summary]).to_csv(output_dir / "summary.csv", index=False)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)

    matched.to_csv(output_dir / "matched.csv", index=False)
    pipeline_only.to_csv(output_dir / "pipeline_only.csv", index=False)
    sextractor_only.to_csv(output_dir / "sextractor_only.csv", index=False)

    return {
        "matched": matched,
        "pipeline_only": pipeline_only,
        "sextractor_only": sextractor_only,
        "frame_summary": pd.DataFrame(frame_summaries),
        "summary": overall_summary,
    }


def build_average_summary(frame_summaries):
    """
    根据逐帧 summary 计算平均指标。

    这里的 summary 是“各帧平均值”，不是把所有检测点合并后的总数统计。
    这样更适合当前 detection-level benchmark。
    """
    metric_names = [
        "pipeline_total",
        "sextractor_total",
        "matched_total",
        "pipeline_only_total",
        "sextractor_only_total",
        "precision",
        "recall",
        "f1",
        "match_distance_mean",
        "match_distance_median",
        "match_distance_max",
    ]

    summary = {
        "summary_type": "frame_average",
        "frame_count": len(frame_summaries),
    }

    for metric_name in metric_names:
        values = [item[metric_name] for item in frame_summaries]
        summary[metric_name] = float(np.mean(values)) if values else 0.0

    return summary


def find_catalog_for_frame(catalog_dir, frame_stem):
    """
    根据 FITS 文件名查找对应的 SExtractor catalog。

    例如输入帧是：
    20260330163205413_9901.fits

    那么这里会优先找：
    20260330163205413_9901.csv
    20260330163205413_9901.cat
    """
    catalog_dir = Path(catalog_dir)
    csv_path = catalog_dir / f"{frame_stem}.csv"
    cat_path = catalog_dir / f"{frame_stem}.cat"

    if csv_path.exists():
        return csv_path
    if cat_path.exists():
        return cat_path

    return None


def add_frame_name(df, frame_name):
    """
    给输出明细表补一列 Frame_Name，方便后续知道每条记录来自哪一帧。
    """
    if df.empty:
        return df

    df = df.copy()
    df["Frame_Name"] = frame_name
    return df


def concat_tables(tables):
    """
    合并多个 DataFrame。
    如果列表里没有有效表，就返回一个空 DataFrame。
    """
    valid_tables = [df for df in tables if not df.empty]
    if not valid_tables:
        return pd.DataFrame()

    return pd.concat(valid_tables, ignore_index=True)
