"""Reproducible FITS analysis and CSV/JSON delivery: python -m src.cli --help."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import logging
import math
import hashlib
import time
from pathlib import Path

import numpy as np

from src.analysis.star_catalog import star_catalog
from src.config import load_config
from src.dataio.fits_loader import FrameSequence
from src.pipeline import analyze_sequence


class _PaddingWarningFilter(logging.Filter):
    """Astropy resets warning registries while reading; deduplicate at its logger."""
    def __init__(self):
        super().__init__()
        self.seen = set()

    def filter(self, record):
        message = record.getMessage()
        if not message.startswith("File may have been truncated"):
            return True
        if message in self.seen:
            return False
        self.seen.add(message)
        return True


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def calibrate_after_detection(run, sequence):
    """Explicit opt-in boundary: truth is loaded only after all target decisions."""
    from src.astrometry.photometry import zero_point_from_truth
    from src.validate.truth import load_truth, match_frames

    if sequence.truth_path is None:
        return None, "没有 .DAT 文件；星等未定标。"
    if len(run.targets) != 1:
        return None, "真值文件没有目标关联标识；仅在恰有一条目标轨迹时支持零点定标。"
    truth = load_truth(sequence.truth_path)
    track = run.targets[0].track
    frames, rows = match_frames(truth, sequence, track.frames)
    if len(set(frames)) != len(frames):
        raise ValueError("多个真值记录对应同一帧，不能用于零点定标")
    # Exposure-normalized fixed aperture, identical to the star measurements.
    rates = run.target_flux_rates(track, frames)
    zp = zero_point_from_truth(rates, truth.mag[rows], exposure_s=1.0)
    return zp, "固定孔径 ADU/s 与 .DAT 星等拟合；滤光波段未给定，含系统误差。"


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_safe(rows))


def run_analysis(args):
    started = time.perf_counter()
    config = load_config(args.config)
    output = args.output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("输出目录已存在且非空，请指定新的 --output 以保留已有结果")
    if getattr(args, 'catalog', None):
        if args.calibrate_from_truth:
            raise ValueError('--catalog 与 --calibrate-from-truth 不能同时使用')
        if not args.catalog.is_file():
            raise FileNotFoundError(f'找不到参考星表：{args.catalog}')
    if not args.no_plot:
        from src.viz.figures import setup_matplotlib
        setup_matplotlib()  # Fail before doing expensive work if fonts are unavailable.
    sequence = FrameSequence.from_directory(args.input)
    run = analyze_sequence(sequence, config, reference_frame=args.reference_frame, frame_range=args.frames)
    zp, note = None, "未请求真值定标；仅输出仪器星等。"
    catalog_fit = None
    if getattr(args, 'catalog', None):
        from src.astrometry.catalog_match import calibrate_catalog
        auxiliary = getattr(sequence.headers[run.reference_frame], 'space_aux', None)
        if auxiliary is None:
            raise ValueError('星表匹配目前需要天基辅助数据提供的光轴指向')
        try:
            zp, catalog_fit = calibrate_catalog(run.stars[run.reference_frame],
                                                sequence.headers[run.reference_frame].exposure_s,
                                                args.catalog, [auxiliary['ra_deg'], auxiliary['dec_deg']],
                                                config['catalog_match'])
            note = 'Gaia DR3 G 参照的近似星等；相机波段未知，零点散度不代表完整误差。'
        except ValueError as exc:
            note = f'星表定标未完成：{exc}；保留仪器星等。'
            logging.warning(note)
    if args.calibrate_from_truth:
        try:
            zp, note = calibrate_after_detection(run, sequence)
        except ValueError as exc:
            note = f"定标未完成：{exc}；保留独立识别结果。"
            logging.warning(note)
    catalog = star_catalog(run.stars, run.registration, run.reference_frame,
                           min_hits=config["star_report"]["min_hits"],
                           radius=config["repeatability"]["match_radius_px"],
                           exposure_s=sequence.headers[run.reference_frame].exposure_s,
                           zero_point=zp)
    catalog["n_detected_sources"] = run.raw_star_count
    catalog["detection_n_sigma"] = config["detect"]["report_n_sigma"]
    catalog["aperture_radius_px"] = config["detect"]["aperture_radius_px"]
    catalog["calibration_note"] = note
    if catalog_fit is not None:
        catalog['magnitude_system'] = 'Gaia DR3 G-referenced approximate magnitude; camera passband unspecified'
        catalog['catalog_fit'] = catalog_fit
    target_rows, verdicts = [], []
    for target_id, verdict in enumerate(run.targets, 1):
        verdicts.append({"target_id": target_id, **verdict.to_dict()})
        track = verdict.track
        rates = run.target_flux_rates(track, track.frames)
        for i, frame in enumerate(track.frames):
            target_rows.append({
                "target_id": target_id, "frame": frame,
                "time_utc": sequence.headers[frame].date_obs.isot,
                "x_det": float(track.xy_det[i, 0]), "y_det": float(track.xy_det[i, 1]),
                "x_sky_px": float(track.xy_sky[i, 0]), "y_sky_px": float(track.xy_sky[i, 1]),
                "aperture_flux_adu_per_s": float(rates[i]),
            })
    report = {
        "schema_version": 1, "dataset_id": sequence.dataset_id,
        "input_frame_count": len(sequence), "analysis_frames": run.frames,
        "reference_file": sequence.headers[run.reference_frame].path.name,
        "algorithm_scope": ("天基恒星背景配准、静止源剔除、按实际时间连接多条运动候选轨迹；目标身份未定。"
                            if getattr(sequence, 'observation_mode', 'ground') == 'space' else
                            "地基望远镜跟踪段；探测器系近静止且天球系运动的目标。"),
        "coordinates": "0-based x=column y=row; sky coordinates are reference-frame pixels, not RA/Dec",
        "truth_used_for_detection": False, "config": config,
        "registration": run.registration.report(), "target_count": len(verdicts),
        "targets": verdicts, "star_catalog": catalog,
        "diagnostics": run.diagnostics,
    }
    if getattr(sequence, 'observation_mode', 'ground') == 'space':
        report['space_telemetry'] = [dict(frame=f, time_utc=sequence.headers[f].date_obs.isot,
                                         **sequence.headers[f].space_aux) for f in run.frames]
        report['input_files'] = []
        for h in sequence.headers:
            with h.path.open('rb') as fh:
                digest = hashlib.file_digest(fh, 'sha256').hexdigest()
            report['input_files'].append(dict(file=h.path.name, bytes=h.path.stat().st_size, sha256=digest))
        report['star_catalog']['note'] += ' 静止星场下，跨帧重复本身不能排除所有固定传感器缺陷。'
        from src.analysis.space_summary import summarize_space_observation
        report['space_summary'] = summarize_space_observation(sequence, run, catalog_fit)
    report['analysis_seconds'] = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2,
                                                  allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output / "stars.csv", catalog["sources"],
              ["source_id", "frame", "x", "y", "aperture_flux_adu", "instrumental_mag_rate", "mag", "hits", "confirmed"])
    write_csv(output / "targets.csv", target_rows,
              ["target_id", "frame", "time_utc", "x_det", "y_det", "x_sky_px", "y_sky_px", "aperture_flux_adu_per_s"])
    if not args.no_plot:
        from src.viz.contest import plot_star_catalog
        plot_star_catalog(sequence.image(run.reference_frame), catalog, output / "faintest-star.png")
        if getattr(sequence, 'observation_mode', 'ground') == 'space':
            from src.viz.contest import plot_motion_tracks
            plot_motion_tracks(sequence, run, output / 'motion-tracks.png')
    print(json.dumps({"output": str(output), "target_count": len(verdicts),
                      "detected_sources": run.raw_star_count,
                      "confirmed_stars": catalog["n_confirmed_stars"],
                      "faintest_confirmed_star": catalog["faintest_confirmed_star"],
                      "calibration_note": note}, ensure_ascii=False, indent=2, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="星图识别：指定图像计数、最暗恒星候选与全部目标轨迹导出")
    parser.add_argument("--input", type=Path, required=True, help="带观测头的 FITS 序列目录")
    parser.add_argument("--output", type=Path, default=Path("output") / datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    parser.add_argument("--reference-frame", type=int, help="比赛指定图像的 0-based 时序编号，默认跟踪段中帧")
    parser.add_argument("--frames", type=int, nargs=2, metavar=("START", "END"), help="显式指定连续分析帧，包含 END")
    parser.add_argument("--config", type=Path, help="完整 YAML 配置，默认 src/config/default.yaml")
    parser.add_argument("--calibrate-from-truth", action="store_true", help="独立识别完成后，允许用 .DAT 拟合星等零点")
    parser.add_argument('--catalog', type=Path, help='本地 Gaia 参考 CSV；识别后匹配并给出有来源的近似星等')
    parser.add_argument("--no-plot", action="store_true", help="仅导出 CSV / JSON")
    parser.add_argument("--verbose", action="store_true", help="显示各模块的详细日志")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("src.pipeline").setLevel(logging.INFO)
    logging.getLogger("src.space_pipeline").setLevel(logging.INFO)
    # Known input padding warning: show once, while preserving other FITS warnings.
    astropy_logger = logging.getLogger("astropy")
    astropy_logger.propagate = False
    astropy_logger.addFilter(_PaddingWarningFilter())
    from src.calib.segment import NoTrackingSegment
    from src.register.solver import RegistrationError
    from src.viz.figures import FontUnavailable
    try:
        run_analysis(args)
    except (ValueError, OSError, KeyError, NoTrackingSegment, RegistrationError, FontUnavailable) as exc:
        parser.exit(1, f"分析未完成：{exc}\n请检查输入格式、配置及上方具体错误；天基辅助数据会自动校验，详见 docs/delivery.md。\n")


if __name__ == "__main__":
    main()
