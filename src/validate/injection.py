"""Paired target injection into real FITS crops, with a frozen holdout protocol.

Run: python -m src.validate.injection --input data/rst19 --output output/injection
Only this evaluator receives injection truth. Production settings are unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from astropy.io import fits
from astropy.time import Time

from src.config import load_config
from src.dataio.fits_loader import FrameSequence
from src.space_pipeline import analyze_space_sequence
from src.target.space_motion import find_space_targets
from src.target.trail_check import check_tracks
from src.validate.benchmark import evaluate_tracks


DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / 'config' / 'injection.yaml'


class ImageSequence:
    """Inference interface containing pixels and acquisition metadata, no truth."""

    def __init__(self, headers, images):
        self.headers, self.images = headers, images

    def __len__(self):
        return len(self.headers)

    def image(self, frame):
        return self.images[frame]

    def times(self):
        return Time([h.date_obs for h in self.headers])


def validate_protocol(protocol, config):
    """Fail before creating artifacts or opening the holdout data."""
    for key in ('tuning_seeds', 'holdout_seeds'):
        values = protocol[key]
        if (not values or any(type(v) is not int or v < 0 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f'{key}: require distinct nonnegative integer seeds')
    if set(protocol['tuning_seeds']) & set(protocol['holdout_seeds']):
        raise ValueError('Tuning and holdout seeds must be disjoint')
    for key in ('fluxes_adu', 'speeds_px_s', 'peak_thresholds_adu'):
        values = np.asarray(protocol[key], dtype=float)
        if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
                or np.any(values <= 0) or len(set(values)) != len(values)):
            raise ValueError(f'{key}: require distinct finite positive values')
    for key in ('psf_sigma_px', 'psf_support_sigma', 'trail_samples_per_px',
                'edge_margin_px', 'match_radius_px'):
        if not np.isfinite(protocol[key]) or protocol[key] <= 0:
            raise ValueError(f'{key}: require a finite positive value')
    if type(protocol['crop_size']) is not int or protocol['crop_size'] < 128:
        raise ValueError('crop_size must be an integer >= 128')
    baseline = config['space']['motion']['min_peak_adu']
    if baseline not in protocol['peak_thresholds_adu']:
        raise ValueError('The sweep must include the production peak threshold')
    motion = config['space']['motion']
    if not all(motion['min_speed_px_s'] <= s <= motion['max_speed_px_s']
               for s in protocol['speeds_px_s']):
        raise ValueError('Injected speeds must be inside the declared motion range')


def crop_bounds(shape, size, split, seed, metadata_rows):
    """Tuning uses the left half; holdout uses the right, without pixel overlap."""
    height, width = shape
    if split not in ('tuning', 'holdout'):
        raise ValueError('Unknown split')
    left, right = (0, width // 2) if split == 'tuning' else (width // 2, width)
    if size > right - left or size > height - metadata_rows:
        raise ValueError('Crop does not fit the split after excluding metadata rows')
    rng = np.random.default_rng(seed)
    x = int(rng.integers(left, right - size + 1))
    y = int(rng.integers(metadata_rows, height - size + 1))
    return x, y, size, size


def load_crop(sequence, bounds):
    x, y, width, height = bounds
    images = []
    for header in sequence.headers:
        with fits.open(header.path) as hdul:
            hdul.verify('silentfix')
            if (len(hdul[0].shape) != 2 or y + height > hdul[0].shape[0]
                    or x + width > hdul[0].shape[1]):
                raise ValueError(f'Crop exceeds FITS image: {header.path.name}')
            # Section access retains signed/scaled FITS semantics, without loading 4096².
            images.append(np.array(hdul[0].section[y:y+height, x:x+width], dtype=float))
    return ImageSequence(sequence.headers, images)


def target_geometry(sequence, seed, speed, protocol):
    stamps = sequence.times().unix
    elapsed = stamps - (stamps[0] + stamps[-1]) / 2
    exposures = np.array([h.exposure_s for h in sequence.headers])
    rng = np.random.default_rng(seed + 100000)
    angle = rng.uniform(0, 2 * np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    # Same midpoint/direction for every brightness and speed at a given seed.
    max_speed = max(protocol['speeds_px_s'])
    extent = (np.max(np.abs(elapsed)) + exposures.max()/2) * max_speed
    margin = (np.abs(direction) * extent + protocol['edge_margin_px']
              + protocol['psf_support_sigma'] * protocol['psf_sigma_px'])
    size_xy = np.array(sequence.image(0).shape[::-1]) - 1
    if np.any(2 * margin >= size_xy):
        raise ValueError('Crop too small for the complete injected track and PSF')
    center = rng.uniform(margin, size_xy - margin)
    velocity = direction * speed
    return center + elapsed[:, None] * velocity, velocity


def trail_patch(shape, position, velocity, exposure, flux, protocol):
    """Discrete flux-normalized Gaussian integrated along a linear exposure."""
    position, velocity = np.asarray(position), np.asarray(velocity)
    half = (np.abs(velocity) * exposure / 2
            + protocol['psf_support_sigma'] * protocol['psf_sigma_px'])
    lower = np.floor(position - half).astype(int)
    upper = np.ceil(position + half).astype(int) + 1
    if np.any(lower < 0) or np.any(upper > np.array(shape[::-1])):
        raise ValueError('Injection support must fit entirely inside the image')
    x0, y0 = lower
    x1, y1 = upper
    yy, xx = np.mgrid[y0:y1, x0:x1]
    length = np.linalg.norm(velocity) * exposure
    count = max(1, int(np.ceil(length * protocol['trail_samples_per_px'])))
    patch = np.zeros_like(xx, dtype=float)
    # Midpoint quadrature avoids over-weighting the endpoints of short trails.
    for offset in (np.arange(count) + .5) / count - .5:
        cx, cy = position + offset * exposure * velocity
        patch += np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*protocol['psf_sigma_px']**2))
    patch *= flux / patch.sum()
    return (slice(y0, y1), slice(x0, x1)), patch


def inject_target(background, seed, flux, speed, protocol, config):
    xy, velocity = target_geometry(background, seed, speed, protocol)
    images, audit = [], []
    for f, header in enumerate(background.headers):
        original = background.image(f)
        image = original.copy()
        region, patch = trail_patch(image.shape, xy[f], velocity, header.exposure_s, flux, protocol)
        pixels = original[region]
        valid = np.isfinite(pixels) & (pixels >= 0) & (pixels < config['space']['saturation_adu'])
        addition = np.where(valid, patch, 0.)
        image[region] += addition
        images.append(image)
        audit.append(dict(frame=f, nominal_flux_adu=float(flux), added_flux_adu=float(addition.sum()),
                          injected_peak_adu=float(patch.max()),
                          signal_masked_fraction=float(1 - addition.sum()/flux)))
    truth = [dict(id=1, xy=xy, visible_frames=list(range(len(background))))]
    return ImageSequence(background.headers, images), truth, audit


def control_truth(verdicts, frame_count):
    truth = []
    for i, verdict in enumerate(verdicts):
        points = np.full((frame_count, 2), np.nan)
        points[verdict.track.frames] = verdict.track.xy_det
        truth.append(dict(id=i, xy=points, visible_frames=verdict.track.frames))
    return truth


def score_injection(predictions, truth, controls, frame_count, radius):
    """Attribute known injections; unmatched real candidates have no FP label."""
    preexisting = evaluate_tracks(controls, truth, radius)
    ambiguous_ids = {m['truth_id'] for m in preexisting['matches']}
    result = evaluate_tracks(predictions, truth, radius)
    matches = [m for m in result['matches'] if m['truth_id'] not in ambiguous_ids]
    recovered = {m['prediction_index'] for m in matches}
    remaining = [v for j, v in enumerate(predictions) if j not in recovered]
    background_matches = evaluate_tracks(remaining, control_truth(controls, frame_count), radius)
    # Recalculate localization/coverage on credited matches only (pre-existing
    # candidates are ambiguous evidence, and must not raise the recovery count).
    credited_ids = {m['truth_id'] for m in matches}
    credited = evaluate_tracks([predictions[m['prediction_index']] for m in matches],
                               [t for t in truth if t['id'] in credited_ids], radius)
    return dict(truth_count=len(truth), recovered_count=len(matches),
                missed_count=len(truth)-len(matches), ambiguous_truth_ids=sorted(ambiguous_ids),
                predicted_count=len(predictions), control_candidate_count=len(controls),
                unmatched_prediction_count=len(remaining),
                matched_control_count=background_matches['tp'],
                new_unmatched_count=background_matches['fp'],
                localization_rmse_px=credited['localization_rmse_px'],
                matched_visible_coverage=credited['matched_visible_coverage'], matches=matches)


def scan_thresholds(sequence, config, thresholds):
    """Rerun upstream per image case; share it only within a peak-only sweep."""
    before = time.perf_counter()
    run = analyze_space_sequence(sequence, config)
    shared_seconds = time.perf_counter() - before
    results = {}
    for threshold in thresholds:
        motion = deepcopy(config['space']['motion'])
        motion['min_peak_adu'] = threshold
        before = time.perf_counter()
        predictions, counts = find_space_targets(run.detections, run.frames, run.registration,
                                                  sequence.times().unix, motion)
        predictions, rejected = check_tracks(sequence, predictions, config['space']['trail_check'])
        results[threshold] = dict(predictions=predictions,
                                  association_and_check_seconds=time.perf_counter()-before,
                                  candidate_counts=counts, rejected_count=len(rejected))
    # analyze_space_sequence includes its original motion pass. The shared time
    # is reported explicitly and is not advertised as a per-threshold full run.
    return run, results, shared_seconds


def aggregate_cases(rows):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in ('split', 'threshold_adu', 'flux_adu', 'speed_px_s'))
        grouped.setdefault(key, []).append(row)
    result = []
    for (split, threshold, flux, speed), group in sorted(grouped.items()):
        total = sum(r['truth_count'] for r in group)
        recovered = sum(r['recovered_count'] for r in group)
        result.append(dict(split=split, threshold_adu=threshold, flux_adu=flux, speed_px_s=speed,
                           cases=len(group), truth_count=total, recovered_count=recovered,
                           recovery_rate=recovered/total,
                           new_unmatched_count=sum(r['new_unmatched_count'] for r in group),
                           ambiguous_truth_count=sum(len(r['ambiguous_truth_ids']) for r in group),
                           mean_association_and_check_seconds=float(np.mean(
                               [r['association_and_check_seconds'] for r in group]))))
    return result


def select_threshold(rows, baseline):
    """Use tuning rows only; tie-break by unexplained count, then baseline proximity."""
    tuning = [r for r in rows if r['split'] == 'tuning']
    if not tuning:
        raise ValueError('No tuning evidence')
    candidates = []
    for threshold in sorted({r['threshold_adu'] for r in tuning}):
        group = [r for r in tuning if r['threshold_adu'] == threshold]
        candidates.append(dict(threshold_adu=threshold,
                               recovered_count=sum(r['recovered_count'] for r in group),
                               new_unmatched_count=sum(r['new_unmatched_count'] for r in group)))
    chosen = min(candidates, key=lambda r: (-r['recovered_count'], r['new_unmatched_count'],
                                           abs(r['threshold_adu']-baseline), -r['threshold_adu']))
    return dict(selected_threshold_adu=chosen['threshold_adu'], baseline_threshold_adu=baseline,
                policy='Tuning only: most recovered, fewest new unmatched, closest to production, higher threshold.',
                frozen_utc=datetime.now(timezone.utc).isoformat(), tuning_totals=candidates,
                production_config_changed=False)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_summary(output, report):
    bins = report['bins']
    with (output/'recovery.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bins[0]))
        writer.writeheader()
        writer.writerows(bins)
    selection = report['selection']
    lines = ['# 真实背景注入回收实验', '',
             f"生产基线：{selection['baseline_threshold_adu']:g} ADU；调参集选择：{selection['selected_threshold_adu']:g} ADU。生产配置未修改。", '',
             '流量是每次曝光注入的总 ADU，不是星等。每格为回收数/注入数。', '',
             '| 分组 | 峰值门限 ADU | 注入流量 ADU | 速度 px/s | 回收 | 新增未归属轨迹 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in bins:
        lines.append(f"| {row['split']} | {row['threshold_adu']:g} | {row['flux_adu']:g} | "
                     f"{row['speed_px_s']:g} | {row['recovered_count']}/{row['truth_count']} | {row['new_unmatched_count']} |")
    lines += ['', '## 口径与限制', ''] + [f'- {s}' for s in report['limitations']]
    lines += ['', '详细协议、输入与代码 SHA-256、实际注入信号、逐条预测和耗时见 JSON；中间结果见 cases.jsonl。', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    speeds = report['protocol']['speeds_px_s']
    fig, axes = plt.subplots(2, len(speeds), figsize=(4*len(speeds), 7), squeeze=False)
    for i, split in enumerate(('tuning', 'holdout')):
        for j, speed in enumerate(speeds):
            ax = axes[i, j]
            for threshold in report['protocol']['peak_thresholds_adu']:
                rows = [r for r in bins if r['split'] == split and r['speed_px_s'] == speed
                        and r['threshold_adu'] == threshold]
                ax.plot([r['flux_adu'] for r in rows], [r['recovery_rate'] for r in rows],
                        marker='o', label=f'Peak >= {threshold:g} ADU')
            ax.set(xscale='log', ylim=(-.05, 1.05), xlabel='Injected total signal / exposure (ADU)',
                   ylabel='Recovered / injected', title=f'{split} | {speed:g} px/s')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Real-background injection recovery | shared field, limited seeds')
    fig.tight_layout()
    fig.savefig(output/'recovery.png', dpi=150)
    plt.close(fig)


def run_injection(input_directory, output, *, protocol=None, config=None):
    protocol = deepcopy(protocol if protocol is not None else load_config(DEFAULT_PROTOCOL))
    config = deepcopy(config if config is not None else load_config())
    validate_protocol(protocol, config)
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Injection output directory must be empty; choose a new directory')
    sequence = FrameSequence.from_directory(input_directory)
    if sequence.observation_mode != 'space':
        raise ValueError('Real-background benchmark requires validated space FITS input')
    if len(sequence) < max(config['space']['motion']['min_track_frames'], config['star_report']['min_hits']):
        raise ValueError('Insufficient frames for injection evaluation')
    with fits.open(sequence.headers[0].path) as hdul:
        hdul.verify('silentfix')
        shape = hdul[0].shape
    if len(shape) != 2:
        raise ValueError('Expected 2-D FITS images')
    crops = [dict(split=split, seed=seed,
                  bounds=list(crop_bounds(shape, protocol['crop_size'], split, seed,
                                          config['space']['metadata_rows'])))
             for split in ('tuning', 'holdout') for seed in protocol[f'{split}_seeds']]
    root = Path(__file__).resolve().parents[2]
    started = time.perf_counter()
    # Crops exclude telemetry. Do not mask their first science row as metadata.
    effective = deepcopy(config)
    effective['space']['metadata_rows'] = 0
    manifest = dict(schema_version=1, kind='real-background-injection',
                    created_utc=datetime.now(timezone.utc).isoformat(), protocol=protocol,
                    production_config=config, effective_config=effective, crops=crops,
                    image_shape=list(shape), frame_count=len(sequence),
                    input_files=[dict(name=h.path.name, bytes=h.path.stat().st_size, sha256=file_hash(h.path),
                                      date_obs_utc=h.date_obs.isot, exposure_s=h.exposure_s)
                                 for h in sequence.headers],
                    source_sha256={p.relative_to(root).as_posix(): file_hash(p)
                                   for p in sorted((root/'src').rglob('*')) if p.suffix in ('.py', '.yaml')},
                    environment=dict(python=sys.version, platform=platform.platform(),
                                     packages={p: version(p) for p in ('numpy', 'scipy', 'astropy', 'photutils', 'matplotlib')}),
                    matching_policy='One-to-one; >=4 matches and >=80% of predicted observations within radius. '
                                    'Pre-existing control matches never receive injection recovery credit.',
                    selection_policy='Freeze tuning-only selection before evaluating any holdout crop.',
                    limitations=[
                        '同一官方序列的左右半幅分别用于调参与留出，种子不重合；仍非独立天区或独立设备验证。',
                        '每例只注入一条目标；使用真实背景与原有噪声，圆高斯 PSF 和匀速曝光拖尾；未模拟额外光子散粒噪声（增益未知）。',
                        '流量为曝光内积分 ADU，非 Gaia 星等；坏像素原值保留，被屏蔽的注入信号逐帧记录。',
                        '位置按几何范围随机生成，不避让恒星；全帧可见的注入均进入分母，遮挡或融合导致的漏检也保留。',
                        '没有完整真实目标真值；未注入对照中的轨迹数量与新增未归属轨迹仅作诊断，不能称为真实误报率或 Precision。',
                        '不同门限共享本例的检测与配准，仅扫描运动候选峰值门限；4σ 单帧检出门限等生产参数保持固定。',
                        'shared_pipeline_seconds 包含原流水线的基线关联；每个门限另记关联/核验耗时，总时间包含全部重复计算和报告导出。',
                        '同一背景和轨迹在多个亮度/速度格中复用，样本相关且数量有限；不等于官方目标召回率。',
                        '调参选择仅为候选建议；无人工核查与独立天区证据时不自动降低生产门限。'])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'protocol.json', manifest)
    rows, control_rows = [], []
    selection = None
    try:
        for crop in crops:
            split, seed = crop['split'], crop['seed']
            if split == 'holdout' and selection is None:
                selection = select_threshold(rows, config['space']['motion']['min_peak_adu'])
                write_json(output/'selection.json', selection)
            print(f"Injection: {split} seed={seed} crop={crop['bounds']} control", flush=True)
            before = time.perf_counter()
            background = load_crop(sequence, crop['bounds'])
            loading_seconds = time.perf_counter() - before
            _, controls, shared_seconds = scan_thresholds(background, effective, protocol['peak_thresholds_adu'])
            for threshold, result in controls.items():
                control_rows.append(dict(**crop, threshold_adu=threshold,
                                         candidate_count=len(result['predictions']),
                                         loading_seconds=loading_seconds, shared_pipeline_seconds=shared_seconds,
                                         association_and_check_seconds=result['association_and_check_seconds'],
                                         predictions=[dict(frames=v.track.frames, xy=v.track.xy_det.tolist())
                                                      for v in result['predictions']]))
            write_json(output/'controls.json', control_rows)
            for flux in protocol['fluxes_adu']:
                for speed in protocol['speeds_px_s']:
                    print(f'Injection: {split} seed={seed} flux={flux:g} speed={speed:g}', flush=True)
                    before = time.perf_counter()
                    injected, truth, audit = inject_target(background, seed, flux, speed, protocol, effective)
                    render_seconds = time.perf_counter() - before
                    run, sweep, shared_seconds = scan_thresholds(injected, effective, protocol['peak_thresholds_adu'])
                    for threshold, result in sweep.items():
                        row = dict(**crop, flux_adu=flux, speed_px_s=speed, threshold_adu=threshold,
                                   render_seconds=render_seconds, shared_pipeline_seconds=shared_seconds,
                                   association_and_check_seconds=result['association_and_check_seconds'],
                                   **score_injection(result['predictions'], truth, controls[threshold]['predictions'],
                                                     len(sequence), protocol['match_radius_px']),
                                   truth_xy=truth[0]['xy'].tolist(), injection_audit=audit,
                                   frame_diagnostics=run.diagnostics['per_frame'],
                                   motion_candidate_counts=result['candidate_counts'],
                                   rejected_trail_count=result['rejected_count'],
                                   predictions=[dict(frames=v.track.frames, xy=v.track.xy_det.tolist(),
                                                     evidence=v.to_dict()) for v in result['predictions']])
                        rows.append(row)
                        with (output/'cases.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            # Release a full crop's pixel stacks before reading the next seed.
            del background, injected
        report = dict(**manifest, selection=selection, controls=control_rows, cases=rows,
                      bins=aggregate_cases(rows))
        save_summary(output, report)
        report['total_seconds'] = time.perf_counter() - started
        write_json(output/'injection.json', report)
    except Exception as exc:
        write_json(output/'failure.json', dict(error=type(exc).__name__, message=str(exc),
                                              completed_rows=len(rows), elapsed_seconds=time.perf_counter()-started))
        raise
    print(f"Completed {len(rows)} rows in {report['total_seconds']:.1f}s: {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    try:
        run_injection(args.input, args.output, protocol=load_config(args.protocol))
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Injection benchmark failed: {exc}\n')


if __name__ == '__main__':
    main()
