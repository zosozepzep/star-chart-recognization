"""Paired 80 ADU / 40 ADU / 40 ADU + exposure-trail relocation experiment.

Run development on the original tuning crops first. Validation requires its
completed report and unchanged source/configuration, and uses new disjoint
crops and seeds from the same sequence (not an independent dataset).
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import time

import numpy as np

from src.config import load_config
from src.dataio.fits_loader import FrameSequence
from src.space_pipeline import analyze_space_sequence
from src.target.space_motion import find_space_targets
from src.target.trail_check import check_tracks
from src.target.trail_refinement import refine_and_relink
from src.validate.benchmark import evaluate_tracks
from src.validate.injection import file_hash, inject_target, load_crop, score_injection, write_json
from src.validate.injection_review import replay_predictions, verify_baseline


ROOT = Path(__file__).resolve().parents[2]
FIT_CONFIG = ROOT/'src/config/trail_refinement.yaml'
VARIANTS = ('baseline80', 'threshold40', 'refined40')
VALIDATION_SEEDS = (5101, 6101, 7103)


def overlaps(a, b):
    return a[0] < b[0]+b[2] and b[0] < a[0]+a[2] and a[1] < b[1]+b[3] and b[1] < a[1]+a[3]


def new_crops(shape, size, old_crops, metadata_rows, seeds=VALIDATION_SEEDS):
    """Choose using geometry only; reject every old and newly reserved rectangle."""
    height, width = shape
    if width < size or height-metadata_rows < size:
        raise ValueError('New crop does not fit image')
    if set(seeds) & {c['seed'] for c in old_crops} or len(set(seeds)) != len(seeds):
        raise ValueError('Validation seeds must be new and distinct')
    occupied = [c['bounds'] for c in old_crops]
    result = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for _ in range(10000):
            bounds = [int(rng.integers(0, width-size+1)),
                      int(rng.integers(metadata_rows, height-size+1)), size, size]
            if not any(overlaps(bounds, b) for b in occupied):
                result.append(dict(split='validation', seed=int(seed), bounds=bounds))
                occupied.append(bounds)
                break
        else:
            raise ValueError('Cannot reserve disjoint validation crops')
    return result


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): file_hash(p)
            for p in sorted((ROOT/'src').rglob('*')) if p.suffix in ('.py', '.yaml')}


def compare_methods(sequence, config, fit_config):
    """Inference receives only images/metadata, never injection truth or seed."""
    before = time.perf_counter()
    run = analyze_space_sequence(sequence, config)
    shared_seconds = time.perf_counter()-before
    results = {}
    for variant, threshold in zip(VARIANTS[:2], (80., 40.)):
        effective = deepcopy(config)
        effective['space']['motion']['min_peak_adu'] = threshold
        before = time.perf_counter()
        raw, counts = find_space_targets(run.detections, run.frames, run.registration,
                                         sequence.times().unix, effective['space']['motion'])
        targets, rejected = check_tracks(sequence, raw, effective['space']['trail_check'])
        results[variant] = dict(predictions=targets, seconds=time.perf_counter()-before,
                                diagnostics=dict(candidate_counts=counts, rejected_trails=rejected))
        if threshold == 40.:
            before = time.perf_counter()
            refined, diagnostics = refine_and_relink(sequence, run, raw, effective, fit_config)
            refinement_seconds = time.perf_counter()-before
            results['refined40'] = dict(predictions=refined,
                                        seconds=results['threshold40']['seconds']+refinement_seconds,
                                        refinement_seconds=refinement_seconds, diagnostics=diagnostics)
    return results, shared_seconds


def serialize_predictions(predictions):
    return [dict(frames=v.track.frames, xy=v.track.xy_det.tolist(), evidence=v.to_dict()) for v in predictions]


def extra_metrics(predictions, truth, score, radius):
    """Duplicates remain uncredited; report all-observation errors separately."""
    eligible = [j for j, v in enumerate(predictions) if evaluate_tracks([v], truth, radius)['tp']]
    errors = [float(np.linalg.norm(xy-truth[0]['xy'][f]))
              for m in score['matches'] for f, xy in zip(predictions[m['prediction_index']].track.frames,
                                                        predictions[m['prediction_index']].track.xy_det)]
    return dict(duplicate_eligible_count=max(0, len(eligible)-1),
                matched_track_squared_error_sum=float(np.square(errors).sum()),
                matched_track_observations=len(errors))


def aggregate(rows, *, by_bin=False):
    groups = {}
    for row in rows:
        key = (row['variant'], row['flux_adu'], row['speed_px_s']) if by_bin else (row['variant'],)
        groups.setdefault(key, []).append(row)
    totals = []
    for key, group in sorted(groups.items()):
        n = sum(r['matched_track_observations'] for r in group)
        fits = [f for r in group for f in r['diagnostics'].get('fits', [])]
        totals.append(dict(variant=key[0], **(dict(flux_adu=key[1], speed_px_s=key[2]) if by_bin else {}),
                           cases=len(group), recovered_count=sum(r['recovered_count'] for r in group),
                           new_unmatched_count=sum(r['new_unmatched_count'] for r in group),
                           duplicate_eligible_count=sum(r['duplicate_eligible_count'] for r in group),
                           all_observation_rmse_px=np.sqrt(sum(r['matched_track_squared_error_sum'] for r in group)/n) if n else None,
                           matched_observations=n,
                           mean_matched_coverage=float(np.mean([r['matched_visible_coverage'] for r in group
                                                                if r['matched_visible_coverage'] is not None])) if n else None,
                           mean_method_seconds=float(np.mean([r['method_seconds'] for r in group])),
                           fit_attempts=len(fits), fit_accepted=sum(f['accepted'] for f in fits),
                           fit_reasons=dict(Counter(f['reason'] for f in fits))))
    return totals


def save_summary(output, report):
    bins = report['bins']
    with (output/'recovery.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = [k for k in bins[0] if k != 'fit_reasons']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(bins)
    lines = ['# 曝光拖尾重定位实验', '', f"阶段：{report['phase']}。生产算法及门限未修改。", '',
             '| 方法 | 回收/注入 | 新增未归属 | 重复可匹配 | 已匹配轨迹全部观测 RMSE px |',
             '|---|---:|---:|---:|---:|']
    for r in report['totals']:
        rmse = f"{r['all_observation_rmse_px']:.3f}" if r['all_observation_rmse_px'] is not None else '无匹配'
        lines.append(f"| {r['variant']} | {r['recovered_count']}/{r['cases']} | {r['new_unmatched_count']} | {r['duplicate_eligible_count']} | {rmse} |")
    lines += ['', '## 边界', ''] + ['- '+s for s in report['limitations']]
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    speeds = report['protocol']['speeds_px_s']
    fig, axes = plt.subplots(1, len(speeds), figsize=(4*len(speeds), 3.7), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, speed in zip(axes, report['protocol']['speeds_px_s']):
        for variant in VARIANTS:
            selected = sorted([r for r in bins if r['variant'] == variant and r['speed_px_s'] == speed], key=lambda r: r['flux_adu'])
            ax.plot([r['flux_adu'] for r in selected], [r['recovered_count']/r['cases'] for r in selected], marker='o', label=variant)
        ax.set(title=f'{speed:g} px/s', xlabel='Injected flux / exposure (ADU)', xscale='log', ylim=(-.05, 1.05))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Injection recovery fraction')
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"Trail relocation: {report['phase']} (same sequence)")
    fig.tight_layout()
    fig.savefig(output/'recovery.png', dpi=150)
    plt.close(fig)


def run_experiment(input_directory, output, baseline_path, *, phase='development', development_path=None):
    input_directory, output, baseline_path = Path(input_directory), Path(output), Path(baseline_path)
    if phase not in ('development', 'validation'):
        raise ValueError('Unknown phase')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Experiment output must be empty')
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    verify_baseline(baseline, input_directory, ROOT)
    config, fit_config = deepcopy(baseline['effective_config']), load_config(FIT_CONFIG)
    if config['space']['motion']['min_peak_adu'] != 80:
        raise ValueError('This paired experiment requires the 80 ADU baseline')
    protocol, sources = deepcopy(baseline['protocol']), source_hashes()
    validation_crops = new_crops(baseline['image_shape'], protocol['crop_size'], baseline['crops'],
                                 baseline['production_config']['space']['metadata_rows'])
    development_hash = None
    if phase == 'validation':
        if development_path is None:
            raise ValueError('Validation requires a completed development report')
        development_path = Path(development_path)
        development = json.loads(development_path.read_text(encoding='utf-8'))
        if (development.get('phase') != 'development' or development.get('kind') != 'trail-refinement-experiment'
                or development.get('source_sha256') != sources
                or development.get('baseline_sha256') != file_hash(baseline_path)
                or development.get('validation_crops') != validation_crops
                or len(development.get('cases', [])) != len(protocol['tuning_seeds'])*len(protocol['fluxes_adu'])*len(protocol['speeds_px_s'])*len(VARIANTS)):
            raise ValueError('Development incomplete or frozen source/protocol changed')
        development_hash = file_hash(development_path)
    crops = ([{**c, 'split': 'development'} for c in baseline['crops'] if c['split'] == 'tuning']
             if phase == 'development' else validation_crops)
    sequence = FrameSequence.from_directory(input_directory)
    if [h.path.name for h in sequence.headers] != [h['name'] for h in baseline['input_files']]:
        raise ValueError('Input frame list differs from baseline')
    started = time.perf_counter()
    manifest = dict(schema_version=1, kind='trail-refinement-experiment', phase=phase,
                    created_utc=datetime.now(timezone.utc).isoformat(), baseline_sha256=file_hash(baseline_path),
                    development_sha256=development_hash, source_sha256=sources,
                    effective_config=config, fit_config=fit_config, protocol=protocol, variants=list(VARIANTS),
                    crops=crops, validation_crops=validation_crops, input_files=baseline['input_files'],
                    environment=baseline['environment'],
                    limitations=[
                        '新验证区域与旧调参/留出及彼此均无像素重叠，使用新种子；仍是同一序列，不能称独立天区或设备验证。',
                        '三组共享检测与配准；重定位只拟合已成轨迹的观测，不补缺帧，不改善未检出弱源。',
                        '模型从检测轨迹估计探测器速度，拟合中心、流量、PSF宽度及背景平面；不接收注入真值或未注入背景。',
                        '拟合采用匀速曝光与圆高斯假设，与注入形状同族；没有额外光子噪声，真实复杂PSF仍待验证。',
                        '沿用至少4个时刻、80%观测在3px内的一对一匹配；重复可匹配轨迹另报且仍计入未归属。',
                        '无完整真实目标标注，新增未归属数不是FP或Precision；回收率不是官方目标召回率。',
                        '汇总RMSE包含成功匹配轨迹的全部观测，另保留原匹配半径内RMSE；不同方法成功样本集合可能不同。',
                        '共享耗时包含生产基线关联；方法耗时不含检测配准，refined40含40ADU初始关联/核验、拟合和重新关联。',
                        '拟合失败保留原检测；保留原峰值/面积/孔径测光，拟合流量仅作诊断，不用于星等。'])
    # Record actual runtime versions, not the earlier run's environment.
    import platform, sys
    from importlib.metadata import version
    manifest['environment'] = dict(python=sys.version, platform=platform.platform(),
                                    packages={p: version(p) for p in ('numpy', 'scipy', 'astropy', 'photutils', 'matplotlib')})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'protocol.json', manifest)
    rows, controls_saved = [], []
    try:
        for crop in crops:
            seed = crop['seed']
            print(f"Trail experiment: {phase} seed={seed} control", flush=True)
            background = load_crop(sequence, crop['bounds'])
            controls, shared = compare_methods(background, config, fit_config)
            for variant in VARIANTS:
                controls_saved.append(dict(**crop, variant=variant, shared_pipeline_seconds=shared,
                                            method_seconds=controls[variant]['seconds'],
                                            diagnostics=controls[variant]['diagnostics'],
                                            predictions=serialize_predictions(controls[variant]['predictions'])))
                if phase == 'development' and variant != 'refined40':
                    old = next(c for c in baseline['controls'] if c['seed'] == seed and c['threshold_adu'] == (80 if variant == 'baseline80' else 40))
                    replay_predictions(controls[variant]['predictions'], old['predictions'])
            write_json(output/'controls.json', controls_saved)
            for flux in protocol['fluxes_adu']:
                for speed in protocol['speeds_px_s']:
                    print(f'Trail experiment: {phase} seed={seed} flux={flux:g} speed={speed:g}', flush=True)
                    injected, truth, audit = inject_target(background, seed, flux, speed, protocol, config)
                    sweep, shared = compare_methods(injected, config, fit_config)
                    for variant in VARIANTS:
                        result = sweep[variant]
                        predictions = result['predictions']
                        score = score_injection(predictions, truth, controls[variant]['predictions'], len(sequence), protocol['match_radius_px'])
                        if phase == 'development' and variant != 'refined40':
                            old = next(c for c in baseline['cases'] if c['seed'] == seed and c['flux_adu'] == flux and c['speed_px_s'] == speed
                                       and c['threshold_adu'] == (80 if variant == 'baseline80' else 40))
                            replay_predictions(predictions, old['predictions'])
                            if any(score[k] != old[k] for k in ('recovered_count', 'new_unmatched_count')):
                                raise ValueError('Original baseline scoring changed')
                        row = dict(**crop, variant=variant, flux_adu=flux, speed_px_s=speed,
                                   shared_pipeline_seconds=shared, method_seconds=result['seconds'],
                                   refinement_seconds=result.get('refinement_seconds', 0.), **score,
                                   **extra_metrics(predictions, truth, score, protocol['match_radius_px']),
                                   truth_xy=truth[0]['xy'].tolist(), injection_audit=audit,
                                   predictions=serialize_predictions(predictions), diagnostics=result['diagnostics'])
                        rows.append(row)
                        with (output/'cases.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        if source_hashes() != sources:
            raise ValueError('Source changed during experiment')
        verify_baseline(baseline, input_directory, ROOT)
        report = dict(**manifest, cases=rows, controls=controls_saved, totals=aggregate(rows), bins=aggregate(rows, by_bin=True))
        save_summary(output, report)
        report['total_seconds'] = time.perf_counter()-started
        write_json(output/'experiment.json', report)
        write_json(output/'receipt.json', dict(completed_rows=len(rows), source_and_input_verified=True,
                                               files={p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}))
    except Exception as exc:
        write_json(output/'failure.json', dict(error=type(exc).__name__, message=str(exc), completed_rows=len(rows)))
        raise
    print(json.dumps(report['totals'], indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--phase', choices=['development', 'validation'], default='development')
    parser.add_argument('--development-report', type=Path)
    args = parser.parse_args()
    run_experiment(args.input, args.output, args.baseline, phase=args.phase, development_path=args.development_report)


if __name__ == '__main__':
    main()
