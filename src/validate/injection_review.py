"""Replay a saved injection experiment and audit where known sources are lost.

This is retrospective diagnosis, not new holdout validation or inference. No
production thresholds or matching criteria are changed by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from photutils.segmentation import SourceCatalog, detect_sources
from scipy.spatial import cKDTree

from src.calib.background import model_background
from src.dataio.fits_loader import FrameSequence
from src.space_pipeline import prepare_space_image
from src.target.space_motion import find_space_targets
from src.target.trail_check import check_tracks
from src.validate.benchmark import evaluate_tracks
from src.validate.injection import (
    control_truth, file_hash, inject_target, load_crop, scan_thresholds,
    score_injection, write_json,
)


STAGES = {
    'recovered': '已回收',
    'detection_or_centroid': '检测或质心：3px内原始检出不足4帧',
    'mask_or_edge': '坏像素或边界过滤：有效检出不足4帧',
    'stationary_filter': '恒星排除：非静止候选不足4帧',
    'area_filter': '面积过滤：候选不足4帧',
    'peak_filter': '峰值门限：合格候选不足4帧',
    'association_or_localization': '关联或定位：候选帧足够，但无合格匹配轨迹',
    'trail_check': '拖尾核验：匹配轨迹被否决',
    'preexisting_ambiguity': '未注入对照已匹配真值，回收证据有歧义',
}


def verify_baseline(report, input_directory, root):
    """Refuse an audit silently mixing the saved experiment and changed inputs."""
    if report.get('kind') != 'real-background-injection':
        raise ValueError('Expected a completed real-background injection report')
    for name, digest in report['source_sha256'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != digest:
            raise ValueError(f'Baseline source mismatch: {name}')
    for entry in report['input_files']:
        name = entry['name']
        if Path(name).name != name or file_hash(input_directory/name) != entry['sha256']:
            raise ValueError(f'Baseline FITS mismatch: {name}')


def replay_predictions(predictions, saved):
    if len(predictions) != len(saved):
        raise ValueError('Replay prediction count differs from saved baseline')
    for actual, expected in zip(predictions, saved):
        if (actual.track.frames != expected['frames']
                or not np.allclose(actual.track.xy_det, expected['xy'], atol=1e-7, rtol=0)):
            raise ValueError('Replay prediction coordinates differ from saved baseline')


def stationary_hits(run, config):
    """Read-only reproduction of the saved motion candidate predicate.

    Each sweep verifies total eligible counts against find_space_targets; this
    diagnostic cannot silently diverge from the production implementation.
    """
    sky = {f: run.registration.to_sky(f, run.detections[f].xy) for f in run.frames}
    trees = {f: cKDTree(xy) for f, xy in sky.items()}
    result = {}
    for f in run.frames:
        hits = np.ones(len(sky[f]), dtype=int)
        for g in run.frames:
            if g != f:
                hits += trees[g].query(sky[f])[0] < config['stationary_radius_px']
        result[f] = hits
    return result


def component_evidence(labels, residual, signal, label_id):
    """Measure signal inside an actual segmentation component, not a centroid pixel.

    A centroid may lie in a hole or between disconnected bright pixels. The
    caller supplies the component label matched through SourceCatalog instead.
    """
    component = labels == label_id
    added = float(signal[component].sum())
    positive_residual = float(np.maximum(residual[component], 0).sum())
    return dict(segment_injected_signal_adu=added,
                segment_positive_residual_adu=positive_residual,
                injected_to_positive_residual_ratio=added/positive_residual if positive_residual else None)


def inspect_frames(background, injected, run, control_run, truth_xy, config, radius):
    """Rebuild segmentation and retain local image evidence around the known path."""
    evidence, windows = [], []
    hits = stationary_hits(run, config['space']['motion'])
    for f in run.frames:
        image, _ = prepare_space_image(injected.image(f), config['space'])
        model = model_background(image, **config['background'])
        residual = model.subtract(image)
        threshold = model.threshold(config['detect']['search_n_sigma'])
        segments = detect_sources(residual, threshold, npixels=config['detect']['npixels'])
        raw_xy, raw_labels = np.empty((0, 2)), np.empty(0, dtype=int)
        labels = np.zeros(image.shape, dtype=int) if segments is None else segments.data
        if segments is not None:
            cat = SourceCatalog(residual, segments).to_table(['label', 'xcentroid', 'ycentroid'])
            raw_xy = np.column_stack((cat['xcentroid'], cat['ycentroid'])).astype(float)
            raw_labels = np.asarray(cat['label'], dtype=int)
            finite = np.isfinite(raw_xy).all(axis=1)
            raw_xy, raw_labels = raw_xy[finite], raw_labels[finite]
        table = run.detections[f]
        if len(table):
            distance, indices = cKDTree(raw_xy).query(table.xy)
            if np.any(distance > 1e-7):
                raise ValueError('Rebuilt segmentation does not match pipeline detections')
            table_labels = raw_labels[indices]
        else:
            table_labels = np.empty(0, dtype=int)
        original = background.image(f)
        signal = np.subtract(injected.image(f), original, out=np.zeros_like(image),
                             where=np.isfinite(injected.image(f)) & np.isfinite(original))
        flux = float(signal.sum())
        raw_near = np.linalg.norm(raw_xy-truth_xy[f], axis=1) <= radius
        valid_distances = np.linalg.norm(table.xy-truth_xy[f], axis=1)
        source_rows = []
        # Report all detections: predictions can lie beyond a truth-centred window.
        signal_by_label = np.bincount(labels.ravel(), weights=signal.ravel())
        for k in range(len(table)):
            source_rows.append(dict(index=k, label=int(table_labels[k]), xy=table.xy[k].tolist(),
                                    error_px=float(valid_distances[k]), peak_adu=float(table.peak[k]),
                                    npixels=int(table.npix[k]), stationary_hits=int(hits[f][k]),
                                    segment_signal_fraction=float(signal_by_label[table_labels[k]]/flux)
                                    if flux else None, segment_injected_signal_adu=None,
                                    segment_positive_residual_adu=None, injected_to_positive_residual_ratio=None,
                                    nearest_control_detection_px=None))
        x, y = np.rint(truth_xy[f]).astype(int)
        evidence.append(dict(frame=f, raw_near_count=int(raw_near.sum()),
                             local_noise_rms_adu=float(model.rms[y, x]),
                             local_detection_threshold_adu=float(threshold[y, x]),
                             injected_peak_adu=float(signal.max()),
                             signal_above_threshold_fraction=float(signal[residual > threshold].sum()/flux)
                             if flux else None,
                             signal_in_segments_fraction=float(signal[labels > 0].sum()/flux) if flux else None,
                             sources=source_rows))
        # Fixed display extent only, never used for source selection or inference.
        half = 24
        x0, x1 = max(0, x-half), min(image.shape[1], x+half+1)
        y0, y1 = max(0, y-half), min(image.shape[0], y+half+1)
        region = np.s_[y0:y1, x0:x1]
        windows.append(dict(bounds=[x0, y0, x1, y1], original=original[region].copy(),
                            injected=injected.image(f)[region].copy(), signal=signal[region].copy(),
                            labels=labels[region].copy()))
        # Only retain expensive component evidence for returned predictions later.
        for source in source_rows:
            if source['error_px'] <= half:
                source.update(component_evidence(labels, residual, signal, source['label']))
                distance, _ = cKDTree(control_run.detections[f].xy).query(source['xy'])
                source['nearest_control_detection_px'] = float(distance) if np.isfinite(distance) else None
    return evidence, windows


def stage_counts(frame_evidence, motion, counts, radius):
    totals = Counter()
    per_frame = []
    for frame in frame_evidence:
        sources = frame['sources']
        near = [s for s in sources if s['error_px'] <= radius]
        moving = [s for s in near if s['stationary_hits'] < motion['stationary_min_hits']]
        area = [s for s in moving if s['npixels'] <= motion['max_npixels']]
        eligible = [s for s in area if s['peak_adu'] >= motion['min_peak_adu']]
        actual_count = sum(s['stationary_hits'] < motion['stationary_min_hits']
                           and s['npixels'] <= motion['max_npixels'] and s['peak_adu'] >= motion['min_peak_adu']
                           for s in sources)
        if actual_count != counts[frame['frame']]:
            raise ValueError('Diagnostic candidate predicate differs from inference')
        flags = dict(raw=frame['raw_near_count'] > 0, valid=bool(near),
                     moving=bool(moving), area=bool(area), eligible=bool(eligible))
        totals.update({k: int(v) for k, v in flags.items()})
        per_frame.append(dict(frame=frame['frame'], **flags,
                              eligible_source_indices=[s['index'] for s in eligible]))
    return dict(totals), per_frame


def loss_stage(totals, recovered, before_trail, ambiguous=False):
    """First failed necessary condition; never claims a unique causal explanation."""
    if recovered:
        return 'recovered'
    if ambiguous:
        return 'preexisting_ambiguity'
    for key, stage in [('raw', 'detection_or_centroid'), ('valid', 'mask_or_edge'),
                       ('moving', 'stationary_filter'), ('area', 'area_filter'), ('eligible', 'peak_filter')]:
        if totals[key] < 4:
            return stage
    return 'trail_check' if before_trail else 'association_or_localization'


def unassigned_indices(predictions, truth, controls, frame_count, radius):
    scored = score_injection(predictions, truth, controls, frame_count, radius)
    credited = {m['prediction_index'] for m in scored['matches']}
    remaining = [i for i in range(len(predictions)) if i not in credited]
    matching = evaluate_tracks([predictions[i] for i in remaining], control_truth(controls, frame_count), radius)
    old = {remaining[m['prediction_index']] for m in matching['matches']}
    return [i for i in remaining if i not in old], scored


def audit_track(verdict, truth_xy, frame_evidence, radius):
    observations = []
    direction = truth_xy[-1] - truth_xy[0]
    direction = direction / np.linalg.norm(direction)
    for f, xy in zip(verdict.track.frames, verdict.track.xy_det):
        sources = frame_evidence[f]['sources']
        source = min(sources, key=lambda s: np.linalg.norm(np.array(s['xy'])-xy))
        if np.linalg.norm(np.array(source['xy'])-xy) > 1e-7:
            raise ValueError('Prediction observation cannot be mapped to a detected component')
        offset = xy - truth_xy[f]
        observations.append(dict(frame=f, **source, along_error_px=float(offset @ direction),
                                 cross_error_px=float(offset @ np.array([-direction[1], direction[0]]))))
    matching = sum(o['error_px'] <= radius for o in observations)
    return dict(observed_frames=len(observations), within_radius_count=matching,
                within_radius_fraction=matching/len(observations),
                required_matches=4, required_fraction=.8,
                error_min_px=min(o['error_px'] for o in observations),
                error_max_px=max(o['error_px'] for o in observations),
                trail_consistent_fraction=verdict.trail_evidence['consistent_fraction'],
                observations=observations)


def plot_track(path, title, track, truth_xy, windows, radius):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle
    observations = track['observations']
    fig, axes = plt.subplots(len(observations), 4, figsize=(11, 2.5*len(observations)), squeeze=False)
    for i, row in enumerate(observations):
        f = row['frame']
        window = windows[f]
        x0, y0, x1, y1 = window['bounds']
        extent = (x0-.5, x1-.5, y1-.5, y0-.5)
        vmin, vmax = np.nanpercentile(np.r_[window['original'].ravel(), window['injected'].ravel()], [5, 99.5])
        selected = np.where(window['labels'] == row['label'], 2, (window['labels'] > 0).astype(int))
        for j, (pixels, label) in enumerate([(window['original'], 'Original background'),
                                             (window['injected'], 'Background + injection'),
                                             (window['signal'], 'Added signal only'),
                                             (selected, '4-sigma segments (selected=yellow)')]):
            ax = axes[i, j]
            limits = dict(vmin=vmin, vmax=vmax) if j < 2 else dict(vmin=0, vmax=max(float(pixels.max()), 1e-9))
            ax.imshow(pixels, origin='upper', extent=extent, cmap='gray' if j < 2 else 'viridis', **limits)
            ax.add_patch(Circle(truth_xy[f], radius, fill=False, color='cyan', lw=1))
            ax.plot(*row['xy'], '+', color='orangered', markersize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label if i == 0 else '', fontsize=9)
        axes[i, 0].set_ylabel(f"f{f} | error {row['error_px']:.2f}px\nsegment signal {row['segment_signal_fraction']:.1%}", fontsize=9)
    fig.suptitle(title+'\nCyan: truth 3px circle; orange: measured centroid', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .965))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_review(report_path, input_directory, output, split='holdout'):
    report_path, input_directory, output = map(Path, (report_path, input_directory, output))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Review output must be empty; choose a new directory')
    if split not in ('tuning', 'holdout'):
        raise ValueError('Unknown split')
    baseline = json.loads(report_path.read_text(encoding='utf-8'))
    root = Path(__file__).resolve().parents[2]
    verify_baseline(baseline, input_directory, root)
    sequence = FrameSequence.from_directory(input_directory)
    if [h.path.name for h in sequence.headers] != [f['name'] for f in baseline['input_files']]:
        raise ValueError('FITS sequence differs from the saved ordered input list')
    config, protocol = baseline['effective_config'], baseline['protocol']
    saved_rows = [r for r in baseline['cases'] if r['split'] == split]
    if not saved_rows:
        raise ValueError('No saved cases for this split')
    output.mkdir(parents=True, exist_ok=True)
    (output/'figures').mkdir()
    started = time.perf_counter()
    manifest = dict(kind='retrospective-injection-review', created_utc=datetime.now(timezone.utc).isoformat(),
                    baseline_report_sha256=file_hash(report_path), split=split,
                    baseline_source_files_verified=len(baseline['source_sha256']),
                    input_files_verified=len(baseline['input_files']),
                    reviewer_sha256=file_hash(Path(__file__)), production_config_changed=False,
                    policy='First failed necessary condition at the unchanged 3px / 4 observations / 80% matching rule. '
                           'Nearby detections may be real stars; stage counts alone do not prove causal attribution. '
                           'This previously held-out data is now retrospective diagnostic evidence, not fresh validation.')
    write_json(output/'manifest.json', manifest)
    rows, tracks, flat_frames = [], [], []
    try:
        for crop in [c for c in baseline['crops'] if c['split'] == split]:
            seed = crop['seed']
            print(f'Review: {split} seed={seed} control', flush=True)
            background = load_crop(sequence, crop['bounds'])
            control_run, controls, _ = scan_thresholds(background, config, protocol['peak_thresholds_adu'])
            for threshold in protocol['peak_thresholds_adu']:
                saved = next(c for c in baseline['controls'] if c['split'] == split
                             and c['seed'] == seed and c['threshold_adu'] == threshold)
                replay_predictions(controls[threshold]['predictions'], saved['predictions'])
            cases = [r for r in saved_rows if r['seed'] == seed]
            for flux, speed in sorted({(r['flux_adu'], r['speed_px_s']) for r in cases}):
                print(f'Review: seed={seed} flux={flux:g} speed={speed:g}', flush=True)
                injected, truth, _ = inject_target(background, seed, flux, speed, protocol, config)
                run, sweep, _ = scan_thresholds(injected, config, protocol['peak_thresholds_adu'])
                frame_evidence, windows = inspect_frames(background, injected, run, control_run,
                                                         truth[0]['xy'], config, protocol['match_radius_px'])
                for threshold, result in sweep.items():
                    saved = next(r for r in cases if r['flux_adu'] == flux and r['speed_px_s'] == speed
                                 and r['threshold_adu'] == threshold)
                    if not np.allclose(truth[0]['xy'], saved['truth_xy'], atol=1e-10, rtol=0):
                        raise ValueError('Reconstructed injection geometry differs from saved truth')
                    predictions = result['predictions']
                    replay_predictions(predictions, saved['predictions'])
                    unknown, score = unassigned_indices(predictions, truth, controls[threshold]['predictions'],
                                                        len(sequence), protocol['match_radius_px'])
                    for key in ('recovered_count', 'new_unmatched_count', 'matched_control_count', 'missed_count'):
                        if score[key] != saved[key]:
                            raise ValueError(f'Replayed score differs: {key}')
                    motion = deepcopy(config['space']['motion'])
                    motion['min_peak_adu'] = threshold
                    totals, flags = stage_counts(frame_evidence, motion, result['candidate_counts'], protocol['match_radius_px'])
                    before, _ = find_space_targets(run.detections, run.frames, run.registration, injected.times().unix, motion)
                    pre_match = evaluate_tracks(before, truth, protocol['match_radius_px'])['tp']
                    _, rejected = check_tracks(injected, before, config['space']['trail_check'])
                    stage = loss_stage(totals, score['recovered_count'], pre_match, bool(score['ambiguous_truth_ids']))
                    case_id = dict(split=split, seed=seed, flux_adu=flux, speed_px_s=speed, threshold_adu=threshold)
                    row = dict(**case_id, stage=stage, **{k+'_frames': v for k, v in totals.items()},
                               recovered_count=score['recovered_count'], before_trail_matches=pre_match,
                               before_trail_track_count=len(before), rejected_trail_count=len(rejected),
                               new_unmatched_count=len(unknown))
                    rows.append(row)
                    with (output/'cases.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(dict(**row, frames=frame_evidence, stage_flags=flags,
                                                     rejected_trails=rejected), ensure_ascii=False, allow_nan=False)+'\n')
                    for frame, flag in zip(frame_evidence, flags):
                        flat_frames.append(dict(**case_id, frame=frame['frame'],
                                                **{k: frame[k] for k in ('local_noise_rms_adu', 'local_detection_threshold_adu',
                                                                         'injected_peak_adu', 'signal_above_threshold_fraction',
                                                                         'signal_in_segments_fraction')},
                                                **{k: flag[k] for k in ('raw', 'valid', 'moving', 'area', 'eligible')}))
                    for index in unknown:
                        audit = audit_track(predictions[index], truth[0]['xy'], frame_evidence, protocol['match_radius_px'])
                        name = f's{seed}-f{flux:g}-v{speed:g}-p{threshold:g}-track{index}.png'
                        plot_track(output/'figures'/name, f'Seed {seed} | {flux:g} ADU | {speed:g} px/s | peak {threshold:g}',
                                   audit, truth[0]['xy'], windows, protocol['match_radius_px'])
                        tracks.append(dict(**case_id, prediction_index=index, figure='figures/'+name, **audit))
        summary = dict(**manifest, cases=rows, unassigned_tracks=tracks,
                       stage_counts=dict(Counter(r['stage'] for r in rows)), total_seconds=time.perf_counter()-started)
        write_csv(output/'stages.csv', rows)
        write_csv(output/'frames.csv', flat_frames)
        observation_rows = [dict(seed=t['seed'], flux_adu=t['flux_adu'], speed_px_s=t['speed_px_s'],
                                 threshold_adu=t['threshold_adu'], prediction_index=t['prediction_index'],
                                 **{k: v for k, v in o.items() if k != 'xy'}, x=o['xy'][0], y=o['xy'][1])
                            for t in tracks for o in t['observations']]
        write_csv(output/'track-observations.csv', observation_rows)
        write_json(output/'review.json', summary)
        lines = ['# 注入实验逐阶段复核', '', '这是对既有实验的回溯诊断，未改变生产配置或评估半径。', '',
                 '| 种子 | 流量 | 速度 | 门限 | 原始/有效/非静止/面积/峰值合格帧 | 首个未满足条件 |',
                 '|---|---:|---:|---:|---|---|']
        for r in rows:
            counts = '/'.join(str(r[k+'_frames']) for k in ('raw', 'valid', 'moving', 'area', 'eligible'))
            lines.append(f"| {r['seed']} | {r['flux_adu']:g} | {r['speed_px_s']:g} | {r['threshold_adu']:g} | {counts} | {STAGES[r['stage']]} |")
        lines += ['', '## 未归属轨迹逐帧对照', '']
        for t in tracks:
            lines += [f"### 种子 {t['seed']}，轨迹 {t['prediction_index']}，门限 {t['threshold_adu']:g}", '',
                      f"3px 内 {t['within_radius_count']}/{t['observed_frames']} 帧；拖尾一致比例 {t['trail_consistent_fraction']:.0%}。", '',
                      f"![原图、注入图、注入分量与分割]({t['figure']})", '']
        lines += ['附近检出不必然属于注入源；首个未满足条件是诊断线索，不是唯一原因。',
                  '原留出集现用于回溯分析，不能把本报告当成新的独立验证。', '']
        (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    except Exception as exc:
        write_json(output/'failure.json', dict(error=type(exc).__name__, message=str(exc), completed_rows=len(rows)))
        raise
    print(f'Review complete: {len(rows)} rows, {len(tracks)} unassigned tracks, {summary["total_seconds"]:.1f}s', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('tuning', 'holdout'), default='holdout')
    args = parser.parse_args()
    try:
        run_review(args.report, args.input, args.output, args.split)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Injection review failed: {exc}\n')


if __name__ == '__main__':
    main()
