"""Replay the remaining refinement cases and save original-pixel evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from src.dataio.fits_loader import FrameSequence
from src.space_pipeline import analyze_space_sequence
from src.target.dual_frame import Track
from src.target.space_motion import MotionVerdict, find_space_targets
from src.target.trail_refinement import refine_and_relink
from src.validate.benchmark import evaluate_tracks
from src.validate.injection import file_hash, inject_target, load_crop, write_json
from src.validate.injection_review import inspect_frames, replay_predictions, unassigned_indices


def restore(saved, frame_times):
    result = []
    for p in saved:
        fs, xy = p['frames'], np.array(p['xy'])
        ones = np.ones(len(fs))
        track = Track(fs, xy, xy.copy(), ones, ones, ones)
        result.append(MotionVerdict(track, np.zeros(2), 0., frame_times[fs[-1]]-frame_times[fs[0]]))
    return result


def plot(path, title, records, windows, truth_xy):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle
    fig, axes = plt.subplots(len(records), 3, figsize=(9, 2.1*len(records)), squeeze=False)
    for i, row in enumerate(records):
        f, w = row['frame'], windows[row['frame']]
        x0, y0, x1, y1 = w['bounds']
        limits = np.nanpercentile(np.r_[w['original'].ravel(), w['injected'].ravel()], [5, 99.5])
        segments = np.where(w['labels'] == row['label'], 2, (w['labels'] > 0).astype(int))
        for j, pixels in enumerate((w['original'], w['injected'], segments)):
            ax = axes[i, j]
            ax.imshow(pixels, origin='upper', extent=(x0-.5, x1-.5, y1-.5, y0-.5),
                       cmap='gray' if j < 2 else 'viridis',
                       vmin=limits[0] if j < 2 else 0, vmax=limits[1] if j < 2 else 2)
            ax.add_patch(Circle(truth_xy[f], 3., fill=False, color='cyan'))
            ax.plot(*row['original_xy'], '+', color='orange', ms=9)
            ax.plot(*row['final_xy'], 'x', color='lime', ms=7)
            if row['fit'].get('fitted_xy') is not None and not row['fit']['accepted']:
                ax.plot(*row['fit']['fitted_xy'], '.', color='magenta', ms=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(('Original background', 'Background + injection', 'Original detected segment')[j], fontsize=9)
        axes[i, 0].set_ylabel(f"f{f}: {row['fit']['reason']}\nerror {row['final_error_px']:.2f}px\nsignal share {row['segment_signal_fraction']:.1%}", fontsize=8)
    fig.suptitle(title+'\nCyan: truth 3px; orange: original; green: final; magenta: rejected fit', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def review(development, validation, input_directory, output):
    root = Path(__file__).resolve().parents[1]
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Review output must be empty')
    reports = {name: json.loads(Path(p).read_text(encoding='utf-8'))
               for name, p in [('development', development), ('validation', validation)]}
    for report in reports.values():
        for name, checksum in report['source_sha256'].items():
            if file_hash(root/name) != checksum:
                raise ValueError(f'Source changed: {name}')
        for entry in report['input_files']:
            if file_hash(Path(input_directory)/entry['name']) != entry['sha256']:
                raise ValueError(f"Input changed: {entry['name']}")
    selected = [('validation', 5101, 1800., 12.), ('validation', 6101, 1800., 12.)]
    selected += [('development', 809, flux, 2.) for flux in (600., 1800., 5400.)]
    started = time.perf_counter()
    sequence = FrameSequence.from_directory(input_directory)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'manifest.json', dict(selected=selected, tool_sha256=file_hash(Path(__file__)),
                                            development_sha256=file_hash(development), validation_sha256=file_hash(validation)))
    rows = []
    try:
        for phase, seed, flux, speed in selected:
            report = reports[phase]
            saved = next(r for r in report['cases'] if (r['seed'], r['flux_adu'], r['speed_px_s'], r['variant'])
                         == (seed, flux, speed, 'refined40'))
            config = deepcopy(report['effective_config'])
            print(f'Review: {phase} {seed} flux={flux:g} speed={speed:g}', flush=True)
            background = load_crop(sequence, saved['bounds'])
            injected, truth, _ = inject_target(background, seed, flux, speed, report['protocol'], config)
            control_run = analyze_space_sequence(background, config)
            run = analyze_space_sequence(injected, config)
            config['space']['motion']['min_peak_adu'] = 40.
            initial, _ = find_space_targets(run.detections, run.frames, run.registration, injected.times().unix, config['space']['motion'])
            predictions, diagnostics = refine_and_relink(injected, run, initial, config, report['fit_config'])
            replay_predictions(predictions, saved['predictions'])
            old_control = next(c for c in report['controls'] if c['seed'] == seed and c['variant'] == 'refined40')
            controls = restore(old_control['predictions'], injected.times().unix)
            unknown, score = unassigned_indices(predictions, truth, controls, len(sequence), 3.)
            assert score['recovered_count'] == saved['recovered_count'] and score['new_unmatched_count'] == saved['new_unmatched_count']
            evidence, windows = inspect_frames(background, injected, run, control_run, truth[0]['xy'], config, 3.)
            fits = {(f['frame'], f['source_index']): f for f in diagnostics['fits']}
            chosen = sorted(set(unknown) | {m['prediction_index'] for m in score['matches']})
            for k in chosen:
                observations = []
                prediction = predictions[k]
                for frame, xy in zip(prediction.track.frames, prediction.track.xy_det):
                    candidates = [(key[1], fit) for key, fit in fits.items() if key[0] == frame and
                                  np.linalg.norm(np.array(fit['fitted_xy'] if fit['accepted'] else fit['original_xy'])-xy) < 1e-7]
                    if candidates:
                        index, fit = min(candidates, key=lambda pair: pair[0])
                    else:
                        distance = np.linalg.norm(run.detections[frame].xy-xy, axis=1)
                        index = int(np.argmin(distance))
                        assert distance[index] < 1e-7
                        fit = dict(reason='not_in_initial_track', accepted=False, original_xy=xy.tolist())
                    source = evidence[frame]['sources'][index]
                    original = np.array(fit['original_xy'])
                    observations.append(dict(frame=frame, source_index=index, label=source['label'],
                                             original_xy=original.tolist(), final_xy=xy.tolist(),
                                             original_error_px=float(np.linalg.norm(original-truth[0]['xy'][frame])),
                                             final_error_px=float(np.linalg.norm(xy-truth[0]['xy'][frame])), fit=fit,
                                             segment_signal_fraction=source['segment_signal_fraction'],
                                             injected_to_positive_residual_ratio=source['injected_to_positive_residual_ratio'],
                                             nearest_control_detection_px=source['nearest_control_detection_px']))
                row = dict(phase=phase, seed=seed, flux_adu=flux, speed_px_s=speed, prediction_index=k,
                           unassigned=k in unknown, individually_matches_injection=bool(evaluate_tracks([prediction], truth, 3.)['tp']),
                           observations=observations)
                name = f's{seed}-f{flux:g}-v{speed:g}-track{k}.png'
                row['figure'] = name
                plot(output/name, f'{phase} seed={seed} flux={flux:g} speed={speed:g} track={k}', observations, windows, truth[0]['xy'])
                rows.append(row)
        write_json(output/'review.json', dict(cases=rows, total_seconds=time.perf_counter()-started,
                                               scope='Retrospective pixel diagnosis; recovery labels and production unchanged.'))
    except Exception as exc:
        write_json(output/'failure.json', dict(error=str(exc), completed_tracks=len(rows)))
        raise
    print(f'Review complete: {len(rows)} tracks, {sum(len(r["observations"]) for r in rows)} observations', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review(args.development, args.validation, args.input, args.output)
