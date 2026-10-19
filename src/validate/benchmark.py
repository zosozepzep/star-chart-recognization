"""Seeded image-level synthetic benchmark and paired motion ablations.

Truth is used ONLY by the evaluator, never passed to detection or association.
These controlled scenes are not an estimate of official-data completeness.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from astropy.time import Time, TimeDelta
from scipy.optimize import linear_sum_assignment

from src.config import load_config
from src.space_pipeline import analyze_space_sequence
from src.target.space_motion import find_space_targets
from src.target.trail_check import check_tracks


class SyntheticSequence:
    """512px star field, trailed targets, fixed defects and a point-flash control."""
    def __init__(self, seed, scenario):
        self.seed, self.scenario = int(seed), scenario
        self.stamps = np.cumsum(np.r_[0., np.random.default_rng(seed).uniform(1.35, 1.65, 11)])
        self.headers = [SimpleNamespace(date_obs=Time('2026-03-30T16:32:05') + TimeDelta(t, format='sec'),
                                        exposure_s=1.5) for t in self.stamps]
        rng = np.random.default_rng(seed + 100)
        self.stars = rng.uniform([25, 25], [487, 487], (100, 2))
        self.amplitudes = rng.uniform(180, 650, len(self.stars))
        self.truth = []
        for k, (start, velocity) in enumerate([([45., 150.], [9., 1.4]), ([370., 280.], [-5., 3.])]):
            visible = [f for f in range(12) if not (scenario == 'gaps' and k == 0 and f in (2, 5, 8))]
            self.truth.append(dict(id=k+1, xy=np.array(start)+self.stamps[:, None]*velocity,
                                   visible_frames=visible, velocity=velocity))

    def __len__(self):
        return len(self.headers)

    def times(self):
        return Time([h.date_obs for h in self.headers])

    @staticmethod
    def add_source(image, position, amplitude, velocity=(0., 0.), exposure=1.5):
        velocity = np.array(velocity)
        length = np.linalg.norm(velocity) * exposure
        half = int(length/2 + 7)
        x, y = position
        x0, x1 = max(0, int(x)-half), min(512, int(x)+half+2)
        y0, y1 = max(0, int(y)-half), min(512, int(y)+half+2)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        samples = max(1, int(np.ceil(length*2)))
        for offset in (np.linspace(-.5, .5, samples) if samples > 1 else [0.]):
            cx, cy = np.array(position) + offset * exposure * velocity
            image[y0:y1, x0:x1] += amplitude/samples * np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*1.15**2))

    def image(self, frame):
        rng = np.random.default_rng(self.seed*1000 + frame)
        noise = 7. if self.scenario == 'faint' else 3.
        yy, xx = np.indices((512, 512))
        image = 100. + .008*xx + .005*yy + rng.normal(0, noise, (512, 512))
        for p, amp in zip(self.stars, self.amplitudes):
            self.add_source(image, p, amp)
        for target in self.truth:
            if frame in target['visible_frames']:
                amp = 240. if self.scenario == 'faint' and target['id'] == 1 else 1100.
                self.add_source(image, target['xy'][frame], amp, target['velocity'])
        # Deliberately coherent but untrailed flashes: a targeted negative control,
        # not a claim about how frequently these occur in astronomical imagery.
        self.add_source(image, [60+11*self.stamps[frame], 415-1.2*self.stamps[frame]], 450.)
        for p in rng.uniform(20, 490, (8, 2)):
            self.add_source(image, p, rng.uniform(150, 500))
        image[30, 470] = 31000  # Fixed saturated defect.
        image[0] = 100         # Non-imaging metadata row.
        return image


def evaluate_tracks(verdicts, truth, radius=3.):
    """Maximum-cardinality one-to-one trajectory matching; duplicates are FP.

    Eligibility: >=4 visible timestamp matches and >=80% of a predicted track's
    observations within radius. Localization is measured only on those matches.
    """
    quality = np.zeros((len(truth), len(verdicts)))
    evidence = {}
    for i, target in enumerate(truth):
        visible = set(target['visible_frames'])
        for j, verdict in enumerate(verdicts):
            errors = [float(np.linalg.norm(xy-target['xy'][f]))
                      for f, xy in zip(verdict.track.frames, verdict.track.xy_det)
                      if f in visible]
            matches = [e for e in errors if e <= radius]
            fraction = len(matches)/verdict.track.length
            if len(matches) >= 4 and fraction >= .8:
                quality[i, j] = 10000 + fraction
                evidence[i, j] = matches
    pairs = []
    if quality.size:
        rows, cols = linear_sum_assignment(-quality)
        pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if quality[i, j] > 0]
    tp, fp, fn = len(pairs), len(verdicts)-len(pairs), len(truth)-len(pairs)
    errors = [e for pair in pairs for e in evidence[pair]]
    coverage = [len(evidence[i, j])/len(truth[i]['visible_frames']) for i, j in pairs]
    return dict(tp=tp, fp=fp, fn=fn, truth_count=len(truth), predicted_count=len(verdicts),
                precision=tp/(tp+fp) if tp+fp else None, recall=tp/len(truth) if truth else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                localization_rmse_px=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                matched_visible_coverage=float(np.mean(coverage)) if coverage else None,
                matches=[dict(truth_id=truth[i]['id'], prediction_index=j,
                              matched_observations=len(evidence[i, j])) for i, j in pairs])


def run_benchmark(output, seeds=(41, 137, 809), scenarios=('clean', 'gaps', 'faint')):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('基准输出目录非空，请另选目录')
    config = load_config()
    started = time.perf_counter()
    rows = []
    for seed in seeds:
        for scenario in scenarios:
            print(f'基准实验：seed={seed} / {scenario}', flush=True)
            seq = SyntheticSequence(seed, scenario)
            run = analyze_space_sequence(seq, config, reference_frame=6)
            # Shared detection and registration hold all upstream factors fixed.
            for variant in ('full', 'without_trail', 'without_gaps'):
                conf = deepcopy(config['space']['motion'])
                if variant == 'without_gaps':
                    conf['max_gap_frames'] = 0
                before = time.perf_counter()
                verdicts, _ = find_space_targets(run.detections, run.frames, run.registration, seq.times().unix, conf)
                if variant != 'without_trail':
                    verdicts, _ = check_tracks(seq, verdicts, config['space']['trail_check'])
                rows.append(dict(seed=int(seed), scenario=scenario, variant=variant,
                                 association_and_check_seconds=time.perf_counter()-before,
                                 **evaluate_tracks(verdicts, seq.truth)))
    aggregate = []
    for variant in ('full', 'without_trail', 'without_gaps'):
        selected = [r for r in rows if r['variant'] == variant]
        tp, fp, fn = [sum(r[key] for r in selected) for key in ('tp', 'fp', 'fn')]
        aggregate.append(dict(variant=variant, cases=len(selected), tp=tp, fp=fp, fn=fn,
                              precision=tp/(tp+fp) if tp+fp else None,
                              recall=tp/(tp+fn) if tp+fn else None,
                              f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None))
    root = Path(__file__).resolve().parents[2]
    sources = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in ('src',) for p in sorted((root/folder).rglob('*'))
               if p.suffix in ('.py', '.yaml')}
    report = dict(schema_version=1, kind='synthetic-image-benchmark',
                  created_utc=datetime.now(timezone.utc).isoformat(), seeds=list(seeds), scenarios=list(scenarios),
                  image_shape=[512, 512], frames_per_case=12, config=config, source_sha256=sources,
                  metrics_policy='一对一轨迹匹配；至少4个时刻且至少80%轨迹观测距注入真值≤3px；重复轨迹计FP。',
                  scope='合成图像的检测、配准、关联和拖尾核验；不含星等定标，不等于官方数据准确率。',
                  controls='圆高斯PSF、线性曝光拖尾、渐变背景、高斯噪声、固定饱和点、随机瞬时源及定向构造的无拖尾移动闪点。',
                  limitations=['每类场景种子数有限；不覆盖真实拥挤场、复杂PSF及全部仪器伪影。',
                               '消融共享同一批检测和配准，只改变漏帧容忍或拖尾核验。',
                               '相容的物理约束与生成模型相关；不能据此主张领域最优。',
                               '弱源场景保留生产峰值门限，预期暴露漏检；不按结果调参。'],
                  aggregate=aggregate, cases=rows, total_seconds=time.perf_counter()-started)
    output.mkdir(parents=True, exist_ok=True)
    (output/'benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[41, 137, 809])
    args = parser.parse_args()
    run_benchmark(args.output, seeds=args.seeds)


if __name__ == '__main__':
    main()
