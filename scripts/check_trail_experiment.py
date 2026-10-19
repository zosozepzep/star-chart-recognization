"""Independently check the two completed trail experiments (stdlib only)."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_report(path, input_directory):
    report = read(path)
    assert report['kind'] == 'trail-refinement-experiment'
    for name, checksum in report['source_sha256'].items():
        source = (ROOT/name).resolve()
        assert source.is_relative_to(ROOT) and digest(source) == checksum, name
    for entry in report['input_files']:
        assert Path(entry['name']).name == entry['name']
        assert digest(input_directory/entry['name']) == entry['sha256'], entry['name']
    receipt = read(path.parent/'receipt.json')
    for name, checksum in receipt['files'].items():
        assert Path(name).name == name
        assert digest(path.parent/name) == checksum, name
    expected = set(itertools.product([c['seed'] for c in report['crops']],
                                     report['protocol']['fluxes_adu'],
                                     report['protocol']['speeds_px_s'], report['variants']))
    keys = [(r['seed'], r['flux_adu'], r['speed_px_s'], r['variant']) for r in report['cases']]
    assert len(keys) == len(set(keys)) and set(keys) == expected
    assert receipt['completed_rows'] == len(keys)
    assert [json.loads(s) for s in (path.parent/'cases.jsonl').read_text(encoding='utf-8').splitlines()] == report['cases']
    for row in report['cases']:
        assert row['recovered_count']+row['missed_count'] == 1
        assert row['predicted_count'] == len(row['predictions'])
        assert row['predicted_count'] == row['recovered_count']+row['matched_control_count']+row['new_unmatched_count']
        errors = []
        for match in row['matches']:
            prediction = row['predictions'][match['prediction_index']]
            assert len(prediction['frames']) == len(set(prediction['frames']))
            measured = [math.dist(xy, row['truth_xy'][f]) for f, xy in zip(prediction['frames'], prediction['xy'])]
            near = sum(e <= report['protocol']['match_radius_px'] for e in measured)
            assert near >= 4 and near/len(measured) >= .8
            assert near == match['matched_observations']
            errors.extend(measured)
        assert len(errors) == row['matched_track_observations']
        assert math.isclose(sum(e*e for e in errors), row['matched_track_squared_error_sum'], abs_tol=1e-8)
        fits = row['diagnostics'].get('fits', [])
        if fits:
            assert Counter(f['reason'] for f in fits) == row['diagnostics']['fit_reasons']
            assert all(f['accepted'] == (f['reason'] == 'accepted') for f in fits)
    for total in report['totals']:
        group = [r for r in report['cases'] if r['variant'] == total['variant']]
        for key in ('recovered_count', 'new_unmatched_count', 'duplicate_eligible_count'):
            assert sum(r[key] for r in group) == total[key]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    development = verify_report(args.development, args.input)
    validation = verify_report(args.validation, args.input)
    assert development['phase'] == 'development' and validation['phase'] == 'validation'
    assert validation['source_sha256'] == development['source_sha256']
    assert validation['development_sha256'] == digest(args.development)
    assert validation['baseline_sha256'] == development['baseline_sha256']
    assert validation['crops'] == development['validation_crops']
    receipt = dict(development_rows=len(development['cases']), validation_rows=len(validation['cases']),
                    source_files=len(validation['source_sha256']), input_files=len(validation['input_files']),
                    checker_sha256=digest(Path(__file__)),
                    development_sha256=digest(args.development), validation_sha256=digest(args.validation),
                    checks='source/input/output hashes, frozen development, complete unique grid, JSONL equality, '
                           'count accounting, original matching eligibility, all-observation errors, fit reasons, totals')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
