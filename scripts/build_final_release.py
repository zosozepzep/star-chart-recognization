"""Build the small source ZIP and a separate complete offline demonstration kit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile

from scripts.package_source import build_archive
from scripts.verify_release import sha256, verify


ROOT = Path(__file__).resolve().parents[1]


def build(output, version, image, evidence=()):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Release destination already exists: {output}')
    inspection = subprocess.run(['docker', 'image', 'inspect', image], check=True, capture_output=True, text=True)
    info = json.loads(inspection.stdout)[0]
    files = sorted((ROOT/'data/rst19').glob('*.fits'))
    if len(files) != 15:
        raise ValueError('Final official dataset must contain exactly 15 FITS files')
    release = dict(version=version, created_utc=datetime.now(timezone.utc).isoformat(),
                    image_id=info['Id'], dataset='data/rst19', default_reference_frame=7,
                    production_peak_threshold_adu=80, refinement='experimental only; not used by production',
                    inputs=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha256(p)) for p in files])
    output.mkdir(parents=True)
    archive_path = output/f'star-chart-source-{version}.zip'
    try:
        receipt = build_archive(ROOT, archive_path, release_info=release)
        (output/(archive_path.name+'.sha256')).write_text(f"{receipt['sha256']}  {archive_path.name}\n", encoding='ascii')
        kit = output/'offline-kit'
        app_parent = kit/'app'
        app_parent.mkdir(parents=True)
        with ZipFile(archive_path) as archive:
            archive.extractall(app_parent)
        app = app_parent/'star-chart-recognization'
        for source in files:
            destination = app/source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        for relative in evidence:
            source = (ROOT/relative).resolve()
            if not source.is_relative_to((ROOT/'output').resolve()):
                raise ValueError('Evidence must be inside the project output directory')
            destination = app/source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination)
            elif source.is_file():
                shutil.copyfile(source, destination)
            else:
                raise FileNotFoundError(source)
        verify(app, environment=True)
        runtime = kit/'runtime'
        runtime.mkdir()
        print('Exporting frozen CPU image...', flush=True)
        subprocess.run(['docker', 'image', 'save', '--output', str(runtime/'star-chart-cpu.tar'), info['Id']], check=True)
        # Keep the start command local and resolve paths relative to this kit.
        (kit/'start.ps1').write_text("param([int]$Port = 8765)\n$ErrorActionPreference = 'Stop'\n"
            "& (Join-Path $PSScriptRoot 'app/star-chart-recognization/scripts/final_demo.ps1') -Port $Port -RuntimeArchive (Join-Path $PSScriptRoot 'runtime/star-chart-cpu.tar')\n",
            encoding='utf-8')
        shutil.copyfile(ROOT/'docs/final-demo.md', kit/'START-HERE.md')
        receipt.update(release=release, app=str(app), runtime_bytes=(runtime/'star-chart-cpu.tar').stat().st_size,
                       runtime_sha256=sha256(runtime/'star-chart-cpu.tar'))
        (output/'build-receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        print(json.dumps(receipt, indent=2, ensure_ascii=False), flush=True)
        return receipt
    except Exception as exc:
        (output/'BUILD-FAILED.json').write_text(json.dumps(dict(error=str(exc)), ensure_ascii=False), encoding='utf-8')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', default='national-final-20261018')
    parser.add_argument('--image', default='star-chart:cpu')
    parser.add_argument('--evidence', nargs='*', default=[])
    args = parser.parse_args()
    build(args.output, args.version, args.image, args.evidence)
