"""Verify frozen source, input data and optional runtime before a demonstration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_path(root, name):
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Manifest path escapes release: {name}')
    return path


def verify(root, *, environment=False):
    root = Path(root).resolve()
    manifest = json.loads((root/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path = checked_path(root, entry['path'])
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f"Frozen source missing or changed: {entry['path']}")
    if 'RELEASE.json' not in {entry['path'] for entry in manifest['files']}:
        raise ValueError('Source manifest does not cover RELEASE.json')
    release = json.loads((root/'RELEASE.json').read_text(encoding='utf-8'))
    for entry in release['inputs']:
        path = checked_path(root, entry['path'])
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f"Official input missing or changed: {entry['path']}")
    dataset = checked_path(root, release['dataset'])
    actual = {p.relative_to(root).as_posix() for p in dataset.iterdir()
              if p.is_file() and p.suffix.lower() in ('.fits', '.fit', '.fts')}
    expected = {entry['path'] for entry in release['inputs']}
    if actual != expected:
        raise ValueError('Official FITS file list differs from frozen release')
    # Added Python modules or altered configuration can change imports without
    # modifying an existing manifest entry. Refuse them in the frozen source.
    known = {e['path'] for e in manifest['files']}
    for folder in ('src', 'scripts'):
        for path in (root/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.html', '.css', '.js', '.ps1', '.sh'):
                if path.relative_to(root).as_posix() not in known:
                    raise ValueError(f'Unmanifested executable/configuration: {path.relative_to(root)}')
    image = release['image_id']
    if environment:
        if sys.version_info < (3, 11):
            raise ValueError('The release launcher requires Python 3.11 or newer')
        process = subprocess.run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
                                 capture_output=True, text=True, timeout=30)
        if process.returncode or process.stdout.strip() != image:
            raise ValueError('Frozen CPU image unavailable; start Docker and load the supplied runtime archive')
    return dict(version=release['version'], source_files=len(manifest['files']),
                input_files=len(release['inputs']), image_id=image, environment_checked=environment,
                source_manifest_sha256=sha256(root/'SOURCE_MANIFEST.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--environment', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.root, environment=args.environment), indent=2))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'Release verification failed: {exc}\n')


if __name__ == '__main__':
    main()
