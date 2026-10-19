import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from scripts.package_source import build_archive, SOURCE_DIRS, TOP_FILES
from scripts.verify_release import verify


@pytest.fixture
def release(tmp_path):
    root = tmp_path/'source'
    root.mkdir()
    for name in TOP_FILES:
        (root/name).write_text('test\n', encoding='utf-8')
    for name in SOURCE_DIRS:
        (root/name).mkdir(parents=True, exist_ok=True)
    (root/'src/main.py').write_text('print(1)\n', encoding='utf-8')
    info = dict(version='test', image_id='sha256:abc', dataset='data/rst19',
                inputs=[dict(path='data/rst19/one.fits', bytes=4, sha256=hashlib.sha256(b'fits').hexdigest())])
    archive = tmp_path/'source.zip'
    build_archive(root, archive, release_info=info)
    with ZipFile(archive) as z:
        z.extractall(tmp_path/'extracted')
    app = tmp_path/'extracted/star-chart-recognization'
    (app/'data/rst19').mkdir()
    (app/'data/rst19/one.fits').write_bytes(b'fits')
    return app


def test_release_metadata_is_covered_and_modified_source_fails(release):
    assert verify(release)['version'] == 'test'
    (release/'src/main.py').write_text('print(2)\n', encoding='utf-8')
    with pytest.raises(ValueError, match='source'):
        verify(release)


def test_release_refuses_changed_data_extra_frames_and_extra_code(release):
    path = release/'data/rst19/one.fits'
    path.write_bytes(b'bad!')
    with pytest.raises(ValueError, match='input'):
        verify(release)
    path.write_bytes(b'fits')
    extra = release/'data/rst19/two.fits'
    extra.write_bytes(b'fits')
    with pytest.raises(ValueError, match='file list'):
        verify(release)
    extra.unlink()
    (release/'src/override.py').write_text('x=1\n')
    with pytest.raises(ValueError, match='Unmanifested'):
        verify(release)


def test_release_checks_exact_image_id(release, monkeypatch):
    monkeypatch.setattr('scripts.verify_release.subprocess.run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='sha256:abc\n'))
    assert verify(release, environment=True)['environment_checked']
    monkeypatch.setattr('scripts.verify_release.subprocess.run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='sha256:other\n'))
    with pytest.raises(ValueError, match='image unavailable'):
        verify(release, environment=True)


def test_release_cannot_escape_manifest_root(release):
    manifest = release/'SOURCE_MANIFEST.json'
    value = json.loads(manifest.read_text())
    value['files'][0]['path'] = '../../outside.py'
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='escapes'):
        verify(release)
