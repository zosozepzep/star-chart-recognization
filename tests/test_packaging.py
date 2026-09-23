"""Submission boundary checks; also runnable without the scientific stack.

    python -m unittest discover -s tests -p test_packaging.py -v
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.package_source import SOURCE_DIRS, TOP_FILES, build_archive


class SourceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        for name in TOP_FILES:
            (self.root / name).write_text("test\n", encoding="utf-8")
        for name in SOURCE_DIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        (self.root / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")

    def test_sources_and_manifest_exclude_data_output_and_local_material(self):
        for name in ("data/images/secret.fits", "output/preview.png", ".git/config",
                     "probe/local.py", "src/__pycache__/main.pyc", "docker/image.tar",
                     "docs/superpowers/private.md"):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"must not ship")
        result = build_archive(self.root, self.root / "dist" / "source.zip")
        with ZipFile(result["archive"]) as archive:
            prefix = "star-chart-recognization/"
            names = {name.removeprefix(prefix) for name in archive.namelist()}
            self.assertIn("src/main.py", names)
            self.assertIn("data/images/.gitkeep", names)
            self.assertNotIn("data/images/secret.fits", names)
            self.assertNotIn("output/preview.png", names)
            self.assertNotIn("docs/superpowers/private.md", names)
            self.assertFalse(any("__pycache__" in name or name.startswith(("probe/", ".git/")) for name in names))
            self.assertNotIn("docker/image.tar", names)
            manifest = json.loads(archive.read(prefix + "SOURCE_MANIFEST.json"))
            self.assertEqual({f["path"] for f in manifest["files"]}, names - {"SOURCE_MANIFEST.json"})
            for entry in manifest["files"]:
                data = archive.read(prefix + entry["path"])
                self.assertEqual(entry["bytes"], len(data))
                self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())

    def test_oversized_archive_is_not_published(self):
        output = self.root / "dist" / "source.zip"
        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            build_archive(self.root, output, max_bytes=1)
        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.iterdir()), [])

    def test_existing_submission_is_preserved(self):
        output = self.root / "source.zip"
        output.write_bytes(b"existing submission")
        with self.assertRaises(FileExistsError):
            build_archive(self.root, output)
        self.assertEqual(output.read_bytes(), b"existing submission")

    def test_same_checkout_produces_identical_bytes(self):
        a = build_archive(self.root, self.root / "one.zip")
        b = build_archive(self.root, self.root / "two.zip")
        self.assertEqual(a["sha256"], b["sha256"])


if __name__ == "__main__":
    unittest.main()
