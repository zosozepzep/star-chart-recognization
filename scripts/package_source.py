"""Build a reproducible source ZIP under the contest's 200 MB limit (stdlib only).

Uses an explicit source allowlist, not git archive: historical tracked output and
uncommitted local data must never enter a submission. Does not modify git or data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 200_000_000  # Decimal MB, stricter than 200 MiB.
TOP_FILES = (
    "README.md", ".gitignore", ".gitattributes", ".dockerignore",
    "requirements.txt", "pytest.ini", "docker-compose.yml",
)
SOURCE_DIRS = {
    "src": {".py", ".yaml"},
    "tests": {".py"},
    "examples": {".py"},
    "scripts": {".py", ".sh", ".ps1"},
    "docker": {".cpu", ".gpu"},
    "docs/reports": {".md"},
    "docs/reference": {".txt", ".md", ".csv"},
}


def source_files(root: Path) -> list[Path]:
    files = [root / name for name in TOP_FILES]
    for name, suffixes in SOURCE_DIRS.items():
        directory = root / name
        if not directory.is_dir():
            raise ValueError(f"Missing source directory: {name}")
        files.extend(p for p in directory.rglob("*") if p.is_file() and p.suffix in suffixes)
    files.extend((root / "docs").glob("*.md"))
    for p in files:
        if not p.is_file():
            raise ValueError(f"Missing source file: {p.relative_to(root)}")
        if p.is_symlink() or any(parent.is_symlink() for parent in p.parents if parent != root):
            raise ValueError(f"Symlinks are not allowed in a source package: {p}")
        if not p.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Source path escapes project: {p}")
    return sorted(set(files), key=lambda p: p.relative_to(root).as_posix())


def build_archive(root: Path, destination: Path, max_bytes: int = MAX_BYTES) -> dict:
    files = source_files(root)
    if destination.exists():
        raise FileExistsError(f"Archive already exists; choose another --output: {destination}")
    entries = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
    # Keep documented mount points even when the organizer supplies data separately.
    entries.update({"data/images/.gitkeep": b"", "output/.gitkeep": b""})
    manifest = {
        "format": 1,
        "size_limit_bytes": max_bytes,
        "excluded": ["raw data", "generated output", "Docker images", "git history",
                     "local probes", "legacy code", "internal planning", "contest attachments"],
        "files": [
            {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(entries.items())
        ],
    }
    entries["SOURCE_MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage in the same directory: failures leave an existing submission untouched.
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".zip.tmp", delete=False) as fh:
        staging = Path(fh.name)
    try:
        with ZipFile(staging, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in sorted(entries.items()):
                info = ZipInfo("star-chart-recognization/" + name, (2026, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
                info.compress_type = ZIP_DEFLATED
                archive.writestr(info, data, compresslevel=9)
        size = staging.stat().st_size
        if size > max_bytes:
            raise ValueError(f"Archive exceeds limit: {size:,} > {max_bytes:,} bytes")
        with ZipFile(staging) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive integrity check failed")
        staging.rename(destination)
    finally:
        staging.unlink(missing_ok=True)
    return {"archive": str(destination), "files": len(entries), "bytes": size,
            "limit_bytes": max_bytes, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "star-chart-source.zip")
    args = parser.parse_args()
    try:
        result = build_archive(ROOT, args.output.resolve())
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Packaging failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
