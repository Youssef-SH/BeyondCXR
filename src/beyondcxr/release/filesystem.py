"""Neutral filesystem primitives shared by release workflows."""

from __future__ import annotations

from pathlib import Path

from beyondcxr.data.errors import ManifestBuildError


def directory_bytes(root: str | Path) -> dict[str, bytes]:
    """Return deterministic relative file bytes for one symlink-free directory tree."""
    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise ManifestBuildError("Release directory is unavailable")
    result: dict[str, bytes] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ManifestBuildError("Release directory tree must not contain symlinks")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = path.read_bytes()
    return result
