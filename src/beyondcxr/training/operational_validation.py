"""Read-only validation of campaign filesystem and database destinations."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from beyondcxr.data.errors import ManifestBuildError


def validate_writable_directory_destination(value: str | Path, context: str) -> Path:
    """Validate one existing or prospective writable directory without creating it."""
    return _validate_destination_kind(value, context, kind="directory")


def validate_writable_file_destination(value: str | Path, context: str) -> Path:
    """Validate one existing or prospective writable regular file without creating it."""
    return _validate_destination_kind(value, context, kind="file")


def validate_existing_sqlite_database(value: str | Path, context: str) -> None:
    """Read-only validate an existing SQLite database without creating or migrating it."""
    path = Path(value).absolute()
    if not path.exists() and not path.is_symlink():
        return
    database = validate_writable_file_destination(path, context)
    try:
        with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as connection:
            result = connection.execute("PRAGMA quick_check").fetchone()
    except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
        raise ManifestBuildError(f"{context} is not a readable valid SQLite database") from exc
    if result != ("ok",):
        raise ManifestBuildError(f"{context} failed SQLite integrity validation")


def _validate_destination_kind(value: str | Path, context: str, *, kind: str) -> Path:
    path = Path(value).absolute()
    if any(candidate.is_symlink() for candidate in (path, *path.parents)):
        raise ManifestBuildError(f"{context} contains a symlink")
    if path.exists():
        if kind == "directory":
            if not path.is_dir() or not os.access(path, os.W_OK | os.X_OK):
                raise ManifestBuildError(f"{context} is not a writable and searchable directory")
        else:
            if not path.is_file() or not os.access(path, os.W_OK):
                raise ManifestBuildError(f"{context} is not a writable file")
            parent = path.parent
            if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
                raise ManifestBuildError(
                    f"{context} parent is not a writable and searchable directory"
                )
        return path.resolve()
    current = path
    while not current.exists():
        if current.parent == current:
            raise ManifestBuildError(f"{context} has no existing parent")
        current = current.parent
    if not current.is_dir() or not os.access(current, os.W_OK | os.X_OK):
        raise ManifestBuildError(f"{context} has no writable and searchable directory ancestor")
    return path.resolve()
