"""Publish generated directories atomically with rollback."""

from __future__ import annotations

import errno
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def staging_directory(destination: str | Path) -> Path:
    """Create a sibling staging directory for a generated destination."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}-staging-", dir=target.parent))


def install_immutable_directory(
    stage: str | Path,
    destination: str | Path,
    validator: Callable[..., Any],
) -> bool:
    """Install on the current POSIX workflow, or validate and reuse a completed winner.

    Identity destinations are required to be either absent or completed non-empty immutable
    objects. The sibling rename and destination are on the same filesystem.
    """
    staged = Path(stage)
    target = Path(destination)
    validator(staged, enforce_directory_name=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        validator(target)
        return False
    try:
        os.rename(staged, target)
    except OSError as exc:
        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
            raise
        validator(target)
        return False
    return True


def validate_path_component(value: object, field: str) -> str:
    """Require one non-special path component on POSIX and Windows."""
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).is_absolute()
        or Path(value).name != value
    ):
        raise ValueError(f"{field} must be one safe path component")
    return value


def update_current_marker(current_path: str | Path, immutable_id: str) -> None:
    """Atomically replace a text pointer to one immutable artifact identity."""
    target = Path(current_path)
    if not immutable_id or Path(immutable_id).name != immutable_id:
        raise ValueError("Immutable artifact identity is invalid")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}-", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(immutable_id + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def publish_directory(stage: str | Path, destination: str | Path) -> None:
    """Publish a complete staged directory, restoring an existing destination on failure."""
    staged = Path(stage)
    target = Path(destination)
    if not staged.is_dir():
        raise ValueError(f"Staged publication directory does not exist: {staged}")
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=f".{target.name}-backup-", dir=target.parent))
    backup.rmdir()
    destination_backed_up = False
    try:
        if target.exists():
            os.replace(target, backup)
            destination_backed_up = True
        try:
            os.replace(staged, target)
        except BaseException:
            if destination_backed_up:
                os.replace(backup, target)
                destination_backed_up = False
            raise
        if destination_backed_up:
            shutil.rmtree(backup)
            destination_backed_up = False
    finally:
        if staged.exists():
            shutil.rmtree(staged)
        if backup.exists():
            if destination_backed_up and not target.exists():
                os.replace(backup, target)
            else:
                shutil.rmtree(backup)
