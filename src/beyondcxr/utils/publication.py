"""Publish generated directories atomically with rollback."""

from __future__ import annotations

import ctypes
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


def is_publication_staging_directory(path: Path) -> bool:
    """Identify unpublished sibling directories, including interrupted publications."""
    return (
        path.name.startswith(".")
        and "-staging-" in path.name
        and not path.is_symlink()
        and path.is_dir()
    )


def install_immutable_directory(
    stage: str | Path,
    destination: str | Path,
    validator: Callable[..., Any],
) -> bool:
    """Atomically install without replacement, or validate a concurrent winner."""
    staged = Path(stage)
    target = Path(destination)
    validator(staged, enforce_directory_name=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        validator(target)
        return False
    try:
        rename_directory_no_replace(staged, target)
    except OSError as exc:
        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
            raise
        validator(target)
        return False
    return True


def rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Use Linux renameat2 to preserve even an empty concurrent destination."""
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOSYS, "Atomic no-replace directory publication is unavailable")
    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    # AT_FDCWD resolves both paths normally; RENAME_NOREPLACE forbids replacement.
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "Atomic no-replace directory publication failed")


def install_regenerable_directory(
    stage: str | Path,
    destination: str | Path,
    validator: Callable[..., Any],
) -> bool:
    """Validate and install a deterministic view, replacing only invalid cached bytes."""
    staged = Path(stage)
    target = Path(destination)
    validator(staged, enforce_directory_name=False)
    if target.exists() or target.is_symlink():
        try:
            validator(target)
        except (OSError, ValueError):
            publish_directory(staged, target)
            validator(target)
            return True
        return False
    publish_directory(staged, target)
    validator(target)
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


def publish_bytes_no_replace(destination: str | Path, content: bytes) -> bool:
    """Atomically publish complete bytes without replacing an existing pathname."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}-", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            return False
        _fsync_directory(target.parent)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def publish_regenerable_bytes(destination: str | Path, content: bytes) -> bool:
    """Atomically install deterministic bytes, reusing an exact existing view."""
    target = Path(destination)
    if target.is_file() and not target.is_symlink() and target.read_bytes() == content:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}-", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    backup = Path(tempfile.mkdtemp(prefix=f".{target.name}-backup-", dir=target.parent))
    backup.rmdir()
    backed_up = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists() or target.is_symlink():
            os.replace(target, backup)
            backed_up = True
        try:
            os.replace(temporary, target)
        except BaseException:
            if backed_up:
                os.replace(backup, target)
                backed_up = False
            raise
        _fsync_directory(target.parent)
        if backed_up:
            _remove_path(backup)
            backed_up = False
        return True
    finally:
        temporary.unlink(missing_ok=True)
        if backup.exists() or backup.is_symlink():
            if backed_up and not target.exists() and not target.is_symlink():
                os.replace(backup, target)
            else:
                _remove_path(backup)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


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
        if target.exists() or target.is_symlink():
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
            _remove_path(backup)
            destination_backed_up = False
    finally:
        if staged.exists():
            shutil.rmtree(staged)
        if backup.exists() or backup.is_symlink():
            if destination_backed_up and not target.exists() and not target.is_symlink():
                os.replace(backup, target)
            else:
                _remove_path(backup)
