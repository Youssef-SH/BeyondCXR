"""Fail-fast local advisory locks for formal campaign transitions."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.operational_validation import validate_directory_chain

try:
    import fcntl
except ImportError:  # pragma: no cover - formal campaigns require POSIX
    fcntl = None  # type: ignore[assignment]


@contextmanager
def campaign_lock(
    control_root: str | Path,
    study: str,
    *,
    trusted_root: str | Path,
) -> Iterator[None]:
    """Hold one exception-safe nonblocking lock for a formal campaign."""
    if fcntl is None:
        raise ManifestBuildError("Formal campaign locking requires a POSIX platform")
    root = Path(control_root).absolute()
    validate_directory_chain(
        trusted_root,
        root,
        f"{study} campaign control root",
        create=True,
    )
    path = root / "campaign.lock"
    if path.is_symlink():
        raise ManifestBuildError(f"{study} campaign lock path is invalid")
    flags = os.O_CREAT | os.O_RDWR | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ManifestBuildError(f"{study} campaign lock path is unusable") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ManifestBuildError(f"{study} campaign lock path is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ManifestBuildError(f"Another {study} campaign is active") from exc
        except OSError as exc:
            raise ManifestBuildError(f"{study} campaign lock acquisition failed") from exc
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
