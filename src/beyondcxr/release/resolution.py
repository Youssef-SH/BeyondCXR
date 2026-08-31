"""Unambiguous resolution of fully validated completed Symile campaigns."""

from __future__ import annotations

import re
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

from beyondcxr.data.errors import ManifestBuildError

_SYMILE_ARCHIVE_NAME = re.compile(
    r"symile-global-result-[0-9a-f]{64}-symile-execution-[0-9a-f]{64}\.zip"
)


def resolve_release_archive(candidate_or_directory: str | Path) -> Path:
    """Select an archive for an independent consumer that owns its own validation lifetime."""
    with resolved_release(candidate_or_directory) as (archive, _campaign):
        return archive


@contextmanager
def _restored_candidate(candidate: Path):
    from beyondcxr.training.preservation import restored_export, validate_archive_checksum
    from beyondcxr.training.symile_campaign import validate_restored_campaign

    validate_archive_checksum(candidate)
    with restored_export(candidate) as root:
        yield validate_restored_campaign(root)


@contextmanager
def resolved_release(candidate_or_directory: str | Path):
    """Keep the uniquely selected, recursively validated restoration alive for its consumers."""
    supplied = Path(candidate_or_directory)
    if supplied.is_symlink():
        raise ManifestBuildError("Release candidate path must not be a symlink")
    explicit = supplied.is_file()
    if not explicit and not supplied.is_dir():
        raise ManifestBuildError("Release candidate path is unavailable")
    candidates = (
        [supplied]
        if explicit
        else [
            path
            for path in sorted(supplied.iterdir())
            if not path.is_symlink()
            and path.is_file()
            and _SYMILE_ARCHIVE_NAME.fullmatch(path.name)
        ]
    )
    with ExitStack() as selected:
        result = None
        for candidate in candidates:
            with ExitStack() as trial:
                try:
                    campaign = trial.enter_context(_restored_candidate(candidate))
                except (OSError, ValueError, zipfile.BadZipFile) as exc:
                    if explicit:
                        raise ManifestBuildError("Release candidate is invalid") from exc
                    continue
                if result is not None:
                    raise ManifestBuildError(
                        "Multiple valid completed Symile release candidates are ambiguous"
                    )
                selected.enter_context(trial.pop_all())
                result = candidate, campaign
        if result is None:
            raise ManifestBuildError("No valid completed Symile release candidate exists")
        yield result
