"""Strict machine-local configuration for the public BeyondCXR workflow."""

from __future__ import annotations

import json
import os
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.operational_validation import validate_writable_directory_destination

LOCAL_CONFIG_FILENAME = ".beyondcxr.local.toml"
DEFAULT_RSNA_SOURCE = Path("data/raw/rsna/extracted")
DEFAULT_SYMILE_SOURCE = Path("data/raw/symile/extracted")


@dataclass(frozen=True)
class LocalWorkflowConfig:
    """Resolved machine paths; none of these values are scientific identity."""

    backup_root: Path
    rsna_source_root: Path
    symile_source_root: Path


def load_local_config(
    repository_root: str | Path,
    *,
    require_backup: bool = True,
) -> LocalWorkflowConfig | None:
    """Validate saved machine paths; require a backup unless explicitly optional."""
    repository = Path(repository_root).resolve()
    document = _read_local_document(repository)
    if document is None:
        if require_backup:
            raise ManifestBuildError("Machine setup is incomplete; run `make setup` first")
        return None
    backup_value = document.get("backup_root")
    if backup_value is None:
        if require_backup:
            raise ManifestBuildError("Machine setup is incomplete; run `make setup` first")
        return None
    backup = _resolve_configured_path(repository, backup_value, "backup_root")
    rsna = _resolve_configured_path(
        repository,
        document.get("rsna", {}).get("source_root", DEFAULT_RSNA_SOURCE.as_posix()),
        "rsna.source_root",
    )
    symile = _resolve_configured_path(
        repository,
        document.get("symile", {}).get("source_root", DEFAULT_SYMILE_SOURCE.as_posix()),
        "symile.source_root",
    )
    if backup.is_relative_to(repository) or not backup.is_dir():
        raise ManifestBuildError("Configured persistent backup root is invalid")
    validate_writable_directory_destination(backup, "Persistent backup root")
    for source, context in ((rsna, "RSNA source root"), (symile, "Symile source root")):
        if source.exists() and not source.is_dir():
            raise ManifestBuildError(f"Configured {context} is invalid")
    return LocalWorkflowConfig(backup, rsna, symile)


def configure_local_machine(
    repository_root: str | Path,
    *,
    backup_root: str | Path | None = None,
    rsna_source_root: str | Path | None = None,
    symile_source_root: str | Path | None = None,
    interactive: bool = False,
    prompt: Callable[[str], str] = input,
) -> LocalWorkflowConfig:
    """Validate and atomically publish machine paths without storing scientific settings."""
    repository = Path(repository_root).resolve()
    document = _read_local_document(repository) or {}
    try:
        existing = load_local_config(repository, require_backup=False)
    except ManifestBuildError:
        existing = None
    selected_backup: str | Path | None = backup_root
    if selected_backup is None and existing is not None:
        selected_backup = existing.backup_root
    if selected_backup is None:
        if not interactive:
            raise ManifestBuildError(
                "A persistent backup root is required; run "
                "`make setup BACKUP_ROOT=/approved/persistent/backup`"
            )
        selected_backup = prompt("Persistent backup root: ").strip()
    backup = _resolve_configured_path(repository, selected_backup, "backup_root")
    if backup.is_relative_to(repository):
        raise ManifestBuildError("The persistent backup root must be outside the repository")
    validate_writable_directory_destination(backup, "Persistent backup root")
    backup.mkdir(parents=True, exist_ok=True)
    validate_writable_directory_destination(backup, "Persistent backup root")

    rsna = _select_source_root(
        repository,
        rsna_source_root,
        document.get("rsna", {}).get("source_root"),
        DEFAULT_RSNA_SOURCE,
        "RSNA source root",
    )
    symile = _select_source_root(
        repository,
        symile_source_root,
        document.get("symile", {}).get("source_root"),
        DEFAULT_SYMILE_SOURCE,
        "Symile source root",
    )
    config = LocalWorkflowConfig(backup, rsna, symile)
    if existing != config or not (repository / LOCAL_CONFIG_FILENAME).is_file():
        _publish_config(repository / LOCAL_CONFIG_FILENAME, repository, config)
    return config


def _read_local_document(repository: Path) -> dict[str, Any] | None:
    path = repository / LOCAL_CONFIG_FILENAME
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ManifestBuildError("Machine-local configuration is not a regular file")
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ManifestBuildError("Machine-local configuration is unreadable") from exc
    if not isinstance(document, dict) or set(document) - {"backup_root", "rsna", "symile"}:
        raise ManifestBuildError("Machine-local configuration contains unsupported fields")
    for section in ("rsna", "symile"):
        value = document.get(section, {})
        if not isinstance(value, dict) or set(value) - {"source_root"}:
            raise ManifestBuildError(
                f"Machine-local configuration section [{section}] contains unsupported fields"
            )
    return document


def _select_source_root(
    repository: Path,
    override: str | Path | None,
    existing: object,
    default: Path,
    context: str,
) -> Path:
    value = override if override is not None else existing
    if value is None:
        value = default
    path = _resolve_configured_path(repository, value, context)
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ManifestBuildError(f"{context} must be a directory or an absent destination")
    return path


def _resolve_configured_path(repository: Path, value: object, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip() or "\x00" in str(value):
        raise ManifestBuildError(f"{field} must be a non-empty filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository / path
    if any(candidate.is_symlink() for candidate in (path, *path.parents)):
        raise ManifestBuildError(f"{field} must not contain a symbolic link")
    return path.resolve(strict=False)


def _config_path(repository: Path, path: Path) -> str:
    try:
        return path.relative_to(repository).as_posix()
    except ValueError:
        return path.as_posix()


def _publish_config(path: Path, repository: Path, config: LocalWorkflowConfig) -> None:
    backup = json.dumps(_config_path(repository, config.backup_root), ensure_ascii=False)
    rsna = json.dumps(_config_path(repository, config.rsna_source_root), ensure_ascii=False)
    symile = json.dumps(_config_path(repository, config.symile_source_root), ensure_ascii=False)
    document = (
        f"backup_root = {backup}\n\n"
        "[rsna]\n"
        f"source_root = {rsna}\n\n"
        "[symile]\n"
        f"source_root = {symile}\n"
    ).encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
