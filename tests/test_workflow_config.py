from __future__ import annotations

import os
from pathlib import Path

import pytest

from beyondcxr import workflow_config
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.workflow_config import (
    LOCAL_CONFIG_FILENAME,
    configure_local_machine,
    load_local_config,
)


def test_setup_requires_explicit_backup_noninteractively(tmp_path: Path) -> None:
    with pytest.raises(ManifestBuildError, match="BACKUP_ROOT=/approved/persistent/backup"):
        configure_local_machine(tmp_path, interactive=False)


def test_setup_round_trips_non_bmp_unicode_roots(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    configured = configure_local_machine(
        repository,
        backup_root=tmp_path / "backup-\U0001f4be",
        rsna_source_root="source-\U0001f4be",
    )
    assert load_local_config(repository) == configured


def test_setup_prompts_once_and_writes_strict_private_config(tmp_path: Path) -> None:
    backup = tmp_path.parent / f"{tmp_path.name}-backup"
    prompts: list[str] = []
    config = configure_local_machine(
        tmp_path,
        interactive=True,
        prompt=lambda message: prompts.append(message) or str(backup),
    )
    assert prompts == ["Persistent backup root: "]
    assert backup.is_dir()
    assert config.backup_root == backup.resolve()
    path = tmp_path / LOCAL_CONFIG_FILENAME
    assert path.stat().st_mode & 0o777 == 0o600
    assert load_local_config(tmp_path) == config
    assert config.rsna_source_root == (tmp_path / "data/raw/rsna/extracted").resolve()
    assert config.symile_source_root == (tmp_path / "data/raw/symile/extracted").resolve()


def test_setup_overrides_take_precedence_and_valid_config_is_reused(tmp_path: Path) -> None:
    first_backup = tmp_path.parent / f"{tmp_path.name}-backup-one"
    second_backup = tmp_path.parent / f"{tmp_path.name}-backup-two"
    first = configure_local_machine(
        tmp_path,
        backup_root=first_backup,
        rsna_source_root="sources/rsna",
        symile_source_root="sources/symile",
    )
    path = tmp_path / LOCAL_CONFIG_FILENAME
    before = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns
    assert configure_local_machine(tmp_path) == first
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == before_mtime
    updated = configure_local_machine(tmp_path, backup_root=second_backup)
    assert updated.backup_root == second_backup.resolve()
    assert updated.rsna_source_root == first.rsna_source_root
    assert updated.symile_source_root == first.symile_source_root


def test_setup_repairs_invalid_saved_backup_with_explicit_override(tmp_path: Path) -> None:
    invalid_backup = tmp_path.parent / f"{tmp_path.name}-invalid-backup"
    invalid_backup.write_text("not a directory", encoding="utf-8")
    path = tmp_path / LOCAL_CONFIG_FILENAME
    path.write_text(
        f'backup_root = "{invalid_backup.as_posix()}"\n\n'
        '[rsna]\nsource_root = "sources/rsna"\n\n'
        '[symile]\nsource_root = "sources/symile"\n',
        encoding="utf-8",
    )
    replacement = tmp_path.parent / f"{tmp_path.name}-replacement-backup"

    repaired = configure_local_machine(tmp_path, backup_root=replacement)

    assert repaired.backup_root == replacement.resolve()
    assert repaired.rsna_source_root == (tmp_path / "sources/rsna").resolve()
    assert repaired.symile_source_root == (tmp_path / "sources/symile").resolve()
    assert load_local_config(tmp_path) == repaired


def test_setup_prompts_to_replace_invalid_saved_backup(tmp_path: Path) -> None:
    invalid_backup = tmp_path.parent / f"{tmp_path.name}-invalid-backup"
    invalid_backup.write_text("not a directory", encoding="utf-8")
    (tmp_path / LOCAL_CONFIG_FILENAME).write_text(
        f'backup_root = "{invalid_backup.as_posix()}"\n', encoding="utf-8"
    )
    replacement = tmp_path.parent / f"{tmp_path.name}-replacement-backup"
    prompts: list[str] = []

    repaired = configure_local_machine(
        tmp_path,
        interactive=True,
        prompt=lambda message: prompts.append(message) or str(replacement),
    )

    assert prompts == ["Persistent backup root: "]
    assert repaired.backup_root == replacement.resolve()


def test_interactive_setup_preserves_sources_when_backup_is_missing(tmp_path: Path) -> None:
    path = tmp_path / LOCAL_CONFIG_FILENAME
    path.write_text('[rsna]\nsource_root = "sources/rsna"\n', encoding="utf-8")
    backup = tmp_path.parent / f"{tmp_path.name}-backup"
    prompts: list[str] = []

    configured = configure_local_machine(
        tmp_path,
        interactive=True,
        prompt=lambda message: prompts.append(message) or str(backup),
    )

    assert prompts == ["Persistent backup root: "]
    assert configured.rsna_source_root == (tmp_path / "sources/rsna").resolve()


def test_setup_atomic_replace_failure_preserves_existing_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_backup = tmp_path.parent / f"{tmp_path.name}-backup-one"
    second_backup = tmp_path.parent / f"{tmp_path.name}-backup-two"
    configure_local_machine(tmp_path, backup_root=first_backup)
    path = tmp_path / LOCAL_CONFIG_FILENAME
    original = path.read_bytes()

    def reject_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated atomic replacement failure")

    monkeypatch.setattr(workflow_config.os, "replace", reject_replace)
    with pytest.raises(OSError, match="simulated atomic replacement failure"):
        configure_local_machine(tmp_path, backup_root=second_backup)

    assert path.read_bytes() == original
    assert not tuple(tmp_path.glob(f".{LOCAL_CONFIG_FILENAME}-*.tmp"))


@pytest.mark.parametrize("saved_value", ['""', "false", "0"])
def test_setup_rejects_invalid_saved_source_without_defaulting(
    tmp_path: Path, saved_value: str
) -> None:
    path = tmp_path / LOCAL_CONFIG_FILENAME
    original = f"[rsna]\nsource_root = {saved_value}\n"
    path.write_text(original, encoding="utf-8")
    backup = tmp_path.parent / f"{tmp_path.name}-backup"

    with pytest.raises(ManifestBuildError, match="non-empty filesystem path"):
        configure_local_machine(tmp_path, backup_root=backup)

    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "content",
    [
        'backup_root = "/tmp/backup"\nsecret = "value"\n',
        'backup_root = "/tmp/backup"\n[rsna]\nworkers = 8\n',
        'backup_root = "/tmp/backup"\n[symile]\nthreshold = 0.5\n',
    ],
)
def test_local_config_rejects_non_path_fields(tmp_path: Path, content: str) -> None:
    (tmp_path / LOCAL_CONFIG_FILENAME).write_text(content, encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="unsupported fields"):
        load_local_config(tmp_path)


def test_setup_rejects_repository_local_backup_and_source_symlink(tmp_path: Path) -> None:
    with pytest.raises(ManifestBuildError, match="outside the repository"):
        configure_local_machine(tmp_path, backup_root=tmp_path / "backup")
    backup = tmp_path.parent / f"{tmp_path.name}-backup"
    source_target = tmp_path / "source-target"
    source_target.mkdir()
    source_link = tmp_path / "source-link"
    os.symlink(source_target, source_link)
    with pytest.raises(ManifestBuildError, match="symbolic link"):
        configure_local_machine(
            tmp_path,
            backup_root=backup,
            rsna_source_root=source_link,
        )
