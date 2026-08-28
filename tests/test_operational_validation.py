from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

import beyondcxr.training.operational_validation as operational_validation
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.operational_validation import (
    validate_existing_sqlite_database,
    validate_writable_directory_destination,
    validate_writable_file_destination,
)


def test_writable_file_destination_accepts_missing_and_existing_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"
    assert validate_writable_file_destination(missing, "tracking database") == missing
    existing = tmp_path / "existing.db"
    existing.write_bytes(b"database")
    assert validate_writable_file_destination(existing, "tracking database") == existing


def test_existing_sqlite_validation_is_read_only_and_rejects_corruption(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"
    validate_existing_sqlite_database(missing, "tracking database")
    assert not missing.exists()

    valid = tmp_path / "valid.db"
    with sqlite3.connect(valid) as connection:
        connection.execute("CREATE TABLE evidence (value INTEGER NOT NULL)")
        connection.execute("INSERT INTO evidence VALUES (1)")
    before = valid.read_bytes()
    validate_existing_sqlite_database(valid, "tracking database")
    assert valid.read_bytes() == before

    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"not a SQLite database")
    with pytest.raises(ManifestBuildError, match="valid SQLite database"):
        validate_existing_sqlite_database(corrupt, "tracking database")


@pytest.mark.parametrize("denied", ("file", "parent"))
def test_existing_file_destination_requires_file_and_parent_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, denied: str
) -> None:
    destination = tmp_path / "mlflow.db"
    destination.write_bytes(b"database")

    def access(path: os.PathLike[str], mode: int) -> bool:
        candidate = Path(path)
        if denied == "file" and candidate == destination and mode == os.W_OK:
            return False
        if denied == "parent" and candidate == tmp_path and mode == os.W_OK | os.X_OK:
            return False
        return True

    monkeypatch.setattr(operational_validation.os, "access", access)

    with pytest.raises(ManifestBuildError, match="writable file|writable and searchable"):
        validate_writable_file_destination(destination, "tracking database")


@pytest.mark.parametrize("failure", ("symlink", "parent-symlink", "directory"))
def test_writable_file_destination_rejects_wrong_or_symlinked_state(
    tmp_path: Path, failure: str
) -> None:
    destination = tmp_path / "tracking.db"
    if failure == "symlink":
        target = tmp_path / "target.db"
        target.write_bytes(b"database")
        destination.symlink_to(target)
    elif failure == "parent-symlink":
        parent = tmp_path / "physical"
        parent.mkdir()
        alias = tmp_path / "alias"
        alias.symlink_to(parent, target_is_directory=True)
        destination = alias / "tracking.db"
    else:
        destination.mkdir()
    with pytest.raises(ManifestBuildError, match="symlink|writable file"):
        validate_writable_file_destination(destination, "tracking database")


def test_writable_directory_destination_accepts_missing_and_existing_directories(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    assert validate_writable_directory_destination(missing, "output root") == missing
    existing = tmp_path / "existing"
    existing.mkdir()
    assert validate_writable_directory_destination(existing, "output root") == existing


@pytest.mark.parametrize("existing", (False, True))
def test_directory_destination_requires_write_and_search_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    destination = tmp_path / "output"
    permission_target = destination if existing else tmp_path
    if existing:
        destination.mkdir()

    def access(path: os.PathLike[str], mode: int) -> bool:
        if Path(path) == permission_target and mode == os.W_OK | os.X_OK:
            return False
        return True

    monkeypatch.setattr(operational_validation.os, "access", access)

    with pytest.raises(ManifestBuildError, match="writable and searchable"):
        validate_writable_directory_destination(destination, "output root")


@pytest.mark.parametrize("failure", ("file", "symlink"))
def test_writable_directory_destination_rejects_files_and_symlinks(
    tmp_path: Path, failure: str
) -> None:
    destination = tmp_path / "output"
    if failure == "file":
        destination.write_bytes(b"not a directory")
    else:
        physical = tmp_path / "physical"
        physical.mkdir()
        destination.symlink_to(physical, target_is_directory=True)
    with pytest.raises(ManifestBuildError, match="symlink|writable.*directory"):
        validate_writable_directory_destination(destination, "output root")
