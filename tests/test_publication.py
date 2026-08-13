from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from radfusion.utils.publication import (
    install_immutable_directory,
    publish_directory,
    staging_directory,
    update_current_marker,
)


def _validate_text_object(path: str | Path, *, enforce_directory_name: bool = True) -> None:
    del enforce_directory_name
    if Path(path, "value.txt").read_text(encoding="utf-8") != "same":
        raise ValueError("invalid immutable object")


def test_immutable_directory_install_reports_creation_and_reuse(tmp_path: Path) -> None:
    destination = tmp_path / "object"
    first = staging_directory(destination)
    (first / "value.txt").write_text("same", encoding="utf-8")
    assert install_immutable_directory(first, destination, _validate_text_object) is True

    repeated = staging_directory(destination)
    (repeated / "value.txt").write_text("same", encoding="utf-8")
    assert install_immutable_directory(repeated, destination, _validate_text_object) is False
    assert (destination / "value.txt").read_text(encoding="utf-8") == "same"


def test_immutable_directory_does_not_replace_incomplete_destination(tmp_path: Path) -> None:
    destination = tmp_path / "object"
    destination.mkdir()
    stage = staging_directory(destination)
    (stage / "value.txt").write_text("same", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        install_immutable_directory(stage, destination, _validate_text_object)
    assert not list(destination.iterdir())


def test_immutable_directory_concurrent_winner_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "object"
    stage = staging_directory(destination)
    (stage / "value.txt").write_text("same", encoding="utf-8")

    def concurrent_winner(source: str | Path, target: str | Path) -> None:
        del source
        winner = Path(target)
        winner.mkdir()
        (winner / "value.txt").write_text("same", encoding="utf-8")
        raise OSError(errno.EEXIST, "concurrent winner")

    monkeypatch.setattr("radfusion.utils.publication.os.rename", concurrent_winner)
    assert install_immutable_directory(stage, destination, _validate_text_object) is False
    assert (destination / "value.txt").read_text(encoding="utf-8") == "same"


def test_current_marker_is_replaced_atomically(tmp_path: Path) -> None:
    current = tmp_path / "CURRENT"
    update_current_marker(current, "artifact-first")
    update_current_marker(current, "artifact-second")

    assert current.read_text(encoding="utf-8") == "artifact-second\n"
    assert not list(tmp_path.glob(".CURRENT-*.tmp"))


def test_successful_directory_publication_replaces_complete_previous_output(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "reports"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    stage = staging_directory(destination)
    (stage / "new.txt").write_text("new", encoding="utf-8")

    publish_directory(stage, destination)

    assert {path.name for path in destination.iterdir()} == {"new.txt"}
    assert (destination / "new.txt").read_text(encoding="utf-8") == "new"


def test_failed_directory_publication_restores_previous_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "reports"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    stage = staging_directory(destination)
    (stage / "partial.txt").write_text("partial", encoding="utf-8")
    real_replace = os.replace
    calls = 0

    def fail_stage_publish(source: str | Path, target: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publication failed")
        real_replace(source, target)

    monkeypatch.setattr("radfusion.utils.publication.os.replace", fail_stage_publish)

    with pytest.raises(OSError):
        publish_directory(stage, destination)

    assert {path.name for path in destination.iterdir()} == {"old.txt"}
    assert (destination / "old.txt").read_text(encoding="utf-8") == "old"
