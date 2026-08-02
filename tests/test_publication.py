from __future__ import annotations

import os
from pathlib import Path

import pytest

from radfusion.utils.publication import publish_directory, staging_directory


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
