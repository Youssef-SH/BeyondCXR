from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest

import beyondcxr.training.campaign_lock as lock_module
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.campaign_lock import campaign_lock


def test_campaign_lock_is_fail_fast_and_exception_safe(tmp_path: Path) -> None:
    with campaign_lock(tmp_path, "Synthetic", trusted_root=tmp_path):
        with pytest.raises(ManifestBuildError, match="active"):
            with campaign_lock(tmp_path, "Synthetic", trusted_root=tmp_path):
                raise AssertionError("unreachable")

    with campaign_lock(tmp_path, "Synthetic", trusted_root=tmp_path):
        pass

    with pytest.raises(RuntimeError, match="interrupted transition"):
        with campaign_lock(tmp_path, "Synthetic", trusted_root=tmp_path):
            raise RuntimeError("interrupted transition")

    with campaign_lock(tmp_path, "Synthetic", trusted_root=tmp_path):
        pass


def _hold_lock(path: str, acquired, release) -> None:
    with campaign_lock(path, "Synthetic", trusted_root=path):
        acquired.set()
        release.wait()


def test_campaign_lock_is_fail_fast_across_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    acquired = context.Event()
    release = context.Event()
    process = context.Process(target=_hold_lock, args=(str(tmp_path), acquired, release))
    process.start()
    try:
        assert acquired.wait(timeout=10)
        with pytest.raises(ManifestBuildError, match="active"):
            with campaign_lock(tmp_path, "Synthetic", trusted_root=tmp_path):
                raise AssertionError("unreachable")
    finally:
        release.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0


@pytest.mark.parametrize("kind", ["directory", "symlink", "fifo"])
def test_campaign_lock_rejects_malformed_lock_path(tmp_path: Path, kind: str) -> None:
    lock = tmp_path / "campaign.lock"
    if kind == "directory":
        lock.mkdir()
    elif kind == "symlink":
        lock.symlink_to(tmp_path / "target")
    else:
        os.mkfifo(lock)
    with pytest.raises(ManifestBuildError, match="lock path"):
        with campaign_lock(tmp_path, "Synthetic", trusted_root=tmp_path):
            raise AssertionError("unreachable")


def test_campaign_lock_reports_posix_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lock_module, "fcntl", None)
    with pytest.raises(ManifestBuildError, match="POSIX"):
        with campaign_lock(tmp_path, "Synthetic", trusted_root=tmp_path):
            raise AssertionError("unreachable")


def test_campaign_lock_rejects_symlinked_control_root_before_external_mutation(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    repository.mkdir()
    outside.mkdir()
    control = repository / "control"
    control.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ManifestBuildError, match="symlink"):
        with campaign_lock(control, "Synthetic", trusted_root=repository):
            raise AssertionError("unreachable")

    assert list(outside.iterdir()) == []


def test_campaign_lock_rejects_symlinked_intermediate_before_external_mutation(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    repository.mkdir()
    outside.mkdir()
    (repository / "private").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ManifestBuildError, match="symlink"):
        with campaign_lock(
            repository / "private/control/symile",
            "Synthetic",
            trusted_root=repository,
        ):
            raise AssertionError("unreachable")

    assert list(outside.iterdir()) == []


def test_campaign_lock_safely_creates_absent_nested_directories(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    control = repository / "private/control/symile"

    with campaign_lock(control, "Synthetic", trusted_root=repository):
        assert (control / "campaign.lock").is_file()


def test_campaign_lock_rejects_parent_traversal_before_mutation(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    with pytest.raises(ManifestBuildError, match="escapes"):
        with campaign_lock(repository / "../outside", "Synthetic", trusted_root=repository):
            pytest.fail("unexpected lock")
    assert not (tmp_path / "outside").exists()


def test_campaign_lock_reuses_existing_real_directory_chain(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    control = repository / "private/control/symile"
    control.mkdir(parents=True)

    with campaign_lock(control, "Synthetic", trusted_root=repository):
        assert (control / "campaign.lock").is_file()


def test_campaign_lock_rejects_non_directory_component(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "private").write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(ManifestBuildError, match="non-directory"):
        with campaign_lock(
            repository / "private/control/symile",
            "Synthetic",
            trusted_root=repository,
        ):
            raise AssertionError("unreachable")
