from __future__ import annotations

import subprocess
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.rsna_acquisition import ensure_rsna_source


def _write_source(root: Path, *, complete: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "patientId": "sample",
                "x": None,
                "y": None,
                "width": None,
                "height": None,
                "Target": 0,
            }
        ]
    ).to_csv(root / "stage_2_train_labels.csv", index=False)
    pd.DataFrame([{"patientId": "sample", "class": "Normal"}]).to_csv(
        root / "stage_2_detailed_class_info.csv", index=False
    )
    images = root / "stage_2_train_images"
    images.mkdir()
    (images / "sample.dcm").write_bytes(b"dicom")
    if complete:
        (root / "stage_2_sample_submission.csv").write_text("patientId,PredictionString\n")
        (root / "stage_2_test_images").mkdir()


def _zip_source(source: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source).as_posix()
            if path.is_dir():
                handle.writestr(relative + "/", b"")
            else:
                handle.write(path, relative)


def test_valid_existing_rsna_source_never_uses_network(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source)

    def forbidden(_command: Sequence[str], _directory: Path) -> None:
        raise AssertionError("network downloader was called")

    assert ensure_rsna_source(source, downloader=forbidden) == source.resolve()


@pytest.mark.parametrize("nested", [False, True])
def test_rsna_source_symlink_is_rejected_before_download(tmp_path: Path, nested: bool) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    source = link / "source" if nested else link
    with pytest.raises(ManifestBuildError, match="symlink"):
        ensure_rsna_source(source, downloader=lambda *_: pytest.fail("unexpected download"))
    assert list(target.iterdir()) == []


def test_absent_rsna_source_is_safely_acquired_and_installed(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    _write_source(fixture, complete=True)
    commands: list[tuple[str, ...]] = []

    def download(command: Sequence[str], directory: Path) -> None:
        commands.append(tuple(command))
        _zip_source(fixture, directory / "release.zip")

    destination = tmp_path / "installed"
    assert ensure_rsna_source(destination, downloader=download) == destination.resolve()
    assert len(commands) == 1
    assert "rsna-pneumonia-detection-challenge" in commands[0]
    assert (destination / "stage_2_train_images/sample.dcm").read_bytes() == b"dicom"
    ensure_rsna_source(destination, downloader=lambda *_args: pytest.fail("unexpected download"))


def test_conflicting_existing_source_fails_without_download(tmp_path: Path) -> None:
    destination = tmp_path / "source"
    destination.mkdir()
    called = False

    def download(_command: Sequence[str], _directory: Path) -> None:
        nonlocal called
        called = True

    with pytest.raises(ManifestBuildError):
        ensure_rsna_source(destination, downloader=download)
    assert called is False


def test_kaggle_authentication_failure_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_download(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, ["kaggle"])

    monkeypatch.setattr("beyondcxr.rsna_acquisition.subprocess.run", reject_download)
    with pytest.raises(
        ManifestBuildError, match="verify Kaggle credentials and competition access"
    ):
        ensure_rsna_source(tmp_path / "source")
    assert not (tmp_path / "source").exists()


@pytest.mark.parametrize(
    "member",
    [
        "../escape.txt",
        "/stage_2_train_labels.csv",
        "unexpected.txt",
        "stage_2_train_images/nested/sample.dcm",
        "stage_2_train_images/notes.txt",
    ],
)
def test_rsna_acquisition_rejects_unsafe_or_unexpected_members(tmp_path: Path, member: str) -> None:
    def download(_command: Sequence[str], directory: Path) -> None:
        with zipfile.ZipFile(directory / "release.zip", "w") as handle:
            handle.writestr(member, b"bad")

    with pytest.raises(ManifestBuildError):
        ensure_rsna_source(tmp_path / "source", downloader=download)
    assert not (tmp_path / "source").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_rsna_acquisition_rejects_archive_symlink(tmp_path: Path) -> None:
    def download(_command: Sequence[str], directory: Path) -> None:
        link = zipfile.ZipInfo("stage_2_train_images/link.dcm")
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        with zipfile.ZipFile(directory / "release.zip", "w") as handle:
            handle.writestr(link, "target")

    with pytest.raises(ManifestBuildError, match="symbolic link"):
        ensure_rsna_source(tmp_path / "source", downloader=download)


def test_rsna_acquisition_rejects_duplicate_member(tmp_path: Path) -> None:
    def download(_command: Sequence[str], directory: Path) -> None:
        with zipfile.ZipFile(directory / "release.zip", "w") as handle:
            handle.writestr("stage_2_train_labels.csv", b"first")
            handle.writestr("stage_2_train_labels.csv", b"second")

    with pytest.warns(UserWarning), pytest.raises(ManifestBuildError, match="duplicate"):
        ensure_rsna_source(tmp_path / "source", downloader=download)


@pytest.mark.parametrize("malformed", [True, False])
def test_rsna_acquisition_rejects_invalid_release_without_installing(
    tmp_path: Path, malformed: bool
) -> None:
    def download(_command: Sequence[str], directory: Path) -> None:
        archive = directory / "release.zip"
        if malformed:
            archive.write_bytes(b"not a ZIP archive")
        else:
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("stage_2_train_labels.csv", b"patientId,Target\n")

    message = "invalid RSNA release archive" if malformed else "incomplete top-level layout"
    with pytest.raises(ManifestBuildError, match=message):
        ensure_rsna_source(tmp_path / "source", downloader=download)
    assert list(tmp_path.iterdir()) == []
