"""Safe, optional acquisition of the public RSNA Stage 2 release."""

from __future__ import annotations

import shutil
import stat
import subprocess
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.rsna_source import RsnaPaths, discover_dicoms, load_source_samples
from beyondcxr.utils.publication import install_immutable_directory, staging_directory

RSNA_KAGGLE_COMPETITION = "rsna-pneumonia-detection-challenge"
_REQUIRED_TOP_LEVEL = {
    "stage_2_train_labels.csv",
    "stage_2_detailed_class_info.csv",
    "stage_2_sample_submission.csv",
    "stage_2_train_images",
    "stage_2_test_images",
}
_MAX_ARCHIVE_MEMBERS = 40_000
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 20 * 1024**3


def ensure_rsna_source(
    source_root: str | Path,
    *,
    downloader: Callable[[Sequence[str], Path], None] | None = None,
) -> Path:
    """Validate an existing source or acquire, stage, validate, and install it once."""
    destination = Path(source_root).absolute()
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ManifestBuildError("RSNA source path must not contain a symlink")
    destination = destination.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        _validate_rsna_source(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ManifestBuildError("RSNA source parent must not be a symlink")
    stage = staging_directory(destination)
    try:
        download_directory = stage / "download"
        extracted = stage / "extracted"
        download_directory.mkdir()
        extracted.mkdir()
        command = (
            "kaggle",
            "competitions",
            "download",
            "--competition",
            RSNA_KAGGLE_COMPETITION,
            "--path",
            download_directory.as_posix(),
        )
        (downloader or _download_with_kaggle)(command, download_directory)
        archives = tuple(
            path
            for path in download_directory.iterdir()
            if not path.is_symlink() and path.is_file() and path.suffix.lower() == ".zip"
        )
        if len(archives) != 1:
            raise ManifestBuildError("Kaggle acquisition did not produce one RSNA release archive")
        _extract_release(archives[0], extracted)
        _validate_rsna_source(extracted, require_complete_release=True)

        def validate_acquired_release(
            path: str | Path, *, enforce_directory_name: bool = True
        ) -> None:
            _validate_rsna_source(
                path,
                require_complete_release=True,
                enforce_directory_name=enforce_directory_name,
            )

        install_immutable_directory(extracted, destination, validate_acquired_release)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return destination


def _download_with_kaggle(command: Sequence[str], _directory: Path) -> None:
    try:
        subprocess.run(list(command), check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise ManifestBuildError(
            "Kaggle acquisition is unavailable; run `make setup` and configure Kaggle credentials"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ManifestBuildError(
            "Kaggle acquisition failed; verify Kaggle credentials and competition access"
        ) from exc


def _extract_release(archive: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(archive) as handle:
            members = handle.infolist()
            if (
                len(members) > _MAX_ARCHIVE_MEMBERS
                or sum(member.file_size for member in members) > _MAX_ARCHIVE_UNCOMPRESSED_BYTES
            ):
                raise ManifestBuildError("RSNA archive exceeds the expected release bounds")
            observed: set[str] = set()
            for member in members:
                name = member.filename
                path = PurePosixPath(name)
                if (
                    not name
                    or "\\" in name
                    or path.is_absolute()
                    or ".." in path.parts
                    or path.as_posix() != name.rstrip("/")
                ):
                    raise ManifestBuildError("RSNA archive contains an unsafe member path")
                normalized = path.as_posix()
                if normalized in observed:
                    raise ManifestBuildError("RSNA archive contains a duplicate member path")
                observed.add(normalized)
                mode = (member.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise ManifestBuildError("RSNA archive contains a symbolic link")
                if not _is_expected_release_member(member, path):
                    raise ManifestBuildError("RSNA archive contains an unexpected release member")
            for member in members:
                relative = PurePosixPath(member.filename)
                target = destination.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    except zipfile.BadZipFile as exc:
        raise ManifestBuildError("Kaggle produced an invalid RSNA release archive") from exc


def _is_expected_release_member(member: zipfile.ZipInfo, path: PurePosixPath) -> bool:
    top_level = path.parts[0]
    if top_level not in _REQUIRED_TOP_LEVEL:
        return False
    if top_level in {"stage_2_train_images", "stage_2_test_images"}:
        if len(path.parts) == 1:
            return member.is_dir()
        return len(path.parts) == 2 and not member.is_dir() and path.suffix.lower() == ".dcm"
    return len(path.parts) == 1 and not member.is_dir()


def _validate_rsna_source(
    source_root: str | Path,
    *,
    require_complete_release: bool = False,
    enforce_directory_name: bool = True,
) -> None:
    del enforce_directory_name
    root = Path(source_root)
    if root.is_symlink() or not root.is_dir():
        raise ManifestBuildError("RSNA source root is not a regular directory")
    if require_complete_release:
        required_files = (
            "stage_2_train_labels.csv",
            "stage_2_detailed_class_info.csv",
            "stage_2_sample_submission.csv",
        )
        required_directories = ("stage_2_train_images", "stage_2_test_images")
        if (
            any(not (root / name).is_file() for name in required_files)
            or any(not (root / name).is_dir() for name in required_directories)
            or any((root / name).is_symlink() for name in _REQUIRED_TOP_LEVEL)
        ):
            raise ManifestBuildError("RSNA release archive has an incomplete top-level layout")
    paths = RsnaPaths.from_root(root)
    paths.validate()
    samples = load_source_samples(paths)
    dicoms = discover_dicoms(paths.images)
    if set(samples) != set(dicoms):
        raise ManifestBuildError("RSNA source labels and DICOM membership differ")
