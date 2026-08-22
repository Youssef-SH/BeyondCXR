"""Deterministic preservation export for a completed Symile campaign."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from radfusion.data.errors import ManifestBuildError
from radfusion.utils.publication import is_publication_staging_directory, validate_path_component

SYMILE_EXPORT_MANIFEST_SCHEMA_VERSION = 1
EXPORT_MANIFEST_FILENAME = "export-manifest.json"


@dataclass(frozen=True)
class SymileExportMember:
    """One validated campaign authority and its canonical restoration target."""

    path: Path
    restore_relative: Path


@dataclass(frozen=True)
class _ValidatedManifestMember:
    archive_identity: str
    restore_relative: PurePosixPath
    kind: str
    files: tuple[tuple[PurePosixPath, str], ...]


def validate_export_paths(
    *, sources: Sequence[str | Path], export_root: str | Path, backup_root: str | Path
) -> None:
    """Reject self-including snapshots and overlapping preservation destinations."""
    export = Path(export_root).resolve()
    backup = Path(backup_root).resolve()
    if export.is_relative_to(backup) or backup.is_relative_to(export):
        raise ManifestBuildError("Symile export and backup destinations must be disjoint")
    for source in sources:
        root = Path(source).resolve()
        if export.is_relative_to(root) or backup.is_relative_to(root):
            raise ManifestBuildError("Symile preservation destinations must be outside sources")


def export_and_verify(
    *,
    members: Sequence[SymileExportMember],
    export_root: str | Path,
    backup_root: str | Path,
    export_name: str,
    restoration_validator: Callable[[Path], None] | None = None,
) -> Path:
    """Archive exact authorities, preserve privately, and certify standalone restoration."""
    validate_path_component(export_name, "Symile export name")
    values = tuple(members)
    if not values or any(not isinstance(member, SymileExportMember) for member in values):
        raise ManifestBuildError("Symile export members are invalid")
    validate_export_paths(
        sources=[member.path for member in values],
        export_root=export_root,
        backup_root=backup_root,
    )
    document, entries = _build_export_manifest(values)
    encoded_manifest = _canonical_json_bytes(document)
    export_directory = Path(export_root)
    export_directory.mkdir(parents=True, exist_ok=True)
    archive = export_directory / f"{export_name}.zip"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive.name}.", suffix=".tmp", dir=export_directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as output:
            _write_zip_bytes(output, EXPORT_MANIFEST_FILENAME, encoded_manifest)
            for name, path in entries:
                info = _zip_file_info(name)
                with path.open("rb") as source, output.open(info, "w") as destination:
                    shutil.copyfileobj(source, destination)
        digest = _file_sha256(temporary)
        _validate_private_file(temporary, expected_sha256=digest)
        restore_and_validate_symile_export(temporary, restoration_validator=restoration_validator)
        _install_or_validate(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    _validate_private_file(archive, expected_sha256=digest)
    checksum = archive.with_suffix(".zip.sha256")
    _write_or_validate(checksum, f"{digest}  {archive.name}\n".encode())
    backup_directory = Path(backup_root)
    backup_directory.mkdir(parents=True, exist_ok=True)
    backup = backup_directory / archive.name
    descriptor, backup_temporary_name = tempfile.mkstemp(
        prefix=f".{backup.name}.", suffix=".tmp", dir=backup_directory
    )
    backup_temporary = Path(backup_temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination, archive.open("rb") as source:
            shutil.copyfileobj(source, destination)
        _install_or_validate(backup_temporary, backup)
    finally:
        backup_temporary.unlink(missing_ok=True)
    _validate_private_file(backup, expected_sha256=digest)
    return archive


def restore_and_validate_symile_export(
    archive: str | Path,
    *,
    restoration_validator: Callable[[Path], None] | None = None,
) -> None:
    """Restore a campaign using only its archive-owned V1 manifest and validate it."""
    source = Path(archive)
    _validate_private_file(source)
    with zipfile.ZipFile(source) as input_archive:
        members = _read_and_validate_archive(input_archive)
        with tempfile.TemporaryDirectory() as temporary_directory:
            restored_root = Path(temporary_directory) / "restored"
            restored_root.mkdir()
            for member in members:
                target = restored_root.joinpath(*member.restore_relative.parts)
                if member.kind == "directory":
                    target.mkdir(parents=True)
                    for relative, expected_hash in member.files:
                        _restore_archived_file(
                            input_archive,
                            f"{member.archive_identity}/{relative.as_posix()}",
                            target.joinpath(*relative.parts),
                            expected_hash,
                        )
                else:
                    relative, expected_hash = member.files[0]
                    _restore_archived_file(
                        input_archive,
                        f"{member.archive_identity}/{relative.as_posix()}",
                        target,
                        expected_hash,
                    )
            if restoration_validator is not None:
                restoration_validator(restored_root)


def _build_export_manifest(
    members: tuple[SymileExportMember, ...],
) -> tuple[dict[str, object], tuple[tuple[str, Path], ...]]:
    documents: list[dict[str, object]] = []
    entries: list[tuple[str, Path]] = []
    for index, member in enumerate(members):
        root = member.path
        if root.is_symlink() or not root.exists():
            raise ManifestBuildError("Symile export source state is invalid")
        restore = _validated_relative_path(member.restore_relative.as_posix(), "restore target")
        identity = f"member-{index:04d}"
        kind = "file" if root.is_file() else "directory" if root.is_dir() else None
        if kind is None:
            raise ManifestBuildError("Symile export source kind is invalid")
        descendants = [] if kind == "file" else sorted(root.rglob("*"))
        stages = [path for path in descendants if is_publication_staging_directory(path)]
        descendants = [
            path for path in descendants if not any(path.is_relative_to(stage) for stage in stages)
        ]
        if any(path.is_symlink() for path in descendants):
            raise ManifestBuildError("Symile export source contains a symlink")
        files = [root] if kind == "file" else [path for path in descendants if path.is_file()]
        if not files:
            raise ManifestBuildError("Symile export authority directory is empty")
        file_documents = []
        for path in files:
            relative = Path(root.name) if kind == "file" else path.relative_to(root)
            relative_posix = relative.as_posix()
            digest = _file_sha256(path)
            file_documents.append({"path": relative_posix, "sha256": digest})
            entries.append((f"{identity}/{relative_posix}", path))
        documents.append(
            {
                "archive_identity": identity,
                "restore_relative": restore.as_posix(),
                "kind": kind,
                "files": file_documents,
            }
        )
    document = {
        "symile_export_manifest_schema_version": SYMILE_EXPORT_MANIFEST_SCHEMA_VERSION,
        "members": documents,
    }
    _validate_export_manifest(document)
    return document, tuple(entries)


def _read_and_validate_archive(
    archive: zipfile.ZipFile,
) -> tuple[_ValidatedManifestMember, ...]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or names.count(EXPORT_MANIFEST_FILENAME) != 1:
        raise ManifestBuildError("Symile export archive membership is invalid")
    if any(info.is_dir() or _zip_entry_is_symlink(info) for info in infos):
        raise ManifestBuildError("Symile export archive contains an unsafe member")
    try:
        raw = archive.read(EXPORT_MANIFEST_FILENAME)
        document = json.loads(raw)
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Symile export manifest is unreadable") from exc
    if raw != _canonical_json_bytes(document):
        raise ManifestBuildError("Symile export manifest serialization is invalid")
    members = _validate_export_manifest(document)
    expected_names = {EXPORT_MANIFEST_FILENAME}
    for member in members:
        for relative, expected_hash in member.files:
            name = f"{member.archive_identity}/{relative.as_posix()}"
            expected_names.add(name)
            try:
                observed_hash = _zip_member_sha256(archive, name)
            except KeyError as exc:
                raise ManifestBuildError(
                    "Symile export archive has missing or unexpected members"
                ) from exc
            if observed_hash != expected_hash:
                raise ManifestBuildError("Symile export archived file hash is invalid")
    if set(names) != expected_names:
        raise ManifestBuildError("Symile export archive has missing or unexpected members")
    return members


def _validate_export_manifest(document: object) -> tuple[_ValidatedManifestMember, ...]:
    schema_version = (
        document.get("symile_export_manifest_schema_version")
        if isinstance(document, dict)
        else None
    )
    if (
        not isinstance(document, dict)
        or set(document) != {"symile_export_manifest_schema_version", "members"}
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SYMILE_EXPORT_MANIFEST_SCHEMA_VERSION
        or not isinstance(document["members"], list)
        or not document["members"]
    ):
        raise ManifestBuildError("Symile export manifest contract is invalid")
    values: list[_ValidatedManifestMember] = []
    archive_ids: set[str] = set()
    restore_targets: list[PurePosixPath] = []
    archive_paths: set[str] = set()
    for index, item in enumerate(document["members"]):
        if not isinstance(item, Mapping) or set(item) != {
            "archive_identity",
            "restore_relative",
            "kind",
            "files",
        }:
            raise ManifestBuildError("Symile export member contract is invalid")
        identity = item["archive_identity"]
        if identity != f"member-{index:04d}" or identity in archive_ids:
            raise ManifestBuildError("Symile export member identity is invalid")
        archive_ids.add(identity)
        restore = _validated_relative_path(item["restore_relative"], "restore target")
        if any(
            restore == existing
            or restore.is_relative_to(existing)
            or existing.is_relative_to(restore)
            for existing in restore_targets
        ):
            raise ManifestBuildError("Symile export restore targets collide")
        restore_targets.append(restore)
        kind = item["kind"]
        files = item["files"]
        if kind not in {"file", "directory"} or not isinstance(files, list) or not files:
            raise ManifestBuildError("Symile export member kind or files are invalid")
        validated_files: list[tuple[PurePosixPath, str]] = []
        for file_item in files:
            if not isinstance(file_item, Mapping) or set(file_item) != {"path", "sha256"}:
                raise ManifestBuildError("Symile export file contract is invalid")
            relative = _validated_relative_path(file_item["path"], "archived file")
            digest = file_item["sha256"]
            archive_path = f"{identity}/{relative.as_posix()}"
            if (
                archive_path in archive_paths
                or any(
                    relative.is_relative_to(existing) or existing.is_relative_to(relative)
                    for existing, _ in validated_files
                )
                or not _valid_sha256(digest)
            ):
                raise ManifestBuildError("Symile export archived file identity is invalid")
            archive_paths.add(archive_path)
            validated_files.append((relative, digest))
        if kind == "file" and (
            len(validated_files) != 1 or validated_files[0][0] != PurePosixPath(restore.name)
        ):
            raise ManifestBuildError("Symile export file member kind is inconsistent")
        if [path.as_posix() for path, _ in validated_files] != sorted(
            path.as_posix() for path, _ in validated_files
        ):
            raise ManifestBuildError("Symile export archived files are not canonically ordered")
        values.append(_ValidatedManifestMember(identity, restore, kind, tuple(validated_files)))
    return tuple(values)


def _validated_relative_path(value: object, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ManifestBuildError(f"Symile export {context} is invalid")
    path = PurePosixPath(value)
    if (
        value == "."
        or path.as_posix() != value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ManifestBuildError(f"Symile export {context} is invalid")
    return path


def _restore_archived_file(
    archive: zipfile.ZipFile, name: str, destination: Path, expected_hash: str
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "wb") as output, archive.open(name) as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != expected_hash:
        raise ManifestBuildError("Restored Symile export file hash is invalid")


def _zip_file_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _write_zip_bytes(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    archive.writestr(_zip_file_info(name), content)


def _zip_entry_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return info.create_system == 3 and stat.S_ISLNK(mode)


def _install_or_validate(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if (
            destination.is_symlink()
            or not destination.is_file()
            or temporary.stat().st_size != destination.stat().st_size
            or _file_sha256(temporary) != _file_sha256(destination)
        ):
            raise ManifestBuildError(
                "Symile export conflicts with existing preserved state"
            ) from None


def _validate_private_file(path: Path, *, expected_sha256: str | None = None) -> None:
    if path.is_symlink() or not path.is_file():
        raise ManifestBuildError("Symile preserved archive is not a regular file")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ManifestBuildError("Symile preserved archive permissions are unsafe")
    if expected_sha256 is not None and _file_sha256(path) != expected_sha256:
        raise ManifestBuildError("Symile preserved archive hash is invalid")


def _write_or_validate(path: Path, content: bytes) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
            _install_or_validate(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
        raise ManifestBuildError("Symile export checksum conflicts with existing state")
    _validate_private_file(path)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ManifestBuildError("Symile export manifest is not serializable") from exc


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_member_sha256(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
