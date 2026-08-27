"""Deterministic preservation export for a completed scientific campaign."""

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
from typing import Any, overload

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.utils.publication import is_publication_staging_directory, validate_path_component

PRESERVATION_MANIFEST_SCHEMA_VERSION = 1
EXPORT_MANIFEST_FILENAME = "export-manifest.json"


@dataclass(frozen=True)
class PreservationMember:
    """One validated campaign authority and its canonical restoration target."""

    path: Path
    restore_relative: Path


@dataclass(frozen=True)
class _ValidatedManifestMember:
    archive_identity: str
    restore_relative: PurePosixPath
    kind: str
    files: tuple[tuple[PurePosixPath, str], ...]


def validate_exact_restored_closure(
    restored_root: str | Path,
    members: Sequence[PreservationMember],
) -> None:
    """Require the restored tree to contain exactly the recomputed campaign closure."""
    root = Path(restored_root).resolve()
    expected: list[tuple[Path, bool]] = []
    for member in members:
        target = (root / member.restore_relative).resolve()
        if not target.is_relative_to(root):
            raise ManifestBuildError("Restored preservation member escapes its authority root")
        if target.is_symlink() or not target.exists():
            raise ManifestBuildError("Restored preservation authority is unavailable")
        expected.append((target, target.is_dir()))
    if len({path for path, _ in expected}) != len(expected):
        raise ManifestBuildError("Restored preservation authority paths contain duplicates")
    for path in root.rglob("*"):
        resolved = path.resolve()
        if path.is_symlink() or not any(
            resolved == authority
            or (is_directory and resolved.is_relative_to(authority))
            or authority.is_relative_to(resolved)
            for authority, is_directory in expected
        ):
            raise ManifestBuildError("Restored campaign contains state outside its exact closure")


def validate_preservation_paths(
    *, sources: Sequence[str | Path], export_root: str | Path, backup_root: str | Path
) -> None:
    """Reject symlinks and every source/destination ancestor overlap."""
    export = _resolved_preservation_path(export_root)
    backup = _resolved_preservation_path(backup_root)
    if _paths_overlap(export, backup):
        raise ManifestBuildError("Export and backup destinations must be disjoint")
    resolved_sources = tuple(_resolved_preservation_path(source) for source in sources)
    for index, root in enumerate(resolved_sources):
        if _paths_overlap(export, root) or _paths_overlap(backup, root):
            raise ManifestBuildError("Preservation sources and destinations must be disjoint")
        if any(_paths_overlap(root, other) for other in resolved_sources[index + 1 :]):
            raise ManifestBuildError("Preservation sources must be mutually disjoint")


def export_and_verify(
    *,
    members: Sequence[PreservationMember],
    export_root: str | Path,
    backup_root: str | Path,
    export_name: str,
    restoration_validator: Callable[[Path], None] | None = None,
) -> Path:
    """Archive exact authorities, preserve privately, and certify standalone restoration."""
    validate_path_component(export_name, "preservation export name")
    values = tuple(members)
    if not values or any(not isinstance(member, PreservationMember) for member in values):
        raise ManifestBuildError("Preservation export members are invalid")
    validate_preservation_paths(
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
        restore_and_validate_export(temporary, restoration_validator=restoration_validator)
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
    restore_and_validate_export(backup, restoration_validator=restoration_validator)
    return archive


def _resolved_preservation_path(value: str | Path) -> Path:
    path = Path(value).absolute()
    for candidate in (path, *path.parents):
        if candidate.exists() or candidate.is_symlink():
            if candidate.is_symlink():
                raise ManifestBuildError("Preservation paths must not contain symlinks")
    return path.resolve()


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


@overload
def restore_and_validate_export(
    archive: str | Path,
    *,
    restoration_validator: None = None,
) -> None: ...


@overload
def restore_and_validate_export[RestorationResult](
    archive: str | Path,
    *,
    restoration_validator: Callable[[Path], RestorationResult],
) -> RestorationResult: ...


def restore_and_validate_export[RestorationResult](
    archive: str | Path,
    *,
    restoration_validator: Callable[[Path], RestorationResult] | None = None,
) -> RestorationResult | None:
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
                return restoration_validator(restored_root)
    return None


def _build_export_manifest(
    members: tuple[PreservationMember, ...],
) -> tuple[dict[str, object], tuple[tuple[str, Path], ...]]:
    documents: list[dict[str, object]] = []
    entries: list[tuple[str, Path]] = []
    for index, member in enumerate(members):
        root = member.path
        if root.is_symlink() or not root.exists():
            raise ManifestBuildError("Preservation export source state is invalid")
        restore = _validated_relative_path(member.restore_relative.as_posix(), "restore target")
        identity = f"member-{index:04d}"
        kind = "file" if root.is_file() else "directory" if root.is_dir() else None
        if kind is None:
            raise ManifestBuildError("Preservation export source kind is invalid")
        descendants = [] if kind == "file" else sorted(root.rglob("*"))
        stages = [path for path in descendants if is_publication_staging_directory(path)]
        descendants = [
            path for path in descendants if not any(path.is_relative_to(stage) for stage in stages)
        ]
        if any(path.is_symlink() for path in descendants):
            raise ManifestBuildError("Preservation export source contains a symlink")
        files = [root] if kind == "file" else [path for path in descendants if path.is_file()]
        if not files:
            raise ManifestBuildError("Preservation export authority directory is empty")
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
        "preservation_manifest_schema_version": PRESERVATION_MANIFEST_SCHEMA_VERSION,
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
        raise ManifestBuildError("Preservation archive membership is invalid")
    if any(info.is_dir() or _zip_entry_is_symlink(info) for info in infos):
        raise ManifestBuildError("Preservation archive contains an unsafe member")
    try:
        raw = archive.read(EXPORT_MANIFEST_FILENAME)
        document = json.loads(raw)
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Preservation manifest is unreadable") from exc
    if raw != _canonical_json_bytes(document):
        raise ManifestBuildError("Preservation manifest serialization is invalid")
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
                    "Preservation archive has missing or unexpected members"
                ) from exc
            if observed_hash != expected_hash:
                raise ManifestBuildError("Preservation archived file hash is invalid")
    if set(names) != expected_names:
        raise ManifestBuildError("Preservation archive has missing or unexpected members")
    return members


def _validate_export_manifest(document: object) -> tuple[_ValidatedManifestMember, ...]:
    schema_version = (
        document.get("preservation_manifest_schema_version") if isinstance(document, dict) else None
    )
    if (
        not isinstance(document, dict)
        or set(document) != {"preservation_manifest_schema_version", "members"}
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != PRESERVATION_MANIFEST_SCHEMA_VERSION
        or not isinstance(document["members"], list)
        or not document["members"]
    ):
        raise ManifestBuildError("Preservation manifest contract is invalid")
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
            raise ManifestBuildError("Preservation member contract is invalid")
        identity = item["archive_identity"]
        if identity != f"member-{index:04d}" or identity in archive_ids:
            raise ManifestBuildError("Preservation member identity is invalid")
        archive_ids.add(identity)
        restore = _validated_relative_path(item["restore_relative"], "restore target")
        if any(
            restore == existing
            or restore.is_relative_to(existing)
            or existing.is_relative_to(restore)
            for existing in restore_targets
        ):
            raise ManifestBuildError("Preservation restore targets collide")
        restore_targets.append(restore)
        kind = item["kind"]
        files = item["files"]
        if kind not in {"file", "directory"} or not isinstance(files, list) or not files:
            raise ManifestBuildError("Preservation member kind or files are invalid")
        validated_files: list[tuple[PurePosixPath, str]] = []
        for file_item in files:
            if not isinstance(file_item, Mapping) or set(file_item) != {"path", "sha256"}:
                raise ManifestBuildError("Preservation file contract is invalid")
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
                raise ManifestBuildError("Preservation archived file identity is invalid")
            archive_paths.add(archive_path)
            validated_files.append((relative, digest))
        if kind == "file" and (
            len(validated_files) != 1 or validated_files[0][0] != PurePosixPath(restore.name)
        ):
            raise ManifestBuildError("Preservation file member kind is inconsistent")
        if [path.as_posix() for path, _ in validated_files] != sorted(
            path.as_posix() for path, _ in validated_files
        ):
            raise ManifestBuildError("Preservation archived files are not canonically ordered")
        values.append(_ValidatedManifestMember(identity, restore, kind, tuple(validated_files)))
    return tuple(values)


def _validated_relative_path(value: object, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ManifestBuildError(f"Preservation {context} is invalid")
    path = PurePosixPath(value)
    if (
        value == "."
        or path.as_posix() != value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ManifestBuildError(f"Preservation {context} is invalid")
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
        raise ManifestBuildError("Restored preservation file hash is invalid")


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
            raise ManifestBuildError("Preservation export conflicts with existing state") from None


def _validate_private_file(path: Path, *, expected_sha256: str | None = None) -> None:
    if path.is_symlink() or not path.is_file():
        raise ManifestBuildError("Preserved archive is not a regular file")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ManifestBuildError("Preserved archive permissions are unsafe")
    if expected_sha256 is not None and _file_sha256(path) != expected_sha256:
        raise ManifestBuildError("Preserved archive hash is invalid")


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
        raise ManifestBuildError("Preservation checksum conflicts with existing state")
    _validate_private_file(path)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ManifestBuildError("Preservation manifest is not serializable") from exc


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
