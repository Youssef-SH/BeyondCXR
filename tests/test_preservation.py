from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

import beyondcxr.training.preservation as preservation
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.preservation import (
    PreservationMember,
    export_and_verify,
    restore_and_validate_export,
    validate_completed_preservation,
    validate_preservation_archive,
    validate_preservation_paths,
)


def _members(tmp_path: Path) -> tuple[PreservationMember, ...]:
    directory = tmp_path / "source"
    directory.mkdir()
    (directory / "manifest.json").write_text('{"schema_version":1}\n', encoding="utf-8")
    (directory / "artifact.bin").write_bytes(b"bounded-streaming-input" * 100)
    single = tmp_path / "test-open.json"
    single.write_text('{"schema_version":1}\n', encoding="utf-8")
    return (
        PreservationMember(directory, Path("reports/symile/result")),
        PreservationMember(single, Path("private/control/symile/test-open.json")),
    )


def _arguments(tmp_path: Path) -> dict[str, object]:
    return {
        "members": _members(tmp_path),
        "export_root": tmp_path / "export",
        "backup_root": tmp_path / "backup",
        "export_name": "campaign",
    }


@pytest.mark.parametrize(
    ("export", "backup"),
    [
        ("export", "export"),
        ("export", "export/backup"),
        ("backup/export", "backup"),
        ("source/export", "backup"),
        ("export", "source/backup"),
    ],
)
def test_export_rejects_overlapping_paths_before_writing(
    tmp_path: Path, export: str, backup: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    artifact = source / "artifact"
    artifact.write_bytes(b"synthetic")
    with pytest.raises(ManifestBuildError, match="disjoint|outside sources"):
        export_and_verify(
            members=[PreservationMember(source, Path("reports/symile/result"))],
            export_root=tmp_path / export,
            backup_root=tmp_path / backup,
            export_name="campaign",
        )
    assert set(tmp_path.rglob("*")) == {source, artifact}


def test_export_is_self_describing_private_and_byte_deterministic(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    restorations = 0

    def validate(root: Path) -> None:
        nonlocal restorations
        assert root.is_relative_to(arguments["export_root"])
        assert (root / "reports/symile/result/artifact.bin").is_file()
        assert (root / "private/control/symile/test-open.json").is_file()
        restorations += 1

    arguments["restoration_validator"] = validate
    previous_umask = os.umask(0o022)
    try:
        first = export_and_verify(**arguments)
    finally:
        os.umask(previous_umask)
    before = first.read_bytes()
    backup = tmp_path / "backup/campaign.zip"
    assert stat.S_IMODE(first.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [
            "export-manifest.json",
            "member-0000/artifact.bin",
            "member-0000/manifest.json",
            "member-0001/test-open.json",
        ]
        manifest = json.loads(archive.read("export-manifest.json"))
        assert manifest["preservation_manifest_schema_version"] == 1
        assert manifest["members"][0]["restore_relative"] == "reports/symile/result"
        assert manifest["members"][0]["kind"] == "directory"
        assert manifest["members"][1]["restore_relative"] == (
            "private/control/symile/test-open.json"
        )
        assert manifest["members"][1]["kind"] == "file"
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
    assert export_and_verify(**arguments).read_bytes() == before
    assert restorations == 2
    assert {path.name for path in first.parent.iterdir()} == {"campaign.zip", "campaign.zip.sha256"}

    os.chmod(backup, 0o644)
    with pytest.raises(ManifestBuildError, match="permissions are unsafe"):
        export_and_verify(**arguments)
    assert backup.read_bytes() == before


def test_export_canonicalizes_nested_files_by_archive_path(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "a").mkdir(parents=True)
    (source / "a.json").write_text("sibling\n", encoding="utf-8")
    (source / "a/manifest.json").write_text("nested\n", encoding="utf-8")

    archive_path = export_and_verify(
        members=[PreservationMember(source, Path("reports/symile/result"))],
        export_root=tmp_path / "export",
        backup_root=tmp_path / "backup",
        export_name="canonical-order",
    )

    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("export-manifest.json"))
        assert [item["path"] for item in manifest["members"][0]["files"]] == [
            "a.json",
            "a/manifest.json",
        ]
        assert archive.namelist() == [
            "export-manifest.json",
            "member-0000/a.json",
            "member-0000/a/manifest.json",
        ]


@pytest.mark.parametrize(
    ("source", "export", "backup"),
    [
        ("export/source", "export", "backup"),
        ("backup/source", "export", "backup"),
        ("source", "export", "export/backup"),
        ("source", "backup/export", "backup"),
    ],
)
def test_path_validation_rejects_overlap_in_both_directions(
    tmp_path: Path, source: str, export: str, backup: str
) -> None:
    with pytest.raises(ManifestBuildError, match="disjoint"):
        validate_preservation_paths(
            sources=[tmp_path / source],
            export_root=tmp_path / export,
            backup_root=tmp_path / backup,
        )


def test_path_validation_rejects_symlinked_components(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ManifestBuildError, match="symlinks"):
        validate_preservation_paths(
            sources=[tmp_path / "source"],
            export_root=link / "export",
            backup_root=tmp_path / "backup",
        )


def test_path_validation_rejects_overlapping_source_members(tmp_path: Path) -> None:
    with pytest.raises(ManifestBuildError, match="mutually disjoint"):
        validate_preservation_paths(
            sources=[tmp_path / "source", tmp_path / "source/nested"],
            export_root=tmp_path / "export",
            backup_root=tmp_path / "backup",
        )


def test_identical_secure_backup_is_reused_and_symlink_is_rejected(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    archive = export_and_verify(**arguments)
    backup = tmp_path / "backup/campaign.zip"
    inode = backup.stat().st_ino
    assert export_and_verify(**arguments).read_bytes() == archive.read_bytes()
    assert backup.stat().st_ino == inode

    other_arguments = {
        **arguments,
        "export_root": tmp_path / "other-export",
        "backup_root": tmp_path / "other-backup",
    }
    (tmp_path / "other-backup").mkdir()
    (tmp_path / "other-backup/campaign.zip").symlink_to(backup)
    with pytest.raises(ManifestBuildError, match="not a regular file"):
        export_and_verify(**other_arguments)


def test_completed_export_is_validated_and_reused_without_rebuilding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)
    archive = export_and_verify(**arguments)
    expected = archive.read_bytes()

    def reject_rebuild(_members: object) -> None:
        raise AssertionError("completed preservation must not be rebuilt")

    monkeypatch.setattr(preservation, "_build_export_manifest", reject_rebuild)
    assert export_and_verify(**arguments).read_bytes() == expected


@pytest.mark.parametrize("target", ["archive", "checksum", "backup"])
def test_completed_export_rejects_conflicting_canonical_state(tmp_path: Path, target: str) -> None:
    arguments = _arguments(tmp_path)
    archive = export_and_verify(**arguments)
    paths = {
        "archive": archive,
        "checksum": archive.with_suffix(".zip.sha256"),
        "backup": tmp_path / "backup/campaign.zip",
    }
    paths[target].write_bytes(b"conflicting completed state")
    os.chmod(paths[target], 0o600)

    with pytest.raises(ManifestBuildError):
        export_and_verify(**arguments)


def test_completed_preservation_validates_sidecar_backup_and_restored_science(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    archive = export_and_verify(**arguments)
    observed: list[bytes] = []

    validate_completed_preservation(
        archive,
        backup_root=tmp_path / "backup",
        restoration_validator=lambda root: observed.append(
            (root / "reports/symile/result/artifact.bin").read_bytes()
        ),
    )
    assert observed == [b"bounded-streaming-input" * 100]
    assert validate_preservation_archive(archive) is None


def test_export_certifies_before_publishing_preservation_files(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)

    def reject_restoration(root: Path) -> None:
        assert (root / "reports/symile/result/manifest.json").is_file()
        raise ManifestBuildError("simulated restoration certification failure")

    arguments["restoration_validator"] = reject_restoration
    with pytest.raises(ManifestBuildError, match="certification failure"):
        export_and_verify(**arguments)

    assert not (tmp_path / "export/campaign.zip").exists()
    assert not (tmp_path / "export/campaign.zip.sha256").exists()
    assert not (tmp_path / "backup/campaign.zip").exists()
    assert not tuple((tmp_path / "export").glob(".campaign.zip.*.tmp"))


def test_export_rejects_unsafe_existing_checksum_permissions(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    archive = export_and_verify(**arguments)
    checksum = archive.with_suffix(".zip.sha256")
    expected = checksum.read_bytes()
    os.chmod(checksum, 0o644)

    with pytest.raises(ManifestBuildError, match="permissions are unsafe"):
        export_and_verify(**arguments)
    assert checksum.read_bytes() == expected


def _valid_manifest() -> dict[str, object]:
    return {
        "preservation_manifest_schema_version": 1,
        "members": [
            {
                "archive_identity": "member-0000",
                "restore_relative": "reports/symile/result",
                "kind": "directory",
                "role": "scientific_authority",
                "files": [
                    {"path": "manifest.json", "sha256": hashlib.sha256(b"content").hexdigest()}
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_version",
        "boolean_version",
        "float_version",
        "missing_field",
        "extra_field",
        "duplicate_restore",
        "absolute_restore",
        "traversal_restore",
        "unsupported_kind",
        "malformed_hash",
        "duplicate_archive_path",
        "file_kind_mismatch",
    ],
)
def test_export_manifest_rejects_hostile_contracts(mutation: str) -> None:
    document = _valid_manifest()
    member = document["members"][0]
    if mutation == "wrong_version":
        document["preservation_manifest_schema_version"] = 2
    elif mutation == "boolean_version":
        document["preservation_manifest_schema_version"] = True
    elif mutation == "float_version":
        document["preservation_manifest_schema_version"] = 1.0
    elif mutation == "missing_field":
        member.pop("files")
    elif mutation == "extra_field":
        member["unexpected"] = True
    elif mutation == "duplicate_restore":
        document["members"].append({**deepcopy(member), "archive_identity": "member-0001"})
    elif mutation == "absolute_restore":
        member["restore_relative"] = "/private"
    elif mutation == "traversal_restore":
        member["restore_relative"] = "../private"
    elif mutation == "unsupported_kind":
        member["kind"] = "symlink"
    elif mutation == "malformed_hash":
        member["files"][0]["sha256"] = "bad"
    elif mutation == "duplicate_archive_path":
        member["files"].append(deepcopy(member["files"][0]))
    else:
        member["kind"] = "file"
    with pytest.raises(ManifestBuildError):
        preservation._validate_export_manifest(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("restore", "."),
        ("restore", "./reports/symile/result"),
        ("restore", "reports//symile/result"),
        ("restore", "reports/./symile/result"),
        ("restore", "reports/symile/result/"),
        ("archive", "nested//manifest.json"),
    ],
)
def test_export_manifest_rejects_textually_noncanonical_paths(field: str, value: str) -> None:
    document = _valid_manifest()
    member = document["members"][0]
    if field == "restore":
        member["restore_relative"] = value
    else:
        member["files"][0]["path"] = value

    with pytest.raises(ManifestBuildError, match="is invalid"):
        preservation._validate_export_manifest(document)


def test_export_file_member_requires_exact_standalone_basename() -> None:
    document = _valid_manifest()
    member = document["members"][0]
    member["restore_relative"] = "private/control/test-open.json"
    member["kind"] = "file"
    member["files"][0]["path"] = "nested/test-open.json"

    with pytest.raises(ManifestBuildError, match="file member kind is inconsistent"):
        preservation._validate_export_manifest(document)


def _write_archive(
    path: Path, document: dict[str, object], entries: list[tuple[str, bytes]]
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        preservation._write_zip_bytes(
            archive,
            "export-manifest.json",
            preservation._canonical_json_bytes(document),
        )
        for name, content in entries:
            preservation._write_zip_bytes(archive, name, content)
    os.chmod(path, 0o600)


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "wrong_hash", "duplicate"])
@pytest.mark.filterwarnings("ignore:Duplicate name:UserWarning")
def test_standalone_restore_rejects_hostile_zip_membership(tmp_path: Path, mutation: str) -> None:
    document = _valid_manifest()
    entries = [("member-0000/manifest.json", b"content")]
    if mutation == "missing":
        entries = []
    elif mutation == "unexpected":
        entries.append(("untracked", b"content"))
    elif mutation == "wrong_hash":
        entries[0] = (entries[0][0], b"changed")
    elif mutation == "duplicate":
        entries.append(entries[0])
    archive = tmp_path / "hostile.zip"
    _write_archive(archive, document, entries)
    with pytest.raises(ManifestBuildError):
        restore_and_validate_export(archive)


def test_standalone_restore_rejects_noncanonical_zip_member_order(tmp_path: Path) -> None:
    document = _valid_manifest()
    archive_path = tmp_path / "reordered.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        preservation._write_zip_bytes(archive, "member-0000/manifest.json", b"content")
        preservation._write_zip_bytes(
            archive,
            "export-manifest.json",
            preservation._canonical_json_bytes(document),
        )
    os.chmod(archive_path, 0o600)

    with pytest.raises(ManifestBuildError, match="not canonically ordered"):
        restore_and_validate_export(archive_path)


def test_standalone_restore_rejects_noncanonical_zip_metadata(tmp_path: Path) -> None:
    document = _valid_manifest()
    archive_path = tmp_path / "noncanonical-metadata.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        preservation._write_zip_bytes(
            archive,
            "export-manifest.json",
            preservation._canonical_json_bytes(document),
        )
        info = preservation._zip_file_info("member-0000/manifest.json")
        info.date_time = (1981, 1, 1, 0, 0, 0)
        archive.writestr(info, b"content")
    os.chmod(archive_path, 0o600)

    with pytest.raises(ManifestBuildError, match="metadata is not canonical"):
        restore_and_validate_export(archive_path)


@pytest.mark.parametrize("content", [None, b"corrupted derivative"])
def test_standalone_restore_rejects_missing_or_corrupt_derivative(
    tmp_path: Path, content: bytes | None
) -> None:
    document = _valid_manifest()
    document["members"][0]["role"] = "derivative"
    entries = [] if content is None else [("member-0000/manifest.json", content)]
    archive = tmp_path / "derivative.zip"
    _write_archive(archive, document, entries)

    with pytest.raises(ManifestBuildError):
        restore_and_validate_export(archive)


def test_standalone_restore_uses_only_backup_manifest_after_sources_are_deleted(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    export_and_verify(**arguments)
    backup = tmp_path / "backup/campaign.zip"
    renamed = tmp_path / "renamed-valid-campaign.zip"
    shutil.copy2(backup, renamed)
    for member in arguments["members"]:
        shutil.rmtree(member.path) if member.path.is_dir() else member.path.unlink()

    def validate(root: Path) -> dict[str, bytes]:
        return {
            "artifact": (root / "reports/symile/result/artifact.bin").read_bytes(),
            "open": (root / "private/control/symile/test-open.json").read_bytes(),
        }

    observed = restore_and_validate_export(backup, restoration_validator=validate)
    assert observed == {
        "artifact": b"bounded-streaming-input" * 100,
        "open": b'{"schema_version":1}\n',
    }
    assert restore_and_validate_export(renamed, restoration_validator=validate) == observed


def test_restoration_context_exists_only_inside_authenticated_archive_validation(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    archive = export_and_verify(**arguments)
    captured = []

    def validate(root: Path) -> None:
        context = preservation._validated_restoration_context(root)
        assert preservation.require_restoration_context(context) is context
        captured.append(context)

    restore_and_validate_export(archive, restoration_validator=validate)
    with pytest.raises(PermissionError, match="authenticated restoration context"):
        preservation.require_restoration_context(captured[0])
    with pytest.raises(PermissionError, match="authenticated restoration context"):
        preservation._validated_restoration_context(tmp_path)


def test_export_cleans_temporary_file_after_archive_install_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)

    def fail_archive_install(temporary: Path, destination: Path) -> None:
        del temporary, destination
        raise OSError("simulated archive publication failure")

    monkeypatch.setattr(preservation, "_install_or_validate", fail_archive_install)
    with pytest.raises(OSError, match="simulated archive publication failure"):
        export_and_verify(**arguments)
    export_root = tmp_path / "export"
    assert not (export_root / "campaign.zip").exists()
    assert not tuple(export_root.glob(".campaign.zip.*.tmp"))


def test_late_checksum_failure_rolls_back_and_allows_clean_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)
    real_write = preservation._write_or_validate

    def fail_checksum(path: Path, content: bytes) -> bool:
        del path, content
        raise OSError("simulated checksum publication failure")

    monkeypatch.setattr(preservation, "_write_or_validate", fail_checksum)
    with pytest.raises(OSError, match="simulated checksum publication failure"):
        export_and_verify(**arguments)

    archive = tmp_path / "export/campaign.zip"
    checksum = archive.with_suffix(".zip.sha256")
    backup = tmp_path / "backup/campaign.zip"
    assert not archive.exists()
    assert not checksum.exists()
    assert not backup.exists()
    assert not tuple(tmp_path.rglob(".*.tmp"))

    monkeypatch.setattr(preservation, "_write_or_validate", real_write)
    published = export_and_verify(**arguments)
    assert published == archive
    assert checksum.is_file()
    assert backup.read_bytes() == archive.read_bytes()
    assert validate_completed_preservation(archive, backup_root=tmp_path / "backup") == archive


def test_terminal_validation_failure_rolls_back_only_attempt_created_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)
    export_root = tmp_path / "export"
    backup_root = tmp_path / "backup"
    export_root.mkdir()
    backup_root.mkdir()
    export_marker = export_root / "pre-existing.txt"
    backup_marker = backup_root / "pre-existing.txt"
    export_marker.write_bytes(b"keep export state")
    backup_marker.write_bytes(b"keep backup state")

    def fail_completed_validation(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        raise ManifestBuildError("simulated terminal preservation failure")

    monkeypatch.setattr(
        preservation,
        "validate_completed_preservation",
        fail_completed_validation,
    )
    with pytest.raises(ManifestBuildError, match="simulated terminal preservation failure"):
        export_and_verify(**arguments)

    archive = export_root / "campaign.zip"
    assert not archive.exists()
    assert not archive.with_suffix(".zip.sha256").exists()
    assert not (backup_root / "campaign.zip").exists()
    assert export_marker.read_bytes() == b"keep export state"
    assert backup_marker.read_bytes() == b"keep backup state"
    assert not tuple(tmp_path.rglob(".*.tmp"))


@pytest.mark.parametrize("missing", [("checksum",), ("backup",), ("checksum", "backup")])
def test_interrupted_installed_archive_recovers_missing_transport_files(
    tmp_path: Path, missing: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)
    archive = export_and_verify(**arguments)
    expected = archive.read_bytes()
    paths = {
        "checksum": archive.with_suffix(".zip.sha256"),
        "backup": tmp_path / "backup/campaign.zip",
    }
    for name in missing:
        paths[name].unlink()

    def forbid_rebuild(_members: object) -> None:
        raise AssertionError("Recovery must retain the installed archive")

    monkeypatch.setattr(preservation, "_build_export_manifest", forbid_rebuild)
    validated = []
    arguments["restoration_validator"] = lambda root: validated.append(root)
    assert export_and_verify(**arguments) == archive
    assert validated
    assert archive.read_bytes() == expected == paths["backup"].read_bytes()
    validate_completed_preservation(archive, backup_root=tmp_path / "backup")


def test_interrupted_archive_requires_valid_science_before_repair(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    archive = export_and_verify(**arguments)
    checksum = archive.with_suffix(".zip.sha256")
    backup = tmp_path / "backup/campaign.zip"
    checksum.unlink()
    backup.unlink()

    def reject(_root: Path) -> None:
        raise ManifestBuildError("invalid restored science")

    arguments["restoration_validator"] = reject
    with pytest.raises(ManifestBuildError, match="invalid restored science"):
        export_and_verify(**arguments)
    assert archive.exists()
    assert not checksum.exists()
    assert not backup.exists()
