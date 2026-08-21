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

import radfusion.training.symile_export as symile_export
from radfusion.data.errors import ManifestBuildError
from radfusion.training.symile_export import (
    SymileExportMember,
    export_and_verify,
    restore_and_validate_symile_export,
)


def _members(tmp_path: Path) -> tuple[SymileExportMember, ...]:
    directory = tmp_path / "source"
    directory.mkdir()
    (directory / "manifest.json").write_text('{"schema_version":1}\n', encoding="utf-8")
    (directory / "artifact.bin").write_bytes(b"bounded-streaming-input" * 100)
    single = tmp_path / "test-open.json"
    single.write_text('{"test_open_schema_version":1}\n', encoding="utf-8")
    return (
        SymileExportMember(directory, Path("reports/symile/result")),
        SymileExportMember(single, Path("private/control/symile/test-open.json")),
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
            members=[SymileExportMember(source, Path("reports/symile/result"))],
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
        assert manifest["symile_export_manifest_schema_version"] == 1
        assert manifest["members"][0]["restore_relative"] == "reports/symile/result"
        assert manifest["members"][0]["kind"] == "directory"
        assert manifest["members"][1]["restore_relative"] == (
            "private/control/symile/test-open.json"
        )
        assert manifest["members"][1]["kind"] == "file"
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
    assert export_and_verify(**arguments).read_bytes() == before
    assert restorations == 2

    os.chmod(backup, 0o644)
    with pytest.raises(ManifestBuildError, match="permissions are unsafe"):
        export_and_verify(**arguments)
    assert backup.read_bytes() == before


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
    with pytest.raises(ManifestBuildError, match="conflicts with existing preserved state"):
        export_and_verify(**other_arguments)


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
        "symile_export_manifest_schema_version": 1,
        "members": [
            {
                "archive_identity": "member-0000",
                "restore_relative": "reports/symile/result",
                "kind": "directory",
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
        document["symile_export_manifest_schema_version"] = 2
    elif mutation == "boolean_version":
        document["symile_export_manifest_schema_version"] = True
    elif mutation == "float_version":
        document["symile_export_manifest_schema_version"] = 1.0
    elif mutation == "missing_field":
        member.pop("files")
    elif mutation == "extra_field":
        member["legacy"] = True
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
        symile_export._validate_export_manifest(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("restore", "."),
        ("restore", "reports//symile/result"),
        ("restore", "reports/./symile/result"),
        ("restore", "reports/symile/result/"),
        ("archive", "."),
        ("archive", "nested//manifest.json"),
        ("archive", "nested/./manifest.json"),
        ("archive", "nested/manifest.json/"),
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
        symile_export._validate_export_manifest(document)


def test_export_file_member_requires_exact_standalone_basename() -> None:
    document = _valid_manifest()
    member = document["members"][0]
    member["restore_relative"] = "private/control/test-open.json"
    member["kind"] = "file"
    member["files"][0]["path"] = "nested/test-open.json"

    with pytest.raises(ManifestBuildError, match="file member kind is inconsistent"):
        symile_export._validate_export_manifest(document)


def _write_archive(
    path: Path, document: dict[str, object], entries: list[tuple[str, bytes]]
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        symile_export._write_zip_bytes(
            archive,
            "export-manifest.json",
            symile_export._canonical_json_bytes(document),
        )
        for name, content in entries:
            symile_export._write_zip_bytes(archive, name, content)
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
        restore_and_validate_symile_export(archive)


def test_standalone_restore_uses_only_backup_manifest_after_sources_are_deleted(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    export_and_verify(**arguments)
    backup = tmp_path / "backup/campaign.zip"
    for member in arguments["members"]:
        shutil.rmtree(member.path) if member.path.is_dir() else member.path.unlink()
    observed: dict[str, bytes] = {}

    def validate(root: Path) -> None:
        observed["artifact"] = (root / "reports/symile/result/artifact.bin").read_bytes()
        observed["open"] = (root / "private/control/symile/test-open.json").read_bytes()

    restore_and_validate_symile_export(backup, restoration_validator=validate)
    assert observed == {
        "artifact": b"bounded-streaming-input" * 100,
        "open": b'{"test_open_schema_version":1}\n',
    }


def test_export_cleans_temporary_file_after_archive_install_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)

    def fail_archive_install(temporary: Path, destination: Path) -> None:
        del temporary, destination
        raise OSError("simulated archive publication failure")

    monkeypatch.setattr(symile_export, "_install_or_validate", fail_archive_install)
    with pytest.raises(OSError, match="simulated archive publication failure"):
        export_and_verify(**arguments)
    export_root = tmp_path / "export"
    assert not (export_root / "campaign.zip").exists()
    assert not tuple(export_root.glob(".campaign.zip.*.tmp"))
