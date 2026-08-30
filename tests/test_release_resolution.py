from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path

import pytest

import beyondcxr.release.resolution as resolution
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.preservation import PreservationMember, export_and_verify


def _archive_name(index: int) -> str:
    return f"symile-global-result-{index:064x}-symile-execution-{index + 100:064x}.zip"


def _candidate(root: Path, index: int) -> Path:
    path = root / _archive_name(index)
    path.write_bytes(f"candidate-{index}".encode())
    return path


def test_automatic_resolution_requires_exactly_one_valid_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid = _candidate(tmp_path, 1)
    valid = _candidate(tmp_path, 2)

    def validate(path: str | Path) -> Path:
        candidate = Path(path)
        if candidate == invalid:
            raise ManifestBuildError("invalid candidate")
        return candidate

    monkeypatch.setattr(resolution, "_restored_candidate", lambda path: nullcontext(validate(path)))
    assert resolution.resolve_release_archive(tmp_path) == valid

    valid.unlink()
    with pytest.raises(ManifestBuildError, match="No valid completed"):
        resolution.resolve_release_archive(tmp_path)


def test_automatic_resolution_rejects_ambiguity_without_newest_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    older = _candidate(tmp_path, 1)
    newer = _candidate(tmp_path, 2)
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    monkeypatch.setattr(
        resolution,
        "_restored_candidate",
        lambda path: nullcontext(Path(path)),
    )

    with pytest.raises(ManifestBuildError, match="ambiguous"):
        resolution.resolve_release_archive(tmp_path)


def test_automatic_resolution_ignores_current_and_noncanonical_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid = _candidate(tmp_path, 1)
    (tmp_path / "CURRENT").write_text(valid.name, encoding="utf-8")
    (tmp_path / "latest.zip").write_bytes(b"not a candidate")
    observed: list[Path] = []

    def validate(path: str | Path) -> Path:
        observed.append(Path(path))
        return Path(path)

    monkeypatch.setattr(resolution, "_restored_candidate", lambda path: nullcontext(validate(path)))
    assert resolution.resolve_release_archive(tmp_path) == valid
    assert observed == [valid]


def test_explicit_historical_candidate_is_always_fully_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact = tmp_path / "historical-name.zip"
    exact.write_bytes(b"candidate")
    observed: list[Path] = []

    def validate(path: str | Path) -> Path:
        observed.append(Path(path))
        return Path(path)

    monkeypatch.setattr(resolution, "_restored_candidate", lambda path: nullcontext(validate(path)))
    assert resolution.resolve_release_archive(exact) == exact
    assert observed == [exact]


def test_explicit_invalid_candidate_cannot_bypass_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact = tmp_path / "historical-name.zip"
    exact.write_bytes(b"candidate")

    def reject(_path: str | Path) -> Path:
        raise ManifestBuildError("recursive validation failed")

    monkeypatch.setattr(resolution, "_restored_candidate", lambda path: nullcontext(reject(path)))
    with pytest.raises(ManifestBuildError, match="candidate is invalid"):
        resolution.resolve_release_archive(exact)


def test_candidate_validity_uses_transport_and_real_restoration_callback_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "authority.json").write_text("{}\n", encoding="utf-8")
    archive = export_and_verify(
        members=[PreservationMember(source, Path("reports/symile/global-result"))],
        export_root=tmp_path / "outbox",
        backup_root=tmp_path / "backup",
        export_name="historical",
    )
    observed: list[Path] = []

    def validate(root: Path) -> object:
        assert (root / "reports/symile/global-result/authority.json").is_file()
        observed.append(root)
        return object()

    monkeypatch.setattr(
        "beyondcxr.training.symile_campaign.validate_restored_campaign",
        validate,
    )
    assert resolution.resolve_release_archive(archive) == archive
    assert len(observed) == 1


def test_invalid_scientific_control_is_not_an_automatic_release_candidate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "test-open.json"
    source.write_text('{"freeze_id":"../outside"}\n', encoding="utf-8")
    archive = export_and_verify(
        members=[PreservationMember(source, Path("private/control/symile/test-open.json"))],
        export_root=tmp_path / "outbox",
        backup_root=tmp_path / "backup",
        export_name=_archive_name(1).removesuffix(".zip"),
    )
    with pytest.raises(ManifestBuildError, match="No valid completed"):
        resolution.resolve_release_archive(archive.parent)


def test_resolved_release_retains_one_authenticated_restoration(tmp_path, monkeypatch):
    from beyondcxr.training.preservation import (
        _validated_restoration_context,
        require_restoration_context,
    )

    source = tmp_path / "source"
    source.mkdir()
    (source / "authority.json").write_text("{}\n")
    archive = export_and_verify(
        members=[PreservationMember(source, Path("reports/result"))],
        export_root=tmp_path / "outbox",
        backup_root=tmp_path / "backup",
        export_name="representative",
    )
    observed = []

    def validate(root):
        observed.append(root)
        return _validated_restoration_context(root)

    monkeypatch.setattr("beyondcxr.training.symile_campaign.validate_restored_campaign", validate)
    with resolution.resolved_release(archive) as (selected, context):
        assert selected == archive
        assert len(observed) == 1
        assert require_restoration_context(context) is context
        assert (context.root / "reports/result/authority.json").is_file()
    with pytest.raises(PermissionError):
        require_restoration_context(context)
