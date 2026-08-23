from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from symile_campaign_test_support import (
    _freeze,
)

import beyondcxr.training.symile_campaign_control as campaign_control
import beyondcxr.training.symile_test_inference as test_inference
import beyondcxr.utils.publication as publication
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.symile_campaign_control import (
    ValidatedPretestFreeze,
    ValidatedTestOpenRecord,
    create_or_validate_test_open_record,
    materialize_official_test,
)


@pytest.mark.parametrize(
    "field,value",
    [
        ("primary_metric", "average_precision"),
        (
            "metric_direction",
            {
                "roc_auc": "higher_is_better",
                "average_precision": "higher_is_better",
                "brier_score": "higher_is_better",
            },
        ),
    ],
)
def test_freeze_rejects_changed_metric_policy(tmp_path, field, value):
    freeze = _freeze(tmp_path)
    document = json.loads(json.dumps(freeze.manifest))
    document["held_out_policy"]["metric_policy"][field] = value
    with pytest.raises(ManifestBuildError, match="fields"):
        campaign_control._validate_freeze_fields(document)


@pytest.mark.parametrize(
    "changes",
    [
        {"device_type": "tpu"},
        {"autocast_dtype": "float16"},
        {"device_type": "cuda", "autocast_dtype": None},
        {"autocast_dtype": "bfloat16"},
        {"gpu_name": "irrelevant"},
        {"cuda_runtime_version": "12.4"},
    ],
)
def test_freeze_rejects_malformed_neural_inference_runtime(tmp_path, changes):
    freeze = _freeze(tmp_path)
    document = json.loads(json.dumps(freeze.manifest))
    document["held_out_policy"]["neural_inference_runtime"].update(changes)
    with pytest.raises(ManifestBuildError, match="fields"):
        campaign_control._validate_freeze_fields(document)


def test_freeze_rejects_tampered_neural_inference_runtime(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path / "freeze")
    manifest_path = freeze.directory / "manifest.json"
    document = json.loads(manifest_path.read_bytes())
    document["held_out_policy"]["neural_inference_runtime"] = {
        "device_type": "cuda",
        "autocast_dtype": "float16",
        "cuda_runtime_version": "12.4",
        "cudnn_version": 9100,
        "gpu_device_name": "GPU-A",
        "gpu_compute_capability": [8, 6],
    }
    manifest_path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    with pytest.raises(ManifestBuildError, match="identity"):
        campaign_control._validate_pretest_freeze_document(freeze.directory)


@pytest.mark.parametrize("value", [True, 1.0])
def test_pretest_freeze_requires_exact_integer_schema_version(
    tmp_path: Path, value: object
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    altered = tmp_path / "altered-freeze"
    shutil.copytree(freeze.directory, altered)
    manifest = json.loads((altered / "manifest.json").read_bytes())
    manifest["pretest_freeze_schema_version"] = value
    (altered / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    with pytest.raises(ManifestBuildError, match="contract"):
        campaign_control._validate_pretest_freeze_document(altered, enforce_directory_name=False)


@pytest.mark.parametrize("value", [True, 1.0])
def test_test_open_requires_exact_integer_schema_version(tmp_path: Path, value: object) -> None:
    freeze = _freeze(tmp_path / "freeze")
    control = tmp_path / "control"
    record = create_or_validate_test_open_record(control_root=control, capability=freeze)
    document = json.loads(record.path.read_bytes())
    document["test_open_schema_version"] = value
    record.path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    with pytest.raises(ManifestBuildError, match="another freeze"):
        create_or_validate_test_open_record(control_root=control, capability=freeze)


def test_test_open_record_is_atomic_identity_only_and_dual_authorization_is_required(
    tmp_path: Path,
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    control = tmp_path / "control"
    record = create_or_validate_test_open_record(control_root=control, capability=freeze)
    original = record.path.read_bytes()
    assert create_or_validate_test_open_record(control_root=control, capability=freeze) == record
    assert record.path.read_bytes() == original
    assert set(json.loads(record.path.read_text())) == {
        "test_open_schema_version",
        "freeze_id",
        "bundle_id",
        "split_assignment_id",
        "science_git_commit",
        "dependency_lock_sha256",
        "created_at_utc",
    }
    with pytest.raises(ManifestBuildError):
        materialize_official_test(
            freeze,
            ValidatedTestOpenRecord(
                record.path,
                "pretest-freeze-" + "9" * 64,
                record.document_sha256,
                record._guard,
            ),
            lambda: "opened",
        )
    forged = ValidatedPretestFreeze(
        freeze.directory, freeze.manifest, freeze.manifest_sha256, object()
    )
    with pytest.raises(ManifestBuildError):
        materialize_official_test(forged, record, lambda: "opened")
    forged_record = ValidatedTestOpenRecord(
        record.path,
        record.freeze_id,
        record.document_sha256,
        object(),
    )
    with pytest.raises(ManifestBuildError):
        materialize_official_test(freeze, forged_record, lambda: "opened")
    assert materialize_official_test(freeze, record, lambda: "opened") == "opened"

    document = json.loads(original)
    document["created_at_utc"] = "2026-01-01T00:00:00Z"
    record.path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    with pytest.raises(ManifestBuildError, match="changed after validation"):
        materialize_official_test(freeze, record, lambda: "opened")


def test_different_freeze_cannot_replace_test_open_record(tmp_path: Path) -> None:
    first = _freeze(tmp_path / "first", "a")
    second = _freeze(tmp_path / "second", "9")
    control = tmp_path / "control"
    create_or_validate_test_open_record(control_root=control, capability=first)
    with pytest.raises(ManifestBuildError, match="another freeze"):
        create_or_validate_test_open_record(control_root=control, capability=second)


def test_malformed_test_open_record_is_not_repaired(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path / "freeze")
    control = tmp_path / "control"
    control.mkdir()
    path = control / "test-open.json"
    path.write_bytes(b'{"test_open_schema_version":')

    with pytest.raises(ManifestBuildError, match="malformed"):
        create_or_validate_test_open_record(control_root=control, capability=freeze)

    assert path.read_bytes() == b'{"test_open_schema_version":'


def test_test_open_publication_loser_validates_same_freeze_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    control = tmp_path / "control"
    real_link = os.link

    def winning_link(source: str | Path, destination: str | Path) -> None:
        real_link(source, destination)
        raise FileExistsError

    monkeypatch.setattr(publication.os, "link", winning_link)
    record = create_or_validate_test_open_record(control_root=control, capability=freeze)

    assert record.path.is_file()
    assert not tuple(control.glob(".test-open.json-*.tmp"))


def test_test_open_publication_loser_rejects_different_freeze_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = _freeze(tmp_path / "requested", "a")
    winner = _freeze(tmp_path / "winner", "9")
    control = tmp_path / "control"

    def winning_link(source: str | Path, destination: str | Path) -> None:
        del source
        document = {
            "test_open_schema_version": 1,
            "freeze_id": winner.freeze_id,
            "bundle_id": winner.manifest["bundle"]["bundle_id"],
            "split_assignment_id": winner.manifest["bundle"]["split_assignment_id"],
            "science_git_commit": winner.manifest["science_git_commit"],
            "dependency_lock_sha256": winner.manifest["dependency_lock_sha256"],
            "created_at_utc": "2026-01-01T00:00:00Z",
        }
        Path(destination).write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        raise FileExistsError

    monkeypatch.setattr(publication.os, "link", winning_link)
    with pytest.raises(ManifestBuildError, match="another freeze"):
        create_or_validate_test_open_record(control_root=control, capability=requested)

    assert not tuple(control.glob(".test-open.json-*.tmp"))


def test_test_open_publication_failure_leaves_no_final_or_temporary_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    control = tmp_path / "control"

    def failed_link(source: str | Path, destination: str | Path) -> None:
        del source, destination
        raise OSError("simulated publication failure")

    monkeypatch.setattr(publication.os, "link", failed_link)
    with pytest.raises(OSError, match="simulated publication failure"):
        create_or_validate_test_open_record(control_root=control, capability=freeze)

    assert not (control / "test-open.json").exists()
    assert not tuple(control.glob(".test-open.json-*.tmp"))


def test_invalid_final_membership_fails_before_official_accessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accessed = False

    def forbidden_accessor(*args: object, **kwargs: object) -> None:
        nonlocal accessed
        accessed = True

    monkeypatch.setattr(test_inference, "materialize_official_test", forbidden_accessor)
    with pytest.raises(ManifestBuildError, match="exactly fourteen"):
        test_inference.publish_frozen_test_predictions(
            capability=_freeze(tmp_path / "freeze"),
            test_open_record=ValidatedTestOpenRecord(
                tmp_path / "missing.json", "wrong", "0" * 64, object()
            ),
            final_packages=[],
            source_root=tmp_path,
            manifest_root=tmp_path,
            private_root=tmp_path,
            runtime=None,
        )
    assert accessed is False
