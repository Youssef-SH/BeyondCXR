from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from symile_lifecycle_test_support import prospective_symile_mutation

import beyondcxr.utils.private_predictions as private_predictions
from beyondcxr.data.hashing import LogicalTableContract, logical_table_sha256, sha256_file
from beyondcxr.utils.private_predictions import (
    _HELD_OUT_PUBLICATION_GUARD,
    PREDICTION_MANIFEST_FILENAME,
    PREDICTION_SCHEMA,
    PREDICTIONS_FILENAME,
    SYMILE_TEST_INFERENCE_POLICY,
    ValidatedHeldOutPublicationAuthority,
    _publish_held_out_prediction_evidence,
    build_prediction_table,
    validate_prediction_evidence,
)
from beyondcxr.utils.private_predictions import (
    publish_prediction_evidence as publish_prediction_evidence_public,
)

_PACKAGE_ID = "model-package-" + "a" * 64
_OTHER_PACKAGE_ID = "model-package-" + "b" * 64
_BUNDLE_ID = "bundle-" + "c" * 64
_FREEZE_ID = "rsna-freeze-" + "d" * 64
_OTHER_FREEZE_ID = "rsna-freeze-" + "e" * 64
_PREDICTION_CONTRACT = LogicalTableContract(
    "private.predictions", 1, PREDICTION_SCHEMA, ("sample_id",), ("sample_id",)
)


@pytest.fixture(autouse=True)
def _isolate_serialization_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep artifact-format tests independent of study lifecycle fixture setup."""
    monkeypatch.setattr(
        private_predictions,
        "_reauthenticate_held_out_publication_authority",
        lambda authority: authority,
    )


def publish_prediction_evidence(**kwargs):
    """Exercise immutable serialization with a guarded test-only authority fixture."""
    if kwargs.get("scope") != "test":
        return publish_prediction_evidence_public(**kwargs)
    dataset_id = kwargs["dataset_id"]
    publication_record = (
        {
            "authority_kind": "rsna-frozen-campaign-v1",
            "execution_id": "rsna-execution-" + "f" * 64,
        }
        if dataset_id == "rsna"
        else {
            "authority_kind": "symile-opened-campaign-v1",
            "execution_id": "symile-execution-" + "f" * 64,
        }
    )
    authority = ValidatedHeldOutPublicationAuthority(
        dataset_id=dataset_id,
        model_package_id=kwargs["model_package_id"],
        task_id=kwargs["task_id"],
        bundle_id=kwargs["bundle_id"],
        freeze_id=kwargs.get("authorized_by_freeze_id"),
        sample_ids=tuple(str(value) for value in kwargs["sample_ids"]),
        targets=tuple(int(value) for value in np.asarray(kwargs["targets"]).reshape(-1)),
        publication_record=publication_record,
        _authorization=object(),
        _projection=None,
        _guard=_HELD_OUT_PUBLICATION_GUARD,
    )
    return _publish_held_out_prediction_evidence(
        publication_authority=authority,
        **kwargs,
    )


def test_generic_publisher_rejects_held_out_coordinates_without_study_authority(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="study-specific validated authority"):
        publish_prediction_evidence_public(
            private_root=tmp_path / "private",
            dataset_id="rsna",
            model_package_id=_PACKAGE_ID,
            task_id="pneumonia",
            bundle_id=_BUNDLE_ID,
            scope="test",
            sample_ids=("rsna:a", "rsna:b"),
            targets=(0, 1),
            logits=(-1.0, 1.0),
            authorized_by_freeze_id=_FREEZE_ID,
        )


def _publish(tmp_path: Path, *, package_id: str = _PACKAGE_ID, freeze_id: str = _FREEZE_ID):
    return publish_prediction_evidence(
        private_root=tmp_path / "private",
        dataset_id="rsna",
        model_package_id=package_id,
        task_id="pneumonia",
        bundle_id=_BUNDLE_ID,
        scope="test",
        sample_ids=("rsna:b", "rsna:a"),
        targets=np.asarray([1, 0], dtype=np.int8),
        logits=np.asarray([2.0, -1.0], dtype=np.float64),
        authorized_by_freeze_id=freeze_id,
    )


def _publish_symile(tmp_path: Path):
    with pytest.MonkeyPatch.context() as monkeypatch:
        with prospective_symile_mutation(tmp_path, monkeypatch) as prospective_mutation:
            return publish_prediction_evidence(
                private_root=tmp_path / "private",
                dataset_id="symile",
                model_package_id="fold-package-" + "e" * 64,
                task_id="pneumonia_strict",
                bundle_id=_BUNDLE_ID,
                scope="outer_fold_oof",
                sample_ids=("symile:2", "symile:1"),
                targets=[1, 0],
                logits=[2.0, -1.0],
                cv_assignment_id="cv-assignment-" + "f" * 64,
                repeat_seed=17,
                outer_fold=0,
                prospective_mutation=prospective_mutation,
            )


def _publish_symile_test(tmp_path: Path, *, freeze: str = "pretest-freeze-" + "1" * 64):
    return publish_prediction_evidence(
        private_root=tmp_path / "private",
        dataset_id="symile",
        model_package_id="final-package-" + "e" * 64,
        task_id="pneumonia_strict",
        bundle_id=_BUNDLE_ID,
        scope="test",
        sample_ids=("symile:2", "symile:1"),
        targets=[1, 0],
        logits=[2.0, -1.0],
        label_policy_version="symile-pneumonia-strict-v1",
        inference_policy=SYMILE_TEST_INFERENCE_POLICY,
        authorized_by_freeze_id=freeze,
    )


def test_prediction_evidence_publishes_exact_canonical_contract(tmp_path: Path) -> None:
    evidence = _publish(tmp_path)
    validated = validate_prediction_evidence(evidence.directory)
    table = pq.read_table(evidence.directory / PREDICTIONS_FILENAME)

    assert evidence.directory == (tmp_path / "private/predictions/rsna" / evidence.prediction_id)
    assert {path.name for path in evidence.directory.iterdir()} == {
        PREDICTIONS_FILENAME,
        PREDICTION_MANIFEST_FILENAME,
    }
    assert validated.manifest["row_count"] == 2
    assert table.column("sample_id").to_pylist() == ["rsna:a", "rsna:b"]
    assert table.column("target").to_pylist() == [0, 1]
    assert table.schema == build_prediction_table(("rsna:a", "rsna:b"), [0, 1], [-1.0, 2.0]).schema


def test_symile_test_authorization_is_held_out_prediction_identity(
    tmp_path: Path,
) -> None:
    evidence = _publish_symile_test(tmp_path)
    other_control = _publish_symile_test(tmp_path / "other", freeze="pretest-freeze-" + "2" * 64)
    manifest = evidence.manifest

    assert other_control.prediction_id != evidence.prediction_id
    assert manifest["authorized_by_freeze_id"] == "pretest-freeze-" + "1" * 64
    assert manifest["label_policy_version"] == "symile-pneumonia-strict-v1"
    assert manifest["inference_policy"] == SYMILE_TEST_INFERENCE_POLICY
    repeated_other = _publish_symile_test(tmp_path, freeze="pretest-freeze-" + "2" * 64)
    assert repeated_other.prediction_id == other_control.prediction_id


@pytest.mark.parametrize("value", [True, 1.0])
def test_symile_test_inference_policy_requires_exact_integer_schema_version(
    tmp_path: Path, value: object
) -> None:
    evidence = _publish_symile_test(tmp_path)
    manifest_path = evidence.directory / PREDICTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_bytes())
    manifest["inference_policy"]["inference_policy_schema_version"] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="scientific policy"):
        validate_prediction_evidence(evidence.directory)


def test_rsna_held_out_prediction_requires_its_authorizing_freeze(tmp_path: Path) -> None:
    evidence = _publish(tmp_path)

    assert evidence.manifest["authorized_by_freeze_id"] == _FREEZE_ID
    assert "label_policy_version" not in evidence.manifest
    assert "inference_policy" not in evidence.manifest
    with pytest.raises(ValueError, match="scientific freeze"):
        publish_prediction_evidence(
            private_root=tmp_path / "missing/private",
            dataset_id="rsna",
            model_package_id=_PACKAGE_ID,
            task_id="pneumonia",
            bundle_id=_BUNDLE_ID,
            scope="test",
            sample_ids=("rsna:a", "rsna:b"),
            targets=[0, 1],
            logits=[-1.0, 2.0],
        )


def test_rsna_held_out_prediction_identity_binds_and_validates_freeze(tmp_path: Path) -> None:
    evidence = _publish(tmp_path / "first")
    other = _publish(tmp_path / "other", freeze_id=_OTHER_FREEZE_ID)

    assert evidence.prediction_id != other.prediction_id
    with pytest.raises(ValueError, match="different scientific freeze"):
        validate_prediction_evidence(
            evidence.directory,
            expected_authorizing_freeze_id=_OTHER_FREEZE_ID,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_ids", ("rsna:a", "rsna:a")),
        ("targets", np.asarray([0, 2])),
        ("logits", np.asarray([-1.0, np.inf])),
    ],
)
def test_prediction_evidence_rejects_invalid_rows(
    tmp_path: Path, field: str, value: object
) -> None:
    values = {
        "sample_ids": ("rsna:a", "rsna:b"),
        "targets": np.asarray([0, 1]),
        "logits": np.asarray([-1.0, 2.0]),
    }
    values[field] = value
    with pytest.raises(ValueError):
        publish_prediction_evidence(
            private_root=tmp_path / "private",
            dataset_id="rsna",
            model_package_id=_PACKAGE_ID,
            task_id="pneumonia",
            bundle_id=_BUNDLE_ID,
            scope="test",
            **values,
            authorized_by_freeze_id=_FREEZE_ID,
        )


def test_prediction_identity_binds_logical_content_and_package(tmp_path: Path) -> None:
    first = _publish(tmp_path / "first")
    repeated = _publish(tmp_path / "second")
    changed_package = _publish(tmp_path / "third", package_id=_OTHER_PACKAGE_ID)
    changed_content = publish_prediction_evidence(
        private_root=tmp_path / "fourth/private",
        dataset_id="rsna",
        model_package_id=_PACKAGE_ID,
        task_id="pneumonia",
        bundle_id=_BUNDLE_ID,
        scope="test",
        sample_ids=("rsna:a", "rsna:b"),
        targets=[0, 1],
        logits=[-0.5, 2.0],
        authorized_by_freeze_id=_FREEZE_ID,
    )
    assert repeated.prediction_id == first.prediction_id
    assert changed_package.prediction_id != first.prediction_id
    assert changed_content.prediction_id != first.prediction_id
    with pytest.raises(ValueError, match="model package"):
        validate_prediction_evidence(
            first.directory,
            expected_model_package_id="model-package-" + "e" * 64,
        )


def test_prediction_identity_is_independent_of_parquet_encoding(tmp_path: Path) -> None:
    evidence = _publish(tmp_path)
    path = evidence.directory / PREDICTIONS_FILENAME
    table = pq.read_table(path)
    original_physical = sha256_file(path)
    pq.write_table(table, path, compression=None)
    assert sha256_file(path) != original_physical
    manifest_path = evidence.directory / PREDICTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prediction_file_sha256"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validated = validate_prediction_evidence(evidence.directory)
    assert validated.prediction_id == evidence.prediction_id
    assert (
        logical_table_sha256(validated.predictions, _PREDICTION_CONTRACT)
        == manifest["prediction_table_sha256"]
    )
    existing_bytes = path.read_bytes()
    repeated = _publish(tmp_path)
    assert repeated.created is False
    assert path.read_bytes() == existing_bytes


@pytest.mark.parametrize(
    ("publisher", "invalid_ids"),
    [
        (_publish, ["bad:1", "rsna:b"]),
        (_publish_symile, ["symile:2", "symile:not-numeric"]),
    ],
)
def test_prediction_validation_revalidates_dataset_sample_ids(
    tmp_path: Path, publisher, invalid_ids: list[str]
) -> None:
    evidence = publisher(tmp_path)
    prediction_path = evidence.directory / PREDICTIONS_FILENAME
    table = pq.read_table(prediction_path).set_column(
        0,
        pa.field("sample_id", pa.string(), nullable=False),
        pa.array(invalid_ids, type=pa.string()),
    )
    assert table.schema == PREDICTION_SCHEMA
    pq.write_table(table, prediction_path)
    manifest_path = evidence.directory / PREDICTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prediction_table_sha256"] = logical_table_sha256(table, _PREDICTION_CONTRACT)
    manifest["prediction_file_sha256"] = sha256_file(prediction_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="sample identity"):
        validate_prediction_evidence(evidence.directory)


@pytest.mark.parametrize("row_count", [True, 2.0, 0, 3])
def test_prediction_validation_requires_strict_positive_integer_row_count(
    tmp_path: Path, row_count: object
) -> None:
    evidence = _publish(tmp_path)
    manifest_path = evidence.directory / PREDICTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_count"] = row_count
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_prediction_evidence(evidence.directory)


@pytest.mark.parametrize("sample_ids", [(1, 2), ("", "rsna:b"), ("bad:a", "rsna:b")])
def test_prediction_rejects_non_string_or_empty_sample_ids(
    tmp_path: Path, sample_ids: tuple[object, object]
) -> None:
    with pytest.raises(ValueError):
        publish_prediction_evidence(
            private_root=tmp_path / "private",
            dataset_id="rsna",
            model_package_id=_PACKAGE_ID,
            task_id="pneumonia",
            bundle_id=_BUNDLE_ID,
            scope="test",
            sample_ids=sample_ids,
            targets=[0, 1],
            logits=[-1.0, 1.0],
            authorized_by_freeze_id=_FREEZE_ID,
        )


@pytest.mark.parametrize("task_id", ["../task", "a/b", "a\\b", ".", ".."])
def test_prediction_rejects_unsafe_components(tmp_path: Path, task_id: str) -> None:
    with pytest.raises(ValueError):
        publish_prediction_evidence(
            private_root=tmp_path / "private",
            dataset_id="rsna",
            model_package_id=_PACKAGE_ID,
            task_id=task_id,
            bundle_id=_BUNDLE_ID,
            scope="test",
            sample_ids=("rsna:a", "rsna:b"),
            targets=[0, 1],
            logits=[-1.0, 1.0],
            authorized_by_freeze_id=_FREEZE_ID,
        )


@pytest.mark.parametrize(
    ("dataset_id", "model_package_id", "task_id"),
    [
        ("rsna", "fold-package-" + "e" * 64, "pneumonia"),
        ("rsna", _PACKAGE_ID, "pneumonia_strict"),
        ("symile", _PACKAGE_ID, "pneumonia_strict"),
        ("symile", "fold-package-" + "e" * 64, "pneumonia"),
    ],
)
def test_prediction_rejects_dataset_incompatible_coordinates(
    tmp_path: Path,
    dataset_id: str,
    model_package_id: str,
    task_id: str,
) -> None:
    symile = dataset_id == "symile"
    with pytest.raises(ValueError):
        publish_prediction_evidence(
            private_root=tmp_path / "private",
            dataset_id=dataset_id,
            model_package_id=model_package_id,
            task_id=task_id,
            bundle_id=_BUNDLE_ID,
            scope="outer_fold_oof" if symile else "test",
            sample_ids=("symile:1", "symile:2") if symile else ("rsna:a", "rsna:b"),
            targets=[0, 1],
            logits=[-1.0, 1.0],
            cv_assignment_id="cv-assignment-" + "f" * 64 if symile else None,
            repeat_seed=17 if symile else None,
            outer_fold=0 if symile else None,
            authorized_by_freeze_id=None if symile else _FREEZE_ID,
        )


def test_prediction_validation_rejects_tampering_and_symlinks(tmp_path: Path) -> None:
    evidence = _publish(tmp_path / "second")
    prediction_path = evidence.directory / PREDICTIONS_FILENAME
    prediction_path.write_bytes(prediction_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_prediction_evidence(evidence.directory)

    evidence = _publish(tmp_path / "third")
    prediction_path = evidence.directory / PREDICTIONS_FILENAME
    target = tmp_path / "predictions.parquet"
    prediction_path.rename(target)
    prediction_path.symlink_to(target)
    with pytest.raises(ValueError):
        validate_prediction_evidence(evidence.directory)


_MISSING_SCHEMA_VERSION = object()


@pytest.mark.parametrize("value", [True, 1.0, 2, _MISSING_SCHEMA_VERSION])
def test_prediction_manifest_requires_integer_schema_version_one(
    tmp_path: Path, value: object
) -> None:
    evidence = _publish(tmp_path)
    manifest_path = evidence.directory / PREDICTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value is _MISSING_SCHEMA_VERSION:
        manifest.pop("prediction_schema_version")
    else:
        manifest["prediction_schema_version"] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_prediction_evidence(evidence.directory)
