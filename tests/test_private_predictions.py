from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from radfusion.data.hashing import logical_arrow_sha256, sha256_file
from radfusion.utils.private_predictions import (
    PREDICTION_MANIFEST_FILENAME,
    PREDICTION_SCHEMA,
    PREDICTIONS_FILENAME,
    build_prediction_table,
    publish_prediction_evidence,
    validate_prediction_evidence,
)

_PACKAGE_ID = "model-package-" + "a" * 64
_OTHER_PACKAGE_ID = "model-package-" + "b" * 64
_BUNDLE_ID = "bundle-" + "c" * 64
_SPLIT_ID = "split-assignment-" + "d" * 64


def _publish(tmp_path: Path, *, package_id: str = _PACKAGE_ID):
    return publish_prediction_evidence(
        private_root=tmp_path / "private",
        dataset_id="rsna",
        model_package_id=package_id,
        task_id="pneumonia",
        bundle_id=_BUNDLE_ID,
        split_assignment_id=_SPLIT_ID,
        scope="test",
        sample_ids=("rsna:b", "rsna:a"),
        targets=np.asarray([1, 0], dtype=np.int8),
        logits=np.asarray([2.0, -1.0], dtype=np.float64),
    )


def _publish_symile(tmp_path: Path):
    return publish_prediction_evidence(
        private_root=tmp_path / "private",
        dataset_id="symile",
        model_package_id="fold-package-" + "e" * 64,
        task_id="pneumonia_strict",
        bundle_id=_BUNDLE_ID,
        split_assignment_id=_SPLIT_ID,
        scope="outer_fold_oof",
        sample_ids=("symile:2", "symile:1"),
        targets=[1, 0],
        logits=[2.0, -1.0],
        cv_assignment_id="cv-assignment-" + "f" * 64,
        repeat_seed=17,
        outer_fold=0,
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
            split_assignment_id=_SPLIT_ID,
            scope="test",
            **values,
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
        split_assignment_id=_SPLIT_ID,
        scope="test",
        sample_ids=("rsna:a", "rsna:b"),
        targets=[0, 1],
        logits=[-0.5, 2.0],
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
    assert logical_arrow_sha256(validated.predictions) == manifest["logical_arrow_sha256"]
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
    manifest["logical_arrow_sha256"] = logical_arrow_sha256(table)
    manifest["prediction_file_sha256"] = sha256_file(prediction_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="sample identity"):
        validate_prediction_evidence(evidence.directory)


@pytest.mark.parametrize("row_count", [True, 2.0, "2", None, 0, -1, 3])
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
            split_assignment_id=_SPLIT_ID,
            scope="test",
            sample_ids=sample_ids,
            targets=[0, 1],
            logits=[-1.0, 1.0],
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
            split_assignment_id=_SPLIT_ID,
            scope="test",
            sample_ids=("rsna:a", "rsna:b"),
            targets=[0, 1],
            logits=[-1.0, 1.0],
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
            split_assignment_id=_SPLIT_ID,
            scope="outer_fold_oof" if symile else "test",
            sample_ids=("symile:1", "symile:2") if symile else ("rsna:a", "rsna:b"),
            targets=[0, 1],
            logits=[-1.0, 1.0],
            cv_assignment_id="cv-assignment-" + "f" * 64 if symile else None,
            repeat_seed=17 if symile else None,
            outer_fold=0 if symile else None,
        )


def test_prediction_validation_rejects_tampering_and_symlinks(tmp_path: Path) -> None:
    evidence = _publish(tmp_path)
    manifest_path = evidence.directory / PREDICTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_count"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_prediction_evidence(evidence.directory)

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


@pytest.mark.parametrize("value", [True, 1.0, "1", None, 0, _MISSING_SCHEMA_VERSION])
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
