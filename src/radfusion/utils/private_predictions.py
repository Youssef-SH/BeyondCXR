"""Publish validated patient-level neural predictions in private local storage."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from radfusion.data.hashing import sha256_file
from radfusion.utils.publication import publish_directory, staging_directory

PRIVATE_PREDICTION_SCHEMA_VERSION = 1
PREDICTIONS_FILENAME = "predictions.parquet"
PREDICTION_MANIFEST_FILENAME = "manifest.json"
PRIVATE_PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("private_patient_key", pa.string(), nullable=False),
        pa.field("target", pa.int8(), nullable=False),
        pa.field("logit", pa.float64(), nullable=False),
        pa.field("probability", pa.float64(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("training_run_id", pa.string(), nullable=False),
        pa.field("test_evaluation_run_id", pa.string(), nullable=False),
        pa.field("model_package_id", pa.string(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
    ]
)


def private_root_for_reports(report_directory: str | Path) -> Path:
    """Return the ignored private workspace adjacent to the public report root."""
    return Path(report_directory).parent / "private"


def publish_private_neural_predictions(
    *,
    private_root: str | Path,
    dataset: str,
    training_run_id: str,
    test_evaluation_run_id: str,
    model_package_id: str,
    seed: int,
    sample_ids: Sequence[str],
    patient_keys: Sequence[str],
    targets: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
) -> Path:
    """Publish one immutable, aligned private neural test-prediction table."""
    for value, name in (
        (dataset, "dataset"),
        (training_run_id, "training_run_id"),
        (test_evaluation_run_id, "test_evaluation_run_id"),
        (model_package_id, "model_package_id"),
    ):
        _safe_component(value, name)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Private prediction seed must be an integer")
    table = _prediction_table(
        training_run_id=training_run_id,
        test_evaluation_run_id=test_evaluation_run_id,
        model_package_id=model_package_id,
        seed=seed,
        sample_ids=sample_ids,
        patient_keys=patient_keys,
        targets=targets,
        logits=logits,
        probabilities=probabilities,
    )
    destination = Path(private_root) / "predictions" / dataset / test_evaluation_run_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Private prediction artifact already exists: {destination}")
    stage = staging_directory(destination)
    try:
        prediction_path = stage / PREDICTIONS_FILENAME
        pq.write_table(
            table,
            prediction_path,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
        )
        manifest = {
            "private_prediction_schema_version": PRIVATE_PREDICTION_SCHEMA_VERSION,
            "dataset": dataset,
            "row_count": table.num_rows,
            "prediction_file_sha256": sha256_file(prediction_path),
            "training_run_id": training_run_id,
            "test_evaluation_run_id": test_evaluation_run_id,
            "model_package_id": model_package_id,
            "seed": seed,
        }
        (stage / PREDICTION_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        validate_private_neural_predictions(stage)
        publish_directory(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return destination


def validate_private_neural_predictions(directory: str | Path) -> dict[str, Any]:
    """Validate one exact private neural prediction artifact without exposing rows."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Private prediction artifact must be a physical directory")
    with os.scandir(root) as entries:
        inspected = list(entries)
    if {entry.name for entry in inspected} != {
        PREDICTIONS_FILENAME,
        PREDICTION_MANIFEST_FILENAME,
    } or any(entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in inspected):
        raise ValueError("Private prediction artifact has an invalid file set")
    try:
        manifest = json.loads((root / PREDICTION_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Private prediction manifest is unreadable") from exc
    fields = {
        "private_prediction_schema_version",
        "dataset",
        "row_count",
        "prediction_file_sha256",
        "training_run_id",
        "test_evaluation_run_id",
        "model_package_id",
        "seed",
    }
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ValueError("Private prediction manifest has an unexpected field set")
    version = manifest["private_prediction_schema_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != PRIVATE_PREDICTION_SCHEMA_VERSION
    ):
        raise ValueError("Private prediction schema version is invalid")
    for field in ("dataset", "training_run_id", "test_evaluation_run_id", "model_package_id"):
        _safe_component(manifest[field], field)
    row_count = manifest["row_count"]
    seed = manifest["seed"]
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("Private prediction manifest counts or seed are invalid")
    prediction_path = root / PREDICTIONS_FILENAME
    if sha256_file(prediction_path) != manifest["prediction_file_sha256"]:
        raise ValueError("Private prediction file SHA-256 mismatch")
    table = pq.read_table(prediction_path)
    if table.schema != PRIVATE_PREDICTION_SCHEMA or table.num_rows != row_count:
        raise ValueError("Private prediction table schema or row count is invalid")
    _validate_prediction_rows(table, manifest)
    return manifest


def _prediction_table(
    *,
    training_run_id: str,
    test_evaluation_run_id: str,
    model_package_id: str,
    seed: int,
    sample_ids: Sequence[str],
    patient_keys: Sequence[str],
    targets: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
) -> pa.Table:
    raw_targets = np.asarray(targets).reshape(-1)
    if set(np.unique(raw_targets).tolist()) - {0, 1}:
        raise ValueError("Private prediction targets must be binary")
    target_values = raw_targets.astype(np.int8, copy=False)
    logit_values = np.asarray(logits, dtype=np.float64).reshape(-1)
    probability_values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    lengths = {
        len(sample_ids),
        len(patient_keys),
        len(target_values),
        len(logit_values),
        len(probability_values),
    }
    if lengths == {0} or len(lengths) != 1:
        raise ValueError("Private prediction columns must have one equal non-zero length")
    if list(sample_ids) != sorted(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Private prediction sample IDs must be unique and ordered")
    if any(not isinstance(value, str) or not value for value in (*sample_ids, *patient_keys)):
        raise ValueError("Private prediction identities must be non-empty strings")
    if not np.isfinite(logit_values).all() or not np.isfinite(probability_values).all():
        raise ValueError("Private prediction numeric values must be finite")
    if ((probability_values < 0.0) | (probability_values > 1.0)).any():
        raise ValueError("Private prediction probabilities must be within [0, 1]")
    if any(
        not math.isclose(probability, _sigmoid(logit), rel_tol=1e-12, abs_tol=1e-15)
        for logit, probability in zip(logit_values, probability_values, strict=True)
    ):
        raise ValueError("Private prediction probabilities do not correspond to stored logits")
    count = len(target_values)
    return pa.Table.from_pydict(
        {
            "sample_id": list(sample_ids),
            "private_patient_key": list(patient_keys),
            "target": target_values,
            "logit": logit_values,
            "probability": probability_values,
            "split": ["test"] * count,
            "training_run_id": [training_run_id] * count,
            "test_evaluation_run_id": [test_evaluation_run_id] * count,
            "model_package_id": [model_package_id] * count,
            "seed": [seed] * count,
        },
        schema=PRIVATE_PREDICTION_SCHEMA,
    )


def _validate_prediction_rows(table: pa.Table, manifest: dict[str, Any]) -> None:
    rows = table.to_pylist()
    sample_ids = [row["sample_id"] for row in rows]
    if sample_ids != sorted(sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Private prediction rows are not uniquely ordered")
    for row in rows:
        if (
            not row["private_patient_key"]
            or row["target"] not in {0, 1}
            or row["split"] != "test"
            or row["training_run_id"] != manifest["training_run_id"]
            or row["test_evaluation_run_id"] != manifest["test_evaluation_run_id"]
            or row["model_package_id"] != manifest["model_package_id"]
            or row["seed"] != manifest["seed"]
            or not math.isfinite(row["logit"])
            or not math.isfinite(row["probability"])
            or not 0.0 <= row["probability"] <= 1.0
            or not math.isclose(
                row["probability"], _sigmoid(row["logit"]), rel_tol=1e-12, abs_tol=1e-15
            )
        ):
            raise ValueError("Private prediction row violates its contract")


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _safe_component(value: object, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"Private prediction {field} must be one safe path component")
