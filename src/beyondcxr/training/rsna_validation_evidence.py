"""Canonical package-bound evidence for deterministic RSNA validation reports."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.hashing import sha256_file
from beyondcxr.data.rsna_artifacts import (
    LABELS_FILENAME,
    SPLITS_FILENAME,
    validate_bundle_directory,
    validate_bundle_reference,
)
from beyondcxr.data.rsna_schemas import PNEUMONIA_LABEL_POLICY_VERSION, PNEUMONIA_TASK_ID
from beyondcxr.training.config import ExperimentConfig, require_runtime_seed

VALIDATION_EVIDENCE_FILENAME = "validation-evidence.json"
VALIDATION_EVIDENCE_SCHEMA_VERSION = 1
CXR_EPOCH_HISTORY_FIELDS = frozenset(
    {
        "global_epoch",
        "stage_epoch",
        "stage",
        "training_loss",
        "validation_average_precision",
        "selected_best",
        "encoder_learning_rate",
        "head_learning_rate",
        "scheduler_last_epoch",
        "no_improvement_count",
    }
)


@dataclass(frozen=True)
class CanonicalValidationCohort:
    """Ordered validation membership and labels derived from one exact RSNA bundle."""

    sample_ids: tuple[str, ...]
    targets: tuple[int, ...]


def load_canonical_validation_cohort(
    bundle_directory: str | Path,
    *,
    expected_bundle_id: str,
    expected_manifest_sha256: str,
    expected_split_assignment_id: str,
    expected_task_id: str,
    expected_label_policy_version: str,
) -> CanonicalValidationCohort:
    """Derive the exact ordered validation cohort from a fully validated bundle."""
    bundle = Path(bundle_directory)
    reference = validate_bundle_reference(
        bundle,
        expected_bundle_id=expected_bundle_id,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    manifest = validate_bundle_directory(bundle, expected_bundle_id=expected_bundle_id)
    tasks = manifest.get("tasks")
    bundle_task = tasks.get(PNEUMONIA_TASK_ID) if isinstance(tasks, dict) else None
    if (
        reference.manifest_sha256 != expected_manifest_sha256
        or manifest["membership"]["split"]["split_assignment_id"] != expected_split_assignment_id
        or expected_task_id != PNEUMONIA_TASK_ID
        or expected_label_policy_version != PNEUMONIA_LABEL_POLICY_VERSION
        or not isinstance(bundle_task, dict)
        or bundle_task.get("label_policy_version") != PNEUMONIA_LABEL_POLICY_VERSION
    ):
        raise ManifestBuildError("RSNA validation cohort authority is inconsistent")
    split_ids = tuple(
        sorted(
            pq.read_table(
                bundle / SPLITS_FILENAME,
                columns=["sample_id"],
                filters=[("split_name", "=", "validation")],
            )
            .to_pandas()["sample_id"]
            .astype(str)
        )
    )
    labels = (
        pq.read_table(
            bundle / LABELS_FILENAME,
            columns=["sample_id", "label_value"],
            filters=[("task_id", "=", PNEUMONIA_TASK_ID)],
        )
        .to_pandas()
        .set_index("sample_id")["label_value"]
        .to_dict()
    )
    if (
        not split_ids
        or len(split_ids) != len(set(split_ids))
        or any(sample_id not in labels for sample_id in split_ids)
    ):
        raise ManifestBuildError("RSNA canonical validation cohort is invalid")
    return CanonicalValidationCohort(
        split_ids,
        tuple(int(labels[sample_id]) for sample_id in split_ids),
    )


def write_validation_evidence(
    path: str | Path,
    *,
    config: ExperimentConfig,
    sample_ids: Sequence[str],
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    epoch_history: Sequence[Mapping[str, object]] = (),
) -> Path:
    """Write the minimal canonical evidence from which a training report is derived."""
    if config.evaluation is None:
        raise ValueError("RSNA validation evidence requires an evaluation policy")
    target_values = np.asarray(targets)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    document = {
        "rsna_validation_evidence_schema_version": VALIDATION_EVIDENCE_SCHEMA_VERSION,
        "dataset_id": config.dataset.dataset_id,
        "bundle_id": config.dataset.bundle_id,
        "bundle_manifest_sha256": config.dataset.bundle_manifest_sha256,
        "split_assignment_id": config.dataset.split_assignment_id,
        "task_id": config.task.task_id,
        "label_policy_version": config.task.label_policy_version,
        "family_id": config.family.family_id,
        "config_source_sha256": config.config_source_sha256,
        "config_semantic_sha256": config.config_semantic_sha256,
        "seed": require_runtime_seed(config),
        "calibration_bins": config.evaluation.calibration_bins,
        "sensitivity_target": config.evaluation.sensitivity_target,
        "sample_ids": list(sample_ids),
        "targets": [int(value) for value in target_values],
        "probabilities": [float(value) for value in probability_values],
        "epoch_history": [dict(item) for item in epoch_history],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical_bytes(document))
    validate_validation_evidence(destination, package=None, config=config)
    return destination


def validate_validation_evidence(
    path: str | Path,
    *,
    package: Mapping[str, Any] | None,
    config: ExperimentConfig,
    expected_cohort: CanonicalValidationCohort | None = None,
) -> dict[str, Any]:
    """Validate evidence bytes and their exact package/config lineage."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("RSNA validation evidence must be a regular file")
    try:
        raw = source.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("RSNA validation evidence is unreadable") from exc
    fields = {
        "rsna_validation_evidence_schema_version",
        "dataset_id",
        "bundle_id",
        "bundle_manifest_sha256",
        "split_assignment_id",
        "task_id",
        "label_policy_version",
        "family_id",
        "config_source_sha256",
        "config_semantic_sha256",
        "seed",
        "calibration_bins",
        "sensitivity_target",
        "sample_ids",
        "targets",
        "probabilities",
        "epoch_history",
    }
    if (
        not isinstance(document, dict)
        or set(document) != fields
        or type(document["rsna_validation_evidence_schema_version"]) is not int
        or document["rsna_validation_evidence_schema_version"] != VALIDATION_EVIDENCE_SCHEMA_VERSION
        or raw != _canonical_bytes(document)
    ):
        raise ValueError("RSNA validation evidence contract is invalid")
    expected = {
        "dataset_id": config.dataset.dataset_id,
        "bundle_id": config.dataset.bundle_id,
        "bundle_manifest_sha256": config.dataset.bundle_manifest_sha256,
        "split_assignment_id": config.dataset.split_assignment_id,
        "task_id": config.task.task_id,
        "label_policy_version": config.task.label_policy_version,
        "family_id": config.family.family_id,
        "config_source_sha256": config.config_source_sha256,
        "config_semantic_sha256": config.config_semantic_sha256,
        "seed": require_runtime_seed(config),
        "calibration_bins": config.evaluation.calibration_bins if config.evaluation else None,
        "sensitivity_target": config.evaluation.sensitivity_target if config.evaluation else None,
    }
    if any(document[key] != value for key, value in expected.items()):
        raise ValueError("RSNA validation evidence differs from its archived configuration")
    sample_ids = document["sample_ids"]
    targets = document["targets"]
    probabilities = document["probabilities"]
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or any(not isinstance(value, str) or not value for value in sample_ids)
        or not isinstance(targets, list)
        or not targets
        or any(type(value) is not int or value not in {0, 1} for value in targets)
        or set(targets) != {0, 1}
        or not isinstance(probabilities, list)
        or len(sample_ids) != len(targets)
        or len(probabilities) != len(targets)
        or any(
            type(value) not in {int, float} or not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in probabilities
        )
        or not isinstance(document["epoch_history"], list)
    ):
        raise ValueError("RSNA validation evidence values are invalid")
    if config.family.family_id == "cxr_densenet":
        validate_cxr_epoch_history(document["epoch_history"], config=config, package=package)
    elif document["epoch_history"]:
        raise ValueError("RSNA validation evidence values are invalid")
    if expected_cohort is not None and (
        tuple(sample_ids) != expected_cohort.sample_ids or tuple(targets) != expected_cohort.targets
    ):
        raise ValueError("RSNA validation evidence differs from the canonical validation cohort")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("RSNA validation evidence sample membership is not unique")
    if package is not None:
        package_seed = package.get("seed")
        training_policy = package.get("training_policy")
        if package_seed is None and isinstance(training_policy, Mapping):
            package_seed = training_policy.get("seed")
        package_expected = {
            key: package.get(key)
            for key in (
                "dataset_id",
                "bundle_id",
                "bundle_manifest_sha256",
                "split_assignment_id",
                "task_id",
                "label_policy_version",
                "family_id",
                "config_source_sha256",
                "config_semantic_sha256",
            )
        }
        package_expected["seed"] = package_seed
        if any(document[key] != value for key, value in package_expected.items()):
            raise ValueError("RSNA validation evidence differs from its model package")
        digest = package.get("validation_evidence_sha256")
        if not isinstance(digest, str) or digest != sha256_file(source):
            raise ValueError("RSNA validation evidence SHA-256 differs from its model package")
        if package.get("validation_evidence_semantic_sha256") != evidence_semantic_sha256(document):
            raise ValueError("RSNA validation evidence identity differs from its model package")
    return document


def validate_cxr_epoch_history(
    value: object,
    *,
    config: ExperimentConfig,
    package: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Validate the exact CXR training lifecycle and selected-checkpoint history."""
    neural = config.neural
    if config.family.family_id != "cxr_densenet" or neural is None:
        raise ValueError("CXR epoch history requires a CXR configuration")
    if not isinstance(value, list) or not value:
        raise ValueError("CXR epoch history must be non-empty")
    history: list[dict[str, Any]] = []
    expected_stage = "warmup"
    expected_stage_epoch = 1
    best_average_precision = float("-inf")
    selected_row: dict[str, Any] | None = None
    no_improvement_count = 0
    for expected_global_epoch, item in enumerate(value, start=1):
        if not isinstance(item, dict) or set(item) != CXR_EPOCH_HISTORY_FIELDS:
            raise ValueError("CXR epoch history row contract is invalid")
        if (
            type(item["global_epoch"]) is not int
            or item["global_epoch"] <= 0
            or type(item["stage_epoch"]) is not int
            or item["stage_epoch"] <= 0
            or item["stage"] not in {"warmup", "fine_tune"}
            or not _finite_nonnegative(item["training_loss"])
            or not _probability(item["validation_average_precision"])
            or type(item["selected_best"]) is not bool
            or (
                item["encoder_learning_rate"] is not None
                and not _finite_nonnegative(item["encoder_learning_rate"])
            )
            or not _finite_nonnegative(item["head_learning_rate"])
            or (
                item["scheduler_last_epoch"] is not None
                and (
                    type(item["scheduler_last_epoch"]) is not int
                    or item["scheduler_last_epoch"] < 0
                )
            )
            or type(item["no_improvement_count"]) is not int
            or item["no_improvement_count"] < 0
        ):
            raise ValueError("CXR epoch history row values are invalid")
        if item["global_epoch"] != expected_global_epoch:
            raise ValueError("CXR epoch history global epoch sequence is invalid")
        stage = item["stage"]
        if stage != expected_stage:
            if (
                expected_stage != "warmup"
                or stage != "fine_tune"
                or expected_stage_epoch != neural.warmup_epochs + 1
            ):
                raise ValueError("CXR epoch history stage transition is invalid")
            expected_stage = "fine_tune"
            expected_stage_epoch = 1
        if item["stage_epoch"] != expected_stage_epoch:
            raise ValueError("CXR epoch history stage epoch sequence is invalid")
        expected_stage_epoch += 1
        if stage == "warmup":
            if (
                item["encoder_learning_rate"] is not None
                or item["scheduler_last_epoch"] is not None
            ):
                raise ValueError("CXR warmup epoch history values are invalid")
            expected_no_improvement = 0
        else:
            if (
                item["encoder_learning_rate"] is None
                or item["scheduler_last_epoch"] is None
                or item["stage_epoch"] > neural.fine_tune_epochs
            ):
                raise ValueError("CXR fine-tune epoch history values are invalid")
            expected_no_improvement = no_improvement_count
        average_precision = item["validation_average_precision"]
        selected_best = average_precision > (
            best_average_precision + neural.early_stopping_min_delta
        )
        if item["selected_best"] is not selected_best:
            raise ValueError("CXR epoch history checkpoint selection is invalid")
        if selected_best:
            best_average_precision = average_precision
            selected_row = item
            no_improvement_count = 0
            expected_no_improvement = 0
        elif stage == "fine_tune":
            no_improvement_count += 1
            expected_no_improvement = no_improvement_count
        if item["no_improvement_count"] != expected_no_improvement:
            raise ValueError("CXR epoch history no-improvement sequence is invalid")
        if (
            stage == "fine_tune"
            and not selected_best
            and no_improvement_count >= neural.early_stopping_patience
            and expected_global_epoch != len(value)
        ):
            raise ValueError("CXR epoch history continues after early stopping")
        history.append(item)
    if expected_stage != "fine_tune" or selected_row is None:
        raise ValueError("CXR epoch history training lifecycle is incomplete")
    final_row = history[-1]
    if not (
        final_row["stage_epoch"] == neural.fine_tune_epochs
        or (
            final_row["selected_best"] is False
            and final_row["no_improvement_count"] >= neural.early_stopping_patience
        )
    ):
        raise ValueError("CXR epoch history termination is invalid")
    if package is not None:
        selection = package.get("selection")
        if (
            not isinstance(selection, Mapping)
            or set(selection)
            != {"selected_epoch", "selected_stage", "validation_average_precision"}
            or type(selection["selected_epoch"]) is not int
            or selection["selected_epoch"] <= 0
            or selection["selected_stage"] not in {"warmup", "fine_tune"}
            or not _probability(selection["validation_average_precision"])
            or selected_row["global_epoch"] != selection["selected_epoch"]
            or selected_row["stage"] != selection["selected_stage"]
            or not math.isclose(
                selected_row["validation_average_precision"],
                selection["validation_average_precision"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("CXR epoch history differs from package checkpoint selection")
    return history


def _finite_nonnegative(value: object) -> bool:
    return bool(type(value) in {int, float} and math.isfinite(value) and value >= 0.0)


def _probability(value: object) -> bool:
    return bool(_finite_nonnegative(value) and value <= 1.0)


def evidence_semantic_sha256(document: Mapping[str, Any]) -> str:
    """Hash only the validation observations and scientific lineage they represent."""
    fields = (
        "dataset_id",
        "bundle_id",
        "bundle_manifest_sha256",
        "split_assignment_id",
        "task_id",
        "label_policy_version",
        "family_id",
        "seed",
        "sample_ids",
        "targets",
        "probabilities",
        "epoch_history",
    )
    payload = {field: document[field] for field in fields}
    return hashlib.sha256(
        b"beyondcxr-rsna-validation-evidence-v1\0"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
