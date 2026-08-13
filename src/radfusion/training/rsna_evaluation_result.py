"""Publish and validate immutable RSNA held-out evaluation results."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from radfusion.evaluation.metrics import (
    evaluate_operating_point,
    evaluate_probabilities,
)
from radfusion.training.config import ExperimentConfig
from radfusion.training.rsna_train_metadata import (
    metrics_document,
    validate_report_set,
    write_run_reports,
)
from radfusion.utils.package_identity import (
    canonical_scientific_id,
    package_scientific_config_payload,
)
from radfusion.utils.privacy import validate_public_reports
from radfusion.utils.private_predictions import (
    ValidatedPredictionEvidence,
    validate_prediction_evidence,
)
from radfusion.utils.publication import (
    install_immutable_directory,
    staging_directory,
    validate_path_component,
)
from radfusion.utils.rsna_model_publication import (
    MODEL_FILENAME as TABULAR_MODEL_FILENAME,
)
from radfusion.utils.rsna_model_publication import (
    validate_published_model,
    validated_threshold_contract,
)
from radfusion.utils.rsna_neural_publication import (
    validate_published_neural_model,
)

EVALUATION_SCHEMA_VERSION = 1
EVALUATION_PREFIX = "evaluation-"
EVALUATION_MANIFEST_FILENAME = "manifest.json"
DERIVATIVES_DIRECTORY = "derivatives"
EVALUATION_POLICY_VERSION = "rsna-held-out-evaluation-v1"
_FIELDS = {
    "evaluation_schema_version",
    "evaluation_id",
    "dataset_id",
    "family_id",
    "modalities",
    "seed",
    "model_package_id",
    "prediction_id",
    "prediction_manifest_sha256",
    "task_id",
    "bundle_id",
    "split_assignment_id",
    "scope",
    "evaluation_policy",
    "thresholds",
    "claims",
}


@dataclass(frozen=True)
class CompletedRsnaEvaluation:
    """Scientific identities and operational references from held-out evaluation."""

    evaluation_id: str
    prediction_id: str
    model_package_id: str
    mlflow_run_id: str
    artifact_directory: Path
    private_prediction_directory: Path
    average_precision: float


@dataclass(frozen=True)
class ValidatedEvaluationResult:
    """Validated aggregate claims derived from one prediction object."""

    directory: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    created: bool = False


def derive_evaluation_claims(
    evidence: ValidatedPredictionEvidence,
    *,
    evaluation_policy: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Derive the complete aggregate metric document from validated evidence."""
    policy = _validate_evaluation_policy(evaluation_policy)
    calibration_bins = policy["calibration_bins"]
    sensitivity_target = policy["threshold_selection"]["sensitivity_target"]
    frame = evidence.predictions.to_pandas()
    targets = frame["target"].to_numpy(dtype=np.int8)
    probabilities = frame["probability"].to_numpy(dtype=np.float64)
    probability = evaluate_probabilities(targets, probabilities, calibration_bins=calibration_bins)
    youden = evaluate_operating_point(
        targets, probabilities, threshold=float(thresholds["youden_j"])
    )
    sensitivity = evaluate_operating_point(
        targets,
        probabilities,
        threshold=float(thresholds["target_sensitivity"]),
    )
    return metrics_document(
        scope="test",
        calibration_bins=calibration_bins,
        sensitivity_target=sensitivity_target,
        thresholds={key: float(value) for key, value in thresholds.items()},
        probability=probability,
        youden=youden,
        target_sensitivity=sensitivity,
    )


def publish_rsna_evaluation(
    *,
    report_root: str | Path,
    private_root: str | Path,
    model_root: str | Path,
    evidence: ValidatedPredictionEvidence,
    evaluation_policy: Mapping[str, Any],
    forbidden_source_values: Sequence[str],
) -> ValidatedEvaluationResult:
    """Publish one evidence-and-policy-identified held-out evaluation."""
    package = validate_rsna_model_package(model_root, evidence.manifest["model_package_id"])
    frozen_threshold_contract = validated_threshold_contract(
        package["threshold_contract"], positive_class=package["positive_class"]
    )
    policy = _validate_evaluation_policy(evaluation_policy)
    if policy["threshold_selection"] != frozen_threshold_contract:
        raise ValueError("Evaluation threshold-selection policy differs from frozen package state")
    thresholds = package["thresholds"]
    semantic = {
        "dataset_id": evidence.manifest["dataset_id"],
        "family_id": package["family_id"],
        "modalities": package["modalities"],
        "seed": _package_seed(package),
        "model_package_id": evidence.manifest["model_package_id"],
        "prediction_id": evidence.manifest["prediction_id"],
        "task_id": evidence.manifest["task_id"],
        "bundle_id": evidence.manifest["bundle_id"],
        "split_assignment_id": evidence.manifest["split_assignment_id"],
        "scope": evidence.manifest["scope"],
        "evaluation_policy": policy,
        "thresholds": {key: float(value) for key, value in thresholds.items()},
    }
    _validate_evaluation_policy(policy)
    frozen_thresholds = _validated_thresholds(thresholds)
    _validate_package_evidence(package, evidence, frozen_thresholds, policy)
    claims = derive_evaluation_claims(
        evidence,
        evaluation_policy=policy,
        thresholds=frozen_thresholds,
    )
    semantic["thresholds"] = frozen_thresholds
    evaluation_id = canonical_scientific_id(EVALUATION_PREFIX, semantic)
    document = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        **semantic,
        "evaluation_id": evaluation_id,
        "prediction_manifest_sha256": evidence.manifest_sha256,
        "claims": claims,
    }
    destination = Path(report_root) / "rsna" / "evaluations" / evaluation_id
    stage = staging_directory(destination)
    try:
        derivatives = stage / DERIVATIVES_DIRECTORY
        derivatives.mkdir()
        frame = evidence.predictions.to_pandas()
        write_run_reports(
            derivatives,
            model_name=str(evidence.manifest["model_package_id"]),
            targets=frame["target"].to_numpy(dtype=np.int8),
            probabilities=frame["probability"].to_numpy(dtype=np.float64),
            document=claims,
        )
        validate_report_set(derivatives)
        validate_public_reports(
            derivatives.iterdir(), forbidden_source_values=forbidden_source_values
        )
        (stage / EVALUATION_MANIFEST_FILENAME).write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        validate_rsna_evaluation(
            stage,
            private_root=private_root,
            model_root=model_root,
            enforce_directory_name=False,
        )
        created = install_immutable_directory(
            stage,
            destination,
            lambda path, **kwargs: validate_rsna_evaluation(
                path,
                private_root=private_root,
                model_root=model_root,
                **kwargs,
            ),
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return replace(
        validate_rsna_evaluation(
            destination,
            private_root=private_root,
            model_root=model_root,
            expected_evaluation_id=evaluation_id,
        ),
        created=created,
    )


def validate_rsna_evaluation(
    directory: str | Path,
    *,
    private_root: str | Path,
    model_root: str | Path,
    expected_evaluation_id: str | None = None,
    enforce_directory_name: bool = True,
) -> ValidatedEvaluationResult:
    """Re-derive and validate one evaluation from its exact prediction evidence."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Evaluation result has an invalid artifact set")
    with os.scandir(root) as entries:
        inspected = {entry.name: entry for entry in entries}
    if set(inspected) != {EVALUATION_MANIFEST_FILENAME, DERIVATIVES_DIRECTORY}:
        raise ValueError("Evaluation result has an invalid artifact set")
    manifest_entry = inspected[EVALUATION_MANIFEST_FILENAME]
    derivatives_entry = inspected[DERIVATIVES_DIRECTORY]
    if (
        manifest_entry.is_symlink()
        or not manifest_entry.is_file(follow_symlinks=False)
        or derivatives_entry.is_symlink()
        or not derivatives_entry.is_dir(follow_symlinks=False)
    ):
        raise ValueError("Evaluation result has an invalid artifact set")
    manifest_bytes = (root / EVALUATION_MANIFEST_FILENAME).read_bytes()
    try:
        document = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Evaluation manifest is unreadable") from exc
    if not isinstance(document, dict) or set(document) != _FIELDS:
        raise ValueError("Evaluation manifest has an unexpected field set")
    schema_version = document["evaluation_schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != EVALUATION_SCHEMA_VERSION
    ):
        raise ValueError("Evaluation schema version is invalid")
    prediction_id = document["prediction_id"]
    _require_identity(document["evaluation_id"], EVALUATION_PREFIX, "evaluation")
    _require_identity(prediction_id, "prediction-", "prediction")
    _require_identity(document["model_package_id"], "model-package-", "model package")
    evidence = validate_prediction_evidence(
        Path(private_root) / "predictions" / "rsna" / prediction_id,
        expected_prediction_id=prediction_id,
        expected_model_package_id=document["model_package_id"],
    )
    if evidence.manifest_sha256 != document["prediction_manifest_sha256"]:
        raise ValueError("Evaluation prediction-manifest integrity witness differs")
    package = validate_rsna_model_package(model_root, document["model_package_id"])
    policy = _validate_evaluation_policy(document["evaluation_policy"])
    thresholds = _validated_thresholds(document["thresholds"])
    _validate_package_evidence(package, evidence, thresholds, policy)
    expected_lineage = {
        "dataset_id": package["dataset_id"],
        "family_id": package["family_id"],
        "modalities": package["modalities"],
        "seed": _package_seed(package),
        "model_package_id": package["model_package_id"],
        "prediction_id": evidence.prediction_id,
        "task_id": package["task_id"],
        "bundle_id": package["bundle_id"],
        "split_assignment_id": package["split_assignment_id"],
        "scope": "test",
    }
    if any(document[field] != value for field, value in expected_lineage.items()):
        raise ValueError("Evaluation manifest lineage differs from package and evidence")
    semantic = {
        key: document[key]
        for key in _FIELDS
        if key
        not in {
            "evaluation_schema_version",
            "evaluation_id",
            "prediction_manifest_sha256",
            "claims",
        }
    }
    evaluation_id = canonical_scientific_id(EVALUATION_PREFIX, semantic)
    if document["evaluation_id"] != evaluation_id:
        raise ValueError("Evaluation identity differs from evidence and policy")
    if expected_evaluation_id is not None and evaluation_id != expected_evaluation_id:
        raise ValueError("Evaluation differs from the expected identity")
    if enforce_directory_name and root.name != evaluation_id:
        raise ValueError("Evaluation directory differs from its semantic identity")
    claims = derive_evaluation_claims(
        evidence,
        evaluation_policy=policy,
        thresholds=thresholds,
    )
    if claims != document["claims"]:
        raise ValueError("Evaluation claims differ from re-derived evidence")
    validate_report_set(root / DERIVATIVES_DIRECTORY)
    with tempfile.TemporaryDirectory(prefix="radfusion-evaluation-validation-") as temporary:
        expected = Path(temporary)
        frame = evidence.predictions.to_pandas()
        write_run_reports(
            expected,
            model_name=str(document["model_package_id"]),
            targets=frame["target"].to_numpy(dtype=np.int8),
            probabilities=frame["probability"].to_numpy(dtype=np.float64),
            document=claims,
        )
        validate_report_set(expected)
        derivatives = root / DERIVATIVES_DIRECTORY
        if any(
            (derivatives / path.name).read_bytes() != path.read_bytes()
            for path in expected.iterdir()
        ):
            raise ValueError("Evaluation derivatives differ from deterministic claims")
    return ValidatedEvaluationResult(
        root,
        document,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def validate_rsna_model_package(model_root: str | Path, package_id: object) -> dict[str, Any]:
    """Resolve and fully validate one RSNA tabular or neural model package."""
    _require_identity(package_id, "model-package-", "model package")
    assert isinstance(package_id, str)
    root = Path(model_root) / "packages" / package_id
    if (root / TABULAR_MODEL_FILENAME).is_file():
        package = validate_published_model(root)
    else:
        package = validate_published_neural_model(root)
    if package["model_package_id"] != package_id:
        raise ValueError("Evaluation resolved the wrong model package")
    return package


def validated_rsna_evaluation_policy(
    package: Mapping[str, Any], evaluation_config: ExperimentConfig
) -> dict[str, Any]:
    """Validate explicit evaluation authority against package-owned fit semantics."""
    if evaluation_config.evaluation is None:
        raise ValueError("RSNA evaluation configuration has no scientific evaluation policy")
    expected = {
        "dataset_id": evaluation_config.dataset.dataset_id,
        "bundle_id": evaluation_config.dataset.bundle_id,
        "bundle_manifest_sha256": evaluation_config.dataset.bundle_manifest_sha256,
        "split_assignment_id": evaluation_config.dataset.split_assignment_id,
        "task_id": evaluation_config.task.task_id,
        "label_policy_version": evaluation_config.task.label_policy_version,
        "family_id": evaluation_config.family.family_id,
        "modalities": list(evaluation_config.family.modalities),
        "fit_config": package_scientific_config_payload(evaluation_config),
    }
    if any(package.get(field) != value for field, value in expected.items()):
        raise ValueError("Evaluation configuration is incompatible with the model package")
    threshold_selection = validated_threshold_contract(
        package["threshold_contract"], positive_class=package["positive_class"]
    )
    if evaluation_config.evaluation.sensitivity_target != threshold_selection["sensitivity_target"]:
        raise ValueError("Evaluation sensitivity target differs from the frozen model package")
    return _evaluation_policy(
        evaluation_config.evaluation.calibration_bins,
        threshold_selection,
    )


def _package_seed(package: Mapping[str, Any]) -> int:
    value = package.get("seed")
    if value is None and isinstance(package.get("training_policy"), Mapping):
        value = package["training_policy"].get("seed")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Model package seed coordinate is invalid")
    return value


def _validate_package_evidence(
    package: Mapping[str, Any],
    evidence: ValidatedPredictionEvidence,
    thresholds: Mapping[str, float],
    evaluation_policy: Mapping[str, Any],
) -> None:
    manifest = evidence.manifest
    expected = {
        "dataset_id": "rsna",
        "model_package_id": package["model_package_id"],
        "task_id": package["task_id"],
        "bundle_id": package["bundle_id"],
        "split_assignment_id": package["split_assignment_id"],
        "scope": "test",
    }
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise ValueError("Evaluation package and prediction lineage differ")
    package_threshold_contract = validated_threshold_contract(
        package["threshold_contract"], positive_class=package["positive_class"]
    )
    if package["dataset_id"] != "rsna" or package["thresholds"] != dict(thresholds):
        raise ValueError("Evaluation thresholds differ from frozen package state")
    if evaluation_policy["threshold_selection"] != package_threshold_contract:
        raise ValueError("Evaluation threshold-selection policy differs from frozen package state")


def _validate_evaluation_policy(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "policy_version",
        "calibration_bins",
        "threshold_selection",
    }:
        raise ValueError("Evaluation policy is invalid")
    bins = value["calibration_bins"]
    if (
        value["policy_version"] != EVALUATION_POLICY_VERSION
        or isinstance(bins, bool)
        or not isinstance(bins, int)
        or not 2 <= bins <= 1000
    ):
        raise ValueError("Evaluation policy is invalid")
    threshold_selection = validated_threshold_contract(
        value["threshold_selection"], positive_class=1
    )
    return {
        "policy_version": EVALUATION_POLICY_VERSION,
        "calibration_bins": bins,
        "threshold_selection": threshold_selection,
    }


def _evaluation_policy(
    calibration_bins: object, threshold_selection: Mapping[str, Any]
) -> dict[str, Any]:
    return _validate_evaluation_policy(
        {
            "policy_version": EVALUATION_POLICY_VERSION,
            "calibration_bins": calibration_bins,
            "threshold_selection": dict(threshold_selection),
        }
    )


def _validated_thresholds(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"youden_j", "target_sensitivity"}:
        raise ValueError("Evaluation thresholds are invalid")
    result = dict(value)
    if any(
        isinstance(item, bool)
        or not isinstance(item, int | float)
        or not np.isfinite(item)
        or not 0.0 <= item <= 1.0
        for item in result.values()
    ):
        raise ValueError("Evaluation thresholds are invalid")
    return {key: float(item) for key, item in result.items()}


def _require_identity(value: object, prefix: str, name: str) -> str:
    validate_path_component(value, name)
    assert isinstance(value, str)
    suffix = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or len(suffix) != 64
        or any(character not in "0123456789abcdef" for character in suffix)
    ):
        raise ValueError(f"{name} identity is invalid")
    return value
