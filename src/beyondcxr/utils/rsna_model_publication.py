"""Publish and validate immutable semantic RSNA tabular model packages."""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from beyondcxr.data.hashing import sha256_file
from beyondcxr.data.rsna_metadata_preprocess import (
    metadata_input_contract,
    validate_metadata_pipeline,
)
from beyondcxr.training.config import load_experiment_config, with_runtime
from beyondcxr.training.rsna_validation_evidence import (
    VALIDATION_EVIDENCE_FILENAME,
    validate_validation_evidence,
    validation_evidence_id,
)
from beyondcxr.utils.package_identity import (
    fitted_object_state_sha256,
    package_scientific_config_payload,
)
from beyondcxr.utils.publication import (
    install_immutable_directory,
    staging_directory,
)
from beyondcxr.utils.semantic_identity import canonical_scientific_id
from beyondcxr.utils.skops_io import load_skops, trusted_types_for_file

MODEL_FILENAME = "model.skops"
CONFIG_FILENAME = "resolved_config.yaml"
MANIFEST_FILENAME = "manifest.json"
MODEL_PACKAGE_SCHEMA_VERSION = 1
MODEL_PACKAGE_ID_PREFIX = "model-package-"
YOUDEN_J_POLICY_VERSION = "youden-j-all-roc-highest-finite-tie-v1"
TARGET_SENSITIVITY_POLICY_VERSION = "target-sensitivity-all-roc-highest-finite-v1"
REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "model_package_schema_version",
        "model_package_id",
        "dataset_id",
        "bundle_id",
        "task_id",
        "label_policy_version",
        "positive_class",
        "family_id",
        "modalities",
        "seed",
        "fit_config",
        "preprocessor_state_sha256",
        "model_state_sha256",
        "source_package_id",
        "model_sha256",
        "validation_evidence_sha256",
        "validation_evidence_id",
        "git_commit",
        "git_dirty",
        "dependency_lock_sha256",
        "python_version",
        "best_iteration",
        "thresholds",
        "threshold_contract",
        "input_contract",
    }
)


@dataclass(frozen=True)
class PublishedModel:
    """Paths and identities for one immutable semantic model package."""

    package_directory: Path
    model_path: Path
    config_path: Path
    manifest_path: Path
    model_package_id: str
    model_sha256: str
    model_size_mib: float
    created: bool


def publish_model_package(
    *,
    model_root: str | Path,
    serialized_model_path: str | Path,
    source_config_bytes: bytes,
    validation_evidence_path: str | Path,
    manifest: Mapping[str, Any],
) -> PublishedModel:
    """Publish one fitted model under its semantic package identity."""
    packages_root = Path(model_root) / "packages"
    packages_root.mkdir(parents=True, exist_ok=True)
    stage = staging_directory(packages_root / "model-package-pending")
    try:
        model_path = stage / MODEL_FILENAME
        config_path = stage / CONFIG_FILENAME
        shutil.copyfile(serialized_model_path, model_path)
        evidence_path = stage / VALIDATION_EVIDENCE_FILENAME
        shutil.copyfile(validation_evidence_path, evidence_path)
        config_path.write_bytes(source_config_bytes)
        trusted_types_for_file(model_path)
        config = with_runtime(
            load_experiment_config(config_path, bundle_id=manifest["bundle_id"]),
            seed=int(manifest["seed"]),
        )
        fitted = _validate_fitted_model(load_skops(model_path), config.family.family_id)
        preprocessor = fitted.named_steps["preprocess"]
        estimator = fitted.named_steps["classifier"]
        document = {
            **dict(manifest),
            "model_package_schema_version": MODEL_PACKAGE_SCHEMA_VERSION,
            "dataset_id": config.dataset.dataset_id,
            "label_policy_version": config.task.label_policy_version,
            "modalities": list(config.family.modalities),
            "fit_config": package_scientific_config_payload(config),
            "preprocessor_state_sha256": fitted_object_state_sha256(preprocessor),
            "model_state_sha256": fitted_object_state_sha256(
                estimator, selected_iteration=manifest["best_iteration"]
            ),
            "source_package_id": None,
            "model_sha256": sha256_file(model_path),
            "validation_evidence_sha256": sha256_file(evidence_path),
            "validation_evidence_id": validation_evidence_id(
                json.loads(evidence_path.read_bytes())
            ),
            "python_version": platform.python_version(),
        }
        document["model_package_id"] = model_package_id(document)
        destination = packages_root / document["model_package_id"]
        _validate_manifest(document, model_path, config_path, evidence_path)
        (stage / MANIFEST_FILENAME).write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        created = install_immutable_directory(stage, destination, validate_published_model)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    final_model = destination / MODEL_FILENAME
    return PublishedModel(
        package_directory=destination,
        model_path=final_model,
        config_path=destination / CONFIG_FILENAME,
        manifest_path=destination / MANIFEST_FILENAME,
        model_package_id=document["model_package_id"],
        model_sha256=sha256_file(final_model),
        model_size_mib=final_model.stat().st_size / (1024.0 * 1024.0),
        created=created,
    )


def validate_published_model(
    package_directory: str | Path, *, enforce_directory_name: bool = True
) -> dict[str, Any]:
    """Validate one reconstructable tabular model package."""
    directory = Path(package_directory)
    if directory.parent.name != "packages" or directory.is_symlink() or not directory.is_dir():
        raise ValueError("Model package must be a physical directory beneath packages")
    expected = {
        MODEL_FILENAME,
        CONFIG_FILENAME,
        MANIFEST_FILENAME,
        VALIDATION_EVIDENCE_FILENAME,
    }
    with os.scandir(directory) as entries:
        inspected = list(entries)
    if {entry.name for entry in inspected} != expected or any(
        entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in inspected
    ):
        raise ValueError("Model package contains an invalid artifact set")
    try:
        document = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Model manifest is unreadable") from exc
    _validate_manifest(
        document,
        directory / MODEL_FILENAME,
        directory / CONFIG_FILENAME,
        directory / VALIDATION_EVIDENCE_FILENAME,
    )
    if enforce_directory_name and directory.name != document["model_package_id"]:
        raise ValueError("Model package directory differs from its semantic identity")
    trusted_types_for_file(directory / MODEL_FILENAME)
    return document


def model_package_id(document: Mapping[str, Any]) -> str:
    """Return semantic identity from fit meaning and reconstructed fitted state."""
    required = REQUIRED_MANIFEST_FIELDS - {"model_package_id"}
    if set(document) not in {required, REQUIRED_MANIFEST_FIELDS}:
        raise ValueError("Model package identity payload has an unexpected field set")
    payload = {
        "dataset_id": document["dataset_id"],
        "bundle_id": document["bundle_id"],
        "task_id": document["task_id"],
        "label_policy_version": document["label_policy_version"],
        "positive_class": document["positive_class"],
        "family_id": document["family_id"],
        "modalities": document["modalities"],
        "seed": document["seed"],
        "fit_config": document["fit_config"],
        "preprocessor_state_sha256": document["preprocessor_state_sha256"],
        "model_state_sha256": document["model_state_sha256"],
        "source_package_id": document["source_package_id"],
        "best_iteration": document["best_iteration"],
        "thresholds": document["thresholds"],
        "threshold_contract": document["threshold_contract"],
        "validation_evidence_id": document["validation_evidence_id"],
    }
    return canonical_scientific_id(
        MODEL_PACKAGE_ID_PREFIX, payload, domain="model-package", schema_version=1
    )


def _validate_manifest(
    document: object, model_path: Path, config_path: Path, evidence_path: Path
) -> None:
    if not isinstance(document, dict) or set(document) != REQUIRED_MANIFEST_FIELDS:
        raise ValueError("Model manifest contains an unexpected field set")
    schema_version = document["model_package_schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != MODEL_PACKAGE_SCHEMA_VERSION
    ):
        raise ValueError("Model package schema version is invalid")
    config = with_runtime(
        load_experiment_config(config_path, bundle_id=document["bundle_id"]), seed=document["seed"]
    )
    expected = {
        "dataset_id": config.dataset.dataset_id,
        "bundle_id": config.dataset.bundle_id,
        "task_id": config.task.task_id,
        "label_policy_version": config.task.label_policy_version,
        "family_id": config.family.family_id,
        "modalities": list(config.family.modalities),
        "fit_config": package_scientific_config_payload(config),
    }
    if any(document[key] != value for key, value in expected.items()):
        raise ValueError("Model package differs from its archived fit configuration")
    if document["positive_class"] != 1 or document["source_package_id"] is not None:
        raise ValueError("Tabular model package semantic contract is invalid")
    for field in (
        "model_sha256",
        "preprocessor_state_sha256",
        "model_state_sha256",
        "dependency_lock_sha256",
        "validation_evidence_sha256",
    ):
        if not _is_sha256(document[field]):
            raise ValueError(f"Model manifest {field} must be a lowercase SHA-256")
    validation_id = document["validation_evidence_id"]
    if (
        not isinstance(validation_id, str)
        or not validation_id.startswith("validation-evidence-")
        or len(validation_id) != len("validation-evidence-") + 64
        or any(character not in "0123456789abcdef" for character in validation_id[20:])
    ):
        raise ValueError("Model manifest validation evidence identity is invalid")
    if document["model_sha256"] != sha256_file(model_path):
        raise ValueError("Model SHA-256 does not match model bytes")
    fitted = _validate_fitted_model(load_skops(model_path), config.family.family_id)
    if (
        fitted_object_state_sha256(fitted.named_steps["preprocess"])
        != document["preprocessor_state_sha256"]
        or fitted_object_state_sha256(
            fitted.named_steps["classifier"], selected_iteration=document["best_iteration"]
        )
        != document["model_state_sha256"]
    ):
        raise ValueError("Reconstructed fitted state differs from semantic package identity")
    if not isinstance(document["git_dirty"], bool):
        raise ValueError("Model manifest git_dirty must be Boolean")
    if not isinstance(document["git_commit"], str) or not document["git_commit"]:
        raise ValueError("Model manifest git_commit must be non-empty text")
    if not isinstance(document["python_version"], str) or not document["python_version"]:
        raise ValueError("Model manifest python_version must be non-empty text")
    best_iteration = document["best_iteration"]
    if best_iteration is not None and (
        isinstance(best_iteration, bool)
        or not isinstance(best_iteration, int)
        or best_iteration <= 0
    ):
        raise ValueError("Model manifest best_iteration must be null or positive")
    fitted_best_iteration = getattr(fitted.named_steps["classifier"], "best_iteration_", None)
    if fitted_best_iteration != best_iteration:
        raise ValueError("Model manifest best iteration differs from fitted estimator state")
    thresholds = document["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != {"youden_j", "target_sensitivity"}:
        raise ValueError("Model manifest thresholds are invalid")
    if any(not _probability(value) for value in thresholds.values()):
        raise ValueError("Model manifest thresholds must be finite probabilities")
    contract = validated_threshold_contract(
        document["threshold_contract"], positive_class=document["positive_class"]
    )
    if contract["sensitivity_target"] != config.evaluation.sensitivity_target:
        raise ValueError("Model package threshold contract differs from archived configuration")
    validate_validation_evidence(evidence_path, package=document, config=config)
    if document["input_contract"] != metadata_input_contract():
        raise ValueError("Model manifest input contract is invalid")
    if document["model_package_id"] != model_package_id(document):
        raise ValueError("Model package ID does not match its semantic payload")


def _validate_fitted_model(model: object, family_id: str) -> Pipeline:
    fitted = validate_metadata_pipeline(model)
    classifier = fitted.named_steps["classifier"]
    expected_type = {
        "metadata_logistic": LogisticRegression,
        "metadata_lightgbm": LGBMClassifier,
    }.get(family_id)
    if expected_type is None or not isinstance(classifier, expected_type):
        raise ValueError("RSNA tabular classifier differs from its configured family")
    return fitted


def threshold_contract(*, sensitivity_target: float, positive_class: int = 1) -> dict[str, Any]:
    """Return the compact contract used to derive validation thresholds."""
    contract = {
        "youden_j_policy_version": YOUDEN_J_POLICY_VERSION,
        "target_sensitivity_policy_version": TARGET_SENSITIVITY_POLICY_VERSION,
        "sensitivity_target": sensitivity_target,
        "positive_class": positive_class,
    }
    return validated_threshold_contract(contract, positive_class=positive_class)


def validated_threshold_contract(contract: object, *, positive_class: object) -> dict[str, Any]:
    """Validate and return the one canonical frozen threshold-selection contract."""
    expected = {
        "youden_j_policy_version",
        "target_sensitivity_policy_version",
        "sensitivity_target",
        "positive_class",
    }
    if not isinstance(contract, dict) or set(contract) != expected:
        raise ValueError("Model manifest threshold contract is invalid")
    if (
        contract["youden_j_policy_version"] != YOUDEN_J_POLICY_VERSION
        or contract["target_sensitivity_policy_version"] != TARGET_SENSITIVITY_POLICY_VERSION
    ):
        raise ValueError("Model manifest threshold policy is invalid")
    if (
        not isinstance(contract["sensitivity_target"], float)
        or not _probability(contract["sensitivity_target"])
        or contract["sensitivity_target"] == 0
    ):
        raise ValueError("Model manifest sensitivity target is invalid")
    if (
        isinstance(contract["positive_class"], bool)
        or not isinstance(contract["positive_class"], int)
        or contract["positive_class"] != positive_class
    ):
        raise ValueError("Model manifest threshold positive class is invalid")
    return dict(contract)


def _probability(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
