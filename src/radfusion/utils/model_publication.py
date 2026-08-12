"""Publish and validate compact run-qualified local model packages."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radfusion.data.hashing import sha256_file
from radfusion.data.tabular_preprocess import metadata_input_contract
from radfusion.utils.skops_io import trusted_types_for_file

MODEL_FILENAME = "model.skops"
CONFIG_FILENAME = "resolved_config.yaml"
MANIFEST_FILENAME = "model_manifest.json"
MODEL_PACKAGE_SCHEMA_VERSION = 1
MODEL_PACKAGE_ID_PREFIX = "model-package-"
REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "model_package_schema_version",
        "model_package_id",
        "model_sha256",
        "training_mlflow_run_id",
        "bundle_id",
        "split_assignment_id",
        "task",
        "positive_class",
        "model",
        "config_source_sha256",
        "config_semantic_sha256",
        "seed",
        "git_commit",
        "git_dirty",
        "dependency_lock_sha256",
        "best_iteration",
        "thresholds",
        "threshold_contract",
        "input_contract",
    }
)
_IDENTITY_FIELDS = REQUIRED_MANIFEST_FIELDS - {"model_package_id"}
YOUDEN_J_POLICY_VERSION = "youden-j-all-roc-highest-finite-tie-v1"
TARGET_SENSITIVITY_POLICY_VERSION = "target-sensitivity-all-roc-highest-finite-v1"


@dataclass(frozen=True)
class PublishedModel:
    """Paths and physical identity for one local training-run package."""

    run_directory: Path
    model_path: Path
    config_path: Path
    manifest_path: Path
    model_package_id: str
    model_sha256: str
    model_size_mib: float


def publish_model_run(
    *,
    model_root: str | Path,
    mlflow_run_id: str,
    serialized_model_path: str | Path,
    source_config_bytes: bytes,
    manifest: Mapping[str, Any],
) -> PublishedModel:
    """Publish one complete model package atomically."""
    _validate_component(mlflow_run_id, "mlflow_run_id")
    runs_root = Path(model_root) / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    final = runs_root / mlflow_run_id
    stage = Path(tempfile.mkdtemp(prefix=f".{mlflow_run_id}-staging-", dir=runs_root))
    try:
        model_path = stage / MODEL_FILENAME
        config_path = stage / CONFIG_FILENAME
        shutil.copyfile(serialized_model_path, model_path)
        config_path.write_bytes(source_config_bytes)
        trusted_types_for_file(model_path)
        document = {
            **dict(manifest),
            "model_package_schema_version": MODEL_PACKAGE_SCHEMA_VERSION,
            "training_mlflow_run_id": mlflow_run_id,
            "model_sha256": sha256_file(model_path),
        }
        document["model_package_id"] = model_package_id(document)
        _validate_manifest(document, mlflow_run_id, model_path, config_path)
        (stage / MANIFEST_FILENAME).write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if final.exists():
            existing = validate_published_model(final)
            if existing != document:
                raise FileExistsError(f"Model run exists with conflicting content: {final}")
        else:
            os.replace(stage, final)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    model_path = final / MODEL_FILENAME
    return PublishedModel(
        run_directory=final,
        model_path=model_path,
        config_path=final / CONFIG_FILENAME,
        manifest_path=final / MANIFEST_FILENAME,
        model_package_id=document["model_package_id"],
        model_sha256=sha256_file(model_path),
        model_size_mib=model_path.stat().st_size / (1024.0 * 1024.0),
    )


def validate_published_model(run_directory: str | Path) -> dict[str, Any]:
    """Validate one run-qualified model package and return its manifest."""
    directory = Path(run_directory)
    if directory.parent.name != "runs" or directory.is_symlink() or not directory.is_dir():
        raise ValueError("Model run must be a physical directory beneath runs")
    expected = {MODEL_FILENAME, CONFIG_FILENAME, MANIFEST_FILENAME}
    with os.scandir(directory) as entries:
        inspected = list(entries)
    if {entry.name for entry in inspected} != expected:
        raise ValueError("Model run contains an unexpected artifact set")
    if any(entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in inspected):
        raise ValueError("Model run entries must be regular non-symlink files")
    try:
        document = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Model manifest is unreadable") from exc
    _validate_manifest(
        document,
        directory.name,
        directory / MODEL_FILENAME,
        directory / CONFIG_FILENAME,
    )
    trusted_types_for_file(directory / MODEL_FILENAME)
    return document


def model_package_id(document: Mapping[str, Any]) -> str:
    """Return the semantic identity of one complete model-package manifest."""
    if set(document) not in {_IDENTITY_FIELDS, REQUIRED_MANIFEST_FIELDS}:
        raise ValueError("Model package identity payload contains an unexpected field set")
    payload = {field: document[field] for field in sorted(_IDENTITY_FIELDS)}
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Model package identity payload is not canonical JSON") from exc
    return MODEL_PACKAGE_ID_PREFIX + hashlib.sha256(encoded).hexdigest()


def _validate_manifest(
    document: Mapping[str, Any],
    run_id: str,
    model_path: Path,
    config_path: Path,
) -> None:
    if set(document) != REQUIRED_MANIFEST_FIELDS:
        raise ValueError("Model manifest contains an unexpected field set")
    schema_version = document["model_package_schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != MODEL_PACKAGE_SCHEMA_VERSION
    ):
        raise ValueError("Model package schema version must be the supported integer")
    if document["training_mlflow_run_id"] != run_id:
        raise ValueError("Model manifest run ID does not match its path")
    for field in ("bundle_id", "split_assignment_id", "task", "model", "git_commit"):
        if not isinstance(document[field], str) or not document[field]:
            raise ValueError(f"Model manifest {field} must be a non-empty string")
    for field in (
        "model_sha256",
        "config_source_sha256",
        "config_semantic_sha256",
        "dependency_lock_sha256",
    ):
        if not _is_sha256(document[field]):
            raise ValueError(f"Model manifest {field} must be a lowercase SHA-256")
    if document["model_sha256"] != sha256_file(model_path):
        raise ValueError("Model SHA-256 does not match model bytes")
    if document["config_source_sha256"] != sha256_file(config_path):
        raise ValueError("Source config SHA-256 does not match archived config bytes")
    positive_class = document["positive_class"]
    if (
        isinstance(positive_class, bool)
        or not isinstance(positive_class, int)
        or positive_class != 1
    ):
        raise ValueError("Model manifest positive class must be integer 1")
    if not isinstance(document["git_dirty"], bool):
        raise ValueError("Model manifest git_dirty must be Boolean")
    if isinstance(document["seed"], bool) or not isinstance(document["seed"], int):
        raise ValueError("Model manifest seed must be an integer")
    best_iteration = document["best_iteration"]
    if best_iteration is not None and (
        isinstance(best_iteration, bool)
        or not isinstance(best_iteration, int)
        or best_iteration <= 0
    ):
        raise ValueError("Model manifest best_iteration must be null or positive")
    thresholds = document["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != {
        "youden_j",
        "target_sensitivity",
    }:
        raise ValueError("Model manifest thresholds are invalid")
    for value in thresholds.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("Model manifest thresholds must be finite probabilities")
    _validate_threshold_contract(document["threshold_contract"], document["positive_class"])
    if document["input_contract"] != metadata_input_contract():
        raise ValueError("Model manifest input contract is invalid")
    if document["model_package_id"] != model_package_id(document):
        raise ValueError("Model package ID does not match its semantic payload")


def threshold_contract(*, sensitivity_target: float, positive_class: int = 1) -> dict[str, Any]:
    """Return the compact contract used to derive validation thresholds."""
    return {
        "youden_j_policy_version": YOUDEN_J_POLICY_VERSION,
        "target_sensitivity_policy_version": TARGET_SENSITIVITY_POLICY_VERSION,
        "sensitivity_target": sensitivity_target,
        "positive_class": positive_class,
    }


def _validate_threshold_contract(contract: object, positive_class: object) -> None:
    if not isinstance(contract, dict) or set(contract) != {
        "youden_j_policy_version",
        "target_sensitivity_policy_version",
        "sensitivity_target",
        "positive_class",
    }:
        raise ValueError("Model manifest threshold contract is invalid")
    if contract["youden_j_policy_version"] != YOUDEN_J_POLICY_VERSION:
        raise ValueError("Model manifest Youden-J threshold policy is invalid")
    if contract["target_sensitivity_policy_version"] != TARGET_SENSITIVITY_POLICY_VERSION:
        raise ValueError("Model manifest target-sensitivity threshold policy is invalid")
    sensitivity = contract["sensitivity_target"]
    if (
        isinstance(sensitivity, bool)
        or not isinstance(sensitivity, int | float)
        or not math.isfinite(sensitivity)
        or not 0.0 < sensitivity <= 1.0
    ):
        raise ValueError("Model manifest sensitivity target is invalid")
    threshold_positive_class = contract["positive_class"]
    if (
        isinstance(threshold_positive_class, bool)
        or not isinstance(threshold_positive_class, int)
        or threshold_positive_class != positive_class
    ):
        raise ValueError("Model manifest threshold positive class is invalid")


def _validate_component(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).name != value
    ):
        raise ValueError(f"{field} must be one safe path component")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
