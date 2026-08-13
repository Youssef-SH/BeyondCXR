"""Safely serialize, publish, and validate neural model packages."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.hashing import sha256_file
from radfusion.data.rsna_cxr_cache import CxrCacheSourceAuthentication
from radfusion.models.fusion_concat import (
    fusion_architecture_contract,
    fusion_structured_input_conversion_contract,
)
from radfusion.training.config import (
    ExperimentConfig,
    load_experiment_config,
    with_runtime,
)
from radfusion.training.execution import LoaderExecutionPolicy
from radfusion.utils.model_publication import threshold_contract

NEURAL_MODEL_FILENAME = "model.pt"
CONFIG_FILENAME = "resolved_config.yaml"
MANIFEST_FILENAME = "model_manifest.json"
STRUCTURED_PREPROCESSOR_FILENAME = "structured_preprocessor.skops"
NEURAL_CHECKPOINT_SCHEMA_VERSION = 1
NEURAL_PACKAGE_SCHEMA_VERSION = 1
NEURAL_PACKAGE_ID_PREFIX = "model-package-"
CHECKPOINT_FIELDS = frozenset(
    {
        "checkpoint_schema_version",
        "model_state_dict",
        "selected_epoch",
        "selected_stage",
        "validation_average_precision",
    }
)
NEURAL_MANIFEST_FIELDS = frozenset(
    {
        "model_package_schema_version",
        "model_package_id",
        "training_mlflow_run_id",
        "modality",
        "model",
        "task",
        "positive_class",
        "bundle_id",
        "bundle_manifest_sha256",
        "split_assignment_id",
        "label_policy_version",
        "config_source_sha256",
        "config_semantic_sha256",
        "checkpoint_sha256",
        "source_provenance",
        "model_identity",
        "input_contract",
        "training_transform_contract",
        "evaluation_transform_contract",
        "training_policy",
        "selection",
        "thresholds",
        "threshold_contract",
        "metrics_policy",
        "source_authentication",
        "runtime_provenance",
    }
)
FUSION_MANIFEST_FIELDS = NEURAL_MANIFEST_FIELDS | {
    "structured_preprocessor_sha256",
    "structured_preprocessor_contract",
    "structured_input_conversion",
    "fusion_architecture",
    "source_cxr_lineage",
}


@dataclass(frozen=True)
class PublishedNeuralModel:
    """Paths and identities for one immutable neural package."""

    run_directory: Path
    model_path: Path
    config_path: Path
    manifest_path: Path
    model_package_id: str
    checkpoint_sha256: str
    model_size_mib: float


def checkpoint_document(
    state_dict: Mapping[str, torch.Tensor],
    *,
    selected_epoch: int,
    selected_stage: str,
    validation_average_precision: float,
) -> dict[str, Any]:
    """Build and validate the exact safe neural checkpoint document."""
    document = {
        "checkpoint_schema_version": NEURAL_CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in state_dict.items()
        },
        "selected_epoch": selected_epoch,
        "selected_stage": selected_stage,
        "validation_average_precision": validation_average_precision,
    }
    _validate_checkpoint(document)
    return document


def save_neural_checkpoint(document: Mapping[str, Any], path: str | Path) -> Path:
    """Serialize one validated plain checkpoint dictionary."""
    _validate_checkpoint(document)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(document), destination)
    load_neural_checkpoint(destination)
    return destination


def load_neural_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a neural checkpoint safely on CPU and validate its exact schema."""
    try:
        document = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        raise ValueError("Neural checkpoint is unreadable by the safe tensor loader") from exc
    _validate_checkpoint(document)
    return document


def strict_load_checkpoint(model: torch.nn.Module, checkpoint: Mapping[str, Any]) -> None:
    """Strictly load a validated CPU state dictionary into a reconstructed model."""
    _validate_checkpoint(checkpoint)
    try:
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError("Neural checkpoint does not match the reconstructed model") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("Neural checkpoint contains missing or unexpected parameters")
    state = model.state_dict()
    if set(state) != set(checkpoint["model_state_dict"]):
        raise ValueError("Neural checkpoint structural verification failed")


def publish_neural_model_run(
    *,
    model_root: str | Path,
    mlflow_run_id: str,
    checkpoint_path: str | Path,
    source_config_bytes: bytes,
    manifest: Mapping[str, Any],
    structured_preprocessor_path: str | Path | None = None,
) -> PublishedNeuralModel:
    """Publish one newly created immutable neural model package."""
    _validate_component(mlflow_run_id, "mlflow_run_id")
    runs_root = Path(model_root) / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    final = runs_root / mlflow_run_id
    if final.exists():
        raise FileExistsError(f"Neural model package already exists: {final}")
    stage = Path(tempfile.mkdtemp(prefix=f".{mlflow_run_id}-staging-", dir=runs_root))
    try:
        model_path = stage / NEURAL_MODEL_FILENAME
        config_path = stage / CONFIG_FILENAME
        shutil.copyfile(checkpoint_path, model_path)
        config_path.write_bytes(source_config_bytes)
        modality = manifest.get("modality")
        if modality == "fusion":
            if structured_preprocessor_path is None:
                raise ValueError("Fusion publication requires a fitted structured preprocessor")
            _require_regular_file(structured_preprocessor_path, "structured preprocessor")
            preprocessor_path = stage / STRUCTURED_PREPROCESSOR_FILENAME
            shutil.copyfile(structured_preprocessor_path, preprocessor_path)
        elif modality == "image":
            if structured_preprocessor_path is not None:
                raise ValueError("Image publication does not accept a structured preprocessor")
        else:
            raise ValueError("Neural publication requires image or fusion modality")
        checkpoint = load_neural_checkpoint(model_path)
        document = {
            **dict(manifest),
            "model_package_schema_version": NEURAL_PACKAGE_SCHEMA_VERSION,
            "training_mlflow_run_id": mlflow_run_id,
            "checkpoint_sha256": sha256_file(model_path),
        }
        document["model_package_id"] = neural_model_package_id(document)
        _validate_manifest(document, mlflow_run_id, model_path, config_path, checkpoint)
        (stage / MANIFEST_FILENAME).write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, final)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return PublishedNeuralModel(
        run_directory=final,
        model_path=final / NEURAL_MODEL_FILENAME,
        config_path=final / CONFIG_FILENAME,
        manifest_path=final / MANIFEST_FILENAME,
        model_package_id=document["model_package_id"],
        checkpoint_sha256=document["checkpoint_sha256"],
        model_size_mib=(final / NEURAL_MODEL_FILENAME).stat().st_size / (1024.0 * 1024.0),
    )


def validate_published_neural_model(run_directory: str | Path) -> dict[str, Any]:
    """Validate one exact neural package without reconstructing its architecture."""
    document = validate_neural_package_metadata(run_directory)
    load_validated_neural_checkpoint(run_directory, document)
    return document


def validate_neural_package_metadata(run_directory: str | Path) -> dict[str, Any]:
    """Validate a neural package's exact files, manifest, and physical identities."""
    directory = Path(run_directory)
    if directory.parent.name != "runs" or directory.is_symlink() or not directory.is_dir():
        raise ValueError("Neural model package must be a physical directory beneath runs")
    with os.scandir(directory) as entries:
        inspected = list(entries)
    if any(entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in inspected):
        raise ValueError("Neural model package entries must be regular non-symlink files")
    actual = {entry.name for entry in inspected}
    if MANIFEST_FILENAME not in actual:
        raise ValueError("Neural model package is missing its manifest")
    try:
        document = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Neural model manifest is unreadable") from exc
    modality = document.get("modality") if isinstance(document, dict) else None
    expected = {NEURAL_MODEL_FILENAME, CONFIG_FILENAME, MANIFEST_FILENAME}
    if modality == "fusion":
        expected.add(STRUCTURED_PREPROCESSOR_FILENAME)
    if actual != expected:
        raise ValueError("Neural model package contains an unexpected artifact set")
    _validate_manifest_metadata(
        document,
        directory.name,
        directory / NEURAL_MODEL_FILENAME,
        directory / CONFIG_FILENAME,
    )
    if modality == "fusion" and sha256_file(
        directory / STRUCTURED_PREPROCESSOR_FILENAME
    ) != document.get("structured_preprocessor_sha256"):
        raise ValueError("Fusion structured preprocessor SHA-256 mismatch")
    return document


def load_validated_neural_checkpoint(
    run_directory: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Safely load a checkpoint and verify its binding to validated package metadata."""
    directory = Path(run_directory)
    checkpoint = load_neural_checkpoint(directory / NEURAL_MODEL_FILENAME)
    _validate_checkpoint_binding(manifest, checkpoint)
    return checkpoint


def neural_model_package_id(document: Mapping[str, Any]) -> str:
    """Return the deterministic exact provenance identity of a neural package."""
    manifest_fields = _manifest_fields(document.get("modality"))
    identity_fields = _identity_fields(manifest_fields)
    if set(document) not in {
        identity_fields,
        manifest_fields,
        manifest_fields - {"model_package_id"},
    }:
        raise ValueError("Neural package identity payload contains an unexpected field set")
    payload = {field: document[field] for field in sorted(identity_fields)}
    if document.get("modality") == "fusion":
        source_lineage = document["source_cxr_lineage"]
        if not isinstance(source_lineage, Mapping):
            raise ValueError("Fusion source CXR lineage must be a mapping")
        payload["source_cxr_lineage"] = {
            key: source_lineage[key] for key in sorted(source_lineage) if key != "training_run_id"
        }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Neural package identity payload is not canonical JSON") from exc
    return NEURAL_PACKAGE_ID_PREFIX + hashlib.sha256(encoded).hexdigest()


def _validate_checkpoint(document: object) -> None:
    if not isinstance(document, dict) or set(document) != CHECKPOINT_FIELDS:
        raise ValueError("Neural checkpoint contains an unexpected field set")
    version = document["checkpoint_schema_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != NEURAL_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("Neural checkpoint schema version is invalid")
    state = document["model_state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("Neural checkpoint state dictionary must be non-empty")
    for key, value in state.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Neural checkpoint state keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise ValueError("Neural checkpoint state values must be tensors")
        if value.device.type != "cpu" or value.requires_grad or not torch.isfinite(value).all():
            raise ValueError("Neural checkpoint tensors must be detached finite CPU tensors")
    epoch = document["selected_epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("Neural checkpoint selected epoch must be a positive integer")
    if document["selected_stage"] not in {"warmup", "fine_tune"}:
        raise ValueError("Neural checkpoint selected stage is invalid")
    average_precision = document["validation_average_precision"]
    if not _probability(average_precision):
        raise ValueError("Neural checkpoint validation Average Precision is invalid")


def _validate_manifest(
    document: object,
    run_id: str,
    model_path: Path,
    config_path: Path,
    checkpoint: Mapping[str, Any],
) -> None:
    _validate_manifest_metadata(document, run_id, model_path, config_path)
    _validate_checkpoint_binding(document, checkpoint)


def _validate_manifest_metadata(
    document: object,
    run_id: str,
    model_path: Path,
    config_path: Path,
) -> None:
    if not isinstance(document, dict):
        raise ValueError("Neural model manifest contains an unexpected field set")
    manifest_fields = _manifest_fields(document.get("modality"))
    if set(document) != manifest_fields:
        raise ValueError("Neural model manifest contains an unexpected field set")
    schema_version = document["model_package_schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != NEURAL_PACKAGE_SCHEMA_VERSION
    ):
        raise ValueError("Neural package schema version is invalid")
    if document["training_mlflow_run_id"] != run_id or document["modality"] not in {
        "image",
        "fusion",
    }:
        raise ValueError("Neural package run or modality identity is invalid")
    for field in (
        "model",
        "task",
        "bundle_id",
        "split_assignment_id",
        "label_policy_version",
    ):
        if not isinstance(document[field], str) or not document[field]:
            raise ValueError(f"Neural model manifest {field} must be a non-empty string")
    positive_class = document["positive_class"]
    if (
        isinstance(positive_class, bool)
        or not isinstance(positive_class, int)
        or positive_class != 1
    ):
        raise ValueError("Neural model manifest positive class must be integer 1")
    for field in (
        "bundle_manifest_sha256",
        "config_source_sha256",
        "config_semantic_sha256",
        "checkpoint_sha256",
    ):
        if not _is_sha256(document[field]):
            raise ValueError(f"Neural model manifest {field} must be a lowercase SHA-256")
    if document["checkpoint_sha256"] != sha256_file(model_path):
        raise ValueError("Neural checkpoint hash does not match package bytes")
    if document["config_source_sha256"] != sha256_file(config_path):
        raise ValueError("Neural config hash does not match package bytes")
    training_policy = document.get("training_policy")
    seed = training_policy.get("seed") if isinstance(training_policy, dict) else None
    config = with_runtime(load_experiment_config(config_path), seed=seed)
    if (
        document["bundle_id"] != config.dataset.bundle_id
        or document["task"] != config.task.task_id
        or document["model"] != config.family.family_id
        or document["config_semantic_sha256"] != config.config_semantic_sha256
    ):
        raise ValueError("Neural package identity differs from archived configuration")
    selection = document["selection"]
    if not isinstance(selection, dict) or set(selection) != {
        "selected_epoch",
        "selected_stage",
        "validation_average_precision",
    }:
        raise ValueError("Neural package selection contract is invalid")
    if (
        isinstance(selection["selected_epoch"], bool)
        or not isinstance(selection["selected_epoch"], int)
        or selection["selected_epoch"] <= 0
        or selection["selected_stage"] not in {"warmup", "fine_tune"}
        or not _probability(selection["validation_average_precision"])
    ):
        raise ValueError("Neural package selection values are invalid")
    thresholds = document["thresholds"]
    if (
        not isinstance(thresholds, dict)
        or set(thresholds)
        != {
            "youden_j",
            "target_sensitivity",
        }
        or any(not _probability(value) for value in thresholds.values())
    ):
        raise ValueError("Neural package thresholds are invalid")
    contract = document["threshold_contract"]
    expected_contract_fields = {
        "youden_j_policy_version",
        "target_sensitivity_policy_version",
        "sensitivity_target",
        "positive_class",
    }
    if not isinstance(contract, dict) or set(contract) != expected_contract_fields:
        raise ValueError("Neural package threshold contract is invalid")
    sensitivity_target = contract["sensitivity_target"]
    threshold_positive_class = contract["positive_class"]
    if (
        not _probability(sensitivity_target)
        or sensitivity_target == 0.0
        or isinstance(threshold_positive_class, bool)
        or not isinstance(threshold_positive_class, int)
        or threshold_positive_class != 1
    ):
        raise ValueError("Neural package threshold contract is invalid")
    if contract != threshold_contract(sensitivity_target=float(sensitivity_target)):
        raise ValueError("Neural package threshold contract is unsupported")
    authentication = document["source_authentication"]
    try:
        CxrCacheSourceAuthentication.from_dict(authentication)
    except ValueError as exc:
        raise ValueError("Neural package source-authentication contract is invalid") from exc
    training_policy = document["training_policy"]
    if not isinstance(training_policy, dict):
        raise ValueError("Neural package training policy is invalid")
    seed = training_policy.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or training_policy.get("permitted_partitions") != ["train", "validation"]
    ):
        raise ValueError("Neural package training policy is invalid")
    _validate_nested_manifest(document, config)
    if document["model_package_id"] != neural_model_package_id(document):
        raise ValueError("Neural package ID does not match its identity payload")


def _validate_nested_manifest(document: dict[str, Any], config: ExperimentConfig) -> None:
    if config.neural is None:
        raise ValueError("Neural package archived configuration is not a neural experiment")
    modality = _package_modality(config)
    if document["modality"] != modality:
        raise ValueError("Neural package modality differs from archived configuration")
    source = _exact_mapping(
        document["source_provenance"],
        {
            "git_commit",
            "git_dirty",
            "dependency_lock_sha256",
            "python_version",
            "torch_version",
            "torchvision_version",
            "torchxrayvision_version",
        },
        "source provenance",
    )
    if not all(
        isinstance(source[field], str) and source[field] for field in source if field != "git_dirty"
    ):
        raise ValueError("Neural package source provenance contains invalid text")
    if not isinstance(source["git_dirty"], bool) or not _is_sha256(
        source["dependency_lock_sha256"]
    ):
        raise ValueError("Neural package source provenance is invalid")

    model = _exact_mapping(
        document["model_identity"],
        {
            "family_id",
            "modalities",
            "encoder_architecture",
            "image_size",
            "embedding_dimension",
            "classifier_output_dimension",
            "pretrained_weight",
        },
        "model identity",
    )
    expected_model = {
        "family_id": config.family.family_id,
        "modalities": list(config.family.modalities),
        "encoder_architecture": config.family.parameters["encoder_name"],
        "image_size": config.family.parameters["image_size"],
        "embedding_dimension": config.family.parameters["embedding_dimension"],
        "classifier_output_dimension": 1,
    }
    if any(model[field] != value for field, value in expected_model.items()):
        raise ValueError("Neural package model identity differs from archived configuration")
    weight = _exact_mapping(
        model["pretrained_weight"],
        {"declared_name", "stable_identifier", "cache_filename", "byte_size", "sha256"},
        "pretrained weight identity",
    )
    if (
        weight["declared_name"] != config.family.parameters["weights"]
        or not all(
            isinstance(weight[field], str) and weight[field]
            for field in ("stable_identifier", "cache_filename")
        )
        or isinstance(weight["byte_size"], bool)
        or not isinstance(weight["byte_size"], int)
        or weight["byte_size"] <= 0
        or not _is_sha256(weight["sha256"])
    ):
        raise ValueError("Neural package pretrained weight identity is invalid")

    image = config.neural
    selection = document["selection"]
    selected_epoch = selection["selected_epoch"]
    if (
        selected_epoch > image.warmup_epochs + image.fine_tune_epochs
        or (selection["selected_stage"] == "warmup" and selected_epoch > image.warmup_epochs)
        or (selection["selected_stage"] == "fine_tune" and selected_epoch <= image.warmup_epochs)
    ):
        raise ValueError("Neural package selected epoch is inconsistent with its stage")
    transform_kwargs = {
        "image_size": int(config.family.parameters["image_size"]),
        "rotation_degrees": image.rotation_degrees,
        "translation_fraction": image.translation_fraction,
        "brightness_jitter": image.brightness_jitter,
        "contrast_jitter": image.contrast_jitter,
    }
    expected_training_transform = StandardCxrTransform(training=True, **transform_kwargs).contract()
    expected_evaluation_transform = StandardCxrTransform(
        training=False, **transform_kwargs
    ).contract()
    if document["training_transform_contract"] != expected_training_transform:
        raise ValueError("Neural package training transform differs from archived configuration")
    if document["evaluation_transform_contract"] != expected_evaluation_transform:
        raise ValueError("Neural package evaluation transform differs from archived configuration")
    if document["input_contract"] != expected_evaluation_transform["input"]:
        raise ValueError("Neural package input contract is invalid")
    _validate_training_and_metrics(document, config, image)
    if modality == "fusion":
        _validate_fusion_manifest(document, config)


def _validate_fusion_manifest(document: dict[str, Any], config: ExperimentConfig) -> None:
    contract = document["structured_preprocessor_contract"]
    if (
        not isinstance(contract, dict)
        or not isinstance(contract.get("transformed_feature_names"), list)
        or isinstance(contract.get("transformed_dimension"), bool)
        or not isinstance(contract.get("transformed_dimension"), int)
        or contract["transformed_dimension"] <= 0
        or len(contract["transformed_feature_names"]) != contract["transformed_dimension"]
        or contract.get("output_structure") != "dense"
        or contract.get("output_dtype") != "float64"
        or not _is_sha256(document["structured_preprocessor_sha256"])
    ):
        raise ValueError("Fusion structured preprocessor identity is invalid")
    if document["structured_input_conversion"] != fusion_structured_input_conversion_contract():
        raise ValueError("Fusion structured tensor conversion contract is invalid")
    expected_architecture = fusion_architecture_contract(
        config.family,
        structured_input_dimension=contract["transformed_dimension"],
    )
    if document["fusion_architecture"] != expected_architecture:
        raise ValueError("Fusion architecture identity is invalid")
    lineage = _exact_mapping(
        document["source_cxr_lineage"],
        {
            "training_run_id",
            "model_package_id",
            "checkpoint_sha256",
            "config_semantic_sha256",
            "git_commit",
            "dependency_lock_sha256",
        },
        "source CXR lineage",
    )
    if not all(isinstance(value, str) and value for value in lineage.values()) or not all(
        _is_sha256(lineage[field])
        for field in (
            "checkpoint_sha256",
            "config_semantic_sha256",
            "dependency_lock_sha256",
        )
    ):
        raise ValueError("Fusion source CXR lineage is invalid")
    source = document["source_provenance"]
    if (
        lineage["git_commit"] != source["git_commit"]
        or lineage["dependency_lock_sha256"] != source["dependency_lock_sha256"]
    ):
        raise ValueError("Fusion source CXR revision differs from fusion package provenance")


def _manifest_fields(modality: object) -> frozenset[str]:
    if modality == "image":
        return NEURAL_MANIFEST_FIELDS
    if modality == "fusion":
        return FUSION_MANIFEST_FIELDS
    raise ValueError("Neural package has an invalid modality")


def _package_modality(config: ExperimentConfig) -> str:
    """Map the canonical family ontology to the frozen RSNA package modality."""
    if config.family.family_id == "cxr_densenet":
        return "image"
    if config.family.family_id == "cxr_metadata_concat":
        return "fusion"
    raise ValueError("Neural package archived configuration has an unsupported family")


def _identity_fields(manifest_fields: frozenset[str]) -> frozenset[str]:
    return manifest_fields - {
        "model_package_id",
        "runtime_provenance",
        "config_source_sha256",
        "training_mlflow_run_id",
    }


def _validate_training_and_metrics(
    document: dict[str, Any], config: ExperimentConfig, image: Any
) -> None:
    policy = _exact_mapping(
        document["training_policy"],
        {
            "seed",
            "permitted_partitions",
            "class_weight",
            "optimizer",
            "warmup",
            "fine_tuning",
            "weight_decay",
            "gradient_clip_norm",
            "scheduler",
            "early_stopping",
        },
        "training policy",
    )
    expected_policy = {
        "seed": config.runtime.seed,
        "permitted_partitions": ["train", "validation"],
        "optimizer": "AdamW",
        "warmup": {
            "epochs": image.warmup_epochs,
            "head_learning_rate": image.warmup_head_learning_rate,
            "encoder_frozen": True,
        },
        "fine_tuning": {
            "maximum_epochs": image.fine_tune_epochs,
            "encoder_learning_rate": image.encoder_learning_rate,
            "head_learning_rate": image.head_learning_rate,
        },
        "weight_decay": image.weight_decay,
        "gradient_clip_norm": image.gradient_clip_norm,
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "max",
            "factor": image.scheduler_factor,
            "patience": image.scheduler_patience,
            "min_lr": image.scheduler_min_learning_rate,
        },
        "early_stopping": {
            "metric": "validation_average_precision",
            "patience": image.early_stopping_patience,
            "minimum_delta": image.early_stopping_min_delta,
        },
    }
    if any(policy[field] != value for field, value in expected_policy.items()):
        raise ValueError("Neural package training policy differs from archived configuration")
    class_weight = _exact_mapping(
        policy["class_weight"],
        {"policy_version", "labels_used", "positive_count", "negative_count", "pos_weight"},
        "class weight policy",
    )
    positive = class_weight["positive_count"]
    negative = class_weight["negative_count"]
    if (
        class_weight["labels_used"] != "train"
        or class_weight["policy_version"] != "training-label-prevalence-pos-weight-v1"
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (positive, negative)
        )
        or not _finite_number(class_weight["pos_weight"])
        or not math.isclose(
            float(class_weight["pos_weight"]), negative / positive, rel_tol=1e-12, abs_tol=0.0
        )
    ):
        raise ValueError("Neural package class weight policy is invalid")

    metrics = _exact_mapping(
        document["metrics_policy"],
        {"version", "calibration_bins", "threshold_policy_version", "sensitivity_target"},
        "metrics policy",
    )
    if (
        metrics["version"] != "binary-probability-and-frozen-operating-points-v1"
        or metrics["threshold_policy_version"] != "validation-frozen-thresholds-v1"
        or metrics["calibration_bins"] != config.evaluation.calibration_bins
        or metrics["sensitivity_target"] != config.evaluation.sensitivity_target
        or document["threshold_contract"]["sensitivity_target"]
        != config.evaluation.sensitivity_target
    ):
        raise ValueError("Neural package metrics policy is invalid")
    _validate_runtime_provenance(document["runtime_provenance"])


def _validate_runtime_provenance(value: object) -> None:
    runtime = _exact_mapping(
        value,
        {
            "requested_device",
            "resolved_device",
            "cuda_available",
            "mixed_precision_requested",
            "mixed_precision_effective",
            "pin_memory_requested",
            "pin_memory_effective",
            "torch_version",
            "torchvision_version",
            "torchxrayvision_version",
            "cuda_runtime_version",
            "cudnn_version",
            "gpu_device_name",
            "gpu_device_index",
            "gpu_compute_capability",
            "loader_execution",
            "cxr_cache_id",
        },
        "runtime provenance",
    )
    if runtime["resolved_device"] not in {"cpu", "cuda"}:
        raise ValueError("Neural package runtime device is invalid")
    if (
        not isinstance(runtime["cxr_cache_id"], str)
        or not runtime["cxr_cache_id"].startswith("cache-")
        or not _is_sha256(runtime["cxr_cache_id"][6:])
    ):
        raise ValueError("Neural package CXR cache identity is invalid")
    try:
        loader_execution = LoaderExecutionPolicy.from_provenance(runtime["loader_execution"])
    except ValueError as exc:
        raise ValueError("Neural package loader execution provenance is invalid") from exc
    if loader_execution.lifecycle != "reused":
        raise ValueError("Neural package loader execution provenance is not epoch-reused")

    if (
        runtime["requested_device"] not in {"auto", "cpu", "cuda"}
        or runtime["pin_memory_requested"] not in {"auto", "enabled", "disabled"}
        or not all(
            isinstance(runtime[field], str) and runtime[field]
            for field in ("torch_version", "torchvision_version", "torchxrayvision_version")
        )
    ):
        raise ValueError("Neural package runtime text provenance is invalid")
    if not all(
        isinstance(runtime[field], bool)
        for field in (
            "cuda_available",
            "mixed_precision_requested",
            "mixed_precision_effective",
            "pin_memory_effective",
        )
    ):
        raise ValueError("Neural package runtime Boolean provenance is invalid")
    gpu_fields = (
        "cuda_runtime_version",
        "cudnn_version",
        "gpu_device_name",
        "gpu_device_index",
        "gpu_compute_capability",
    )
    if runtime["resolved_device"] == "cpu" and any(
        runtime[field] is not None for field in gpu_fields
    ):
        raise ValueError("CPU runtime provenance contains GPU values")
    if runtime["resolved_device"] == "cuda":
        capability = runtime["gpu_compute_capability"]
        if (
            not isinstance(runtime["cuda_runtime_version"], str)
            or isinstance(runtime["cudnn_version"], bool)
            or not isinstance(runtime["cudnn_version"], int)
            or not isinstance(runtime["gpu_device_name"], str)
            or isinstance(runtime["gpu_device_index"], bool)
            or not isinstance(runtime["gpu_device_index"], int)
            or not isinstance(capability, list)
            or len(capability) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in capability)
        ):
            raise ValueError("CUDA runtime provenance is invalid")


def _exact_mapping(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Neural package {name} has an unexpected field set")
    return value


def _finite_number(value: object) -> bool:
    return bool(
        not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)
    )


def _validate_checkpoint_binding(
    document: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> None:
    selection = document["selection"]
    if selection != {
        "selected_epoch": checkpoint["selected_epoch"],
        "selected_stage": checkpoint["selected_stage"],
        "validation_average_precision": checkpoint["validation_average_precision"],
    }:
        raise ValueError("Neural package selection differs from its checkpoint")


def _probability(value: object) -> bool:
    return bool(
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


def _validate_component(value: object, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).name != value
    ):
        raise ValueError(f"{field} must be one safe path component")


def _require_regular_file(path: str | Path, name: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Neural package {name} must be a regular non-symlink file")
    return source
