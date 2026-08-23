"""Safely serialize, publish, and validate RSNA neural model packages."""

from __future__ import annotations

import json
import math
import os
import pickle
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.hashing import sha256_file
from beyondcxr.data.rsna_cxr_cache import CxrCacheSourceAuthentication
from beyondcxr.models.fusion_concat import (
    fusion_architecture_contract,
    fusion_structured_input_conversion_contract,
)
from beyondcxr.training.config import (
    ExperimentConfig,
    load_experiment_config,
    with_runtime,
)
from beyondcxr.training.execution import LoaderExecutionPolicy
from beyondcxr.utils.package_identity import (
    canonical_scientific_id,
    fitted_object_state_sha256,
    package_scientific_config_payload,
    pretrained_weight_semantic_identity,
    tensor_state_sha256,
)
from beyondcxr.utils.publication import (
    install_immutable_directory,
    staging_directory,
)
from beyondcxr.utils.rsna_model_publication import threshold_contract, validated_threshold_contract
from beyondcxr.utils.skops_io import load_skops

NEURAL_MODEL_FILENAME = "model.pt"
CONFIG_FILENAME = "resolved_config.yaml"
MANIFEST_FILENAME = "manifest.json"
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
        "dataset_id",
        "family_id",
        "modalities",
        "task_id",
        "positive_class",
        "bundle_id",
        "bundle_manifest_sha256",
        "split_assignment_id",
        "label_policy_version",
        "config_source_sha256",
        "config_semantic_sha256",
        "checkpoint_sha256",
        "fit_config",
        "model_state_sha256",
        "preprocessor_state_sha256",
        "source_package_id",
        "source_provenance",
        "model_identity",
        "input_contract",
        "training_transform_contract",
        "evaluation_transform_contract",
        "training_policy",
        "selection",
        "thresholds",
        "threshold_contract",
        "source_authentication",
        "runtime_provenance",
    }
)
FUSION_MANIFEST_FIELDS = NEURAL_MANIFEST_FIELDS | {
    "structured_preprocessor_sha256",
    "structured_preprocessor_contract",
    "structured_input_conversion",
    "fusion_architecture",
}


@dataclass(frozen=True)
class PublishedNeuralModel:
    """Paths and identities for one immutable neural package."""

    package_directory: Path
    model_path: Path
    config_path: Path
    manifest_path: Path
    model_package_id: str
    checkpoint_sha256: str
    model_size_mib: float
    created: bool


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


def publish_neural_model_package(
    *,
    model_root: str | Path,
    checkpoint_path: str | Path,
    source_config_bytes: bytes,
    manifest: Mapping[str, Any],
    structured_preprocessor_path: str | Path | None = None,
) -> PublishedNeuralModel:
    """Publish one newly created immutable neural model package."""
    packages_root = Path(model_root) / "packages"
    packages_root.mkdir(parents=True, exist_ok=True)
    stage = staging_directory(packages_root / "model-package-pending")
    try:
        model_path = stage / NEURAL_MODEL_FILENAME
        config_path = stage / CONFIG_FILENAME
        shutil.copyfile(checkpoint_path, model_path)
        config_path.write_bytes(source_config_bytes)
        package_kind = _manifest_package_kind(manifest)
        if package_kind == "fusion":
            if structured_preprocessor_path is None:
                raise ValueError("Fusion publication requires a fitted structured preprocessor")
            _require_regular_file(structured_preprocessor_path, "structured preprocessor")
            preprocessor_path = stage / STRUCTURED_PREPROCESSOR_FILENAME
            shutil.copyfile(structured_preprocessor_path, preprocessor_path)
        elif package_kind == "cxr":
            if structured_preprocessor_path is not None:
                raise ValueError("CXR publication does not accept a structured preprocessor")
        else:
            raise ValueError("Neural publication requires CXR or fusion modality")
        checkpoint = load_neural_checkpoint(model_path)
        config = with_runtime(
            load_experiment_config(config_path), seed=int(manifest["training_policy"]["seed"])
        )
        preprocessor_state = None
        if package_kind == "fusion":
            preprocessor_state = fitted_object_state_sha256(
                load_skops(stage / STRUCTURED_PREPROCESSOR_FILENAME)
            )
        source_package_id = manifest.get("source_package_id")
        document = {
            **dict(manifest),
            "model_package_schema_version": NEURAL_PACKAGE_SCHEMA_VERSION,
            "dataset_id": config.dataset.dataset_id,
            "family_id": config.family.family_id,
            "modalities": list(config.family.modalities),
            "task_id": config.task.task_id,
            "checkpoint_sha256": sha256_file(model_path),
            "fit_config": package_scientific_config_payload(config),
            "model_state_sha256": tensor_state_sha256(checkpoint["model_state_dict"]),
            "preprocessor_state_sha256": preprocessor_state,
            "source_package_id": source_package_id,
        }
        document["model_package_id"] = neural_model_package_id(document)
        final = packages_root / document["model_package_id"]
        _validate_manifest(document, model_path, config_path, checkpoint)
        if package_kind == "fusion":
            _validate_fusion_source_package(stage, document)
        (stage / MANIFEST_FILENAME).write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        created = install_immutable_directory(stage, final, validate_published_neural_model)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    published_document = validate_published_neural_model(final)
    return PublishedNeuralModel(
        package_directory=final,
        model_path=final / NEURAL_MODEL_FILENAME,
        config_path=final / CONFIG_FILENAME,
        manifest_path=final / MANIFEST_FILENAME,
        model_package_id=published_document["model_package_id"],
        checkpoint_sha256=published_document["checkpoint_sha256"],
        model_size_mib=(final / NEURAL_MODEL_FILENAME).stat().st_size / (1024.0 * 1024.0),
        created=created,
    )


def validate_published_neural_model(
    package_directory: str | Path, *, enforce_directory_name: bool = True
) -> dict[str, Any]:
    """Validate one exact neural package without reconstructing its architecture."""
    document = validate_neural_package_metadata(
        package_directory, enforce_directory_name=enforce_directory_name
    )
    load_validated_neural_checkpoint(package_directory, document)
    return document


def validate_neural_package_metadata(
    package_directory: str | Path, *, enforce_directory_name: bool = True
) -> dict[str, Any]:
    """Validate a neural package's exact files, manifest, and physical identities."""
    directory = Path(package_directory)
    if directory.parent.name != "packages" or directory.is_symlink() or not directory.is_dir():
        raise ValueError("Neural model package must be a physical directory beneath packages")
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
    package_kind = _manifest_package_kind(document) if isinstance(document, dict) else None
    expected = {NEURAL_MODEL_FILENAME, CONFIG_FILENAME, MANIFEST_FILENAME}
    if package_kind == "fusion":
        expected.add(STRUCTURED_PREPROCESSOR_FILENAME)
    if actual != expected:
        raise ValueError("Neural model package contains an unexpected artifact set")
    _validate_manifest_metadata(
        document,
        directory / NEURAL_MODEL_FILENAME,
        directory / CONFIG_FILENAME,
    )
    if package_kind == "fusion":
        _validate_fusion_source_package(directory, document)
    if enforce_directory_name and directory.name != document["model_package_id"]:
        raise ValueError("Neural package directory differs from its semantic identity")
    if package_kind == "fusion" and sha256_file(
        directory / STRUCTURED_PREPROCESSOR_FILENAME
    ) != document.get("structured_preprocessor_sha256"):
        raise ValueError("Fusion structured preprocessor SHA-256 mismatch")
    return document


def load_validated_neural_checkpoint(
    package_directory: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Safely load a checkpoint and verify its binding to validated package metadata."""
    directory = Path(package_directory)
    checkpoint = load_neural_checkpoint(directory / NEURAL_MODEL_FILENAME)
    _validate_checkpoint_binding(manifest, checkpoint)
    return checkpoint


def neural_model_package_id(document: Mapping[str, Any]) -> str:
    """Return semantic identity from fit meaning and reconstructed fitted state."""
    manifest_fields = _manifest_fields(_manifest_package_kind(document))
    identity_fields = _identity_fields(manifest_fields)
    if set(document) not in {
        identity_fields,
        manifest_fields,
        manifest_fields - {"model_package_id"},
    }:
        raise ValueError("Neural package identity payload contains an unexpected field set")
    payload = {
        "dataset_id": document["dataset_id"],
        "bundle_id": document["bundle_id"],
        "split_assignment_id": document["split_assignment_id"],
        "task_id": document["task_id"],
        "label_policy_version": document["label_policy_version"],
        "positive_class": document["positive_class"],
        "family_id": document["family_id"],
        "modalities": document["modalities"],
        "seed": document["training_policy"]["seed"],
        "fit_config": document["fit_config"],
        "preprocessor_state_sha256": document["preprocessor_state_sha256"],
        "model_state_sha256": document["model_state_sha256"],
        "source_package_id": document["source_package_id"],
        "selected_state": {
            "selected_epoch": document["selection"]["selected_epoch"],
            "selected_stage": document["selection"]["selected_stage"],
        },
        "thresholds": document["thresholds"],
        "threshold_contract": document["threshold_contract"],
        "model_identity": {
            **document["model_identity"],
            "pretrained_weight": pretrained_weight_semantic_identity(
                document["model_identity"]["pretrained_weight"]
            ),
        },
        "input_contract": document["input_contract"],
        "training_transform_contract": document["training_transform_contract"],
        "evaluation_transform_contract": document["evaluation_transform_contract"],
        "training_policy": document["training_policy"],
    }
    return canonical_scientific_id(NEURAL_PACKAGE_ID_PREFIX, payload)


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
    model_path: Path,
    config_path: Path,
    checkpoint: Mapping[str, Any],
) -> None:
    _validate_manifest_metadata(document, model_path, config_path)
    _validate_checkpoint_binding(document, checkpoint)


def _validate_manifest_metadata(
    document: object,
    model_path: Path,
    config_path: Path,
) -> None:
    if not isinstance(document, dict):
        raise ValueError("Neural model manifest contains an unexpected field set")
    package_kind = _manifest_package_kind(document)
    manifest_fields = _manifest_fields(package_kind)
    if set(document) != manifest_fields:
        raise ValueError("Neural model manifest contains an unexpected field set")
    schema_version = document["model_package_schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != NEURAL_PACKAGE_SCHEMA_VERSION
    ):
        raise ValueError("Neural package schema version is invalid")
    if package_kind not in {"cxr", "fusion"}:
        raise ValueError("Neural package family/modalities contract is invalid")
    for field in (
        "family_id",
        "task_id",
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
        document["dataset_id"] != config.dataset.dataset_id
        or document["bundle_id"] != config.dataset.bundle_id
        or document["task_id"] != config.task.task_id
        or document["family_id"] != config.family.family_id
        or document["modalities"] != list(config.family.modalities)
        or document["fit_config"] != package_scientific_config_payload(config)
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
    contract = validated_threshold_contract(
        document["threshold_contract"], positive_class=positive_class
    )
    if contract != threshold_contract(sensitivity_target=contract["sensitivity_target"]):
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
    if document["model_state_sha256"] != tensor_state_sha256(
        load_neural_checkpoint(model_path)["model_state_dict"]
    ) or (package_kind == "cxr" and document["preprocessor_state_sha256"] is not None):
        raise ValueError("Neural reconstructed fitted state differs from semantic identity")
    if package_kind == "fusion":
        state_hash = fitted_object_state_sha256(
            load_skops(model_path.parent / STRUCTURED_PREPROCESSOR_FILENAME)
        )
        if state_hash != document["preprocessor_state_sha256"]:
            raise ValueError("Fusion fitted preprocessor differs from semantic identity")
    if document["model_package_id"] != neural_model_package_id(document):
        raise ValueError("Neural package ID does not match its identity payload")


def _validate_nested_manifest(document: dict[str, Any], config: ExperimentConfig) -> None:
    if config.neural is None:
        raise ValueError("Neural package archived configuration is not a neural experiment")
    package_kind = _config_package_kind(config)
    if _manifest_package_kind(document) != package_kind:
        raise ValueError("Neural package kind differs from archived configuration")
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

    neural = config.neural
    selection = document["selection"]
    selected_epoch = selection["selected_epoch"]
    if (
        selected_epoch > neural.warmup_epochs + neural.fine_tune_epochs
        or (selection["selected_stage"] == "warmup" and selected_epoch > neural.warmup_epochs)
        or (selection["selected_stage"] == "fine_tune" and selected_epoch <= neural.warmup_epochs)
    ):
        raise ValueError("Neural package selected epoch is inconsistent with its stage")
    transform_kwargs = {
        "image_size": int(config.family.parameters["image_size"]),
        "rotation_degrees": neural.rotation_degrees,
        "translation_fraction": neural.translation_fraction,
        "brightness_jitter": neural.brightness_jitter,
        "contrast_jitter": neural.contrast_jitter,
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
    _validate_training_policy(document, config, neural)
    if package_kind == "fusion":
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
    source_package_id = document["source_package_id"]
    if (
        not isinstance(source_package_id, str)
        or not source_package_id.startswith(NEURAL_PACKAGE_ID_PREFIX)
        or not _is_sha256(source_package_id.removeprefix(NEURAL_PACKAGE_ID_PREFIX))
    ):
        raise ValueError("Fusion source CXR package identity is invalid")


def _validate_fusion_source_package(directory: Path, document: Mapping[str, Any]) -> None:
    source_package_id = document["source_package_id"]
    source_directory = directory.parent / source_package_id
    try:
        source = validate_published_neural_model(source_directory)
    except (OSError, ValueError) as exc:
        raise ValueError("Fusion source CXR package is missing or invalid") from exc
    required = {
        "dataset_id": document["dataset_id"],
        "task_id": document["task_id"],
        "bundle_id": document["bundle_id"],
        "split_assignment_id": document["split_assignment_id"],
        "label_policy_version": document["label_policy_version"],
        "family_id": "cxr_densenet",
        "modalities": ["cxr"],
        "model_package_id": source_package_id,
    }
    if (
        any(source[field] != value for field, value in required.items())
        or source["training_policy"]["seed"] != document["training_policy"]["seed"]
    ):
        raise ValueError("Fusion source CXR package scientific lineage is incompatible")
    source_config = load_experiment_config(source_directory / CONFIG_FILENAME)
    fusion_config = load_experiment_config(directory / CONFIG_FILENAME)
    encoder_fields = ("encoder_name", "weights", "image_size", "embedding_dimension")
    if (
        source_config.neural != fusion_config.neural
        or any(
            source_config.family.parameters[field] != fusion_config.family.parameters[field]
            for field in encoder_fields
        )
        or pretrained_weight_semantic_identity(source["model_identity"]["pretrained_weight"])
        != pretrained_weight_semantic_identity(document["model_identity"]["pretrained_weight"])
        or source["training_transform_contract"] != document["training_transform_contract"]
        or source["evaluation_transform_contract"] != document["evaluation_transform_contract"]
        or source["input_contract"] != document["input_contract"]
    ):
        raise ValueError("Fusion source CXR package scientific contract is incompatible")


def _manifest_fields(package_kind: object) -> frozenset[str]:
    if package_kind == "cxr":
        return NEURAL_MANIFEST_FIELDS
    if package_kind == "fusion":
        return FUSION_MANIFEST_FIELDS
    raise ValueError("Neural package has an invalid kind")


def _manifest_package_kind(document: Mapping[str, Any]) -> str:
    family_id = document.get("family_id")
    modalities = document.get("modalities")
    if family_id == "cxr_densenet" and modalities == ["cxr"]:
        return "cxr"
    if family_id == "cxr_metadata_concat" and modalities == ["cxr", "metadata"]:
        return "fusion"
    raise ValueError("Neural package has an invalid family/modalities contract")


def _config_package_kind(config: ExperimentConfig) -> str:
    """Map the canonical family ontology to the RSNA package kind."""
    if config.family.family_id == "cxr_densenet":
        return "cxr"
    if config.family.family_id == "cxr_metadata_concat":
        return "fusion"
    raise ValueError("Neural package archived configuration has an unsupported family")


def _identity_fields(manifest_fields: frozenset[str]) -> frozenset[str]:
    return manifest_fields - {
        "model_package_id",
        "runtime_provenance",
        "config_source_sha256",
    }


def _validate_training_policy(
    document: dict[str, Any], config: ExperimentConfig, neural: Any
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
            "epochs": neural.warmup_epochs,
            "head_learning_rate": neural.warmup_head_learning_rate,
            "encoder_frozen": True,
        },
        "fine_tuning": {
            "maximum_epochs": neural.fine_tune_epochs,
            "encoder_learning_rate": neural.encoder_learning_rate,
            "head_learning_rate": neural.head_learning_rate,
        },
        "weight_decay": neural.weight_decay,
        "gradient_clip_norm": neural.gradient_clip_norm,
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "max",
            "factor": neural.scheduler_factor,
            "patience": neural.scheduler_patience,
            "min_lr": neural.scheduler_min_learning_rate,
        },
        "early_stopping": {
            "metric": "validation_average_precision",
            "patience": neural.early_stopping_patience,
            "minimum_delta": neural.early_stopping_min_delta,
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

    if document["threshold_contract"]["sensitivity_target"] != config.evaluation.sensitivity_target:
        raise ValueError("Neural package threshold contract is invalid")
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


def _require_regular_file(path: str | Path, name: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Neural package {name} must be a regular non-symlink file")
    return source
