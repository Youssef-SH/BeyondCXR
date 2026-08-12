"""Resolve one explicit verified source CXR training package for fusion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from radfusion.training.completed_runs import require_completed_run
from radfusion.training.config import (
    ExperimentConfig,
    load_experiment_config,
    require_runtime_seed,
    with_runtime,
)
from radfusion.utils.neural_publication import (
    CONFIG_FILENAME,
    load_validated_neural_checkpoint,
    validate_neural_package_metadata,
)


@dataclass(frozen=True)
class SourceCxrLineage:
    """Exact source CXR package identities bound into a fusion run."""

    training_run_id: str
    model_package_id: str
    checkpoint_sha256: str
    config_semantic_sha256: str
    git_commit: str
    dependency_lock_sha256: str

    def as_dict(self) -> dict[str, str]:
        """Return serializable source-package lineage."""
        return asdict(self)


@dataclass(frozen=True)
class VerifiedSourceCxr:
    """Validated source lineage and safely loaded CXR checkpoint."""

    lineage: SourceCxrLineage
    manifest: Mapping[str, Any]
    checkpoint: Mapping[str, Any]


def resolve_source_cxr_training_run(
    client,
    training_run_id: str,
    fusion_config: ExperimentConfig,
    *,
    current_git_commit: str,
    current_git_dirty: bool,
    current_dependency_lock_sha256: str,
) -> VerifiedSourceCxr:
    """Resolve and verify one explicitly supplied same-seed image training run."""
    if not isinstance(training_run_id, str) or not training_run_id.strip():
        raise ValueError("Fusion training requires an explicit source CXR training run ID")
    if fusion_config.family.family_id != "cxr_metadata_concat" or fusion_config.neural is None:
        raise ValueError("Source CXR resolution requires a fusion experiment configuration")
    run = client.get_run(training_run_id)
    record = require_completed_run(run)
    if (
        record.run_kind != "training"
        or record.evaluation_scope != "validation"
        or record.modality != "image"
        or record.model != "cxr_densenet"
    ):
        raise ValueError("Fusion source must be a completed cxr_densenet training run")
    if record.integer_seed() != require_runtime_seed(fusion_config):
        raise ValueError("Fusion and source CXR training seeds differ")
    model_path = Path(record.local_model_path)
    if model_path.name != "model.pt":
        raise ValueError("Source CXR run has an invalid local package path")
    package_directory = model_path.parent
    manifest = validate_neural_package_metadata(package_directory)
    source_config = with_runtime(
        load_experiment_config(package_directory / CONFIG_FILENAME), seed=record.integer_seed()
    )
    _validate_source_contract(
        record,
        source_config,
        manifest,
        fusion_config,
        training_run_id=training_run_id,
        current_git_commit=current_git_commit,
        current_git_dirty=current_git_dirty,
        current_dependency_lock_sha256=current_dependency_lock_sha256,
    )
    checkpoint = load_validated_neural_checkpoint(package_directory, manifest)
    return VerifiedSourceCxr(
        SourceCxrLineage(
            training_run_id=training_run_id,
            model_package_id=str(manifest["model_package_id"]),
            checkpoint_sha256=str(manifest["checkpoint_sha256"]),
            config_semantic_sha256=str(manifest["config_semantic_sha256"]),
            git_commit=str(manifest["source_provenance"]["git_commit"]),
            dependency_lock_sha256=str(manifest["source_provenance"]["dependency_lock_sha256"]),
        ),
        manifest,
        checkpoint,
    )


def source_encoder_state(source: VerifiedSourceCxr) -> dict[str, torch.Tensor]:
    """Return the verified source checkpoint state for encoder-only extraction."""
    state = source.checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Source CXR checkpoint has no model state")
    return dict(state)


def _validate_source_contract(
    record,
    source_config: ExperimentConfig,
    manifest: Mapping[str, Any],
    fusion_config: ExperimentConfig,
    *,
    training_run_id: str,
    current_git_commit: str,
    current_git_dirty: bool,
    current_dependency_lock_sha256: str,
) -> None:
    if source_config.family.family_id != "cxr_densenet" or source_config.neural is None:
        raise ValueError("Source CXR package does not archive an image experiment")
    expected_record = {
        "run_id": training_run_id,
        "dataset": fusion_config.dataset.dataset_id,
        "task": fusion_config.task.task_id,
        "bundle_id": fusion_config.dataset.bundle_id,
        "model_package_id": manifest["model_package_id"],
        "split_assignment_id": manifest["split_assignment_id"],
        "label_policy_version": manifest["label_policy_version"],
        "config_semantic_sha256": manifest["config_semantic_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "local_model_sha256": manifest["checkpoint_sha256"],
    }
    for field, expected in expected_record.items():
        if getattr(record, field) != expected:
            raise ValueError(f"Source CXR run {field} differs from its package or fusion config")
    expected_manifest = {
        "training_mlflow_run_id": training_run_id,
        "modality": "image",
        "model": "cxr_densenet",
        "task": fusion_config.task.task_id,
        "bundle_id": fusion_config.dataset.bundle_id,
        "config_source_sha256": source_config.config_source_sha256,
        "training_policy_seed": require_runtime_seed(fusion_config),
    }
    for field, expected in expected_manifest.items():
        observed = (
            manifest["training_policy"]["seed"]
            if field == "training_policy_seed"
            else manifest[field]
        )
        if observed != expected:
            raise ValueError(f"Source CXR package {field} is incompatible with fusion")
    common_parameters = ("encoder_name", "weights", "image_size", "embedding_dimension")
    if any(
        source_config.family.parameters[field] != fusion_config.family.parameters[field]
        for field in common_parameters
    ):
        raise ValueError("Source CXR encoder identity differs from fusion configuration")
    if source_config.neural != fusion_config.neural:
        raise ValueError("Source CXR transform and neural lifecycle differ from fusion")
    source = manifest["source_provenance"]
    if (
        source["git_dirty"]
        or current_git_dirty
        or source["git_commit"] != current_git_commit
        or source["dependency_lock_sha256"] != current_dependency_lock_sha256
        or record.git_commit != source["git_commit"]
        or record.git_dirty != "false"
        or record.dependency_lock_sha256 != source["dependency_lock_sha256"]
    ):
        raise ValueError("Fusion requires matching clean source revision and dependency lock")
    if record.bundle_manifest_sha256 != manifest["bundle_manifest_sha256"]:
        raise ValueError("Source CXR observed bundle-manifest identity is inconsistent")
    pretrained_name = manifest["model_identity"]["pretrained_weight"]["declared_name"]
    if pretrained_name != fusion_config.family.parameters["weights"]:
        raise ValueError("Source CXR pretrained-weight identity differs from fusion")
