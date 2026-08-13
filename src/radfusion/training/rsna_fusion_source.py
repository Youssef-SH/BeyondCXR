"""Resolve one explicit verified source CXR model package for RSNA fusion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from radfusion.training.config import (
    ExperimentConfig,
    load_experiment_config,
    require_runtime_seed,
    with_runtime,
)
from radfusion.training.rsna_evaluation_result import validate_rsna_model_package
from radfusion.utils.rsna_neural_publication import (
    CONFIG_FILENAME,
    load_validated_neural_checkpoint,
)


@dataclass(frozen=True)
class VerifiedSourceCxr:
    """Validated source package and safely loaded CXR checkpoint."""

    source_package_id: str
    manifest: Mapping[str, Any]
    checkpoint: Mapping[str, Any]


def resolve_source_cxr_package(
    source_package_id: str,
    fusion_config: ExperimentConfig,
) -> VerifiedSourceCxr:
    """Resolve one explicitly supplied scientifically compatible CXR package."""
    if fusion_config.family.family_id != "cxr_metadata_concat" or fusion_config.neural is None:
        raise ValueError("Source CXR resolution requires a fusion experiment configuration")
    manifest = validate_rsna_model_package(
        fusion_config.runtime.model_directory,
        source_package_id,
    )
    package_directory = fusion_config.runtime.model_directory / "packages" / source_package_id
    source_config = with_runtime(
        load_experiment_config(package_directory / CONFIG_FILENAME),
        seed=int(manifest["training_policy"]["seed"]),
    )
    _validate_source_contract(source_config, manifest, fusion_config, source_package_id)
    checkpoint = load_validated_neural_checkpoint(package_directory, manifest)
    return VerifiedSourceCxr(
        source_package_id,
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
    source_config: ExperimentConfig,
    manifest: Mapping[str, Any],
    fusion_config: ExperimentConfig,
    source_package_id: str,
) -> None:
    if source_config.family.family_id != "cxr_densenet" or source_config.neural is None:
        raise ValueError("Source CXR package does not archive a CXR experiment")
    expected = {
        "model_package_id": source_package_id,
        "dataset_id": fusion_config.dataset.dataset_id,
        "bundle_id": fusion_config.dataset.bundle_id,
        "split_assignment_id": fusion_config.dataset.split_assignment_id,
        "task_id": fusion_config.task.task_id,
        "label_policy_version": fusion_config.task.label_policy_version,
        "family_id": "cxr_densenet",
        "modalities": ["cxr"],
    }
    if any(manifest.get(field) != value for field, value in expected.items()):
        raise ValueError("Source CXR package scientific lineage is incompatible with fusion")
    if manifest["training_policy"]["seed"] != require_runtime_seed(fusion_config):
        raise ValueError("Fusion and source CXR package seeds differ")
    common_parameters = ("encoder_name", "weights", "image_size", "embedding_dimension")
    if any(
        source_config.family.parameters[field] != fusion_config.family.parameters[field]
        for field in common_parameters
    ):
        raise ValueError("Source CXR encoder identity differs from fusion configuration")
    if source_config.neural != fusion_config.neural:
        raise ValueError("Source CXR transform and neural lifecycle differ from fusion")
    pretrained_name = manifest["model_identity"]["pretrained_weight"]["declared_name"]
    if pretrained_name != fusion_config.family.parameters["weights"]:
        raise ValueError("Source CXR pretrained-weight identity differs from fusion")
