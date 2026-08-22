"""Publish and validate the frozen full-development Symile final packages."""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from radfusion.data.errors import ManifestBuildError
from radfusion.data.symile_preprocess import (
    LAB_ECDF_POLICY_VERSION,
    SymileLabEcdfTransformer,
    load_symile_lab_preprocessor,
    save_symile_lab_preprocessor,
    validate_symile_lab_preprocessor,
)
from radfusion.models.cxr_baseline import CxrBinaryClassifier, StandardCxrEncoder
from radfusion.models.symile_ecg_fusion import build_symile_trimodal_gated_model
from radfusion.models.symile_fusion import build_symile_concat_model, build_symile_gated_model
from radfusion.training.config import ExperimentConfig
from radfusion.training.symile_families import (
    FINAL_FUSION_FAMILIES,
    FINAL_NEURAL_FAMILIES,
    FINAL_NEURAL_MEMBER_SEEDS,
    FINAL_TABULAR_FAMILIES,
    FINAL_TABULAR_SEED,
    SYMILE_ECG_GATED_FAMILY,
)
from radfusion.utils.package_identity import (
    canonical_scientific_id,
    fitted_object_state_sha256,
    package_scientific_config_payload,
    pretrained_weight_semantic_identity,
    tensor_state_sha256,
)
from radfusion.utils.publication import install_immutable_directory, staging_directory
from radfusion.utils.skops_io import load_skops, save_skops

FINAL_PACKAGE_PREFIX = "final-package-"
FINAL_PACKAGE_SCHEMA_VERSION = 1
FINAL_NEURAL_CHECKPOINT_SCHEMA_VERSION = 1
_MANIFEST_FILENAME = "manifest.json"
_CONFIG_FILENAME = "final_fit_config.json"
_NEURAL_FILENAME = "model.pt"
_TABULAR_FILENAME = "model.skops"
_LAB_PREPROCESSOR_FILENAME = "lab_preprocessor.skops"


@dataclass(frozen=True)
class FinalTrainingPlan:
    """The development-authoritative median budget rendered into final fitting."""

    family_id: str
    budget: int | None
    stage1_epochs: int | None
    stage2_epochs: int | None


@dataclass(frozen=True)
class ValidatedFinalPackage:
    """A validated, independently reconstructable final fitted package."""

    directory: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str

    @property
    def package_id(self) -> str:
        return str(self.manifest["final_package_id"])


@dataclass(frozen=True)
class FinalPackageConfig:
    """Only the authenticated, package-owned fields needed after final fitting."""

    dataset_id: str
    bundle_id: str
    bundle_manifest_sha256: str
    split_assignment_id: str
    task_id: str
    label_policy_version: str
    family_id: str
    modalities: tuple[str, ...]
    family_parameters: Mapping[str, object]
    preprocessing: Mapping[str, object]
    training_parameters: Mapping[str, object]
    loader: Mapping[str, object]
    augmentation: Mapping[str, object]

    @property
    def batch_size(self) -> int:
        return int(self.loader["batch_size"])

    @property
    def mixed_precision(self) -> bool:
        return bool(self.training_parameters["mixed_precision"])


def final_training_plan(family_id: str, budget: int | None) -> FinalTrainingPlan:
    """Convert the frozen family budget into the exact final training schedule."""
    if family_id == "labs_logistic":
        if budget is not None:
            raise ManifestBuildError("Final Labs Logistic Regression has no selected budget")
        return FinalTrainingPlan(family_id, None, None, None)
    if family_id == "labs_lightgbm":
        _positive_int(budget, "Final LightGBM median best_iteration")
        if int(budget) > 500:
            raise ManifestBuildError("Final LightGBM budget exceeds its frozen maximum")
        return FinalTrainingPlan(family_id, int(budget), None, None)
    if family_id not in FINAL_NEURAL_FAMILIES:
        raise ManifestBuildError("Unsupported final Symile family")
    _positive_int(budget, "Final neural median epoch")
    if not isinstance(budget, int):
        raise ManifestBuildError("Final neural median epoch is invalid")
    if budget > 30:
        raise ManifestBuildError("Final neural budget exceeds its frozen maximum")
    return FinalTrainingPlan(
        family_id=family_id,
        budget=budget,
        stage1_epochs=min(budget, 2),
        stage2_epochs=max(budget - 2, 0),
    )


def publish_final_neural_package(
    *,
    model_root: str | Path,
    config: ExperimentConfig,
    family_development_id: str,
    plan: FinalTrainingPlan,
    seed: int,
    state_dict: Mapping[str, torch.Tensor],
    lab_preprocessor: SymileLabEcdfTransformer | None,
    source_cxr_package_id: str | None,
    pretrained_weight: Mapping[str, object] | None,
    operational: Mapping[str, object],
) -> ValidatedFinalPackage:
    """Publish one terminal neural member with no validation-derived state."""
    _validate_neural_plan(config, family_development_id, plan, seed, source_cxr_package_id)
    _validate_execution_provenance(operational, neural=True)
    _validate_pretrained_weight(config, pretrained_weight)
    _validate_state_dict(state_dict)
    if config.family.family_id in FINAL_FUSION_FAMILIES:
        if lab_preprocessor is None:
            raise ManifestBuildError("Final fusion package lacks its fitted laboratory transform")
        validate_symile_lab_preprocessor(lab_preprocessor)
    elif lab_preprocessor is not None:
        raise ManifestBuildError("CXR-only final package cannot declare laboratory preprocessing")
    model_hash = tensor_state_sha256(state_dict)
    preprocessor_hash = (
        fitted_object_state_sha256(lab_preprocessor) if lab_preprocessor is not None else None
    )
    semantic = _neural_semantic(
        config=config,
        family_development_id=family_development_id,
        plan=plan,
        seed=seed,
        source_cxr_package_id=source_cxr_package_id,
        pretrained_weight=pretrained_weight,
        model_hash=model_hash,
        preprocessor_hash=preprocessor_hash,
    )
    package_id = canonical_scientific_id(FINAL_PACKAGE_PREFIX, semantic)
    destination = _package_destination(model_root, package_id)
    stage = staging_directory(destination)
    try:
        checkpoint = {
            "checkpoint_schema_version": FINAL_NEURAL_CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": _cpu_state(state_dict),
            "terminal_training": {
                "stage1_epochs": plan.stage1_epochs,
                "stage2_epochs": plan.stage2_epochs,
                "scheduler": None,
                "early_stopping": None,
                "selection": None,
            },
        }
        torch.save(checkpoint, stage / _NEURAL_FILENAME)
        (stage / _CONFIG_FILENAME).write_bytes(_final_config_bytes(config))
        if lab_preprocessor is not None:
            save_symile_lab_preprocessor(lab_preprocessor, stage / _LAB_PREPROCESSOR_FILENAME)
        document = _neural_document(
            package_id=package_id,
            semantic=semantic,
            bundle_manifest_sha256=config.dataset.bundle_manifest_sha256,
            pretrained_weight=pretrained_weight,
            artifacts=_artifact_hashes(stage),
        )
        document["execution_provenance"] = dict(operational)
        _write_manifest(stage, document)
        install_immutable_directory(stage, destination, validate_final_package)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return validate_final_package(destination, expected_package_id=package_id)


def publish_final_tabular_package(
    *,
    model_root: str | Path,
    config: ExperimentConfig,
    family_development_id: str | None,
    plan: FinalTrainingPlan,
    model: Pipeline,
    operational: Mapping[str, object],
) -> ValidatedFinalPackage:
    """Publish one final LR or fixed-iteration LightGBM package."""
    _validate_execution_provenance(operational, neural=False)
    family = config.family.family_id
    if family not in FINAL_TABULAR_FAMILIES or plan.family_id != family:
        raise ManifestBuildError("Final tabular package family is invalid")
    classifier = model.named_steps.get("classifier") if isinstance(model, Pipeline) else None
    if family == "labs_logistic":
        if family_development_id is not None or plan.budget is not None:
            raise ManifestBuildError("Final Labs Logistic Regression contract is invalid")
    elif family_development_id is None or plan.budget is None:
        raise ManifestBuildError("Final Labs LightGBM contract is invalid")
    _validate_final_tabular_estimator(classifier, config, plan)
    preprocessor = model.named_steps.get("preprocess")
    if not isinstance(preprocessor, SymileLabEcdfTransformer):
        raise ManifestBuildError("Final tabular package lacks its fitted laboratory transform")
    preprocessor_hash = fitted_object_state_sha256(preprocessor)
    model_hash = fitted_object_state_sha256(model, selected_iteration=plan.budget)
    semantic = {
        "dataset_id": "symile",
        "execution_scope": "full_development",
        "input": _final_input_projection(config),
        "family_development_id": family_development_id,
        "final_training_budget": plan.budget,
        "final_stage_budgets": None,
        "seed_policy": FINAL_TABULAR_SEED,
        "terminal_fixed_budget_policy": "no_validation_no_scheduler_no_early_stopping",
        "model_state_sha256": model_hash,
        "preprocessor_state_sha256": preprocessor_hash,
        "source_cxr_package_id": None,
        "pretrained_scientific_identity": None,
    }
    package_id = canonical_scientific_id(FINAL_PACKAGE_PREFIX, semantic)
    destination = _package_destination(model_root, package_id)
    stage = staging_directory(destination)
    try:
        save_skops(model, stage / _TABULAR_FILENAME)
        (stage / _CONFIG_FILENAME).write_bytes(_final_config_bytes(config))
        document = {
            "final_package_schema_version": FINAL_PACKAGE_SCHEMA_VERSION,
            "final_package_id": package_id,
            "package_kind": "tabular",
            "bundle_manifest_sha256": config.dataset.bundle_manifest_sha256,
            **semantic,
            "pretrained_weight_fingerprint": None,
            "artifacts": _artifact_hashes(stage),
            "execution_provenance": dict(operational),
        }
        _write_manifest(stage, document)
        install_immutable_directory(stage, destination, validate_final_package)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return validate_final_package(destination, expected_package_id=package_id)


def validate_final_package(
    directory: str | Path,
    *,
    expected_package_id: str | None = None,
    enforce_directory_name: bool = True,
) -> ValidatedFinalPackage:
    """Validate exact artifacts, fitted state, and narrowed final semantic identity."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ManifestBuildError("Final package directory is invalid")
    raw = (root / _MANIFEST_FILENAME).read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Final package manifest is unreadable") from exc
    required = {
        "final_package_schema_version",
        "final_package_id",
        "package_kind",
        "dataset_id",
        "bundle_manifest_sha256",
        "execution_scope",
        "input",
        "family_development_id",
        "final_training_budget",
        "final_stage_budgets",
        "seed_policy",
        "terminal_fixed_budget_policy",
        "model_state_sha256",
        "preprocessor_state_sha256",
        "source_cxr_package_id",
        "pretrained_scientific_identity",
        "pretrained_weight_fingerprint",
        "artifacts",
        "execution_provenance",
    }
    schema_version = (
        document.get("final_package_schema_version") if isinstance(document, dict) else None
    )
    if (
        not isinstance(document, dict)
        or set(document) != required
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != FINAL_PACKAGE_SCHEMA_VERSION
        or document["dataset_id"] != "symile"
        or not _sha256_text(document["bundle_manifest_sha256"])
        or document["execution_scope"] != "full_development"
        or document["terminal_fixed_budget_policy"]
        != "no_validation_no_scheduler_no_early_stopping"
    ):
        raise ManifestBuildError("Final package manifest contract is invalid")
    expected_files = _expected_files(document)
    if {item.name for item in root.iterdir()} != expected_files:
        raise ManifestBuildError("Final package file set is invalid")
    if document["artifacts"] != _artifact_hashes(root):
        raise ManifestBuildError("Final package physical integrity is invalid")
    config = _validated_config(root, document)
    if document["package_kind"] == "neural":
        _validate_neural_final(root, document, config)
    elif document["package_kind"] == "tabular":
        _validate_tabular_final(root, document, config)
    else:
        raise ManifestBuildError("Final package kind is invalid")
    _validate_execution_provenance(
        document["execution_provenance"], neural=document["package_kind"] == "neural"
    )
    semantic = {
        key: document[key]
        for key in (
            "dataset_id",
            "execution_scope",
            "input",
            "family_development_id",
            "final_training_budget",
            "final_stage_budgets",
            "seed_policy",
            "terminal_fixed_budget_policy",
            "model_state_sha256",
            "preprocessor_state_sha256",
            "source_cxr_package_id",
            "pretrained_scientific_identity",
        )
    }
    expected = canonical_scientific_id(FINAL_PACKAGE_PREFIX, semantic)
    if (
        document["final_package_id"] != expected
        or (expected_package_id is not None and expected != expected_package_id)
        or (enforce_directory_name and root.name != expected)
    ):
        raise ManifestBuildError("Final package semantic identity is invalid")
    return ValidatedFinalPackage(root, document, hashlib.sha256(raw).hexdigest())


def load_final_neural_state(package: ValidatedFinalPackage) -> Mapping[str, torch.Tensor]:
    """Load one validated terminal neural state without any upstream weight cache."""
    if package.manifest["package_kind"] != "neural":
        raise ManifestBuildError("Final package is not neural")
    return _load_final_neural_checkpoint(package.directory / _NEURAL_FILENAME)["model_state_dict"]


def load_final_package_config(package: ValidatedFinalPackage) -> FinalPackageConfig:
    """Load the package-bound configuration after the package validator has passed."""
    return _validated_config(package.directory, package.manifest)


def validate_final_fit_provenance(value: object) -> None:
    """Validate provenance that every successful final fit must receive up front."""
    base = {"git_commit", "git_dirty", "dependency_lock_sha256", "mlflow_run_id"}
    if not isinstance(value, Mapping) or set(value) != base:
        raise ManifestBuildError("Final fit execution provenance is invalid")
    if (
        not _sha256_text(value["dependency_lock_sha256"])
        or not isinstance(value["mlflow_run_id"], str)
        or not value["mlflow_run_id"]
        or not isinstance(value["git_commit"], str)
        or len(value["git_commit"]) != 40
        or any(character not in "0123456789abcdef" for character in value["git_commit"])
        or not isinstance(value["git_dirty"], bool)
    ):
        raise ManifestBuildError("Final fit operational lineage is invalid")


def _validate_execution_provenance(value: object, *, neural: bool) -> None:
    """Validate bounded nonidentity execution provenance."""
    base = {"git_commit", "git_dirty", "dependency_lock_sha256", "mlflow_run_id"}
    expected = base | ({"runtime_provenance", "loader_execution"} if neural else set())
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ManifestBuildError("Final package execution provenance is invalid")
    validate_final_fit_provenance({key: value[key] for key in base})
    if neural:
        if not isinstance(value["runtime_provenance"], Mapping) or not isinstance(
            value["runtime_provenance"].get("pin_memory_effective"), bool
        ):
            raise ManifestBuildError("Final neural runtime provenance is invalid")
        loader = value["loader_execution"]
        loader_fields = {
            "lifecycle",
            "num_workers",
            "pin_memory",
            "batch_size",
            "drop_last",
            "shuffle",
            "sampler",
            "persistent_workers",
            "prefetch_factor",
            "multiprocessing_context",
        }
        if not isinstance(loader, Mapping) or set(loader) != loader_fields:
            raise ManifestBuildError("Final neural loader provenance is invalid")
        if (
            loader["lifecycle"] != "reused"
            or not isinstance(loader["num_workers"], int)
            or isinstance(loader["num_workers"], bool)
            or loader["num_workers"] < 0
            or not isinstance(loader["batch_size"], int)
            or isinstance(loader["batch_size"], bool)
            or loader["batch_size"] <= 0
            or loader["drop_last"] is not False
            or loader["shuffle"] is not False
            or loader["sampler"] != "epoch_permutation"
            or not isinstance(loader["pin_memory"], bool)
            or not isinstance(loader["persistent_workers"], bool)
            or loader["pin_memory"] != value["runtime_provenance"]["pin_memory_effective"]
            or (loader["num_workers"] == 0 and loader["persistent_workers"])
            or (loader["num_workers"] == 0 and loader["prefetch_factor"] is not None)
            or (loader["num_workers"] == 0 and loader["multiprocessing_context"] is not None)
            or (loader["num_workers"] > 0 and loader["prefetch_factor"] != 2)
            or (loader["num_workers"] > 0 and loader["multiprocessing_context"] != "spawn")
        ):
            raise ManifestBuildError("Final neural loader execution contract is invalid")


def load_final_neural_model(package: ValidatedFinalPackage) -> torch.nn.Module:
    """Reconstruct a final neural model using only its packaged state."""
    config = load_final_package_config(package)
    if package.manifest["package_kind"] != "neural":
        raise ManifestBuildError("Final package is not neural")
    model = _reconstruct_neural(config)
    try:
        incompatible = model.load_state_dict(load_final_neural_state(package), strict=True)
    except RuntimeError as exc:
        raise ManifestBuildError("Final neural state cannot load strictly") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ManifestBuildError("Final neural state is structurally incomplete")
    return model


def load_final_tabular_model(package: ValidatedFinalPackage) -> Pipeline:
    """Load a validated final tabular pipeline without mutable estimator state."""
    if package.manifest["package_kind"] != "tabular":
        raise ManifestBuildError("Final package is not tabular")
    try:
        model = load_skops(package.directory / _TABULAR_FILENAME)
    except (OSError, ValueError, TypeError) as exc:
        raise ManifestBuildError("Final tabular model is unreadable") from exc
    if not isinstance(model, Pipeline):
        raise ManifestBuildError("Final tabular model type is invalid")
    return model


def load_final_lab_preprocessor(
    package: ValidatedFinalPackage,
) -> SymileLabEcdfTransformer:
    """Load a validated fusion preprocessor bound to the final package."""
    if package.manifest["input"]["family"]["family_id"] not in FINAL_FUSION_FAMILIES:
        raise ManifestBuildError("Final package does not use laboratory fusion")
    return load_symile_lab_preprocessor(package.directory / _LAB_PREPROCESSOR_FILENAME)


def _neural_document(
    *,
    package_id: str,
    semantic: Mapping[str, object],
    bundle_manifest_sha256: str,
    pretrained_weight: Mapping[str, object] | None,
    artifacts: Mapping[str, str],
) -> dict[str, object]:
    return {
        "final_package_schema_version": FINAL_PACKAGE_SCHEMA_VERSION,
        "final_package_id": package_id,
        "package_kind": "neural",
        "bundle_manifest_sha256": bundle_manifest_sha256,
        **semantic,
        "pretrained_weight_fingerprint": (
            dict(pretrained_weight) if pretrained_weight is not None else None
        ),
        "artifacts": dict(artifacts),
    }


def _neural_semantic(
    *,
    config: ExperimentConfig,
    family_development_id: str,
    plan: FinalTrainingPlan,
    seed: int,
    source_cxr_package_id: str | None,
    pretrained_weight: Mapping[str, object] | None,
    model_hash: str,
    preprocessor_hash: str | None,
) -> dict[str, object]:
    return {
        "dataset_id": "symile",
        "execution_scope": "full_development",
        "input": _final_input_projection(config),
        "family_development_id": family_development_id,
        "final_training_budget": plan.budget,
        "final_stage_budgets": {
            "stage1_epochs": plan.stage1_epochs,
            "stage2_epochs": plan.stage2_epochs,
        },
        "seed_policy": seed,
        "terminal_fixed_budget_policy": "no_validation_no_scheduler_no_early_stopping",
        "model_state_sha256": model_hash,
        "preprocessor_state_sha256": preprocessor_hash,
        "source_cxr_package_id": source_cxr_package_id,
        "pretrained_scientific_identity": (
            pretrained_weight_semantic_identity(pretrained_weight)
            if pretrained_weight is not None
            else None
        ),
    }


def _final_input_projection(config: ExperimentConfig) -> dict[str, object]:
    return final_input_projection_from_development(package_scientific_config_payload(config))


def final_input_projection_from_development(
    fit_config: Mapping[str, object],
) -> dict[str, object]:
    """Narrow one validated development fit configuration for terminal fitting."""
    try:
        dataset = fit_config["dataset"]
        task = fit_config["task"]
        family = fit_config["family"]
        preprocessing = fit_config["preprocessing"]
        training = fit_config["training"]
        parameters = dict(training["parameters"])
    except (KeyError, TypeError) as exc:
        raise ManifestBuildError("Development fit configuration is invalid") from exc
    for key in (
        "warmup_epochs",
        "fine_tune_epochs",
        "scheduler_factor",
        "scheduler_patience",
        "scheduler_min_learning_rate",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "early_stopping_rounds",
    ):
        parameters.pop(key, None)
    return {
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "bundle_id": dataset["bundle_id"],
            "split_assignment_id": dataset["split_assignment_id"],
            "cohort": "official_train_plus_validation_strict_pneumonia",
        },
        "task": dict(task),
        "family": {
            "family_id": family["family_id"],
            "modalities": list(family["modalities"]),
            "parameters": dict(family["parameters"]),
        },
        "preprocessing": dict(preprocessing),
        "training": {
            "parameters": parameters,
            "loader": dict(training["loader"]),
            "augmentation": dict(training["augmentation"]),
        },
    }


def final_cxr_ancestry_projection(
    final_input: Mapping[str, object], bundle_manifest_sha256: str
) -> dict[str, object]:
    """Return only shared, package-owned CXR semantics required by fusion ancestry."""
    family = final_input["family"]
    parameters = family["parameters"]
    return {
        "dataset": final_input["dataset"],
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "task": final_input["task"],
        "encoder": {
            key: parameters[key]
            for key in ("encoder_name", "weights", "image_size", "embedding_dimension")
        },
        "cxr_transform_policy": final_input["preprocessing"]["cxr_transform_policy"],
    }


def _final_config_bytes(config: ExperimentConfig) -> bytes:
    document = {
        "final_fit_config_schema_version": 1,
        "bundle_manifest_sha256": config.dataset.bundle_manifest_sha256,
        **_final_input_projection(config),
    }
    return (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _validate_neural_plan(
    config: ExperimentConfig,
    family_development_id: str,
    plan: FinalTrainingPlan,
    seed: int,
    source_cxr_package_id: str | None,
) -> None:
    family = config.family.family_id
    if (
        family != plan.family_id
        or family not in {"cxr_densenet", *FINAL_FUSION_FAMILIES}
        or not _identity(family_development_id, "development-")
        or seed not in FINAL_NEURAL_MEMBER_SEEDS
        or plan.stage1_epochs != min(int(plan.budget or 0), 2)
        or plan.stage2_epochs != max(int(plan.budget or 0) - 2, 0)
    ):
        raise ManifestBuildError("Final neural package plan is invalid")
    if family in FINAL_FUSION_FAMILIES:
        if not _identity(source_cxr_package_id, FINAL_PACKAGE_PREFIX):
            raise ManifestBuildError("Final fusion package requires same-seed final CXR ancestry")
    elif source_cxr_package_id is not None:
        raise ManifestBuildError("Final CXR package cannot declare a source CXR package")


def _validate_neural_final(
    root: Path, document: Mapping[str, object], config: FinalPackageConfig
) -> None:
    checkpoint = _load_final_neural_checkpoint(root / _NEURAL_FILENAME)
    state = checkpoint["model_state_dict"]
    if tensor_state_sha256(state) != document["model_state_sha256"]:
        raise ManifestBuildError("Final neural fitted state differs from manifest")
    family = config.family_id
    plan = final_training_plan(family, document["final_training_budget"])
    _validate_pretrained_weight(config, document["pretrained_weight_fingerprint"])
    expected_pretrained = (
        pretrained_weight_semantic_identity(document["pretrained_weight_fingerprint"])
        if document["pretrained_weight_fingerprint"] is not None
        else None
    )
    if document["pretrained_scientific_identity"] != expected_pretrained:
        raise ManifestBuildError("Final neural pretrained scientific identity is invalid")
    terminal = checkpoint["terminal_training"]
    expected_terminal = {
        "stage1_epochs": plan.stage1_epochs,
        "stage2_epochs": plan.stage2_epochs,
        "scheduler": None,
        "early_stopping": None,
        "selection": None,
    }
    if terminal != expected_terminal:
        raise ManifestBuildError("Final neural checkpoint uses selection or scheduler state")
    if document["final_stage_budgets"] != {
        "stage1_epochs": plan.stage1_epochs,
        "stage2_epochs": plan.stage2_epochs,
    }:
        raise ManifestBuildError("Final neural stage budgets differ from the frozen plan")
    model = _reconstruct_neural(config)
    try:
        incompatible = model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ManifestBuildError("Final neural package cannot reconstruct exactly") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ManifestBuildError("Final neural package state is structurally incomplete")
    if family in FINAL_FUSION_FAMILIES:
        transformer = load_symile_lab_preprocessor(root / _LAB_PREPROCESSOR_FILENAME)
        if fitted_object_state_sha256(transformer) != document["preprocessor_state_sha256"]:
            raise ManifestBuildError("Final neural laboratory preprocessor differs from manifest")
    elif document["preprocessor_state_sha256"] is not None:
        raise ManifestBuildError("Final CXR package declares a laboratory preprocessor")


def _validate_tabular_final(
    root: Path, document: Mapping[str, object], config: FinalPackageConfig
) -> None:
    try:
        model = load_skops(root / _TABULAR_FILENAME)
    except (OSError, ValueError, TypeError) as exc:
        raise ManifestBuildError("Final tabular package is unreadable") from exc
    if not isinstance(model, Pipeline) or tuple(model.named_steps) != ("preprocess", "classifier"):
        raise ManifestBuildError("Final tabular package is not a fitted lab pipeline")
    family = config.family_id
    classifier = model.named_steps["classifier"]
    plan = final_training_plan(family, document["final_training_budget"])
    if document["final_stage_budgets"] is not None:
        raise ManifestBuildError("Final tabular package cannot declare neural stage budgets")
    if (
        document["pretrained_scientific_identity"] is not None
        or document["pretrained_weight_fingerprint"] is not None
    ):
        raise ManifestBuildError("Final tabular package cannot declare pretrained CXR state")
    _validate_final_tabular_estimator(classifier, config, plan)
    preprocessor = model.named_steps["preprocess"]
    if (
        config.preprocessing.get("lab_policy") != LAB_ECDF_POLICY_VERSION
        or not isinstance(preprocessor, SymileLabEcdfTransformer)
        or fitted_object_state_sha256(preprocessor) != document["preprocessor_state_sha256"]
        or fitted_object_state_sha256(model, selected_iteration=plan.budget)
        != document["model_state_sha256"]
    ):
        raise ManifestBuildError("Final tabular fitted state differs from manifest")


def _validate_final_tabular_estimator(
    classifier: object,
    config: ExperimentConfig | FinalPackageConfig,
    plan: FinalTrainingPlan,
) -> None:
    """Prove the fitted estimator's effective parameters match its frozen config."""
    if isinstance(config, FinalPackageConfig):
        family = config.family_id
        parameters = config.training_parameters
        family_parameters = config.family_parameters
    else:
        family = config.family.family_id
        parameters = config.training.parameters
        family_parameters = config.family.parameters
    if family == "labs_logistic":
        if not isinstance(classifier, LogisticRegression):
            raise ManifestBuildError("Final Labs Logistic Regression estimator type is invalid")
        expected = {
            "l1_ratio": float(parameters["l1_ratio"]),
            "solver": str(parameters["solver"]),
            "C": float(parameters["C"]),
            "max_iter": int(parameters["max_iter"]),
            "class_weight": parameters["class_weight"],
            "random_state": FINAL_TABULAR_SEED,
        }
    elif family == "labs_lightgbm":
        if not isinstance(classifier, LGBMClassifier):
            raise ManifestBuildError("Final Labs LightGBM estimator type is invalid")
        expected = {
            "objective": str(family_parameters["objective"]),
            "n_estimators": plan.budget,
            "learning_rate": float(parameters["learning_rate"]),
            "num_leaves": int(family_parameters["num_leaves"]),
            "min_child_samples": int(family_parameters["min_child_samples"]),
            "subsample": float(parameters["subsample"]),
            "subsample_freq": int(parameters["subsample_freq"]),
            "colsample_bytree": float(parameters["colsample_bytree"]),
            "reg_lambda": float(parameters["reg_lambda"]),
            "class_weight": parameters["class_weight"],
            "random_state": FINAL_TABULAR_SEED,
            "bagging_seed": FINAL_TABULAR_SEED,
            "feature_fraction_seed": FINAL_TABULAR_SEED,
            "data_random_seed": FINAL_TABULAR_SEED,
            "drop_seed": FINAL_TABULAR_SEED,
            "extra_seed": FINAL_TABULAR_SEED,
            "deterministic": True,
            "force_col_wise": True,
            "n_jobs": 1,
            "metric": "None",
            "verbosity": -1,
        }
    else:
        raise ManifestBuildError("Final tabular estimator family is invalid")
    actual = classifier.get_params(deep=False)
    if any(actual.get(name) != value for name, value in expected.items()):
        raise ManifestBuildError("Final tabular estimator differs from frozen configuration")


def _validated_config(root: Path, document: Mapping[str, object]) -> FinalPackageConfig:
    try:
        source = root / _CONFIG_FILENAME
        source_bytes = source.read_bytes()
        packaged = json.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Final package configuration is invalid") from exc
    schema_version = (
        packaged.get("final_fit_config_schema_version") if isinstance(packaged, dict) else None
    )
    if (
        not isinstance(packaged, dict)
        or set(packaged)
        != {
            "final_fit_config_schema_version",
            "bundle_manifest_sha256",
            *document["input"],
        }
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
        or packaged["bundle_manifest_sha256"] != document["bundle_manifest_sha256"]
        or {
            key: value
            for key, value in packaged.items()
            if key not in {"final_fit_config_schema_version", "bundle_manifest_sha256"}
        }
        != document["input"]
    ):
        raise ManifestBuildError("Final package configuration differs from manifest")
    return _final_package_config_from_input(
        document["input"],
        bundle_manifest_sha256=str(document["bundle_manifest_sha256"]),
    )


def _final_package_config_from_input(
    value: object,
    *,
    bundle_manifest_sha256: str,
) -> FinalPackageConfig:
    """Build the truthful reconstruction view from an authenticated final-fit contract."""
    if not isinstance(value, Mapping) or set(value) != {
        "dataset",
        "task",
        "family",
        "preprocessing",
        "training",
    }:
        raise ManifestBuildError("Final-fit configuration fields are invalid")
    dataset = value["dataset"]
    task = value["task"]
    family = value["family"]
    preprocessing = value["preprocessing"]
    training = value["training"]
    if (
        not isinstance(dataset, Mapping)
        or set(dataset)
        != {
            "dataset_id",
            "bundle_id",
            "split_assignment_id",
            "cohort",
        }
        or dataset["dataset_id"] != "symile"
        or dataset["cohort"] != "official_train_plus_validation_strict_pneumonia"
        or not isinstance(task, Mapping)
        or set(task) != {"task_id", "label_policy_version"}
        or not isinstance(family, Mapping)
        or set(family) != {"family_id", "modalities", "parameters"}
        or not isinstance(family["family_id"], str)
        or family["family_id"] not in {*FINAL_NEURAL_FAMILIES, *FINAL_TABULAR_FAMILIES}
        or not isinstance(family["modalities"], list)
        or not all(isinstance(item, str) for item in family["modalities"])
        or not isinstance(family["parameters"], Mapping)
        or not isinstance(preprocessing, Mapping)
        or not isinstance(training, Mapping)
        or set(training) != {"parameters", "loader", "augmentation"}
        or not isinstance(training["parameters"], Mapping)
        or not isinstance(training["loader"], Mapping)
        or not isinstance(training["augmentation"], Mapping)
    ):
        raise ManifestBuildError("Final-fit configuration contract is invalid")
    parameters = MappingProxyType(dict(training["parameters"]))
    loader = MappingProxyType(dict(training["loader"]))
    augmentation = MappingProxyType(dict(training["augmentation"]))
    if family["family_id"] in FINAL_NEURAL_FAMILIES:
        batch_size = loader.get("batch_size")
        mixed_precision = parameters.get("mixed_precision")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
            or not isinstance(mixed_precision, bool)
            or any(
                isinstance(augmentation.get(name), bool)
                or not isinstance(augmentation.get(name), int | float)
                for name in (
                    "rotation_degrees",
                    "translation_fraction",
                    "brightness_jitter",
                    "contrast_jitter",
                )
            )
        ):
            raise ManifestBuildError("Final neural fit configuration is incomplete")
    return FinalPackageConfig(
        dataset_id="symile",
        bundle_id=str(dataset["bundle_id"]),
        bundle_manifest_sha256=bundle_manifest_sha256,
        split_assignment_id=str(dataset["split_assignment_id"]),
        task_id=str(task["task_id"]),
        label_policy_version=str(task["label_policy_version"]),
        family_id=str(family["family_id"]),
        modalities=tuple(family["modalities"]),
        family_parameters=MappingProxyType(dict(family["parameters"])),
        preprocessing=MappingProxyType(dict(preprocessing)),
        training_parameters=parameters,
        loader=loader,
        augmentation=augmentation,
    )


def _validate_pretrained_weight(
    config: ExperimentConfig | FinalPackageConfig,
    fingerprint: object,
) -> None:
    family = config.family_id if isinstance(config, FinalPackageConfig) else config.family.family_id
    if family in FINAL_FUSION_FAMILIES:
        if fingerprint is not None:
            raise ManifestBuildError("Final fusion package duplicates CXR pretrained lineage")
        return
    if family != "cxr_densenet" or not isinstance(fingerprint, Mapping):
        raise ManifestBuildError("Final CXR package lacks pretrained-weight lineage")
    try:
        scientific = pretrained_weight_semantic_identity(fingerprint)
    except ValueError as exc:
        raise ManifestBuildError("Final CXR pretrained-weight fingerprint is invalid") from exc
    parameters = (
        config.family_parameters
        if isinstance(config, FinalPackageConfig)
        else config.family.parameters
    )
    if scientific["declared_name"] != parameters["weights"]:
        raise ManifestBuildError("Final CXR pretrained weight differs from configuration")


def _reconstruct_neural(config: FinalPackageConfig) -> torch.nn.Module:
    parameters = config.family_parameters
    if config.family_id == "cxr_densenet":
        return CxrBinaryClassifier(
            StandardCxrEncoder(
                weights=None,
                expected_embedding_dimension=int(parameters["embedding_dimension"]),
                image_size=int(parameters["image_size"]),
            ),
            embedding_dimension=int(parameters["embedding_dimension"]),
            image_size=int(parameters["image_size"]),
        )
    if config.family_id == "cxr_labs_concat":
        return build_symile_concat_model(parameters, weights=None)
    if config.family_id == "cxr_labs_gated":
        return build_symile_gated_model(parameters, weights=None)
    if config.family_id == SYMILE_ECG_GATED_FAMILY:
        return build_symile_trimodal_gated_model(parameters, weights=None)
    raise ManifestBuildError("Final neural reconstruction family is unsupported")


def _load_final_neural_checkpoint(path: Path) -> Mapping[str, object]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        raise ManifestBuildError("Final neural checkpoint is unreadable") from exc
    schema_version = (
        checkpoint.get("checkpoint_schema_version") if isinstance(checkpoint, Mapping) else None
    )
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != {"checkpoint_schema_version", "model_state_dict", "terminal_training"}
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != FINAL_NEURAL_CHECKPOINT_SCHEMA_VERSION
        or not isinstance(checkpoint["terminal_training"], Mapping)
    ):
        raise ManifestBuildError("Final neural checkpoint contract is invalid")
    _validate_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def _expected_files(document: Mapping[str, object]) -> set[str]:
    files = {_MANIFEST_FILENAME, _CONFIG_FILENAME}
    if document["package_kind"] == "neural":
        files.add(_NEURAL_FILENAME)
        family = document["input"]["family"]["family_id"]
        if family in FINAL_FUSION_FAMILIES:
            files.add(_LAB_PREPROCESSOR_FILENAME)
    elif document["package_kind"] == "tabular":
        files.add(_TABULAR_FILENAME)
    return files


def _package_destination(model_root: str | Path, package_id: str) -> Path:
    return Path(model_root) / "final" / "packages" / package_id


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        item.name: _sha256_file(item)
        for item in sorted(directory.iterdir())
        if item.name != _MANIFEST_FILENAME
    }


def _write_manifest(directory: Path, document: Mapping[str, object]) -> None:
    (directory / _MANIFEST_FILENAME).write_text(
        json.dumps(dict(document), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _cpu_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _validate_state_dict(state: object) -> None:
    if (
        not isinstance(state, Mapping)
        or not state
        or any(
            not isinstance(key, str)
            or not isinstance(value, torch.Tensor)
            or not torch.isfinite(value).all()
            for key, value in state.items()
        )
    ):
        raise ManifestBuildError("Final neural state is invalid")


def _identity(value: object, prefix: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(prefix)
        and len(value) == len(prefix) + 64
        and all(character in "0123456789abcdef" for character in value[len(prefix) :])
    )


def _positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestBuildError(f"{label} must be a positive integer")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
