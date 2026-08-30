"""Execute one prespecified Symile family over the repeated-CV design."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_preprocess import (
    LAB_ECDF_POLICY_VERSION,
    LAB_FEATURE_COLUMNS,
    SymileLabEcdfTransformer,
    load_symile_lab_preprocessor,
)
from beyondcxr.data.symile_schemas import OUTER_FOLDS, REPEAT_SEEDS
from beyondcxr.data.symile_source import DevelopmentReleaseAsset
from beyondcxr.models.cxr_baseline import (
    CxrBinaryClassifier,
    StandardCxrEncoder,
    fingerprint_pretrained_weights,
)
from beyondcxr.models.fusion_concat import initialize_fusion_encoder
from beyondcxr.models.symile_ecg_fusion import build_symile_trimodal_gated_model
from beyondcxr.models.symile_fusion import build_symile_concat_model, build_symile_gated_model
from beyondcxr.models.symile_tabular import (
    fit_symile_labs_lightgbm,
    fit_symile_labs_logistic,
    symile_tabular_logits,
)
from beyondcxr.training.config import (
    ConfigError,
    ExperimentConfig,
    load_symile_development_config,
)
from beyondcxr.training.device import ResolvedDevice, resolve_device
from beyondcxr.training.execution import LoaderExecutionPolicy, reused_loader_policy
from beyondcxr.training.neural import (
    FineTuneScope,
    SelectionMetricName,
    build_evaluation_loader,
    build_image_loaders,
    configure_neural_determinism,
    deterministic_inference,
    fit_two_stage_binary_model,
    seed_neural_runtime,
)
from beyondcxr.training.symile_data import (
    SymileCxrStore,
    SymileDevelopmentData,
    SymileInnerSplit,
    SymileNeuralDataset,
    SymileOuterFold,
    derive_inner_split,
    load_symile_development,
    materialize_outer_fold,
)
from beyondcxr.training.symile_ecg_data import SymileEcgStore, SymileTriModalDataset
from beyondcxr.training.symile_execution import (
    ValidatedSymileProspectiveMutation,
    build_symile_formal_preflight,
    require_symile_prospective_mutation,
    require_symile_prospective_path,
    symile_prospective_mutation,
    validate_active_symile_execution,
    validate_current_symile_execution,
    validate_symile_lifecycle,
)
from beyondcxr.training.symile_families import (
    SYMILE_ALL_FUSION_FAMILIES,
    SYMILE_ALL_NEURAL_FAMILIES,
    SYMILE_CORE_DEVELOPMENT_FAMILIES,
    SYMILE_ECG_GATED_FAMILY,
)
from beyondcxr.training.symile_statistics import metrics as symile_headline_metrics
from beyondcxr.utils.mlflow_utils import (
    DEFAULT_TRACKING_URI,
    configure_mlflow,
    discover_repository_root,
    environment_provenance,
    log_source_config,
    serialize_modalities,
    tracked_run,
)
from beyondcxr.utils.operational_logging import add_logging_argument, configure_logging
from beyondcxr.utils.package_identity import package_scientific_config_payload
from beyondcxr.utils.private_predictions import (
    ValidatedPredictionEvidence,
    build_prediction_table,
    publish_prediction_evidence,
    validate_prediction_evidence,
)
from beyondcxr.utils.provenance import git_revision, uv_lock_sha256
from beyondcxr.utils.publication import is_publication_staging_directory
from beyondcxr.utils.skops_io import load_skops
from beyondcxr.utils.symile_publication import (
    LAB_PREPROCESSOR_FILENAME,
    NEURAL_MODEL_FILENAME,
    TABULAR_MODEL_FILENAME,
    ValidatedDevelopmentResult,
    ValidatedFoldPackage,
    load_symile_neural_checkpoint,
    neural_checkpoint_document,
    publish_development_result,
    publish_fold_package,
    reconstruct_fold_neural_model,
    validate_development_result,
    validate_fold_package,
    validate_fold_prediction_pair,
)


@dataclass(frozen=True)
class SymileFoldExecutionContext:
    """Reusable validated state for one family invocation."""

    data: SymileDevelopmentData
    cxr_store: SymileCxrStore | None
    runtime: ResolvedDevice | None
    loader_execution: LoaderExecutionPolicy | None
    git_commit: str
    dependency_lock_sha256: str
    python_version: str
    source_cxr_folds: Mapping[tuple[int, int], ValidatedFoldPackage]
    source_cxr_development_id: str | None
    ecg_store: SymileEcgStore | None = None


@dataclass(frozen=True)
class CompletedSymileFold:
    """One independently valid model package and its private OOF evidence."""

    package: ValidatedFoldPackage
    prediction: ValidatedPredictionEvidence


def run_symile_development(
    config: ExperimentConfig,
    *,
    prospective_mutation: ValidatedSymileProspectiveMutation,
    source_cxr_development_id: str | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    execution: LoaderExecutionPolicy | None = None,
    repository_root: str | Path = ".",
    development_assets: tuple[DevelopmentReleaseAsset, ...] | None = None,
) -> ValidatedDevelopmentResult:
    """Execute exactly 3 x 5 folds and publish one complete family authority."""
    require_symile_prospective_mutation(prospective_mutation)
    require_symile_prospective_path(
        prospective_mutation,
        config.runtime.model_directory,
        "Symile development model root",
    )
    require_symile_prospective_path(
        prospective_mutation,
        config.runtime.report_directory,
        "Symile development report root",
    )
    require_symile_prospective_path(
        prospective_mutation,
        config.runtime.private_output_directory,
        "Symile development private root",
    )
    family_id = config.family.family_id
    _validate_source_argument(family_id, source_cxr_development_id)
    commit, dirty = git_revision(repository_root)
    if dirty:
        raise ValueError("Formal Symile development requires a clean Git worktree")
    lock_hash = uv_lock_sha256(Path(repository_root) / "uv.lock")
    configure_mlflow(experiment_name=config.runtime.experiment_name, tracking_uri=tracking_uri)
    data = load_symile_development(config)
    runtime = None
    loader_execution = None
    cxr_store = None
    if family_id in SYMILE_ALL_NEURAL_FAMILIES:
        if config.neural is None or config.runtime.source_root is None:
            raise ConfigError("Symile neural development configuration is incomplete")
        runtime = resolve_device(
            config.runtime.device,
            mixed_precision=config.neural.mixed_precision,
            pin_memory_policy=config.runtime.pin_memory_policy,
        )
        loader_execution = execution or reused_loader_policy(
            num_workers=config.runtime.num_workers,
            pin_memory=runtime.pin_memory_effective,
        )
        cxr_store = SymileCxrStore(
            data.bundle,
            config.runtime.source_root,
            development_assets=development_assets,
        )
    ecg_store = (
        SymileEcgStore(
            data.bundle,
            config.runtime.source_root,
            development_assets=development_assets,
        )
        if family_id == SYMILE_ECG_GATED_FAMILY and config.runtime.source_root is not None
        else None
    )
    source_folds: Mapping[tuple[int, int], ValidatedFoldPackage] = {}
    if source_cxr_development_id is not None:
        source_folds = _resolve_source_cxr_folds(
            config,
            source_cxr_development_id,
        )
    context = SymileFoldExecutionContext(
        data=data,
        cxr_store=cxr_store,
        runtime=runtime,
        loader_execution=loader_execution,
        git_commit=commit,
        dependency_lock_sha256=lock_hash,
        python_version=str(
            prospective_mutation.lifecycle.current_execution.execution.manifest[
                "framework_versions"
            ]["python"]
        ),
        source_cxr_folds=source_folds,
        source_cxr_development_id=source_cxr_development_id,
        ecg_store=ecg_store,
    )
    completed = [
        execute_symile_outer_fold(
            config,
            context,
            prospective_mutation=prospective_mutation,
            repeat_seed=seed,
            outer_fold=fold,
        )
        for seed in REPEAT_SEEDS
        for fold in OUTER_FOLDS
    ]
    return publish_development_result(
        report_root=config.runtime.report_directory,
        model_root=config.runtime.model_directory,
        prediction_root=config.runtime.private_output_directory,
        manifest_root=config.runtime.manifest_directory,
        family=family_id,
        folds=[item.package for item in completed],
        predictions=[item.prediction for item in completed],
        prospective_mutation=prospective_mutation,
    )


def execute_symile_outer_fold(
    config: ExperimentConfig,
    context: SymileFoldExecutionContext,
    *,
    prospective_mutation: ValidatedSymileProspectiveMutation,
    repeat_seed: int,
    outer_fold: int,
) -> CompletedSymileFold:
    """Fit, select, infer once, and publish one immutable outer-fold package."""
    require_symile_prospective_mutation(prospective_mutation)
    require_symile_prospective_path(
        prospective_mutation,
        config.runtime.model_directory,
        "Symile fold model root",
    )
    outer = materialize_outer_fold(context.data, repeat_seed=repeat_seed, outer_fold=outer_fold)
    inner = derive_inner_split(outer)
    source = context.source_cxr_folds.get((repeat_seed, outer_fold))
    if (
        source is not None
        and source.manifest["inner_split"]["inner_split_id"] != inner.inner_split_id
    ):
        raise ManifestBuildError("Source CXR fold uses a different inner split")
    source_lineage = _source_cxr_lineage(context.source_cxr_development_id, source)
    resumed = _resume_outer_fold(
        config, context, outer, inner, source_lineage, prospective_mutation
    )
    if resumed is not None:
        return resumed
    tags = {
        "run_kind": "training",
        "evaluation_scope": "oof",
        "dataset_id": config.dataset.dataset_id,
        "task_id": config.task.task_id,
        "label_policy_version": config.task.label_policy_version,
        "family_id": config.family.family_id,
        "modalities": serialize_modalities(config.family.modalities),
        "bundle_id": config.dataset.bundle_id,
        "cv_assignment_id": config.dataset.cv_assignment_id,
        "repeat_seed": str(repeat_seed),
        "outer_fold": str(outer_fold),
        "inner_split_id": inner.inner_split_id,
        "git_commit": context.git_commit,
        "git_dirty": "false",
        "dependency_lock_sha256": context.dependency_lock_sha256,
        "run_complete": "false",
    }
    if source_lineage is not None:
        tags["source_development_id"] = source_lineage["development_id"]
        tags["source_package_id"] = source_lineage["fold_package_id"]
    published: ValidatedFoldPackage | None = None
    prediction: ValidatedPredictionEvidence | None = None
    with tracked_run(
        run_name=f"{config.family.family_id}-r{repeat_seed}-f{outer_fold}",
        tags=tags,
        parameters={
            "repeat_seed": repeat_seed,
            "outer_fold": outer_fold,
            "inner_seed": inner.inner_seed,
            **environment_provenance(),
        },
    ) as run_id:
        log_source_config(config)
        fit = _fit_outer_fold(config, context, outer, inner, source)
        oof = build_prediction_table(
            outer.holdout["sample_id"].astype(str).tolist(),
            outer.holdout["target"].to_numpy(dtype=np.int8),
            fit["logits"],
        )
        lineage = _fold_lineage(config, context, fit.get("pretrained_weight"))
        operational = {
            "mlflow_run_id": run_id,
            "python_version": context.python_version,
            "runtime_provenance": (
                context.runtime.provenance() if context.runtime is not None else None
            ),
            "loader_execution": (
                context.loader_execution.provenance()
                if context.loader_execution is not None
                else None
            ),
        }
        published = publish_fold_package(
            prospective_mutation=prospective_mutation,
            model_root=config.runtime.model_directory,
            family=config.family.family_id,
            repeat_seed=repeat_seed,
            outer_fold=outer_fold,
            config_bytes=config.source_bytes,
            lineage=lineage,
            inner_split={
                "inner_split_id": inner.inner_split_id,
                "inner_seed": inner.inner_seed,
                "policy": inner.policy,
            },
            selection=fit["selection"],
            model=fit["model"],
            lab_preprocessor=fit.get("lab_preprocessor"),
            training_history=fit.get("training_history"),
            source_cxr=source_lineage,
            operational=operational,
        )
        prediction = publish_prediction_evidence(
            prospective_mutation=prospective_mutation,
            private_root=config.runtime.private_output_directory,
            dataset_id="symile",
            model_package_id=published.manifest["fold_package_id"],
            task_id=config.task.task_id,
            bundle_id=config.dataset.bundle_id,
            scope="outer_fold_oof",
            sample_ids=oof["sample_id"].to_pylist(),
            targets=oof["target"].to_pylist(),
            logits=oof["logit"].to_pylist(),
            cv_assignment_id=config.dataset.cv_assignment_id,
            repeat_seed=repeat_seed,
            outer_fold=outer_fold,
        )
        probabilities = oof["probability"].to_numpy()
        targets = oof["target"].to_numpy()
        metrics = _probability_metrics(targets, probabilities)
        mlflow.log_metrics(metrics)
        selection_parameters = {
            key: value
            for key, value in {
                "selected_epoch": fit["selection"].get("selected_epoch"),
                "best_iteration": fit["selection"].get("best_iteration"),
            }.items()
            if value is not None
        }
        if selection_parameters:
            mlflow.log_params(selection_parameters)
        mlflow.set_tags(
            {
                "package_kind": "fold",
                "package_id": published.manifest["fold_package_id"],
                "prediction_id": prediction.prediction_id,
            }
        )
        mlflow.set_tag("run_complete", "true")
    if published is None or prediction is None:
        raise RuntimeError("Symile fold lifecycle completed without its package/evidence pair")
    return CompletedSymileFold(published, prediction)


def _resume_outer_fold(
    config: ExperimentConfig,
    context: SymileFoldExecutionContext,
    outer: SymileOuterFold,
    inner: SymileInnerSplit,
    source_lineage: Mapping[str, object] | None,
    prospective_mutation: ValidatedSymileProspectiveMutation,
) -> CompletedSymileFold | None:
    """Resume an exact fitted coordinate, recovering only missing OOF inference."""
    from beyondcxr.training.symile_campaign_control import validate_fold_execution_provenance

    candidates = []
    for directory, manifest in _published_fold_manifests(
        config.runtime.model_directory / "packages"
    ):
        if (
            manifest.get("family_id") == config.family.family_id
            and manifest.get("repeat_seed") == outer.repeat_seed
            and manifest.get("outer_fold") == outer.outer_fold
        ):
            package = validate_fold_package(directory)
            manifest = package.manifest
            expected_lineage = _fold_lineage(
                config, context, manifest["lineage"]["pretrained_weight"]
            )
            if (
                manifest["config"]["fit_config"] != package_scientific_config_payload(config)
                or manifest["lineage"] != expected_lineage
                or manifest["source_cxr"] != source_lineage
                or manifest["inner_split"]
                != {
                    "inner_split_id": inner.inner_split_id,
                    "inner_seed": inner.inner_seed,
                    "policy": inner.policy,
                }
            ):
                raise ManifestBuildError("Existing fold conflicts with the prospective campaign")
            validate_fold_execution_provenance(
                package, prospective_mutation.lifecycle.current_execution.execution.manifest
            )
            candidates.append(package)
    if len(candidates) > 1:
        raise ManifestBuildError("Symile fold selection is ambiguous")
    if not candidates:
        return None
    package = candidates[0]
    package_id = package.manifest["fold_package_id"]
    authority = outer.holdout.assign(repeat_seed=outer.repeat_seed, outer_fold=outer.outer_fold)
    predictions = []
    root = config.runtime.private_output_directory / "predictions/symile/oof"
    for directory, manifest in _published_fold_manifests(root):
        if manifest.get("model_package_id") == package_id:
            evidence = validate_prediction_evidence(directory, expected_model_package_id=package_id)
            validate_fold_prediction_pair(package, evidence, authority)
            predictions.append(evidence)
    if len(predictions) > 1:
        raise ManifestBuildError("Symile fold prediction selection is ambiguous")
    if predictions:
        return CompletedSymileFold(package, predictions[0])
    if config.family.family_id not in SYMILE_ALL_NEURAL_FAMILIES:
        model = load_skops(package.directory / TABULAR_MODEL_FILENAME)
        logits = symile_tabular_logits(model, outer.holdout[list(LAB_FEATURE_COLUMNS)])
    else:
        checkpoint = load_symile_neural_checkpoint(package.directory / NEURAL_MODEL_FILENAME)
        model = reconstruct_fold_neural_model(config, checkpoint)
        if context.runtime is None or context.cxr_store is None:
            raise ConfigError("Symile neural fold context is incomplete")
        model.to(context.runtime.device)
        labs = None
        if config.family.family_id in SYMILE_ALL_FUSION_FAMILIES:
            transformer = load_symile_lab_preprocessor(
                package.directory / LAB_PREPROCESSOR_FILENAME
            )
            labs = transformer.transform(outer.holdout[list(LAB_FEATURE_COLUMNS)])
        logits = _infer_neural_holdout(config, context, outer, model, labs)
    evidence = publish_prediction_evidence(
        prospective_mutation=prospective_mutation,
        private_root=config.runtime.private_output_directory,
        dataset_id="symile",
        model_package_id=package_id,
        task_id=config.task.task_id,
        bundle_id=config.dataset.bundle_id,
        scope="outer_fold_oof",
        sample_ids=outer.holdout["sample_id"].astype(str).tolist(),
        targets=outer.holdout["target"].to_numpy(dtype=np.int8),
        logits=logits,
        cv_assignment_id=config.dataset.cv_assignment_id,
        repeat_seed=outer.repeat_seed,
        outer_fold=outer.outer_fold,
    )
    validate_fold_prediction_pair(package, evidence, authority)
    return CompletedSymileFold(package, evidence)


def _published_fold_manifests(root: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Inspect bounded publication coordinates before recursive candidate validation."""
    if not root.exists() and not root.is_symlink():
        return
    if root.is_symlink() or not root.is_dir():
        raise ManifestBuildError("Symile fold publication root is invalid")
    for directory in root.iterdir():
        if is_publication_staging_directory(directory):
            continue
        path = directory / "manifest.json"
        if directory.is_symlink() or not directory.is_dir() or path.is_symlink():
            raise ManifestBuildError("Symile fold publication member is invalid")
        try:
            document = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestBuildError("Symile fold publication manifest is unreadable") from exc
        if not isinstance(document, dict):
            raise ManifestBuildError("Symile fold publication manifest is invalid")
        yield directory, document


def _fit_outer_fold(
    config: ExperimentConfig,
    context: SymileFoldExecutionContext,
    outer: SymileOuterFold,
    inner: SymileInnerSplit,
    source: ValidatedFoldPackage | None,
) -> dict[str, Any]:
    lab_columns = list(LAB_FEATURE_COLUMNS)
    targets = outer.training["target"].to_numpy(dtype=np.int8)
    family_id = config.family.family_id
    if family_id == "labs_logistic":
        fit = fit_symile_labs_logistic(
            outer.training[lab_columns],
            targets,
            parameters=config.training.parameters,
            selection_metric=config.training.selection_metric,
            lab_policy=str(config.preprocessing["lab_policy"]),
            training_seed=outer.repeat_seed,
        )
        return {
            "model": fit.pipeline,
            "logits": symile_tabular_logits(fit.pipeline, outer.holdout[lab_columns]),
            "selection": {
                "metric": config.training.selection_metric,
                "selected_epoch": None,
                "best_iteration": None,
            },
        }
    if family_id == "labs_lightgbm":
        fit = fit_symile_labs_lightgbm(
            outer.training[lab_columns],
            targets,
            parameters={**config.family.parameters, **config.training.parameters},
            selection_metric=config.training.selection_metric,
            lab_policy=str(config.preprocessing["lab_policy"]),
            inner_training_indices=inner.training_indices,
            inner_validation_indices=inner.validation_indices,
            training_seed=outer.repeat_seed,
        )
        return {
            "model": fit.pipeline,
            "logits": symile_tabular_logits(fit.pipeline, outer.holdout[lab_columns]),
            "selection": {
                "metric": config.training.selection_metric,
                "selected_epoch": None,
                "best_iteration": fit.best_iteration,
            },
        }
    return _fit_neural_outer_fold(config, context, outer, inner, source)


def _fit_neural_outer_fold(
    config: ExperimentConfig,
    context: SymileFoldExecutionContext,
    outer: SymileOuterFold,
    inner: SymileInnerSplit,
    source: ValidatedFoldPackage | None,
) -> dict[str, Any]:
    if config.neural is None or context.cxr_store is None or context.runtime is None:
        raise ConfigError("Symile neural fold context is incomplete")
    seed_neural_runtime(outer.repeat_seed)
    lab_preprocessor = None
    transformed_training = None
    transformed_holdout = None
    if config.family.family_id in SYMILE_ALL_FUSION_FAMILIES:
        if config.preprocessing["lab_policy"] != LAB_ECDF_POLICY_VERSION:
            raise ConfigError("Symile laboratory preprocessing policy is unsupported")
        lab_preprocessor = SymileLabEcdfTransformer().fit(outer.training[list(LAB_FEATURE_COLUMNS)])
        transformed_training = lab_preprocessor.transform(outer.training[list(LAB_FEATURE_COLUMNS)])
        transformed_holdout = lab_preprocessor.transform(outer.holdout[list(LAB_FEATURE_COLUMNS)])
    training_transform = _transform(config, training=True)
    evaluation_transform = _transform(config, training=False)
    inner_training = outer.training.iloc[inner.training_indices].reset_index(drop=True)
    inner_validation = outer.training.iloc[inner.validation_indices].reset_index(drop=True)
    train_labs = (
        transformed_training[inner.training_indices] if transformed_training is not None else None
    )
    validation_labs = (
        transformed_training[inner.validation_indices] if transformed_training is not None else None
    )
    if config.family.family_id == SYMILE_ECG_GATED_FAMILY:
        if context.ecg_store is None or train_labs is None or validation_labs is None:
            raise ConfigError("Symile ECG development context is incomplete")
        train_dataset = SymileTriModalDataset(
            inner_training,
            cxr_store=context.cxr_store,
            ecg_store=context.ecg_store,
            transform=training_transform,
            training_seed=outer.repeat_seed,
            labs=train_labs,
        )
        validation_dataset = SymileTriModalDataset(
            inner_validation,
            cxr_store=context.cxr_store,
            ecg_store=context.ecg_store,
            transform=evaluation_transform,
            training_seed=outer.repeat_seed,
            labs=validation_labs,
        )
    else:
        train_dataset = SymileNeuralDataset(
            inner_training,
            cxr_store=context.cxr_store,
            transform=training_transform,
            training_seed=outer.repeat_seed,
            labs=train_labs,
        )
        validation_dataset = SymileNeuralDataset(
            inner_validation,
            cxr_store=context.cxr_store,
            transform=evaluation_transform,
            training_seed=outer.repeat_seed,
            labs=validation_labs,
        )
    loaders = build_image_loaders(
        train_dataset,
        validation_dataset,
        config=config.neural,
        runtime=context.runtime,
        seed=outer.repeat_seed,
        execution=context.loader_execution,
    )
    model, pretrained = _build_neural_model(config, source)
    model.to(context.runtime.device)
    input_keys = (
        ("image", "structured", "ecg")
        if config.family.family_id == SYMILE_ECG_GATED_FAMILY
        else (("image",) if config.family.family_id == "cxr_densenet" else ("image", "structured"))
    )
    fit = fit_two_stage_binary_model(
        model,
        loaders.train,
        loaders.validation,
        input_keys=input_keys,
        config=config.neural,
        runtime=context.runtime,
        pos_weight=float(config.training.parameters["pos_weight"]),
        selection_metric=cast(SelectionMetricName, config.training.selection_metric),
        fine_tune_scope=cast(FineTuneScope, config.training.parameters["fine_tune_scope"]),
    )
    incompatible = model.load_state_dict(fit.selected_state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("Selected Symile state did not restore strictly")
    restored_validation = deterministic_inference(
        model,
        loaders.validation,
        runtime=context.runtime,
        input_keys=input_keys,
    )
    restored_roc_auc = float(
        roc_auc_score(restored_validation.targets, restored_validation.probabilities)
    )
    if not np.isclose(restored_roc_auc, fit.selected_validation_metric, rtol=0.0, atol=1e-12):
        raise ValueError("Restored Symile checkpoint does not reproduce selected AUROC")
    logits = _infer_neural_holdout(config, context, outer, model, transformed_holdout)
    checkpoint = neural_checkpoint_document(
        fit.selected_state_dict,
        selected_epoch=fit.selected_epoch,
        selected_stage=fit.selected_stage,
        selected_validation_roc_auc=fit.selected_validation_metric,
    )
    return {
        "model": checkpoint,
        "lab_preprocessor": lab_preprocessor,
        "training_history": [
            {
                "epoch": record.global_epoch,
                "stage": record.stage,
                "training_loss": record.training_loss,
                "validation_metric": record.validation_metric,
                "encoder_learning_rate": record.encoder_learning_rate,
                "head_learning_rate": record.head_learning_rate,
            }
            for record in fit.history
        ],
        "logits": logits,
        "selection": {
            "metric": "roc_auc",
            "selected_epoch": fit.selected_epoch,
            "selected_stage": fit.selected_stage,
            "selected_validation_metric": fit.selected_validation_metric,
            "best_iteration": None,
        },
        "pretrained_weight": pretrained,
    }


def _infer_neural_holdout(
    config: ExperimentConfig,
    context: SymileFoldExecutionContext,
    outer: SymileOuterFold,
    model: torch.nn.Module,
    transformed_holdout: np.ndarray | None,
) -> np.ndarray:
    if config.neural is None or context.runtime is None or context.cxr_store is None:
        raise ConfigError("Symile neural holdout context is incomplete")
    configure_neural_determinism()
    evaluation_transform = _transform(config, training=False)
    input_keys = (
        ("image", "structured", "ecg")
        if config.family.family_id == SYMILE_ECG_GATED_FAMILY
        else (("image",) if config.family.family_id == "cxr_densenet" else ("image", "structured"))
    )
    if config.family.family_id == SYMILE_ECG_GATED_FAMILY:
        if context.ecg_store is None or transformed_holdout is None:
            raise ConfigError("Symile ECG holdout context is incomplete")
        holdout_dataset = SymileTriModalDataset(
            outer.holdout,
            cxr_store=context.cxr_store,
            ecg_store=context.ecg_store,
            transform=evaluation_transform,
            training_seed=outer.repeat_seed,
            labs=transformed_holdout,
        )
    else:
        holdout_dataset = SymileNeuralDataset(
            outer.holdout,
            cxr_store=context.cxr_store,
            transform=evaluation_transform,
            training_seed=outer.repeat_seed,
            labs=transformed_holdout,
        )
    holdout_loader = build_evaluation_loader(
        holdout_dataset,
        batch_size=config.neural.batch_size,
        runtime=context.runtime,
    )
    inference = deterministic_inference(
        model,
        holdout_loader,
        runtime=context.runtime,
        input_keys=input_keys,
    )
    if inference.sample_ids != tuple(outer.holdout["sample_id"].astype(str)):
        raise ManifestBuildError("Symile neural OOF inference order is invalid")
    return inference.logits


def _build_neural_model(
    config: ExperimentConfig,
    source: ValidatedFoldPackage | None,
) -> tuple[torch.nn.Module, Mapping[str, object] | None]:
    family_id = config.family.family_id
    parameters = config.family.parameters
    if family_id == "cxr_densenet":
        weights = str(parameters["weights"])
        before = fingerprint_pretrained_weights(weights)
        model = CxrBinaryClassifier(
            StandardCxrEncoder(
                weights=str(parameters["weights"]),
                expected_embedding_dimension=int(parameters["embedding_dimension"]),
                image_size=int(parameters["image_size"]),
            ),
            embedding_dimension=int(parameters["embedding_dimension"]),
            image_size=int(parameters["image_size"]),
        )
        after = fingerprint_pretrained_weights(weights)
        if before != after:
            raise ValueError("Pretrained CXR weight bytes changed during model construction")
        return model, before.as_dict()
    if source is None:
        raise ManifestBuildError("Symile fusion requires its matching CXR fold package")
    if family_id == "cxr_labs_concat":
        model = build_symile_concat_model(parameters, weights=None)
    elif family_id == SYMILE_ECG_GATED_FAMILY:
        model = build_symile_trimodal_gated_model(parameters, weights=None)
    else:
        model = build_symile_gated_model(parameters, weights=None)
    checkpoint = load_symile_neural_checkpoint(source.directory / NEURAL_MODEL_FILENAME)
    state = checkpoint["model_state_dict"]
    if not isinstance(state, Mapping):
        raise ManifestBuildError("Source CXR checkpoint state is invalid")
    initialize_fusion_encoder(model, state)
    return model, None


def _resolve_source_cxr_folds(
    config: ExperimentConfig,
    development_id: str,
) -> dict[tuple[int, int], ValidatedFoldPackage]:
    directory = config.runtime.report_directory / "families" / development_id
    development = validate_development_result(
        directory,
        model_root=config.runtime.model_directory,
        prediction_root=config.runtime.private_output_directory,
        manifest_root=config.runtime.manifest_directory,
        expected_development_id=development_id,
    )
    if development.manifest["family_id"] != "cxr_densenet":
        raise ManifestBuildError("Fusion source development result is not CXR-only")
    result: dict[tuple[int, int], ValidatedFoldPackage] = {}
    folds_root = config.runtime.model_directory / "packages"
    for reference in development.manifest["fold_packages"]:
        package = validate_fold_package(
            folds_root / reference["fold_package_id"],
            expected_fold_package_id=reference["fold_package_id"],
        )
        lineage = package.manifest["lineage"]
        required = {
            "bundle_id": config.dataset.bundle_id,
            "cv_assignment_id": config.dataset.cv_assignment_id,
            "task_id": config.task.task_id,
            "encoder_identity": _encoder_identity(config),
            "transform_contract": _transform(config, training=False).contract(),
        }
        if any(lineage.get(key) != value for key, value in required.items()):
            raise ManifestBuildError("Source CXR fold is incompatible with fusion development")
        coordinate = (package.manifest["repeat_seed"], package.manifest["outer_fold"])
        result[coordinate] = package
    if set(result) != {(seed, fold) for seed in REPEAT_SEEDS for fold in OUTER_FOLDS}:
        raise ManifestBuildError("Source CXR development lacks one or more exact folds")
    return result


def _source_cxr_lineage(
    development_id: str | None,
    source: ValidatedFoldPackage | None,
) -> dict[str, object] | None:
    if development_id is None and source is None:
        return None
    if development_id is None or source is None:
        raise ManifestBuildError("Source CXR lineage is incomplete")
    return {
        "development_id": development_id,
        "fold_package_id": source.manifest["fold_package_id"],
    }


def _fold_lineage(
    config: ExperimentConfig,
    context: SymileFoldExecutionContext,
    pretrained_weight: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "bundle_id": config.dataset.bundle_id,
        "cv_assignment_id": config.dataset.cv_assignment_id,
        "task_id": config.task.task_id,
        "git_commit": context.git_commit,
        "dependency_lock_sha256": context.dependency_lock_sha256,
        "encoder_identity": (
            _encoder_identity(config)
            if config.family.family_id in SYMILE_ALL_NEURAL_FAMILIES
            else None
        ),
        "transform_contract": (
            _transform(config, training=False).contract()
            if config.family.family_id in SYMILE_ALL_NEURAL_FAMILIES
            else None
        ),
        "pretrained_weight": dict(pretrained_weight) if pretrained_weight is not None else None,
    }


def _transform(config: ExperimentConfig, *, training: bool) -> StandardCxrTransform:
    if config.neural is None:
        raise ConfigError("Symile neural transform requires neural configuration")
    return StandardCxrTransform(
        training=training,
        policy_version=str(config.preprocessing["cxr_transform_policy"]),
        image_size=int(config.family.parameters["image_size"]),
        rotation_degrees=config.neural.rotation_degrees,
        translation_fraction=config.neural.translation_fraction,
        brightness_jitter=config.neural.brightness_jitter,
        contrast_jitter=config.neural.contrast_jitter,
    )


def _encoder_identity(config: ExperimentConfig) -> dict[str, object]:
    return {
        "library": "torchxrayvision",
        "architecture": str(config.family.parameters["encoder_name"]),
        "weights": str(config.family.parameters["weights"]),
        "embedding_dimension": int(config.family.parameters["embedding_dimension"]),
    }


def _probability_metrics(targets: object, probabilities: object) -> dict[str, float]:
    truth = np.asarray(targets, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.shape != scores.shape or truth.ndim != 1 or set(truth.tolist()) != {0, 1}:
        raise ManifestBuildError("Symile metric inputs are invalid")
    return symile_headline_metrics(truth, scores)


def _validate_source_argument(family: str, development_id: str | None) -> None:
    if family in SYMILE_ALL_FUSION_FAMILIES:
        if development_id is None:
            raise ConfigError("Symile fusion development requires a source CXR development ID")
    elif development_id is not None:
        raise ConfigError("This Symile family does not accept a source CXR development ID")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-cxr-development-id", default=None)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--backup-root", type=Path, required=True)
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        repository_root = discover_repository_root()
        config = load_symile_development_config(args.config)
        if config.family.family_id not in SYMILE_CORE_DEVELOPMENT_FAMILIES:
            raise ConfigError("The public development command is limited to the six core families")
        execution = validate_active_symile_execution(repository_root / "private/control/symile")
        config = load_symile_development_config(args.config, **execution.manifest["dataset"])
        current = validate_current_symile_execution(
            execution,
            build_symile_formal_preflight(repository_root, **execution.manifest["dataset"]),
        )
        lifecycle = validate_symile_lifecycle(
            current,
            repository_root=repository_root,
            private_root=repository_root / "private",
            backup_root=args.backup_root,
        )
        with symile_prospective_mutation(lifecycle) as prospective_mutation:
            result = run_symile_development(
                config,
                prospective_mutation=prospective_mutation,
                source_cxr_development_id=args.source_cxr_development_id,
                tracking_uri=args.tracking_uri,
                repository_root=repository_root,
            )
    except (ConfigError, ManifestBuildError, OSError, RuntimeError, ValueError) as exc:
        print(f"Symile development failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "family_id": result.manifest["family_id"],
                "development_id": result.manifest["development_id"],
                "report_directory": result.directory.as_posix(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
