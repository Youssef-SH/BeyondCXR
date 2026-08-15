"""Execute one frozen six-family Symile repeated-CV development lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.errors import ManifestBuildError
from radfusion.data.symile_preprocess import (
    LAB_ECDF_POLICY_VERSION,
    LAB_FEATURE_COLUMNS,
    SymileLabEcdfTransformer,
)
from radfusion.data.symile_schemas import OUTER_FOLDS, REPEAT_SEEDS
from radfusion.models.cxr_baseline import (
    CxrBinaryClassifier,
    StandardCxrEncoder,
    fingerprint_pretrained_weights,
)
from radfusion.models.fusion_concat import initialize_fusion_encoder
from radfusion.models.symile_fusion import build_symile_concat_model, build_symile_gated_model
from radfusion.models.symile_tabular import (
    fit_symile_labs_lightgbm,
    fit_symile_labs_logistic,
    symile_tabular_logits,
)
from radfusion.training.config import (
    ConfigError,
    ExperimentConfig,
    load_symile_development_config,
)
from radfusion.training.device import ResolvedDevice, resolve_device
from radfusion.training.execution import LoaderExecutionPolicy, reused_loader_policy
from radfusion.training.neural import (
    FineTuneScope,
    SelectionMetricName,
    build_evaluation_loader,
    build_image_loaders,
    deterministic_inference,
    fit_two_stage_binary_model,
    seed_neural_runtime,
)
from radfusion.training.symile_data import (
    SymileCxrStore,
    SymileDevelopmentData,
    SymileInnerSplit,
    SymileNeuralDataset,
    SymileOuterFold,
    derive_inner_split,
    load_symile_development,
    materialize_outer_fold,
)
from radfusion.training.symile_families import SYMILE_FUSION_FAMILIES, SYMILE_NEURAL_FAMILIES
from radfusion.utils.mlflow_utils import (
    DEFAULT_TRACKING_URI,
    configure_mlflow,
    environment_provenance,
    git_revision,
    log_source_config,
    serialize_modalities,
    tracked_run,
    uv_lock_sha256,
)
from radfusion.utils.operational_logging import add_logging_argument, configure_logging
from radfusion.utils.private_predictions import (
    ValidatedPredictionEvidence,
    build_prediction_table,
    publish_prediction_evidence,
)
from radfusion.utils.symile_publication import (
    NEURAL_MODEL_FILENAME,
    ValidatedDevelopmentResult,
    ValidatedFoldPackage,
    load_symile_neural_checkpoint,
    neural_checkpoint_document,
    publish_development_result,
    publish_fold_package,
    validate_development_result,
    validate_fold_package,
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
    source_cxr_folds: Mapping[tuple[int, int], ValidatedFoldPackage]
    source_cxr_development_id: str | None


@dataclass(frozen=True)
class CompletedSymileFold:
    """One independently valid model package and its private OOF evidence."""

    package: ValidatedFoldPackage
    prediction: ValidatedPredictionEvidence


def run_symile_development(
    config: ExperimentConfig,
    *,
    source_cxr_development_id: str | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    execution: LoaderExecutionPolicy | None = None,
) -> ValidatedDevelopmentResult:
    """Execute exactly 3 x 5 folds and publish one complete family authority."""
    family_id = config.family.family_id
    _validate_source_argument(family_id, source_cxr_development_id)
    configure_mlflow(experiment_name=config.runtime.experiment_name, tracking_uri=tracking_uri)
    commit, dirty = git_revision()
    if dirty:
        raise ValueError("Formal Symile development requires a clean Git worktree")
    lock_hash = uv_lock_sha256()
    data = load_symile_development(config)
    runtime = None
    loader_execution = None
    cxr_store = None
    if family_id in SYMILE_NEURAL_FAMILIES:
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
        cxr_store = SymileCxrStore(data.bundle, config.runtime.source_root)
    source_folds: Mapping[tuple[int, int], ValidatedFoldPackage] = {}
    if source_cxr_development_id is not None:
        source_folds = _resolve_source_cxr_folds(
            config,
            source_cxr_development_id,
        )
    context = SymileFoldExecutionContext(
        data,
        cxr_store,
        runtime,
        loader_execution,
        commit,
        lock_hash,
        source_folds,
        source_cxr_development_id,
    )
    completed = [
        execute_symile_outer_fold(config, context, repeat_seed=seed, outer_fold=fold)
        for seed in REPEAT_SEEDS
        for fold in OUTER_FOLDS
    ]
    return publish_development_result(
        report_root=config.runtime.report_directory,
        model_root=config.runtime.model_directory,
        prediction_root=config.runtime.private_output_directory,
        manifest_root=config.runtime.manifest_directory,
        family=family_id,
        config_semantic_sha256=config.config_semantic_sha256,
        folds=[item.package for item in completed],
        predictions=[item.prediction for item in completed],
    )


def execute_symile_outer_fold(
    config: ExperimentConfig,
    context: SymileFoldExecutionContext,
    *,
    repeat_seed: int,
    outer_fold: int,
) -> CompletedSymileFold:
    """Fit, select, infer once, and publish one immutable outer-fold package."""
    outer = materialize_outer_fold(context.data, repeat_seed=repeat_seed, outer_fold=outer_fold)
    inner = derive_inner_split(outer)
    semantic_hash = config.config_semantic_sha256
    source = context.source_cxr_folds.get((repeat_seed, outer_fold))
    if (
        source is not None
        and source.manifest["inner_split"]["inner_split_id"] != inner.inner_split_id
    ):
        raise ManifestBuildError("Source CXR fold uses a different inner split")
    source_lineage = _source_cxr_lineage(context.source_cxr_development_id, source)
    tags = {
        "run_kind": "training",
        "evaluation_scope": "oof",
        "dataset_id": config.dataset.dataset_id,
        "task_id": config.task.task_id,
        "family_id": config.family.family_id,
        "modalities": serialize_modalities(config.family.modalities),
        "bundle_id": config.dataset.bundle_id,
        "bundle_manifest_sha256": config.dataset.bundle_manifest_sha256,
        "split_assignment_id": config.dataset.split_assignment_id,
        "cv_assignment_id": config.dataset.cv_assignment_id,
        "repeat_seed": str(repeat_seed),
        "outer_fold": str(outer_fold),
        "inner_split_id": inner.inner_split_id,
        "config_source_sha256": config.config_source_sha256,
        "config_semantic_sha256": semantic_hash,
        "git_commit": context.git_commit,
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
            "runtime_provenance": (
                context.runtime.provenance() if context.runtime is not None else None
            ),
        }
        published = publish_fold_package(
            model_root=config.runtime.model_directory,
            family=config.family.family_id,
            repeat_seed=repeat_seed,
            outer_fold=outer_fold,
            config_bytes=config.source_bytes,
            config_sha256=config.config_source_sha256,
            config_semantic_sha256=semantic_hash,
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
            private_root=config.runtime.private_output_directory,
            dataset_id="symile",
            model_package_id=published.manifest["fold_package_id"],
            task_id=config.task.task_id,
            bundle_id=config.dataset.bundle_id,
            split_assignment_id=config.dataset.split_assignment_id,
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
            repeat_seed=outer.repeat_seed,
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
            repeat_seed=outer.repeat_seed,
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
    if config.family.family_id in SYMILE_FUSION_FAMILIES:
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
    train_dataset = SymileNeuralDataset(
        inner_training,
        cxr_store=context.cxr_store,
        transform=training_transform,
        repeat_seed=outer.repeat_seed,
        labs=train_labs,
    )
    validation_dataset = SymileNeuralDataset(
        inner_validation,
        cxr_store=context.cxr_store,
        transform=evaluation_transform,
        repeat_seed=outer.repeat_seed,
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
        ("image",) if config.family.family_id == "cxr_densenet" else ("image", "structured")
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
    holdout_dataset = SymileNeuralDataset(
        outer.holdout,
        cxr_store=context.cxr_store,
        transform=evaluation_transform,
        repeat_seed=outer.repeat_seed,
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
        "logits": inference.logits,
        "selection": {
            "metric": "roc_auc",
            "selected_epoch": fit.selected_epoch,
            "selected_stage": fit.selected_stage,
            "selected_validation_metric": fit.selected_validation_metric,
            "best_iteration": None,
        },
        "pretrained_weight": pretrained,
    }


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
        if package.manifest_sha256 != reference["fold_manifest_sha256"]:
            raise ManifestBuildError("Source CXR fold manifest hash differs from family authority")
        lineage = package.manifest["lineage"]
        required = {
            "bundle_id": config.dataset.bundle_id,
            "bundle_manifest_sha256": config.dataset.bundle_manifest_sha256,
            "split_assignment_id": config.dataset.split_assignment_id,
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
        "fold_manifest_sha256": source.manifest_sha256,
    }


def _fold_lineage(
    config: ExperimentConfig,
    context: SymileFoldExecutionContext,
    pretrained_weight: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "bundle_id": config.dataset.bundle_id,
        "bundle_manifest_sha256": context.data.bundle_manifest_sha256,
        "split_assignment_id": config.dataset.split_assignment_id,
        "cv_assignment_id": config.dataset.cv_assignment_id,
        "cv_manifest_sha256": context.data.cv_reference.manifest_sha256,
        "task_id": config.task.task_id,
        "git_commit": context.git_commit,
        "dependency_lock_sha256": context.dependency_lock_sha256,
        "encoder_identity": (
            _encoder_identity(config) if config.family.family_id in SYMILE_NEURAL_FAMILIES else None
        ),
        "transform_contract": (
            _transform(config, training=False).contract()
            if config.family.family_id in SYMILE_NEURAL_FAMILIES
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
    return {
        "roc_auc": float(roc_auc_score(truth, scores)),
        "average_precision": float(average_precision_score(truth, scores)),
        "brier_score": float(brier_score_loss(truth, scores)),
    }


def _validate_source_argument(family: str, development_id: str | None) -> None:
    if family in SYMILE_FUSION_FAMILIES:
        if development_id is None:
            raise ConfigError("Symile fusion development requires a source CXR development ID")
    elif development_id is not None:
        raise ConfigError("This Symile family does not accept a source CXR development ID")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-cxr-development-id", default=None)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_symile_development_config(args.config)
        result = run_symile_development(
            config,
            source_cxr_development_id=args.source_cxr_development_id,
            tracking_uri=args.tracking_uri,
        )
    except (ConfigError, ManifestBuildError, OSError, RuntimeError, ValueError) as exc:
        print(f"Symile development failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "family_id": result.manifest["family_id"],
                "development_id": result.manifest["development_id"],
                "development_manifest_sha256": result.manifest_sha256,
                "report_directory": result.directory.as_posix(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
