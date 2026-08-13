"""Train and validate one configured RSNA CXR family."""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
from torch import nn

from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.rsna_cxr_cache import ValidatedCxrCache
from radfusion.evaluation.metrics import (
    OperatingPointMetrics,
    ProbabilityMetrics,
    evaluate_operating_point,
    evaluate_probabilities,
    target_sensitivity_threshold,
    youden_j_threshold,
)
from radfusion.models.cxr_baseline import fingerprint_pretrained_weights
from radfusion.training.config import (
    ExperimentConfig,
    require_runtime_seed,
)
from radfusion.training.device import resolve_device
from radfusion.training.execution import LoaderExecutionPolicy, reused_loader_policy
from radfusion.training.neural import (
    CLASS_WEIGHT_POLICY_VERSION,
    EpochThroughput,
    SelectedTrainingResult,
    TrainingEpochRecord,
    build_image_loaders,
    deterministic_inference,
    fit_rsna_cxr_model,
    seed_neural_runtime,
    training_class_weight,
)
from radfusion.training.rsna_datasets import (
    CxrRunData,
    RsnaCachedImageDataset,
    RsnaDataset,
    expected_rsna_cxr_cache_identity,
    prepare_rsna_cxr_cache,
)
from radfusion.training.rsna_interfaces import RsnaCxrModelImplementation
from radfusion.training.rsna_registry import get_dataset, get_model
from radfusion.training.rsna_train_metadata import (
    metrics_document,
    mlflow_metrics,
    validate_report_set,
    write_run_reports,
)
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
from radfusion.utils.operational_logging import (
    CountProgress,
    get_operational_logger,
    log_event,
    timed_phase,
)
from radfusion.utils.privacy import validate_public_reports
from radfusion.utils.publication import publish_directory, staging_directory
from radfusion.utils.rsna_model_publication import threshold_contract
from radfusion.utils.rsna_neural_publication import (
    checkpoint_document,
    load_neural_checkpoint,
    publish_neural_model_package,
    save_neural_checkpoint,
    strict_load_checkpoint,
)

_LOGGER = get_operational_logger(__name__)


@dataclass(frozen=True)
class CxrModelResult:
    """Published outputs from one completed CXR training run."""

    run_id: str
    validation_probability: ProbabilityMetrics
    validation_youden_j: OperatingPointMetrics
    validation_target_sensitivity: OperatingPointMetrics
    thresholds: dict[str, float]
    model_path: Path
    model_sha256: str
    model_package_id: str
    artifact_directory: Path
    model_size_mib: float


def train_cxr_experiment(
    config: ExperimentConfig,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    cache: ValidatedCxrCache | None = None,
    execution: LoaderExecutionPolicy | None = None,
) -> CxrModelResult:
    """Train on CXR train/validation partitions and publish one selected package."""
    if config.family.family_id != "cxr_densenet" or config.neural is None:
        raise ValueError("CXR training requires a complete CXR experiment configuration")
    if config.evaluation is None:
        raise ValueError("RSNA CXR training requires evaluation policy")
    seed = require_runtime_seed(config)
    neural = config.neural
    family = config.family
    configure_mlflow(
        experiment_name=config.runtime.experiment_name,
        tracking_uri=tracking_uri,
    )
    commit, dirty = git_revision()
    lock_hash = uv_lock_sha256()
    environment = environment_provenance()
    base_tags = {
        "run_kind": "training",
        "evaluation_scope": "validation",
        "dataset_id": config.dataset.dataset_id,
        "task_id": config.task.task_id,
        "family_id": family.family_id,
        "modalities": serialize_modalities(family.modalities),
        "bundle_id": config.dataset.bundle_id,
        "bundle_manifest_sha256": config.dataset.bundle_manifest_sha256,
        "split_assignment_id": config.dataset.split_assignment_id,
        "seed": str(seed),
        "git_commit": commit,
        "dependency_lock_sha256": lock_hash,
        "config_source_sha256": config.config_source_sha256,
        "config_semantic_sha256": config.config_semantic_sha256,
        "run_complete": "false",
    }
    initial_parameters = {
        "training_seed": seed,
        "requested_device": config.runtime.device,
        "requested_mixed_precision": neural.mixed_precision,
        "requested_pin_memory_policy": config.runtime.pin_memory_policy,
        "sensitivity_target": config.evaluation.sensitivity_target,
        "calibration_bins": config.evaluation.calibration_bins,
        **dict(family.parameters),
        **environment,
    }
    with tracked_run(
        run_name=family.family_id,
        tags=base_tags,
        parameters=initial_parameters,
    ) as run_id:
        log_source_config(config)
        context = {"run_id": run_id, "family_id": family.family_id}
        dataset_adapter = get_dataset(config.dataset.dataset_id)
        with timed_phase(_LOGGER, "dataset_loading", **context):
            cxr_data = dataset_adapter.load_cxr_train_validation(config)
        if cxr_data.lineage.split_assignment_id != config.dataset.split_assignment_id:
            raise ValueError("Loaded split assignment differs from the configuration")
        with timed_phase(_LOGGER, "cxr_runtime_preparation", **context):
            seed_neural_runtime(seed)
            train_transform = _transform(config, training=True)
            evaluation_transform = _transform(config, training=False)
            resolved_cache = cache or prepare_rsna_cxr_cache(
                cast(RsnaDataset, dataset_adapter), config, evaluation_transform
            )
            expected_cache_identity = expected_rsna_cxr_cache_identity(
                lineage=cxr_data.lineage,
                bundle_manifest_sha256=cxr_data.bundle_manifest_sha256,
                source_inventory=cxr_data.source_inventory,
                transform=evaluation_transform,
            )
            authentication = resolved_cache.source_authentication.as_dict()
            mlflow.log_param("source_authentication_policy", authentication["policy_version"])
            train_dataset = RsnaCachedImageDataset(
                cxr_data.train,
                cache=resolved_cache,
                expected_cache_identity=expected_cache_identity,
                partition="train",
                transform=train_transform,
                training_seed=seed,
            )
            validation_dataset = RsnaCachedImageDataset(
                cxr_data.validation,
                cache=resolved_cache,
                expected_cache_identity=expected_cache_identity,
                partition="validation",
                transform=evaluation_transform,
                training_seed=seed,
            )
            runtime = resolve_device(
                config.runtime.device,
                mixed_precision=neural.mixed_precision,
                pin_memory_policy=config.runtime.pin_memory_policy,
            )
            log_event(_LOGGER, "device_resolved", device=runtime.device.type, **context)
            loader_execution = execution or reused_loader_policy(
                num_workers=config.runtime.num_workers,
                pin_memory=runtime.pin_memory_effective,
            )
            loaders = build_image_loaders(
                train_dataset,
                validation_dataset,
                config=neural,
                runtime=runtime,
                seed=seed,
                execution=loader_execution,
            )
            train_targets = cxr_data.train["target"].to_numpy(dtype=np.int8)
            positive_count, negative_count, pos_weight = training_class_weight(train_targets)
            mlflow.log_params(
                {
                    f"loader_{key}": value if value is not None else "not_applicable"
                    for key, value in loader_execution.provenance().items()
                    if not isinstance(value, dict)
                }
            )
        model_builder = cast(RsnaCxrModelImplementation, get_model(family.family_id))
        with timed_phase(_LOGGER, "model_construction", **context):
            weight_identity = fingerprint_pretrained_weights(str(family.parameters["weights"]))
            model = model_builder.build(family)
            if fingerprint_pretrained_weights(str(family.parameters["weights"])) != weight_identity:
                raise RuntimeError("Pretrained weight file changed during model construction")
            if not isinstance(model, nn.Module):
                raise TypeError("Registered CXR model builder must return torch.nn.Module")
            model.to(runtime.device)
        log_event(_LOGGER, "pretrained_weight_fingerprint_stable", **context)

        epoch_started_at = time.perf_counter()

        def stage_started(stage: str, planned_epochs: int) -> None:
            log_event(
                _LOGGER,
                "training_stage_started",
                stage=stage,
                planned_epochs=planned_epochs,
                **context,
            )

        def epoch_started(stage: str, global_epoch: int, stage_epoch: int) -> None:
            nonlocal epoch_started_at
            epoch_started_at = time.perf_counter()
            log_event(
                _LOGGER,
                "epoch_started",
                stage=stage,
                global_epoch=global_epoch,
                stage_epoch=stage_epoch,
                **context,
            )

        def epoch_completed(record: TrainingEpochRecord) -> None:
            nonlocal epoch_started_at
            now = time.perf_counter()
            log_event(
                _LOGGER,
                "epoch_completed",
                stage=record.stage,
                global_epoch=record.global_epoch,
                stage_epoch=record.stage_epoch,
                training_loss=record.training_loss,
                validation_average_precision=record.validation_metric,
                selected_best=record.selected_best,
                encoder_learning_rate=record.encoder_learning_rate,
                head_learning_rate=record.head_learning_rate,
                no_improvement_count=record.no_improvement_count,
                elapsed_s=now - epoch_started_at,
                **context,
            )
            epoch_started_at = now
            if (
                record.stage == "fine_tune"
                and not record.selected_best
                and record.no_improvement_count >= neural.early_stopping_patience
            ):
                log_event(
                    _LOGGER,
                    "early_stopping_triggered",
                    stage=record.stage,
                    global_epoch=record.global_epoch,
                    patience=neural.early_stopping_patience,
                    **context,
                )

        def epoch_throughput(record: TrainingEpochRecord, throughput: EpochThroughput) -> None:
            log_event(
                _LOGGER,
                "epoch_throughput",
                stage=record.stage,
                global_epoch=record.global_epoch,
                training_elapsed_s=throughput.training_elapsed_s,
                validation_elapsed_s=throughput.validation_elapsed_s,
                training_batches_per_second=throughput.training_batches_per_second,
                validation_batches_per_second=throughput.validation_batches_per_second,
                training_samples_per_second=throughput.training_samples_per_second,
                validation_samples_per_second=throughput.validation_samples_per_second,
                **context,
            )

        operation_progress: dict[tuple[str, str, int], CountProgress] = {}

        def neural_progress(
            operation: str,
            stage: str,
            global_epoch: int,
            completed: int,
            total: int,
        ) -> None:
            key = (operation, stage, global_epoch)
            reporter = operation_progress.get(key)
            if reporter is None and total > 0:
                reporter = CountProgress(
                    _LOGGER,
                    "neural_operation_progress",
                    total=total,
                    unit="batches",
                    count_interval=100,
                    fields={
                        "operation": operation,
                        "stage": stage,
                        "global_epoch": global_epoch,
                        **context,
                    },
                )
                operation_progress[key] = reporter
            if reporter is not None:
                reporter.update(completed)

        with timed_phase(_LOGGER, "cxr_training", **context):
            fit = fit_rsna_cxr_model(
                model,
                loaders,
                config=neural,
                runtime=runtime,
                pos_weight=pos_weight,
                epoch_callback=epoch_completed,
                epoch_started_callback=epoch_started,
                stage_callback=stage_started,
                progress_callback=neural_progress,
                throughput_callback=epoch_throughput,
            )
        log_event(
            _LOGGER,
            "checkpoint_selected",
            selected_epoch=fit.selected_epoch,
            selected_stage=fit.selected_stage,
            validation_average_precision=fit.selected_validation_metric,
            **context,
        )
        model.load_state_dict(fit.selected_state_dict, strict=True)
        model.to(runtime.device)
        model.eval()
        with timed_phase(_LOGGER, "validation_inference", **context):
            validation_progress: CountProgress | None = None

            def report_validation_progress(completed: int, total: int) -> None:
                nonlocal validation_progress
                if validation_progress is None:
                    validation_progress = CountProgress(
                        _LOGGER,
                        "inference_progress",
                        total=total,
                        unit="batches",
                        count_interval=100,
                        fields={"partition": "validation", **context},
                    )
                validation_progress.update(completed)

            final_validation = deterministic_inference(
                model,
                loaders.validation,
                runtime=runtime,
                progress_callback=report_validation_progress,
            )
        if not math.isclose(
            final_validation.average_precision,
            fit.selected_validation_metric,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Restored CXR checkpoint changed validation Average Precision")
        thresholds = {
            "youden_j": youden_j_threshold(
                final_validation.targets, final_validation.probabilities
            ),
            "target_sensitivity": target_sensitivity_threshold(
                final_validation.targets,
                final_validation.probabilities,
                sensitivity=config.evaluation.sensitivity_target,
            ),
        }
        probability_metrics = evaluate_probabilities(
            final_validation.targets,
            final_validation.probabilities,
            calibration_bins=config.evaluation.calibration_bins,
        )
        youden_metrics = evaluate_operating_point(
            final_validation.targets,
            final_validation.probabilities,
            threshold=thresholds["youden_j"],
        )
        sensitivity_metrics = evaluate_operating_point(
            final_validation.targets,
            final_validation.probabilities,
            threshold=thresholds["target_sensitivity"],
        )
        document = metrics_document(
            scope="validation",
            calibration_bins=config.evaluation.calibration_bins,
            sensitivity_target=config.evaluation.sensitivity_target,
            thresholds=thresholds,
            probability=probability_metrics,
            youden=youden_metrics,
            target_sensitivity=sensitivity_metrics,
        )
        document["cxr_training"] = {
            "lineage": {
                "bundle_id": cxr_data.lineage.bundle_id,
                "bundle_manifest_sha256": cxr_data.bundle_manifest_sha256,
                "split_assignment_id": cxr_data.lineage.split_assignment_id,
                "task_id": cxr_data.lineage.task_id,
                "label_policy_version": cxr_data.lineage.label_policy_version,
            },
            "source_authentication": authentication,
            "model_identity": {
                "family_id": family.family_id,
                "modalities": list(family.modalities),
                "encoder_architecture": family.parameters["encoder_name"],
                "image_size": family.parameters["image_size"],
                "embedding_dimension": family.parameters["embedding_dimension"],
                "classifier_output_dimension": 1,
                "pretrained_weight": weight_identity.as_dict(),
            },
            "input_contract": evaluation_transform.contract()["input"],
            "training_transform_contract": train_transform.contract(),
            "evaluation_transform_contract": evaluation_transform.contract(),
            "class_weighting": {
                "policy_version": CLASS_WEIGHT_POLICY_VERSION,
                "labels_used": "train",
                "positive_count": positive_count,
                "negative_count": negative_count,
                "pos_weight": pos_weight,
            },
            "runtime": runtime.provenance(),
            "epoch_history": [
                {
                    ("validation_average_precision" if key == "validation_metric" else key): value
                    for key, value in asdict(record).items()
                }
                for record in fit.history
            ],
            "selection": {
                "selected_epoch": fit.selected_epoch,
                "selected_stage": fit.selected_stage,
                "validation_average_precision": final_validation.average_precision,
            },
            "limitations": [
                "The challenge target is radiology-derived.",
                "Test data were not loaded, decoded, authenticated, or evaluated.",
            ],
        }
        report_directory = (
            config.runtime.report_directory / config.dataset.dataset_id / "runs" / run_id
        )
        if report_directory.exists():
            raise FileExistsError(f"CXR validation report already exists: {report_directory}")
        report_stage = staging_directory(report_directory)
        temporary_model_root = Path(tempfile.mkdtemp(prefix="radfusion-neural-model-"))
        published = None
        report_published = False
        try:
            checkpoint_path = save_neural_checkpoint(
                checkpoint_document(
                    fit.selected_state_dict,
                    selected_epoch=fit.selected_epoch,
                    selected_stage=fit.selected_stage,
                    validation_average_precision=final_validation.average_precision,
                ),
                temporary_model_root / "model.pt",
            )
            loaded_checkpoint = load_neural_checkpoint(checkpoint_path)
            strict_load_checkpoint(model, loaded_checkpoint)
            manifest = _manifest(
                config=config,
                cxr_data=cxr_data,
                source_authentication=authentication,
                commit=commit,
                dirty=dirty,
                lock_hash=lock_hash,
                environment=environment,
                runtime={
                    **runtime.provenance(),
                    "loader_execution": loader_execution.provenance(),
                    "cxr_cache_id": resolved_cache.identity.cache_id,
                },
                weight_identity=weight_identity.as_dict(),
                train_transform=train_transform.contract(),
                evaluation_transform=evaluation_transform.contract(),
                positive_count=positive_count,
                negative_count=negative_count,
                pos_weight=pos_weight,
                fit=fit,
                final_average_precision=final_validation.average_precision,
                thresholds=thresholds,
            )
            published = publish_neural_model_package(
                model_root=config.runtime.model_directory,
                checkpoint_path=checkpoint_path,
                source_config_bytes=config.source_bytes,
                manifest=manifest,
            )
            document["cxr_training"]["package"] = {
                "training_run_id": run_id,
                "model_package_id": published.model_package_id,
                "checkpoint_sha256": published.checkpoint_sha256,
                "checkpoint_byte_size": published.model_path.stat().st_size,
            }
            write_run_reports(
                report_stage,
                model_name=family.family_id,
                targets=final_validation.targets,
                probabilities=final_validation.probabilities,
                document=document,
            )
            validate_report_set(report_stage)
            validate_public_reports(
                report_stage.iterdir(),
                forbidden_source_values={
                    *final_validation.sample_ids,
                    *final_validation.patient_ids,
                    *cxr_data.train["sample_id"].astype(str),
                    *cxr_data.train["patient_id"].astype(str),
                },
            )
            mlflow.log_params(
                {
                    "train_positive_count": positive_count,
                    "train_negative_count": negative_count,
                    "pos_weight": pos_weight,
                    "class_weight_policy": CLASS_WEIGHT_POLICY_VERSION,
                    "source_inventory_file_sha256": authentication["source_inventory_file_sha256"],
                    "source_inventory_arrow_sha256": authentication[
                        "source_inventory_arrow_sha256"
                    ],
                    "source_authentication_policy_version": authentication["policy_version"],
                    "selected_epoch": fit.selected_epoch,
                    "selected_stage": fit.selected_stage,
                    **{f"runtime_{key}": value for key, value in runtime.provenance().items()},
                }
            )
            metrics = mlflow_metrics(
                scope="validation",
                document=document,
                latency_ms=None,
                model_size_mib=published.model_size_mib,
            )
            mlflow.log_metrics(metrics)
            publish_directory(report_stage, report_directory)
            report_published = True
            mlflow.log_params(
                {
                    "checkpoint_sha256": published.checkpoint_sha256,
                    "model_path": published.model_path.as_posix(),
                    "report_directory": report_directory.as_posix(),
                    "threshold_youden_j": thresholds["youden_j"],
                    "threshold_target_sensitivity": thresholds["target_sensitivity"],
                }
            )
            mlflow.set_tags({"package_kind": "model", "package_id": published.model_package_id})
            mlflow.set_tag("run_complete", "true")
        except BaseException:
            if report_published and report_directory.exists():
                shutil.rmtree(report_directory)
            raise
        finally:
            if report_stage.exists():
                shutil.rmtree(report_stage)
            shutil.rmtree(temporary_model_root, ignore_errors=True)
        if published is None:
            raise RuntimeError("CXR training completed without a published package")
        log_event(_LOGGER, "publication_completed", artifact="model_package", **context)
        log_event(_LOGGER, "publication_completed", artifact="validation_report", **context)
    return CxrModelResult(
        run_id=run_id,
        validation_probability=probability_metrics,
        validation_youden_j=youden_metrics,
        validation_target_sensitivity=sensitivity_metrics,
        thresholds=thresholds,
        model_path=published.model_path,
        model_sha256=published.checkpoint_sha256,
        model_package_id=published.model_package_id,
        artifact_directory=report_directory,
        model_size_mib=published.model_size_mib,
    )


def _transform(config: ExperimentConfig, *, training: bool) -> StandardCxrTransform:
    neural = config.neural
    if neural is None:
        raise ValueError("CXR transform requires CXR configuration")
    return StandardCxrTransform(
        training=training,
        policy_version=str(config.preprocessing["cxr_transform_policy"]),
        image_size=int(config.family.parameters["image_size"]),
        rotation_degrees=neural.rotation_degrees,
        translation_fraction=neural.translation_fraction,
        brightness_jitter=neural.brightness_jitter,
        contrast_jitter=neural.contrast_jitter,
    )


def _manifest(
    *,
    config: ExperimentConfig,
    cxr_data: CxrRunData,
    source_authentication: dict[str, object],
    commit: str,
    dirty: bool,
    lock_hash: str,
    environment: dict[str, str],
    runtime: dict[str, Any],
    weight_identity: dict[str, object],
    train_transform: dict[str, Any],
    evaluation_transform: dict[str, Any],
    positive_count: int,
    negative_count: int,
    pos_weight: float,
    fit: SelectedTrainingResult,
    final_average_precision: float,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    neural = config.neural
    if neural is None:
        raise ValueError("CXR manifest requires CXR configuration")
    return {
        "family_id": config.family.family_id,
        "modalities": list(config.family.modalities),
        "task_id": cxr_data.lineage.task_id,
        "positive_class": 1,
        "bundle_id": cxr_data.lineage.bundle_id,
        "bundle_manifest_sha256": cxr_data.bundle_manifest_sha256,
        "split_assignment_id": cxr_data.lineage.split_assignment_id,
        "label_policy_version": cxr_data.lineage.label_policy_version,
        "config_source_sha256": config.config_source_sha256,
        "config_semantic_sha256": config.config_semantic_sha256,
        "source_provenance": {
            "git_commit": commit,
            "git_dirty": dirty,
            "dependency_lock_sha256": lock_hash,
            "python_version": environment["environment_python_version"],
            "torch_version": runtime["torch_version"],
            "torchvision_version": runtime["torchvision_version"],
            "torchxrayvision_version": runtime["torchxrayvision_version"],
        },
        "model_identity": {
            "family_id": config.family.family_id,
            "modalities": list(config.family.modalities),
            "encoder_architecture": config.family.parameters["encoder_name"],
            "image_size": config.family.parameters["image_size"],
            "embedding_dimension": config.family.parameters["embedding_dimension"],
            "classifier_output_dimension": 1,
            "pretrained_weight": weight_identity,
        },
        "input_contract": evaluation_transform["input"],
        "training_transform_contract": train_transform,
        "evaluation_transform_contract": evaluation_transform,
        "training_policy": {
            "seed": require_runtime_seed(config),
            "permitted_partitions": ["train", "validation"],
            "class_weight": {
                "policy_version": CLASS_WEIGHT_POLICY_VERSION,
                "labels_used": "train",
                "positive_count": positive_count,
                "negative_count": negative_count,
                "pos_weight": pos_weight,
            },
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
        },
        "selection": {
            "selected_epoch": fit.selected_epoch,
            "selected_stage": fit.selected_stage,
            "validation_average_precision": final_average_precision,
        },
        "thresholds": thresholds,
        "threshold_contract": threshold_contract(
            sensitivity_target=config.evaluation.sensitivity_target
        ),
        "source_authentication": source_authentication,
        "runtime_provenance": runtime,
    }
