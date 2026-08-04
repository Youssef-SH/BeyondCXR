"""Train and validate one configured RSNA image experiment."""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
from torch import nn

from radfusion.data.cxr_cache import ValidatedCxrCache
from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.evaluation.metrics import (
    OperatingPointMetrics,
    ProbabilityMetrics,
    evaluate_operating_point,
    evaluate_probabilities,
    target_sensitivity_threshold,
    youden_j_threshold,
)
from radfusion.models.cxr_baseline import fingerprint_pretrained_weights
from radfusion.training.config import ExperimentConfig, image_semantic_config_sha256
from radfusion.training.datasets import (
    ImageRunData,
    RsnaCachedImageDataset,
    RsnaDataset,
    expected_rsna_cxr_cache_identity,
    prepare_rsna_cxr_cache,
)
from radfusion.training.device import resolve_device
from radfusion.training.execution import LoaderExecutionPolicy, configured_loader_policy
from radfusion.training.interfaces import ImageModelImplementation
from radfusion.training.neural import (
    CLASS_WEIGHT_POLICY_VERSION,
    EpochRecord,
    EpochThroughput,
    NeuralFitResult,
    build_image_loaders,
    deterministic_inference,
    fit_image_model,
    seed_neural_runtime,
    training_class_weight,
)
from radfusion.training.registry import get_dataset, get_model
from radfusion.training.train_tabular import (
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
    tracked_run,
    uv_lock_sha256,
)
from radfusion.utils.model_publication import threshold_contract
from radfusion.utils.neural_publication import (
    checkpoint_document,
    load_neural_checkpoint,
    publish_neural_model_run,
    save_neural_checkpoint,
    strict_load_checkpoint,
)
from radfusion.utils.operational_logging import (
    CountProgress,
    get_operational_logger,
    log_event,
    timed_phase,
)
from radfusion.utils.privacy import validate_public_reports
from radfusion.utils.publication import publish_directory, staging_directory

NEURAL_METRICS_POLICY_VERSION = "binary-probability-and-frozen-operating-points-v1"
NEURAL_THRESHOLD_POLICY_VERSION = "validation-frozen-thresholds-v1"
_LOGGER = get_operational_logger(__name__)


@dataclass(frozen=True)
class ImageModelResult:
    """Published outputs from one completed image training run."""

    model_name: str
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


def train_image_experiment(
    config: ExperimentConfig,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    cache: ValidatedCxrCache | None = None,
    execution: LoaderExecutionPolicy | None = None,
) -> ImageModelResult:
    """Train on image train/validation partitions and publish one selected package."""
    if config.model.modality != "image" or config.image is None:
        raise ValueError("Image training requires a complete image experiment configuration")
    configure_mlflow(
        experiment_name=config.mlflow.experiment_name,
        tracking_uri=tracking_uri,
    )
    commit, dirty = git_revision()
    lock_hash = uv_lock_sha256()
    environment = environment_provenance()
    base_tags = {
        "run_kind": "training",
        "evaluation_scope": "validation",
        "experiment_name": config.name,
        "dataset": config.dataset.registry_key,
        "dataset_bundle_id": config.dataset.bundle_id,
        "task": config.dataset.task_id,
        "modality": "image",
        "model": config.model.registry_key,
        "seed": str(config.training.seed),
        "git_commit": commit,
        "git_dirty": str(dirty).lower(),
        "dependency_lock_sha256": lock_hash,
        "source_config_sha256": config.source_sha256,
        "semantic_config_sha256": image_semantic_config_sha256(config),
        "run_complete": "false",
    }
    initial_parameters = {
        "training_seed": config.training.seed,
        "requested_device": config.image.device,
        "requested_mixed_precision": config.image.mixed_precision,
        "requested_pin_memory_policy": config.image.pin_memory_policy,
        "sensitivity_target": config.evaluation.sensitivity_target,
        "calibration_bins": config.evaluation.calibration_bins,
        **dict(config.model.parameters),
        **environment,
    }
    with tracked_run(
        run_name=config.name,
        tags=base_tags,
        parameters=initial_parameters,
    ) as run_id:
        log_source_config(config)
        context = {"run_id": run_id, "model": config.model.registry_key}
        dataset_adapter = get_dataset(config.dataset.registry_key)
        with timed_phase(_LOGGER, "dataset_loading", **context):
            image_data = dataset_adapter.load_image_train_validation(config.dataset)
        mlflow.set_tags(
            {
                "split_assignment_id": image_data.lineage.split_assignment_id,
                "label_policy_version": image_data.lineage.label_policy_version,
            }
        )
        mlflow.log_param("bundle_manifest_sha256", image_data.bundle_manifest_sha256)
        with timed_phase(_LOGGER, "image_runtime_preparation", **context):
            seed_neural_runtime(config.training.seed)
            train_transform = _transform(config, training=True)
            evaluation_transform = _transform(config, training=False)
            resolved_cache = cache or prepare_rsna_cxr_cache(
                cast(RsnaDataset, dataset_adapter), config.dataset, evaluation_transform
            )
            expected_cache_identity = expected_rsna_cxr_cache_identity(
                lineage=image_data.lineage,
                bundle_manifest_sha256=image_data.bundle_manifest_sha256,
                source_inventory=image_data.source_inventory,
                transform=evaluation_transform,
            )
            authentication = resolved_cache.source_authentication.as_dict()
            mlflow.set_tag("source_authentication_policy", authentication["policy_version"])
            train_dataset = RsnaCachedImageDataset(
                image_data.train,
                cache=resolved_cache,
                expected_cache_identity=expected_cache_identity,
                partition="train",
                transform=train_transform,
                training_seed=config.training.seed,
            )
            validation_dataset = RsnaCachedImageDataset(
                image_data.validation,
                cache=resolved_cache,
                expected_cache_identity=expected_cache_identity,
                partition="validation",
                transform=evaluation_transform,
                training_seed=config.training.seed,
            )
            runtime = resolve_device(
                config.image.device,
                mixed_precision=config.image.mixed_precision,
                pin_memory_policy=config.image.pin_memory_policy,
            )
            log_event(_LOGGER, "device_resolved", device=runtime.device.type, **context)
            loader_execution = execution or configured_loader_policy(
                num_workers=config.image.num_workers,
                pin_memory=runtime.pin_memory_effective,
            )
            loaders = build_image_loaders(
                train_dataset,
                validation_dataset,
                config=config.image,
                runtime=runtime,
                seed=config.training.seed,
                execution=loader_execution,
            )
            train_targets = image_data.train["target"].to_numpy(dtype=np.int8)
            positive_count, negative_count, pos_weight = training_class_weight(train_targets)
            mlflow.log_params(
                {
                    f"loader_{key}": value if value is not None else "not_applicable"
                    for key, value in loader_execution.provenance().items()
                    if not isinstance(value, dict)
                }
            )
        model_builder = cast(ImageModelImplementation, get_model(config.model.registry_key))
        with timed_phase(_LOGGER, "model_construction", **context):
            weight_identity = fingerprint_pretrained_weights(
                str(config.model.parameters["weights"])
            )
            model = model_builder.build(config.model)
            if (
                fingerprint_pretrained_weights(str(config.model.parameters["weights"]))
                != weight_identity
            ):
                raise RuntimeError("Pretrained weight file changed during model construction")
            if not isinstance(model, nn.Module):
                raise TypeError("Registered image model builder must return torch.nn.Module")
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

        def epoch_completed(record: EpochRecord) -> None:
            nonlocal epoch_started_at
            now = time.perf_counter()
            log_event(
                _LOGGER,
                "epoch_completed",
                stage=record.stage,
                global_epoch=record.global_epoch,
                stage_epoch=record.stage_epoch,
                training_loss=record.training_loss,
                validation_average_precision=record.validation_average_precision,
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
                and record.no_improvement_count >= config.image.early_stopping_patience
            ):
                log_event(
                    _LOGGER,
                    "early_stopping_triggered",
                    stage=record.stage,
                    global_epoch=record.global_epoch,
                    patience=config.image.early_stopping_patience,
                    **context,
                )

        def epoch_throughput(record: EpochRecord, throughput: EpochThroughput) -> None:
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

        with timed_phase(_LOGGER, "image_training", **context):
            fit = fit_image_model(
                model,
                loaders,
                config=config.image,
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
            validation_average_precision=fit.selected_validation_average_precision,
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
            fit.selected_validation_average_precision,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Restored image checkpoint changed validation Average Precision")
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
        document["image_run"] = {
            "lineage": {
                "bundle_id": image_data.lineage.bundle_id,
                "bundle_manifest_sha256": image_data.bundle_manifest_sha256,
                "split_assignment_id": image_data.lineage.split_assignment_id,
                "task": image_data.lineage.task_id,
                "label_policy_version": image_data.lineage.label_policy_version,
            },
            "source_authentication": authentication,
            "model_identity": {
                "registry_key": config.model.registry_key,
                "modality": config.model.modality,
                "encoder_architecture": config.model.parameters["encoder_name"],
                "image_size": config.model.parameters["image_size"],
                "embedding_dimension": config.model.parameters["embedding_dimension"],
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
            "epoch_history": [record.as_dict() for record in fit.history],
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
            config.training.report_directory / config.dataset.registry_key / "runs" / run_id
        )
        if report_directory.exists():
            raise FileExistsError(f"Image validation report already exists: {report_directory}")
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
                image_data=image_data,
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
            published = publish_neural_model_run(
                model_root=config.training.model_directory,
                mlflow_run_id=run_id,
                checkpoint_path=checkpoint_path,
                source_config_bytes=config.source_bytes,
                manifest=manifest,
            )
            document["image_run"]["package"] = {
                "training_run_id": run_id,
                "model_package_id": published.model_package_id,
                "checkpoint_sha256": published.checkpoint_sha256,
                "checkpoint_byte_size": published.model_path.stat().st_size,
            }
            write_run_reports(
                report_stage,
                model_name=config.model.registry_key,
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
                    *image_data.train["sample_id"].astype(str),
                    *image_data.train["patient_id"].astype(str),
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
            mlflow.set_tags(
                {
                    "local_model_path": published.model_path.as_posix(),
                    "local_model_sha256": published.checkpoint_sha256,
                    "checkpoint_sha256": published.checkpoint_sha256,
                    "model_package_id": published.model_package_id,
                    "report_directory": report_directory.as_posix(),
                    "threshold_youden_j": str(thresholds["youden_j"]),
                    "threshold_target_sensitivity": str(thresholds["target_sensitivity"]),
                }
            )
            mlflow.set_tag("run_complete", "true")
        except BaseException:
            if published is not None and published.run_directory.exists():
                shutil.rmtree(published.run_directory)
            if report_published and report_directory.exists():
                shutil.rmtree(report_directory)
            raise
        finally:
            if report_stage.exists():
                shutil.rmtree(report_stage)
            shutil.rmtree(temporary_model_root, ignore_errors=True)
        if published is None:
            raise RuntimeError("Image training completed without a published package")
        log_event(_LOGGER, "publication_completed", artifact="model_package", **context)
        log_event(_LOGGER, "publication_completed", artifact="validation_report", **context)
    return ImageModelResult(
        model_name=config.model.registry_key,
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
    image = config.image
    if image is None:
        raise ValueError("Image transform requires image configuration")
    return StandardCxrTransform(
        training=training,
        image_size=int(config.model.parameters["image_size"]),
        rotation_degrees=image.rotation_degrees,
        translation_fraction=image.translation_fraction,
        brightness_jitter=image.brightness_jitter,
        contrast_jitter=image.contrast_jitter,
    )


def _manifest(
    *,
    config: ExperimentConfig,
    image_data: ImageRunData,
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
    fit: NeuralFitResult,
    final_average_precision: float,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    image = config.image
    if image is None:
        raise ValueError("Image manifest requires image configuration")
    return {
        "modality": "image",
        "model": config.model.registry_key,
        "task": image_data.lineage.task_id,
        "positive_class": 1,
        "bundle_id": image_data.lineage.bundle_id,
        "bundle_manifest_sha256": image_data.bundle_manifest_sha256,
        "split_assignment_id": image_data.lineage.split_assignment_id,
        "label_policy_version": image_data.lineage.label_policy_version,
        "source_config_sha256": config.source_sha256,
        "semantic_config_sha256": image_semantic_config_sha256(config),
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
            "registry_key": config.model.registry_key,
            "modality": config.model.modality,
            "encoder_architecture": config.model.parameters["encoder_name"],
            "image_size": config.model.parameters["image_size"],
            "embedding_dimension": config.model.parameters["embedding_dimension"],
            "classifier_output_dimension": 1,
            "pretrained_weight": weight_identity,
        },
        "input_contract": evaluation_transform["input"],
        "training_transform_contract": train_transform,
        "evaluation_transform_contract": evaluation_transform,
        "training_policy": {
            "seed": config.training.seed,
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
        "metrics_policy": {
            "version": NEURAL_METRICS_POLICY_VERSION,
            "calibration_bins": config.evaluation.calibration_bins,
            "threshold_policy_version": NEURAL_THRESHOLD_POLICY_VERSION,
            "sensitivity_target": config.evaluation.sensitivity_target,
        },
        "source_authentication": source_authentication,
        "runtime_provenance": runtime,
    }
