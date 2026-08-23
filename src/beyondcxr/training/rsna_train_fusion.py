"""Train and validate one configured RSNA CXR-metadata fusion family."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
from sklearn.pipeline import Pipeline

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.hashing import sha256_file
from beyondcxr.data.rsna_cxr_cache import ValidatedCxrCache
from beyondcxr.data.rsna_metadata_preprocess import (
    SOURCE_FEATURES,
    build_rsna_preprocessor,
    fitted_rsna_preprocessor_contract,
    load_preprocessor,
    save_preprocessor,
    transform_rsna_metadata,
    validate_fitted_rsna_preprocessor,
)
from beyondcxr.evaluation.metrics import (
    OperatingPointMetrics,
    ProbabilityMetrics,
    evaluate_operating_point,
    evaluate_probabilities,
    target_sensitivity_threshold,
    youden_j_threshold,
)
from beyondcxr.models.fusion_concat import (
    RsnaConcatFusionModel,
    RsnaCxrMetadataConcatModel,
    fusion_architecture_contract,
    fusion_structured_input_conversion_contract,
    initialize_fusion_encoder,
)
from beyondcxr.training.config import (
    ExperimentConfig,
    require_runtime_seed,
)
from beyondcxr.training.device import resolve_device
from beyondcxr.training.execution import LoaderExecutionPolicy, reused_loader_policy
from beyondcxr.training.neural import (
    CLASS_WEIGHT_POLICY_VERSION,
    EpochThroughput,
    TrainingEpochRecord,
    build_image_loaders,
    deterministic_inference,
    fit_rsna_two_stage_binary_model,
    seed_neural_runtime,
    training_class_weight,
)
from beyondcxr.training.rsna_datasets import (
    FusionRunData,
    RsnaCachedFusionDataset,
    RsnaDataset,
    expected_rsna_cxr_cache_identity,
    prepare_rsna_cxr_cache,
)
from beyondcxr.training.rsna_fusion_source import resolve_source_cxr_package, source_encoder_state
from beyondcxr.training.rsna_registry import get_dataset, get_model
from beyondcxr.training.rsna_train_metadata import (
    metrics_document,
    mlflow_metrics,
    validate_report_set,
    write_run_reports,
)
from beyondcxr.utils.mlflow_utils import (
    DEFAULT_TRACKING_URI,
    configure_mlflow,
    environment_provenance,
    git_revision,
    log_source_config,
    serialize_modalities,
    tracked_run,
    uv_lock_sha256,
)
from beyondcxr.utils.operational_logging import get_operational_logger, log_event, timed_phase
from beyondcxr.utils.privacy import validate_public_reports
from beyondcxr.utils.publication import publish_directory, staging_directory
from beyondcxr.utils.rsna_model_publication import threshold_contract
from beyondcxr.utils.rsna_neural_publication import (
    STRUCTURED_PREPROCESSOR_FILENAME,
    checkpoint_document,
    publish_neural_model_package,
    save_neural_checkpoint,
    strict_load_checkpoint,
)

_LOGGER = get_operational_logger(__name__)


@dataclass(frozen=True)
class FusionModelResult:
    """Published outputs from one completed fusion training run."""

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


def train_fusion_experiment(
    config: ExperimentConfig,
    *,
    source_cxr_package_id: str,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    cache: ValidatedCxrCache | None = None,
    execution: LoaderExecutionPolicy | None = None,
) -> FusionModelResult:
    """Train fusion from one explicit verified same-seed CXR package."""
    if config.family.family_id != "cxr_metadata_concat" or config.neural is None:
        raise ValueError("Fusion training requires a complete fusion experiment configuration")
    if config.evaluation is None:
        raise ValueError("RSNA fusion training requires evaluation policy")
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
    source = resolve_source_cxr_package(source_cxr_package_id, config)
    source_package_id = source.source_package_id
    tags = {
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
        "source_package_id": source_package_id,
        "git_commit": commit,
        "dependency_lock_sha256": lock_hash,
        "config_source_sha256": config.config_source_sha256,
        "config_semantic_sha256": config.config_semantic_sha256,
        "run_complete": "false",
    }
    with tracked_run(
        run_name=family.family_id,
        tags=tags,
        parameters={
            "training_seed": seed,
            "source_package_id": source_package_id,
            **dict(family.parameters),
            **environment,
        },
    ) as run_id:
        log_source_config(config)
        context = {"run_id": run_id, "family_id": family.family_id}
        dataset_adapter = get_dataset(config.dataset.dataset_id)
        with timed_phase(_LOGGER, "dataset_loading", **context):
            data = dataset_adapter.load_fusion_train_validation(config)
        _validate_source_dataset_lineage(data, source.manifest)
        if data.lineage.split_assignment_id != config.dataset.split_assignment_id:
            raise ValueError("Loaded split assignment differs from the configuration")
        with timed_phase(_LOGGER, "fusion_runtime_preparation", **context):
            seed_neural_runtime(seed)
            preprocessor, contract, train_matrix, validation_matrix = _fit_structured(data)
            train_transform = _transform(config, training=True)
            evaluation_transform = _transform(config, training=False)
            resolved_cache = cache or prepare_rsna_cxr_cache(
                cast(RsnaDataset, dataset_adapter), config, evaluation_transform
            )
            expected_cache_identity = expected_rsna_cxr_cache_identity(
                lineage=data.lineage,
                bundle_manifest_sha256=data.bundle_manifest_sha256,
                source_inventory=data.source_inventory,
                transform=evaluation_transform,
            )
            source_authentication = resolved_cache.source_authentication.as_dict()
            mlflow.log_param(
                "source_authentication_policy", source_authentication["policy_version"]
            )
            train_dataset = RsnaCachedFusionDataset(
                data.train,
                train_matrix,
                structured_sample_ids=tuple(data.train["sample_id"].astype(str)),
                cache=resolved_cache,
                expected_cache_identity=expected_cache_identity,
                partition="train",
                transform=train_transform,
                training_seed=seed,
            )
            validation_dataset = RsnaCachedFusionDataset(
                data.validation,
                validation_matrix,
                structured_sample_ids=tuple(data.validation["sample_id"].astype(str)),
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
            mlflow.log_params(
                {
                    f"loader_{key}": value if value is not None else "not_applicable"
                    for key, value in loader_execution.provenance().items()
                    if not isinstance(value, dict)
                }
            )
            positive_count, negative_count, pos_weight = training_class_weight(
                data.train["target"].to_numpy(dtype=np.int8)
            )
        builder = cast(RsnaCxrMetadataConcatModel, get_model(family.family_id))
        model = builder.build(
            family,
            structured_dimension=int(contract["transformed_dimension"]),
            weights=None,
        )
        if not isinstance(model, RsnaConcatFusionModel):
            raise TypeError("Registered fusion builder returned an invalid model")
        initialize_fusion_encoder(model, source_encoder_state(source))
        model.to(runtime.device)

        def epoch_completed(record: TrainingEpochRecord) -> None:
            log_event(
                _LOGGER,
                "epoch_completed",
                stage=record.stage,
                global_epoch=record.global_epoch,
                stage_epoch=record.stage_epoch,
                training_loss=record.training_loss,
                validation_average_precision=record.validation_metric,
                selected_best=record.selected_best,
                no_improvement_count=record.no_improvement_count,
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

        with timed_phase(_LOGGER, "fusion_training", **context):
            fit = fit_rsna_two_stage_binary_model(
                model,
                loaders.train,
                loaders.validation,
                input_keys=("image", "structured"),
                config=neural,
                runtime=runtime,
                pos_weight=pos_weight,
                epoch_callback=epoch_completed,
                throughput_callback=epoch_throughput,
            )
        loaded = checkpoint_document(
            fit.selected_state_dict,
            selected_epoch=fit.selected_epoch,
            selected_stage=fit.selected_stage,
            validation_average_precision=fit.selected_validation_metric,
        )
        strict_load_checkpoint(model, loaded)
        final_validation = deterministic_inference(
            model,
            loaders.validation,
            runtime=runtime,
            input_keys=("image", "structured"),
        )
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
        probability = evaluate_probabilities(
            final_validation.targets,
            final_validation.probabilities,
            calibration_bins=config.evaluation.calibration_bins,
        )
        youden = evaluate_operating_point(
            final_validation.targets,
            final_validation.probabilities,
            threshold=thresholds["youden_j"],
        )
        sensitivity = evaluate_operating_point(
            final_validation.targets,
            final_validation.probabilities,
            threshold=thresholds["target_sensitivity"],
        )
        report = metrics_document(
            scope="validation",
            calibration_bins=config.evaluation.calibration_bins,
            sensitivity_target=config.evaluation.sensitivity_target,
            thresholds=thresholds,
            probability=probability,
            youden=youden,
            target_sensitivity=sensitivity,
        )
        report_directory = (
            config.runtime.report_directory / config.dataset.dataset_id / "runs" / run_id
        )
        if report_directory.exists():
            raise FileExistsError(f"Fusion validation report already exists: {report_directory}")
        report_stage = staging_directory(report_directory)
        temporary = Path(tempfile.mkdtemp(prefix="beyondcxr-fusion-package-"))
        published = None
        try:
            checkpoint_path = save_neural_checkpoint(loaded, temporary / "model.pt")
            preprocessor_path = save_preprocessor(
                preprocessor, temporary / "structured_preprocessor.skops"
            )
            manifest = _manifest(
                config=config,
                data=data,
                source_authentication=source_authentication,
                commit=commit,
                dirty=dirty,
                lock_hash=lock_hash,
                environment=environment,
                runtime={
                    **runtime.provenance(),
                    "loader_execution": loader_execution.provenance(),
                    "cxr_cache_id": resolved_cache.identity.cache_id,
                },
                source_package_id=source_package_id,
                source_pretrained_weight=dict(
                    source.manifest["model_identity"]["pretrained_weight"]
                ),
                structured_contract=contract,
                preprocessor_sha256=sha256_file(preprocessor_path),
                train_transform=train_transform.contract(),
                evaluation_transform=evaluation_transform.contract(),
                positive_count=positive_count,
                negative_count=negative_count,
                pos_weight=pos_weight,
                fit=fit,
                thresholds=thresholds,
            )
            published = publish_neural_model_package(
                model_root=config.runtime.model_directory,
                checkpoint_path=checkpoint_path,
                source_config_bytes=config.source_bytes,
                manifest=manifest,
                structured_preprocessor_path=preprocessor_path,
            )
            load_validated_rsna_fusion_preprocessor(published.package_directory, manifest)
            write_run_reports(
                report_stage,
                model_name=family.family_id,
                targets=final_validation.targets,
                probabilities=final_validation.probabilities,
                document=report,
            )
            validate_report_set(report_stage)
            validate_public_reports(
                report_stage.iterdir(),
                forbidden_source_values={
                    *final_validation.sample_ids,
                    *final_validation.patient_ids,
                },
            )
            publish_directory(report_stage, report_directory)
            mlflow.log_metrics(
                mlflow_metrics(
                    scope="validation",
                    document=report,
                    latency_ms=None,
                    model_size_mib=published.model_size_mib,
                )
            )
            mlflow.log_params(
                {
                    "selected_epoch": fit.selected_epoch,
                    "selected_stage": fit.selected_stage,
                    "structured_dimension": contract["transformed_dimension"],
                    "structured_preprocessor_sha256": sha256_file(preprocessor_path),
                }
            )
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
            if report_directory.exists():
                shutil.rmtree(report_directory)
            raise
        finally:
            shutil.rmtree(temporary)
            if report_stage.exists():
                shutil.rmtree(report_stage)
        log_event(_LOGGER, "publication_completed", artifact="fusion_package", **context)
    return FusionModelResult(
        run_id=run_id,
        validation_probability=probability,
        validation_youden_j=youden,
        validation_target_sensitivity=sensitivity,
        thresholds=thresholds,
        model_path=published.model_path,
        model_sha256=published.checkpoint_sha256,
        model_package_id=published.model_package_id,
        artifact_directory=report_directory,
        model_size_mib=published.model_size_mib,
    )


def _fit_structured(
    data: FusionRunData,
) -> tuple[Pipeline, dict[str, Any], np.ndarray, np.ndarray]:
    preprocessor = build_rsna_preprocessor()
    preprocessor.fit(data.train.loc[:, SOURCE_FEATURES])
    fitted = validate_fitted_rsna_preprocessor(preprocessor)
    contract = fitted_rsna_preprocessor_contract(fitted)
    train = transform_rsna_metadata(fitted, data.train.loc[:, SOURCE_FEATURES])
    validation = transform_rsna_metadata(fitted, data.validation.loc[:, SOURCE_FEATURES])
    return (
        fitted,
        contract,
        np.ascontiguousarray(train.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(validation.to_numpy(dtype=np.float64)),
    )


def load_validated_rsna_fusion_preprocessor(
    package_directory: str | Path,
    manifest: Mapping[str, Any],
) -> Pipeline:
    """Load the package-bound fitted RSNA structured preprocessor."""
    if manifest.get("family_id") != "cxr_metadata_concat" or manifest.get("modalities") != [
        "cxr",
        "metadata",
    ]:
        raise ValueError("RSNA structured preprocessing requires a fusion package")
    path = Path(package_directory) / STRUCTURED_PREPROCESSOR_FILENAME
    if path.is_symlink() or not path.is_file():
        raise ValueError("Fusion structured preprocessor must be a regular non-symlink file")
    if sha256_file(path) != manifest.get("structured_preprocessor_sha256"):
        raise ValueError("Fusion structured preprocessor SHA-256 mismatch")
    preprocessor = load_preprocessor(path)
    if fitted_rsna_preprocessor_contract(preprocessor) != manifest.get(
        "structured_preprocessor_contract"
    ):
        raise ValueError("Fusion structured preprocessor contract mismatch")
    return preprocessor


def _validate_source_dataset_lineage(data: FusionRunData, manifest: Mapping[str, Any]) -> None:
    if (
        data.lineage.bundle_id != manifest["bundle_id"]
        or data.bundle_manifest_sha256 != manifest["bundle_manifest_sha256"]
        or data.lineage.split_assignment_id != manifest["split_assignment_id"]
        or data.lineage.task_id != manifest["task_id"]
        or data.lineage.label_policy_version != manifest["label_policy_version"]
    ):
        raise ValueError("Fusion dataset lineage differs from the source CXR package")


def _transform(config: ExperimentConfig, *, training: bool) -> StandardCxrTransform:
    neural = config.neural
    if neural is None:
        raise ValueError("Fusion transform requires CXR configuration")
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
    data: FusionRunData,
    source_authentication: dict[str, object],
    commit: str,
    dirty: bool,
    lock_hash: str,
    environment: dict[str, str],
    runtime: dict[str, Any],
    source_package_id: str,
    source_pretrained_weight: dict[str, Any],
    structured_contract: dict[str, Any],
    preprocessor_sha256: str,
    train_transform: dict[str, Any],
    evaluation_transform: dict[str, Any],
    positive_count: int,
    negative_count: int,
    pos_weight: float,
    fit,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    neural = config.neural
    if neural is None:
        raise ValueError("Fusion manifest requires CXR configuration")
    return {
        "family_id": config.family.family_id,
        "modalities": list(config.family.modalities),
        "task_id": data.lineage.task_id,
        "positive_class": 1,
        "bundle_id": data.lineage.bundle_id,
        "bundle_manifest_sha256": data.bundle_manifest_sha256,
        "split_assignment_id": data.lineage.split_assignment_id,
        "label_policy_version": data.lineage.label_policy_version,
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
            "pretrained_weight": source_pretrained_weight,
        },
        "source_package_id": source_package_id,
        "structured_preprocessor_sha256": preprocessor_sha256,
        "structured_preprocessor_contract": structured_contract,
        "structured_input_conversion": fusion_structured_input_conversion_contract(),
        "fusion_architecture": fusion_architecture_contract(
            config.family,
            structured_input_dimension=structured_contract["transformed_dimension"],
        ),
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
            "validation_average_precision": fit.selected_validation_metric,
        },
        "thresholds": thresholds,
        "threshold_contract": threshold_contract(
            sensitivity_target=config.evaluation.sensitivity_target
        ),
        "source_authentication": source_authentication,
        "runtime_provenance": runtime,
    }
