"""Train and validate one configured RSNA image-metadata fusion experiment."""

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

from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.hashing import sha256_file
from radfusion.data.tabular_preprocess import (
    SOURCE_FEATURES,
    build_rsna_preprocessor,
    fitted_rsna_preprocessor_contract,
    load_preprocessor,
    save_preprocessor,
    transform_rsna_metadata,
    validate_fitted_rsna_preprocessor,
)
from radfusion.evaluation.metrics import (
    OperatingPointMetrics,
    ProbabilityMetrics,
    evaluate_operating_point,
    evaluate_probabilities,
    target_sensitivity_threshold,
    youden_j_threshold,
)
from radfusion.models.fusion_concat import (
    FusionConcatModel,
    RsnaConcatFusionModel,
    initialize_fusion_encoder,
)
from radfusion.training.config import (
    ExperimentConfig,
    fusion_architecture_contract,
    fusion_semantic_config_sha256,
    fusion_structured_input_conversion_contract,
)
from radfusion.training.datasets import FusionRunData, RsnaFusionDataset
from radfusion.training.device import resolve_device
from radfusion.training.fusion_source import (
    SourceCxrLineage,
    resolve_source_cxr_training_run,
    source_encoder_state,
)
from radfusion.training.neural import (
    CLASS_WEIGHT_POLICY_VERSION,
    build_image_loaders,
    deterministic_inference,
    fit_two_stage_binary_model,
    seed_neural_runtime,
    training_class_weight,
)
from radfusion.training.registry import get_dataset, get_model
from radfusion.training.train_image import (
    NEURAL_METRICS_POLICY_VERSION,
    NEURAL_THRESHOLD_POLICY_VERSION,
)
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
    STRUCTURED_PREPROCESSOR_FILENAME,
    checkpoint_document,
    publish_neural_model_run,
    save_neural_checkpoint,
    strict_load_checkpoint,
)
from radfusion.utils.operational_logging import get_operational_logger, log_event, timed_phase
from radfusion.utils.privacy import validate_public_reports
from radfusion.utils.publication import publish_directory, staging_directory

_LOGGER = get_operational_logger(__name__)


@dataclass(frozen=True)
class FusionModelResult:
    """Published outputs from one completed fusion training run."""

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


def train_fusion_experiment(
    config: ExperimentConfig,
    *,
    source_training_run_id: str,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> FusionModelResult:
    """Train fusion from one explicit verified same-seed CXR package."""
    if config.model.modality != "fusion" or config.image is None:
        raise ValueError("Fusion training requires a complete fusion experiment configuration")
    client = configure_mlflow(
        experiment_name=config.mlflow.experiment_name,
        tracking_uri=tracking_uri,
    )
    commit, dirty = git_revision()
    lock_hash = uv_lock_sha256()
    environment = environment_provenance()
    source = resolve_source_cxr_training_run(
        client,
        source_training_run_id,
        config,
        current_git_commit=commit,
        current_git_dirty=dirty,
        current_dependency_lock_sha256=lock_hash,
    )
    source_lineage = source.lineage
    tags = {
        "run_kind": "training",
        "evaluation_scope": "validation",
        "experiment_name": config.name,
        "dataset": config.dataset.registry_key,
        "dataset_bundle_id": config.dataset.bundle_id,
        "task": config.dataset.task_id,
        "modality": "fusion",
        "model": config.model.registry_key,
        "seed": str(config.training.seed),
        "source_cxr_training_run_id": source_lineage.training_run_id,
        "source_cxr_model_package_id": source_lineage.model_package_id,
        "source_cxr_checkpoint_sha256": source_lineage.checkpoint_sha256,
        "git_commit": commit,
        "git_dirty": str(dirty).lower(),
        "dependency_lock_sha256": lock_hash,
        "source_config_sha256": config.source_sha256,
        "semantic_config_sha256": fusion_semantic_config_sha256(config),
        "run_complete": "false",
    }
    with tracked_run(
        run_name=config.name,
        tags=tags,
        parameters={
            "training_seed": config.training.seed,
            "source_cxr_training_run_id": source_lineage.training_run_id,
            "source_cxr_model_package_id": source_lineage.model_package_id,
            "source_cxr_checkpoint_sha256": source_lineage.checkpoint_sha256,
            "source_cxr_semantic_config_sha256": source_lineage.semantic_config_sha256,
            "source_cxr_git_commit": source_lineage.git_commit,
            "source_cxr_dependency_lock_sha256": source_lineage.dependency_lock_sha256,
            **dict(config.model.parameters),
            **environment,
        },
    ) as run_id:
        log_source_config(config)
        context = {"run_id": run_id, "model": config.model.registry_key}
        dataset_adapter = get_dataset(config.dataset.registry_key)
        with timed_phase(_LOGGER, "dataset_loading", **context):
            data = dataset_adapter.load_fusion_train_validation(config.dataset)
        _validate_source_dataset_lineage(data, source.manifest)
        mlflow.set_tags(
            {
                "split_assignment_id": data.lineage.split_assignment_id,
                "label_policy_version": data.lineage.label_policy_version,
                "source_authentication_success": "true",
                "source_authentication_policy": data.authentication.policy_version,
            }
        )
        mlflow.log_param("bundle_manifest_sha256", data.bundle_manifest_sha256)
        with timed_phase(_LOGGER, "fusion_runtime_preparation", **context):
            seed_neural_runtime(config.training.seed)
            preprocessor, contract, train_matrix, validation_matrix = _fit_structured(data)
            train_transform = _transform(config, training=True)
            evaluation_transform = _transform(config, training=False)
            train_dataset = RsnaFusionDataset(
                data.train,
                train_matrix,
                structured_sample_ids=tuple(data.train["sample_id"].astype(str)),
                dataset_root=_required_dataset_root(config),
                partition="train",
                transform=train_transform,
            )
            validation_dataset = RsnaFusionDataset(
                data.validation,
                validation_matrix,
                structured_sample_ids=tuple(data.validation["sample_id"].astype(str)),
                dataset_root=_required_dataset_root(config),
                partition="validation",
                transform=evaluation_transform,
            )
            runtime = resolve_device(
                config.image.device,
                mixed_precision=config.image.mixed_precision,
                pin_memory_policy=config.image.pin_memory_policy,
            )
            loaders = build_image_loaders(
                train_dataset,
                validation_dataset,
                config=config.image,
                runtime=runtime,
                seed=config.training.seed,
            )
            positive_count, negative_count, pos_weight = training_class_weight(
                data.train["target"].to_numpy(dtype=np.int8)
            )
        builder = cast(FusionConcatModel, get_model(config.model.registry_key))
        model = builder.build(
            config.model,
            structured_dimension=int(contract["transformed_dimension"]),
            weights=None,
        )
        if not isinstance(model, RsnaConcatFusionModel):
            raise TypeError("Registered fusion builder returned an invalid model")
        initialize_fusion_encoder(model, source_encoder_state(source))
        model.to(runtime.device)
        with timed_phase(_LOGGER, "fusion_training", **context):
            fit = fit_two_stage_binary_model(
                model,
                loaders.train,
                loaders.validation,
                input_keys=("image", "structured"),
                config=config.image,
                runtime=runtime,
                pos_weight=pos_weight,
            )
        loaded = checkpoint_document(
            fit.selected_state_dict,
            selected_epoch=fit.selected_epoch,
            selected_stage=fit.selected_stage,
            validation_average_precision=fit.selected_validation_average_precision,
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
            config.training.report_directory / config.dataset.registry_key / "runs" / run_id
        )
        if report_directory.exists():
            raise FileExistsError(f"Fusion validation report already exists: {report_directory}")
        report_stage = staging_directory(report_directory)
        temporary = Path(tempfile.mkdtemp(prefix="radfusion-fusion-package-"))
        published = None
        try:
            checkpoint_path = save_neural_checkpoint(loaded, temporary / "model.pt")
            preprocessor_path = save_preprocessor(
                preprocessor, temporary / "structured_preprocessor.skops"
            )
            manifest = _manifest(
                config=config,
                data=data,
                commit=commit,
                dirty=dirty,
                lock_hash=lock_hash,
                environment=environment,
                runtime=runtime.provenance(),
                source_lineage=source_lineage,
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
            published = publish_neural_model_run(
                model_root=config.training.model_directory,
                mlflow_run_id=run_id,
                checkpoint_path=checkpoint_path,
                source_config_bytes=config.source_bytes,
                manifest=manifest,
                structured_preprocessor_path=preprocessor_path,
            )
            load_validated_rsna_fusion_preprocessor(published.run_directory, manifest)
            write_run_reports(
                report_stage,
                model_name=config.model.registry_key,
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
            mlflow.set_tags(
                {
                    "model_package_id": published.model_package_id,
                    "checkpoint_sha256": published.checkpoint_sha256,
                    "local_model_sha256": published.checkpoint_sha256,
                    "local_model_path": published.model_path.as_posix(),
                    "report_directory": report_directory.as_posix(),
                    "threshold_youden_j": str(thresholds["youden_j"]),
                    "threshold_target_sensitivity": str(thresholds["target_sensitivity"]),
                }
            )
            mlflow.set_tag("run_complete", "true")
        except BaseException:
            if published is not None and published.run_directory.exists():
                shutil.rmtree(published.run_directory)
            if report_directory.exists():
                shutil.rmtree(report_directory)
            raise
        finally:
            shutil.rmtree(temporary)
            if report_stage.exists():
                shutil.rmtree(report_stage)
        log_event(_LOGGER, "publication_completed", artifact="fusion_package", **context)
    return FusionModelResult(
        model_name=config.model.registry_key,
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
    if manifest.get("modality") != "fusion":
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
        or data.lineage.task_id != manifest["task"]
        or data.lineage.label_policy_version != manifest["label_policy_version"]
    ):
        raise ValueError("Fusion dataset lineage differs from the source CXR package")


def _transform(config: ExperimentConfig, *, training: bool) -> StandardCxrTransform:
    image = config.image
    if image is None:
        raise ValueError("Fusion transform requires image configuration")
    return StandardCxrTransform(
        training=training,
        image_size=int(config.model.parameters["image_size"]),
        rotation_degrees=image.rotation_degrees,
        translation_fraction=image.translation_fraction,
        brightness_jitter=image.brightness_jitter,
        contrast_jitter=image.contrast_jitter,
    )


def _required_dataset_root(config: ExperimentConfig) -> Path:
    if config.dataset.dataset_root is None:
        raise ValueError("Fusion experiment requires dataset.dataset_root")
    return config.dataset.dataset_root


def _manifest(
    *,
    config: ExperimentConfig,
    data: FusionRunData,
    commit: str,
    dirty: bool,
    lock_hash: str,
    environment: dict[str, str],
    runtime: dict[str, Any],
    source_lineage: SourceCxrLineage,
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
    image = config.image
    if image is None:
        raise ValueError("Fusion manifest requires image configuration")
    return {
        "modality": "fusion",
        "model": config.model.registry_key,
        "task": data.lineage.task_id,
        "positive_class": 1,
        "bundle_id": data.lineage.bundle_id,
        "bundle_manifest_sha256": data.bundle_manifest_sha256,
        "split_assignment_id": data.lineage.split_assignment_id,
        "label_policy_version": data.lineage.label_policy_version,
        "source_config_sha256": config.source_sha256,
        "semantic_config_sha256": fusion_semantic_config_sha256(config),
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
            "modality": "fusion",
            "encoder_architecture": config.model.parameters["encoder_name"],
            "image_size": config.model.parameters["image_size"],
            "embedding_dimension": config.model.parameters["embedding_dimension"],
            "classifier_output_dimension": 1,
            "pretrained_weight": source_pretrained_weight,
        },
        "source_cxr_lineage": source_lineage.as_dict(),
        "structured_preprocessor_sha256": preprocessor_sha256,
        "structured_preprocessor_contract": structured_contract,
        "structured_input_conversion": fusion_structured_input_conversion_contract(),
        "fusion_architecture": fusion_architecture_contract(
            config.model,
            structured_input_dimension=structured_contract["transformed_dimension"],
        ),
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
            "validation_average_precision": fit.selected_validation_average_precision,
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
        "source_authentication": data.authentication.as_dict(),
        "runtime_provenance": runtime,
    }
