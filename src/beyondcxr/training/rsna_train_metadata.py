"""Train and validate one configured RSNA metadata family."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from beyondcxr.data.rsna_metadata_preprocess import (
    metadata_input_contract,
    validate_metadata_pipeline,
)
from beyondcxr.evaluation.latency import (
    LATENCY_MEASURED_CALLS,
    LATENCY_SAMPLE_POLICY,
    LATENCY_WARMUP_CALLS,
    benchmark_single_sample_latency_ms,
)
from beyondcxr.evaluation.metrics import (
    OperatingPointMetrics,
    ProbabilityMetrics,
    evaluate_operating_point,
    evaluate_probabilities,
    target_sensitivity_threshold,
    youden_j_threshold,
)
from beyondcxr.evaluation.probabilities import positive_class_probabilities
from beyondcxr.training.config import (
    ExperimentConfig,
    require_runtime_seed,
)
from beyondcxr.training.rsna_formal import rsna_run_report_root
from beyondcxr.training.rsna_registry import get_dataset, get_model
from beyondcxr.training.rsna_training_report import (
    metrics_document,
    mlflow_metrics,
    validate_training_report,
    write_run_reports,
)
from beyondcxr.training.rsna_validation_evidence import (
    VALIDATION_EVIDENCE_FILENAME,
    write_validation_evidence,
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
from beyondcxr.utils.rsna_model_publication import publish_model_package, threshold_contract
from beyondcxr.utils.skops_io import load_skops, save_skops

_LOGGER = get_operational_logger(__name__)


@dataclass(frozen=True)
class MetadataModelResult:
    """Published outputs from one completed training run."""

    run_id: str
    validation_probability: ProbabilityMetrics
    validation_youden_j: OperatingPointMetrics
    validation_target_sensitivity: OperatingPointMetrics
    thresholds: dict[str, float]
    model_path: Path
    model_sha256: str
    model_package_id: str
    artifact_directory: Path
    latency_ms: float
    model_size_mib: float


def train_metadata_experiment(
    config: ExperimentConfig,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> MetadataModelResult:
    """Fit on train, select thresholds on validation, and publish the model."""
    seed = require_runtime_seed(config)
    if config.evaluation is None:
        raise ValueError("RSNA tabular training requires evaluation policy")
    configure_mlflow(
        experiment_name=config.runtime.experiment_name,
        tracking_uri=tracking_uri,
    )
    commit, dirty = git_revision()
    lock_hash = uv_lock_sha256()
    base_tags = {
        "run_kind": "training",
        "evaluation_scope": "validation",
        "dataset_id": config.dataset.dataset_id,
        "task_id": config.task.task_id,
        "family_id": config.family.family_id,
        "modalities": serialize_modalities(config.family.modalities),
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
    base_parameters = {
        "training_seed": seed,
        "sensitivity_target": config.evaluation.sensitivity_target,
        "calibration_bins": config.evaluation.calibration_bins,
        "latency_sample_policy": LATENCY_SAMPLE_POLICY,
        "latency_warmup_calls": LATENCY_WARMUP_CALLS,
        "latency_measured_calls": LATENCY_MEASURED_CALLS,
        **dict(config.family.parameters),
        **dict(config.training.parameters),
    }
    with tracked_run(
        run_name=config.family.family_id,
        tags=base_tags,
        parameters=base_parameters,
    ) as run_id:
        log_source_config(config)
        context = {"run_id": run_id, "family_id": config.family.family_id}
        with timed_phase(_LOGGER, "dataset_loading", **context):
            dataset = get_dataset(config.dataset.dataset_id).load_train_validation(config)
        log_event(
            _LOGGER,
            "dataset_loaded",
            train_count=len(dataset.train.targets),
            validation_count=len(dataset.validation.targets),
            **context,
        )
        if dataset.lineage.split_assignment_id != config.dataset.split_assignment_id:
            raise ValueError("Loaded split assignment differs from the configuration")
        with timed_phase(_LOGGER, "model_fitting", **context):
            model_fit = get_model(config.family.family_id).fit(
                config,
                seed,
                dataset.train.features,
                dataset.train.targets,
                dataset.validation.features,
                dataset.validation.targets,
            )
        best_iteration = _best_iteration(model_fit.derived_parameters)
        with timed_phase(_LOGGER, "validation_evaluation", **context):
            probabilities = positive_class_probabilities(
                model_fit.pipeline,
                dataset.validation.features,
                best_iteration=best_iteration,
            )
            thresholds = {
                "youden_j": youden_j_threshold(dataset.validation.targets, probabilities),
                "target_sensitivity": target_sensitivity_threshold(
                    dataset.validation.targets,
                    probabilities,
                    sensitivity=config.evaluation.sensitivity_target,
                ),
            }
            probability_metrics = evaluate_probabilities(
                dataset.validation.targets,
                probabilities,
                calibration_bins=config.evaluation.calibration_bins,
            )
            youden_metrics = evaluate_operating_point(
                dataset.validation.targets,
                probabilities,
                threshold=thresholds["youden_j"],
            )
            sensitivity_metrics = evaluate_operating_point(
                dataset.validation.targets,
                probabilities,
                threshold=thresholds["target_sensitivity"],
            )
            latency_ms = benchmark_single_sample_latency_ms(
                model_fit.pipeline,
                dataset.validation.features,
                warmup_calls=LATENCY_WARMUP_CALLS,
                measured_calls=LATENCY_MEASURED_CALLS,
                best_iteration=best_iteration,
            )
        log_event(
            _LOGGER,
            "validation_completed",
            average_precision=probability_metrics.average_precision,
            roc_auc=probability_metrics.roc_auc,
            brier_score=probability_metrics.brier_score,
            **context,
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
        report_directory = rsna_run_report_root(config.runtime.report_directory) / run_id
        report_stage = staging_directory(report_directory)
        temporary_model_root = Path(tempfile.mkdtemp(prefix="beyondcxr-model-"))
        published = None
        try:
            serialized = save_skops(model_fit.pipeline, temporary_model_root / "model.skops")
            restored = load_skops(serialized)
            validate_metadata_pipeline(restored)
            restored_probabilities = positive_class_probabilities(
                restored,
                dataset.validation.features,
                best_iteration=best_iteration,
            )
            if not np.array_equal(restored_probabilities, probabilities):
                raise ValueError("Serialized model probabilities differ from fitted probabilities")
            mlflow.log_params(
                {
                    **dict(model_fit.derived_parameters),
                    **environment_provenance(),
                    "train_positive_count": int((dataset.train.targets == 1).sum()),
                    "train_negative_count": int((dataset.train.targets == 0).sum()),
                }
            )
            mlflow.log_metrics(
                mlflow_metrics(
                    scope="validation",
                    document=document,
                    latency_ms=latency_ms,
                    model_size_mib=serialized.stat().st_size / (1024.0 * 1024.0),
                )
            )
            published = publish_model_package(
                model_root=config.runtime.model_directory,
                serialized_model_path=serialized,
                source_config_bytes=config.source_bytes,
                validation_evidence_path=write_validation_evidence(
                    temporary_model_root / VALIDATION_EVIDENCE_FILENAME,
                    config=config,
                    sample_ids=dataset.validation.sample_ids,
                    targets=dataset.validation.targets,
                    probabilities=probabilities,
                ),
                manifest={
                    "bundle_id": dataset.lineage.bundle_id,
                    "split_assignment_id": dataset.lineage.split_assignment_id,
                    "task_id": dataset.lineage.task_id,
                    "positive_class": 1,
                    "family_id": config.family.family_id,
                    "config_source_sha256": config.config_source_sha256,
                    "config_semantic_sha256": config.config_semantic_sha256,
                    "seed": seed,
                    "git_commit": commit,
                    "git_dirty": dirty,
                    "dependency_lock_sha256": lock_hash,
                    "best_iteration": best_iteration,
                    "thresholds": thresholds,
                    "threshold_contract": threshold_contract(
                        sensitivity_target=config.evaluation.sensitivity_target,
                    ),
                    "input_contract": metadata_input_contract(),
                },
            )
            document["training_package"] = {
                "run_id": run_id,
                "model_package_id": published.model_package_id,
                "family_id": config.family.family_id,
                "seed": seed,
            }
            write_run_reports(
                report_stage,
                model_name=config.family.family_id,
                targets=dataset.validation.targets,
                probabilities=probabilities,
                document=document,
            )
            validate_training_report(
                report_stage,
                run_id=run_id,
                model_package_id=published.model_package_id,
                family_id=config.family.family_id,
                seed=seed,
                package=json.loads(published.manifest_path.read_text(encoding="utf-8")),
                package_directory=published.package_directory,
            )
            validate_public_reports(
                report_stage.iterdir(),
                forbidden_source_values={
                    *dataset.train.sample_ids,
                    *dataset.train.patient_ids,
                    *dataset.validation.sample_ids,
                    *dataset.validation.patient_ids,
                },
            )
            publish_directory(report_stage, report_directory)
            mlflow.log_params(
                {
                    "model_sha256": published.model_sha256,
                    "model_path": published.model_path.as_posix(),
                    "report_directory": report_directory.as_posix(),
                }
            )
            mlflow.set_tags({"package_kind": "model", "package_id": published.model_package_id})
            mlflow.set_tag("run_complete", "true")
        finally:
            if report_stage.exists():
                shutil.rmtree(report_stage)
            shutil.rmtree(temporary_model_root, ignore_errors=True)
        log_event(_LOGGER, "publication_completed", artifact="model_package", **context)
        log_event(_LOGGER, "publication_completed", artifact="validation_report", **context)

    return MetadataModelResult(
        run_id=run_id,
        validation_probability=probability_metrics,
        validation_youden_j=youden_metrics,
        validation_target_sensitivity=sensitivity_metrics,
        thresholds=thresholds,
        model_path=published.model_path,
        model_sha256=published.model_sha256,
        model_package_id=published.model_package_id,
        artifact_directory=report_directory,
        latency_ms=latency_ms,
        model_size_mib=published.model_size_mib,
    )


def _best_iteration(parameters: Any) -> int | None:
    value = parameters.get("best_iteration")
    return int(value) if value is not None and int(value) > 0 else None
