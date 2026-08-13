"""Train and validate one configured tabular metadata experiment."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "radfusion-matplotlib"))

import mlflow
import numpy as np

from radfusion.data.rsna_metadata_preprocess import (
    metadata_input_contract,
    validate_metadata_pipeline,
)
from radfusion.evaluation.latency import (
    LATENCY_MEASURED_CALLS,
    LATENCY_SAMPLE_POLICY,
    LATENCY_WARMUP_CALLS,
    benchmark_single_sample_latency_ms,
)
from radfusion.evaluation.metrics import (
    CALIBRATION_BINNING_STRATEGY,
    OperatingPointMetrics,
    ProbabilityMetrics,
    evaluate_operating_point,
    evaluate_probabilities,
    target_sensitivity_threshold,
    youden_j_threshold,
)
from radfusion.evaluation.plots import write_evaluation_plots
from radfusion.evaluation.probabilities import positive_class_probabilities
from radfusion.training.config import ExperimentConfig, require_runtime_seed
from radfusion.training.rsna_registry import get_dataset, get_model
from radfusion.utils.mlflow_utils import (
    DEFAULT_TRACKING_URI,
    configure_mlflow,
    environment_provenance,
    git_revision,
    log_source_config,
    tracked_run,
    uv_lock_sha256,
)
from radfusion.utils.model_publication import publish_model_run, threshold_contract
from radfusion.utils.operational_logging import get_operational_logger, log_event, timed_phase
from radfusion.utils.privacy import validate_public_reports
from radfusion.utils.publication import publish_directory, staging_directory
from radfusion.utils.skops_io import load_skops, save_skops

REQUIRED_REPORT_FILENAMES = frozenset(
    {
        "metrics.json",
        "evaluation_report.md",
        "confusion_summary.md",
        "roc_curve.png",
        "precision_recall_curve.png",
        "calibration_curve.png",
        "confusion_matrix_youden_j.png",
        "confusion_matrix_target_sensitivity.png",
    }
)
_LOGGER = get_operational_logger(__name__)


@dataclass(frozen=True)
class ModelResult:
    """Published outputs from one completed training run."""

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
    latency_ms: float
    model_size_mib: float


def train_configured_experiment(
    config: ExperimentConfig,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> ModelResult:
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
        "experiment_name": config.family.family_id,
        "dataset": config.dataset.dataset_id,
        "dataset_bundle_id": config.dataset.bundle_id,
        "task": config.task.task_id,
        "modality": "metadata",
        "model": config.family.family_id,
        "seed": str(seed),
        "git_commit": commit,
        "git_dirty": str(dirty).lower(),
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
        context = {"run_id": run_id, "model": config.family.family_id}
        with timed_phase(_LOGGER, "dataset_loading", **context):
            dataset = get_dataset(config.dataset.dataset_id).load_train_validation(config)
        log_event(
            _LOGGER,
            "dataset_loaded",
            train_count=len(dataset.train.targets),
            validation_count=len(dataset.validation.targets),
            **context,
        )
        mlflow.set_tags(
            {
                "split_assignment_id": dataset.lineage.split_assignment_id,
                "label_policy_version": dataset.lineage.label_policy_version,
            }
        )
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
        report_directory = (
            config.runtime.report_directory / config.dataset.dataset_id / "runs" / run_id
        )
        report_stage = staging_directory(report_directory)
        temporary_model_root = Path(tempfile.mkdtemp(prefix="radfusion-model-"))
        try:
            write_run_reports(
                report_stage,
                model_name=config.family.family_id,
                targets=dataset.validation.targets,
                probabilities=probabilities,
                document=document,
            )
            validate_report_set(report_stage)
            validate_public_reports(
                report_stage.iterdir(),
                forbidden_source_values={
                    *dataset.train.sample_ids,
                    *dataset.train.patient_ids,
                    *dataset.validation.sample_ids,
                    *dataset.validation.patient_ids,
                },
            )
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
            published = publish_model_run(
                model_root=config.runtime.model_directory,
                mlflow_run_id=run_id,
                serialized_model_path=serialized,
                source_config_bytes=config.source_bytes,
                manifest={
                    "bundle_id": dataset.lineage.bundle_id,
                    "split_assignment_id": dataset.lineage.split_assignment_id,
                    "task": dataset.lineage.task_id,
                    "positive_class": 1,
                    "model": config.family.family_id,
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
            publish_directory(report_stage, report_directory)
            mlflow.set_tags(
                {
                    "local_model_path": published.model_path.as_posix(),
                    "local_model_sha256": published.model_sha256,
                    "model_package_id": published.model_package_id,
                    "report_directory": report_directory.as_posix(),
                }
            )
            mlflow.set_tag("run_complete", "true")
        finally:
            if report_stage.exists():
                shutil.rmtree(report_stage)
            shutil.rmtree(temporary_model_root, ignore_errors=True)
        log_event(_LOGGER, "publication_completed", artifact="model_package", **context)
        log_event(_LOGGER, "publication_completed", artifact="validation_report", **context)

    return ModelResult(
        model_name=config.family.family_id,
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


def metrics_document(
    *,
    scope: str,
    calibration_bins: int,
    sensitivity_target: float,
    thresholds: dict[str, float],
    probability: ProbabilityMetrics,
    youden: OperatingPointMetrics,
    target_sensitivity: OperatingPointMetrics,
) -> dict[str, Any]:
    """Build one aggregate metrics document."""
    return {
        "evaluation_scope": scope,
        "calibration": {
            "calibration_bins": calibration_bins,
            "calibration_binning_strategy": CALIBRATION_BINNING_STRATEGY,
        },
        "probability_metrics": probability.as_dict(),
        "operating_points": {
            "youden_j": {
                "threshold": thresholds["youden_j"],
                "metrics": youden.as_dict(),
            },
            "target_sensitivity": {
                "configured_target_sensitivity": sensitivity_target,
                "threshold": thresholds["target_sensitivity"],
                "metrics": target_sensitivity.as_dict(),
            },
        },
    }


def write_run_reports(
    directory: Path,
    *,
    model_name: str,
    targets: np.ndarray,
    probabilities: np.ndarray,
    document: dict[str, Any],
) -> None:
    """Render the aggregate report set for one evaluation scope."""
    directory.mkdir(parents=True, exist_ok=True)
    operating = document["operating_points"]
    write_evaluation_plots(
        targets,
        probabilities,
        youden_j_threshold=operating["youden_j"]["threshold"],
        target_sensitivity_threshold=operating["target_sensitivity"]["threshold"],
        calibration_bins=document["calibration"]["calibration_bins"],
        output_directory=directory,
    )
    (directory / "metrics.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_evaluation_report(directory / "evaluation_report.md", model_name, document)
    _write_confusion_summary(directory / "confusion_summary.md", document)


def validate_report_set(directory: str | Path) -> None:
    """Require the complete aggregate report set and no additional entries."""
    report_directory = Path(directory)
    with os.scandir(report_directory) as entries:
        inspected = list(entries)
    actual = {entry.name for entry in inspected}
    if actual != REQUIRED_REPORT_FILENAMES:
        missing = sorted(REQUIRED_REPORT_FILENAMES - actual)
        unexpected = sorted(actual - REQUIRED_REPORT_FILENAMES)
        raise ValueError(f"Run report set is invalid: missing={missing}, unexpected={unexpected}")
    if any(entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in inspected):
        raise ValueError("Run reports must be regular non-symlink files")


def mlflow_metrics(
    *,
    scope: str,
    document: dict[str, Any],
    latency_ms: float | None,
    model_size_mib: float,
) -> dict[str, float]:
    """Flatten one aggregate document into stable MLflow metric names."""
    metrics = {
        f"{scope}_{key}": float(value) for key, value in document["probability_metrics"].items()
    }
    for policy, values in document["operating_points"].items():
        metrics[f"{scope}_{policy}_threshold"] = float(values["threshold"])
        metrics.update(
            {f"{scope}_{policy}_{key}": float(value) for key, value in values["metrics"].items()}
        )
    if latency_ms is not None:
        metrics[f"{scope}_latency_ms"] = latency_ms
    metrics["model_size_mib"] = model_size_mib
    return metrics


def _write_evaluation_report(path: Path, model_name: str, document: dict[str, Any]) -> None:
    scope = document["evaluation_scope"]
    probability = document["probability_metrics"]
    lines = [
        f"# {model_name} {scope} evaluation",
        "",
        "| Average Precision | ROC-AUC | Brier | ECE | Calibration slope | Calibration intercept |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {probability['average_precision']:.6f} | {probability['roc_auc']:.6f} | "
        f"{probability['brier_score']:.6f} | "
        f"{probability['expected_calibration_error']:.6f} | "
        f"{probability['calibration_slope']:.6f} | "
        f"{probability['calibration_intercept']:.6f} |",
    ]
    for policy, title in (
        ("youden_j", "Youden-J operating point"),
        ("target_sensitivity", "Target-sensitivity operating point"),
    ):
        values = document["operating_points"][policy]
        metrics = values["metrics"]
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"Validation-derived threshold: `{values['threshold']:.10f}`.",
                "",
                "| Precision | Recall | Specificity | F1 |",
                "| ---: | ---: | ---: | ---: |",
                f"| {metrics['precision']:.6f} | {metrics['recall']:.6f} | "
                f"{metrics['specificity']:.6f} | {metrics['f1']:.6f} |",
            ]
        )
    if image_run := document.get("image_run"):
        selection = image_run["selection"]
        authentication = image_run["source_authentication"]
        lines.extend(
            [
                "",
                "## Neural training summary",
                "",
                f"- Selected state: {selection['selected_stage']} epoch "
                f"{selection['selected_epoch']}",
                f"- Authenticated train/validation files: {authentication['file_count']}",
                "- Test data were not loaded, decoded, authenticated, or evaluated.",
            ]
        )
    if image_evaluation := document.get("image_evaluation"):
        counts = image_evaluation["test_counts"]
        lines.extend(
            [
                "",
                "## Verified neural test evaluation",
                "",
                f"- Test samples: {counts['sample_count']}",
                f"- Positive samples: {counts['positive_count']}",
                f"- Negative samples: {counts['negative_count']}",
                "- Package and checkpoint verification completed before test access.",
                "- Operating points were frozen on validation.",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_confusion_summary(path: Path, document: dict[str, Any]) -> None:
    lines = [f"# Aggregate {document['evaluation_scope']} confusion summary", ""]
    for policy, title in (
        ("youden_j", "Youden-J operating point"),
        ("target_sensitivity", "Target-sensitivity operating point"),
    ):
        values = document["operating_points"][policy]["metrics"]
        lines.extend(
            [
                f"## {title}",
                "",
                f"- True negatives: {values['true_negative']:,}",
                f"- False positives: {values['false_positive']:,}",
                f"- False negatives: {values['false_negative']:,}",
                f"- True positives: {values['true_positive']:,}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _best_iteration(parameters: Any) -> int | None:
    value = parameters.get("best_iteration")
    return int(value) if value is not None and int(value) > 0 else None
