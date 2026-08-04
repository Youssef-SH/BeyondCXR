"""Evaluate one completed training run on its pinned test partition."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from sqlalchemy.exc import SQLAlchemyError

from radfusion.data.cxr_cache import ValidatedCxrCache
from radfusion.data.hashing import sha256_file
from radfusion.data.tabular_preprocess import validate_metadata_pipeline
from radfusion.evaluation.latency import benchmark_single_sample_latency_ms
from radfusion.evaluation.metrics import evaluate_operating_point, evaluate_probabilities
from radfusion.evaluation.probabilities import positive_class_probabilities
from radfusion.training.config import ConfigError, ExperimentConfig, load_experiment_config
from radfusion.training.evaluate_fusion import (
    FusionTestEvaluationResult,
    evaluate_fusion_training_run,
)
from radfusion.training.evaluate_image import (
    ImageTestEvaluationResult,
    evaluate_image_training_run,
)
from radfusion.training.execution import LoaderExecutionPolicy
from radfusion.training.registry import RegistryError, get_dataset
from radfusion.training.train_tabular import (
    metrics_document,
    mlflow_metrics,
    validate_report_set,
    write_run_reports,
)
from radfusion.utils.mlflow_utils import (
    DEFAULT_TRACKING_URI,
    configure_mlflow,
    git_revision,
    tracked_run,
    uv_lock_sha256,
)
from radfusion.utils.model_publication import threshold_contract, validate_published_model
from radfusion.utils.operational_logging import (
    add_logging_argument,
    configure_logging,
    get_operational_logger,
    log_event,
    timed_phase,
)
from radfusion.utils.privacy import validate_public_reports
from radfusion.utils.publication import publish_directory, staging_directory
from radfusion.utils.skops_io import load_skops

_LOGGER = get_operational_logger(__name__)


@dataclass(frozen=True)
class TestEvaluationResult:
    """Outputs from one completed explicit test-evaluation run."""

    run_id: str
    training_run_id: str
    artifact_directory: Path
    average_precision: float


def evaluate_training_run(
    training_run_id: str,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    cache: ValidatedCxrCache | None = None,
    execution: LoaderExecutionPolicy | None = None,
) -> TestEvaluationResult | ImageTestEvaluationResult | FusionTestEvaluationResult:
    """Apply a completed training run's model and thresholds to test data."""
    client = configure_mlflow(tracking_uri=tracking_uri)
    source_run = client.get_run(training_run_id)
    source_modality = source_run.data.tags.get("modality")
    if source_modality == "image":
        return evaluate_image_training_run(
            training_run_id,
            tracking_uri=tracking_uri,
            cache=cache,
            execution=execution,
        )
    if source_modality == "fusion":
        return evaluate_fusion_training_run(
            training_run_id,
            tracking_uri=tracking_uri,
            cache=cache,
            execution=execution,
        )
    if source_modality != "metadata":
        raise ValueError("Source training run has no valid modality")
    if (
        source_run.info.status != "FINISHED"
        or source_run.data.tags.get("run_kind") != "training"
        or source_run.data.tags.get("evaluation_scope") != "validation"
        or source_run.data.tags.get("run_complete") != "true"
    ):
        raise ValueError("Source training run is not complete")
    model_path = Path(source_run.data.tags["local_model_path"])
    package = model_path.parent
    manifest = validate_published_model(package)
    config = load_experiment_config(package / "resolved_config.yaml")
    evaluator_commit, evaluator_dirty = git_revision()
    evaluator_lock_hash = uv_lock_sha256()
    _verify_training_lineage(
        source_run,
        config,
        manifest,
        model_path,
        evaluator_commit=evaluator_commit,
        evaluator_dirty=evaluator_dirty,
        evaluator_lock_hash=evaluator_lock_hash,
    )
    model = validate_metadata_pipeline(load_skops(model_path))
    log_event(_LOGGER, "training_package_verified", training_run_id=training_run_id)
    dataset_implementation = get_dataset(config.dataset.registry_key)
    pinned_lineage = dataset_implementation.load_lineage(config.dataset)
    if (
        pinned_lineage.bundle_id != manifest["bundle_id"]
        or pinned_lineage.split_assignment_id != manifest["split_assignment_id"]
        or pinned_lineage.task_id != manifest["task"]
    ):
        raise ValueError("Pinned bundle lineage differs from the trained model")
    configure_mlflow(
        experiment_name=config.mlflow.experiment_name,
        tracking_uri=tracking_uri,
    )
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
    if experiment is None or source_run.info.experiment_id != experiment.experiment_id:
        raise ValueError("Source training run belongs to a different MLflow experiment")
    best_iteration = manifest["best_iteration"]
    tags = {
        "run_kind": "test_evaluation",
        "evaluation_scope": "test",
        "source_training_run_id": training_run_id,
        "experiment_name": config.name,
        "dataset": config.dataset.registry_key,
        "dataset_bundle_id": manifest["bundle_id"],
        "split_assignment_id": manifest["split_assignment_id"],
        "task": manifest["task"],
        "modality": "metadata",
        "model": config.model.registry_key,
        "seed": str(manifest["seed"]),
        "model_sha256": manifest["model_sha256"],
        "model_package_id": manifest["model_package_id"],
        "git_commit": evaluator_commit,
        "git_dirty": str(evaluator_dirty).lower(),
        "dependency_lock_sha256": evaluator_lock_hash,
        "run_complete": "false",
    }
    with tracked_run(
        run_name=f"{config.name}-test",
        tags=tags,
        parameters={
            "source_training_run_id": training_run_id,
            "calibration_bins": config.evaluation.calibration_bins,
            "latency_warmup_calls": config.evaluation.latency_warmup_calls,
            "latency_measured_calls": config.evaluation.latency_measured_calls,
            "best_iteration": best_iteration,
        },
    ) as evaluation_run_id:
        context = {"run_id": evaluation_run_id, "model": config.model.registry_key}
        with timed_phase(_LOGGER, "test_dataset_loading", **context):
            test, lineage = dataset_implementation.load_test(config.dataset)
        if (
            lineage.bundle_id != manifest["bundle_id"]
            or lineage.split_assignment_id != manifest["split_assignment_id"]
            or lineage.task_id != manifest["task"]
        ):
            raise ValueError("Test bundle lineage differs from the trained model")
        with timed_phase(_LOGGER, "test_evaluation", **context):
            probabilities = positive_class_probabilities(
                model,
                test.features,
                best_iteration=best_iteration,
            )
            thresholds = {key: float(value) for key, value in manifest["thresholds"].items()}
            probability_metrics = evaluate_probabilities(
                test.targets,
                probabilities,
                calibration_bins=config.evaluation.calibration_bins,
            )
            youden = evaluate_operating_point(
                test.targets,
                probabilities,
                threshold=thresholds["youden_j"],
            )
            target_sensitivity = evaluate_operating_point(
                test.targets,
                probabilities,
                threshold=thresholds["target_sensitivity"],
            )
            latency_ms = benchmark_single_sample_latency_ms(
                model,
                test.features,
                warmup_calls=config.evaluation.latency_warmup_calls,
                measured_calls=config.evaluation.latency_measured_calls,
                best_iteration=best_iteration,
            )
        document = metrics_document(
            scope="test",
            calibration_bins=config.evaluation.calibration_bins,
            sensitivity_target=config.evaluation.sensitivity_target,
            thresholds=thresholds,
            probability=probability_metrics,
            youden=youden,
            target_sensitivity=target_sensitivity,
        )
        report_directory = (
            config.training.report_directory
            / config.dataset.registry_key
            / "runs"
            / evaluation_run_id
        )
        report_stage = staging_directory(report_directory)
        try:
            write_run_reports(
                report_stage,
                model_name=config.model.registry_key,
                targets=test.targets,
                probabilities=probabilities,
                document=document,
            )
            validate_report_set(report_stage)
            validate_public_reports(
                report_stage.iterdir(),
                forbidden_source_values={*test.sample_ids, *test.patient_ids},
            )
            mlflow.log_metrics(
                mlflow_metrics(
                    scope="test",
                    document=document,
                    latency_ms=latency_ms,
                    model_size_mib=model_path.stat().st_size / (1024.0 * 1024.0),
                )
            )
            publish_directory(report_stage, report_directory)
            mlflow.set_tags(
                {
                    "report_directory": report_directory.as_posix(),
                    "threshold_youden_j": str(thresholds["youden_j"]),
                    "threshold_target_sensitivity": str(thresholds["target_sensitivity"]),
                }
            )
            mlflow.set_tag("run_complete", "true")
        finally:
            if report_stage.exists():
                shutil.rmtree(report_stage)
        log_event(_LOGGER, "publication_completed", artifact="test_report", **context)
    return TestEvaluationResult(
        run_id=evaluation_run_id,
        training_run_id=training_run_id,
        artifact_directory=report_directory,
        average_precision=probability_metrics.average_precision,
    )


def _verify_training_lineage(
    run,
    config: ExperimentConfig,
    manifest,
    model_path: Path,
    *,
    evaluator_commit: str,
    evaluator_dirty: bool,
    evaluator_lock_hash: str,
) -> None:
    tags = run.data.tags
    expected_package = config.training.model_directory / "runs" / run.info.run_id
    if model_path.parent.resolve() != expected_package.resolve():
        raise ValueError("Training run points outside its configured model package")
    if manifest["training_mlflow_run_id"] != run.info.run_id:
        raise ValueError("Model manifest training run ID mismatch")
    checks = {
        "source_config_sha256": config.source_sha256,
        "bundle_id": config.dataset.bundle_id,
        "task": config.dataset.task_id,
        "seed": config.training.seed,
        "model": config.model.registry_key,
        "model_sha256": sha256_file(model_path),
    }
    for field, expected in checks.items():
        if manifest[field] != expected:
            raise ValueError(f"Model manifest {field} mismatch")
    if manifest["positive_class"] != 1:
        raise ValueError("Model positive-class contract is invalid")
    expected_tags = {
        "dataset_bundle_id": manifest["bundle_id"],
        "split_assignment_id": manifest["split_assignment_id"],
        "task": manifest["task"],
        "model": config.model.registry_key,
        "seed": str(manifest["seed"]),
        "git_commit": manifest["git_commit"],
        "git_dirty": str(manifest["git_dirty"]).lower(),
        "source_config_sha256": manifest["source_config_sha256"],
        "dependency_lock_sha256": manifest["dependency_lock_sha256"],
        "local_model_sha256": manifest["model_sha256"],
        "model_package_id": manifest["model_package_id"],
    }
    for name, expected in expected_tags.items():
        if tags.get(name) != expected:
            raise ValueError(f"Training run tag {name} mismatch")
    _verify_validation_choices(run, config, manifest)
    if manifest["threshold_contract"] != threshold_contract(
        sensitivity_target=config.evaluation.sensitivity_target,
    ):
        raise ValueError("Model threshold contract differs from the resolved configuration")
    if manifest["git_dirty"]:
        raise ValueError("Formal test evaluation rejects a dirty training package")
    if evaluator_dirty:
        raise ValueError("Formal test evaluation requires a clean current working tree")
    if evaluator_commit != manifest["git_commit"]:
        raise ValueError("Current Git commit does not match the model package")
    if evaluator_lock_hash != manifest["dependency_lock_sha256"]:
        raise ValueError("Current dependency lock does not match the model package")


def _verify_validation_choices(run, config: ExperimentConfig, manifest) -> None:
    for policy in ("youden_j", "target_sensitivity"):
        metric_name = f"validation_{policy}_threshold"
        value = run.data.metrics.get(metric_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"Training run metric {metric_name} is invalid")
        if not math.isclose(
            float(manifest["thresholds"][policy]),
            float(value),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Training run metric {metric_name} mismatch")

    source_best_iteration = run.data.params.get("best_iteration")
    if config.model.registry_key == "metadata_lightgbm":
        try:
            parsed_best_iteration = int(source_best_iteration)
        except (TypeError, ValueError) as exc:
            raise ValueError("Training run best_iteration is invalid") from exc
        if parsed_best_iteration <= 0 or str(parsed_best_iteration) != source_best_iteration:
            raise ValueError("Training run best_iteration is invalid")
        if manifest["best_iteration"] != parsed_best_iteration:
            raise ValueError("Training run best_iteration mismatch")
    elif config.model.registry_key == "metadata_logistic":
        if manifest["best_iteration"] is not None:
            raise ValueError("Logistic Regression manifest declares best_iteration")
        if source_best_iteration not in {None, "not_applicable"}:
            raise ValueError("Logistic Regression run declares best_iteration")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Completed training MLflow run ID")
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow SQLite tracking URI",
    )
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate one training run and print aggregate test lineage."""
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        result = evaluate_training_run(
            args.run_id,
            tracking_uri=args.tracking_uri,
        )
    except (
        ConfigError,
        RegistryError,
        MlflowException,
        SQLAlchemyError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"Test evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "training_run_id": result.training_run_id,
                "test_evaluation_run_id": result.run_id,
                "test_average_precision": result.average_precision,
                "artifact_directory": result.artifact_directory.as_posix(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
