"""Evaluate one package authorized by the complete formal RSNA package freeze."""

from __future__ import annotations

from pathlib import Path

import mlflow

from beyondcxr.data.rsna_cxr_cache import ValidatedCxrCache
from beyondcxr.data.rsna_metadata_preprocess import validate_metadata_pipeline
from beyondcxr.evaluation.latency import (
    LATENCY_MEASURED_CALLS,
    LATENCY_WARMUP_CALLS,
    benchmark_single_sample_latency_ms,
)
from beyondcxr.evaluation.probabilities import canonical_binary_raw_scores
from beyondcxr.training.config import (
    ExperimentConfig,
    require_runtime_seed,
    with_runtime,
)
from beyondcxr.training.device import ResolvedDevice, neural_inference_runtime_policy
from beyondcxr.training.execution import LoaderExecutionPolicy
from beyondcxr.training.neural import configure_neural_determinism
from beyondcxr.training.rsna_campaign_control import (
    ValidatedRsnaPackageFreeze,
)
from beyondcxr.training.rsna_evaluation_result import (
    CompletedRsnaEvaluation,
    publish_rsna_evaluation,
    validate_rsna_model_package,
    validated_rsna_evaluation_policy,
)
from beyondcxr.training.rsna_registry import get_dataset
from beyondcxr.training.rsna_training_report import mlflow_metrics
from beyondcxr.utils.mlflow_utils import (
    DEFAULT_TRACKING_URI,
    configure_mlflow,
    environment_provenance,
    serialize_modalities,
    tracked_run,
)
from beyondcxr.utils.operational_logging import (
    get_operational_logger,
    log_event,
    timed_phase,
)
from beyondcxr.utils.private_predictions import publish_prediction_evidence
from beyondcxr.utils.rsna_model_publication import (
    MODEL_FILENAME,
    validate_published_model,
)
from beyondcxr.utils.skops_io import load_skops

_LOGGER = get_operational_logger(__name__)


def evaluate_model_package(
    model_package_id: str,
    *,
    authorization: ValidatedRsnaPackageFreeze,
    evaluation_config: ExperimentConfig,
    runtime: ResolvedDevice,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    model_directory: str | Path = "models/rsna",
    cache: ValidatedCxrCache | None = None,
    execution: LoaderExecutionPolicy | None = None,
    private_output_directory: str | Path | None = None,
    report_directory: str | Path | None = None,
) -> CompletedRsnaEvaluation:
    """Dispatch held-out evaluation only after validating the complete package freeze."""
    frozen = authorization.require(model_package_id)
    position = authorization.packages.index(frozen)
    configured = authorization.execution.manifest["configs"][position]
    if (
        configured["family_id"] != evaluation_config.family.family_id
        or configured["seed"] != require_runtime_seed(evaluation_config)
        or configured["config_source_sha256"] != evaluation_config.config_source_sha256
        or configured["config_semantic_sha256"] != evaluation_config.config_semantic_sha256
    ):
        raise ValueError("Held-out evaluation config differs from the frozen formal plan")
    manifest = validate_rsna_model_package(model_directory, model_package_id)
    if manifest["family_id"] != configured["family_id"]:
        raise ValueError("Held-out package family differs from the frozen formal plan")
    if evaluation_config.neural is not None:
        if (
            neural_inference_runtime_policy(runtime)
            != authorization.execution.manifest["training_evaluation_runtime"]
        ):
            raise ValueError("Held-out numerical runtime differs from the frozen execution")
        configure_neural_determinism()
    package = Path(model_directory) / "packages" / model_package_id
    if (package / MODEL_FILENAME).is_file():
        return _evaluate_tabular_package(
            package,
            authorization=authorization,
            evaluation_config=evaluation_config,
            tracking_uri=tracking_uri,
            private_output_directory=private_output_directory,
            report_directory=report_directory,
        )
    from beyondcxr.training.rsna_evaluate_cxr import evaluate_cxr_model_package
    from beyondcxr.training.rsna_evaluate_fusion import evaluate_fusion_model_package

    if manifest["family_id"] == "cxr_densenet":
        return evaluate_cxr_model_package(
            model_package_id,
            authorization=authorization,
            evaluation_config=evaluation_config,
            tracking_uri=tracking_uri,
            model_directory=model_directory,
            cache=cache,
            execution=execution,
            runtime=runtime,
            private_output_directory=private_output_directory,
            report_directory=report_directory,
        )
    return evaluate_fusion_model_package(
        model_package_id,
        authorization=authorization,
        evaluation_config=evaluation_config,
        tracking_uri=tracking_uri,
        model_directory=model_directory,
        cache=cache,
        execution=execution,
        runtime=runtime,
        private_output_directory=private_output_directory,
        report_directory=report_directory,
    )


def _evaluate_tabular_package(
    package: Path,
    *,
    authorization: ValidatedRsnaPackageFreeze,
    evaluation_config: ExperimentConfig,
    tracking_uri: str,
    private_output_directory: str | Path | None,
    report_directory: str | Path | None,
) -> CompletedRsnaEvaluation:
    manifest = validate_published_model(package)
    policy = validated_rsna_evaluation_policy(manifest, evaluation_config)
    config = with_runtime(
        evaluation_config,
        seed=int(manifest["seed"]),
        model_directory=package.parent.parent,
        private_output_directory=private_output_directory,
        report_directory=report_directory,
    )
    model = validate_metadata_pipeline(load_skops(package / MODEL_FILENAME))
    dataset = get_dataset(config.dataset.dataset_id)
    pinned = dataset.load_lineage(config)
    if (
        pinned.bundle_id != manifest["bundle_id"]
        or pinned.split_assignment_id != manifest["split_assignment_id"]
        or pinned.task_id != manifest["task_id"]
    ):
        raise ValueError("Pinned bundle lineage differs from the model package")
    configure_mlflow(experiment_name=config.runtime.experiment_name, tracking_uri=tracking_uri)
    tags = {
        "run_kind": "test_evaluation",
        "evaluation_scope": "test",
        "dataset_id": config.dataset.dataset_id,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": manifest["bundle_manifest_sha256"],
        "split_assignment_id": manifest["split_assignment_id"],
        "task_id": manifest["task_id"],
        "family_id": config.family.family_id,
        "modalities": serialize_modalities(manifest["modalities"]),
        "seed": str(manifest["seed"]),
        "config_source_sha256": config.config_source_sha256,
        "config_semantic_sha256": config.config_semantic_sha256,
        "git_commit": manifest["git_commit"],
        "dependency_lock_sha256": manifest["dependency_lock_sha256"],
        "package_kind": "model",
        "package_id": manifest["model_package_id"],
        "run_complete": "false",
    }
    with tracked_run(
        run_name=f"{config.family.family_id}-test",
        tags=tags,
        parameters={"package_id": manifest["model_package_id"], **environment_provenance()},
    ) as run_id:
        context = {"run_id": run_id, "family_id": config.family.family_id}
        with timed_phase(_LOGGER, "test_dataset_loading", **context):
            test, lineage = dataset.load_test(
                config,
                authorization=authorization,
                package_id=str(manifest["model_package_id"]),
            )
        if (
            lineage.bundle_id != manifest["bundle_id"]
            or lineage.split_assignment_id != manifest["split_assignment_id"]
            or lineage.task_id != manifest["task_id"]
        ):
            raise ValueError("Test bundle lineage differs from the model package")
        logits = canonical_binary_raw_scores(
            model,
            test.features,
            best_iteration=manifest["best_iteration"],
        )
        latency_ms = benchmark_single_sample_latency_ms(
            model,
            test.features,
            warmup_calls=LATENCY_WARMUP_CALLS,
            measured_calls=LATENCY_MEASURED_CALLS,
            best_iteration=manifest["best_iteration"],
        )
        evidence = publish_prediction_evidence(
            private_root=config.runtime.private_output_directory,
            dataset_id="rsna",
            model_package_id=manifest["model_package_id"],
            task_id=manifest["task_id"],
            bundle_id=manifest["bundle_id"],
            split_assignment_id=manifest["split_assignment_id"],
            scope="test",
            sample_ids=test.sample_ids,
            targets=test.targets,
            logits=logits,
        )
        result = publish_rsna_evaluation(
            report_root=config.runtime.report_directory,
            model_root=config.runtime.model_directory,
            private_root=config.runtime.private_output_directory,
            evidence=evidence,
            evaluation_policy=policy,
            forbidden_source_values=(*test.sample_ids, *test.patient_ids),
        )
        claims = result.manifest["claims"]
        mlflow.log_metrics(
            mlflow_metrics(
                scope="test",
                document=claims,
                latency_ms=latency_ms,
                model_size_mib=(package / MODEL_FILENAME).stat().st_size / (1024.0 * 1024.0),
            )
        )
        mlflow.log_param("report_directory", result.directory.as_posix())
        mlflow.set_tags(
            {
                "prediction_id": evidence.manifest["prediction_id"],
                "evaluation_id": result.manifest["evaluation_id"],
                "run_complete": "true",
            }
        )
        log_event(_LOGGER, "publication_completed", artifact="evaluation_result", **context)
    return CompletedRsnaEvaluation(
        evaluation_id=result.manifest["evaluation_id"],
        prediction_id=evidence.manifest["prediction_id"],
        model_package_id=manifest["model_package_id"],
        mlflow_run_id=run_id,
        artifact_directory=result.directory,
        private_prediction_directory=evidence.directory,
        average_precision=float(
            result.manifest["claims"]["probability_metrics"]["average_precision"]
        ),
    )
