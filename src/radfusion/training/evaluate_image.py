"""Evaluate one verified image package on its pinned RSNA test partition."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlflow
from torch import nn

from radfusion.data.cxr_cache import ValidatedCxrCache
from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.evaluation.metrics import evaluate_operating_point, evaluate_probabilities
from radfusion.models.cxr_baseline import ImageDenseNetModel
from radfusion.training.config import (
    ExperimentConfig,
    image_semantic_config_sha256,
    load_experiment_config,
)
from radfusion.training.datasets import (
    RsnaCachedImageDataset,
    RsnaDataset,
    expected_rsna_cxr_cache_identity,
    prepare_rsna_cxr_cache,
)
from radfusion.training.device import resolve_device
from radfusion.training.execution import LoaderExecutionPolicy, configured_loader_policy
from radfusion.training.neural import (
    build_evaluation_loader,
    deterministic_inference,
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
    tracked_run,
    uv_lock_sha256,
)
from radfusion.utils.neural_publication import (
    load_validated_neural_checkpoint,
    strict_load_checkpoint,
    validate_neural_package_metadata,
)
from radfusion.utils.operational_logging import (
    CountProgress,
    get_operational_logger,
    log_event,
    timed_phase,
)
from radfusion.utils.privacy import validate_public_reports
from radfusion.utils.private_predictions import (
    private_root_for_reports,
    publish_private_neural_predictions,
)
from radfusion.utils.publication import publish_directory, staging_directory

_LOGGER = get_operational_logger(__name__)


@dataclass(frozen=True)
class ImageTestEvaluationResult:
    """Outputs from one completed explicit neural test-evaluation run."""

    run_id: str
    training_run_id: str
    artifact_directory: Path
    private_prediction_directory: Path
    average_precision: float


def evaluate_image_training_run(
    training_run_id: str,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    cache: ValidatedCxrCache | None = None,
    execution: LoaderExecutionPolicy | None = None,
) -> ImageTestEvaluationResult:
    """Verify one explicit image package before accessing its test partition."""
    client = configure_mlflow(tracking_uri=tracking_uri)
    source_run = client.get_run(training_run_id)
    experiment_name = source_run.data.tags.get("experiment_name", "")
    if not experiment_name:
        raise ValueError("Source training run has no experiment identity")
    mlflow.set_experiment(experiment_id=source_run.info.experiment_id)
    evaluator_commit, evaluator_dirty = git_revision()
    evaluator_lock_hash = uv_lock_sha256()
    environment = environment_provenance()
    initial_tags = {
        "run_kind": "test_evaluation",
        "evaluation_scope": "test",
        "source_training_run_id": training_run_id,
        "experiment_name": experiment_name,
        "dataset": source_run.data.tags.get("dataset", ""),
        "dataset_bundle_id": source_run.data.tags.get("dataset_bundle_id", ""),
        "task": source_run.data.tags.get("task", ""),
        "modality": "image",
        "model": source_run.data.tags.get("model", ""),
        "seed": source_run.data.tags.get("seed", ""),
        "git_commit": evaluator_commit,
        "git_dirty": str(evaluator_dirty).lower(),
        "dependency_lock_sha256": evaluator_lock_hash,
        "run_complete": "false",
    }
    with tracked_run(
        run_name=f"{experiment_name}-test",
        tags=initial_tags,
        parameters={
            "source_training_run_id": training_run_id,
            **environment,
        },
    ) as evaluation_run_id:
        context = {"run_id": evaluation_run_id, "model": initial_tags["model"]}
        if (
            source_run.info.status != "FINISHED"
            or source_run.data.tags.get("run_kind") != "training"
            or source_run.data.tags.get("evaluation_scope") != "validation"
            or source_run.data.tags.get("modality") != "image"
            or source_run.data.tags.get("run_complete") != "true"
        ):
            raise ValueError("Source image training run is not complete")
        model_path = Path(source_run.data.tags["local_model_path"])
        package_directory = model_path.parent
        manifest = validate_neural_package_metadata(package_directory)
        config = load_experiment_config(package_directory / "resolved_config.yaml")
        verify_image_training_package(
            source_run,
            config,
            manifest,
            training_run_id=training_run_id,
            evaluator_commit=evaluator_commit,
            evaluator_dirty=evaluator_dirty,
            evaluator_lock_hash=evaluator_lock_hash,
        )
        checkpoint = load_validated_neural_checkpoint(package_directory, manifest)
        model_builder = cast(ImageDenseNetModel, get_model(config.model.registry_key))
        model = model_builder.build_architecture(config.model)
        if not isinstance(model, nn.Module):
            raise TypeError("Registered image model builder must return torch.nn.Module")
        strict_load_checkpoint(model, checkpoint)
        log_event(_LOGGER, "training_package_verified", **context)
        dataset_adapter = get_dataset(config.dataset.registry_key)
        with timed_phase(_LOGGER, "dataset_loading", **context):
            image_data = dataset_adapter.load_image_test(
                config.dataset,
                expected_manifest_sha256=manifest["bundle_manifest_sha256"],
            )
        if (
            image_data.lineage.bundle_id != manifest["bundle_id"]
            or image_data.lineage.split_assignment_id != manifest["split_assignment_id"]
            or image_data.lineage.task_id != manifest["task"]
            or image_data.lineage.label_policy_version != manifest["label_policy_version"]
            or image_data.bundle_manifest_sha256 != manifest["bundle_manifest_sha256"]
        ):
            raise ValueError("Test bundle lineage differs from the neural package")
        image = config.image
        if image is None or config.dataset.dataset_root is None:
            raise ValueError("Verified image package has an incomplete configuration")
        evaluation_transform = StandardCxrTransform(
            training=False,
            image_size=int(config.model.parameters["image_size"]),
            rotation_degrees=image.rotation_degrees,
            translation_fraction=image.translation_fraction,
            brightness_jitter=image.brightness_jitter,
            contrast_jitter=image.contrast_jitter,
        )
        if evaluation_transform.contract() != manifest["evaluation_transform_contract"]:
            raise ValueError("Evaluation transform differs from the neural package")
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
        _verify_source_authentication(authentication, manifest)
        if manifest["runtime_provenance"]["cxr_cache_id"] != resolved_cache.identity.cache_id:
            raise ValueError("Validated cache identity differs from the image package")
        test_dataset = RsnaCachedImageDataset(
            image_data.test,
            cache=resolved_cache,
            expected_cache_identity=expected_cache_identity,
            partition="test",
            transform=evaluation_transform,
            training_seed=config.training.seed,
        )
        runtime = resolve_device(
            image.device,
            mixed_precision=image.mixed_precision,
            pin_memory_policy=image.pin_memory_policy,
        )
        loader_execution = execution or configured_loader_policy(
            num_workers=image.num_workers,
            pin_memory=runtime.pin_memory_effective,
        )
        test_loader = build_evaluation_loader(
            test_dataset,
            config=image,
            runtime=runtime,
            execution=loader_execution,
        )
        model.to(runtime.device)
        with timed_phase(_LOGGER, "test_inference", **context):
            inference_progress: CountProgress | None = None

            def report_inference_progress(completed: int, total: int) -> None:
                nonlocal inference_progress
                if inference_progress is None:
                    inference_progress = CountProgress(
                        _LOGGER,
                        "inference_progress",
                        total=total,
                        unit="batches",
                        count_interval=100,
                        fields={"partition": "test", **context},
                    )
                inference_progress.update(completed)

            inference = deterministic_inference(
                model,
                test_loader,
                runtime=runtime,
                progress_callback=report_inference_progress,
            )
        thresholds = {key: float(value) for key, value in manifest["thresholds"].items()}
        probability_metrics = evaluate_probabilities(
            inference.targets,
            inference.probabilities,
            calibration_bins=config.evaluation.calibration_bins,
        )
        youden = evaluate_operating_point(
            inference.targets,
            inference.probabilities,
            threshold=thresholds["youden_j"],
        )
        target_sensitivity = evaluate_operating_point(
            inference.targets,
            inference.probabilities,
            threshold=thresholds["target_sensitivity"],
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
        document["image_evaluation"] = {
            "evaluation_run_id": evaluation_run_id,
            "source_training_run_id": training_run_id,
            "model_package_id": manifest["model_package_id"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "bundle_id": manifest["bundle_id"],
            "bundle_manifest_sha256": manifest["bundle_manifest_sha256"],
            "split_assignment_id": manifest["split_assignment_id"],
            "task": manifest["task"],
            "label_policy_version": manifest["label_policy_version"],
            "partition": "test",
            "thresholds_frozen_from": "validation",
            "source_authentication": authentication,
            "runtime": runtime.provenance(),
            "test_counts": {
                "sample_count": len(inference.targets),
                "positive_count": int((inference.targets == 1).sum()),
                "negative_count": int((inference.targets == 0).sum()),
            },
            "package_verification": True,
            "limitations": [
                "The challenge target is radiology-derived.",
                "Operating points were frozen on validation before test access.",
            ],
        }
        report_directory = (
            config.training.report_directory
            / config.dataset.registry_key
            / "runs"
            / evaluation_run_id
        )
        if report_directory.exists():
            raise FileExistsError(f"Image test report already exists: {report_directory}")
        report_stage = staging_directory(report_directory)
        report_published = False
        private_prediction_directory = (
            private_root_for_reports(config.training.report_directory)
            / "predictions"
            / config.dataset.registry_key
            / evaluation_run_id
        )
        private_predictions_published = False
        try:
            write_run_reports(
                report_stage,
                model_name=config.model.registry_key,
                targets=inference.targets,
                probabilities=inference.probabilities,
                document=document,
            )
            validate_report_set(report_stage)
            validate_public_reports(
                report_stage.iterdir(),
                forbidden_source_values={*inference.sample_ids, *inference.patient_ids},
            )
            publish_private_neural_predictions(
                private_root=private_root_for_reports(config.training.report_directory),
                dataset=config.dataset.registry_key,
                training_run_id=training_run_id,
                test_evaluation_run_id=evaluation_run_id,
                model_package_id=str(manifest["model_package_id"]),
                seed=config.training.seed,
                sample_ids=inference.sample_ids,
                patient_keys=inference.patient_ids,
                targets=inference.targets,
                logits=inference.logits,
                probabilities=inference.probabilities,
            )
            private_predictions_published = True
            metrics = mlflow_metrics(
                scope="test",
                document=document,
                latency_ms=None,
                model_size_mib=model_path.stat().st_size / (1024.0 * 1024.0),
            )
            mlflow.log_params(
                {
                    "bundle_manifest_sha256": image_data.bundle_manifest_sha256,
                    "source_inventory_file_sha256": authentication["source_inventory_file_sha256"],
                    "source_inventory_arrow_sha256": authentication[
                        "source_inventory_arrow_sha256"
                    ],
                    "source_authentication_policy_version": authentication["policy_version"],
                    **{
                        f"evaluation_runtime_{key}": value
                        for key, value in runtime.provenance().items()
                    },
                    "evaluation_cxr_cache_id": resolved_cache.identity.cache_id,
                    **{
                        f"evaluation_loader_{key}": (
                            value if value is not None else "not_applicable"
                        )
                        for key, value in loader_execution.provenance().items()
                        if not isinstance(value, dict)
                    },
                }
            )
            mlflow.log_metrics(metrics)
            publish_directory(report_stage, report_directory)
            report_published = True
            mlflow.set_tags(
                {
                    "split_assignment_id": manifest["split_assignment_id"],
                    "label_policy_version": manifest["label_policy_version"],
                    "source_authentication_policy": authentication["policy_version"],
                    "model_package_id": manifest["model_package_id"],
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                    "local_model_sha256": manifest["checkpoint_sha256"],
                    "report_directory": report_directory.as_posix(),
                    "comparison_publication": "completed-mlflow-run-ledger",
                    "threshold_youden_j": str(thresholds["youden_j"]),
                    "threshold_target_sensitivity": str(thresholds["target_sensitivity"]),
                }
            )
            mlflow.set_tag("run_complete", "true")
        except BaseException:
            if report_published and report_directory.exists():
                shutil.rmtree(report_directory)
            if private_predictions_published and private_prediction_directory.exists():
                shutil.rmtree(private_prediction_directory)
            raise
        finally:
            if report_stage.exists():
                shutil.rmtree(report_stage)
        log_event(_LOGGER, "publication_completed", artifact="test_report", **context)
    return ImageTestEvaluationResult(
        run_id=evaluation_run_id,
        training_run_id=training_run_id,
        artifact_directory=report_directory,
        private_prediction_directory=private_prediction_directory,
        average_precision=probability_metrics.average_precision,
    )


def verify_image_training_package(
    source_run,
    config: ExperimentConfig,
    manifest: dict[str, Any],
    *,
    training_run_id: str,
    evaluator_commit: str,
    evaluator_dirty: bool,
    evaluator_lock_hash: str,
) -> None:
    if config.model.modality != "image":
        raise ValueError("Neural package does not contain an image configuration")
    expected_package = config.training.model_directory / "runs" / training_run_id
    observed_package = Path(source_run.data.tags["local_model_path"]).parent.resolve()
    if observed_package != expected_package.resolve():
        raise ValueError("Image training run points outside its configured package")
    source = manifest["source_provenance"]
    checks = {
        "training_mlflow_run_id": training_run_id,
        "source_config_sha256": config.source_sha256,
        "semantic_config_sha256": image_semantic_config_sha256(config),
        "bundle_id": config.dataset.bundle_id,
        "task": config.dataset.task_id,
        "model": config.model.registry_key,
    }
    for field, expected in checks.items():
        if manifest[field] != expected:
            raise ValueError(f"Neural package {field} mismatch")
    expected_tags = {
        "dataset_bundle_id": manifest["bundle_id"],
        "split_assignment_id": manifest["split_assignment_id"],
        "task": manifest["task"],
        "model": manifest["model"],
        "seed": str(manifest["training_policy"]["seed"]),
        "git_commit": source["git_commit"],
        "git_dirty": str(source["git_dirty"]).lower(),
        "dependency_lock_sha256": source["dependency_lock_sha256"],
        "source_config_sha256": manifest["source_config_sha256"],
        "semantic_config_sha256": manifest["semantic_config_sha256"],
        "local_model_sha256": manifest["checkpoint_sha256"],
        "model_package_id": manifest["model_package_id"],
    }
    for field, expected in expected_tags.items():
        if source_run.data.tags.get(field) != expected:
            raise ValueError(f"Image training run tag {field} mismatch")
    if source_run.data.params.get("bundle_manifest_sha256") != manifest["bundle_manifest_sha256"]:
        raise ValueError("Image training run parameter bundle_manifest_sha256 mismatch")
    for policy in ("youden_j", "target_sensitivity"):
        observed = source_run.data.metrics.get(f"validation_{policy}_threshold")
        expected = manifest["thresholds"][policy]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int | float)
            or not math.isfinite(observed)
            or not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(f"Image training run threshold {policy} mismatch")
    numeric_checks = {
        "validation_average_precision": manifest["selection"]["validation_average_precision"],
        "model_size_mib": Path(source_run.data.tags["local_model_path"]).stat().st_size
        / (1024.0 * 1024.0),
    }
    for field, expected in numeric_checks.items():
        observed = source_run.data.metrics.get(field)
        if (
            not isinstance(observed, int | float)
            or isinstance(observed, bool)
            or not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(f"Image training run metric {field} mismatch")
    for field in ("selected_stage", "selected_epoch"):
        if source_run.data.params.get(field) != str(manifest["selection"][field]):
            raise ValueError(f"Image training run parameter {field} mismatch")
    if source["git_dirty"] or evaluator_dirty:
        raise ValueError("Formal image test evaluation requires clean training and evaluation")
    if evaluator_commit != source["git_commit"]:
        raise ValueError("Current Git commit does not match the neural package")
    if evaluator_lock_hash != source["dependency_lock_sha256"]:
        raise ValueError("Current dependency lock does not match the neural package")


def _verify_source_authentication(observed: dict[str, object], manifest: dict[str, Any]) -> None:
    expected = manifest["source_authentication"]
    if observed != expected:
        raise ValueError("Validated cache source authentication differs from the neural package")
