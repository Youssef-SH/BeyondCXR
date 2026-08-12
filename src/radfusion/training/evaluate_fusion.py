"""Evaluate one verified RSNA fusion package on its pinned test partition."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlflow
import numpy as np

from radfusion.data.cxr_cache import ValidatedCxrCache
from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.tabular_preprocess import SOURCE_FEATURES, transform_rsna_metadata
from radfusion.evaluation.metrics import evaluate_operating_point, evaluate_probabilities
from radfusion.models.fusion_concat import (
    FusionConcatModel,
    RsnaConcatFusionModel,
    fusion_structured_input_conversion_contract,
)
from radfusion.training.config import (
    load_experiment_config,
    require_runtime_seed,
    with_runtime,
)
from radfusion.training.datasets import (
    RsnaCachedFusionDataset,
    RsnaDataset,
    expected_rsna_cxr_cache_identity,
    prepare_rsna_cxr_cache,
)
from radfusion.training.device import resolve_device
from radfusion.training.execution import LoaderExecutionPolicy, one_shot_loader_policy
from radfusion.training.fusion_source import resolve_source_cxr_training_run
from radfusion.training.neural import build_evaluation_loader, deterministic_inference
from radfusion.training.registry import get_dataset, get_model
from radfusion.training.train_fusion import load_validated_rsna_fusion_preprocessor
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
    CONFIG_FILENAME,
    load_validated_neural_checkpoint,
    strict_load_checkpoint,
    validate_neural_package_metadata,
)
from radfusion.utils.operational_logging import get_operational_logger, log_event, timed_phase
from radfusion.utils.privacy import validate_public_reports
from radfusion.utils.private_predictions import (
    publish_private_neural_predictions,
)
from radfusion.utils.publication import publish_directory, staging_directory

_LOGGER = get_operational_logger(__name__)


@dataclass(frozen=True)
class FusionTestEvaluationResult:
    """Outputs from one completed explicit fusion test-evaluation run."""

    run_id: str
    training_run_id: str
    artifact_directory: Path
    private_prediction_directory: Path
    average_precision: float


def evaluate_fusion_training_run(
    training_run_id: str,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    cache: ValidatedCxrCache | None = None,
    execution: LoaderExecutionPolicy | None = None,
    private_output_directory: str | Path | None = None,
) -> FusionTestEvaluationResult:
    """Verify one explicit fusion package before accessing its test partition."""
    client = configure_mlflow(tracking_uri=tracking_uri)
    training_run = client.get_run(training_run_id)
    experiment_name = training_run.data.tags.get("experiment_name", "")
    if not experiment_name:
        raise ValueError("Source fusion training run has no experiment identity")
    mlflow.set_experiment(experiment_id=training_run.info.experiment_id)
    commit, dirty = git_revision()
    lock_hash = uv_lock_sha256()
    environment = environment_provenance()
    initial_tags = {
        "run_kind": "test_evaluation",
        "evaluation_scope": "test",
        "source_training_run_id": training_run_id,
        "experiment_name": experiment_name,
        "dataset": training_run.data.tags.get("dataset", ""),
        "dataset_bundle_id": training_run.data.tags.get("dataset_bundle_id", ""),
        "task": training_run.data.tags.get("task", ""),
        "modality": "fusion",
        "model": training_run.data.tags.get("model", ""),
        "seed": training_run.data.tags.get("seed", ""),
        "git_commit": commit,
        "git_dirty": str(dirty).lower(),
        "dependency_lock_sha256": lock_hash,
        "run_complete": "false",
    }
    with tracked_run(
        run_name=f"{experiment_name}-test",
        tags=initial_tags,
        parameters={"source_training_run_id": training_run_id, **environment},
    ) as evaluation_run_id:
        if (
            training_run.info.status != "FINISHED"
            or training_run.data.tags.get("run_kind") != "training"
            or training_run.data.tags.get("evaluation_scope") != "validation"
            or training_run.data.tags.get("modality") != "fusion"
            or training_run.data.tags.get("run_complete") != "true"
        ):
            raise ValueError("Source fusion training run is not complete")
        model_path = Path(training_run.data.tags["local_model_path"])
        package = model_path.parent
        manifest = validate_neural_package_metadata(package)
        training_report = Path(training_run.data.tags["report_directory"])
        config = with_runtime(
            load_experiment_config(package / CONFIG_FILENAME),
            seed=int(manifest["training_policy"]["seed"]),
            model_directory=package.parent.parent,
            report_directory=training_report.parents[2],
            private_output_directory=private_output_directory,
        )
        _verify_fusion_training_package(
            training_run,
            config,
            manifest,
            training_run_id=training_run_id,
            current_commit=commit,
            current_dirty=dirty,
            current_lock_hash=lock_hash,
        )
        source = resolve_source_cxr_training_run(
            client,
            str(manifest["source_cxr_lineage"]["training_run_id"]),
            config,
            current_git_commit=commit,
            current_git_dirty=dirty,
            current_dependency_lock_sha256=lock_hash,
        )
        if source.lineage.as_dict() != manifest["source_cxr_lineage"]:
            raise ValueError("Fusion package source CXR lineage cannot be reproduced")
        preprocessor = load_validated_rsna_fusion_preprocessor(package, manifest)
        checkpoint = load_validated_neural_checkpoint(package, manifest)
        contract = manifest["structured_preprocessor_contract"]
        builder = cast(FusionConcatModel, get_model(config.family.family_id))
        model = builder.build(
            config.family,
            structured_dimension=int(contract["transformed_dimension"]),
            weights=None,
        )
        if not isinstance(model, RsnaConcatFusionModel):
            raise TypeError("Registered fusion builder returned an invalid model")
        strict_load_checkpoint(model, checkpoint)
        log_event(
            _LOGGER,
            "training_package_verified",
            run_id=evaluation_run_id,
            model=config.family.family_id,
        )
        dataset = get_dataset(config.dataset.dataset_id)
        with timed_phase(
            _LOGGER,
            "dataset_loading",
            run_id=evaluation_run_id,
            model=config.family.family_id,
        ):
            data = dataset.load_fusion_test(
                config,
                expected_manifest_sha256=str(manifest["bundle_manifest_sha256"]),
            )
        _verify_test_lineage(data, manifest)
        transformed = transform_rsna_metadata(
            preprocessor,
            data.test.loc[:, SOURCE_FEATURES],
        )
        if tuple(transformed.columns) != tuple(contract["transformed_feature_names"]):
            raise ValueError("Fusion test transformed feature order differs from the package")
        matrix = np.ascontiguousarray(transformed.to_numpy(dtype=np.float64))
        if manifest["structured_input_conversion"] != fusion_structured_input_conversion_contract():
            raise ValueError("Fusion package has an unsupported structured conversion")
        image = config.neural
        if image is None or config.runtime.source_root is None:
            raise ValueError("Verified fusion package has an incomplete configuration")
        transform = StandardCxrTransform(
            training=False,
            policy_version=str(config.preprocessing["cxr_transform_policy"]),
            image_size=int(config.family.parameters["image_size"]),
            rotation_degrees=image.rotation_degrees,
            translation_fraction=image.translation_fraction,
            brightness_jitter=image.brightness_jitter,
            contrast_jitter=image.contrast_jitter,
        )
        if transform.contract() != manifest["evaluation_transform_contract"]:
            raise ValueError("Fusion evaluation transform differs from the package")
        resolved_cache = cache or prepare_rsna_cxr_cache(
            cast(RsnaDataset, dataset), config, transform
        )
        expected_cache_identity = expected_rsna_cxr_cache_identity(
            lineage=data.lineage,
            bundle_manifest_sha256=data.bundle_manifest_sha256,
            source_inventory=data.source_inventory,
            transform=transform,
        )
        if resolved_cache.source_authentication.as_dict() != manifest["source_authentication"]:
            raise ValueError("Validated cache source authentication differs from fusion package")
        if manifest["runtime_provenance"]["cxr_cache_id"] != resolved_cache.identity.cache_id:
            raise ValueError("Validated cache identity differs from the fusion package")
        test_dataset = RsnaCachedFusionDataset(
            data.test,
            matrix,
            structured_sample_ids=tuple(data.test["sample_id"].astype(str)),
            cache=resolved_cache,
            expected_cache_identity=expected_cache_identity,
            partition="test",
            transform=transform,
            training_seed=require_runtime_seed(config),
        )
        runtime = resolve_device(
            config.runtime.device,
            mixed_precision=image.mixed_precision,
            pin_memory_policy=config.runtime.pin_memory_policy,
        )
        loader_execution = execution or one_shot_loader_policy(
            pin_memory=runtime.pin_memory_effective
        )
        loader = build_evaluation_loader(
            test_dataset, config=image, runtime=runtime, execution=loader_execution
        )
        model.to(runtime.device)
        inference = deterministic_inference(
            model,
            loader,
            runtime=runtime,
            input_keys=("image", "structured"),
        )
        thresholds = {key: float(value) for key, value in manifest["thresholds"].items()}
        probability = evaluate_probabilities(
            inference.targets,
            inference.probabilities,
            calibration_bins=config.evaluation.calibration_bins,
        )
        youden = evaluate_operating_point(
            inference.targets,
            inference.probabilities,
            threshold=thresholds["youden_j"],
        )
        sensitivity = evaluate_operating_point(
            inference.targets,
            inference.probabilities,
            threshold=thresholds["target_sensitivity"],
        )
        document = metrics_document(
            scope="test",
            calibration_bins=config.evaluation.calibration_bins,
            sensitivity_target=config.evaluation.sensitivity_target,
            thresholds=thresholds,
            probability=probability,
            youden=youden,
            target_sensitivity=sensitivity,
        )
        report_directory = (
            config.runtime.report_directory / config.dataset.dataset_id / "runs" / evaluation_run_id
        )
        if report_directory.exists():
            raise FileExistsError(f"Fusion test report already exists: {report_directory}")
        stage = staging_directory(report_directory)
        published = False
        private_prediction_directory = (
            config.runtime.private_output_directory
            / "predictions"
            / config.dataset.dataset_id
            / evaluation_run_id
        )
        private_predictions_published = False
        try:
            write_run_reports(
                stage,
                model_name=config.family.family_id,
                targets=inference.targets,
                probabilities=inference.probabilities,
                document=document,
            )
            validate_report_set(stage)
            validate_public_reports(
                stage.iterdir(),
                forbidden_source_values={*inference.sample_ids, *inference.patient_ids},
            )
            publish_private_neural_predictions(
                private_root=config.runtime.private_output_directory,
                dataset=config.dataset.dataset_id,
                training_run_id=training_run_id,
                test_evaluation_run_id=evaluation_run_id,
                model_package_id=str(manifest["model_package_id"]),
                seed=require_runtime_seed(config),
                sample_ids=inference.sample_ids,
                patient_keys=inference.patient_ids,
                targets=inference.targets,
                logits=inference.logits,
                probabilities=inference.probabilities,
            )
            private_predictions_published = True
            mlflow.log_metrics(
                mlflow_metrics(
                    scope="test",
                    document=document,
                    latency_ms=None,
                    model_size_mib=model_path.stat().st_size / (1024.0 * 1024.0),
                )
            )
            mlflow.log_params(
                {
                    "bundle_manifest_sha256": data.bundle_manifest_sha256,
                    "evaluation_cxr_cache_id": resolved_cache.identity.cache_id,
                    **{
                        f"evaluation_runtime_{key}": value
                        for key, value in runtime.provenance().items()
                    },
                    **{
                        f"evaluation_loader_{key}": (
                            value if value is not None else "not_applicable"
                        )
                        for key, value in loader_execution.provenance().items()
                        if not isinstance(value, dict)
                    },
                }
            )
            publish_directory(stage, report_directory)
            published = True
            mlflow.set_tags(
                {
                    "split_assignment_id": manifest["split_assignment_id"],
                    "label_policy_version": manifest["label_policy_version"],
                    "model_package_id": manifest["model_package_id"],
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                    "local_model_sha256": manifest["checkpoint_sha256"],
                    "source_cxr_training_run_id": manifest["source_cxr_lineage"]["training_run_id"],
                    "source_cxr_model_package_id": manifest["source_cxr_lineage"][
                        "model_package_id"
                    ],
                    "source_cxr_checkpoint_sha256": manifest["source_cxr_lineage"][
                        "checkpoint_sha256"
                    ],
                    "report_directory": report_directory.as_posix(),
                    "threshold_youden_j": str(thresholds["youden_j"]),
                    "threshold_target_sensitivity": str(thresholds["target_sensitivity"]),
                }
            )
            mlflow.set_tag("run_complete", "true")
        except BaseException:
            if published and report_directory.exists():
                shutil.rmtree(report_directory)
            if private_predictions_published and private_prediction_directory.exists():
                shutil.rmtree(private_prediction_directory)
            raise
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return FusionTestEvaluationResult(
        run_id=evaluation_run_id,
        training_run_id=training_run_id,
        artifact_directory=report_directory,
        private_prediction_directory=private_prediction_directory,
        average_precision=probability.average_precision,
    )


def _verify_fusion_training_package(
    run,
    config,
    manifest,
    *,
    training_run_id: str,
    current_commit: str,
    current_dirty: bool,
    current_lock_hash: str,
) -> None:
    if config.family.family_id != "cxr_metadata_concat" or config.neural is None:
        raise ValueError("Fusion package does not archive a fusion configuration")
    source = manifest["source_provenance"]
    expected = {
        "training_mlflow_run_id": training_run_id,
        "config_semantic_sha256": config.config_semantic_sha256,
        "bundle_id": config.dataset.bundle_id,
        "task": config.task.task_id,
        "model": config.family.family_id,
    }
    if any(manifest[field] != value for field, value in expected.items()):
        raise ValueError("Fusion package identity differs from its archived configuration")
    expected_tags = {
        "model_package_id": manifest["model_package_id"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "local_model_sha256": manifest["checkpoint_sha256"],
        "dataset_bundle_id": manifest["bundle_id"],
        "split_assignment_id": manifest["split_assignment_id"],
        "label_policy_version": manifest["label_policy_version"],
        "seed": str(manifest["training_policy"]["seed"]),
        "source_cxr_training_run_id": manifest["source_cxr_lineage"]["training_run_id"],
        "source_cxr_model_package_id": manifest["source_cxr_lineage"]["model_package_id"],
    }
    if any(run.data.tags.get(field) != value for field, value in expected_tags.items()):
        raise ValueError("Fusion training run tags differ from its package")
    if run.data.params.get("bundle_manifest_sha256") != manifest["bundle_manifest_sha256"]:
        raise ValueError("Fusion training run bundle-manifest identity differs from its package")
    if source["git_dirty"] or current_dirty:
        raise ValueError("Formal fusion evaluation requires clean source and evaluation states")
    if (
        current_commit != source["git_commit"]
        or current_lock_hash != source["dependency_lock_sha256"]
    ):
        raise ValueError("Current source revision or dependency lock differs from fusion package")
    for policy in ("youden_j", "target_sensitivity"):
        observed = run.data.metrics.get(f"validation_{policy}_threshold")
        expected_threshold = manifest["thresholds"][policy]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int | float)
            or not math.isfinite(observed)
            or not math.isclose(observed, expected_threshold, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(f"Fusion training threshold {policy} differs from its package")


def _verify_test_lineage(data, manifest) -> None:
    if (
        data.lineage.bundle_id != manifest["bundle_id"]
        or data.bundle_manifest_sha256 != manifest["bundle_manifest_sha256"]
        or data.lineage.split_assignment_id != manifest["split_assignment_id"]
        or data.lineage.task_id != manifest["task"]
        or data.lineage.label_policy_version != manifest["label_policy_version"]
    ):
        raise ValueError("Fusion test bundle lineage differs from the package")
