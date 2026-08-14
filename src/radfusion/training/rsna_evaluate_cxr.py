"""Evaluate one verified CXR package on its pinned RSNA test partition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import mlflow
from torch import nn

from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.rsna_cxr_cache import ValidatedCxrCache
from radfusion.training.config import (
    ExperimentConfig,
    require_runtime_seed,
    with_runtime,
)
from radfusion.training.device import resolve_device
from radfusion.training.execution import LoaderExecutionPolicy, one_shot_loader_policy
from radfusion.training.neural import (
    build_evaluation_loader,
    deterministic_inference,
)
from radfusion.training.rsna_datasets import (
    RsnaCachedImageDataset,
    RsnaDataset,
    expected_rsna_cxr_cache_identity,
    prepare_rsna_cxr_cache,
)
from radfusion.training.rsna_evaluation_result import (
    CompletedRsnaEvaluation,
    publish_rsna_evaluation,
    validate_rsna_model_package,
    validated_rsna_evaluation_policy,
)
from radfusion.training.rsna_interfaces import RsnaCxrModelImplementation
from radfusion.training.rsna_registry import get_dataset, get_model
from radfusion.training.rsna_train_metadata import (
    mlflow_metrics,
)
from radfusion.utils.mlflow_utils import (
    DEFAULT_TRACKING_URI,
    configure_mlflow,
    environment_provenance,
    serialize_modalities,
    tracked_run,
)
from radfusion.utils.operational_logging import (
    CountProgress,
    get_operational_logger,
    log_event,
    timed_phase,
)
from radfusion.utils.private_predictions import publish_prediction_evidence
from radfusion.utils.rsna_neural_publication import (
    load_validated_neural_checkpoint,
    strict_load_checkpoint,
)

_LOGGER = get_operational_logger(__name__)


def evaluate_cxr_model_package(
    model_package_id: str,
    *,
    evaluation_config: ExperimentConfig,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    cache: ValidatedCxrCache | None = None,
    execution: LoaderExecutionPolicy | None = None,
    private_output_directory: str | Path | None = None,
    model_directory: str | Path = "models/rsna",
    report_directory: str | Path | None = None,
) -> CompletedRsnaEvaluation:
    """Verify one explicit CXR package before accessing its test partition."""
    manifest = validate_rsna_model_package(model_directory, model_package_id)
    package_directory = Path(model_directory) / "packages" / model_package_id
    policy = validated_rsna_evaluation_policy(manifest, evaluation_config)
    config = with_runtime(
        evaluation_config,
        seed=int(manifest["training_policy"]["seed"]),
        model_directory=Path(model_directory),
        private_output_directory=private_output_directory,
        report_directory=report_directory,
    )
    if manifest["family_id"] != "cxr_densenet" or manifest["model_package_id"] != model_package_id:
        raise ValueError("Requested package is not the expected CXR package")
    configure_mlflow(experiment_name=config.runtime.experiment_name, tracking_uri=tracking_uri)
    environment = environment_provenance()
    initial_tags = {
        "run_kind": "test_evaluation",
        "evaluation_scope": "test",
        "dataset_id": config.dataset.dataset_id,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": manifest["bundle_manifest_sha256"],
        "split_assignment_id": manifest["split_assignment_id"],
        "task_id": manifest["task_id"],
        "family_id": manifest["family_id"],
        "modalities": serialize_modalities(manifest["modalities"]),
        "seed": str(manifest["training_policy"]["seed"]),
        "config_source_sha256": config.config_source_sha256,
        "config_semantic_sha256": config.config_semantic_sha256,
        "git_commit": manifest["source_provenance"]["git_commit"],
        "dependency_lock_sha256": manifest["source_provenance"]["dependency_lock_sha256"],
        "package_kind": "model",
        "package_id": model_package_id,
        "run_complete": "false",
    }
    with tracked_run(
        run_name=f"{config.family.family_id}-test",
        tags=initial_tags,
        parameters={
            "package_id": model_package_id,
            **environment,
        },
    ) as evaluation_run_id:
        context = {"run_id": evaluation_run_id, "family_id": initial_tags["family_id"]}
        model_path = package_directory / "model.pt"
        checkpoint = load_validated_neural_checkpoint(package_directory, manifest)
        model_builder = cast(RsnaCxrModelImplementation, get_model(config.family.family_id))
        model = model_builder.build_architecture(config.family)
        if not isinstance(model, nn.Module):
            raise TypeError("Registered CXR model builder must return torch.nn.Module")
        strict_load_checkpoint(model, checkpoint)
        log_event(_LOGGER, "training_package_verified", **context)
        dataset_adapter = get_dataset(config.dataset.dataset_id)
        with timed_phase(_LOGGER, "dataset_loading", **context):
            cxr_data = dataset_adapter.load_cxr_test(
                config,
                expected_manifest_sha256=manifest["bundle_manifest_sha256"],
            )
        if (
            cxr_data.lineage.bundle_id != manifest["bundle_id"]
            or cxr_data.lineage.split_assignment_id != manifest["split_assignment_id"]
            or cxr_data.lineage.task_id != manifest["task_id"]
            or cxr_data.lineage.label_policy_version != manifest["label_policy_version"]
            or cxr_data.bundle_manifest_sha256 != manifest["bundle_manifest_sha256"]
        ):
            raise ValueError("Test bundle lineage differs from the neural package")
        neural = config.neural
        if neural is None or config.runtime.source_root is None:
            raise ValueError("Verified CXR package has an incomplete configuration")
        evaluation_transform = StandardCxrTransform(
            training=False,
            policy_version=str(config.preprocessing["cxr_transform_policy"]),
            image_size=int(config.family.parameters["image_size"]),
            rotation_degrees=neural.rotation_degrees,
            translation_fraction=neural.translation_fraction,
            brightness_jitter=neural.brightness_jitter,
            contrast_jitter=neural.contrast_jitter,
        )
        if evaluation_transform.contract() != manifest["evaluation_transform_contract"]:
            raise ValueError("Evaluation transform differs from the neural package")
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
        _verify_source_authentication(authentication, manifest)
        if manifest["runtime_provenance"]["cxr_cache_id"] != resolved_cache.identity.cache_id:
            raise ValueError("Validated cache identity differs from the CXR package")
        test_dataset = RsnaCachedImageDataset(
            cxr_data.test,
            cache=resolved_cache,
            expected_cache_identity=expected_cache_identity,
            partition="test",
            transform=evaluation_transform,
            training_seed=require_runtime_seed(config),
        )
        runtime = resolve_device(
            config.runtime.device,
            mixed_precision=neural.mixed_precision,
            pin_memory_policy=config.runtime.pin_memory_policy,
        )
        loader_execution = execution or one_shot_loader_policy(
            pin_memory=runtime.pin_memory_effective
        )
        test_loader = build_evaluation_loader(
            test_dataset,
            config=neural,
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
        evidence = publish_prediction_evidence(
            private_root=config.runtime.private_output_directory,
            dataset_id="rsna",
            model_package_id=model_package_id,
            task_id=manifest["task_id"],
            bundle_id=manifest["bundle_id"],
            split_assignment_id=manifest["split_assignment_id"],
            scope="test",
            sample_ids=inference.sample_ids,
            targets=inference.targets,
            logits=inference.logits,
        )
        result = publish_rsna_evaluation(
            report_root=config.runtime.report_directory,
            model_root=config.runtime.model_directory,
            private_root=config.runtime.private_output_directory,
            evidence=evidence,
            evaluation_policy=policy,
            forbidden_source_values=(*inference.sample_ids, *inference.patient_ids),
        )
        claims = result.manifest["claims"]
        mlflow.log_metrics(
            mlflow_metrics(
                scope="test",
                document=claims,
                latency_ms=None,
                model_size_mib=model_path.stat().st_size / (1024.0 * 1024.0),
            )
        )
        mlflow.log_params(
            {
                "bundle_manifest_sha256": cxr_data.bundle_manifest_sha256,
                "evaluation_cxr_cache_id": resolved_cache.identity.cache_id,
                **{
                    f"evaluation_runtime_{key}": value
                    for key, value in runtime.provenance().items()
                },
                **{
                    f"evaluation_loader_{key}": value
                    for key, value in loader_execution.provenance().items()
                },
            }
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
        model_package_id=model_package_id,
        mlflow_run_id=evaluation_run_id,
        artifact_directory=result.directory,
        private_prediction_directory=evidence.directory,
        average_precision=float(
            result.manifest["claims"]["probability_metrics"]["average_precision"]
        ),
    )


def _verify_source_authentication(observed: dict[str, object], manifest: dict[str, Any]) -> None:
    expected = manifest["source_authentication"]
    if observed != expected:
        raise ValueError("Validated cache source authentication differs from the neural package")
