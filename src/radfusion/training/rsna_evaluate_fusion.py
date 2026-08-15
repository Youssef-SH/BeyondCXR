"""Evaluate one verified RSNA fusion package on its pinned test partition."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import mlflow
import numpy as np

from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.rsna_cxr_cache import ValidatedCxrCache
from radfusion.data.rsna_metadata_preprocess import SOURCE_FEATURES, transform_rsna_metadata
from radfusion.models.fusion_concat import (
    RsnaConcatFusionModel,
    RsnaCxrMetadataConcatModel,
    fusion_structured_input_conversion_contract,
)
from radfusion.training.config import (
    ExperimentConfig,
    require_runtime_seed,
    with_runtime,
)
from radfusion.training.device import resolve_device
from radfusion.training.execution import LoaderExecutionPolicy, one_shot_loader_policy
from radfusion.training.neural import build_evaluation_loader, deterministic_inference
from radfusion.training.rsna_datasets import (
    RsnaCachedFusionDataset,
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
from radfusion.training.rsna_fusion_source import resolve_source_cxr_package
from radfusion.training.rsna_registry import get_dataset, get_model
from radfusion.training.rsna_train_fusion import load_validated_rsna_fusion_preprocessor
from radfusion.training.rsna_train_metadata import mlflow_metrics
from radfusion.utils.mlflow_utils import (
    DEFAULT_TRACKING_URI,
    configure_mlflow,
    environment_provenance,
    serialize_modalities,
    tracked_run,
)
from radfusion.utils.operational_logging import get_operational_logger, log_event, timed_phase
from radfusion.utils.private_predictions import publish_prediction_evidence
from radfusion.utils.rsna_neural_publication import (
    load_validated_neural_checkpoint,
    strict_load_checkpoint,
)

_LOGGER = get_operational_logger(__name__)


def evaluate_fusion_model_package(
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
    """Verify one explicit fusion package before accessing its test partition."""
    manifest = validate_rsna_model_package(model_directory, model_package_id)
    package = Path(model_directory) / "packages" / model_package_id
    policy = validated_rsna_evaluation_policy(manifest, evaluation_config)
    config = with_runtime(
        evaluation_config,
        seed=int(manifest["training_policy"]["seed"]),
        model_directory=Path(model_directory),
        private_output_directory=private_output_directory,
        report_directory=report_directory,
    )
    if (
        manifest["family_id"] != "cxr_metadata_concat"
        or manifest["model_package_id"] != model_package_id
    ):
        raise ValueError("Requested package is not the expected fusion package")
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
        "source_package_id": manifest["source_package_id"],
        "git_commit": manifest["source_provenance"]["git_commit"],
        "dependency_lock_sha256": manifest["source_provenance"]["dependency_lock_sha256"],
        "package_kind": "model",
        "package_id": model_package_id,
        "run_complete": "false",
    }
    with tracked_run(
        run_name=f"{config.family.family_id}-test",
        tags=initial_tags,
        parameters={"package_id": model_package_id, **environment},
    ) as evaluation_run_id:
        model_path = package / "model.pt"
        resolve_source_cxr_package(str(manifest["source_package_id"]), config)
        preprocessor = load_validated_rsna_fusion_preprocessor(package, manifest)
        checkpoint = load_validated_neural_checkpoint(package, manifest)
        contract = manifest["structured_preprocessor_contract"]
        builder = cast(RsnaCxrMetadataConcatModel, get_model(config.family.family_id))
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
            family_id=config.family.family_id,
        )
        dataset = get_dataset(config.dataset.dataset_id)
        with timed_phase(
            _LOGGER,
            "dataset_loading",
            run_id=evaluation_run_id,
            family_id=config.family.family_id,
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
        neural = config.neural
        if neural is None or config.runtime.source_root is None:
            raise ValueError("Verified fusion package has an incomplete configuration")
        transform = StandardCxrTransform(
            training=False,
            policy_version=str(config.preprocessing["cxr_transform_policy"]),
            image_size=int(config.family.parameters["image_size"]),
            rotation_degrees=neural.rotation_degrees,
            translation_fraction=neural.translation_fraction,
            brightness_jitter=neural.brightness_jitter,
            contrast_jitter=neural.contrast_jitter,
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
            mixed_precision=neural.mixed_precision,
            pin_memory_policy=config.runtime.pin_memory_policy,
        )
        loader_execution = execution or one_shot_loader_policy(
            pin_memory=runtime.pin_memory_effective
        )
        loader = build_evaluation_loader(
            test_dataset,
            batch_size=neural.batch_size,
            runtime=runtime,
            execution=loader_execution,
        )
        model.to(runtime.device)
        inference = deterministic_inference(
            model,
            loader,
            runtime=runtime,
            input_keys=("image", "structured"),
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
        mlflow.log_metrics(
            mlflow_metrics(
                scope="test",
                document=result.manifest["claims"],
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


def _verify_test_lineage(data, manifest) -> None:
    if (
        data.lineage.bundle_id != manifest["bundle_id"]
        or data.bundle_manifest_sha256 != manifest["bundle_manifest_sha256"]
        or data.lineage.split_assignment_id != manifest["split_assignment_id"]
        or data.lineage.task_id != manifest["task_id"]
        or data.lineage.label_policy_version != manifest["label_policy_version"]
    ):
        raise ValueError("Fusion test bundle lineage differs from the package")
