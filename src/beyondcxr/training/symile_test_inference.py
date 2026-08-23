"""Dual-authorized reconstruction and raw Symile test prediction publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_schemas import LABEL_POLICY_VERSION, TASK_ID
from beyondcxr.models.symile_tabular import symile_tabular_logits
from beyondcxr.training.device import (
    ResolvedDevice,
    neural_inference_runtime_policy,
    resolve_device,
)
from beyondcxr.training.neural import (
    build_evaluation_loader,
    configure_neural_determinism,
    deterministic_inference,
)
from beyondcxr.training.symile_campaign_control import (
    ValidatedPretestFreeze,
    ValidatedTestOpenRecord,
    materialize_official_test,
    validated_pretest_freeze_manifest,
)
from beyondcxr.training.symile_families import (
    FINAL_PACKAGE_COUNT,
    FINAL_TABULAR_FAMILIES,
    SYMILE_ECG_GATED_FAMILY,
)
from beyondcxr.training.symile_final_packages import (
    ValidatedFinalPackage,
    load_final_lab_preprocessor,
    load_final_neural_model,
    load_final_package_config,
    load_final_tabular_model,
    validate_final_package,
)
from beyondcxr.training.symile_test_data import (
    FrozenSymileNeuralTestDataset,
    FrozenSymileTestCxrStore,
    FrozenSymileTestData,
    FrozenSymileTestEcgStore,
    HeldOutEvaluationProjection,
    test_laboratory_frame,
    validate_prediction_against_test_projection,
)
from beyondcxr.utils.private_predictions import (
    SYMILE_TEST_INFERENCE_POLICY,
    ValidatedPredictionEvidence,
    publish_prediction_evidence,
    validate_prediction_evidence,
)
from beyondcxr.utils.publication import is_publication_staging_directory


def publish_frozen_test_predictions(
    *,
    capability: ValidatedPretestFreeze,
    test_open_record: ValidatedTestOpenRecord,
    final_packages: Sequence[ValidatedFinalPackage],
    source_root: str | Path,
    manifest_root: str | Path,
    private_root: str | Path,
    runtime: ResolvedDevice,
) -> tuple[tuple[ValidatedPredictionEvidence, ...], FrozenSymileTestData]:
    """Validate all fourteen packages before the accessor may materialize test state."""
    packages = _validated_package_membership(capability, final_packages)
    _validate_neural_inference_runtime(capability, runtime)
    configure_neural_determinism()

    def evaluate() -> tuple[tuple[ValidatedPredictionEvidence, ...], FrozenSymileTestData]:
        data = FrozenSymileTestData(
            capability,
            test_open_record,
            load_final_package_config(packages[0]),
            manifest_root=manifest_root,
        )
        projection = data.evaluation_projection()
        expected = {package.package_id for package in packages}
        existing = validate_existing_test_predictions(
            private_root,
            capability=capability,
            expected_packages=expected,
            projection=projection,
        )
        predictions = _infer_all(
            capability,
            test_open_record,
            packages,
            data,
            source_root,
            private_root,
            runtime,
            existing,
        )
        return predictions, data

    return materialize_official_test(capability, test_open_record, evaluate)


def resolve_held_out_neural_runtime(
    packages: Sequence[ValidatedFinalPackage], device: str
) -> ResolvedDevice:
    if len(packages) != FINAL_PACKAGE_COUNT:
        raise ManifestBuildError("Neural runtime resolution requires fourteen final packages")
    validated = tuple(validate_final_package(package.directory) for package in packages)
    if len({package.package_id for package in validated}) != FINAL_PACKAGE_COUNT:
        raise ManifestBuildError("Final package membership contains duplicates")
    neural_configs = [
        config
        for package in validated
        if (config := load_final_package_config(package)).family_id not in FINAL_TABULAR_FAMILIES
    ]
    mixed_precision = {config.mixed_precision for config in neural_configs}
    if len(mixed_precision) != 1:
        raise ManifestBuildError("Final neural packages disagree on mixed-precision policy")
    return resolve_device(
        device,
        mixed_precision=mixed_precision.pop(),
        pin_memory_policy="auto",
    )


def _validate_neural_inference_runtime(
    capability: ValidatedPretestFreeze, runtime: ResolvedDevice
) -> None:
    frozen = validated_pretest_freeze_manifest(capability)["held_out_policy"][
        "neural_inference_runtime"
    ]
    if neural_inference_runtime_policy(runtime) != frozen:
        raise ManifestBuildError("Neural inference runtime differs from the pre-test freeze")


def _validated_package_membership(
    capability: ValidatedPretestFreeze,
    packages: Sequence[ValidatedFinalPackage],
) -> tuple[ValidatedFinalPackage, ...]:
    if len(packages) != FINAL_PACKAGE_COUNT:
        raise ManifestBuildError("Test inference requires exactly fourteen final packages")
    validated = tuple(validate_final_package(package.directory) for package in packages)
    if len({package.package_id for package in validated}) != FINAL_PACKAGE_COUNT:
        raise ManifestBuildError("Final package membership contains duplicates")
    frozen = validated_pretest_freeze_manifest(capability)["final_packages"]
    observed = [
        {"package_id": package.package_id, "manifest_sha256": package.manifest_sha256}
        for package in validated
    ]
    if observed != frozen:
        raise ManifestBuildError("Final package membership differs from validated freeze")
    return validated


def _infer_all(
    capability: ValidatedPretestFreeze,
    record: ValidatedTestOpenRecord,
    packages: Sequence[ValidatedFinalPackage],
    data: FrozenSymileTestData,
    source_root: str | Path,
    private_root: str | Path,
    runtime: ResolvedDevice,
    existing: Mapping[str, ValidatedPredictionEvidence],
) -> tuple[ValidatedPredictionEvidence, ...]:
    results = dict(existing)
    pending = [package for package in packages if package.package_id not in results]
    cxr_store = (
        FrozenSymileTestCxrStore(capability, record, data.bundle, source_root)
        if any("cxr" in load_final_package_config(package).modalities for package in pending)
        else None
    )
    ecg_store = None
    for package in pending:
        config = load_final_package_config(package)
        family = config.family_id
        if family in FINAL_TABULAR_FAMILIES:
            logits = symile_tabular_logits(
                load_final_tabular_model(package), test_laboratory_frame(data)
            )
        else:
            if cxr_store is None:
                raise ManifestBuildError("Final neural test dependencies are unavailable")
            model = load_final_neural_model(package).to(runtime.device)
            transformer = (
                load_final_lab_preprocessor(package) if "labs" in config.modalities else None
            )
            labs = (
                None if transformer is None else transformer.transform(test_laboratory_frame(data))
            )
            if family == SYMILE_ECG_GATED_FAMILY and ecg_store is None:
                ecg_store = FrozenSymileTestEcgStore(capability, record, data.bundle, source_root)
            dataset = FrozenSymileNeuralTestDataset(
                capability,
                record,
                data.frame,
                cxr_store=cxr_store,
                transform=_evaluation_transform(config),
                labs=labs,
                ecg_store=ecg_store if family == SYMILE_ECG_GATED_FAMILY else None,
            )
            logits = deterministic_inference(
                model,
                build_evaluation_loader(dataset, batch_size=config.batch_size, runtime=runtime),
                runtime=runtime,
                input_keys=_input_keys(family),
            ).logits
        results[package.package_id] = _publish(
            package.package_id, data, logits, private_root, capability.freeze_id
        )
    if len(results) != FINAL_PACKAGE_COUNT:
        raise ManifestBuildError("Test inference did not produce fourteen prediction objects")
    return tuple(results[package.package_id] for package in packages)


def validate_existing_test_predictions(
    private_root: str | Path,
    *,
    capability: ValidatedPretestFreeze,
    expected_packages: set[str],
    projection: HeldOutEvaluationProjection,
) -> dict[str, ValidatedPredictionEvidence]:
    root = Path(private_root) / "predictions" / "symile" / "test"
    if not root.exists():
        return {}
    if root.is_symlink() or not root.is_dir():
        raise ManifestBuildError("Test prediction root is invalid")
    found = {}
    for directory in root.iterdir():
        if is_publication_staging_directory(directory):
            continue
        evidence = validate_prediction_evidence(directory)
        validate_prediction_against_test_projection(
            evidence,
            capability=capability,
            projection=projection,
            expected_package_ids=tuple(expected_packages),
        )
        package_id = str(evidence.manifest["model_package_id"])
        if package_id not in expected_packages or package_id in found:
            raise ManifestBuildError("Persisted test prediction membership is invalid")
        found[package_id] = evidence
    return found


def _publish(
    package_id: str,
    data: FrozenSymileTestData,
    logits: np.ndarray,
    private_root: str | Path,
    freeze_id: str,
) -> ValidatedPredictionEvidence:
    return publish_prediction_evidence(
        private_root=private_root,
        dataset_id="symile",
        model_package_id=package_id,
        task_id=TASK_ID,
        bundle_id=data.bundle.bundle_id,
        split_assignment_id=data.split_assignment_id,
        scope="test",
        sample_ids=data.frame["sample_id"].astype(str).tolist(),
        targets=data.frame["target"].to_numpy(dtype=np.int8),
        logits=logits,
        label_policy_version=LABEL_POLICY_VERSION,
        inference_policy=SYMILE_TEST_INFERENCE_POLICY,
        authorized_by_pretest_freeze_id=freeze_id,
    )


def _evaluation_transform(config):
    return StandardCxrTransform(
        training=False,
        policy_version=str(config.preprocessing["cxr_transform_policy"]),
        image_size=int(config.family_parameters["image_size"]),
        rotation_degrees=float(config.augmentation["rotation_degrees"]),
        translation_fraction=float(config.augmentation["translation_fraction"]),
        brightness_jitter=float(config.augmentation["brightness_jitter"]),
        contrast_jitter=float(config.augmentation["contrast_jitter"]),
    )


def _input_keys(family: str) -> tuple[str, ...]:
    if family == "cxr_densenet":
        return ("image",)
    if family == SYMILE_ECG_GATED_FAMILY:
        return ("image", "structured", "ecg")
    return ("image", "structured")
