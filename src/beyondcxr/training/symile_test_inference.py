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
)
from beyondcxr.training.neural import (
    build_evaluation_loader,
    configure_neural_determinism,
    deterministic_inference,
)
from beyondcxr.training.symile_campaign_control import (
    ValidatedHeldOutCapability,
    ValidatedPretestFreeze,
    ValidatedSymileExecution,
    ValidatedTestOpenEvidence,
    materialize_official_test,
    require_validated_held_out_capability,
    require_validated_test_open_evidence,
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
from beyondcxr.utils.operational_logging import get_operational_logger, timed_phase
from beyondcxr.utils.private_predictions import (
    SYMILE_TEST_INFERENCE_POLICY,
    ValidatedPredictionEvidence,
    _publish_held_out_prediction_evidence,
    authorize_symile_held_out_prediction_publication,
    validate_prediction_evidence,
)
from beyondcxr.utils.publication import is_publication_staging_directory


def publish_frozen_test_predictions(
    *,
    capability: ValidatedHeldOutCapability,
    final_packages: Sequence[ValidatedFinalPackage],
    source_root: str | Path,
    manifest_root: str | Path,
    private_root: str | Path,
    runtime: ResolvedDevice,
) -> tuple[tuple[ValidatedPredictionEvidence, ...], FrozenSymileTestData]:
    """Validate all fourteen packages before the accessor may materialize test state."""
    packages = _validated_package_membership(capability.freeze, final_packages)
    _validate_neural_inference_runtime(capability, runtime)
    configure_neural_determinism()

    def evaluate() -> tuple[tuple[ValidatedPredictionEvidence, ...], FrozenSymileTestData]:
        data = FrozenSymileTestData(
            capability,
            load_final_package_config(packages[0]),
            manifest_root=manifest_root,
            source_root=source_root,
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
            packages,
            data,
            source_root,
            private_root,
            runtime,
            existing,
        )
        return predictions, data

    return materialize_official_test(capability, evaluate)


def _validate_neural_inference_runtime(
    capability: ValidatedHeldOutCapability, runtime: ResolvedDevice
) -> None:
    frozen = capability.execution.manifest["neural_runtime"]
    if neural_inference_runtime_policy(runtime) != frozen:
        raise ManifestBuildError("Neural inference runtime differs from the formal execution")


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
        {
            "package_id": package.package_id,
            "family_id": package.manifest["input"]["family"]["family_id"],
            "seed": package.manifest["seed_policy"],
        }
        for package in validated
    ]
    if observed != frozen:
        raise ManifestBuildError("Final package membership differs from validated freeze")
    return validated


def _infer_all(
    capability: ValidatedHeldOutCapability,
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
        FrozenSymileTestCxrStore(capability, data.bundle, source_root)
        if any("cxr" in load_final_package_config(package).modalities for package in pending)
        else None
    )
    ecg_store = None
    for package in pending:
        config = load_final_package_config(package)
        family = config.family_id
        with timed_phase(
            get_operational_logger(__name__),
            "held_out_member",
            package_id=package.package_id,
            family_id=family,
            seed=package.manifest["seed_policy"],
        ):
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
                    None
                    if transformer is None
                    else transformer.transform(test_laboratory_frame(data))
                )
                if family == SYMILE_ECG_GATED_FAMILY and ecg_store is None:
                    ecg_store = FrozenSymileTestEcgStore(capability, data.bundle, source_root)
                dataset = FrozenSymileNeuralTestDataset(
                    capability,
                    data,
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
                package.package_id, data, logits, private_root, capability
            )
    if len(results) != FINAL_PACKAGE_COUNT:
        raise ManifestBuildError("Test inference did not produce fourteen prediction objects")
    return tuple(results[package.package_id] for package in packages)


def validate_existing_test_predictions(
    private_root: str | Path,
    *,
    capability: ValidatedHeldOutCapability,
    expected_packages: set[str],
    projection: HeldOutEvaluationProjection,
) -> dict[str, ValidatedPredictionEvidence]:
    capability = require_validated_held_out_capability(capability)
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
        _validate_live_publication_authority(evidence, capability)
        validate_prediction_against_test_projection(
            evidence,
            capability=capability.freeze,
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
    capability: ValidatedHeldOutCapability,
) -> ValidatedPredictionEvidence:
    capability = require_validated_held_out_capability(capability)
    frozen_packages = {
        str(item["package_id"])
        for item in validated_pretest_freeze_manifest(capability.freeze)["final_packages"]
    }
    if package_id not in frozen_packages:
        raise ManifestBuildError("Test prediction package is not a frozen final package")
    projection = data.evaluation_projection()
    publication_authority = authorize_symile_held_out_prediction_publication(
        capability,
        model_package_id=package_id,
        projection=projection,
    )
    evidence = _publish_held_out_prediction_evidence(
        publication_authority=publication_authority,
        private_root=private_root,
        dataset_id="symile",
        model_package_id=package_id,
        task_id=TASK_ID,
        bundle_id=data.bundle.bundle_id,
        scope="test",
        sample_ids=data.frame["sample_id"].astype(str).tolist(),
        targets=data.frame["target"].to_numpy(dtype=np.int8),
        logits=logits,
        label_policy_version=LABEL_POLICY_VERSION,
        inference_policy=SYMILE_TEST_INFERENCE_POLICY,
        authorized_by_freeze_id=capability.freeze.freeze_id,
    )
    _validate_live_publication_authority(evidence, capability)
    validate_prediction_against_test_projection(
        evidence,
        capability=capability.freeze,
        projection=projection,
        expected_package_ids=tuple(sorted(frozen_packages)),
    )
    return evidence


def _validate_live_publication_authority(
    evidence: ValidatedPredictionEvidence,
    capability: ValidatedHeldOutCapability,
) -> None:
    authority = require_validated_held_out_capability(capability)
    expected = {
        "authority_kind": "symile-opened-campaign-v1",
        "execution_id": authority.execution.execution_id,
    }
    if evidence.manifest.get("held_out_publication") != expected:
        raise ManifestBuildError("Test prediction lacks its official-opening authority")


def validate_restored_test_prediction_publication(
    evidence: ValidatedPredictionEvidence,
    *,
    freeze: ValidatedPretestFreeze,
    execution: ValidatedSymileExecution,
    opening: ValidatedTestOpenEvidence,
) -> None:
    """Bind restored prediction publication to preserved historical opening evidence."""
    historical_opening = require_validated_test_open_evidence(opening, freeze=freeze)
    if historical_opening.execution_id != execution.execution_id:
        raise ManifestBuildError("Restored prediction opening differs from its execution")
    expected = {
        "authority_kind": "symile-opened-campaign-v1",
        "execution_id": execution.execution_id,
    }
    if evidence.manifest.get("held_out_publication") != expected:
        raise ManifestBuildError("Restored prediction lacks its historical opening authority")


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
