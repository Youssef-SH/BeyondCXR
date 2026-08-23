"""Run the single safety-coupled Symile core and ECG-extension scientific campaign."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import mlflow

from beyondcxr.data.bundle_contract import BUNDLES_DIRECTORY
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_cv import CV_DIRECTORY
from beyondcxr.data.symile_preprocess import LAB_OBSERVED_COLUMNS
from beyondcxr.training.config import load_symile_development_config, with_runtime
from beyondcxr.training.device import ResolvedDevice, neural_inference_runtime_policy
from beyondcxr.training.symile_analysis import analyze_symile_development
from beyondcxr.training.symile_campaign_control import (
    PRETEST_FREEZE_PREFIX,
    ValidatedGlobalResult,
    ValidatedPretestFreeze,
    ValidatedTestOpenRecord,
    create_or_validate_test_open_record,
    publish_error_review,
    publish_global_result,
    publish_pretest_freeze,
    validate_error_review,
    validate_global_result,
    validate_pretest_freeze,
)
from beyondcxr.training.symile_data import load_symile_development_cohort
from beyondcxr.training.symile_development import run_symile_development
from beyondcxr.training.symile_ecg_extension_result import (
    ValidatedEcgExtensionResult,
    publish_ecg_extension_result,
    publish_focused_subgroup_derivative,
    validate_ecg_extension_result,
    validate_focused_subgroup_derivative,
)
from beyondcxr.training.symile_export import (
    SymileExportMember,
    export_and_verify,
    validate_export_paths,
)
from beyondcxr.training.symile_families import (
    FINAL_NEURAL_FAMILIES,
    FINAL_NEURAL_MEMBER_SEEDS,
    FINAL_PACKAGE_COUNT,
    FINAL_TABULAR_FAMILIES,
    FINAL_TABULAR_SEED,
    SYMILE_CORE_DEVELOPMENT_FAMILIES,
    SYMILE_ECG_GATED_FAMILY,
)
from beyondcxr.training.symile_final_packages import (
    ValidatedFinalPackage,
    load_final_package_config,
    validate_final_package,
)
from beyondcxr.training.symile_final_training import (
    fit_final_neural_member,
    fit_final_tabular_family,
)
from beyondcxr.training.symile_statistics import focused_development_subgroups
from beyondcxr.training.symile_test_data import (
    FrozenSymileTestData,
    validate_prediction_against_test_projection,
)
from beyondcxr.training.symile_test_inference import (
    publish_frozen_test_predictions,
    resolve_held_out_neural_runtime,
)
from beyondcxr.utils.mlflow_utils import (
    discover_repository_root,
    environment_provenance,
    git_revision,
    log_source_config,
    serialize_modalities,
    tracked_run,
    uv_lock_sha256,
)
from beyondcxr.utils.private_predictions import (
    ValidatedPredictionEvidence,
    validate_prediction_evidence,
)
from beyondcxr.utils.publication import validate_path_component
from beyondcxr.utils.symile_publication import (
    validate_analysis_result,
    validate_development_result,
    validated_development_repeat_oof,
)

_CONFIGS = {
    family: Path("configs") / f"symile_{family}.yaml"
    for family in (*SYMILE_CORE_DEVELOPMENT_FAMILIES, SYMILE_ECG_GATED_FAMILY)
}


def run_symile_campaign(
    *,
    source_root: str | Path,
    device: str = "auto",
    workers: int = 2,
    backup_root: str | Path,
) -> dict[str, str]:
    """Execute the frozen ordered lifecycle; no internal stage is a public command."""
    repository_root = discover_repository_root()
    manifest_root = repository_root / "data" / "manifests"
    model_root = repository_root / "models" / "symile"
    report_root = repository_root / "reports" / "symile"
    private_root = repository_root / "private"
    export_root = repository_root / "outbox"
    _validate_canonical_formal_layout(repository_root)
    source_root = Path(source_root).resolve()
    backup_root = _repository_path(repository_root, backup_root)
    if backup_root.resolve().is_relative_to(repository_root.resolve()):
        raise ManifestBuildError("Symile backup must reside outside the repository root")
    if any(
        root.resolve().is_relative_to(backup_root.resolve())
        for root in (source_root, repository_root)
    ):
        raise ManifestBuildError("Symile backup must not contain campaign source state")
    validate_export_paths(
        sources=(
            source_root,
            manifest_root,
            model_root,
            report_root,
            private_root,
            *(repository_root / name for name in ("data", "models", "reports", "private")),
        ),
        export_root=export_root,
        backup_root=backup_root,
    )
    tracking_uri = f"sqlite:///{repository_root / 'mlflow.db'}"
    commit, dirty = git_revision(repository_root)
    if dirty:
        raise ManifestBuildError("Formal Symile campaign requires a clean Git worktree")
    lock_hash = uv_lock_sha256(repository_root / "uv.lock")
    configs = {
        family: with_runtime(
            load_symile_development_config(repository_root / path),
            manifest_directory=manifest_root,
            source_root=source_root,
            model_directory=Path(model_root) / "development",
            report_directory=Path(report_root) / "development",
            private_output_directory=private_root,
            device=device,
            num_workers=workers,
        )
        for family, path in _CONFIGS.items()
    }
    control_root = Path(private_root) / "control" / "symile"
    test_open_path = control_root / "test-open.json"
    if test_open_path.exists() or test_open_path.is_symlink():
        freeze, record, ecg_extension, packages = _resolve_opened_pretest_state(
            control_root=control_root,
            model_root=model_root,
            report_root=report_root,
            private_root=private_root,
            manifest_root=manifest_root,
            science_git_commit=commit,
            dependency_lock_sha256=lock_hash,
        )
        runtime = resolve_held_out_neural_runtime(packages, device)
        return _complete_opened_campaign(
            freeze=freeze,
            record=record,
            ecg_extension=ecg_extension,
            packages=packages,
            source_root=source_root,
            manifest_root=manifest_root,
            model_root=model_root,
            report_root=report_root,
            private_root=private_root,
            export_root=export_root,
            backup_root=backup_root,
            runtime=runtime,
        )
    developments = {}
    for family in SYMILE_CORE_DEVELOPMENT_FAMILIES:
        source_id = (
            developments["cxr_densenet"].manifest["development_id"]
            if family in {"cxr_labs_concat", "cxr_labs_gated", "cxr_labs_gated_no_observedness"}
            else None
        )
        developments[family] = run_symile_development(
            configs[family],
            source_cxr_development_id=source_id,
            tracking_uri=tracking_uri,
            repository_root=repository_root,
        )
    _, analysis_directory = analyze_symile_development(
        [
            developments[family].manifest["development_id"]
            for family in SYMILE_CORE_DEVELOPMENT_FAMILIES
        ],
        report_root=Path(report_root) / "development",
        model_root=Path(model_root) / "development",
        prediction_root=private_root,
        manifest_root=manifest_root,
    )
    developments[SYMILE_ECG_GATED_FAMILY] = run_symile_development(
        configs[SYMILE_ECG_GATED_FAMILY],
        source_cxr_development_id=developments["cxr_densenet"].manifest["development_id"],
        tracking_uri=tracking_uri,
        repository_root=repository_root,
    )
    primary_repeat_oof = validated_development_repeat_oof(
        developments["cxr_labs_gated"], private_root
    )
    cxr_repeat_oof = validated_development_repeat_oof(developments["cxr_densenet"], private_root)
    cohort = load_symile_development_cohort(configs["cxr_labs_gated"]).frame
    attributes = cohort.loc[:, ["sample_id", "age_years", "sex", "view_position"]].copy()
    attributes["observed_lab_count"] = cohort.loc[:, LAB_OBSERVED_COLUMNS].sum(axis=1)
    subgroup_reference = publish_focused_subgroup_derivative(
        report_root=report_root,
        summary=focused_development_subgroups(cxr_repeat_oof, primary_repeat_oof, attributes),
    )
    ecg_extension = publish_ecg_extension_result(
        report_root=report_root,
        core_analysis_directory=analysis_directory,
        ecg_development=developments[SYMILE_ECG_GATED_FAMILY],
        focused_subgroup_derivative=subgroup_reference,
        model_root=Path(model_root) / "development",
        prediction_root=private_root,
        manifest_root=manifest_root,
    )
    provenance = {
        "git_commit": commit,
        "git_dirty": False,
        "dependency_lock_sha256": lock_hash,
    }
    final_configs = {
        family: with_runtime(
            configs[family],
            manifest_directory=manifest_root,
            source_root=source_root,
            model_directory=model_root,
            report_directory=report_root,
            private_output_directory=private_root,
            device=device,
            num_workers=workers,
        )
        for family in (*FINAL_NEURAL_FAMILIES, *FINAL_TABULAR_FAMILIES)
    }
    packages = []
    for family in FINAL_TABULAR_FAMILIES:
        config = final_configs[family]
        packages.append(
            _tracked_final_fit(
                config,
                member_seed=FINAL_TABULAR_SEED,
                provenance=provenance,
                fit=lambda operational, family=family, config=config: fit_final_tabular_family(
                    config,
                    development=None if family == "labs_logistic" else developments[family],
                    operational=operational,
                ),
            )
        )
    cxr_by_seed = {}
    for seed in FINAL_NEURAL_MEMBER_SEEDS:
        config = final_configs["cxr_densenet"]
        package = _tracked_final_fit(
            config,
            member_seed=seed,
            provenance=provenance,
            fit=lambda operational, seed=seed, config=config: fit_final_neural_member(
                config,
                development=developments["cxr_densenet"],
                member_seed=seed,
                expected_pretrained_scientific_identity=ecg_extension.manifest[
                    "final_family_authorities"
                ]["cxr_densenet"]["pretrained_scientific_identity"],
                operational=operational,
            ),
        )
        packages.append(package)
        cxr_by_seed[seed] = package
    for family in FINAL_NEURAL_FAMILIES:
        if family == "cxr_densenet":
            continue
        for seed in FINAL_NEURAL_MEMBER_SEEDS:
            config = final_configs[family]
            packages.append(
                _tracked_final_fit(
                    config,
                    member_seed=seed,
                    provenance=provenance,
                    fit=lambda operational, family=family, seed=seed, config=config: (
                        fit_final_neural_member(
                            config,
                            development=developments[family],
                            member_seed=seed,
                            expected_pretrained_scientific_identity=None,
                            source_cxr_package=cxr_by_seed[seed],
                            operational=operational,
                        )
                    ),
                )
            )
    first = configs["labs_logistic"]
    runtime = resolve_held_out_neural_runtime(packages, device)
    freeze = publish_pretest_freeze(
        control_root=control_root,
        bundle={
            "bundle_id": first.dataset.bundle_id,
            "bundle_manifest_sha256": first.dataset.bundle_manifest_sha256,
            "split_assignment_id": first.dataset.split_assignment_id,
        },
        task={
            "task_id": first.task.task_id,
            "label_policy_version": first.task.label_policy_version,
        },
        ecg_extension_result=ecg_extension,
        final_packages=packages,
        neural_inference_runtime=neural_inference_runtime_policy(runtime),
        science_git_commit=commit,
        dependency_lock_sha256=lock_hash,
    )
    record = create_or_validate_test_open_record(control_root=control_root, capability=freeze)
    return _complete_opened_campaign(
        freeze=freeze,
        record=record,
        ecg_extension=ecg_extension,
        packages=packages,
        source_root=source_root,
        manifest_root=manifest_root,
        model_root=model_root,
        report_root=report_root,
        private_root=private_root,
        export_root=export_root,
        backup_root=backup_root,
        runtime=runtime,
    )


def _resolve_opened_pretest_state(
    *,
    control_root: Path,
    model_root: Path,
    report_root: Path,
    private_root: Path,
    manifest_root: Path,
    science_git_commit: str,
    dependency_lock_sha256: str,
) -> tuple[
    ValidatedPretestFreeze,
    ValidatedTestOpenRecord,
    ValidatedEcgExtensionResult,
    tuple[ValidatedFinalPackage, ...],
]:
    test_open_path = control_root / "test-open.json"
    if test_open_path.is_symlink() or not test_open_path.is_file():
        raise ManifestBuildError("Official-test opening record is invalid")
    try:
        opened = json.loads(test_open_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Official-test opening record is malformed") from exc
    freeze_id = opened.get("freeze_id") if isinstance(opened, dict) else None
    if (
        not isinstance(freeze_id, str)
        or not freeze_id.startswith(PRETEST_FREEZE_PREFIX)
        or len(freeze_id) != len(PRETEST_FREEZE_PREFIX) + 64
    ):
        raise ManifestBuildError("Official-test opening record is malformed")
    validate_path_component(freeze_id, "Symile pre-test freeze identity")
    freeze_root = control_root / "freezes" / freeze_id
    freeze_manifest_path = freeze_root / "manifest.json"
    if freeze_root.is_symlink() or not freeze_root.is_dir() or freeze_manifest_path.is_symlink():
        raise ManifestBuildError("Persisted pre-test freeze is invalid")
    try:
        frozen = json.loads(freeze_manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Persisted pre-test freeze is malformed") from exc
    extension_id = frozen.get("ecg_extension_result_id") if isinstance(frozen, dict) else None
    package_refs = frozen.get("final_packages") if isinstance(frozen, dict) else None
    if (
        not isinstance(extension_id, str)
        or not isinstance(package_refs, list)
        or len(package_refs) != FINAL_PACKAGE_COUNT
        or any(
            not isinstance(reference, dict)
            or set(reference) != {"package_id", "manifest_sha256"}
            or not isinstance(reference["package_id"], str)
            for reference in package_refs
        )
    ):
        raise ManifestBuildError("Persisted pre-test freeze is malformed")
    validate_path_component(extension_id, "ECG extension result identity")
    for reference in package_refs:
        validate_path_component(reference["package_id"], "Final package identity")
    extension = validate_ecg_extension_result(
        report_root / "ecg-extension-results" / extension_id,
        report_root=report_root,
        model_root=model_root / "development",
        prediction_root=private_root,
        manifest_root=manifest_root,
    )
    packages = tuple(
        validate_final_package(
            model_root / "final" / "packages" / reference["package_id"],
            expected_package_id=reference["package_id"],
        )
        for reference in package_refs
    )
    freeze = validate_pretest_freeze(
        freeze_root,
        ecg_extension_result=extension,
        final_packages=packages,
    )
    if (
        freeze.manifest["science_git_commit"] != science_git_commit
        or freeze.manifest["dependency_lock_sha256"] != dependency_lock_sha256
    ):
        raise ManifestBuildError("Opened pre-test freeze differs from current source provenance")
    record = create_or_validate_test_open_record(control_root=control_root, capability=freeze)
    return freeze, record, extension, packages


def _complete_opened_campaign(
    *,
    freeze: ValidatedPretestFreeze,
    record: ValidatedTestOpenRecord,
    ecg_extension: ValidatedEcgExtensionResult,
    packages: list[ValidatedFinalPackage] | tuple[ValidatedFinalPackage, ...],
    source_root: Path,
    manifest_root: Path,
    model_root: Path,
    report_root: Path,
    private_root: Path,
    export_root: Path,
    backup_root: Path,
    runtime: ResolvedDevice,
) -> dict[str, str]:
    analysis_id = str(ecg_extension.manifest["core_analysis_id"])
    predictions, test_data = publish_frozen_test_predictions(
        capability=freeze,
        test_open_record=record,
        final_packages=packages,
        source_root=source_root,
        manifest_root=manifest_root,
        private_root=private_root,
        runtime=runtime,
    )
    global_result = publish_global_result(
        report_root=report_root,
        capability=freeze,
        predictions=predictions,
        final_packages=packages,
        test_data=test_data,
    )
    error_review = publish_error_review(
        private_root=private_root,
        capability=freeze,
        predictions=predictions,
        final_packages=packages,
        test_data=test_data,
    )
    export_members = _campaign_export_members(
        freeze=freeze,
        record=record,
        ecg_extension=ecg_extension,
        packages=packages,
        predictions=predictions,
        global_result=global_result,
        error_review=error_review,
        test_data=test_data,
        manifest_root=manifest_root,
        model_root=model_root,
        report_root=report_root,
        private_root=private_root,
    )

    archive = export_and_verify(
        members=export_members,
        export_root=export_root,
        backup_root=backup_root,
        export_name=global_result.result_id,
        restoration_validator=_validate_restored_campaign,
    )
    return {
        "core_analysis_id": analysis_id,
        "ecg_extension_result_id": ecg_extension.result_id,
        "pretest_freeze_id": freeze.freeze_id,
        "global_result_id": global_result.result_id,
        "export": archive.as_posix(),
    }


def _validate_restored_campaign(restored_root: Path) -> None:
    """Recursively certify one campaign using only its restored canonical tree."""
    manifest_root = restored_root / "data" / "manifests"
    model_root = restored_root / "models" / "symile"
    report_root = restored_root / "reports" / "symile"
    private_root = restored_root / "private"
    control_root = private_root / "control" / "symile"
    test_open_path = control_root / "test-open.json"
    try:
        opened = json.loads(test_open_path.read_bytes())
        freeze_id = opened["freeze_id"]
        frozen = json.loads((control_root / "freezes" / freeze_id / "manifest.json").read_bytes())
        extension_id = frozen["ecg_extension_result_id"]
        package_references = frozen["final_packages"]
    except (OSError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Restored campaign control lineage is invalid") from exc
    validate_path_component(freeze_id, "Restored Symile freeze identity")
    validate_path_component(extension_id, "Restored ECG extension identity")
    if not isinstance(package_references, list) or len(package_references) != FINAL_PACKAGE_COUNT:
        raise ManifestBuildError("Restored campaign package lineage is invalid")
    package_ids = []
    for reference in package_references:
        if not isinstance(reference, dict) or not isinstance(reference.get("package_id"), str):
            raise ManifestBuildError("Restored campaign package lineage is invalid")
        validate_path_component(reference["package_id"], "Restored final package identity")
        package_ids.append(reference["package_id"])
    extension = validate_ecg_extension_result(
        report_root / "ecg-extension-results" / extension_id,
        report_root=report_root,
        model_root=model_root / "development",
        prediction_root=private_root,
        manifest_root=manifest_root,
    )
    analysis_id = str(extension.manifest["core_analysis_id"])
    analysis = validate_analysis_result(
        report_root / "development" / "analyses" / analysis_id,
        report_root=report_root / "development",
        model_root=model_root / "development",
        prediction_root=private_root,
        manifest_root=manifest_root,
        expected_analysis_id=analysis_id,
    )
    for development_id in (
        *analysis["family_development_ids"].values(),
        extension.manifest["ecg_development_id"],
    ):
        validate_development_result(
            report_root / "development" / "families" / development_id,
            model_root=model_root / "development",
            prediction_root=private_root,
            manifest_root=manifest_root,
            expected_development_id=development_id,
        )
    subgroup_id = str(extension.manifest["focused_subgroup_derivative"])
    validate_focused_subgroup_derivative(
        report_root / "development-subgroups" / f"{subgroup_id}.json",
        expected_id=subgroup_id,
    )
    packages = tuple(
        validate_final_package(
            model_root / "final" / "packages" / package_id,
            expected_package_id=package_id,
        )
        for package_id in package_ids
    )
    freeze = validate_pretest_freeze(
        control_root / "freezes" / freeze_id,
        ecg_extension_result=extension,
        final_packages=packages,
    )
    record = create_or_validate_test_open_record(control_root=control_root, capability=freeze)
    prediction_root = private_root / "predictions" / "symile" / "test"
    if prediction_root.is_symlink() or not prediction_root.is_dir():
        raise ManifestBuildError("Restored held-out prediction root is invalid")
    unordered = tuple(
        validate_prediction_evidence(path)
        for path in sorted(prediction_root.iterdir())
        if not path.name.startswith(".")
    )
    by_package = {item.manifest["model_package_id"]: item for item in unordered}
    if (
        len(unordered) != FINAL_PACKAGE_COUNT
        or len(by_package) != FINAL_PACKAGE_COUNT
        or set(by_package) != set(package_ids)
    ):
        raise ManifestBuildError("Restored held-out prediction membership is invalid")
    predictions = tuple(by_package[package_id] for package_id in package_ids)
    test_data = FrozenSymileTestData(
        freeze,
        record,
        load_final_package_config(packages[0]),
        manifest_root=manifest_root,
    )
    projection = test_data.evaluation_projection()
    for evidence in predictions:
        validate_prediction_against_test_projection(
            evidence,
            capability=freeze,
            projection=projection,
            expected_package_ids=tuple(package_ids),
        )
    global_directories = tuple((report_root / "global-results").iterdir())
    error_reviews = tuple((private_root / "error-review" / "symile").iterdir())
    if len(global_directories) != 1 or len(error_reviews) != 1:
        raise ManifestBuildError("Restored campaign result membership is invalid")
    validate_global_result(
        global_directories[0],
        capability=freeze,
        predictions=predictions,
        final_packages=packages,
        test_data=test_data,
    )
    validate_error_review(
        error_reviews[0],
        capability=freeze,
        predictions=predictions,
        final_packages=packages,
        test_data=test_data,
    )


def _campaign_export_members(
    *,
    freeze: ValidatedPretestFreeze,
    record: ValidatedTestOpenRecord,
    ecg_extension: ValidatedEcgExtensionResult,
    packages: list[ValidatedFinalPackage] | tuple[ValidatedFinalPackage, ...],
    predictions: tuple[ValidatedPredictionEvidence, ...],
    global_result: ValidatedGlobalResult,
    error_review: Path,
    test_data: FrozenSymileTestData,
    manifest_root: Path,
    model_root: Path,
    report_root: Path,
    private_root: Path,
) -> tuple[SymileExportMember, ...]:
    """Resolve the exact recursively validated authority closure for one campaign."""
    analysis_id = str(ecg_extension.manifest["core_analysis_id"])
    analysis = validate_analysis_result(
        report_root / "development" / "analyses" / analysis_id,
        report_root=report_root / "development",
        model_root=model_root / "development",
        prediction_root=private_root,
        manifest_root=manifest_root,
        expected_analysis_id=analysis_id,
    )
    family_development_ids = analysis["family_development_ids"]
    if set(family_development_ids) != set(SYMILE_CORE_DEVELOPMENT_FAMILIES):
        raise ManifestBuildError("Campaign core analysis does not contain the exact six families")
    development_ids = [
        *(str(family_development_ids[family]) for family in SYMILE_CORE_DEVELOPMENT_FAMILIES),
        str(ecg_extension.manifest["ecg_development_id"]),
    ]
    developments = [
        validate_development_result(
            report_root / "development" / "families" / development_id,
            model_root=model_root / "development",
            prediction_root=private_root,
            manifest_root=manifest_root,
        )
        for development_id in development_ids
    ]
    if [item.manifest["development_id"] for item in developments] != development_ids:
        raise ManifestBuildError("Campaign development authority order is invalid")
    cv_ids = {item.manifest["scientific_context"]["cv_assignment_id"] for item in developments}
    bundle_ids = {item.manifest["scientific_context"]["bundle_id"] for item in developments}
    if (
        cv_ids == {None}
        or len(cv_ids) != 1
        or bundle_ids != {freeze.manifest["bundle"]["bundle_id"]}
    ):
        raise ManifestBuildError("Campaign data authorities are inconsistent")
    cv_id = str(next(iter(cv_ids)))

    validated_extension = validate_ecg_extension_result(
        report_root / "ecg-extension-results" / ecg_extension.result_id,
        report_root=report_root,
        model_root=model_root / "development",
        prediction_root=private_root,
        manifest_root=manifest_root,
    )
    validated_packages = tuple(
        validate_final_package(
            model_root / "final" / "packages" / package.package_id,
            expected_package_id=package.package_id,
        )
        for package in packages
    )
    validated_freeze = validate_pretest_freeze(
        private_root / "control" / "symile" / "freezes" / freeze.freeze_id,
        ecg_extension_result=validated_extension,
        final_packages=validated_packages,
    )
    if validated_freeze.manifest_sha256 != freeze.manifest_sha256:
        raise ManifestBuildError("Campaign freeze changed before export")
    validated_record = create_or_validate_test_open_record(
        control_root=private_root / "control" / "symile", capability=freeze
    )
    if validated_record != record:
        raise ManifestBuildError("Campaign test-open record changed before export")
    validated_global_result = validate_global_result(
        report_root / "global-results" / global_result.result_id,
        capability=freeze,
        predictions=predictions,
        final_packages=packages,
        test_data=test_data,
    )
    validate_error_review(
        error_review,
        capability=freeze,
        predictions=predictions,
        final_packages=packages,
        test_data=test_data,
    )

    members: list[SymileExportMember] = []

    def add(path: Path, restore_relative: Path) -> None:
        if (
            path.is_symlink()
            or not path.exists()
            or restore_relative.is_absolute()
            or ".." in restore_relative.parts
        ):
            raise ManifestBuildError("Campaign export authority path is invalid")
        members.append(SymileExportMember(path, restore_relative))

    bundle_id = str(freeze.manifest["bundle"]["bundle_id"])
    add(
        manifest_root / "symile" / BUNDLES_DIRECTORY / bundle_id,
        Path("data/manifests/symile") / BUNDLES_DIRECTORY / bundle_id,
    )
    add(
        manifest_root / "symile" / CV_DIRECTORY / cv_id,
        Path("data/manifests/symile") / CV_DIRECTORY / cv_id,
    )
    for development in developments:
        development_id = str(development.manifest["development_id"])
        add(
            report_root / "development" / "families" / development_id,
            Path("reports/symile/development/families") / development_id,
        )
        for reference in development.manifest["fold_packages"]:
            fold_id = str(reference["fold_package_id"])
            prediction_id = str(reference["prediction_id"])
            add(
                model_root / "development" / "packages" / fold_id,
                Path("models/symile/development/packages") / fold_id,
            )
            add(
                private_root / "predictions" / "symile" / "oof" / prediction_id,
                Path("private/predictions/symile/oof") / prediction_id,
            )
    add(
        report_root / "development" / "analyses" / analysis_id,
        Path("reports/symile/development/analyses") / analysis_id,
    )
    subgroup_id = str(ecg_extension.manifest["focused_subgroup_derivative"])
    add(
        report_root / "development-subgroups" / f"{subgroup_id}.json",
        Path("reports/symile/development-subgroups") / f"{subgroup_id}.json",
    )
    add(
        validated_extension.directory,
        Path("reports/symile/ecg-extension-results") / validated_extension.result_id,
    )
    for package in validated_packages:
        add(
            package.directory,
            Path("models/symile/final/packages") / package.package_id,
        )
    add(
        validated_freeze.directory,
        Path("private/control/symile/freezes") / validated_freeze.freeze_id,
    )
    add(validated_record.path, Path("private/control/symile/test-open.json"))
    for prediction in predictions:
        evidence = validate_prediction_evidence(
            private_root / "predictions" / "symile" / "test" / prediction.prediction_id,
            expected_prediction_id=prediction.prediction_id,
        )
        add(
            evidence.directory,
            Path("private/predictions/symile/test") / evidence.prediction_id,
        )
    add(
        validated_global_result.directory,
        Path("reports/symile/global-results") / validated_global_result.result_id,
    )
    add(error_review, Path("private/error-review/symile") / error_review.name)
    resolved = [member.path.resolve() for member in members]
    if len(resolved) != len(set(resolved)):
        raise ManifestBuildError("Campaign export authority paths contain duplicates")
    return tuple(members)


def _tracked_final_fit(
    config,
    *,
    member_seed: int,
    provenance: dict[str, object],
    fit: Callable[[dict[str, object]], ValidatedFinalPackage],
) -> ValidatedFinalPackage:
    tags = {
        "run_kind": "final_training",
        "evaluation_scope": "full_development",
        "dataset_id": config.dataset.dataset_id,
        "task_id": config.task.task_id,
        "label_policy_version": config.task.label_policy_version,
        "family_id": config.family.family_id,
        "modalities": serialize_modalities(config.family.modalities),
        "bundle_id": config.dataset.bundle_id,
        "bundle_manifest_sha256": config.dataset.bundle_manifest_sha256,
        "split_assignment_id": config.dataset.split_assignment_id,
        "member_seed": str(member_seed),
        "config_source_sha256": config.config_source_sha256,
        "config_semantic_sha256": config.config_semantic_sha256,
        "git_commit": str(provenance["git_commit"]),
        "git_dirty": "false",
        "dependency_lock_sha256": str(provenance["dependency_lock_sha256"]),
        "run_complete": "false",
    }
    with tracked_run(
        run_name=f"final-{config.family.family_id}-seed-{member_seed}",
        tags=tags,
        parameters={"member_seed": member_seed, **environment_provenance()},
    ) as run_id:
        log_source_config(config)
        package = fit({**provenance, "mlflow_run_id": run_id})
        mlflow.set_tags({"package_kind": "final", "package_id": package.package_id})
        mlflow.set_tag("run_complete", "true")
    return package


def _repository_path(repository_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _validate_canonical_formal_layout(repository_root: Path) -> None:
    """Reject redirected, aliased, or mistyped checkout-owned formal state."""
    root = repository_root.resolve()
    directory_roots = tuple(
        repository_root / path
        for path in (
            "data/manifests/symile",
            "models/symile/development",
            "models/symile/final",
            "reports/symile/development",
            "reports/symile/development-subgroups",
            "reports/symile/ecg-extension-results",
            "reports/symile/global-results",
            "private/control/symile",
            "private/predictions/symile",
            "private/error-review/symile",
            "outbox",
            "mlartifacts",
        )
    )
    file_roots = tuple(
        repository_root / name for name in ("mlflow.db", "mlflow.db-wal", "mlflow.db-shm")
    )
    for target in (*directory_roots, *file_roots):
        try:
            relative = target.relative_to(repository_root)
        except ValueError as exc:
            raise ManifestBuildError("Canonical Symile layout is outside the repository") from exc
        current = repository_root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ManifestBuildError("Canonical Symile layout contains a symlink redirect")
            if current.exists() and current != target and not current.is_dir():
                raise ManifestBuildError("Canonical Symile layout has a parent type mismatch")
        if target.exists():
            expected = target in directory_roots
            if (expected and not target.is_dir()) or (not expected and not target.is_file()):
                raise ManifestBuildError("Canonical Symile layout has a root type mismatch")
            if not target.resolve().is_relative_to(root):
                raise ManifestBuildError("Canonical Symile layout escapes the repository")
    existing = [path for path in (*directory_roots, *file_roots) if path.exists()]
    for index, left in enumerate(existing):
        for right in existing[index + 1 :]:
            if left.samefile(right):
                raise ManifestBuildError("Canonical Symile layout roots are aliased")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backup-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_symile_campaign(**vars(args))
    except Exception as exc:
        print(f"Symile campaign failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
