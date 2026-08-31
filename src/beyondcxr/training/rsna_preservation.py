"""Explicit preservation closure for a completed RSNA campaign."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.rsna_artifacts import (
    LABELS_FILENAME,
    SPLITS_FILENAME,
    validate_bundle_directory,
    validate_bundle_reference,
)
from beyondcxr.data.rsna_audit import generate_rsna_audit
from beyondcxr.data.rsna_schemas import PNEUMONIA_LABEL_POLICY_VERSION, PNEUMONIA_TASK_ID
from beyondcxr.training.campaign_observability import campaign_log_members
from beyondcxr.training.preservation import (
    PreservationMember,
    PreservationRole,
    _validated_restoration_context,
    validate_exact_restored_closure,
    validate_preservation_members,
)
from beyondcxr.training.rsna_campaign_control import (
    ValidatedRsnaExecution,
    ValidatedRsnaFreeze,
    _load_evaluation_record_from_established_context,
    validate_execution,
    validate_execution_closure,
    validate_freeze,
)
from beyondcxr.training.rsna_compare import (
    ComparisonResult,
    regenerate_comparison,
)
from beyondcxr.training.rsna_evaluation_result import (
    CompletedRsnaEvaluation,
    validate_rsna_model_package,
)
from beyondcxr.training.rsna_localize import validate_localization_report
from beyondcxr.training.rsna_seed_summary import (
    SeedSummaryResult,
    publish_seed_summary,
)
from beyondcxr.training.rsna_training_report import validate_training_report
from beyondcxr.training.rsna_validation_evidence import load_canonical_validation_cohort
from beyondcxr.utils.semantic_identity import canonical_json_bytes


@dataclass(frozen=True)
class RsnaTrainingDerivative:
    model_package_id: str
    model_path: Path
    artifact_directory: Path


def campaign_export_members(
    *,
    execution: ValidatedRsnaExecution,
    freeze: ValidatedRsnaFreeze,
    training_results: Sequence[object],
    evaluations: Sequence[CompletedRsnaEvaluation],
    summaries: Sequence[SeedSummaryResult],
    localization: Path,
    comparison: ComparisonResult,
    audit_directory: Path,
    private_root: Path,
    report_root: Path,
) -> tuple[PreservationMember, ...]:
    """Prepare the RSNA closure and derivatives before attaching attempt evidence."""
    validate_execution_closure(execution, freeze)
    bundle_directory = freeze.bundle_directory
    members: list[PreservationMember] = []

    def add(
        path: Path,
        restore_relative: Path,
        *,
        role: PreservationRole = "scientific_authority",
    ) -> None:
        members.append(PreservationMember(path, restore_relative, role))

    add(
        bundle_directory,
        Path("data/manifests/rsna/bundles") / bundle_directory.name,
    )
    add(
        execution.directory,
        Path("private/control/rsna/executions") / execution.execution_id,
        role="operational_evidence",
    )
    add(
        execution.directory.parent.parent / "evaluations" / execution.execution_id,
        Path("private/control/rsna/evaluations") / execution.execution_id,
        role="operational_evidence",
    )
    add(
        execution.directory.parent.parent / "active-execution.json",
        Path("private/control/rsna/active-execution.json"),
        role="operational_evidence",
    )
    add(
        freeze.directory,
        Path("private/control/rsna/freezes") / freeze.freeze_id,
    )
    if tuple(result.model_package_id for result in training_results) != tuple(
        item.model_package_id for item in freeze.packages
    ):
        raise ManifestBuildError("RSNA training derivatives differ from frozen packages")
    for result in training_results:
        package = Path(result.model_path).parent
        add(package, Path("models/rsna/packages") / package.name)
        report = Path(result.artifact_directory)
        add(report, Path("reports") / report.relative_to(report_root), role="derivative")
    for result in evaluations:
        add(
            Path(result.artifact_directory),
            Path("reports/rsna/evaluations") / result.evaluation_id,
        )
        add(
            Path(result.private_prediction_directory),
            Path("private/predictions/rsna") / result.prediction_id,
        )
    evaluation_ids_by_family: dict[str, list[str]] = {}
    for result in evaluations:
        family = _package_family(Path(freeze.model_root), result.model_package_id)
        evaluation_ids_by_family.setdefault(family, []).append(result.evaluation_id)
    summary_families = ("cxr_densenet", "cxr_metadata_concat")
    if any(family not in evaluation_ids_by_family for family in summary_families):
        raise ManifestBuildError("RSNA evaluation closure cannot regenerate seed summaries")
    regenerated_summaries = tuple(
        publish_seed_summary(
            evaluation_ids_by_family[family],
            output_directory=report_root,
            model_directory=freeze.model_root,
            private_directory=private_root,
            freeze=freeze,
            execution=execution,
        )
        for family in summary_families
    )
    if {summary.directory.resolve() for summary in summaries} != {
        summary.directory.resolve() for summary in regenerated_summaries
    }:
        raise ManifestBuildError("RSNA seed-summary derivatives use unexpected coordinates")
    for summary in regenerated_summaries:
        add(
            summary.directory,
            Path("reports/rsna/seed-summaries") / summary.directory.name,
            role="derivative",
        )
    add(localization, Path("reports/rsna/localization") / localization.name)
    add(
        private_root / "localization" / localization.name,
        Path("private/localization") / localization.name,
    )
    regenerated_comparison = regenerate_comparison(
        [result.evaluation_id for result in evaluations],
        output_directory=report_root,
        private_directory=private_root,
        model_directory=freeze.model_root,
        freeze=freeze,
        execution=execution,
    )
    if comparison.directory.resolve() != regenerated_comparison.directory.resolve():
        raise ManifestBuildError("RSNA comparison derivative uses an unexpected coordinate")
    add(
        regenerated_comparison.directory,
        Path("reports/rsna/comparisons/comparison"),
        role="derivative",
    )
    bundle_id = freeze.bundle_directory.name
    generate_rsna_audit(
        freeze.bundle_directory.parents[2],
        report_root / "rsna/audit",
        bundle_id=bundle_id,
    )
    regenerated_audit = report_root / "rsna/audit" / bundle_id
    if audit_directory.resolve() != regenerated_audit.resolve():
        raise ManifestBuildError("RSNA audit derivative uses an unexpected coordinate")
    add(regenerated_audit, Path("reports/rsna/audit") / bundle_id, role="derivative")
    validate_preservation_members(members)
    return tuple(members)


def validate_restored_rsna_campaign(restored_root: Path) -> None:
    """Recursively validate an RSNA campaign using only restored closure members."""
    restored_root = _validated_restoration_context(restored_root).root
    execution_root = restored_root / "private/control/rsna/executions"
    execution_directories = _directories(execution_root)
    if len(execution_directories) != 1:
        raise ManifestBuildError("Restored RSNA closure must contain one execution control")
    execution = validate_execution(execution_directories[0])
    active_execution = restored_root / "private/control/rsna/active-execution.json"
    expected_active = {"schema_version": 1, "execution_id": execution.execution_id}
    try:
        active_raw = active_execution.read_bytes()
        active_document = json.loads(active_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Restored RSNA active execution receipt is unreadable") from exc
    if (
        active_execution.is_symlink()
        or not active_execution.is_file()
        or active_document != expected_active
        or active_raw != canonical_json_bytes(expected_active)
    ):
        raise ManifestBuildError("Restored RSNA active execution receipt is invalid")
    dataset = execution.manifest["dataset"]
    assert isinstance(dataset, dict)
    bundle_id = str(dataset["bundle_id"])
    bundle = restored_root / "data/manifests/rsna/bundles" / bundle_id
    validate_bundle_reference(bundle, expected_bundle_id=bundle_id)
    manifest = validate_bundle_directory(bundle, expected_bundle_id=bundle_id)
    task = execution.manifest["task"]
    tasks = manifest.get("tasks")
    bundle_task = tasks.get(PNEUMONIA_TASK_ID) if isinstance(tasks, dict) else None
    if (
        not isinstance(task, dict)
        or task.get("task_id") != PNEUMONIA_TASK_ID
        or task.get("label_policy_version") != PNEUMONIA_LABEL_POLICY_VERSION
        or not isinstance(bundle_task, dict)
        or bundle_task.get("label_policy_version") != PNEUMONIA_LABEL_POLICY_VERSION
    ):
        raise ManifestBuildError("Restored RSNA localization task authority is invalid")
    test_ids = set(
        pq.read_table(
            bundle / SPLITS_FILENAME,
            columns=["sample_id"],
            filters=[("split_name", "=", "test")],
        )
        .to_pandas()["sample_id"]
        .astype(str)
    )
    positive_label_ids = set(
        pq.read_table(
            bundle / LABELS_FILENAME,
            columns=["sample_id"],
            filters=[("task_id", "=", PNEUMONIA_TASK_ID), ("label_value", "=", 1)],
        )
        .to_pandas()["sample_id"]
        .astype(str)
    )
    expected_positive_sample_ids = tuple(sorted(test_ids & positive_label_ids))
    if not expected_positive_sample_ids:
        raise ManifestBuildError("Restored RSNA positive held-out cohort is empty")

    model_root = restored_root / "models/rsna"
    report_root = restored_root / "reports"
    private_root = restored_root / "private"
    freeze_directories = _directories(restored_root / "private/control/rsna/freezes")
    if len(freeze_directories) != 1:
        raise ManifestBuildError("Restored RSNA closure must contain one scientific freeze")
    frozen = validate_freeze(
        freeze_directories[0],
        model_root=model_root,
        bundle_directory=bundle,
    )
    validate_execution_closure(execution, frozen)
    training_results = _restored_training_derivatives(
        frozen,
        execution=execution,
        bundle=bundle,
        model_root=model_root,
        report_root=report_root,
    )
    evaluations = []
    for package in frozen:
        result = _load_evaluation_record_from_established_context(
            execution,
            freeze=frozen,
            package_id=package.model_package_id,
            report_root=report_root,
            private_root=private_root,
            model_root=model_root,
        )
        if result is None:
            raise ManifestBuildError("Restored RSNA closure lacks one evaluation record")
        evaluations.append(result)

    evaluation_ids_by_family: dict[str, list[str]] = {}
    for result in evaluations:
        family = _package_family(model_root, result.model_package_id)
        evaluation_ids_by_family.setdefault(family, []).append(result.evaluation_id)
    summaries = tuple(
        publish_seed_summary(
            evaluation_ids_by_family[family],
            output_directory=report_root,
            model_directory=model_root,
            private_directory=private_root,
            freeze=frozen,
            execution=execution,
        )
        for family in ("cxr_densenet", "cxr_metadata_concat")
    )

    localization_directories = _directories(report_root / "rsna/localization")
    if len(localization_directories) != 1:
        raise ManifestBuildError("Restored RSNA closure must contain one localization result")
    cxr_packages = tuple(
        package.model_package_id
        for package in frozen
        if _package_family(model_root, package.model_package_id) == "cxr_densenet"
    )
    localization = validate_localization_report(
        localization_directories[0],
        private_directory=private_root / "localization" / localization_directories[0].name,
        expected_positive_sample_ids=expected_positive_sample_ids,
        expected_model_package_ids=cxr_packages,
    )

    comparison = regenerate_comparison(
        [result.evaluation_id for result in evaluations],
        output_directory=report_root,
        private_directory=private_root,
        model_directory=model_root,
        freeze=frozen,
        execution=execution,
    )
    comparison_evaluation_ids = comparison.manifest["evaluation_ids"]
    if len(comparison_evaluation_ids) != len(evaluations) or set(comparison_evaluation_ids) != {
        result.evaluation_id for result in evaluations
    }:
        raise ManifestBuildError("Restored RSNA comparison membership is incomplete")

    generate_rsna_audit(
        restored_root / "data/manifests",
        report_root / "rsna/audit",
        bundle_id=bundle_id,
    )
    audit = report_root / "rsna/audit" / bundle_id
    campaign_log = report_root / "rsna/campaigns/export/execution.log"
    validate_exact_restored_closure(
        restored_root,
        campaign_export_members(
            execution=execution,
            freeze=frozen,
            training_results=training_results,
            evaluations=evaluations,
            summaries=summaries,
            localization=localization,
            comparison=comparison,
            audit_directory=audit,
            private_root=private_root,
            report_root=report_root,
        )
        + campaign_log_members(campaign_log, study="rsna", execution_id=execution.execution_id),
    )


def _restored_training_derivatives(
    freeze: ValidatedRsnaFreeze,
    *,
    execution: ValidatedRsnaExecution,
    bundle: Path,
    model_root: Path,
    report_root: Path,
) -> tuple[RsnaTrainingDerivative, ...]:
    cohort = load_canonical_validation_cohort(
        bundle,
        expected_bundle_id=str(execution.manifest["dataset"]["bundle_id"]),
        expected_task_id=PNEUMONIA_TASK_ID,
        expected_label_policy_version=PNEUMONIA_LABEL_POLICY_VERSION,
    )
    run_root = report_root / "rsna/runs"
    candidates = _directories(run_root)
    by_package: dict[str, list[RsnaTrainingDerivative]] = {}
    for directory in candidates:
        try:
            document = json.loads((directory / "metrics.json").read_bytes())
            binding = document["training_package"]
            package_id = str(binding["model_package_id"])
            package = validate_rsna_model_package(model_root, package_id)
            package_directory = model_root / "packages" / package_id
            model_candidates = tuple(
                path
                for path in (package_directory / "model.skops", package_directory / "model.pt")
                if path.is_file() and not path.is_symlink()
            )
            if len(model_candidates) != 1:
                raise ManifestBuildError("Restored RSNA package model is invalid")
            validate_training_report(
                directory,
                run_id=directory.name,
                model_package_id=package_id,
                family_id=str(binding["family_id"]),
                seed=int(binding["seed"]),
                package=package,
                package_directory=package_directory,
                canonical_validation_cohort=cohort,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ManifestBuildError("Restored RSNA training derivative is invalid") from exc
        by_package.setdefault(package_id, []).append(
            RsnaTrainingDerivative(package_id, model_candidates[0], directory)
        )
    expected = tuple(item.model_package_id for item in freeze.packages)
    if set(by_package) != set(expected):
        raise ManifestBuildError("Restored RSNA training derivative membership is incomplete")
    return tuple(
        min(by_package[package_id], key=lambda derivative: derivative.artifact_directory.name)
        for package_id in expected
    )


def _directories(root: Path) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        raise ManifestBuildError(f"Restored RSNA closure directory is missing: {root}")
    values = tuple(
        sorted(path for path in root.iterdir() if path.is_dir() and not path.is_symlink())
    )
    if len(values) != len(tuple(root.iterdir())):
        raise ManifestBuildError(f"Restored RSNA closure directory has unexpected members: {root}")
    return values


def _package_family(model_root: Path, package_id: str) -> str:
    return str(validate_rsna_model_package(model_root, package_id)["family_id"])
