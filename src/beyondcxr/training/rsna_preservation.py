"""Explicit preservation closure for a completed RSNA campaign."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.rsna_artifacts import (
    LABELS_FILENAME,
    SPLITS_FILENAME,
    validate_bundle_directory,
    validate_bundle_reference,
)
from beyondcxr.data.rsna_audit import REPORT_FILENAMES, generate_rsna_audit
from beyondcxr.data.rsna_schemas import PNEUMONIA_LABEL_POLICY_VERSION, PNEUMONIA_TASK_ID
from beyondcxr.training.preservation import PreservationMember, validate_exact_restored_closure
from beyondcxr.training.rsna_campaign_control import (
    ValidatedRsnaExecution,
    ValidatedRsnaPackageFreeze,
    load_evaluation_record,
    validate_execution,
    validate_execution_closure,
    validate_package_freeze,
)
from beyondcxr.training.rsna_compare import ComparisonResult, validate_comparison
from beyondcxr.training.rsna_evaluation_result import (
    CompletedRsnaEvaluation,
    validate_rsna_model_package,
)
from beyondcxr.training.rsna_localize import validate_localization_report
from beyondcxr.training.rsna_seed_summary import SeedSummaryResult, validate_seed_summary


def campaign_export_members(
    *,
    package_freeze: ValidatedRsnaPackageFreeze,
    evaluations: Sequence[CompletedRsnaEvaluation],
    summaries: Sequence[SeedSummaryResult],
    localization: Path,
    comparison: ComparisonResult,
    audit_directory: Path,
    campaign_log: Path,
    private_root: Path,
) -> tuple[PreservationMember, ...]:
    """Return the exact non-overlapping authority closure for one RSNA campaign."""
    validate_execution_closure(package_freeze)
    execution = package_freeze.execution
    bundle_directory = package_freeze.bundle_directory
    report_root = package_freeze.report_root
    members: list[PreservationMember] = []

    def add(path: Path, restore_relative: Path) -> None:
        resolved = path.resolve()
        if resolved in {member.path.resolve() for member in members}:
            raise ManifestBuildError("RSNA preservation closure contains a duplicate source")
        members.append(PreservationMember(path, restore_relative))

    add(
        bundle_directory,
        Path("data/manifests/rsna/bundles") / bundle_directory.name,
    )
    add(
        execution.directory,
        Path("private/control/rsna/executions") / execution.execution_id,
    )
    for result in package_freeze.packages:
        package = Path(result.model_path).parent
        add(package, Path("models/rsna/packages") / package.name)
        report = Path(result.artifact_directory)
        add(report, Path("reports") / report.relative_to(report_root))
    for result in evaluations:
        add(
            Path(result.artifact_directory),
            Path("reports/rsna/evaluations") / result.evaluation_id,
        )
        add(
            Path(result.private_prediction_directory),
            Path("private/predictions/rsna") / result.prediction_id,
        )
    for summary in summaries:
        add(
            summary.directory,
            Path("reports/rsna/seed-summaries") / summary.seed_summary_id,
        )
    add(localization, Path("reports/rsna/localization") / localization.name)
    add(
        private_root / "localization" / localization.name,
        Path("private/localization") / localization.name,
    )
    add(comparison.directory, Path("reports/rsna/comparisons") / comparison.comparison_id)
    add(audit_directory, Path("reports/rsna/audit") / audit_directory.name)
    add(
        campaign_log,
        Path("reports/rsna/campaigns") / campaign_log.parent.name / "execution.log",
    )
    return tuple(members)


def validate_restored_rsna_campaign(restored_root: Path) -> None:
    """Recursively validate an RSNA campaign using only restored closure members."""
    execution_root = restored_root / "private/control/rsna/executions"
    execution_directories = _directories(execution_root)
    if len(execution_directories) != 1:
        raise ManifestBuildError("Restored RSNA closure must contain one execution control")
    execution = validate_execution(execution_directories[0])
    dataset = execution.manifest["dataset"]
    assert isinstance(dataset, dict)
    bundle_id = str(dataset["bundle_id"])
    bundle = restored_root / "data/manifests/rsna/bundles" / bundle_id
    reference = validate_bundle_reference(
        bundle,
        expected_bundle_id=bundle_id,
        expected_manifest_sha256=str(dataset["bundle_manifest_sha256"]),
    )
    manifest = validate_bundle_directory(bundle, expected_bundle_id=bundle_id)
    if (
        reference.manifest_sha256 != dataset["bundle_manifest_sha256"]
        or manifest["membership"]["split"]["split_assignment_id"] != dataset["split_assignment_id"]
    ):
        raise ManifestBuildError("Restored RSNA bundle differs from execution authority")
    expected_positive_sample_ids = _canonical_positive_heldout_sample_ids(
        bundle,
        manifest=manifest,
        execution=execution,
    )

    model_root = restored_root / "models/rsna"
    report_root = restored_root / "reports"
    private_root = restored_root / "private"
    frozen = validate_package_freeze(
        execution,
        model_root=model_root,
        report_root=report_root,
        bundle_directory=bundle,
    )
    evaluations = []
    for package in frozen:
        result = load_evaluation_record(
            execution,
            package_id=package.model_package_id,
            report_root=report_root,
            private_root=private_root,
            model_root=model_root,
        )
        if result is None:
            raise ManifestBuildError("Restored RSNA closure lacks one evaluation record")
        evaluations.append(result)

    summary_directories = _directories(report_root / "rsna/seed-summaries")
    if len(summary_directories) != 2:
        raise ManifestBuildError("Restored RSNA closure must contain two seed summaries")
    summaries = tuple(
        validate_seed_summary(
            directory,
            report_root=report_root,
            model_root=model_root,
            private_root=private_root,
        )
        for directory in summary_directories
    )
    if {summary.manifest["family_id"] for summary in summaries} != {
        "cxr_densenet",
        "cxr_metadata_concat",
    }:
        raise ManifestBuildError("Restored RSNA seed-summary families are incomplete")

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

    comparison_directories = _directories(report_root / "rsna/comparisons")
    if len(comparison_directories) != 1:
        raise ManifestBuildError("Restored RSNA closure must contain one comparison")
    comparison = validate_comparison(
        comparison_directories[0],
        report_root=report_root,
        private_root=private_root,
        model_root=model_root,
    )
    comparison_evaluation_ids = comparison.manifest["evaluation_ids"]
    if len(comparison_evaluation_ids) != len(evaluations) or set(comparison_evaluation_ids) != {
        result.evaluation_id for result in evaluations
    }:
        raise ManifestBuildError("Restored RSNA comparison membership is incomplete")

    audit = report_root / "rsna/audit" / bundle_id
    if (
        audit.is_symlink()
        or not audit.is_dir()
        or {path.name for path in audit.iterdir()} != set(REPORT_FILENAMES)
        or any(path.is_symlink() or not path.is_file() for path in audit.iterdir())
    ):
        raise ManifestBuildError("Restored RSNA audit artifact set is invalid")
    with tempfile.TemporaryDirectory(prefix="beyondcxr-rsna-audit-validation-") as temporary:
        regenerated_root = Path(temporary) / "audit"
        generate_rsna_audit(
            restored_root / "data/manifests",
            regenerated_root,
            bundle_id=bundle_id,
        )
        regenerated = regenerated_root / bundle_id
        if any(
            (audit / filename).read_bytes() != (regenerated / filename).read_bytes()
            for filename in REPORT_FILENAMES
        ):
            raise ManifestBuildError("Restored RSNA audit differs from its bundle authority")
    logs = tuple((report_root / "rsna/campaigns").glob("*/execution.log"))
    if len(logs) != 1 or logs[0].is_symlink() or not logs[0].is_file():
        raise ManifestBuildError("Restored RSNA campaign log is invalid")
    if "event=campaign_ready_for_export" not in logs[0].read_text(encoding="utf-8"):
        raise ManifestBuildError("Restored RSNA campaign log is incomplete")
    validate_exact_restored_closure(
        restored_root,
        campaign_export_members(
            package_freeze=frozen,
            evaluations=evaluations,
            summaries=summaries,
            localization=localization,
            comparison=comparison,
            audit_directory=audit,
            campaign_log=logs[0],
            private_root=private_root,
        ),
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


def _canonical_positive_heldout_sample_ids(
    bundle: Path,
    *,
    manifest: dict[str, object],
    execution: ValidatedRsnaExecution,
) -> tuple[str, ...]:
    """Derive the exact canonical positive test cohort from one validated bundle."""
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
    splits = pq.read_table(
        bundle / SPLITS_FILENAME,
        columns=["sample_id"],
        filters=[("split_name", "=", "test")],
    ).to_pandas()
    labels = pq.read_table(
        bundle / LABELS_FILENAME,
        columns=["sample_id"],
        filters=[("task_id", "=", PNEUMONIA_TASK_ID), ("label_value", "=", 1)],
    ).to_pandas()
    test_ids = set(splits["sample_id"].astype(str))
    positive_ids = tuple(sorted(set(labels["sample_id"].astype(str)) & test_ids))
    if not positive_ids:
        raise ManifestBuildError("Restored RSNA positive held-out cohort is empty")
    return positive_ids


def _package_family(model_root: Path, package_id: str) -> str:
    return str(validate_rsna_model_package(model_root, package_id)["family_id"])
