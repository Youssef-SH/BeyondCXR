"""Execute the formal RSNA campaign from one exact prepublished authority."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.rsna_artifacts import (
    BUNDLES_DIRECTORY,
    validate_bundle_directory,
    validate_bundle_reference,
)
from beyondcxr.data.rsna_audit import generate_rsna_audit
from beyondcxr.data.rsna_cxr_cache import preprocessing_identity
from beyondcxr.models.cxr_baseline import fingerprint_pretrained_weights
from beyondcxr.training.campaign_lock import campaign_lock
from beyondcxr.training.campaign_observability import campaign_log_members, campaign_session
from beyondcxr.training.config import (
    ExperimentConfig,
    load_experiment_config,
    require_runtime_seed,
    with_runtime,
)
from beyondcxr.training.device import ResolvedDevice, resolve_device
from beyondcxr.training.execution import (
    LoaderExecutionPolicy,
    one_shot_loader_policy,
    reused_loader_policy,
)
from beyondcxr.training.operational_validation import (
    validate_existing_sqlite_database,
    validate_storage_capacity,
    validate_writable_directory_destination,
    validate_writable_file_destination,
)
from beyondcxr.training.preservation import (
    export_and_verify,
    validate_preservation_paths,
)
from beyondcxr.training.rsna_campaign_control import (
    FrozenScientificPackage,
    ValidatedRsnaFreeze,
    authorize_rsna_held_out,
    load_evaluation_record,
    publish_evaluation_record,
    publish_or_validate_execution,
    publish_or_validate_freeze,
    resolve_completed_rsna_packages,
    validate_freeze,
)
from beyondcxr.training.rsna_compare import ComparisonResult, regenerate_comparison
from beyondcxr.training.rsna_datasets import (
    RsnaDataset,
    prepare_rsna_cxr_cache,
    validate_rsna_cache_source_availability,
)
from beyondcxr.training.rsna_evaluate import evaluate_model_package
from beyondcxr.training.rsna_evaluation_result import validate_rsna_model_package
from beyondcxr.training.rsna_formal import (
    _FORMAL_PLAN_GUARD,
    RSNA_FORMAL_RUN_PLAN,
    RsnaAuthorityCoordinates,
    RsnaFormalRoots,
    RsnaFormalRunSpec,
    ValidatedRsnaPlan,
    rsna_run_report_root,
)
from beyondcxr.training.rsna_localize import generate_localization_report
from beyondcxr.training.rsna_preservation import (
    campaign_export_members,
    validate_restored_rsna_campaign,
)
from beyondcxr.training.rsna_registry import get_dataset
from beyondcxr.training.rsna_seed_summary import publish_seed_summary
from beyondcxr.training.rsna_train_cxr import train_cxr_experiment
from beyondcxr.training.rsna_train_fusion import train_fusion_experiment
from beyondcxr.training.rsna_train_metadata import train_metadata_experiment
from beyondcxr.training.rsna_training_report import (
    regenerate_training_report,
    validate_training_report,
)
from beyondcxr.training.rsna_validation_evidence import load_canonical_validation_cohort
from beyondcxr.utils.mlflow_utils import discover_repository_root
from beyondcxr.utils.operational_logging import (
    get_operational_logger,
    log_event,
    operational_output,
    timed_phase,
)
from beyondcxr.utils.package_identity import package_scientific_config_payload
from beyondcxr.utils.provenance import git_revision, uv_lock_sha256
from beyondcxr.utils.publication import install_immutable_directory, staging_directory

_RSNA_WORKSPACE_MINIMUM_FREE_BYTES = 16 * 1024**3
# Eight fitted packages (six DenseNet-based, two metadata) plus bundle/prediction/
# localization evidence and derivatives. Raw DICOMs, CXR cache and MLflow are excluded.
# 256 MiB per package gives headroom over the ~30 MiB float32 DenseNet state;
# the separate 1 GiB allowance covers the remaining tabular/aggregate closure.
# One copy for backup; local archive and extraction coexist on the export filesystem.
_RSNA_PRESERVATION_COPY_BYTES = len(RSNA_FORMAL_RUN_PLAN) * 256 * 1024**2 + 1024**3
_LOGGER = get_operational_logger(__name__)


def _rsna_export_name(*, freeze_id: str, execution_id: str) -> str:
    """Return the specification-defined stable RSNA preservation name."""
    return f"rsna-{freeze_id}-{execution_id}"


@dataclass(frozen=True)
class CampaignResult:
    """Exact produced identities and final transport artifacts."""

    campaign_id: str
    model_package_ids: tuple[str, ...]
    evaluation_ids: tuple[str, ...]
    training_run_ids: tuple[str, ...]
    evaluation_run_ids: tuple[str, ...]
    archive_path: Path
    checksum_path: Path
    campaign_log_path: Path


@dataclass(frozen=True)
class _TrainingDerivative:
    run_id: str
    model_package_id: str
    model_path: Path
    artifact_directory: Path


def execute_rsna_campaign(
    *,
    backup_root: str | Path,
    bundle_id: str,
    source_root: str | Path = "data/raw/rsna/extracted",
) -> CampaignResult:
    """Run or resume one exact RSNA execution after mutation-free preflight."""
    with operational_output():
        with timed_phase(_LOGGER, "campaign_preflight"):
            plan = _preflight_rsna_campaign(
                backup_root=backup_root, source_root=source_root, bundle_id=bundle_id
            )
        with campaign_lock(
            plan.roots.control_root,
            "RSNA",
            trusted_root=plan.roots.repository_root,
        ):
            return _execute_locked_rsna_campaign(plan)


def _execute_locked_rsna_campaign(plan: ValidatedRsnaPlan) -> CampaignResult:
    """Mutate one RSNA execution while holding its campaign lock."""
    with campaign_session(
        repository_root=plan.roots.repository_root,
        report_root=plan.roots.report_root / "rsna",
        study="rsna",
        operation="campaign",
    ) as session:
        campaign_id = session.attempt_id
        log_path = session.log_path
        execution = publish_or_validate_execution(plan)
        session.bind_execution(execution.execution_id)
        with timed_phase(_LOGGER, "dataset_audit"):
            bundle_id = plan.authority.bundle_id
            generate_rsna_audit(
                plan.roots.manifest_root,
                plan.roots.report_root / "rsna" / "audit",
                bundle_id=bundle_id,
            )
            audit_directory = plan.roots.report_root / "rsna" / "audit" / bundle_id
        cxr_configs = tuple(
            config
            for spec, config in zip(plan.runs, plan.configs, strict=True)
            if spec.family_id == "cxr_densenet"
        )
        cxr_reference = cxr_configs[0]
        transform = _transform(cxr_reference, training=False)
        with timed_phase(_LOGGER, "cxr_cache_preparation"):
            cache = prepare_rsna_cxr_cache(
                plan.dataset, cxr_reference, transform, cache_root=plan.roots.cache_root
            )
        existing_freezes = _validated_rsna_freezes(plan)
        if existing_freezes:
            freeze = existing_freezes[0]
            training_results = _ensure_training_report_derivatives(plan, freeze.packages, plan.runs)
            log_event(_LOGGER, "training_resumed_from_scientific_freeze", total=8)
        else:
            completed_packages = resolve_completed_rsna_packages(plan, execution)
            completed_list = []
            cxr_packages = {}
            for spec, config, existing in zip(
                plan.runs, plan.configs, completed_packages, strict=True
            ):
                if existing is not None:
                    result = _ensure_training_report_derivatives(plan, (existing,), (spec,))[0]
                elif spec.family_id in {"metadata_logistic", "metadata_lightgbm"}:
                    result = train_metadata_experiment(config, tracking_uri=plan.roots.tracking_uri)
                elif spec.family_id == "cxr_densenet":
                    result = train_cxr_experiment(
                        config,
                        tracking_uri=plan.roots.tracking_uri,
                        cache=cache,
                        execution=plan.training_execution,
                        runtime=plan.runtime,
                    )
                else:
                    result = train_fusion_experiment(
                        config,
                        source_cxr_package_id=cxr_packages[spec.seed],
                        tracking_uri=plan.roots.tracking_uri,
                        cache=cache,
                        execution=plan.training_execution,
                        runtime=plan.runtime,
                    )
                if spec.family_id == "cxr_densenet":
                    cxr_packages[spec.seed] = result.model_package_id
                completed_list.append(result)
            completed = tuple(completed_list)
            freeze = publish_or_validate_freeze(
                plan=plan,
                execution=execution,
                results=completed,
            )
            training_results = completed
        held_out = authorize_rsna_held_out(
            plan=plan,
            execution=execution,
            freeze=freeze,
        )
        log_event(_LOGGER, "training_boundary_completed", total=len(training_results))
        evaluation_results_list = []
        for result, config in zip(
            training_results,
            plan.configs,
            strict=True,
        ):
            evaluation = load_evaluation_record(
                execution,
                freeze=freeze,
                package_id=result.model_package_id,
                report_root=plan.roots.report_root,
                private_root=plan.roots.private_root,
                model_root=plan.roots.model_root,
            )
            if evaluation is None:
                evaluation = evaluate_model_package(
                    result.model_package_id,
                    authorization=held_out,
                    evaluation_config=config,
                    runtime=plan.runtime,
                    tracking_uri=plan.roots.tracking_uri,
                    model_directory=plan.roots.model_root,
                    cache=cache,
                    execution=plan.evaluation_execution,
                    private_output_directory=plan.roots.private_root,
                    report_directory=plan.roots.report_root,
                )
                publish_evaluation_record(
                    execution,
                    evaluation,
                    freeze=freeze,
                    report_root=plan.roots.report_root,
                    private_root=plan.roots.private_root,
                    model_root=plan.roots.model_root,
                )
            evaluation_results_list.append(evaluation)
        evaluation_results = tuple(evaluation_results_list)
        cxr_tests = evaluation_results[2:5]
        fusion_tests = evaluation_results[5:]
        cxr_evaluation_ids = tuple(result.evaluation_id for result in cxr_tests)
        fusion_evaluation_ids = tuple(result.evaluation_id for result in fusion_tests)
        cxr_summary = publish_seed_summary(
            cxr_evaluation_ids,
            output_directory=plan.roots.report_root,
            model_directory=plan.roots.model_root,
            private_directory=plan.roots.private_root,
            freeze=freeze,
            execution=execution,
        )
        fusion_summary = publish_seed_summary(
            fusion_evaluation_ids,
            output_directory=plan.roots.report_root,
            model_directory=plan.roots.model_root,
            private_directory=plan.roots.private_root,
            freeze=freeze,
            execution=execution,
        )
        localization = generate_localization_report(
            cxr_evaluation_ids,
            authorization=held_out,
            output_directory=plan.roots.report_root,
            model_directory=plan.roots.model_root,
            private_directory=plan.roots.private_root,
            cache=cache,
            runtime=plan.runtime,
        )
        comparison = regenerate_comparison(
            tuple(result.evaluation_id for result in evaluation_results),
            output_directory=plan.roots.report_root,
            private_directory=plan.roots.private_root,
            model_directory=plan.roots.model_root,
            freeze=freeze,
            execution=execution,
        )
        with timed_phase(_LOGGER, "output_validation"):
            _validate_outputs(
                training_results,
                evaluation_results,
                audit_directory,
                cxr_summary.report_directory,
                fusion_summary.report_directory,
                localization,
                comparison,
                log_path,
                roots=plan.roots,
            )
        summaries = (cxr_summary, fusion_summary)
        members = campaign_export_members(
            execution=execution,
            freeze=freeze,
            training_results=training_results,
            evaluations=evaluation_results,
            summaries=summaries,
            localization=localization,
            comparison=comparison,
            audit_directory=audit_directory,
            private_root=plan.roots.private_root,
            report_root=plan.roots.report_root,
        )
        log_event(_LOGGER, "campaign_ready_for_export", phase="export")
        members += campaign_log_members(log_path, study="rsna", execution_id=execution.execution_id)
        archive_started_at = time.perf_counter()
        archive = export_and_verify(
            members=members,
            export_root=plan.roots.export_root,
            backup_root=plan.roots.backup_root,
            export_name=_rsna_export_name(
                freeze_id=freeze.freeze_id,
                execution_id=execution.execution_id,
            ),
            restoration_validator=validate_restored_rsna_campaign,
        )
        log_event(
            _LOGGER,
            "archive_created",
            elapsed_s=time.perf_counter() - archive_started_at,
        )
        checksum = archive.with_suffix(".zip.sha256")
    return CampaignResult(
        campaign_id=campaign_id,
        model_package_ids=tuple(result.model_package_id for result in training_results),
        evaluation_ids=tuple(result.evaluation_id for result in evaluation_results),
        training_run_ids=tuple(result.run_id for result in training_results),
        evaluation_run_ids=tuple(result.mlflow_run_id for result in evaluation_results),
        archive_path=archive,
        checksum_path=checksum,
        campaign_log_path=log_path,
    )


def _validated_rsna_freezes(
    plan: ValidatedRsnaPlan,
) -> tuple[ValidatedRsnaFreeze, ...]:
    root = plan.roots.control_root / "freezes"
    if not root.exists() and not root.is_symlink():
        return ()
    if root.is_symlink() or not root.is_dir():
        raise ManifestBuildError("RSNA scientific freeze root is invalid")
    matches = []
    for directory in sorted(root.iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            raise ManifestBuildError("RSNA scientific freeze root contains invalid state")
        try:
            freeze = validate_freeze(
                directory,
                model_root=plan.roots.model_root,
                bundle_directory=plan.authority.bundle_directory,
            )
        except (OSError, ValueError, KeyError, ManifestBuildError) as exc:
            raise ManifestBuildError("Persisted RSNA scientific freeze is invalid") from exc
        matches.append(freeze)
    if len(matches) > 1:
        raise ManifestBuildError("RSNA execution has ambiguous scientific freezes")
    return tuple(matches)


def _ensure_training_report_derivatives(
    plan: ValidatedRsnaPlan,
    packages: Sequence[FrozenScientificPackage],
    runs: Sequence[RsnaFormalRunSpec],
) -> tuple[_TrainingDerivative, ...]:
    cohort = load_canonical_validation_cohort(
        plan.authority.bundle_directory,
        expected_bundle_id=plan.authority.bundle_id,
        expected_task_id=plan.authority.task_id,
        expected_label_policy_version=plan.authority.label_policy_version,
    )
    root = rsna_run_report_root(plan.roots.report_root)
    derivatives = []
    for frozen, spec in zip(packages, runs, strict=True):
        package = validate_rsna_model_package(plan.roots.model_root, frozen.model_package_id)
        package_directory = plan.roots.model_root / "packages" / frozen.model_package_id
        matches: list[Path] = []
        if root.is_dir() and not root.is_symlink():
            for candidate in sorted(root.iterdir()):
                if candidate.is_symlink() or not candidate.is_dir():
                    continue
                try:
                    report_document = json.loads((candidate / "metrics.json").read_bytes())
                    binding = (
                        report_document.get("training_package")
                        if isinstance(report_document, dict)
                        else None
                    )
                    if (
                        not isinstance(binding, dict)
                        or binding.get("model_package_id") != frozen.model_package_id
                    ):
                        continue
                    validate_training_report(
                        candidate,
                        run_id=candidate.name,
                        model_package_id=frozen.model_package_id,
                        family_id=spec.family_id,
                        seed=spec.seed,
                        package=package,
                        package_directory=package_directory,
                        canonical_validation_cohort=cohort,
                    )
                except (OSError, ValueError, KeyError, TypeError, ManifestBuildError):
                    continue
                matches.append(candidate)
        if not matches:
            destination = root / frozen.model_package_id
            if destination.exists() or destination.is_symlink():
                _quarantine_invalid_training_report(destination)
            stage = staging_directory(destination)
            try:
                regenerate_training_report(
                    stage,
                    run_id=destination.name,
                    model_package_id=frozen.model_package_id,
                    family_id=spec.family_id,
                    seed=spec.seed,
                    package=package,
                    package_directory=package_directory,
                    canonical_validation_cohort=cohort,
                )

                def validate_derivative(
                    path: Path,
                    *,
                    _run_id: str = destination.name,
                    _package_id: str = frozen.model_package_id,
                    _family_id: str = spec.family_id,
                    _seed: int = spec.seed,
                    _package: Mapping[str, Any] = package,
                    _package_directory: Path = package_directory,
                    **_kwargs: object,
                ) -> None:
                    validate_training_report(
                        path,
                        run_id=_run_id,
                        model_package_id=_package_id,
                        family_id=_family_id,
                        seed=_seed,
                        package=_package,
                        package_directory=_package_directory,
                        canonical_validation_cohort=cohort,
                    )

                install_immutable_directory(stage, destination, validate_derivative)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
            matches.append(destination)
        selected = min(matches, key=lambda path: path.name)
        derivatives.append(
            _TrainingDerivative(
                selected.name,
                frozen.model_package_id,
                frozen.model_path,
                selected,
            )
        )
    return tuple(derivatives)


def _quarantine_invalid_training_report(destination: Path) -> Path:
    """Move a colliding unusable derivative aside before canonical regeneration."""
    root = destination.parent
    quarantine = root / ".invalid-training-reports"
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ManifestBuildError("RSNA training report root is invalid")
    quarantine.mkdir(parents=True, exist_ok=True)
    if quarantine.is_symlink() or not quarantine.is_dir():
        raise ManifestBuildError("RSNA training report quarantine is invalid")
    for ordinal in range(1_000_000):
        target = quarantine / f"{destination.name}-{ordinal:06d}"
        if target.exists() or target.is_symlink():
            continue
        try:
            os.replace(destination, target)
        except OSError as exc:
            raise ManifestBuildError(
                "Invalid RSNA training report could not be quarantined"
            ) from exc
        return target
    raise ManifestBuildError("RSNA training report quarantine is exhausted")


def _preflight_rsna_campaign(
    *,
    backup_root: str | Path,
    bundle_id: str,
    source_root: str | Path = "data/raw/rsna/extracted",
) -> ValidatedRsnaPlan:
    """Validate every formal input before creating campaign-owned output."""
    repository_root = discover_repository_root()
    if Path.cwd().resolve() != repository_root:
        raise ManifestBuildError("Formal RSNA campaign must run from the repository root")
    _validate_canonical_rsna_layout(repository_root)
    source_input = Path(source_root)
    source_root = (
        source_input if source_input.is_absolute() else repository_root / source_input
    ).resolve()
    manifest_root = repository_root / "data/manifests"
    cache_root = repository_root / "data/cache/rsna"
    model_root = repository_root / "models/rsna"
    report_root = repository_root / "reports"
    private_root = repository_root / "private"
    control_root = private_root / "control/rsna"
    export_root = repository_root / "outbox"
    backup_input = Path(backup_root)
    backup = validate_writable_directory_destination(backup_input, "RSNA backup destination")
    if backup.is_relative_to(repository_root):
        raise ManifestBuildError("RSNA backup must reside outside the repository root")
    validate_preservation_paths(
        sources=(
            source_root,
            manifest_root,
            model_root,
            report_root / "rsna",
            private_root,
        ),
        export_root=export_root,
        backup_root=backup,
    )
    tracking_database = repository_root / "mlflow.db"
    for root in (
        cache_root,
        report_root,
        model_root,
        private_root,
        export_root,
    ):
        validate_writable_directory_destination(root, "RSNA campaign directory destination")
    validate_writable_file_destination(
        tracking_database, "RSNA campaign tracking database destination"
    )
    validate_existing_sqlite_database(tracking_database, "RSNA campaign tracking database")
    commit, dirty = git_revision(repository_root)
    if dirty:
        raise ManifestBuildError("Formal RSNA campaign requires a clean Git worktree")
    lock_hash = uv_lock_sha256(repository_root / "uv.lock")
    if not source_root.is_dir():
        raise FileNotFoundError(f"RSNA raw dataset is missing: {source_root}")
    required_raw = (
        source_root / "stage_2_train_images",
        source_root / "stage_2_train_labels.csv",
        source_root / "stage_2_detailed_class_info.csv",
    )
    if not all(path.is_dir() if path.suffix == "" else path.is_file() for path in required_raw):
        raise FileNotFoundError("RSNA raw dataset is incomplete")
    config_paths = tuple(repository_root / spec.config_relative for spec in RSNA_FORMAL_RUN_PLAN)
    if missing := [path for path in config_paths if not path.is_file()]:
        raise FileNotFoundError(f"RSNA campaign configs are missing: {missing}")
    configs = tuple(
        with_runtime(
            load_experiment_config(path, bundle_id=bundle_id),
            seed=spec.seed,
            manifest_directory=manifest_root,
            source_root=source_root,
            model_directory=model_root,
            report_directory=report_root,
            private_output_directory=private_root,
            device="cuda",
        )
        for spec, path in zip(RSNA_FORMAL_RUN_PLAN, config_paths, strict=True)
    )
    _validate_neural_campaign_configs(configs)
    _validate_config_agreement(configs)
    reference = configs[0]
    bundle_directory = (
        reference.runtime.manifest_directory
        / "rsna"
        / BUNDLES_DIRECTORY
        / reference.dataset.bundle_id
    )
    validate_bundle_reference(bundle_directory, expected_bundle_id=reference.dataset.bundle_id)
    validate_bundle_directory(
        bundle_directory,
        expected_bundle_id=reference.dataset.bundle_id,
    )
    dataset = get_dataset("rsna")
    if not isinstance(dataset, RsnaDataset):
        raise TypeError("RSNA campaign requires the concrete RSNA dataset adapter")
    cxr_reference = next(
        config
        for spec, config in zip(RSNA_FORMAL_RUN_PLAN, configs, strict=True)
        if spec.family_id == "cxr_densenet"
    )
    if cxr_reference.runtime.source_root is None:
        raise ManifestBuildError("RSNA campaign requires an explicit source root")
    validate_rsna_cache_source_availability(cxr_reference)
    validate_storage_capacity(
        (
            (repository_root, _RSNA_WORKSPACE_MINIMUM_FREE_BYTES),
            (export_root, 2 * _RSNA_PRESERVATION_COPY_BYTES),
            (backup, _RSNA_PRESERVATION_COPY_BYTES),
        ),
        context="RSNA campaign storage",
    )
    runtime = _required_cuda_runtime(cxr_reference)
    weight = fingerprint_pretrained_weights("densenet121-res224-chex")
    roots = RsnaFormalRoots(
        repository_root=repository_root,
        source_root=source_root,
        manifest_root=manifest_root,
        cache_root=cache_root,
        model_root=model_root,
        report_root=report_root,
        private_root=private_root,
        control_root=control_root,
        export_root=export_root,
        backup_root=backup,
        tracking_database=tracking_database,
    )
    authority = RsnaAuthorityCoordinates(
        dataset_id=reference.dataset.dataset_id,
        bundle_id=reference.dataset.bundle_id,
        task_id=reference.task.task_id,
        label_policy_version=reference.task.label_policy_version,
        bundle_directory=bundle_directory,
    )
    return ValidatedRsnaPlan(
        authority=authority,
        roots=roots,
        runs=RSNA_FORMAL_RUN_PLAN,
        configs=configs,
        dataset=dataset,
        git_commit=commit,
        dependency_lock_sha256=lock_hash,
        pretrained_weight=weight,
        runtime=runtime,
        training_execution=_training_execution_policy(cxr_reference, runtime),
        evaluation_execution=one_shot_loader_policy(pin_memory=runtime.pin_memory_effective),
        _guard=_FORMAL_PLAN_GUARD,
    )


def _validate_canonical_rsna_layout(repository_root: Path) -> None:
    """Reject redirected, aliased, or mistyped checkout-owned RSNA state."""
    root = repository_root.resolve()
    directory_roots = tuple(
        repository_root / path
        for path in (
            "data/manifests/rsna",
            "data/cache/rsna",
            "models/rsna/packages",
            "reports/rsna/runs",
            "reports/rsna/evaluations",
            "reports/rsna/seed-summaries",
            "reports/rsna/localization",
            "reports/rsna/comparisons",
            "reports/rsna/audit",
            "reports/rsna/campaigns",
            "private/control/rsna",
            "private/predictions/rsna",
            "private/localization",
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
            raise ManifestBuildError("Canonical RSNA layout is outside the repository") from exc
        current = repository_root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ManifestBuildError("Canonical RSNA layout contains a symlink redirect")
            if current.exists() and current != target and not current.is_dir():
                raise ManifestBuildError("Canonical RSNA layout has a parent type mismatch")
        if target.exists():
            expects_directory = target in directory_roots
            if (expects_directory and not target.is_dir()) or (
                not expects_directory and not target.is_file()
            ):
                raise ManifestBuildError("Canonical RSNA layout has a root type mismatch")
            if not target.resolve().is_relative_to(root):
                raise ManifestBuildError("Canonical RSNA layout escapes the repository")
    existing = [path for path in (*directory_roots, *file_roots) if path.exists()]
    for index, left in enumerate(existing):
        for right in existing[index + 1 :]:
            if left.samefile(right):
                raise ManifestBuildError("Canonical RSNA layout roots are aliased")


def _validate_neural_campaign_configs(configs: Sequence[ExperimentConfig]) -> None:
    """Validate all six neural configs against one campaign execution contract."""
    cxr = tuple(config for config in configs if config.family.family_id == "cxr_densenet")
    fusions = tuple(
        config for config in configs if config.family.family_id == "cxr_metadata_concat"
    )
    expected_seeds = tuple(
        spec.seed for spec in RSNA_FORMAL_RUN_PLAN if spec.family_id == "cxr_densenet"
    )
    if (
        tuple(require_runtime_seed(config) for config in cxr) != expected_seeds
        or tuple(require_runtime_seed(config) for config in fusions) != expected_seeds
    ):
        raise ValueError("RSNA campaign neural seeds differ from the formal run plan")
    cxr_configs = {
        json.dumps(package_scientific_config_payload(config), sort_keys=True) for config in cxr
    }
    fusion_configs = {
        json.dumps(package_scientific_config_payload(config), sort_keys=True) for config in fusions
    }
    if len(cxr_configs) != 1:
        raise ValueError("RSNA CXR configurations do not form one scientific family")
    if len(fusion_configs) != 1:
        raise ValueError("RSNA fusion configurations do not form one scientific family")
    neural_configs = (*cxr, *fusions)
    neural_settings = tuple(config.neural for config in neural_configs)
    if any(neural is None for neural in neural_settings):
        raise ValueError("RSNA campaign requires six complete neural configurations")
    resolved = cast(tuple[Any, ...], neural_settings)
    execution_contracts = {
        (
            neural.batch_size,
            config.runtime.num_workers,
            config.runtime.device,
            neural.mixed_precision,
            config.runtime.pin_memory_policy,
            neural.rotation_degrees,
            neural.translation_fraction,
            neural.brightness_jitter,
            neural.contrast_jitter,
        )
        for config, neural in zip(neural_configs, resolved, strict=True)
    }
    cache_identities = {
        preprocessing_identity(_transform(config, training=False)) for config in neural_configs
    }
    if (
        len(execution_contracts) != 1
        or len(cache_identities) != 1
        or resolved[0].batch_size != 32
        or neural_configs[0].runtime.device == "cpu"
        or neural_configs[0].runtime.pin_memory_policy == "disabled"
    ):
        raise ValueError("RSNA neural configurations do not share the campaign execution contract")


def _validate_config_agreement(configs: Sequence[ExperimentConfig]) -> None:
    coordinates = {
        (
            config.dataset.dataset_id,
            config.dataset.bundle_id,
            config.task.task_id,
            config.task.label_policy_version,
        )
        for config in configs
    }
    observed_matrix = tuple(
        (config.family.family_id, require_runtime_seed(config)) for config in configs
    )
    expected_matrix = tuple((spec.family_id, spec.seed) for spec in RSNA_FORMAL_RUN_PLAN)
    if (
        len(configs) != len(RSNA_FORMAL_RUN_PLAN)
        or len(coordinates) != 1
        or observed_matrix != expected_matrix
    ):
        raise ManifestBuildError("RSNA campaign configurations disagree on authority or task")


def _required_cuda_runtime(config: ExperimentConfig) -> ResolvedDevice:
    neural = config.neural
    if neural is None:
        raise ValueError("RSNA CXR configuration is incomplete")
    runtime = resolve_device(
        "cuda",
        mixed_precision=neural.mixed_precision,
        pin_memory_policy=config.runtime.pin_memory_policy,
    )
    if runtime.device.type != "cuda":
        raise RuntimeError("RSNA campaign neural runtime did not resolve to CUDA")
    return runtime


def _training_execution_policy(
    config: ExperimentConfig, runtime: ResolvedDevice
) -> LoaderExecutionPolicy:
    """Resolve the reviewed persistent policy for epoch-reused loaders."""
    neural = config.neural
    if neural is None:
        raise ValueError("RSNA CXR configuration is incomplete")
    return reused_loader_policy(
        num_workers=config.runtime.num_workers,
        pin_memory=runtime.pin_memory_effective,
    )


def _transform(config: ExperimentConfig, *, training: bool) -> StandardCxrTransform:
    neural = config.neural
    if neural is None:
        raise ValueError("RSNA CXR configuration is incomplete")
    return StandardCxrTransform(
        training=training,
        policy_version=str(config.preprocessing["cxr_transform_policy"]),
        image_size=int(config.family.parameters["image_size"]),
        rotation_degrees=neural.rotation_degrees,
        translation_fraction=neural.translation_fraction,
        brightness_jitter=neural.brightness_jitter,
        contrast_jitter=neural.contrast_jitter,
    )


def _validate_outputs(
    training_results: Sequence[Any],
    evaluation_results: Sequence[Any],
    audit_directory: Path,
    cxr_summary: Path,
    fusion_summary: Path,
    localization: Path,
    comparison: ComparisonResult,
    log_path: Path,
    *,
    roots: RsnaFormalRoots,
) -> None:
    private_root = roots.private_root
    if len(training_results) != 8 or len(evaluation_results) != 8:
        raise ValueError("RSNA campaign requires eight training and eight evaluation results")
    package_ids = tuple(result.model_package_id for result in training_results)
    evaluation_packages = tuple(result.model_package_id for result in evaluation_results)
    if len(set(package_ids)) != 8 or evaluation_packages != package_ids:
        raise ValueError("RSNA evaluation lineage does not match the exact package family")
    if len({result.evaluation_id for result in evaluation_results}) != 8:
        raise ValueError("RSNA evaluation IDs must be unique")
    required = [
        *(Path(result.model_path) for result in training_results),
        *(Path(result.artifact_directory) for result in training_results),
        *(Path(result.artifact_directory) for result in evaluation_results),
        audit_directory,
        cxr_summary,
        fusion_summary,
        localization,
        comparison.directory,
        log_path,
    ]
    for result in evaluation_results:
        private = getattr(result, "private_prediction_directory", None)
        if private is not None:
            required.append(Path(private))
    private_localization = private_root / "localization" / localization.name
    required.append(private_localization)
    if missing := [path for path in required if not path.exists()]:
        raise FileNotFoundError(f"Mandatory RSNA campaign outputs are missing: {missing}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the campaign and print its final transport paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute_rsna_campaign(backup_root=args.backup_root, bundle_id=args.bundle_id)
    except Exception as exc:
        print(f"RSNA campaign failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "campaign_id": result.campaign_id,
                "model_package_ids": list(result.model_package_ids),
                "evaluation_ids": list(result.evaluation_ids),
                "training_run_ids": list(result.training_run_ids),
                "evaluation_run_ids": list(result.evaluation_run_ids),
                "archive": result.archive_path.as_posix(),
                "checksum": result.checksum_path.as_posix(),
                "campaign_log": result.campaign_log_path.as_posix(),
            },
            indent=2,
        )
    )
    return 0
