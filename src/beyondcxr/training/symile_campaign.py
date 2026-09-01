"""Run the single safety-coupled Symile scientific campaign."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import mlflow

from beyondcxr.data.bundle_contract import BUNDLES_DIRECTORY
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_artifacts import validate_symile_bundle_reference
from beyondcxr.data.symile_cv import CV_DIRECTORY
from beyondcxr.data.symile_preprocess import LAB_OBSERVED_COLUMNS
from beyondcxr.data.symile_source import (
    DevelopmentReleaseAsset,
    establish_development_release_assets,
)
from beyondcxr.models.cxr_baseline import fingerprint_pretrained_weights
from beyondcxr.training.campaign_observability import (
    CampaignSession,
    campaign_log_members,
    campaign_session,
)
from beyondcxr.training.config import ExperimentConfig, load_symile_development_config, with_runtime
from beyondcxr.training.device import ResolvedDevice, resolve_device
from beyondcxr.training.operational_validation import (
    validate_existing_sqlite_database,
    validate_storage_capacity,
    validate_writable_directory_destination,
    validate_writable_file_destination,
)
from beyondcxr.training.preservation import (
    PreservationMember,
    PreservationRole,
    ValidatedRestorationContext,
    _validated_restoration_context,
    export_and_verify,
    require_restoration_context,
    validate_exact_restored_closure,
    validate_preservation_members,
    validate_preservation_paths,
)
from beyondcxr.training.symile_campaign_control import (
    PRETEST_FREEZE_PREFIX,
    CertifiedGlobalResult,
    ValidatedHeldOutCapability,
    ValidatedPretestFreeze,
    ValidatedSymileFrozenCampaign,
    _validate_preserved_test_open,
    held_out_operation,
    open_official_test,
    publish_error_review,
    publish_global_result,
    publish_pretest_freeze,
    require_certified_global_result,
    require_symile_frozen_campaign,
    require_validated_held_out_capability,
    validate_error_review,
    validate_global_result,
    validate_global_result_integrity,
    validate_pretest_freeze,
    validate_symile_execution_freeze_science,
    validate_symile_frozen_campaign,
)
from beyondcxr.training.symile_data import (
    SymileCxrStore,
    load_symile_development,
    load_symile_development_cohort,
)
from beyondcxr.training.symile_development import run_symile_development
from beyondcxr.training.symile_development_decision import (
    ValidatedDevelopmentDecision,
    publish_development_decision,
    validate_development_decision,
)
from beyondcxr.training.symile_development_reports import (
    FOCUSED_SUBGROUP_FILENAME,
    analyze_symile_development,
    derive_focused_subgroup_derivative,
    publish_focused_subgroup_derivative,
    publish_operation_analysis,
    publish_operation_subgroups,
    validate_analysis_result,
    validate_focused_subgroup_derivative,
)
from beyondcxr.training.symile_ecg_data import SymileEcgStore
from beyondcxr.training.symile_execution import (
    ValidatedCurrentSymileExecution,
    ValidatedSymileExecution,
    ValidatedSymileProspectiveMutation,
    build_symile_formal_preflight,
    publish_or_validate_symile_execution,
    symile_prospective_mutation,
    symile_test_is_open,
    validate_active_symile_execution,
    validate_current_symile_execution,
    validate_symile_execution,
    validate_symile_lifecycle,
)
from beyondcxr.training.symile_families import (
    DEVELOPMENT_FAMILY_ORDER,
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
from beyondcxr.training.symile_formal import SYMILE_FORMAL_RUN_PLAN
from beyondcxr.training.symile_statistics import focused_development_subgroups
from beyondcxr.training.symile_test_data import (
    FrozenSymileTestData,
    RestoredSymileTestData,
    validate_prediction_against_test_projection,
)
from beyondcxr.training.symile_test_inference import (
    publish_frozen_test_predictions,
    validate_restored_test_prediction_publication,
)
from beyondcxr.utils.immutable_json import immutable_json
from beyondcxr.utils.mlflow_utils import (
    configure_mlflow,
    discover_repository_root,
    environment_provenance,
    log_source_config,
    serialize_modalities,
    tracked_run,
)
from beyondcxr.utils.operational_logging import get_operational_logger, log_event, timed_phase
from beyondcxr.utils.private_predictions import (
    ValidatedPredictionEvidence,
    validate_prediction_evidence,
)
from beyondcxr.utils.publication import validate_path_component
from beyondcxr.utils.semantic_identity import canonical_json_bytes
from beyondcxr.utils.symile_publication import (
    ValidatedDevelopmentResult,
    validate_development_result,
    validated_development_repeat_oof,
)
from beyondcxr.utils.validation_context import operation_scoped_validation

_LOGGER = get_operational_logger(__name__)

_CONFIGS = {spec.family_id: spec.config_relative for spec in SYMILE_FORMAL_RUN_PLAN}
_SYMILE_WORKSPACE_MINIMUM_FREE_BYTES = 32 * 1024**3
# One complete preservation closure, including packages and prediction evidence.
# Reserve one copy for backup and two local copies: archive plus extraction.
_SYMILE_PRESERVATION_COPY_BYTES = 8 * 1024**3
_RESTORED_CAMPAIGN_GUARD = object()


def _symile_export_name(*, global_result_id: str, execution_id: str) -> str:
    """Return the specification-defined stable Symile preservation name."""
    return f"symile-{global_result_id}-{execution_id}"


class RestoredCampaignClosure:
    """Immutable view of the exact campaign state certified during restoration."""

    __slots__ = (
        "__freeze",
        "__execution",
        "__development_decision",
        "__analysis",
        "__subgroup",
        "__developments",
        "__packages",
        "__predictions",
        "__global_result",
        "__test_data",
        "__restored_root",
        "__restoration_context",
        "__authority_state",
        "__guard",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("RestoredCampaignClosure is created only by restoration validation")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("RestoredCampaignClosure is immutable")

    def __copy__(self) -> RestoredCampaignClosure:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> RestoredCampaignClosure:
        memo[id(self)] = self
        return self

    @property
    def freeze(self) -> ValidatedPretestFreeze:
        return self.__freeze

    @property
    def execution(self) -> ValidatedSymileExecution:
        return self.__execution

    @property
    def development_decision(self) -> ValidatedDevelopmentDecision:
        return self.__development_decision

    @property
    def analysis(self) -> Mapping[str, object]:
        return self.__analysis

    @property
    def subgroup(self) -> Mapping[str, object]:
        return self.__subgroup

    @property
    def developments(self) -> tuple[ValidatedDevelopmentResult, ...]:
        return self.__developments

    @property
    def packages(self) -> tuple[ValidatedFinalPackage, ...]:
        return self.__packages

    @property
    def predictions(self) -> tuple[ValidatedPredictionEvidence, ...]:
        return self.__predictions

    @property
    def global_result(self) -> CertifiedGlobalResult:
        return self.__global_result

    @property
    def test_data(self) -> FrozenSymileTestData:
        return self.__test_data

    @property
    def restored_root(self) -> Path:
        return self.__restored_root

    @property
    def restoration_context(self) -> ValidatedRestorationContext:
        return self.__restoration_context


def _restored_authority_state(value: RestoredCampaignClosure) -> tuple[object, ...]:
    return (
        immutable_json(value.freeze.manifest),
        immutable_json(value.execution.manifest),
        immutable_json(value.development_decision.manifest),
        tuple(immutable_json(item.manifest) for item in value.developments),
        tuple(immutable_json(item.manifest) for item in value.packages),
        tuple(immutable_json(item.manifest) for item in value.predictions),
    )


def require_restored_campaign(value: object) -> RestoredCampaignClosure:
    """Require a deeply certified campaign within its active restoration lifetime."""
    if (
        not isinstance(value, RestoredCampaignClosure)
        or value._RestoredCampaignClosure__guard is not _RESTORED_CAMPAIGN_GUARD
    ):
        raise PermissionError("Serving deployment requires a validated restored campaign")
    if not value.restored_root.is_dir():
        raise ManifestBuildError("Restored campaign authority is no longer available")
    restoration = require_restoration_context(value.restoration_context)
    if restoration.root != value.restored_root:
        raise PermissionError("Restored campaign belongs to another restoration root")
    if not value.global_result.directory.resolve().is_relative_to(value.restored_root):
        raise PermissionError("Certified global result belongs to another restoration root")
    require_certified_global_result(value.global_result)
    if _restored_authority_state(value) != value._RestoredCampaignClosure__authority_state:
        raise ManifestBuildError("Restored campaign authority changed after validation")
    return value


def run_symile_campaign(
    *,
    bundle_id: str,
    cv_assignment_id: str,
    source_root: str | Path,
    backup_root: str | Path,
) -> dict[str, str]:
    """Run or resume Symile through its immutable pre-test scientific freeze."""
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
    preservation_sources = _preservation_topology_sources(
        repository_root=repository_root,
        configurable_roots=(
            source_root,
            manifest_root,
            model_root,
            report_root,
            private_root,
        ),
    )
    validate_preservation_paths(
        sources=preservation_sources,
        export_root=export_root,
        backup_root=backup_root,
    )
    tracking_database = repository_root / "mlflow.db"
    for destination in (
        model_root,
        report_root,
        private_root,
        export_root,
        backup_root,
    ):
        validate_writable_directory_destination(
            destination, "Symile campaign directory destination"
        )
    validate_writable_file_destination(
        tracking_database, "Symile campaign tracking database destination"
    )
    validate_existing_sqlite_database(tracking_database, "Symile campaign tracking database")
    _validate_symile_storage_readiness(repository_root, backup_root)
    with campaign_session(
        repository_root=repository_root,
        report_root=report_root,
        study="symile",
        operation="development",
    ) as session:
        tracking_uri = f"sqlite:///{tracking_database}"
        control_root = Path(private_root) / "control" / "symile"
        if symile_test_is_open(
            control_root / "test-open.json",
            backup_root / "control/symile/test-open.json",
        ):
            raise ManifestBuildError(
                "Official Symile test data are already open; resume only the held-out phase"
            )
        with timed_phase(_LOGGER, "development_preflight"):
            configs = {
                family: with_runtime(
                    load_symile_development_config(
                        repository_root / path,
                        bundle_id=bundle_id,
                        cv_assignment_id=cv_assignment_id,
                    ),
                    manifest_directory=manifest_root,
                    source_root=source_root,
                    model_directory=Path(model_root) / "development",
                    report_directory=Path(report_root) / "development",
                    private_output_directory=private_root,
                    device="cuda",
                    num_workers=2,
                )
                for family, path in _CONFIGS.items()
            }
            development_assets = _preflight_symile_campaign(
                configs,
                source_root=source_root,
                require_pretrained_weights=True,
            )
            preflight = build_symile_formal_preflight(
                repository_root, bundle_id=bundle_id, cv_assignment_id=cv_assignment_id
            )
        frozen = _resolve_unique_pretest_state(
            control_root=control_root,
            model_root=model_root,
            report_root=report_root,
            private_root=private_root,
            manifest_root=manifest_root,
        )
        if frozen is not None:
            freeze, _decision, _packages = frozen
            execution = validate_active_symile_execution(control_root)
            current = validate_current_symile_execution(execution, preflight)
            session.bind_execution(execution.execution_id)
            lifecycle = validate_symile_lifecycle(
                current,
                repository_root=repository_root,
                private_root=private_root,
                backup_root=backup_root,
            )
            with symile_prospective_mutation(lifecycle):
                validate_symile_frozen_campaign(freeze=freeze, lifecycle=lifecycle)
                return {
                    "status": "pretest_ready",
                    "execution_id": current.execution.execution_id,
                    "freeze_id": freeze.freeze_id,
                }
        execution = publish_or_validate_symile_execution(preflight)
        session.bind_execution(execution.execution_id)
        current_execution = validate_current_symile_execution(
            execution,
            build_symile_formal_preflight(
                repository_root, bundle_id=bundle_id, cv_assignment_id=cv_assignment_id
            ),
        )
        lifecycle = validate_symile_lifecycle(
            current_execution,
            repository_root=repository_root,
            private_root=private_root,
            backup_root=backup_root,
        )
        with symile_prospective_mutation(lifecycle) as prospective_mutation:
            return _run_symile_prospective_campaign(
                prospective_mutation=prospective_mutation,
                repository_root=repository_root,
                configs=configs,
                source_root=source_root,
                manifest_root=manifest_root,
                model_root=model_root,
                report_root=report_root,
                private_root=private_root,
                control_root=control_root,
                tracking_uri=tracking_uri,
                current_execution=current_execution,
                development_assets=development_assets,
            )


def _run_symile_prospective_campaign(
    *,
    prospective_mutation: ValidatedSymileProspectiveMutation,
    repository_root: Path,
    configs: dict[str, ExperimentConfig],
    source_root: Path,
    manifest_root: Path,
    model_root: Path,
    report_root: Path,
    private_root: Path,
    control_root: Path,
    tracking_uri: str,
    current_execution: ValidatedCurrentSymileExecution,
    development_assets: tuple[DevelopmentReleaseAsset, ...] | None,
) -> dict[str, str]:
    """Create the complete prospectively fixed Symile scientific state."""
    with timed_phase(_LOGGER, "cross_validation"):
        developments = {}
        for family in SYMILE_CORE_DEVELOPMENT_FAMILIES:
            source_id = (
                developments["cxr_densenet"].manifest["development_id"]
                if family in {"cxr_labs_concat", "cxr_labs_gated", "cxr_labs_gated_no_observedness"}
                else None
            )
            developments[family] = run_symile_development(
                configs[family],
                prospective_mutation=prospective_mutation,
                source_cxr_development_id=source_id,
                tracking_uri=tracking_uri,
                repository_root=repository_root,
                development_assets=development_assets,
            )
        developments[SYMILE_ECG_GATED_FAMILY] = run_symile_development(
            configs[SYMILE_ECG_GATED_FAMILY],
            prospective_mutation=prospective_mutation,
            source_cxr_development_id=developments["cxr_densenet"].manifest["development_id"],
            tracking_uri=tracking_uri,
            repository_root=repository_root,
            development_assets=development_assets,
        )
    with timed_phase(_LOGGER, "development_decision"):
        development_decision = publish_development_decision(
            prospective_mutation=prospective_mutation,
            report_root=report_root,
            family_developments=developments,
            model_root=Path(model_root) / "development",
            prediction_root=private_root,
            manifest_root=manifest_root,
        )
        analyze_symile_development(
            [
                developments[family].manifest["development_id"]
                for family in SYMILE_CORE_DEVELOPMENT_FAMILIES
            ],
            report_root=Path(report_root) / "development",
            model_root=Path(model_root) / "development",
            prediction_root=private_root,
            manifest_root=manifest_root,
        )
        primary_repeat_oof = validated_development_repeat_oof(
            developments["cxr_labs_gated"], private_root
        )
        cxr_repeat_oof = validated_development_repeat_oof(
            developments["cxr_densenet"], private_root
        )
        cohort = load_symile_development_cohort(configs["cxr_labs_gated"]).frame
        attributes = cohort.loc[:, ["sample_id", "age_years", "sex", "view_position"]].copy()
        attributes["observed_lab_count"] = cohort.loc[:, LAB_OBSERVED_COLUMNS].sum(axis=1)
        publish_focused_subgroup_derivative(
            report_root=report_root,
            summary=focused_development_subgroups(cxr_repeat_oof, primary_repeat_oof, attributes),
        )
    with timed_phase(_LOGGER, "final_fitting"):
        provenance = {
            "git_commit": current_execution.execution.manifest["git_commit"],
            "git_dirty": False,
            "dependency_lock_sha256": current_execution.execution.manifest[
                "dependency_lock_sha256"
            ],
            "python_version": current_execution.execution.manifest["framework_versions"]["python"],
        }
        final_configs = {
            family: with_runtime(
                configs[family],
                manifest_directory=manifest_root,
                source_root=source_root,
                model_directory=model_root,
                report_directory=report_root,
                private_output_directory=private_root,
                device="cuda",
                num_workers=2,
            )
            for family in (*FINAL_NEURAL_FAMILIES, *FINAL_TABULAR_FAMILIES)
        }
        packages = []
        for family in FINAL_TABULAR_FAMILIES:
            config = final_configs[family]
            packages.append(
                _tracked_final_fit(
                    config,
                    tracking_uri=tracking_uri,
                    member_seed=FINAL_TABULAR_SEED,
                    provenance=provenance,
                    fit=lambda operational, family=family, config=config: fit_final_tabular_family(
                        config,
                        prospective_mutation=prospective_mutation,
                        development=developments[family],
                        operational=operational,
                    ),
                )
            )
        cxr_by_seed = {}
        for seed in FINAL_NEURAL_MEMBER_SEEDS:
            config = final_configs["cxr_densenet"]
            package = _tracked_final_fit(
                config,
                tracking_uri=tracking_uri,
                member_seed=seed,
                provenance=provenance,
                fit=lambda operational, seed=seed, config=config: fit_final_neural_member(
                    config,
                    prospective_mutation=prospective_mutation,
                    development=developments["cxr_densenet"],
                    member_seed=seed,
                    expected_pretrained_scientific_identity=development_decision.manifest[
                        "selected_family_authorities"
                    ]["cxr_densenet"]["pretrained_scientific_identity"],
                    operational=operational,
                    development_assets=development_assets,
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
                        tracking_uri=tracking_uri,
                        member_seed=seed,
                        provenance=provenance,
                        fit=lambda operational, family=family, seed=seed, config=config: (
                            fit_final_neural_member(
                                config,
                                prospective_mutation=prospective_mutation,
                                development=developments[family],
                                member_seed=seed,
                                expected_pretrained_scientific_identity=None,
                                source_cxr_package=cxr_by_seed[seed],
                                operational=operational,
                                development_assets=development_assets,
                            )
                        ),
                    )
                )
    with timed_phase(_LOGGER, "pretest_freeze"):
        first = configs["labs_logistic"]
        freeze = publish_pretest_freeze(
            prospective_mutation=prospective_mutation,
            control_root=control_root,
            bundle_id=first.dataset.bundle_id,
            cv_assignment_id=str(first.dataset.cv_assignment_id),
            task={
                "task_id": first.task.task_id,
                "label_policy_version": first.task.label_policy_version,
            },
            development_decision=development_decision,
            final_packages=packages,
        )
    return {
        "status": "pretest_ready",
        "execution_id": current_execution.execution.execution_id,
        "freeze_id": freeze.freeze_id,
    }


def _preservation_topology_sources(
    *,
    repository_root: Path,
    configurable_roots: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Return mutually nonredundant roots for early preservation topology validation."""
    checkout_sources = tuple(
        (repository_root / name).absolute() for name in ("data", "models", "reports", "private")
    )
    external: list[Path] = []
    for configured in configurable_roots:
        candidate = configured.absolute()
        if any(candidate.is_relative_to(root) for root in checkout_sources):
            continue
        if any(candidate.is_relative_to(root) for root in external):
            continue
        external = [root for root in external if not root.is_relative_to(candidate)]
        external.append(candidate)
    return (*checkout_sources, *external)


def complete_symile_test_phase(
    *,
    session: CampaignSession,
    frozen_campaign: ValidatedSymileFrozenCampaign,
    development_decision: ValidatedDevelopmentDecision,
    packages: tuple[ValidatedFinalPackage, ...],
    source_root: Path,
    manifest_root: Path,
    model_root: Path,
    report_root: Path,
    private_root: Path,
    export_root: Path,
    backup_root: Path,
    runtime: ResolvedDevice,
) -> dict[str, str]:
    """Open or resume only the irreversible/post-open Symile phase."""
    frozen_campaign = require_symile_frozen_campaign(frozen_campaign)
    lifecycle = frozen_campaign.lifecycle
    if (
        Path(private_root).resolve() != lifecycle.private_root
        or Path(backup_root).resolve() != lifecycle.backup_root
    ):
        raise ManifestBuildError("Symile held-out phase roots differ from validated lifecycle")
    with timed_phase(_LOGGER, "test_opening"):
        capability = open_official_test(
            frozen_campaign=frozen_campaign,
        )
    with held_out_operation(capability) as capability:
        return _complete_opened_campaign(
            session=session,
            capability=capability,
            development_decision=development_decision,
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


def resume_symile_test_phase(
    *,
    source_root: str | Path,
    backup_root: str | Path,
) -> dict[str, str]:
    """Resolve the sole frozen execution and open or resume its held-out phase."""
    repository_root = discover_repository_root()
    if Path.cwd().resolve() != repository_root:
        raise ManifestBuildError("Formal Symile campaign must run from the repository root")
    _validate_canonical_formal_layout(repository_root)
    source = Path(source_root).resolve()
    backup = _repository_path(repository_root, backup_root)
    if backup.resolve().is_relative_to(repository_root.resolve()):
        raise ManifestBuildError("Symile backup must reside outside the repository root")
    manifest_root = repository_root / "data" / "manifests"
    model_root = repository_root / "models" / "symile"
    report_root = repository_root / "reports" / "symile"
    private_root = repository_root / "private"
    export_root = repository_root / "outbox"
    control_root = private_root / "control" / "symile"
    preservation_sources = _preservation_topology_sources(
        repository_root=repository_root,
        configurable_roots=(source, manifest_root, model_root, report_root, private_root),
    )
    validate_preservation_paths(
        sources=preservation_sources,
        export_root=export_root,
        backup_root=backup,
    )
    for destination in (model_root, report_root, private_root, export_root, backup):
        validate_writable_directory_destination(
            destination, "Symile campaign directory destination"
        )
    _validate_symile_storage_readiness(repository_root, backup)
    with campaign_session(
        repository_root=repository_root,
        report_root=report_root,
        study="symile",
        operation="test",
    ) as session:
        with timed_phase(_LOGGER, "test_preflight"):
            execution = validate_active_symile_execution(control_root)
            bundle_id = execution.manifest["dataset"]["bundle_id"]
            cv_assignment_id = execution.manifest["dataset"]["cv_assignment_id"]
            configs = {
                family: with_runtime(
                    load_symile_development_config(
                        repository_root / path,
                        bundle_id=bundle_id,
                        cv_assignment_id=cv_assignment_id,
                    ),
                    manifest_directory=manifest_root,
                    source_root=source,
                    model_directory=model_root / "development",
                    report_directory=report_root / "development",
                    private_output_directory=private_root,
                    device="cuda",
                    num_workers=2,
                )
                for family, path in _CONFIGS.items()
            }
            _preflight_symile_campaign(configs, source_root=source, require_pretrained_weights=True)
            preflight = build_symile_formal_preflight(
                repository_root, bundle_id=bundle_id, cv_assignment_id=cv_assignment_id
            )
            execution = validate_active_symile_execution(control_root)
            current = validate_current_symile_execution(execution, preflight)
            session.bind_execution(execution.execution_id)
            lifecycle = validate_symile_lifecycle(
                current,
                repository_root=repository_root,
                private_root=private_root,
                backup_root=backup,
            )
            resolved = _resolve_unique_pretest_state(
                control_root=control_root,
                model_root=model_root,
                report_root=report_root,
                private_root=private_root,
                manifest_root=manifest_root,
            )
            if resolved is None:
                raise ManifestBuildError(
                    "No completed Symile pre-test freeze exists; run `make symile` first"
                )
            freeze, decision, packages = resolved
            frozen_campaign = validate_symile_frozen_campaign(freeze=freeze, lifecycle=lifecycle)
            neural_config = configs["cxr_labs_gated"]
            if neural_config.neural is None:
                raise ManifestBuildError("Symile formal neural runtime configuration is missing")
            runtime = resolve_device(
                neural_config.runtime.device,
                mixed_precision=neural_config.neural.mixed_precision,
                pin_memory_policy=neural_config.runtime.pin_memory_policy,
            )
        return complete_symile_test_phase(
            session=session,
            frozen_campaign=frozen_campaign,
            development_decision=decision,
            packages=packages,
            source_root=source,
            manifest_root=manifest_root,
            model_root=model_root,
            report_root=report_root,
            private_root=private_root,
            export_root=export_root,
            backup_root=backup,
            runtime=runtime,
        )


def _preflight_symile_campaign(
    configs: dict[str, ExperimentConfig],
    *,
    source_root: Path,
    require_pretrained_weights: bool = True,
) -> tuple[DevelopmentReleaseAsset, ...]:
    """Prove all formal authorities, source assets, and runtimes before development."""
    if set(configs) != set(_CONFIGS):
        raise ManifestBuildError("Symile campaign configuration family set is incomplete")
    typed = [configs[family] for family in _CONFIGS]
    coordinates = {
        (
            config.dataset.dataset_id,
            config.dataset.bundle_id,
            config.dataset.cv_assignment_id,
            config.task.task_id,
            config.task.label_policy_version,
        )
        for config in typed
    }
    if len(coordinates) != 1 or any(
        config.family.family_id != family for family, config in configs.items()
    ):
        raise ManifestBuildError("Symile campaign configurations disagree on authority or family")
    reference_config = typed[0]
    bundle_directory = (
        reference_config.runtime.manifest_directory
        / "symile"
        / BUNDLES_DIRECTORY
        / reference_config.dataset.bundle_id
    )
    data = load_symile_development(reference_config)
    reference = validate_symile_bundle_reference(
        bundle_directory,
        expected_bundle_id=reference_config.dataset.bundle_id,
    )
    development_assets = establish_development_release_assets(
        source_root,
        checksum_manifest_sha256=str(reference.manifest["source"]["checksum_manifest_sha256"]),
    )
    SymileCxrStore(data.bundle, source_root, development_assets=development_assets)
    SymileEcgStore(data.bundle, source_root, development_assets=development_assets)
    neural_weights = {
        str(config.family.parameters["weights"]) for config in typed if config.neural is not None
    }
    if len(neural_weights) != 1:
        raise ManifestBuildError("Symile neural configurations disagree on pretrained weights")
    if require_pretrained_weights:
        fingerprint_pretrained_weights(next(iter(neural_weights)))
    runtime_contracts = set()
    for config in typed:
        if config.neural is None:
            continue
        runtime = resolve_device(
            config.runtime.device,
            mixed_precision=config.neural.mixed_precision,
            pin_memory_policy=config.runtime.pin_memory_policy,
        )
        runtime_contracts.add(
            (
                runtime.device.type,
                runtime.mixed_precision_effective,
                runtime.pin_memory_effective,
                config.runtime.num_workers,
            )
        )
    if len(runtime_contracts) != 1:
        raise ManifestBuildError("Symile neural configurations disagree on runtime contract")
    return development_assets


def _resolve_unique_pretest_state(
    *,
    control_root: Path,
    model_root: Path,
    report_root: Path,
    private_root: Path,
    manifest_root: Path,
) -> (
    tuple[
        ValidatedPretestFreeze,
        ValidatedDevelopmentDecision,
        tuple[ValidatedFinalPackage, ...],
    ]
    | None
):
    freezes_root = control_root / "freezes"
    if not freezes_root.exists() and not freezes_root.is_symlink():
        return None
    if freezes_root.is_symlink() or not freezes_root.is_dir():
        raise ManifestBuildError("Symile scientific freeze root is invalid")
    entries = tuple(freezes_root.iterdir())
    if any(path.is_symlink() or not path.is_dir() for path in entries):
        raise ManifestBuildError("Symile scientific freeze root contains invalid state")
    directories = entries
    if not directories:
        return None
    if len(directories) != 1:
        raise ManifestBuildError("Symile scientific freeze selection is ambiguous")
    freeze_root = directories[0]
    freeze_id = freeze_root.name
    if not freeze_id.startswith(PRETEST_FREEZE_PREFIX):
        raise ManifestBuildError("Persisted pre-test freeze identity is malformed")
    validate_path_component(freeze_id, "Symile pre-test freeze identity")
    freeze_manifest_path = freeze_root / "manifest.json"
    if freeze_root.is_symlink() or not freeze_root.is_dir() or freeze_manifest_path.is_symlink():
        raise ManifestBuildError("Persisted pre-test freeze is invalid")
    try:
        frozen = json.loads(freeze_manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Persisted pre-test freeze is malformed") from exc
    decision_id = frozen.get("development_decision_id") if isinstance(frozen, dict) else None
    package_refs = frozen.get("final_packages") if isinstance(frozen, dict) else None
    if (
        not isinstance(decision_id, str)
        or not isinstance(package_refs, list)
        or len(package_refs) != FINAL_PACKAGE_COUNT
        or any(
            not isinstance(reference, dict)
            or set(reference) != {"package_id", "family_id", "seed"}
            or not isinstance(reference["package_id"], str)
            for reference in package_refs
        )
    ):
        raise ManifestBuildError("Persisted pre-test freeze is malformed")
    validate_path_component(decision_id, "Symile development decision identity")
    for reference in package_refs:
        validate_path_component(reference["package_id"], "Final package identity")
    development_decision = validate_development_decision(
        report_root / "development-decisions" / decision_id,
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
        development_decision=development_decision,
        final_packages=packages,
    )
    return freeze, development_decision, packages


def _decision_family_ids(
    development_decision: ValidatedDevelopmentDecision,
) -> dict[str, str]:
    references = development_decision.manifest.get("family_development_ids")
    if (
        not isinstance(references, list)
        or [item.get("family_id") for item in references if isinstance(item, dict)]
        != list(DEVELOPMENT_FAMILY_ORDER)
        or any(
            not isinstance(item, dict)
            or set(item) != {"family_id", "development_id"}
            or not isinstance(item["development_id"], str)
            for item in references
        )
    ):
        raise ManifestBuildError("Symile development decision family membership is invalid")
    return {str(item["family_id"]): str(item["development_id"]) for item in references}


def _ensure_decision_subgroup_derivative(
    development_decision: ValidatedDevelopmentDecision,
    *,
    report_root: Path,
    model_root: Path,
    private_root: Path,
    manifest_root: Path,
) -> Path:
    family_ids = _decision_family_ids(development_decision)
    primary = validate_development_result(
        report_root / "development/families" / family_ids["cxr_labs_gated"],
        model_root=model_root / "development",
        prediction_root=private_root,
        manifest_root=manifest_root,
        expected_development_id=family_ids["cxr_labs_gated"],
    )
    summary = derive_focused_subgroup_derivative(
        cxr_development_id=family_ids["cxr_densenet"],
        primary_repeats=validated_development_repeat_oof(primary, private_root),
        report_root=report_root,
        model_root=model_root / "development",
        prediction_root=private_root,
        manifest_root=manifest_root,
    )
    return publish_focused_subgroup_derivative(report_root=report_root, summary=summary)


def _complete_opened_campaign(
    *,
    session: CampaignSession,
    capability: ValidatedHeldOutCapability,
    development_decision: ValidatedDevelopmentDecision,
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
    freeze = capability.freeze
    with timed_phase(_LOGGER, "held_out_predictions"):
        predictions, test_data = publish_frozen_test_predictions(
            capability=capability,
            final_packages=packages,
            source_root=source_root,
            manifest_root=manifest_root,
            private_root=private_root,
            runtime=runtime,
        )
    with timed_phase(_LOGGER, "global_result"):
        global_result = publish_global_result(
            report_root=report_root,
            capability=freeze,
            predictions=predictions,
            final_packages=packages,
            test_data=test_data,
        )
    with timed_phase(_LOGGER, "error_review"):
        error_review = publish_error_review(
            private_root=private_root,
            capability=freeze,
            predictions=predictions,
            final_packages=packages,
            test_data=test_data,
        )
    export_members = _campaign_export_members(
        freeze=freeze,
        execution=capability.execution,
        opening_path=capability.opening.path,
        live_capability=capability,
        development_decision=development_decision,
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
    log_event(_LOGGER, "campaign_ready_for_export", phase="export")
    export_members += campaign_log_members(
        session.log_path, study="symile", execution_id=capability.execution.execution_id
    )

    archive_started_at = time.perf_counter()
    archive = export_and_verify(
        members=export_members,
        export_root=export_root,
        backup_root=backup_root,
        export_name=_symile_export_name(
            global_result_id=global_result.result_id,
            execution_id=capability.execution.execution_id,
        ),
        restoration_validator=validate_restored_campaign,
    )
    log_event(_LOGGER, "archive_created", elapsed_s=time.perf_counter() - archive_started_at)
    return {
        "development_decision_id": development_decision.decision_id,
        "pretest_freeze_id": freeze.freeze_id,
        "global_result_id": global_result.result_id,
        "export": archive.as_posix(),
    }


@operation_scoped_validation
def validate_restored_campaign(restored_root: Path) -> RestoredCampaignClosure:
    """Recursively validate one campaign from its restored artifact tree."""
    restoration = _validated_restoration_context(restored_root)
    restored_root = restoration.root
    manifest_root = restored_root / "data" / "manifests"
    model_root = restored_root / "models" / "symile"
    report_root = restored_root / "reports" / "symile"
    private_root = restored_root / "private"
    control_root = private_root / "control" / "symile"
    test_open_path = control_root / "test-open.json"
    try:
        opened = json.loads(test_open_path.read_bytes())
        freeze_id = opened["freeze_id"]
        validate_path_component(freeze_id, "Restored Symile freeze identity")
        frozen = json.loads((control_root / "freezes" / freeze_id / "manifest.json").read_bytes())
        decision_id = frozen["development_decision_id"]
        package_references = frozen["final_packages"]
    except (OSError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Restored campaign control lineage is invalid") from exc
    validate_path_component(decision_id, "Restored development decision identity")
    if not isinstance(package_references, list) or len(package_references) != FINAL_PACKAGE_COUNT:
        raise ManifestBuildError("Restored campaign package lineage is invalid")
    package_ids = []
    for reference in package_references:
        if not isinstance(reference, dict) or not isinstance(reference.get("package_id"), str):
            raise ManifestBuildError("Restored campaign package lineage is invalid")
        validate_path_component(reference["package_id"], "Restored final package identity")
        package_ids.append(reference["package_id"])
    development_decision = validate_development_decision(
        report_root / "development-decisions" / decision_id,
        report_root=report_root,
        model_root=model_root / "development",
        prediction_root=private_root,
        manifest_root=manifest_root,
    )
    analysis_path = analyze_symile_development(
        [
            item["development_id"]
            for item in development_decision.manifest["family_development_ids"]
            if item["family_id"] in SYMILE_CORE_DEVELOPMENT_FAMILIES
        ],
        report_root=report_root / "development",
        model_root=model_root / "development",
        prediction_root=private_root,
        manifest_root=manifest_root,
    )
    analysis = validate_analysis_result(
        analysis_path,
        report_root=report_root / "development",
        model_root=model_root / "development",
        prediction_root=private_root,
        manifest_root=manifest_root,
    )
    development_ids = tuple(
        item["development_id"] for item in development_decision.manifest["family_development_ids"]
    )
    developments = tuple(
        validate_development_result(
            report_root / "development" / "families" / development_id,
            model_root=model_root / "development",
            prediction_root=private_root,
            manifest_root=manifest_root,
            expected_development_id=development_id,
        )
        for development_id in development_ids
    )
    subgroup_path = _ensure_decision_subgroup_derivative(
        development_decision,
        report_root=report_root,
        model_root=model_root,
        private_root=private_root,
        manifest_root=manifest_root,
    )
    subgroup = validate_focused_subgroup_derivative(subgroup_path)
    packages = tuple(
        validate_final_package(
            model_root / "final" / "packages" / package_id,
            expected_package_id=package_id,
        )
        for package_id in package_ids
    )
    freeze = validate_pretest_freeze(
        control_root / "freezes" / freeze_id,
        development_decision=development_decision,
        final_packages=packages,
    )
    active_execution = control_root / "active-execution.json"
    try:
        active_raw = active_execution.read_bytes()
        active_document = json.loads(active_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Restored active execution receipt is unreadable") from exc
    execution_id = (
        active_document.get("execution_id") if isinstance(active_document, dict) else None
    )
    expected_active = {"schema_version": 1, "execution_id": execution_id}
    if (
        active_execution.is_symlink()
        or not active_execution.is_file()
        or not isinstance(execution_id, str)
        or active_document != expected_active
        or active_raw != canonical_json_bytes(expected_active)
    ):
        raise ManifestBuildError("Restored active execution receipt is invalid")
    validate_path_component(execution_id, "Restored Symile execution identity")
    execution = validate_symile_execution(
        control_root / "executions" / execution_id,
        expected_execution_id=execution_id,
    )
    validate_symile_execution_freeze_science(
        execution,
        freeze,
        report_root=report_root,
        model_root=model_root,
        private_root=private_root,
        manifest_root=manifest_root,
    )
    opening_evidence = _validate_preserved_test_open(
        path=test_open_path,
        restoration=restoration,
        freeze=freeze,
        execution=execution,
    )
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
    test_data = RestoredSymileTestData(
        opening_evidence,
        freeze,
        load_final_package_config(packages[0]),
        manifest_root=manifest_root,
    )
    projection = test_data.evaluation_projection()
    for evidence in predictions:
        validate_restored_test_prediction_publication(
            evidence,
            freeze=freeze,
            execution=execution,
            opening=opening_evidence,
        )
        validate_prediction_against_test_projection(
            evidence,
            capability=freeze,
            projection=projection,
            expected_package_ids=tuple(package_ids),
        )
    global_directories = tuple((report_root / "global-results").iterdir())
    if len(global_directories) != 1:
        raise ManifestBuildError("Restored campaign result membership is invalid")
    global_result = validate_global_result(
        global_directories[0],
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
    validate_exact_restored_closure(
        restored_root,
        _project_campaign_export_members(
            freeze=freeze,
            execution=execution,
            opening_path=opening_evidence.path,
            development_decision=development_decision,
            developments=developments,
            packages=packages,
            predictions=predictions,
            global_result=global_result,
            error_review=error_review,
            manifest_root=manifest_root,
            model_root=model_root,
            report_root=report_root,
            private_root=private_root,
            analysis_path=analysis_path,
            subgroup_path=subgroup_path,
        )
        + campaign_log_members(
            report_root / "campaigns/export/execution.log",
            study="symile",
            execution_id=execution.execution_id,
        ),
    )
    restored = object.__new__(RestoredCampaignClosure)
    fields = {
        "freeze": freeze,
        "execution": execution,
        "development_decision": development_decision,
        "analysis": immutable_json(analysis),
        "subgroup": immutable_json(subgroup),
        "developments": tuple(developments),
        "packages": tuple(packages),
        "predictions": tuple(predictions),
        "global_result": require_certified_global_result(global_result),
        "test_data": test_data,
        "restored_root": restored_root.resolve(),
        "restoration_context": restoration,
    }
    for name, item in fields.items():
        object.__setattr__(restored, f"_RestoredCampaignClosure__{name}", item)
    object.__setattr__(
        restored,
        "_RestoredCampaignClosure__authority_state",
        _restored_authority_state(restored),
    )
    object.__setattr__(
        restored,
        "_RestoredCampaignClosure__guard",
        _RESTORED_CAMPAIGN_GUARD,
    )
    return restored


def _campaign_export_members(
    *,
    freeze: ValidatedPretestFreeze,
    execution: ValidatedSymileExecution,
    opening_path: Path,
    live_capability: ValidatedHeldOutCapability | None,
    development_decision: ValidatedDevelopmentDecision,
    packages: list[ValidatedFinalPackage] | tuple[ValidatedFinalPackage, ...],
    predictions: tuple[ValidatedPredictionEvidence, ...],
    global_result: CertifiedGlobalResult,
    error_review: Path,
    test_data: FrozenSymileTestData,
    manifest_root: Path,
    model_root: Path,
    report_root: Path,
    private_root: Path,
) -> tuple[PreservationMember, ...]:
    """Resolve the exact recursively validated authority closure for one campaign."""
    global_result = require_certified_global_result(global_result)
    if live_capability is None or live_capability._operation is None:
        raise PermissionError("Preservation requires an active held-out operation")
    live_capability = require_validated_held_out_capability(live_capability)
    analysis_path = publish_operation_analysis(
        live_capability, report_root=report_root / "development"
    )
    validated_freeze = live_capability.freeze
    validated_decision = validated_freeze.development_decision
    validated_packages = validated_freeze.final_packages
    developments = live_capability.frozen_campaign.developments
    if (
        validated_freeze != freeze
        or validated_decision != development_decision
        or tuple(package.package_id for package in validated_packages)
        != tuple(package.package_id for package in packages)
        or live_capability.execution != execution
        or live_capability.opening.path != opening_path
    ):
        raise ManifestBuildError("Campaign preservation lifecycle authorities disagree")
    validated_global_result = validate_global_result_integrity(
        report_root / "global-results" / global_result.result_id,
        capability=freeze,
        predictions=predictions,
        final_packages=packages,
        test_data=test_data,
    )
    if not global_result.matches_integrity(validated_global_result):
        raise ManifestBuildError("Campaign global result changed before export")
    validate_error_review(
        error_review,
        capability=freeze,
        predictions=predictions,
        final_packages=packages,
        test_data=test_data,
    )

    validated_predictions = tuple(
        validate_prediction_evidence(
            private_root / "predictions" / "symile" / "test" / prediction.prediction_id,
            expected_prediction_id=prediction.prediction_id,
        )
        for prediction in predictions
    )
    subgroup_path = publish_operation_subgroups(
        live_capability,
        report_root=report_root,
        model_root=model_root,
        manifest_root=manifest_root,
    )
    return _project_campaign_export_members(
        freeze=validated_freeze,
        execution=execution,
        opening_path=opening_path,
        development_decision=validated_decision,
        developments=tuple(developments),
        packages=validated_packages,
        predictions=validated_predictions,
        global_result=global_result,
        error_review=error_review,
        manifest_root=manifest_root,
        model_root=model_root,
        report_root=report_root,
        private_root=private_root,
        analysis_path=analysis_path,
        subgroup_path=subgroup_path,
    )


def _project_campaign_export_members(
    *,
    freeze: ValidatedPretestFreeze,
    execution: ValidatedSymileExecution,
    opening_path: Path,
    development_decision: ValidatedDevelopmentDecision,
    developments: tuple[ValidatedDevelopmentResult, ...],
    packages: tuple[ValidatedFinalPackage, ...],
    predictions: tuple[ValidatedPredictionEvidence, ...],
    global_result: CertifiedGlobalResult,
    error_review: Path,
    manifest_root: Path,
    model_root: Path,
    report_root: Path,
    private_root: Path,
    analysis_path: Path,
    subgroup_path: Path,
) -> tuple[PreservationMember, ...]:
    """Project already validated campaign authorities into exact archive membership."""
    cv_ids = {item.manifest["scientific_context"]["cv_assignment_id"] for item in developments}
    if len(cv_ids) != 1 or None in cv_ids:
        raise ManifestBuildError("Campaign data authorities are inconsistent")
    cv_id = str(next(iter(cv_ids)))

    members: list[PreservationMember] = []

    def add(
        path: Path,
        restore_relative: Path,
        *,
        role: PreservationRole = "scientific_authority",
    ) -> None:
        if (
            path.is_symlink()
            or not path.exists()
            or restore_relative.is_absolute()
            or ".." in restore_relative.parts
        ):
            raise ManifestBuildError("Campaign export authority path is invalid")
        members.append(PreservationMember(path, restore_relative, role))

    bundle_id = str(freeze.manifest["bundle_id"])
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
    add(analysis_path, Path("reports/symile/development/analyses/core"), role="derivative")
    add(
        subgroup_path,
        Path("reports/symile/development-subgroups") / FOCUSED_SUBGROUP_FILENAME,
        role="derivative",
    )
    add(
        development_decision.directory,
        Path("reports/symile/development-decisions") / development_decision.decision_id,
    )
    for package in packages:
        add(
            package.directory,
            Path("models/symile/final/packages") / package.package_id,
        )
    add(
        freeze.directory,
        Path("private/control/symile/freezes") / freeze.freeze_id,
    )
    add(
        execution.directory,
        Path("private/control/symile/executions") / execution.execution_id,
        role="operational_evidence",
    )
    add(
        execution.directory.parent.parent / "active-execution.json",
        Path("private/control/symile/active-execution.json"),
        role="operational_evidence",
    )
    add(
        opening_path,
        Path("private/control/symile/test-open.json"),
        role="operational_evidence",
    )
    for evidence in predictions:
        add(
            evidence.directory,
            Path("private/predictions/symile/test") / evidence.prediction_id,
        )
    add(
        global_result.directory,
        Path("reports/symile/global-results") / global_result.result_id,
    )
    add(
        error_review,
        Path("private/error-review/symile") / error_review.name,
        role="derivative",
    )
    validate_preservation_members(members)
    return tuple(members)


def _tracked_final_fit(
    config,
    *,
    tracking_uri: str,
    member_seed: int,
    provenance: dict[str, object],
    fit: Callable[[dict[str, object]], ValidatedFinalPackage],
) -> ValidatedFinalPackage:
    configure_mlflow(
        experiment_name=config.runtime.experiment_name,
        tracking_uri=tracking_uri,
    )
    tags = {
        "run_kind": "final_training",
        "evaluation_scope": "full_development",
        "dataset_id": config.dataset.dataset_id,
        "task_id": config.task.task_id,
        "label_policy_version": config.task.label_policy_version,
        "family_id": config.family.family_id,
        "modalities": serialize_modalities(config.family.modalities),
        "bundle_id": config.dataset.bundle_id,
        "member_seed": str(member_seed),
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


def _validate_symile_storage_readiness(
    repository_root: Path,
    backup_root: Path,
) -> None:
    """Account for working state, local archive/extraction, and external backup."""
    validate_storage_capacity(
        (
            (repository_root, _SYMILE_WORKSPACE_MINIMUM_FREE_BYTES),
            (repository_root / "outbox", 2 * _SYMILE_PRESERVATION_COPY_BYTES),
            (backup_root, _SYMILE_PRESERVATION_COPY_BYTES),
        ),
        context="Symile campaign storage",
    )


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
            "reports/symile/development-decisions",
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
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--cv-assignment-id", required=True)
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
