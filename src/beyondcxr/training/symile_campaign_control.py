"""Scientific freeze, held-out authorization, and global-result validation for Symile."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.calibration import calibration_curve

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_schemas import LABEL_POLICY_VERSION, TASK_ID
from beyondcxr.training.campaign_lock import campaign_lock
from beyondcxr.training.preservation import (
    ValidatedRestorationContext,
    require_restoration_context,
)
from beyondcxr.training.symile_development_decision import (
    DEVELOPMENT_DECISION_POLICY,
    ValidatedDevelopmentDecision,
    require_validated_development_decision,
    validate_development_decision,
)
from beyondcxr.training.symile_execution import (
    ValidatedCurrentSymileExecution,
    ValidatedSymileExecution,
    ValidatedSymileLifecycle,
    ValidatedSymileProspectiveMutation,
    prepare_symile_test_open_hierarchy,
    require_current_symile_execution,
    require_symile_execution,
    require_symile_lifecycle,
    require_symile_prospective_mutation,
    require_symile_prospective_path,
)
from beyondcxr.training.symile_families import (
    DEVELOPMENT_FAMILY_ORDER,
    FINAL_NEURAL_MEMBER_SEEDS,
    FINAL_PACKAGE_COUNT,
    FINAL_PACKAGE_POLICY,
    FINAL_TABULAR_SEED,
    NEURAL_MEMBER_REDUCTION,
    SINGLE_MEMBER_REDUCTION,
    SYMILE_ECG_GATED_FAMILY,
    final_predictor_reduction_policy,
    final_predictor_view_policy,
)
from beyondcxr.training.symile_final_packages import ValidatedFinalPackage, validate_final_package
from beyondcxr.training.symile_statistics import (
    BOOTSTRAP_POLICY,
    ERROR_REVIEW_POLICY,
    METRIC_POLICY,
    SUBGROUP_POLICY,
    cluster_bootstrap_effect,
    deterministic_error_cases,
    final_member_mean_logit_ensemble,
    headline_metric_names,
    operating_point_metrics,
    raw_probability_metrics,
)
from beyondcxr.utils.immutable_json import immutable_json
from beyondcxr.utils.package_identity import package_scientific_config_from_normalized
from beyondcxr.utils.private_predictions import (
    SYMILE_TEST_INFERENCE_POLICY,
    ValidatedPredictionEvidence,
    validate_prediction_evidence,
)
from beyondcxr.utils.publication import (
    install_immutable_directory,
    publish_bytes_no_replace,
    publish_regenerable_bytes,
    staging_directory,
)
from beyondcxr.utils.semantic_identity import (
    canonical_json_bytes,
    canonical_scientific_id,
)
from beyondcxr.utils.symile_publication import (
    ValidatedFoldPackage,
    validate_development_result,
)
from beyondcxr.utils.validation_context import memoized_directory_validation

if TYPE_CHECKING:
    from beyondcxr.training.symile_test_data import (
        FrozenSymileTestData,
        HeldOutEvaluationProjection,
    )

PRETEST_FREEZE_PREFIX = "pretest-freeze-"
GLOBAL_RESULT_PREFIX = "global-result-"
_FREEZE_GUARD = object()
_FROZEN_CAMPAIGN_GUARD = object()
_TEST_OPEN_GUARD = object()
_PRESERVED_TEST_OPEN_GUARD = object()
_HELD_OUT_GUARD = object()
_GLOBAL_RESULT_CERTIFICATION_GUARD = object()
GLOBAL_EFFECT_COMPARISONS = (
    ("primary", "cxr_labs_gated", "cxr_densenet"),
    ("concat_vs_cxr", "cxr_labs_concat", "cxr_densenet"),
    ("gated_vs_concat", "cxr_labs_gated", "cxr_labs_concat"),
    ("ecg_vs_gated", SYMILE_ECG_GATED_FAMILY, "cxr_labs_gated"),
)
RELIABILITY_POLICY = {"bins": 10, "strategy": "uniform", "role": "descriptive"}


@dataclass(frozen=True)
class ValidatedPretestFreeze:
    directory: Path
    manifest: Mapping[str, Any]
    development_decision: ValidatedDevelopmentDecision
    final_packages: tuple[ValidatedFinalPackage, ...]
    _guard: object
    _operation: _HeldOutOperation | None = field(default=None, compare=False, repr=False)

    @property
    def freeze_id(self) -> str:
        return str(self.manifest["pretest_freeze_id"])


@dataclass(frozen=True)
class ValidatedTestOpenRecord:
    path: Path
    external_path: Path
    freeze_id: str
    execution_id: str
    document_bytes: bytes
    _guard: object


@dataclass(frozen=True)
class ValidatedTestOpenEvidence:
    """One preserved historical opening receipt without live-access authority."""

    path: Path
    restoration: ValidatedRestorationContext
    freeze_id: str
    execution_id: str
    bundle_id: str
    document_bytes: bytes
    _guard: object


@dataclass(frozen=True)
class ValidatedSymileFrozenCampaign:
    """One scientific freeze joined to its current canonical formal lifecycle."""

    freeze: ValidatedPretestFreeze
    lifecycle: ValidatedSymileLifecycle
    _guard: object
    developments: tuple = ()

    @property
    def current_execution(self) -> ValidatedCurrentSymileExecution:
        return self.lifecycle.current_execution


@dataclass(frozen=True)
class ValidatedHeldOutCapability:
    """Matching freeze, execution, and durable opening evidence."""

    frozen_campaign: ValidatedSymileFrozenCampaign
    opening: ValidatedTestOpenRecord
    _guard: object
    _operation: _HeldOutOperation | None = None

    @property
    def freeze(self) -> ValidatedPretestFreeze:
        return self.frozen_campaign.freeze

    @property
    def current_execution(self) -> ValidatedCurrentSymileExecution:
        return self.frozen_campaign.current_execution

    @property
    def lifecycle(self) -> ValidatedSymileLifecycle:
        return self.frozen_campaign.lifecycle

    @property
    def execution(self) -> ValidatedSymileExecution:
        return self.current_execution.execution


@dataclass
class _HeldOutOperation:
    capability: ValidatedHeldOutCapability
    state: object
    active: bool = True


def _held_out_state(value: ValidatedHeldOutCapability) -> object:
    freeze = value.freeze
    return immutable_json(
        {
            "freeze": freeze.manifest,
            "execution": value.execution.manifest,
            "decision": freeze.development_decision.manifest,
            "packages": [package.manifest for package in freeze.final_packages],
            "developments": [
                {
                    "authority": item.manifest,
                    "folds": [fold.manifest for fold in item.folds],
                    "predictions": [prediction.manifest for prediction in item.predictions],
                }
                for item in value.frozen_campaign.developments
            ],
        }
    )


@contextmanager
def held_out_operation(value: ValidatedHeldOutCapability):
    """Authenticate once while holding the campaign lock; revoke on scope exit.

    Receipt checks remain live at every materialization. Independently opened
    package and source files retain their own complete boundary validation.
    """
    roots = value.lifecycle
    with campaign_lock(roots.control_root, "Symile", trusted_root=roots.repository_root):
        validated = require_validated_held_out_capability(value)
        operation = _HeldOutOperation(validated, _held_out_state(validated))
        try:
            freeze = replace(validated.freeze, _operation=operation)
            joined = replace(validated.frozen_campaign, freeze=freeze)
            scoped = replace(validated, frozen_campaign=joined, _operation=operation)
            operation.capability = scoped
            yield scoped
        finally:
            operation.active = False


@dataclass(frozen=True)
class GlobalResultIntegrity:
    """Integrity and lineage observation without scientific claim certification."""

    directory: Path
    manifest: Mapping[str, Any]
    claims_sha256: str
    reliability_sha256: str

    @property
    def result_id(self) -> str:
        return str(self.manifest["global_result_id"])


class CertifiedGlobalResult:
    """Immutable authority for one exactly attested, scientifically certified result."""

    __slots__ = (
        "__directory",
        "__manifest",
        "__claims_sha256",
        "__reliability_sha256",
        "__guard",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("CertifiedGlobalResult is created only by scientific validation")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("CertifiedGlobalResult is immutable")

    def __copy__(self) -> CertifiedGlobalResult:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> CertifiedGlobalResult:
        memo[id(self)] = self
        return self

    @property
    def directory(self) -> Path:
        return self.__directory

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self.__manifest

    @property
    def claims_sha256(self) -> str:
        return self.__claims_sha256

    @property
    def reliability_sha256(self) -> str:
        return self.__reliability_sha256

    @property
    def result_id(self) -> str:
        return str(self.manifest["global_result_id"])

    def matches_integrity(self, value: object) -> bool:
        """Return whether an integrity observation is exactly the certified evidence."""
        return (
            isinstance(value, GlobalResultIntegrity)
            and value.directory == self.directory
            and immutable_json(value.manifest) == self.manifest
            and value.claims_sha256 == self.claims_sha256
            and value.reliability_sha256 == self.reliability_sha256
        )


def require_certified_global_result(value: object) -> CertifiedGlobalResult:
    """Require scientific certification rather than artifact integrity alone."""
    if (
        not isinstance(value, CertifiedGlobalResult)
        or value._CertifiedGlobalResult__guard is not _GLOBAL_RESULT_CERTIFICATION_GUARD
    ):
        raise PermissionError("Global result requires full scientific certification")
    return value


def publish_pretest_freeze(
    *,
    prospective_mutation: ValidatedSymileProspectiveMutation,
    control_root: str | Path,
    bundle_id: str,
    cv_assignment_id: str,
    task: Mapping[str, str],
    development_decision: ValidatedDevelopmentDecision,
    final_packages: Sequence[ValidatedFinalPackage],
) -> ValidatedPretestFreeze:
    """Validate and publish complete scientific membership under an unopened campaign lock."""
    require_symile_prospective_mutation(prospective_mutation)
    require_symile_prospective_path(
        prospective_mutation,
        control_root,
        "Symile freeze control root",
        relative="private/control/symile",
    )
    development = require_validated_development_decision(development_decision)
    packages = _validated_packages(
        final_packages,
        selected_family_authorities=development.manifest["selected_family_authorities"],
    )
    document = pretest_freeze_document(
        bundle_id=bundle_id,
        cv_assignment_id=cv_assignment_id,
        task=task,
        development_decision_id=development.decision_id,
        package_members=[
            {
                "package_id": package.package_id,
                "family_id": package.manifest["input"]["family"]["family_id"],
                "seed": package.manifest["seed_policy"],
            }
            for package in packages
        ],
    )
    freeze_id = document["pretest_freeze_id"]
    destination = Path(control_root) / "freezes" / freeze_id

    def validator(path: Path, **kwargs: object) -> ValidatedPretestFreeze:
        return validate_pretest_freeze(
            path,
            development_decision=development,
            final_packages=packages,
            **kwargs,
        )

    _publish_json_object(destination, document, validator)
    return validator(destination)


def pretest_freeze_document(
    *,
    bundle_id: str,
    cv_assignment_id: str,
    task: Mapping[str, object],
    development_decision_id: str,
    package_members: list[dict[str, object]],
) -> dict[str, object]:
    """Construct the sole scientific freeze document from explicit coordinates."""
    semantic = {
        "dataset_id": "symile",
        "bundle_id": bundle_id,
        "cv_assignment_id": cv_assignment_id,
        "task": dict(task),
        "development_decision_id": development_decision_id,
        "final_packages": package_members,
        "held_out_policy": {
            "predictor_views": final_predictor_view_policy(package_members),
            "inference_policy": SYMILE_TEST_INFERENCE_POLICY,
            "metric_policy": METRIC_POLICY,
            "operating_point_policy": DEVELOPMENT_DECISION_POLICY["threshold_derivation"],
            "subgroup_policy": SUBGROUP_POLICY,
            "bootstrap": BOOTSTRAP_POLICY,
            "calibration_policy": {
                "coefficients": "logit_calibration_slope_and_intercept",
                "reliability": RELIABILITY_POLICY,
            },
            "error_review_policy": ERROR_REVIEW_POLICY,
        },
    }
    freeze_id = canonical_scientific_id(
        PRETEST_FREEZE_PREFIX, semantic, domain="pretest-freeze", schema_version=1
    )
    document = {
        "pretest_freeze_schema_version": 1,
        "pretest_freeze_id": freeze_id,
        **semantic,
    }
    return document


@memoized_directory_validation("symile-pretest-freeze")
def validate_pretest_freeze(
    directory: str | Path,
    *,
    development_decision: ValidatedDevelopmentDecision,
    final_packages: Sequence[ValidatedFinalPackage],
    enforce_directory_name: bool = True,
) -> ValidatedPretestFreeze:
    """Revalidate frozen policy and membership against the decision and final packages."""
    root, raw, document = _validate_pretest_freeze_document(
        directory, enforce_directory_name=enforce_directory_name
    )
    development = require_validated_development_decision(development_decision)
    packages = _validated_packages(
        final_packages,
        selected_family_authorities=development.manifest["selected_family_authorities"],
    )
    expected_packages = [
        {
            "package_id": package.package_id,
            "family_id": package.manifest["input"]["family"]["family_id"],
            "seed": package.manifest["seed_policy"],
        }
        for package in packages
    ]
    if (
        document["development_decision_id"] != development.decision_id
        or document["final_packages"] != expected_packages
        or document["bundle_id"] != development.manifest["bundle_id"]
        or document["cv_assignment_id"] != development.manifest["cv_assignment_id"]
        or document["held_out_policy"]["operating_point_policy"]
        != development.manifest["decision_policy"]["threshold_derivation"]
    ):
        raise ManifestBuildError("Pre-test freeze referenced authorities are invalid")
    package_dataset = packages[0].manifest["input"]["dataset"]
    package_task = packages[0].manifest["input"]["task"]
    if document["bundle_id"] != package_dataset["bundle_id"] or document["task"] != package_task:
        raise ManifestBuildError("Pre-test freeze data or task authority is invalid")
    return ValidatedPretestFreeze(root, document, development, packages, _FREEZE_GUARD)


def _validate_pretest_freeze_document(
    directory: str | Path,
    *,
    enforce_directory_name: bool = True,
) -> tuple[Path, bytes, dict[str, Any]]:
    root, raw, document = _read_single_json(directory)
    required = {
        "pretest_freeze_schema_version",
        "pretest_freeze_id",
        "dataset_id",
        "bundle_id",
        "cv_assignment_id",
        "task",
        "development_decision_id",
        "final_packages",
        "held_out_policy",
    }
    schema_version = document.get("pretest_freeze_schema_version")
    if (
        set(document) != required
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ManifestBuildError("Pre-test freeze contract is invalid")
    _validate_freeze_fields(document)
    semantic = {
        key: document[key]
        for key in required - {"pretest_freeze_schema_version", "pretest_freeze_id"}
    }
    expected = canonical_scientific_id(
        PRETEST_FREEZE_PREFIX, semantic, domain="pretest-freeze", schema_version=1
    )
    if document["pretest_freeze_id"] != expected or (
        enforce_directory_name and root.name != expected
    ):
        raise ManifestBuildError("Pre-test freeze identity is invalid")
    return root, raw, document


def validate_symile_frozen_campaign(
    *,
    freeze: ValidatedPretestFreeze,
    lifecycle: ValidatedSymileLifecycle,
) -> ValidatedSymileFrozenCampaign:
    """Join the unique canonical scientific freeze to the active current execution."""
    scientific = _require_capability(freeze)
    roots = require_symile_lifecycle(lifecycle)
    current = require_current_symile_execution(roots.current_execution)
    execution_dataset = current.execution.manifest["dataset"]
    expected_directory = roots.control_root / "freezes" / scientific.freeze_id
    freeze_root = roots.control_root / "freezes"
    try:
        entries = tuple(freeze_root.iterdir())
    except OSError as exc:
        raise ManifestBuildError("Canonical Symile freeze root is unavailable") from exc
    if (
        scientific.directory.absolute() != expected_directory.absolute()
        or scientific.directory.resolve() != expected_directory.resolve()
        or scientific.directory.is_symlink()
        or not scientific.directory.is_dir()
        or freeze_root.is_symlink()
        or {entry.name for entry in entries} != {scientific.freeze_id}
        or any(entry.is_symlink() or not entry.is_dir() for entry in entries)
        or scientific.manifest["bundle_id"] != execution_dataset["bundle_id"]
        or scientific.manifest["cv_assignment_id"] != execution_dataset["cv_assignment_id"]
    ):
        raise ManifestBuildError("Symile freeze is not the active canonical frozen campaign")
    developments = validate_symile_execution_freeze_science(
        current.execution,
        scientific,
        report_root=roots.repository_root / "reports" / "symile",
        model_root=roots.repository_root / "models" / "symile",
        private_root=roots.private_root,
        manifest_root=roots.repository_root / "data" / "manifests",
    )
    return ValidatedSymileFrozenCampaign(
        scientific, roots, _FROZEN_CAMPAIGN_GUARD, developments or ()
    )


def validate_symile_execution_freeze_science(
    execution: ValidatedSymileExecution,
    freeze: ValidatedPretestFreeze,
    *,
    report_root: str | Path,
    model_root: str | Path,
    private_root: str | Path,
    manifest_root: str | Path,
) -> tuple:
    """Join historical execution science to recursively validated frozen authorities."""
    historical = require_symile_execution(execution)
    report = Path(report_root)
    models = Path(model_root)
    private = Path(private_root)
    manifests = Path(manifest_root)
    decision_id = str(freeze.manifest["development_decision_id"])
    decision = validate_development_decision(
        report / "development-decisions" / decision_id,
        report_root=report,
        model_root=models / "development",
        prediction_root=private,
        manifest_root=manifests,
    )
    development_references = decision.manifest["family_development_ids"]
    developments = tuple(
        validate_development_result(
            report / "development" / "families" / str(reference["development_id"]),
            model_root=models / "development",
            prediction_root=private,
            manifest_root=manifests,
            expected_development_id=str(reference["development_id"]),
        )
        for reference in development_references
    )
    package_references = freeze.manifest["final_packages"]
    packages = tuple(
        validate_final_package(
            models / "final" / "packages" / str(reference["package_id"]),
            expected_package_id=str(reference["package_id"]),
        )
        for reference in package_references
    )
    scientific = validate_pretest_freeze(
        freeze.directory,
        development_decision=decision,
        final_packages=packages,
        enforce_directory_name=False,
    )
    try:
        dataset = historical.manifest["dataset"]
        plan = historical.manifest["formal_run_plan"]
        if (
            scientific.freeze_id != freeze.freeze_id
            or scientific.manifest != freeze.manifest
            or dataset
            != {
                "bundle_id": scientific.manifest["bundle_id"],
                "cv_assignment_id": scientific.manifest["cv_assignment_id"],
            }
            or [item["family_id"] for item in plan] != list(DEVELOPMENT_FAMILY_ORDER)
            or [item["family_id"] for item in development_references]
            != list(DEVELOPMENT_FAMILY_ORDER)
        ):
            raise ManifestBuildError("Symile execution and scientific freeze disagree")
        for configured, reference, development in zip(
            plan, development_references, developments, strict=True
        ):
            normalized = configured["normalized_config"]
            context = development.manifest["scientific_context"]
            if (
                reference["development_id"] != development.manifest["development_id"]
                or configured["family_id"] != development.manifest["family_id"]
                or context["bundle_id"] != dataset["bundle_id"]
                or context["cv_assignment_id"] != dataset["cv_assignment_id"]
                or context["task_id"] != normalized["task"]["task_id"]
                or context["fit_config"] != package_scientific_config_from_normalized(normalized)
            ):
                raise ManifestBuildError("Symile execution and scientific freeze disagree")
            _validate_development_execution_provenance(
                development,
                execution=historical.manifest,
            )
        for package in packages:
            if not final_package_belongs_to_execution(package, historical.manifest):
                raise ManifestBuildError("Symile execution and scientific freeze disagree")
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestBuildError("Symile execution and scientific freeze disagree") from exc
    return developments


def _validate_development_execution_provenance(
    development,
    *,
    execution: Mapping[str, Any],
) -> None:
    if [item.manifest["fold_package_id"] for item in development.folds] != [
        item["fold_package_id"] for item in development.manifest["fold_packages"]
    ]:
        raise ManifestBuildError("Development fitted evidence membership differs")
    for package in development.folds:
        validate_fold_execution_provenance(package, execution)


def validate_fold_execution_provenance(
    package: ValidatedFoldPackage, execution: Mapping[str, Any]
) -> None:
    """Check a validated fold against the formal execution that admitted it."""
    lineage = package.manifest["lineage"]
    operational = package.manifest["operational"]
    if (
        lineage["git_commit"] != execution["git_commit"]
        or lineage["dependency_lock_sha256"] != execution["dependency_lock_sha256"]
        or operational["python_version"] != execution["framework_versions"]["python"]
    ):
        raise ManifestBuildError("Symile development provenance differs from execution")
    runtime = operational["runtime_provenance"]
    loader = operational["loader_execution"]
    if runtime is None:
        if loader is not None:
            raise ManifestBuildError("Symile tabular provenance differs from execution")
        return
    if (
        runtime.get("torch_version") != execution["framework_versions"]["torch"]
        or runtime.get("torchvision_version") != execution["framework_versions"]["torchvision"]
        or runtime.get("torchxrayvision_version")
        != execution["framework_versions"]["torchxrayvision"]
        or not _runtime_provenance_matches(runtime, execution["neural_runtime"])
        or not _loader_provenance_matches(loader, execution["training_loader"])
    ):
        raise ManifestBuildError("Symile neural provenance differs from execution")
    weight = lineage["pretrained_weight"]
    if weight is not None and not _pretrained_materialization_matches(
        weight, execution["pretrained_weight"]
    ):
        raise ManifestBuildError("Symile pretrained provenance differs from execution")


def final_package_belongs_to_execution(
    package: ValidatedFinalPackage, execution: Mapping[str, Any]
) -> bool:
    """Match a validated final package's provenance to its admitted execution."""
    provenance = package.manifest["execution_provenance"]
    if (
        provenance["git_commit"] != execution["git_commit"]
        or provenance["git_dirty"] is not False
        or provenance["dependency_lock_sha256"] != execution["dependency_lock_sha256"]
        or provenance["python_version"] != execution["framework_versions"]["python"]
    ):
        return False
    if package.manifest["package_kind"] == "tabular":
        return True
    runtime = provenance["runtime_provenance"]
    if (
        runtime.get("torch_version") != execution["framework_versions"]["torch"]
        or runtime.get("torchvision_version") != execution["framework_versions"]["torchvision"]
        or runtime.get("torchxrayvision_version")
        != execution["framework_versions"]["torchxrayvision"]
        or not _runtime_provenance_matches(runtime, execution["neural_runtime"])
        or not _loader_provenance_matches(
            provenance["loader_execution"], execution["training_loader"]
        )
    ):
        return False
    weight = package.manifest["pretrained_weight_fingerprint"]
    return weight is None or _pretrained_materialization_matches(
        weight, execution["pretrained_weight"]
    )


def _pretrained_materialization_matches(value: object, expected: object) -> bool:
    if not isinstance(value, Mapping) or not isinstance(expected, Mapping):
        return False
    return {
        key: value.get(key) for key in ("declared_name", "stable_identifier", "byte_size", "sha256")
    } == dict(expected)


def _runtime_provenance_matches(value: object, expected: object) -> bool:
    if not isinstance(value, Mapping) or not isinstance(expected, Mapping):
        return False
    return (
        value.get("resolved_device") == expected.get("device_type")
        and ("float16" if value.get("mixed_precision_effective") is True else None)
        == expected.get("autocast_dtype")
        and value.get("cuda_runtime_version") == expected.get("cuda_runtime_version")
        and value.get("cudnn_version") == expected.get("cudnn_version")
        and value.get("gpu_device_name") == expected.get("gpu_device_name")
        and value.get("gpu_compute_capability") == expected.get("gpu_compute_capability")
    )


def _loader_provenance_matches(value: object, expected: object) -> bool:
    if not isinstance(value, Mapping) or not isinstance(expected, Mapping):
        return False
    return all(
        value.get(key) == expected.get(key) for key in ("lifecycle", "num_workers", "pin_memory")
    )


def require_symile_frozen_campaign(
    value: object,
) -> ValidatedSymileFrozenCampaign:
    if (
        not isinstance(value, ValidatedSymileFrozenCampaign)
        or value._guard is not _FROZEN_CAMPAIGN_GUARD
    ):
        raise PermissionError("Symile test opening requires a validated frozen campaign")
    observed = validate_symile_frozen_campaign(
        freeze=value.freeze,
        lifecycle=value.lifecycle,
    )
    if observed != value:
        raise ManifestBuildError("Symile frozen campaign changed after validation")
    return observed


def open_official_test(
    *,
    frozen_campaign: ValidatedSymileFrozenCampaign,
) -> ValidatedHeldOutCapability:
    """Reconcile both permanent opening records before granting held-out access."""
    joined = require_symile_frozen_campaign(frozen_campaign)
    freeze = joined.freeze
    roots = joined.lifecycle
    current = joined.current_execution
    execution = current.execution
    control = roots.control_root
    with campaign_lock(control, "Symile", trusted_root=roots.repository_root):
        local, external = prepare_symile_test_open_hierarchy(roots)
        local_raw = _optional_test_open_record(local)
        external_raw = _optional_test_open_record(external)
        if local_raw is None and external_raw is None:
            document = {
                "schema_version": 1,
                "freeze_id": freeze.freeze_id,
                "execution_id": execution.execution_id,
                "opened_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }
            local_raw = canonical_json_bytes(document)
            publish_bytes_no_replace(local, local_raw)
            publish_bytes_no_replace(external, local_raw)
        elif local_raw is None:
            assert external_raw is not None
            _validate_test_open_bytes(external_raw, freeze, execution)
            publish_bytes_no_replace(local, external_raw)
        elif external_raw is None:
            _validate_test_open_bytes(local_raw, freeze, execution)
            publish_bytes_no_replace(external, local_raw)
        local_raw, _ = _read_test_open_record(local)
        external_raw, _ = _read_test_open_record(external)
        if local_raw != external_raw:
            raise ManifestBuildError("Local and external official-test opening records differ")
        _validate_test_open_bytes(local_raw, freeze, execution)
    record = ValidatedTestOpenRecord(
        local,
        external,
        freeze.freeze_id,
        execution.execution_id,
        local_raw,
        _TEST_OPEN_GUARD,
    )
    return ValidatedHeldOutCapability(joined, record, _HELD_OUT_GUARD)


def _validate_preserved_test_open(
    *,
    path: str | Path,
    restoration: ValidatedRestorationContext,
    freeze: ValidatedPretestFreeze,
    execution: ValidatedSymileExecution,
) -> ValidatedTestOpenEvidence:
    """Mint archive-bound historical evidence for restored-campaign validation."""
    root = require_restoration_context(restoration).root
    freeze = _require_capability(freeze)
    execution = require_symile_execution(execution)
    opening = Path(path)
    bundle_id = freeze.manifest.get("bundle_id")
    expected_control = root / "private" / "control" / "symile"
    if (
        not isinstance(bundle_id, str)
        or opening.resolve() != expected_control / "test-open.json"
        or freeze.directory.resolve() != expected_control / "freezes" / freeze.freeze_id
        or execution.directory.resolve() != expected_control / "executions" / execution.execution_id
    ):
        raise ManifestBuildError("Preserved opening evidence is outside its restored campaign")
    raw, _ = _read_test_open_record(opening)
    _validate_test_open_bytes(raw, freeze, execution)
    return ValidatedTestOpenEvidence(
        opening,
        restoration,
        freeze.freeze_id,
        execution.execution_id,
        bundle_id,
        raw,
        _PRESERVED_TEST_OPEN_GUARD,
    )


def require_validated_test_open_evidence(
    value: object,
    *,
    freeze: ValidatedPretestFreeze,
) -> ValidatedTestOpenEvidence:
    """Reauthenticate preserved opening evidence without granting live source authority."""
    scientific = _require_capability(freeze)
    if (
        not isinstance(value, ValidatedTestOpenEvidence)
        or value._guard is not _PRESERVED_TEST_OPEN_GUARD
    ):
        raise PermissionError("Restored held-out access requires validated opening evidence")
    root = require_restoration_context(value.restoration).root
    if (
        value.freeze_id != scientific.freeze_id
        or value.bundle_id != scientific.manifest.get("bundle_id")
        or value.path.resolve() != root / "private" / "control" / "symile" / "test-open.json"
        or scientific.directory.resolve()
        != root / "private" / "control" / "symile" / "freezes" / scientific.freeze_id
    ):
        raise PermissionError("Restored held-out access requires validated opening evidence")
    raw, document = _read_test_open_record(value.path)
    if (
        raw != value.document_bytes
        or document.get("freeze_id") != value.freeze_id
        or document.get("execution_id") != value.execution_id
    ):
        raise ManifestBuildError("Preserved official-test opening evidence changed")
    return value


def _optional_test_open_record(path: Path) -> bytes | None:
    if not path.exists() and not path.is_symlink():
        return None
    raw, _ = _read_test_open_record(path)
    return raw


def _validate_test_open_bytes(
    raw: bytes, freeze: ValidatedPretestFreeze, execution: ValidatedSymileExecution
) -> None:
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Official-test opening record is malformed") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "freeze_id", "execution_id", "opened_at_utc"}
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or document["freeze_id"] != freeze.freeze_id
        or document["execution_id"] != execution.execution_id
        or not _valid_utc_timestamp(document["opened_at_utc"])
        or raw != canonical_json_bytes(document)
    ):
        raise ManifestBuildError("Official-test opening is bound to another formal campaign")


def _read_test_open_record(path: Path) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ManifestBuildError("Official-test opening record is invalid")
    try:
        raw = path.read_bytes()
        observed = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Official-test opening record is malformed") from exc
    if not isinstance(observed, dict):
        raise ManifestBuildError("Official-test opening record is malformed")
    return raw, observed


def materialize_official_test[T](
    capability: ValidatedHeldOutCapability,
    operation: Callable[[], T],
) -> T:
    """Reauthenticate live held-out authority before invoking the supplied source operation."""
    require_validated_held_out_capability(capability)
    return operation()


def validated_pretest_freeze_manifest(
    capability: ValidatedPretestFreeze,
) -> Mapping[str, Any]:
    return _require_capability(capability).manifest


def publish_global_result(
    *,
    report_root: str | Path,
    capability: ValidatedPretestFreeze,
    predictions: Sequence[ValidatedPredictionEvidence],
    final_packages: Sequence[ValidatedFinalPackage],
    test_data: FrozenSymileTestData,
) -> CertifiedGlobalResult:
    """Derive, publish, and certify held-out claims from frozen packages and predictions."""
    capability = _require_capability(capability)
    packages = _validated_frozen_packages(capability, final_packages)
    projection = _frozen_evaluation_projection(capability, test_data)
    evidence = _validated_predictions(capability, packages, predictions, projection)
    claims = _derive_global_claims(capability, packages, evidence, projection)
    semantic = {
        "pretest_freeze_id": capability.freeze_id,
        "prediction_ids": [item.prediction_id for item in evidence],
        "evaluation_policy": _global_evaluation_policy(),
    }
    result_id = canonical_scientific_id(
        GLOBAL_RESULT_PREFIX, semantic, domain="global-result", schema_version=1
    )
    document = {"global_result_schema_version": 1, "global_result_id": result_id, **semantic}
    destination = Path(report_root) / "global-results" / result_id
    stage = staging_directory(destination)
    try:
        (stage / "manifest.json").write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (stage / "claims.json").write_text(
            json.dumps(claims, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (stage / "reliability.svg").write_text(
            _reliability_svg(claims["reliability_curves"]),
            encoding="utf-8",
        )

        def validator(path: Path, **kwargs: object) -> GlobalResultIntegrity:
            return validate_global_result_integrity(
                path,
                capability=capability,
                predictions=evidence,
                final_packages=packages,
                test_data=test_data,
                **kwargs,
            )

        install_immutable_directory(stage, destination, validator)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return validate_global_result(
        destination,
        capability=capability,
        predictions=evidence,
        final_packages=packages,
        test_data=test_data,
    )


def validate_global_result(
    directory: str | Path,
    *,
    capability: ValidatedPretestFreeze,
    predictions: Sequence[ValidatedPredictionEvidence],
    final_packages: Sequence[ValidatedFinalPackage],
    test_data: FrozenSymileTestData,
    enforce_directory_name: bool = True,
) -> CertifiedGlobalResult:
    """Scientifically certify a global result by independently rederiving its claims."""
    validated, packages, projection, evidence, observed_claims = _validate_global_result_artifact(
        directory,
        capability=capability,
        predictions=predictions,
        final_packages=final_packages,
        test_data=test_data,
        enforce_directory_name=enforce_directory_name,
    )
    expected_claims = _derive_global_claims(capability, packages, evidence, projection)
    if observed_claims != expected_claims:
        raise ManifestBuildError("Global-result claims do not rederive from prediction evidence")
    certified = object.__new__(CertifiedGlobalResult)
    object.__setattr__(
        certified,
        "_CertifiedGlobalResult__directory",
        validated.directory,
    )
    object.__setattr__(
        certified,
        "_CertifiedGlobalResult__manifest",
        immutable_json(validated.manifest),
    )
    object.__setattr__(
        certified,
        "_CertifiedGlobalResult__claims_sha256",
        validated.claims_sha256,
    )
    object.__setattr__(
        certified,
        "_CertifiedGlobalResult__reliability_sha256",
        validated.reliability_sha256,
    )
    object.__setattr__(
        certified,
        "_CertifiedGlobalResult__guard",
        _GLOBAL_RESULT_CERTIFICATION_GUARD,
    )
    return certified


def validate_global_result_integrity(
    directory: str | Path,
    *,
    capability: ValidatedPretestFreeze,
    predictions: Sequence[ValidatedPredictionEvidence],
    final_packages: Sequence[ValidatedFinalPackage],
    test_data: FrozenSymileTestData,
    enforce_directory_name: bool = True,
) -> GlobalResultIntegrity:
    """Validate immutable global-result identity and lineage without recomputing statistics."""
    validated, _, _, _, _ = _validate_global_result_artifact(
        directory,
        capability=capability,
        predictions=predictions,
        final_packages=final_packages,
        test_data=test_data,
        enforce_directory_name=enforce_directory_name,
    )
    return validated


def _validate_global_result_artifact(
    directory: str | Path,
    *,
    capability: ValidatedPretestFreeze,
    predictions: Sequence[ValidatedPredictionEvidence],
    final_packages: Sequence[ValidatedFinalPackage],
    test_data: FrozenSymileTestData,
    enforce_directory_name: bool,
) -> tuple[
    GlobalResultIntegrity,
    tuple[ValidatedFinalPackage, ...],
    HeldOutEvaluationProjection,
    tuple[ValidatedPredictionEvidence, ...],
    dict[str, object],
]:
    """Validate the stored result contract shared by certification and routine consumers."""
    capability = _require_capability(capability)
    root = Path(directory)
    if (
        root.is_symlink()
        or not root.is_dir()
        or {item.name for item in root.iterdir()}
        != {
            "manifest.json",
            "claims.json",
            "reliability.svg",
        }
        or any(item.is_symlink() or not item.is_file() for item in root.iterdir())
    ):
        raise ManifestBuildError("Global-result directory is invalid")
    document = json.loads((root / "manifest.json").read_bytes())
    required = {
        "global_result_schema_version",
        "global_result_id",
        "pretest_freeze_id",
        "prediction_ids",
        "evaluation_policy",
    }
    schema_version = (
        document.get("global_result_schema_version") if isinstance(document, dict) else None
    )
    if (
        not isinstance(document, dict)
        or set(document) != required
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
        or len(document["prediction_ids"]) != FINAL_PACKAGE_COUNT
        or document["pretest_freeze_id"] != capability.freeze_id
        or document["evaluation_policy"] != _global_evaluation_policy()
    ):
        raise ManifestBuildError("Global-result contract is invalid")
    semantic = {
        key: document[key]
        for key in required - {"global_result_schema_version", "global_result_id"}
    }
    expected = canonical_scientific_id(
        GLOBAL_RESULT_PREFIX, semantic, domain="global-result", schema_version=1
    )
    if document["global_result_id"] != expected or (
        enforce_directory_name and root.name != expected
    ):
        raise ManifestBuildError("Global-result identity is invalid")
    packages = _validated_frozen_packages(capability, final_packages)
    projection = _frozen_evaluation_projection(capability, test_data)
    evidence = _validated_predictions(capability, packages, predictions, projection)
    if document["prediction_ids"] != [item.prediction_id for item in evidence]:
        raise ManifestBuildError("Global-result prediction order is invalid")
    claims_bytes = (root / "claims.json").read_bytes()
    observed_claims = json.loads(claims_bytes)
    if not isinstance(observed_claims, dict) or "reliability_curves" not in observed_claims:
        raise ManifestBuildError("Global-result claims contract is invalid")
    reliability_bytes = (root / "reliability.svg").read_bytes()
    if reliability_bytes.decode("utf-8") != _reliability_svg(observed_claims["reliability_curves"]):
        raise ManifestBuildError("Global-result reliability plot differs from stored claims")
    return (
        GlobalResultIntegrity(
            root,
            document,
            hashlib.sha256(claims_bytes).hexdigest(),
            hashlib.sha256(reliability_bytes).hexdigest(),
        ),
        packages,
        projection,
        evidence,
        observed_claims,
    )


def publish_error_review(
    *,
    private_root: str | Path,
    capability: ValidatedPretestFreeze,
    predictions: Sequence[ValidatedPredictionEvidence],
    final_packages: Sequence[ValidatedFinalPackage],
    test_data: FrozenSymileTestData,
) -> Path:
    """Publish a restricted regenerable derivative, never a global-result authority."""
    document = _error_review_document(capability, predictions, final_packages, test_data)
    encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = Path(private_root) / "error-review" / "symile" / "error-review.json"
    publish_regenerable_bytes(path, encoded)
    if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
        raise ManifestBuildError("Private error review conflicts with existing content")
    return path


def validate_error_review(
    path: str | Path,
    *,
    capability: ValidatedPretestFreeze,
    predictions: Sequence[ValidatedPredictionEvidence],
    final_packages: Sequence[ValidatedFinalPackage],
    test_data: FrozenSymileTestData,
) -> None:
    """Independently rederive private cases from recursively validated evidence."""
    expected = _error_review_document(capability, predictions, final_packages, test_data)
    source = Path(path)
    encoded = (json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if (
        source.is_symlink()
        or not source.is_file()
        or source.name != "error-review.json"
        or source.read_bytes() != encoded
    ):
        raise ManifestBuildError("Private error review does not rederive from evidence")


def _error_review_document(capability, predictions, final_packages, test_data):
    packages = _validated_frozen_packages(capability, final_packages)
    projection = _frozen_evaluation_projection(capability, test_data)
    evidence = _validated_predictions(capability, packages, predictions, projection)
    primary = predictor_views(packages, evidence)["cxr_labs_gated"]
    threshold = capability.development_decision.manifest["primary_thresholds"]["youden_j"]
    semantic = {
        "pretest_freeze_id": capability.freeze_id,
        "predictor": "cxr_labs_gated",
        "prediction_ids": [
            item.prediction_id
            for package, item in zip(packages, evidence, strict=True)
            if package.manifest["input"]["family"]["family_id"] == "cxr_labs_gated"
        ],
        "operating_point": {"name": "youden_j", "threshold": threshold},
        "ranking_policy": ERROR_REVIEW_POLICY,
        "cases": deterministic_error_cases(
            primary[["sample_id", "target", "probability"]], threshold
        ),
    }
    return {
        "error_review_schema_version": 1,
        **semantic,
    }


def _validated_predictions(
    capability: ValidatedPretestFreeze,
    packages: Sequence[ValidatedFinalPackage],
    predictions: Sequence[ValidatedPredictionEvidence],
    projection: HeldOutEvaluationProjection,
) -> tuple[ValidatedPredictionEvidence, ...]:
    from beyondcxr.training.symile_test_data import validate_prediction_against_test_projection

    if len(predictions) != FINAL_PACKAGE_COUNT:
        raise ManifestBuildError("Global result requires fourteen prediction objects")
    values = tuple(
        validate_prediction_evidence(
            item.directory,
            expected_prediction_id=item.prediction_id,
            enforce_directory_name=False,
        )
        for item in predictions
    )
    frozen_package_ids = tuple(package.package_id for package in packages)
    values = tuple(
        validate_prediction_against_test_projection(
            item,
            capability=capability,
            projection=projection,
            expected_package_ids=frozen_package_ids,
        )
        for item in values
    )
    for package, evidence in zip(packages, values, strict=True):
        if (
            evidence.manifest["model_package_id"] != package.package_id
            or evidence.manifest.get("authorized_by_freeze_id") != capability.freeze_id
        ):
            raise ManifestBuildError("Global-result prediction lineage is invalid")
    if len({item.prediction_id for item in values}) != FINAL_PACKAGE_COUNT:
        raise ManifestBuildError("Global-result prediction membership contains duplicates")
    return values


def _global_evaluation_policy() -> dict[str, object]:
    return {
        "metric_policy": METRIC_POLICY,
        "predictor_views": final_predictor_reduction_policy(),
        "raw_probability_metrics": [
            *headline_metric_names(),
            "calibration_slope",
            "calibration_intercept",
        ],
        "paired_effects": [name for name, _, _ in GLOBAL_EFFECT_COMPARISONS],
        "bootstrap": dict(BOOTSTRAP_POLICY),
        "primary_operating_points": ["youden_j", "target_sensitivity"],
        "reliability_plot": dict(RELIABILITY_POLICY),
    }


def _derive_global_claims(
    capability: ValidatedPretestFreeze,
    packages: Sequence[ValidatedFinalPackage],
    predictions: Sequence[ValidatedPredictionEvidence],
    projection: HeldOutEvaluationProjection,
) -> dict[str, object]:
    views = predictor_views(packages, predictions)
    subjects = _validated_projection_subjects(capability, projection, views["labs_logistic"])
    effects: dict[str, object] = {}
    for name, candidate, comparator in GLOBAL_EFFECT_COMPARISONS:
        frame = (
            views[candidate]
            .merge(
                views[comparator][["sample_id", "probability"]],
                on="sample_id",
                suffixes=("_candidate", "_comparator"),
            )
            .merge(subjects, on="sample_id", validate="one_to_one")
            .rename(
                columns={
                    "probability_candidate": "candidate",
                    "probability_comparator": "comparator",
                }
            )
        )
        effects[name] = {
            metric: cluster_bootstrap_effect(
                frame, candidate="candidate", comparator="comparator", metric=metric
            )
            for metric in headline_metric_names()
        }
    primary = views["cxr_labs_gated"]
    operating_points = {
        name: operating_point_metrics(primary["target"], primary["probability"], float(threshold))
        for name, threshold in capability.development_decision.manifest[
            "primary_thresholds"
        ].items()
    }
    return {
        "predictor_views": {
            family: raw_probability_metrics(frame["target"], frame["probability"])
            for family, frame in views.items()
        },
        "reliability_curves": {
            family: _reliability_curve(frame["target"], frame["probability"])
            for family, frame in views.items()
        },
        "paired_effects": effects,
        "primary_operating_points": operating_points,
    }


def _reliability_curve(targets: pd.Series, probabilities: pd.Series) -> dict[str, list[float]]:
    observed, predicted = calibration_curve(
        targets,
        probabilities,
        n_bins=int(RELIABILITY_POLICY["bins"]),
        strategy=str(RELIABILITY_POLICY["strategy"]),
    )
    return {
        "mean_predicted_probability": [float(value) for value in predicted],
        "observed_positive_fraction": [float(value) for value in observed],
    }


def _reliability_svg(curves: object) -> str:
    if not isinstance(curves, Mapping) or set(curves) != set(FINAL_PACKAGE_POLICY):
        raise ManifestBuildError("Reliability-curve membership is invalid")
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="600" viewBox="0 0 720 600">',
        '<rect width="720" height="600" fill="white"/>',
        '<text x="300" y="30" text-anchor="middle" font-family="sans-serif" '
        'font-size="18">Raw reliability curves (descriptive)</text>',
        '<line x1="70" y1="530" x2="530" y2="70" stroke="#888" stroke-dasharray="5 5"/>',
        '<line x1="70" y1="530" x2="530" y2="530" stroke="black"/>',
        '<line x1="70" y1="530" x2="70" y2="70" stroke="black"/>',
        '<text x="300" y="575" text-anchor="middle" font-family="sans-serif" '
        'font-size="14">Mean predicted probability</text>',
        '<text x="18" y="300" text-anchor="middle" transform="rotate(-90 18 300)" '
        'font-family="sans-serif" font-size="14">Observed positive fraction</text>',
    ]
    for index, family in enumerate(FINAL_PACKAGE_POLICY):
        curve = curves[family]
        if not isinstance(curve, Mapping) or set(curve) != {
            "mean_predicted_probability",
            "observed_positive_fraction",
        }:
            raise ManifestBuildError("Reliability-curve schema is invalid")
        predicted = curve["mean_predicted_probability"]
        observed = curve["observed_positive_fraction"]
        if (
            not isinstance(predicted, list)
            or not isinstance(observed, list)
            or not predicted
            or len(predicted) != len(observed)
        ):
            raise ManifestBuildError("Reliability-curve values are invalid")
        points = " ".join(
            f"{70 + 460 * float(x):.3f},{530 - 460 * float(y):.3f}"
            for x, y in zip(predicted, observed, strict=True)
        )
        color = colors[index]
        lines.extend(
            (
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>',
                f'<line x1="555" y1="{90 + 28 * index}" x2="575" '
                f'y2="{90 + 28 * index}" stroke="{color}" stroke-width="2"/>',
                f'<text x="582" y="{95 + 28 * index}" font-family="sans-serif" '
                f'font-size="12">{family}</text>',
            )
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def predictor_views(
    packages: Sequence[ValidatedFinalPackage],
    predictions: Sequence[ValidatedPredictionEvidence],
) -> dict[str, pd.DataFrame]:
    policy = final_predictor_view_policy(
        [
            {
                "package_id": package.package_id,
                "family_id": package.manifest["input"]["family"]["family_id"],
                "seed": package.manifest["seed_policy"],
            }
            for package in packages
        ]
    )
    grouped: dict[str, list[pd.DataFrame]] = {}
    for package, evidence in zip(packages, predictions, strict=True):
        family = str(package.manifest["input"]["family"]["family_id"])
        frame = evidence.predictions.to_pandas()
        frame["member_seed"] = package.manifest["seed_policy"]
        grouped.setdefault(family, []).append(frame)
    if set(grouped) != set(FINAL_PACKAGE_POLICY):
        raise ManifestBuildError("Global result requires exactly six predictor views")
    views = {}
    for view in policy:
        family = str(view["family_id"])
        frames = grouped[family]
        if view["reduction"] == SINGLE_MEMBER_REDUCTION:
            views[family] = frames[0].loc[:, ["sample_id", "target", "logit", "probability"]]
        elif view["reduction"] == NEURAL_MEMBER_REDUCTION:
            views[family] = final_member_mean_logit_ensemble(
                pd.concat(frames, ignore_index=True).loc[
                    :, ["sample_id", "target", "logit", "member_seed"]
                ]
            )
        else:
            raise ManifestBuildError("Symile predictor-view reduction policy is invalid")
    reference = views["labs_logistic"].loc[:, ["sample_id", "target"]].reset_index(drop=True)
    if any(
        not frame.loc[:, ["sample_id", "target"]].reset_index(drop=True).equals(reference)
        for frame in views.values()
    ):
        raise ManifestBuildError("Global-result predictor views are not exactly aligned")
    return views


def _validated_projection_subjects(
    capability: ValidatedPretestFreeze,
    projection: HeldOutEvaluationProjection,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    try:
        frame = projection.frame()
        lineage = (
            projection.dataset_id,
            projection.bundle_id,
            projection.task_id,
            projection.label_policy_version,
            projection.scope,
            projection.inference_policy,
            projection.freeze_id,
        )
    except (AttributeError, TypeError) as exc:
        raise ManifestBuildError("Global-result held-out projection is invalid") from exc
    frozen_task = capability.manifest["task"]
    if (
        lineage
        != (
            "symile",
            capability.manifest["bundle_id"],
            frozen_task["task_id"],
            frozen_task["label_policy_version"],
            "test",
            SYMILE_TEST_INFERENCE_POLICY,
            capability.freeze_id,
        )
        or not isinstance(frame, pd.DataFrame)
        or tuple(frame.columns) != ("sample_id", "target", "subject_id")
        or frame.empty
        or frame.isna().any().any()
        or frame["sample_id"].duplicated().any()
        or not pd.api.types.is_integer_dtype(frame["subject_id"].dtype)
    ):
        raise ManifestBuildError("Global-result held-out projection is invalid")
    ordered = frame.sort_values("sample_id", kind="stable").reset_index(drop=True)
    if not ordered.loc[:, ["sample_id", "target"]].equals(
        reference.loc[:, ["sample_id", "target"]].reset_index(drop=True)
    ):
        raise ManifestBuildError(
            "Global-result held-out projection differs from prediction evidence"
        )
    return ordered.loc[:, ["sample_id", "subject_id"]]


def _frozen_evaluation_projection(
    capability: ValidatedPretestFreeze,
    test_data: FrozenSymileTestData,
) -> HeldOutEvaluationProjection:
    from beyondcxr.training.symile_test_data import canonical_held_out_evaluation_projection

    try:
        return canonical_held_out_evaluation_projection(test_data, freeze=capability)
    except PermissionError as exc:
        raise ManifestBuildError(
            "Global result requires official test data authorized by the same pre-test freeze"
        ) from exc


def _validated_frozen_packages(
    capability: ValidatedPretestFreeze,
    packages: Sequence[ValidatedFinalPackage],
) -> tuple[ValidatedFinalPackage, ...]:
    values = _validated_packages(packages)
    observed = [
        {
            "package_id": package.package_id,
            "family_id": package.manifest["input"]["family"]["family_id"],
            "seed": package.manifest["seed_policy"],
        }
        for package in values
    ]
    if observed != capability.manifest["final_packages"]:
        raise ManifestBuildError("Global result packages differ from the pre-test freeze")
    return values


def _validated_packages(
    packages: Sequence[ValidatedFinalPackage],
    *,
    selected_family_authorities: Mapping[str, object] | None = None,
) -> tuple[ValidatedFinalPackage, ...]:
    values = tuple(
        validate_final_package(
            package.directory,
            expected_package_id=package.package_id,
            enforce_directory_name=False,
        )
        for package in packages
    )
    by_family: dict[str, list[ValidatedFinalPackage]] = {}
    for package in values:
        family = package.manifest["input"]["family"]["family_id"]
        by_family.setdefault(family, []).append(package)
        if selected_family_authorities is not None:
            authority = selected_family_authorities.get(family)
            if (
                not isinstance(authority, Mapping)
                or (
                    family != "labs_logistic"
                    and package.manifest["family_development_id"] != authority.get("development_id")
                )
                or (
                    family == "labs_logistic"
                    and package.manifest["family_development_id"] is not None
                )
                or package.manifest["final_training_budget"]
                != authority.get("final_training_budget")
                or package.manifest["input"] != authority.get("final_input")
                or package.manifest["pretrained_scientific_identity"]
                != authority.get("pretrained_scientific_identity")
            ):
                raise ManifestBuildError(
                    "Final package differs from its development-derived family authority"
                )
    expected = {family: int(policy["members"]) for family, policy in FINAL_PACKAGE_POLICY.items()}
    if (
        len(values) != FINAL_PACKAGE_COUNT
        or {family: len(items) for family, items in by_family.items()} != expected
    ):
        raise ManifestBuildError("Pre-test freeze requires the exact fourteen-package authority")
    expected_order = [
        family
        for family, policy in FINAL_PACKAGE_POLICY.items()
        for _ in range(int(policy["members"]))
    ]
    if [package.manifest["input"]["family"]["family_id"] for package in values] != expected_order:
        raise ManifestBuildError("Final packages are not in canonical family order")
    datasets = [package.manifest["input"]["dataset"] for package in values]
    tasks = [package.manifest["input"]["task"] for package in values]
    if any(dataset != datasets[0] for dataset in datasets[1:]) or any(
        task != tasks[0] for task in tasks[1:]
    ):
        raise ManifestBuildError("Final packages do not share one dataset and task authority")
    cxr_by_seed = {
        package.manifest["seed_policy"]: package.package_id for package in by_family["cxr_densenet"]
    }
    expected_seeds = set(FINAL_NEURAL_MEMBER_SEEDS)
    if set(cxr_by_seed) != expected_seeds:
        raise ManifestBuildError("Final CXR membership has invalid seeds")
    cxr_pretrained = [
        package.manifest["pretrained_scientific_identity"] for package in by_family["cxr_densenet"]
    ]
    if cxr_pretrained[0] is None or any(
        identity != cxr_pretrained[0] for identity in cxr_pretrained[1:]
    ):
        raise ManifestBuildError("Final CXR members have inconsistent pretrained identities")
    for family, items in by_family.items():
        if len(items) == 1:
            if items[0].manifest["seed_policy"] != FINAL_TABULAR_SEED:
                raise ManifestBuildError("Final tabular seed policy is invalid")
            continue
        seeds = [item.manifest["seed_policy"] for item in items]
        if seeds != list(FINAL_NEURAL_MEMBER_SEEDS):
            raise ManifestBuildError("Final neural members are not in canonical seed order")
        if family != "cxr_densenet" and any(
            item.manifest["source_cxr_package_id"] != cxr_by_seed[item.manifest["seed_policy"]]
            for item in items
        ):
            raise ManifestBuildError("Final fusion package lacks same-seed CXR ancestry")
    return values


def _require_capability(value: ValidatedPretestFreeze) -> ValidatedPretestFreeze:
    if not isinstance(value, ValidatedPretestFreeze) or value._guard is not _FREEZE_GUARD:
        raise ManifestBuildError("A genuine ValidatedPretestFreeze is required")
    if value._operation is not None:
        operation = value._operation
        if value is not operation.capability.freeze:
            raise PermissionError("Freeze does not belong to the authenticated operation")
        require_validated_held_out_capability(operation.capability)
        return value
    observed = validate_pretest_freeze(
        value.directory,
        development_decision=value.development_decision,
        final_packages=value.final_packages,
        enforce_directory_name=False,
    )
    if observed != value:
        raise ManifestBuildError("Validated pre-test freeze changed after recursive validation")
    return value


def require_validated_held_out_capability(
    value: object,
) -> ValidatedHeldOutCapability:
    """Reauthenticate the complete live official-test authority."""
    if not isinstance(value, ValidatedHeldOutCapability) or value._guard is not _HELD_OUT_GUARD:
        raise PermissionError("Official-test access requires a validated held-out capability")
    operation = value._operation
    if operation is None:
        joined = require_symile_frozen_campaign(value.frozen_campaign)
    else:
        if (
            not operation.active
            or value.frozen_campaign is not operation.capability.frozen_campaign
            or value.opening is not operation.capability.opening
            or _held_out_state(value) != operation.state
        ):
            raise PermissionError("Official-test operation is expired or changed")
        joined = value.frozen_campaign
    freeze = joined.freeze
    roots = joined.lifecycle
    current = joined.current_execution
    execution = current.execution
    record = value.opening
    if (
        not isinstance(record, ValidatedTestOpenRecord)
        or record._guard is not _TEST_OPEN_GUARD
        or record.freeze_id != freeze.freeze_id
        or record.execution_id != execution.execution_id
        or record.path != roots.control_root / "test-open.json"
        or record.external_path != roots.backup_root / "control/symile/test-open.json"
    ):
        raise ManifestBuildError("Official-test capability authorities disagree")
    local, _ = _read_test_open_record(record.path)
    external, _ = _read_test_open_record(record.external_path)
    if local != external or local != record.document_bytes:
        raise ManifestBuildError("Official-test opening evidence changed after validation")
    _validate_test_open_bytes(local, freeze, execution)
    return value


def _valid_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo == UTC


def _validate_freeze_fields(document: Mapping[str, Any]) -> None:
    task = document["task"]
    packages = document["final_packages"]
    held_out = document["held_out_policy"]
    if (
        document["dataset_id"] != "symile"
        or not _identity(document["bundle_id"], "bundle-")
        or not _identity(document["cv_assignment_id"], "cv-assignment-")
        or not isinstance(task, Mapping)
        or task
        != {
            "task_id": TASK_ID,
            "label_policy_version": LABEL_POLICY_VERSION,
        }
        or not _identity(document["development_decision_id"], "development-decision-")
        or not isinstance(packages, list)
        or len(packages) != FINAL_PACKAGE_COUNT
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"package_id", "family_id", "seed"}
            or not _identity(item["package_id"], "final-package-")
            or not isinstance(item["family_id"], str)
            or type(item["seed"]) is not int
            for item in packages
        )
        or len({item["package_id"] for item in packages}) != FINAL_PACKAGE_COUNT
        or not isinstance(held_out, Mapping)
        or held_out
        != {
            "predictor_views": final_predictor_view_policy(packages),
            "inference_policy": SYMILE_TEST_INFERENCE_POLICY,
            "metric_policy": METRIC_POLICY,
            "operating_point_policy": DEVELOPMENT_DECISION_POLICY["threshold_derivation"],
            "subgroup_policy": SUBGROUP_POLICY,
            "bootstrap": BOOTSTRAP_POLICY,
            "calibration_policy": {
                "coefficients": "logit_calibration_slope_and_intercept",
                "reliability": RELIABILITY_POLICY,
            },
            "error_review_policy": ERROR_REVIEW_POLICY,
        }
    ):
        raise ManifestBuildError("Pre-test freeze fields are invalid")


def _identity(value: object, prefix: str) -> bool:
    return isinstance(value, str) and value.startswith(prefix) and _sha256(value[len(prefix) :])


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _publish_json_object(
    destination: Path,
    document: Mapping[str, object],
    validator: Callable[[Path], object],
) -> None:
    stage = staging_directory(destination)
    try:
        (stage / "manifest.json").write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        install_immutable_directory(stage, destination, validator)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _read_single_json(directory: str | Path) -> tuple[Path, bytes, dict[str, Any]]:
    root = Path(directory)
    if (
        root.is_symlink()
        or not root.is_dir()
        or {path.name for path in root.iterdir()} != {"manifest.json"}
        or (root / "manifest.json").is_symlink()
        or not (root / "manifest.json").is_file()
    ):
        raise ManifestBuildError("Scientific control object directory is invalid")
    raw = (root / "manifest.json").read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ManifestBuildError("Scientific control manifest is invalid")
    return root, raw, document
