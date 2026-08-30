"""Append-only control state for one exact RSNA formal execution."""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.config import normalized_scientific_config, require_runtime_seed
from beyondcxr.training.device import (
    full_precision_neural_runtime_policy,
    neural_inference_runtime_policy,
)
from beyondcxr.training.rsna_evaluation_result import (
    EVALUATION_POLICY_VERSION,
    CompletedRsnaEvaluation,
    validate_rsna_evaluation,
    validate_rsna_model_package,
    validated_rsna_evaluation_policy,
)
from beyondcxr.training.rsna_formal import (
    RSNA_FORMAL_RUN_PLAN,
    ValidatedRsnaPlan,
    require_validated_rsna_plan,
)
from beyondcxr.training.rsna_held_out_policy import RSNA_PREDICTION_POLICY, localization_policy
from beyondcxr.training.rsna_training_report import validate_training_report
from beyondcxr.training.rsna_validation_evidence import (
    load_canonical_validation_cohort,
    validate_package_validation_cohort,
)
from beyondcxr.utils.package_identity import (
    package_scientific_config_from_normalized,
    package_scientific_config_payload,
)
from beyondcxr.utils.publication import (
    is_publication_staging_directory,
    publish_bytes_no_replace,
    validate_path_component,
)
from beyondcxr.utils.semantic_identity import (
    canonical_json_bytes,
    canonical_scientific_id,
    semantic_sha256,
)

RSNA_EXECUTION_SCHEMA_VERSION = 1
RSNA_FREEZE_SCHEMA_VERSION = 1
RSNA_EVALUATION_RECORD_SCHEMA_VERSION = 1
EXECUTION_PREFIX = "rsna-execution-"
RSNA_FREEZE_PREFIX = "rsna-freeze-"
THREAD_POLICY = "framework_default_no_explicit_override"
_EXECUTION_GUARD = object()
_CURRENT_EXECUTION_GUARD = object()
_FREEZE_GUARD = object()
_HELD_OUT_GUARD = object()


@dataclass(frozen=True)
class ValidatedRsnaExecution:
    execution_id: str
    directory: Path
    manifest: Mapping[str, Any]
    _guard: object


@dataclass(frozen=True)
class ValidatedCurrentRsnaExecution:
    """Stored execution matched to one genuine current formal preflight plan."""

    execution: ValidatedRsnaExecution
    plan: ValidatedRsnaPlan
    _guard: object


@dataclass(frozen=True)
class FrozenScientificPackage:
    model_package_id: str
    model_path: Path


@dataclass(frozen=True)
class ValidatedRsnaFreeze:
    """Validated scientific package membership and held-out policy; no live access alone."""

    freeze_id: str
    directory: Path
    packages: tuple[FrozenScientificPackage, ...]
    model_root: Path
    bundle_directory: Path
    _guard: object

    def require(self, package_id: str) -> FrozenScientificPackage:
        require_validated_freeze(self)
        matches = tuple(item for item in self.packages if item.model_package_id == package_id)
        if len(matches) != 1:
            raise ManifestBuildError("RSNA package is not authorized by the complete freeze")
        return matches[0]

    def __iter__(self):
        return iter(self.packages)


def require_validated_freeze(value: object) -> ValidatedRsnaFreeze:
    """Reauthenticate one held-out capability against its canonical authorities."""
    if not isinstance(value, ValidatedRsnaFreeze) or value._guard is not _FREEZE_GUARD:
        raise PermissionError("RSNA held-out access requires a validated scientific freeze")
    canonical = validate_freeze(
        value.directory,
        model_root=value.model_root,
        bundle_directory=value.bundle_directory,
    )
    if canonical != value:
        raise ManifestBuildError(
            "RSNA scientific freeze capability differs from its canonical authority"
        )
    return value


@dataclass(frozen=True)
class ValidatedRsnaHeldOutCapability:
    """Scientific freeze joined to the independently current formal execution."""

    freeze: ValidatedRsnaFreeze
    current_execution: ValidatedCurrentRsnaExecution
    _guard: object

    def require(self, package_id: str) -> FrozenScientificPackage:
        require_validated_rsna_held_out(self)
        return self.freeze.require(package_id)

    @property
    def packages(self) -> tuple[FrozenScientificPackage, ...]:
        return self.freeze.packages


def authorize_rsna_held_out(
    *,
    plan: ValidatedRsnaPlan,
    execution: ValidatedRsnaExecution,
    freeze: ValidatedRsnaFreeze,
) -> ValidatedRsnaHeldOutCapability:
    """Issue held-out authority from the currently valid complete campaign join."""
    current = validate_current_rsna_execution(plan, execution)
    scientific = require_validated_freeze(freeze)
    document = _load_freeze_document(scientific.directory)
    configured = current.execution.manifest["configs"]
    freeze_root = plan.roots.control_root / "freezes"
    expected_directory = freeze_root / scientific.freeze_id
    try:
        freeze_entries = tuple(freeze_root.iterdir())
    except OSError as exc:
        raise ManifestBuildError("Canonical RSNA freeze root is unavailable") from exc
    if (
        scientific.directory.absolute() != expected_directory.absolute()
        or scientific.directory.resolve() != expected_directory.resolve()
        or freeze_root.is_symlink()
        or {entry.name for entry in freeze_entries} != {scientific.freeze_id}
        or any(entry.is_symlink() or not entry.is_dir() for entry in freeze_entries)
        or scientific.model_root != plan.roots.model_root.resolve()
        or scientific.bundle_directory != plan.authority.bundle_directory.resolve()
        or document["bundle_id"] != current.execution.manifest["dataset"]["bundle_id"]
        or current.execution.manifest["dataset"]
        != {
            "dataset_id": plan.authority.dataset_id,
            "bundle_id": plan.authority.bundle_id,
        }
        or current.execution.manifest["task"]
        != {
            "task_id": plan.authority.task_id,
            "label_policy_version": plan.authority.label_policy_version,
        }
    ):
        raise ManifestBuildError("RSNA freeze differs from the current formal execution plan")
    _validate_current_freeze_members(
        plan=plan,
        scientific=scientific,
        document=document,
        configured=configured,
    )
    validate_rsna_execution_freeze_science(current.execution, scientific)
    return ValidatedRsnaHeldOutCapability(scientific, current, _HELD_OUT_GUARD)


def _validate_current_freeze_members(
    *,
    plan: ValidatedRsnaPlan,
    scientific: ValidatedRsnaFreeze,
    document: Mapping[str, Any],
    configured: Sequence[Mapping[str, Any]],
) -> None:
    """Join every frozen package and policy to the current normalized run config."""
    try:
        for frozen, item, execution_item, config in zip(
            scientific.packages,
            document["packages"],
            configured,
            plan.configs,
            strict=True,
        ):
            package = validate_rsna_model_package(plan.roots.model_root, frozen.model_package_id)
            expected_seed = require_runtime_seed(config)
            package_seed = package.get("seed")
            if package_seed is None and isinstance(package.get("training_policy"), Mapping):
                package_seed = package["training_policy"].get("seed")
            if (
                execution_item["family_id"] != config.family.family_id
                or execution_item["seed"] != expected_seed
                or execution_item["normalized_config"] != normalized_scientific_config(config)
                or item["package_id"] != frozen.model_package_id
                or item["family_id"] != config.family.family_id
                or item["seed"] != expected_seed
                or package.get("dataset_id") != plan.authority.dataset_id
                or package.get("bundle_id") != plan.authority.bundle_id
                or package.get("task_id") != plan.authority.task_id
                or package.get("label_policy_version") != plan.authority.label_policy_version
                or package.get("family_id") != config.family.family_id
                or package_seed != expected_seed
                or package.get("fit_config") != package_scientific_config_payload(config)
                or item["evaluation_policy"] != validated_rsna_evaluation_policy(package, config)
            ):
                raise ManifestBuildError(
                    "RSNA freeze differs from the current formal execution plan"
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestBuildError(
            "RSNA freeze differs from the current formal execution plan"
        ) from exc


def validate_rsna_execution_freeze_science(
    execution: ValidatedRsnaExecution,
    freeze: ValidatedRsnaFreeze,
) -> None:
    """Join a historical execution to its exact recursively validated scientific freeze."""
    historical = _reauthenticate_execution(execution)
    scientific = require_validated_freeze(freeze)
    _validate_established_execution_freeze_science(historical, scientific)


def _validate_established_execution_freeze_science(
    historical: ValidatedRsnaExecution,
    scientific: ValidatedRsnaFreeze,
) -> None:
    """Check the relationship between locally established execution and freeze evidence."""
    document = _load_freeze_document(scientific.directory)
    try:
        if historical.manifest["dataset"] != {
            "dataset_id": document["dataset_id"],
            "bundle_id": document["bundle_id"],
        } or historical.manifest["task"] != {
            "task_id": "pneumonia",
            "label_policy_version": "rsna-stage-2-target-v1",
        }:
            raise ManifestBuildError("RSNA execution and scientific freeze disagree")
        for frozen, item, configured in zip(
            scientific.packages,
            document["packages"],
            historical.manifest["configs"],
            strict=True,
        ):
            package = validate_rsna_model_package(scientific.model_root, frozen.model_package_id)
            fit_config = package_scientific_config_from_normalized(configured["normalized_config"])
            package_seed = package.get("seed")
            if package_seed is None and isinstance(package.get("training_policy"), Mapping):
                package_seed = package["training_policy"].get("seed")
            evaluation = configured["normalized_config"]["evaluation"]
            if not isinstance(evaluation, Mapping):
                raise ManifestBuildError("RSNA execution evaluation policy is invalid")
            expected_evaluation = {
                "policy_version": EVALUATION_POLICY_VERSION,
                "calibration_bins": evaluation["calibration_bins"],
                "threshold_selection": package["threshold_contract"],
            }
            if (
                item["package_id"] != frozen.model_package_id
                or item["family_id"] != configured["family_id"]
                or item["seed"] != configured["seed"]
                or package.get("dataset_id") != document["dataset_id"]
                or package.get("bundle_id") != document["bundle_id"]
                or package.get("task_id") != historical.manifest["task"]["task_id"]
                or package.get("label_policy_version")
                != historical.manifest["task"]["label_policy_version"]
                or package.get("family_id") != configured["family_id"]
                or package_seed != configured["seed"]
                or package.get("fit_config") != fit_config
                or evaluation["sensitivity_target"]
                != package["threshold_contract"]["sensitivity_target"]
                or item["evaluation_policy"] != expected_evaluation
                or not _rsna_package_belongs_to_execution(package, historical.manifest)
            ):
                raise ManifestBuildError("RSNA execution and scientific freeze disagree")
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestBuildError("RSNA execution and scientific freeze disagree") from exc


def _rsna_package_belongs_to_execution(
    package: Mapping[str, Any], execution: Mapping[str, Any]
) -> bool:
    """Match package operational evidence to the formal execution that admitted it."""
    if "source_provenance" not in package:
        return (
            package.get("git_commit") == execution["git_commit"]
            and package.get("git_dirty") is False
            and package.get("dependency_lock_sha256") == execution["dependency_lock_sha256"]
            and package.get("python_version") == execution["framework_versions"]["python"]
        )
    source = package["source_provenance"]
    runtime = package["runtime_provenance"]
    loader = runtime.get("loader_execution") if isinstance(runtime, Mapping) else None
    weight = package["model_identity"]["pretrained_weight"]
    return (
        isinstance(source, Mapping)
        and source.get("git_commit") == execution["git_commit"]
        and source.get("git_dirty") is False
        and source.get("dependency_lock_sha256") == execution["dependency_lock_sha256"]
        and source.get("python_version") == execution["framework_versions"]["python"]
        and source.get("torch_version") == execution["framework_versions"]["torch"]
        and source.get("torchvision_version") == execution["framework_versions"]["torchvision"]
        and source.get("torchxrayvision_version")
        == execution["framework_versions"]["torchxrayvision"]
        and _pretrained_materialization_matches(weight, execution["pretrained_weight"])
        and _runtime_provenance_matches(runtime, execution["training_evaluation_runtime"])
        and _loader_provenance_matches(loader, execution["training_loader"])
    )


def resolve_completed_rsna_packages(
    plan: ValidatedRsnaPlan, execution: ValidatedRsnaExecution
) -> tuple[FrozenScientificPackage | None, ...]:
    """Resolve unique completed fits for the current, still-unfrozen formal plan."""
    require_validated_rsna_plan(plan)
    _require_execution_for_plan(plan, execution)
    root = plan.roots.model_root / "packages"
    completed: list[FrozenScientificPackage | None] = [None] * len(plan.configs)
    if not root.exists() and not root.is_symlink():
        return tuple(completed)
    if root.is_symlink() or not root.is_dir():
        raise ManifestBuildError("RSNA package root is invalid")
    sources: dict[int, str] = {}
    cohort = load_canonical_validation_cohort(
        plan.authority.bundle_directory,
        expected_bundle_id=plan.authority.bundle_id,
        expected_task_id=plan.authority.task_id,
        expected_label_policy_version=plan.authority.label_policy_version,
    )
    for directory in sorted(root.iterdir()):
        if is_publication_staging_directory(directory):
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise ManifestBuildError("RSNA package entry is invalid")
        package = validate_rsna_model_package(plan.roots.model_root, directory.name)
        seed = package.get("seed")
        if seed is None and isinstance(package.get("training_policy"), Mapping):
            seed = package["training_policy"].get("seed")
        for index, config in enumerate(plan.configs):
            if (
                package.get("bundle_id") != plan.authority.bundle_id
                or package.get("family_id") != config.family.family_id
                or seed != require_runtime_seed(config)
            ):
                continue
            expected = {
                "dataset_id": plan.authority.dataset_id,
                "task_id": plan.authority.task_id,
                "label_policy_version": plan.authority.label_policy_version,
                "fit_config": package_scientific_config_payload(config),
            }
            if (
                any(package.get(key) != value for key, value in expected.items())
                or not _rsna_package_belongs_to_execution(package, execution.manifest)
                or package["threshold_contract"]["sensitivity_target"]
                != config.evaluation.sensitivity_target
            ):
                raise ManifestBuildError("RSNA completed package differs from the formal plan")
            if completed[index] is not None:
                raise ManifestBuildError("RSNA completed package selection is ambiguous")
            validate_package_validation_cohort(directory, package=package, expected_cohort=cohort)
            models = [
                directory / name
                for name in ("model.skops", "model.pt")
                if (directory / name).is_file() and not (directory / name).is_symlink()
            ]
            if len(models) != 1:
                raise ManifestBuildError("RSNA completed package model is unavailable")
            completed[index] = FrozenScientificPackage(directory.name, models[0])
            if config.family.family_id == "cxr_metadata_concat":
                sources[index] = package["source_package_id"]
    cxr_members = {
        require_runtime_seed(config): package.model_package_id
        for config, package in zip(plan.configs, completed, strict=True)
        if config.family.family_id == "cxr_densenet" and package is not None
    }
    if any(
        source != cxr_members.get(require_runtime_seed(plan.configs[index]))
        for index, source in sources.items()
    ):
        raise ManifestBuildError("RSNA completed fusion package has conflicting CXR membership")
    return tuple(completed)


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


def require_validated_rsna_held_out(value: object) -> ValidatedRsnaHeldOutCapability:
    """Reauthenticate live RSNA held-out authority for one dataset and task."""
    if not isinstance(value, ValidatedRsnaHeldOutCapability) or value._guard is not _HELD_OUT_GUARD:
        raise PermissionError(
            "RSNA held-out access requires a validated current-execution capability"
        )
    current = require_current_rsna_execution(value.current_execution)
    observed = authorize_rsna_held_out(
        plan=current.plan,
        execution=current.execution,
        freeze=value.freeze,
    )
    if (
        observed.freeze != value.freeze
        or observed.current_execution.execution != value.current_execution.execution
        or observed.current_execution.plan.roots.repository_root
        != value.current_execution.plan.roots.repository_root
        or observed.current_execution.plan.roots.backup_root
        != value.current_execution.plan.roots.backup_root
    ):
        raise ManifestBuildError("RSNA held-out capability changed after validation")
    return observed


def require_validated_execution(value: object) -> ValidatedRsnaExecution:
    """Reject execution objects that did not originate from canonical validation."""
    if not isinstance(value, ValidatedRsnaExecution) or value._guard is not _EXECUTION_GUARD:
        raise PermissionError("RSNA control access requires a validated execution")
    return value


def validate_current_rsna_execution(
    plan: ValidatedRsnaPlan,
    execution: ValidatedRsnaExecution,
) -> ValidatedCurrentRsnaExecution:
    """Match stored execution to independently reconstructed current preflight state."""
    require_validated_rsna_plan(plan)
    value = _reauthenticate_execution(execution)
    expected = _execution_document(plan)
    if dict(value.manifest) != expected:
        raise ManifestBuildError("Current environment differs from the formal RSNA execution")
    _validate_active_execution(plan.roots.control_root, value.execution_id)
    return ValidatedCurrentRsnaExecution(value, plan, _CURRENT_EXECUTION_GUARD)


def require_current_rsna_execution(value: object) -> ValidatedCurrentRsnaExecution:
    if (
        not isinstance(value, ValidatedCurrentRsnaExecution)
        or value._guard is not _CURRENT_EXECUTION_GUARD
    ):
        raise PermissionError("RSNA lifecycle requires a current formal execution")
    from beyondcxr.training.rsna_campaign import _preflight_rsna_campaign

    fresh_plan = _preflight_rsna_campaign(
        backup_root=value.plan.roots.backup_root,
        source_root=value.plan.roots.source_root,
    )
    if fresh_plan.roots.repository_root != value.plan.roots.repository_root:
        raise ManifestBuildError("Current RSNA repository differs from the validated execution")
    observed = validate_current_rsna_execution(fresh_plan, value.execution)
    if (
        observed.execution != value.execution
        or observed.plan.roots.repository_root != value.plan.roots.repository_root
        or observed.plan.roots.backup_root != value.plan.roots.backup_root
    ):
        raise ManifestBuildError("Current RSNA execution changed after validation")
    return observed


def _publish_or_validate_active_execution(control_root: Path, execution_id: str) -> None:
    path = control_root / "active-execution.json"
    expected = {"schema_version": 1, "execution_id": execution_id}
    encoded = canonical_json_bytes(expected)
    if publish_bytes_no_replace(path, encoded) is False and path.read_bytes() != encoded:
        raise ManifestBuildError("RSNA campaign is active under another execution")
    _validate_active_execution(control_root, execution_id)


def _validate_active_initialization(control_root: Path, execution_id: str) -> None:
    path = control_root / "active-execution.json"
    if path.exists() or path.is_symlink():
        if _active_execution_id(control_root) != execution_id:
            raise ManifestBuildError("RSNA campaign is active under another execution")
    elif _rsna_protected_state_exists(control_root):
        raise ManifestBuildError("RSNA protected state lacks an active execution receipt")


def _rsna_protected_state_exists(control_root: Path) -> bool:
    repository = control_root.parents[2] if len(control_root.parents) > 2 else None
    freezes = control_root / "freezes"
    return (freezes.exists() or freezes.is_symlink()) or (
        repository is not None
        and any(
            path.exists() or path.is_symlink()
            for path in (
                repository / "models/rsna/packages",
                repository / "reports/rsna/runs",
                repository / "private/predictions/rsna",
            )
        )
    )


def _validate_active_execution(control_root: Path, execution_id: str) -> None:
    if _active_execution_id(control_root) != execution_id:
        raise ManifestBuildError("RSNA active execution receipt is invalid")


def _active_execution_id(control_root: Path) -> str:
    path = control_root / "active-execution.json"
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("RSNA active execution receipt is unavailable") from exc
    execution_id = document.get("execution_id") if isinstance(document, dict) else None
    expected = {"schema_version": 1, "execution_id": execution_id}
    if (
        path.is_symlink()
        or not _identity(execution_id, EXECUTION_PREFIX)
        or document != expected
        or raw != canonical_json_bytes(expected)
    ):
        raise ManifestBuildError("RSNA active execution receipt is invalid")
    return execution_id


def publish_or_validate_execution(
    plan: ValidatedRsnaPlan,
) -> ValidatedRsnaExecution:
    """Publish or reuse the immutable identity of one prevalidated formal execution."""
    require_validated_rsna_plan(plan)
    if plan.runs != RSNA_FORMAL_RUN_PLAN or len(plan.configs) != len(RSNA_FORMAL_RUN_PLAN):
        raise ManifestBuildError("RSNA formal plan membership is invalid")
    for spec, config in zip(plan.runs, plan.configs, strict=True):
        if (
            config.family.family_id != spec.family_id
            or require_runtime_seed(config) != spec.seed
            or config.dataset.dataset_id != plan.authority.dataset_id
            or config.dataset.bundle_id != plan.authority.bundle_id
            or config.task.task_id != plan.authority.task_id
            or config.task.label_policy_version != plan.authority.label_policy_version
        ):
            raise ManifestBuildError("RSNA formal plan config agreement is invalid")
    document = _execution_document(plan)
    execution_id = str(document["execution_id"])
    execution_root = plan.roots.control_root / "executions"
    _validate_active_initialization(plan.roots.control_root, execution_id)
    directory = execution_root / execution_id
    path = directory / "manifest.json"
    encoded = canonical_json_bytes(document)
    if publish_bytes_no_replace(path, encoded) is False and path.read_bytes() != encoded:
        raise ManifestBuildError("Existing RSNA execution control differs from current preflight")
    execution = validate_execution(directory, expected_execution_id=execution_id)
    _publish_or_validate_active_execution(plan.roots.control_root, execution_id)
    return execution


def _execution_document(plan: ValidatedRsnaPlan) -> dict[str, Any]:
    semantic = {
        "dataset": {
            "dataset_id": plan.authority.dataset_id,
            "bundle_id": plan.authority.bundle_id,
        },
        "task": {
            "task_id": plan.authority.task_id,
            "label_policy_version": plan.authority.label_policy_version,
        },
        "configs": [
            {
                "family_id": config.family.family_id,
                "seed": require_runtime_seed(config),
                "config_path": spec.config_relative.as_posix(),
                "normalized_config": normalized_scientific_config(config),
            }
            for spec, config in zip(plan.runs, plan.configs, strict=True)
        ],
        "framework_versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": plan.runtime.torchvision_version,
            "torchxrayvision": plan.runtime.torchxrayvision_version,
        },
        "pretrained_weight": _execution_pretrained_weight(plan.pretrained_weight.as_dict()),
        "training_evaluation_runtime": neural_inference_runtime_policy(plan.runtime),
        "localization_runtime": full_precision_neural_runtime_policy(plan.runtime),
        "training_loader": {
            **plan.training_execution.provenance(),
            "prefetch_factor": plan.training_execution.prefetch_factor,
        },
        "evaluation_loader": {
            **plan.evaluation_execution.provenance(),
            "prefetch_factor": plan.evaluation_execution.prefetch_factor,
        },
        "thread_policy": THREAD_POLICY,
        "git_commit": plan.git_commit,
        "dependency_lock_sha256": plan.dependency_lock_sha256,
    }
    execution_id = EXECUTION_PREFIX + semantic_sha256(
        "rsna-formal-execution", RSNA_EXECUTION_SCHEMA_VERSION, semantic
    )
    return {
        "rsna_execution_schema_version": RSNA_EXECUTION_SCHEMA_VERSION,
        "execution_id": execution_id,
        **semantic,
    }


def validate_execution(
    directory: str | Path,
    *,
    expected_execution_id: str | None = None,
) -> ValidatedRsnaExecution:
    """Validate a stored formal execution without requiring its historical runtime."""
    root = Path(directory)
    path = root / "manifest.json"
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("RSNA execution control is unreadable") from exc
    fields = {
        "rsna_execution_schema_version",
        "execution_id",
        "dataset",
        "task",
        "configs",
        "framework_versions",
        "pretrained_weight",
        "training_evaluation_runtime",
        "localization_runtime",
        "training_loader",
        "evaluation_loader",
        "thread_policy",
        "git_commit",
        "dependency_lock_sha256",
    }
    if (
        root.is_symlink()
        or not root.is_dir()
        or path.is_symlink()
        or not isinstance(document, dict)
        or set(document) != fields
        or document["rsna_execution_schema_version"] != RSNA_EXECUTION_SCHEMA_VERSION
        or type(document["rsna_execution_schema_version"]) is not int
        or raw != canonical_json_bytes(document)
    ):
        raise ManifestBuildError("RSNA execution control contract is invalid")
    execution_id = document["execution_id"]
    semantic = {
        key: document[key]
        for key in fields
        if key not in {"execution_id", "rsna_execution_schema_version"}
    }
    expected = EXECUTION_PREFIX + semantic_sha256(
        "rsna-formal-execution", RSNA_EXECUTION_SCHEMA_VERSION, semantic
    )
    if (
        execution_id != expected
        or root.name != expected
        or (expected_execution_id is not None and execution_id != expected_execution_id)
    ):
        raise ManifestBuildError("RSNA execution control identity is invalid")
    _validate_plan_fields(document)
    entries = tuple(root.iterdir())
    if {path.name for path in entries} != {"manifest.json"} or any(
        path.is_symlink() or not path.is_file() for path in entries
    ):
        raise ManifestBuildError("RSNA execution directory membership is invalid")
    return ValidatedRsnaExecution(execution_id, root, document, _EXECUTION_GUARD)


def publish_or_validate_freeze(
    *,
    plan: ValidatedRsnaPlan,
    execution: ValidatedRsnaExecution,
    results: Sequence[object],
) -> ValidatedRsnaFreeze:
    """Freeze the exact complete package set before any held-out evaluation."""
    require_validated_rsna_plan(plan)
    _require_execution_for_plan(plan, execution)
    values = tuple(results)
    if len(values) != 8:
        raise ManifestBuildError("RSNA scientific freeze requires eight packages")
    models = plan.roots.model_root.resolve()
    cohort = load_canonical_validation_cohort(
        plan.authority.bundle_directory,
        expected_bundle_id=plan.authority.bundle_id,
        expected_task_id=plan.authority.task_id,
        expected_label_policy_version=plan.authority.label_policy_version,
    )
    members = []
    for result, config in zip(values, plan.configs, strict=True):
        package_id = str(result.model_package_id)
        package = validate_rsna_model_package(models, package_id)
        model_path = _validated_output_path(Path(result.model_path), models, "model")
        expected_lineage = {
            "bundle_id": plan.authority.bundle_id,
            "task_id": plan.authority.task_id,
            "label_policy_version": plan.authority.label_policy_version,
            "family_id": config.family.family_id,
            "fit_config": package_scientific_config_payload(config),
        }
        package_seed = package.get("seed")
        if package_seed is None and isinstance(package.get("training_policy"), Mapping):
            package_seed = package["training_policy"].get("seed")
        if (
            any(package.get(key) != value for key, value in expected_lineage.items())
            or package_seed != require_runtime_seed(config)
            or model_path.name not in {"model.skops", "model.pt"}
            or not model_path.is_file()
        ):
            raise ManifestBuildError("RSNA scientific freeze output is invalid")
        validate_training_report(
            result.artifact_directory,
            run_id=str(result.run_id),
            model_package_id=package_id,
            family_id=config.family.family_id,
            seed=require_runtime_seed(config),
            package=package,
            package_directory=models / "packages" / package_id,
            canonical_validation_cohort=cohort,
        )
        members.append(
            {
                "package_id": package_id,
                "family_id": config.family.family_id,
                "seed": require_runtime_seed(config),
                "evaluation_policy": validated_rsna_evaluation_policy(package, config),
            }
        )
    if len({item["package_id"] for item in members}) != 8:
        raise ManifestBuildError("RSNA scientific freeze contains duplicate packages")
    semantic = {
        "dataset_id": "rsna",
        "bundle_id": plan.authority.bundle_id,
        "packages": members,
        "prediction_policy": RSNA_PREDICTION_POLICY,
        "localization_policy": localization_policy(),
    }
    freeze_id = canonical_scientific_id(
        RSNA_FREEZE_PREFIX, semantic, domain="rsna-freeze", schema_version=1
    )
    document = {
        "schema_version": RSNA_FREEZE_SCHEMA_VERSION,
        "freeze_id": freeze_id,
        **semantic,
    }
    directory = plan.roots.control_root / "freezes" / freeze_id
    path = directory / "manifest.json"
    encoded = canonical_json_bytes(document)
    if publish_bytes_no_replace(path, encoded) is False and path.read_bytes() != encoded:
        raise ManifestBuildError("Existing RSNA scientific freeze differs from completed training")
    return validate_freeze(
        directory,
        model_root=models,
        bundle_directory=plan.authority.bundle_directory,
    )


def validate_freeze(
    directory: str | Path,
    *,
    model_root: str | Path,
    bundle_directory: str | Path,
) -> ValidatedRsnaFreeze:
    """Revalidate frozen package membership, scientific policy, and semantic identity."""
    root = Path(directory)
    document = _load_freeze_document(root)
    models = Path(model_root).resolve()
    bundle = Path(bundle_directory).resolve()
    cohort = load_canonical_validation_cohort(
        bundle,
        expected_bundle_id=str(document["bundle_id"]),
        expected_task_id="pneumonia",
        expected_label_policy_version="rsna-stage-2-target-v1",
    )
    results = []
    for item, configured in zip(document["packages"], RSNA_FORMAL_RUN_PLAN, strict=True):
        if not isinstance(item, dict) or set(item) != {
            "package_id",
            "family_id",
            "seed",
            "evaluation_policy",
        }:
            raise ManifestBuildError("RSNA scientific freeze member is invalid")
        package_id = validate_path_component(item["package_id"], "RSNA package ID")
        package = validate_rsna_model_package(models, package_id)
        package_directory = models / "packages" / package_id
        candidates = tuple(
            path
            for path in (package_directory / "model.skops", package_directory / "model.pt")
            if path.is_file() and not path.is_symlink()
        )
        if len(candidates) != 1:
            raise ManifestBuildError("RSNA frozen package model is unavailable")
        model_path = candidates[0]
        validate_package_validation_cohort(
            package_directory,
            package=package,
            expected_cohort=cohort,
        )
        package_seed = package.get("seed")
        if package_seed is None and isinstance(package.get("training_policy"), Mapping):
            package_seed = package["training_policy"].get("seed")
        expected_lineage = {
            "bundle_id": document["bundle_id"],
            "task_id": "pneumonia",
            "label_policy_version": "rsna-stage-2-target-v1",
            "family_id": configured.family_id,
        }
        if (
            any(package.get(key) != value for key, value in expected_lineage.items())
            or package_seed != configured.seed
            or item["family_id"] != configured.family_id
            or item["seed"] != configured.seed
            or not _frozen_evaluation_policy_matches(item["evaluation_policy"], package)
        ):
            raise ManifestBuildError("RSNA scientific freeze output is unavailable")
        results.append(FrozenScientificPackage(package_id, model_path))
    if len({result.model_package_id for result in results}) != 8:
        raise ManifestBuildError("RSNA scientific freeze contains duplicate packages")
    return ValidatedRsnaFreeze(
        str(document["freeze_id"]),
        root,
        tuple(results),
        models,
        bundle,
        _FREEZE_GUARD,
    )


def _load_freeze_document(root: Path) -> dict[str, Any]:
    """Load and validate the canonical schema-1 scientific freeze document."""
    path = root / "manifest.json"
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("RSNA scientific freeze is unreadable") from exc
    if (
        root.is_symlink()
        or not root.is_dir()
        or path.is_symlink()
        or not isinstance(document, dict)
        or set(document)
        != {
            "schema_version",
            "freeze_id",
            "dataset_id",
            "bundle_id",
            "packages",
            "prediction_policy",
            "localization_policy",
        }
        or type(document["schema_version"]) is not int
        or document["schema_version"] != RSNA_FREEZE_SCHEMA_VERSION
        or raw != canonical_json_bytes(document)
        or document["dataset_id"] != "rsna"
        or not _identity(document["bundle_id"], "bundle-")
        or document["prediction_policy"] != RSNA_PREDICTION_POLICY
        or document["localization_policy"] != localization_policy()
        or not isinstance(document["packages"], list)
        or len(document["packages"]) != 8
    ):
        raise ManifestBuildError("RSNA scientific freeze contract is invalid")
    semantic = {
        key: document[key] for key in document if key not in {"schema_version", "freeze_id"}
    }
    expected = canonical_scientific_id(
        RSNA_FREEZE_PREFIX, semantic, domain="rsna-freeze", schema_version=1
    )
    if document["freeze_id"] != expected or root.name != expected:
        raise ManifestBuildError("RSNA scientific freeze identity is invalid")
    if {entry.name for entry in root.iterdir()} != {"manifest.json"}:
        raise ManifestBuildError("RSNA scientific freeze directory membership is invalid")
    return document


def _frozen_evaluation_policy_matches(value: object, package: Mapping[str, Any]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"policy_version", "calibration_bins", "threshold_selection"}
        and value["policy_version"] == EVALUATION_POLICY_VERSION
        and type(value["calibration_bins"]) is int
        and 2 <= value["calibration_bins"] <= 1000
        and value["threshold_selection"] == package.get("threshold_contract")
    )


def load_evaluation_record(
    execution: ValidatedRsnaExecution,
    *,
    freeze: ValidatedRsnaFreeze,
    package_id: str,
    report_root: str | Path,
    private_root: str | Path,
    model_root: str | Path,
) -> CompletedRsnaEvaluation | None:
    """Load one completed package-bound evaluation, if its record exists."""
    execution = _reauthenticate_execution(execution)
    freeze = require_validated_freeze(freeze)
    _validate_established_execution_freeze_science(execution, freeze)
    return _load_evaluation_record_from_established_context(
        execution,
        freeze=freeze,
        package_id=package_id,
        report_root=report_root,
        private_root=private_root,
        model_root=model_root,
    )


def _load_evaluation_record_from_established_context(
    execution: ValidatedRsnaExecution,
    *,
    freeze: ValidatedRsnaFreeze,
    package_id: str,
    report_root: str | Path,
    private_root: str | Path,
    model_root: str | Path,
) -> CompletedRsnaEvaluation | None:
    """Load one record under a locally established execution/freeze relationship."""
    freeze.require(package_id)
    path = _evaluation_record_root(execution) / f"{package_id}.json"
    if not path.exists() and not path.is_symlink():
        return None
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("RSNA evaluation completion record is unreadable") from exc
    fields = {
        "rsna_evaluation_record_schema_version",
        "execution_id",
        "package_id",
        "evaluation_id",
        "prediction_id",
        "mlflow_run_id",
    }
    if (
        path.is_symlink()
        or not isinstance(document, dict)
        or set(document) != fields
        or type(document["rsna_evaluation_record_schema_version"]) is not int
        or document["rsna_evaluation_record_schema_version"]
        != RSNA_EVALUATION_RECORD_SCHEMA_VERSION
        or document["execution_id"] != execution.execution_id
        or document["package_id"] != package_id
        or raw != canonical_json_bytes(document)
    ):
        raise ManifestBuildError("RSNA evaluation completion record is invalid")
    evaluation_id = validate_path_component(document["evaluation_id"], "RSNA evaluation ID")
    validated = validate_rsna_evaluation(
        Path(report_root) / "rsna" / "evaluations" / evaluation_id,
        private_root=private_root,
        model_root=model_root,
        expected_evaluation_id=evaluation_id,
        freeze=freeze,
        execution=execution,
    )
    if (
        validated.manifest["model_package_id"] != package_id
        or validated.manifest["prediction_id"] != document["prediction_id"]
    ):
        raise ManifestBuildError("RSNA evaluation completion lineage is invalid")
    return CompletedRsnaEvaluation(
        evaluation_id=evaluation_id,
        prediction_id=str(document["prediction_id"]),
        model_package_id=package_id,
        mlflow_run_id=str(document["mlflow_run_id"]),
        artifact_directory=validated.directory,
        private_prediction_directory=(
            Path(private_root) / "predictions" / "rsna" / str(document["prediction_id"])
        ),
        average_precision=float(
            validated.manifest["claims"]["probability_metrics"]["average_precision"]
        ),
    )


def publish_evaluation_record(
    execution: ValidatedRsnaExecution,
    result: CompletedRsnaEvaluation,
    *,
    freeze: ValidatedRsnaFreeze,
    report_root: str | Path,
    private_root: str | Path,
    model_root: str | Path,
) -> None:
    execution = _reauthenticate_execution(execution)
    freeze = require_validated_freeze(freeze)
    _validate_established_execution_freeze_science(execution, freeze)
    freeze.require(result.model_package_id)
    validated = validate_rsna_evaluation(
        Path(report_root) / "rsna" / "evaluations" / result.evaluation_id,
        private_root=private_root,
        model_root=model_root,
        freeze=freeze,
        execution=execution,
        expected_evaluation_id=result.evaluation_id,
    )
    if (
        validated.manifest["prediction_id"] != result.prediction_id
        or validated.manifest["model_package_id"] != result.model_package_id
        or result.artifact_directory.resolve() != validated.directory.resolve()
        or result.private_prediction_directory.resolve()
        != (Path(private_root) / "predictions" / "rsna" / result.prediction_id).resolve()
    ):
        raise ManifestBuildError("RSNA evaluation completion authority is invalid")
    document = {
        "rsna_evaluation_record_schema_version": RSNA_EVALUATION_RECORD_SCHEMA_VERSION,
        "execution_id": execution.execution_id,
        "package_id": result.model_package_id,
        "evaluation_id": result.evaluation_id,
        "prediction_id": result.prediction_id,
        "mlflow_run_id": result.mlflow_run_id,
    }
    path = _evaluation_record_root(execution) / f"{result.model_package_id}.json"
    encoded = canonical_json_bytes(document)
    if publish_bytes_no_replace(path, encoded) is False and path.read_bytes() != encoded:
        raise ManifestBuildError("Existing RSNA evaluation record differs from current result")


def validate_execution_closure(
    execution: ValidatedRsnaExecution,
    freeze: ValidatedRsnaFreeze,
) -> None:
    """Require the exact completed control-tree membership for one execution."""
    frozen = require_validated_freeze(freeze)
    execution = _reauthenticate_execution(execution)
    _validate_established_execution_freeze_science(execution, frozen)
    expected_root = {"manifest.json"}
    entries = tuple(execution.directory.iterdir())
    if {entry.name for entry in entries} != expected_root or any(
        entry.is_symlink() for entry in entries
    ):
        raise ManifestBuildError("RSNA execution control membership is invalid")
    evaluations = _evaluation_record_root(execution)
    if evaluations.is_symlink() or not evaluations.is_dir():
        raise ManifestBuildError("RSNA execution records directory is invalid")
    expected_records = {f"{item.model_package_id}.json" for item in frozen.packages}
    records = tuple(evaluations.iterdir())
    if {record.name for record in records} != expected_records or any(
        record.is_symlink() or not record.is_file() for record in records
    ):
        raise ManifestBuildError("RSNA evaluation record membership is invalid")


def _evaluation_record_root(execution: ValidatedRsnaExecution) -> Path:
    return execution.directory.parent.parent / "evaluations" / execution.execution_id


def _require_execution_for_plan(plan: ValidatedRsnaPlan, execution: ValidatedRsnaExecution) -> None:
    require_validated_execution(execution)
    expected = _execution_document(plan)
    try:
        observed = validate_execution(
            execution.directory,
            expected_execution_id=str(expected["execution_id"]),
        )
    except ManifestBuildError as exc:
        raise ManifestBuildError(
            "RSNA execution control does not match the validated plan"
        ) from exc
    if dict(observed.manifest) != expected or observed != execution:
        raise ManifestBuildError("RSNA execution control does not match the validated plan")


def _reauthenticate_execution(
    execution: ValidatedRsnaExecution,
) -> ValidatedRsnaExecution:
    require_validated_execution(execution)
    observed = validate_execution(
        execution.directory,
        expected_execution_id=execution.execution_id,
    )
    if observed != execution:
        raise ManifestBuildError("RSNA execution capability differs from its canonical authority")
    return observed


def _validate_plan_fields(document: Mapping[str, Any]) -> None:
    dataset = document.get("dataset")
    task = document.get("task")
    configs = document.get("configs")
    weight = document.get("pretrained_weight")
    numerical = document.get("training_evaluation_runtime")
    localization = document.get("localization_runtime")
    frameworks = document.get("framework_versions")
    training_loader = document.get("training_loader")
    evaluation_loader = document.get("evaluation_loader")
    if (
        not isinstance(dataset, dict)
        or set(dataset) != {"dataset_id", "bundle_id"}
        or not isinstance(task, dict)
        or set(task) != {"task_id", "label_policy_version"}
        or not isinstance(configs, list)
        or len(configs) != 8
        or any(
            not isinstance(item, dict)
            or set(item) != {"family_id", "seed", "config_path", "normalized_config"}
            for item in configs
        )
        or not isinstance(frameworks, dict)
        or set(frameworks) != {"python", "torch", "torchvision", "torchxrayvision"}
        or any(not isinstance(value, str) or not value for value in frameworks.values())
        or not isinstance(weight, dict)
        or set(weight) != {"declared_name", "stable_identifier", "byte_size", "sha256"}
        or not isinstance(numerical, dict)
        or not isinstance(localization, dict)
        or set(numerical)
        != {
            "device_type",
            "autocast_dtype",
            "deterministic_algorithms",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "cuda_runtime_version",
            "cudnn_version",
            "gpu_device_name",
            "gpu_compute_capability",
        }
        or set(localization) != set(numerical)
        or training_loader
        != {
            "lifecycle": "reused",
            "num_workers": 2,
            "pin_memory": True,
            "prefetch_factor": 2,
        }
        or evaluation_loader
        != {
            "lifecycle": "one_shot",
            "num_workers": 0,
            "pin_memory": True,
            "prefetch_factor": None,
        }
        or document.get("thread_policy") != THREAD_POLICY
    ):
        raise ManifestBuildError("RSNA execution control fields are invalid")
    assert isinstance(dataset, dict) and isinstance(task, dict) and isinstance(configs, list)
    if (
        dataset["dataset_id"] != "rsna"
        or not _identity(dataset["bundle_id"], "bundle-")
        or task["task_id"] != "pneumonia"
        or not isinstance(task["label_policy_version"], str)
        or not task["label_policy_version"]
        or not isinstance(document.get("git_commit"), str)
        or len(document["git_commit"]) != 40
        or any(character not in "0123456789abcdef" for character in document["git_commit"])
        or not _sha256(document.get("dependency_lock_sha256"))
        or weight["declared_name"] != "densenet121-res224-chex"
        or not isinstance(weight["stable_identifier"], str)
        or not weight["stable_identifier"]
        or type(weight["byte_size"]) is not int
        or weight["byte_size"] <= 0
        or not _sha256(weight["sha256"])
        or numerical["device_type"] != "cuda"
        or numerical["autocast_dtype"] != "float16"
        or numerical["deterministic_algorithms"] != "enabled_warn_only"
        or numerical["cudnn_deterministic"] is not True
        or numerical["cudnn_benchmark"] is not False
        or not isinstance(numerical["cuda_runtime_version"], str)
        or not numerical["cuda_runtime_version"]
        or type(numerical["cudnn_version"]) is not int
        or not isinstance(numerical["gpu_device_name"], str)
        or not numerical["gpu_device_name"]
        or not isinstance(numerical["gpu_compute_capability"], list)
        or len(numerical["gpu_compute_capability"]) != 2
        or any(type(value) is not int for value in numerical["gpu_compute_capability"])
        or localization != {**numerical, "autocast_dtype": None}
    ):
        raise ManifestBuildError("RSNA execution control coordinates are invalid")
    if any(
        item["family_id"] != family
        or item["seed"] != seed
        or item["config_path"] != spec.config_relative.as_posix()
        or not isinstance(item["normalized_config"], dict)
        for item, spec in zip(configs, RSNA_FORMAL_RUN_PLAN, strict=True)
        for family, seed in ((spec.family_id, spec.seed),)
    ):
        raise ManifestBuildError("RSNA execution configuration matrix is invalid")


def _identity(value: object, prefix: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(prefix)
        and len(value) == len(prefix) + 64
        and all(character in "0123456789abcdef" for character in value.removeprefix(prefix))
    )


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _execution_pretrained_weight(value: Mapping[str, object]) -> dict[str, object]:
    required = {"declared_name", "stable_identifier", "cache_filename", "byte_size", "sha256"}
    if set(value) != required:
        raise ManifestBuildError("RSNA pretrained materialization evidence is invalid")
    return {
        "declared_name": value["declared_name"],
        "stable_identifier": value["stable_identifier"],
        "byte_size": value["byte_size"],
        "sha256": value["sha256"],
    }


def _validated_output_path(path: Path, root: Path, kind: str) -> Path:
    absolute = path if path.is_absolute() else Path.cwd() / path
    resolved_root = root.resolve()
    try:
        relative = absolute.relative_to(root if root.is_absolute() else Path.cwd() / root)
    except ValueError as exc:
        raise ManifestBuildError(f"RSNA {kind} output is outside its controlled root") from exc
    current = root if root.is_absolute() else Path.cwd() / root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ManifestBuildError(f"RSNA {kind} output contains a symlink")
    resolved = absolute.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ManifestBuildError(f"RSNA {kind} output escapes its controlled root")
    return resolved
