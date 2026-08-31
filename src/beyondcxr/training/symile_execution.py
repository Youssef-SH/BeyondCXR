"""Operational authority for one formal Symile execution."""

from __future__ import annotations

import json
import os
import platform
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_artifacts import resolve_symile_bundle
from beyondcxr.data.symile_cv import CV_DIRECTORY, validate_symile_cv
from beyondcxr.models.cxr_baseline import fingerprint_pretrained_weights
from beyondcxr.training.campaign_lock import campaign_lock
from beyondcxr.training.config import (
    ExperimentConfig,
    load_symile_development_config,
    normalized_scientific_config,
)
from beyondcxr.training.device import (
    ResolvedDevice,
    neural_inference_runtime_policy,
    resolve_device,
)
from beyondcxr.training.execution import one_shot_loader_policy, reused_loader_policy
from beyondcxr.training.operational_validation import validate_directory_chain
from beyondcxr.training.symile_formal import SYMILE_FORMAL_RUN_PLAN
from beyondcxr.utils.provenance import git_revision, uv_lock_sha256
from beyondcxr.utils.publication import publish_bytes_no_replace
from beyondcxr.utils.semantic_identity import (
    canonical_json_bytes,
    semantic_sha256,
)

SYMILE_EXECUTION_SCHEMA_VERSION = 1
SYMILE_EXECUTION_PREFIX = "symile-execution-"
THREAD_POLICY = "framework_default_no_explicit_override"
_EXECUTION_GUARD = object()
_PREFLIGHT_GUARD = object()
_CURRENT_EXECUTION_GUARD = object()
_LIFECYCLE_GUARD = object()
_ACTIVE_PROSPECTIVE_MUTATIONS: dict[int, ValidatedSymileProspectiveMutation] = {}


@dataclass(frozen=True)
class ValidatedSymileExecution:
    """One structurally validated immutable operational execution record."""

    execution_id: str
    directory: Path
    manifest: Mapping[str, Any]
    _guard: object


@dataclass(frozen=True)
class ValidatedSymileFormalPreflight:
    """Mutation-free current formal inputs derived from reviewed local state."""

    repository_root: Path
    control_root: Path
    configs: tuple[ExperimentConfig, ...]
    runtime: ResolvedDevice
    document: Mapping[str, Any]
    _guard: object


@dataclass(frozen=True)
class ValidatedCurrentSymileExecution:
    """Stored execution proven equal to independently reconstructed preflight state."""

    execution: ValidatedSymileExecution
    repository_root: Path
    _guard: object

    @property
    def control_root(self) -> Path:
        return self.repository_root / "private" / "control" / "symile"


@dataclass(frozen=True)
class ValidatedSymileLifecycle:
    """Canonical local control and approved persistent backup coordinates."""

    current_execution: ValidatedCurrentSymileExecution
    repository_root: Path
    private_root: Path
    control_root: Path
    backup_root: Path
    _guard: object


@dataclass(frozen=True, slots=True, init=False)
class ValidatedSymileProspectiveMutation:
    """One non-transferable mutation transaction held under the campaign lock."""

    lifecycle: ValidatedSymileLifecycle


def build_symile_formal_preflight(
    repository_root: str | Path, *, bundle_id: str, cv_assignment_id: str
) -> ValidatedSymileFormalPreflight:
    """Derive the complete current formal execution contract without mutation."""
    repository = Path(repository_root).resolve(strict=True)
    control = repository / "private" / "control" / "symile"
    configs = tuple(
        load_symile_development_config(
            repository / spec.config_relative,
            bundle_id=bundle_id,
            cv_assignment_id=cv_assignment_id,
        )
        for spec in SYMILE_FORMAL_RUN_PLAN
    )
    if any(
        config.family.family_id != spec.family_id
        for config, spec in zip(configs, SYMILE_FORMAL_RUN_PLAN, strict=True)
    ):
        raise ManifestBuildError("Reviewed Symile formal configuration plan is invalid")
    bundle_ids = {config.dataset.bundle_id for config in configs}
    cv_ids = {config.dataset.cv_assignment_id for config in configs}
    if len(bundle_ids) != 1 or len(cv_ids) != 1 or None in cv_ids:
        raise ManifestBuildError("Reviewed Symile formal dataset coordinates disagree")
    bundle_id = str(next(iter(bundle_ids)))
    cv_assignment_id = str(next(iter(cv_ids)))
    manifest_root = repository / "data" / "manifests"
    bundle = resolve_symile_bundle(
        manifest_root,
        bundle_id=bundle_id,
        full_validation=True,
    )
    validate_symile_cv(
        manifest_root / "symile" / CV_DIRECTORY / cv_assignment_id,
        bundle=bundle,
        expected_assignment_id=cv_assignment_id,
    )
    commit, dirty = git_revision(repository)
    if dirty:
        raise ManifestBuildError("Formal Symile campaign requires a clean Git worktree")
    reference = next(config for config in configs if config.neural is not None)
    runtime = resolve_device(
        "cuda",
        mixed_precision=reference.neural.mixed_precision,
        pin_memory_policy=reference.runtime.pin_memory_policy,
    )
    if runtime.device.type != "cuda":
        raise ManifestBuildError("Formal Symile execution requires CUDA")
    weight = fingerprint_pretrained_weights(str(reference.family.parameters["weights"]))
    training_loader = reused_loader_policy(
        num_workers=2,
        pin_memory=runtime.pin_memory_effective,
    )
    inference_loader = one_shot_loader_policy(pin_memory=runtime.pin_memory_effective)
    document = symile_execution_document(
        bundle_id=bundle_id,
        cv_assignment_id=cv_assignment_id,
        configs=[
            {
                "family_id": spec.family_id,
                "config_path": spec.config_relative.as_posix(),
                "normalized_config": normalized_scientific_config(config),
            }
            for spec, config in zip(SYMILE_FORMAL_RUN_PLAN, configs, strict=True)
        ],
        git_commit=commit,
        dependency_lock_sha256=uv_lock_sha256(repository / "uv.lock"),
        framework_versions={
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": runtime.torchvision_version,
            "torchxrayvision": runtime.torchxrayvision_version,
        },
        neural_runtime=neural_inference_runtime_policy(runtime),
        training_loader={
            **training_loader.provenance(),
            "prefetch_factor": training_loader.prefetch_factor,
        },
        inference_loader={
            **inference_loader.provenance(),
            "prefetch_factor": inference_loader.prefetch_factor,
        },
        pretrained_weight=weight.as_dict(),
    )
    return ValidatedSymileFormalPreflight(
        repository, control, configs, runtime, document, _PREFLIGHT_GUARD
    )


def require_symile_formal_preflight(
    value: object,
) -> ValidatedSymileFormalPreflight:
    if (
        not isinstance(value, ValidatedSymileFormalPreflight)
        or value._guard is not _PREFLIGHT_GUARD
    ):
        raise PermissionError("Symile execution requires a genuine formal preflight")
    if value.control_root != value.repository_root / "private" / "control" / "symile":
        raise ManifestBuildError("Symile formal preflight control root is not canonical")
    _validate_execution_fields(value.document)
    return value


def publish_or_validate_symile_execution(
    preflight: ValidatedSymileFormalPreflight,
) -> ValidatedSymileExecution:
    """Publish the preflight-derived execution record before protected work."""
    current = require_symile_formal_preflight(preflight)
    document = dict(current.document)
    execution_id = str(document["execution_id"])
    control = current.control_root
    validate_directory_chain(
        current.repository_root,
        control,
        "Symile lifecycle control path",
        create=True,
    )
    validate_directory_chain(
        current.repository_root,
        control,
        "Symile lifecycle control path",
    )
    _validate_active_initialization(control, execution_id)
    execution_root = control / "executions"
    destination = execution_root / execution_id
    encoded = canonical_json_bytes(document)
    path = destination / "manifest.json"
    if publish_bytes_no_replace(path, encoded) is False and path.read_bytes() != encoded:
        raise ManifestBuildError("Existing Symile execution differs from formal preflight")
    execution = validate_symile_execution(destination, expected_execution_id=execution_id)
    _publish_or_validate_active_execution(control, execution_id)
    return execution


def symile_execution_document(
    *,
    bundle_id: str,
    cv_assignment_id: str,
    configs: Sequence[Mapping[str, object]],
    git_commit: str,
    dependency_lock_sha256: str,
    framework_versions: Mapping[str, str],
    neural_runtime: Mapping[str, object],
    training_loader: Mapping[str, object],
    inference_loader: Mapping[str, object],
    pretrained_weight: Mapping[str, object],
) -> dict[str, object]:
    """Construct and validate the exact preflight-owned execution document."""
    semantic: dict[str, object] = {
        "study": "symile",
        "dataset": {"bundle_id": bundle_id, "cv_assignment_id": cv_assignment_id},
        "formal_run_plan": [dict(item) for item in configs],
        "git_commit": git_commit,
        "dependency_lock_sha256": dependency_lock_sha256,
        "framework_versions": dict(framework_versions),
        "neural_runtime": dict(neural_runtime),
        "training_loader": dict(training_loader),
        "inference_loader": dict(inference_loader),
        "thread_policy": THREAD_POLICY,
        "pretrained_weight": _execution_pretrained_weight(pretrained_weight),
    }
    execution_id = SYMILE_EXECUTION_PREFIX + semantic_sha256(
        "symile-formal-execution", SYMILE_EXECUTION_SCHEMA_VERSION, semantic
    )
    document = {
        "schema_version": SYMILE_EXECUTION_SCHEMA_VERSION,
        "execution_id": execution_id,
        **semantic,
    }
    _validate_execution_fields(document)
    return document


def validate_symile_execution(
    directory: str | Path, *, expected_execution_id: str | None = None
) -> ValidatedSymileExecution:
    """Validate a stored execution without requiring its historical hardware."""
    root = Path(directory)
    path = root / "manifest.json"
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Symile execution record is unreadable") from exc
    fields = {
        "schema_version",
        "execution_id",
        "study",
        "dataset",
        "formal_run_plan",
        "git_commit",
        "dependency_lock_sha256",
        "framework_versions",
        "neural_runtime",
        "training_loader",
        "inference_loader",
        "thread_policy",
        "pretrained_weight",
    }
    if (
        root.is_symlink()
        or not root.is_dir()
        or path.is_symlink()
        or not path.is_file()
        or not isinstance(document, dict)
        or set(document) != fields
        or type(document["schema_version"]) is not int
        or document["schema_version"] != SYMILE_EXECUTION_SCHEMA_VERSION
        or raw != canonical_json_bytes(document)
    ):
        raise ManifestBuildError("Symile execution record contract is invalid")
    _validate_execution_fields(document)
    semantic = {
        key: document[key] for key in fields if key not in {"schema_version", "execution_id"}
    }
    expected = SYMILE_EXECUTION_PREFIX + semantic_sha256(
        "symile-formal-execution", SYMILE_EXECUTION_SCHEMA_VERSION, semantic
    )
    if (
        document["execution_id"] != expected
        or root.name != expected
        or (expected_execution_id is not None and expected != expected_execution_id)
    ):
        raise ManifestBuildError("Symile execution identity is invalid")
    _validate_execution_directory_membership(root)
    return ValidatedSymileExecution(expected, root, document, _EXECUTION_GUARD)


def require_symile_execution(value: object) -> ValidatedSymileExecution:
    """Reauthenticate an execution capability against its stored authority."""
    if not isinstance(value, ValidatedSymileExecution) or value._guard is not _EXECUTION_GUARD:
        raise PermissionError("Symile lifecycle requires a validated formal execution")
    observed = validate_symile_execution(value.directory, expected_execution_id=value.execution_id)
    if observed != value:
        raise ManifestBuildError("Symile execution changed after validation")
    return observed


def validate_current_symile_execution(
    execution: ValidatedSymileExecution,
    preflight: ValidatedSymileFormalPreflight,
) -> ValidatedCurrentSymileExecution:
    """Independently reconstruct and match current formal preflight state."""
    value = require_symile_execution(execution)
    current = require_symile_formal_preflight(preflight)
    if dict(value.manifest) != dict(current.document):
        raise ManifestBuildError("Current environment differs from the formal Symile execution")
    root = current.control_root
    _validate_active_execution(root, value.execution_id)
    return ValidatedCurrentSymileExecution(value, current.repository_root, _CURRENT_EXECUTION_GUARD)


def require_current_symile_execution(value: object) -> ValidatedCurrentSymileExecution:
    """Reauthenticate a current-execution capability and its active receipt."""
    if (
        not isinstance(value, ValidatedCurrentSymileExecution)
        or value._guard is not _CURRENT_EXECUTION_GUARD
    ):
        raise PermissionError("Symile lifecycle requires a current formal execution")
    preflight = build_symile_formal_preflight(
        value.repository_root, **value.execution.manifest["dataset"]
    )
    observed = validate_current_symile_execution(value.execution, preflight)
    if observed != value:
        raise ManifestBuildError("Current Symile execution changed after validation")
    return observed


def validate_active_symile_execution(control_root: str | Path) -> ValidatedSymileExecution:
    """Resolve the sealed campaign execution only through its explicit active receipt."""
    root = Path(control_root)
    path = root / "active-execution.json"
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Symile active execution receipt is unavailable") from exc
    if (
        path.is_symlink()
        or not isinstance(document, dict)
        or set(document) != {"schema_version", "execution_id"}
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or not _identity(document["execution_id"], SYMILE_EXECUTION_PREFIX)
        or raw != canonical_json_bytes(document)
    ):
        raise ManifestBuildError("Symile active execution receipt is invalid")
    return validate_symile_execution(
        root / "executions" / str(document["execution_id"]),
        expected_execution_id=str(document["execution_id"]),
    )


def validate_symile_lifecycle(
    current_execution: ValidatedCurrentSymileExecution,
    *,
    repository_root: str | Path,
    private_root: str | Path,
    backup_root: str | Path,
) -> ValidatedSymileLifecycle:
    """Validate the one canonical local control root and persistent receipt root."""
    current = require_current_symile_execution(current_execution)
    repository_path = Path(repository_root).absolute()
    private_path = Path(private_root).absolute()
    control_path = private_path / "control" / "symile"
    backup = Path(backup_root).absolute()
    if any(path.is_symlink() for path in (repository_path, *repository_path.parents)):
        raise ManifestBuildError("Symile repository path contains a symlink")
    if private_path != repository_path / "private":
        raise ManifestBuildError("Symile lifecycle root is not the canonical campaign root")
    validate_directory_chain(
        repository_path,
        control_path,
        "Symile lifecycle control path",
    )
    repository = repository_path.resolve()
    private = private_path.resolve()
    control = control_path.resolve()
    if current.control_root.absolute() != control_path or current.control_root.resolve() != control:
        raise ManifestBuildError("Symile lifecycle root is not the canonical campaign root")
    if any(path.is_symlink() for path in (backup, *backup.parents)):
        raise ManifestBuildError("Symile lifecycle backup path contains a symlink")
    validate_directory_chain(backup, backup, "Symile lifecycle backup path")
    resolved_backup = backup.resolve()
    if (
        resolved_backup.is_relative_to(repository)
        or resolved_backup.is_relative_to(control.resolve())
        or control.resolve().is_relative_to(resolved_backup)
        or not resolved_backup.is_dir()
        or not os.access(resolved_backup, os.W_OK | os.X_OK)
    ):
        raise ManifestBuildError("Symile lifecycle backup destination is invalid")
    return ValidatedSymileLifecycle(
        current,
        repository,
        private,
        control,
        resolved_backup,
        _LIFECYCLE_GUARD,
    )


def prepare_symile_test_open_hierarchy(
    lifecycle: ValidatedSymileLifecycle,
) -> tuple[Path, Path]:
    """Create and revalidate the canonical local/external opening parent chains."""
    roots = require_symile_lifecycle(lifecycle)
    validate_directory_chain(
        roots.repository_root,
        roots.control_root,
        "Symile lifecycle control path",
    )
    external_parent = roots.backup_root / "control" / "symile"
    validate_directory_chain(
        roots.backup_root,
        external_parent,
        "Symile lifecycle backup control path",
        create=True,
    )
    validate_directory_chain(
        roots.backup_root,
        external_parent,
        "Symile lifecycle backup control path",
    )
    return roots.control_root / "test-open.json", external_parent / "test-open.json"


def require_symile_lifecycle(value: object) -> ValidatedSymileLifecycle:
    if not isinstance(value, ValidatedSymileLifecycle) or value._guard is not _LIFECYCLE_GUARD:
        raise PermissionError("Symile test opening requires validated lifecycle roots")
    current = require_current_symile_execution(value.current_execution)
    observed = validate_symile_lifecycle(
        current,
        repository_root=value.repository_root,
        private_root=value.private_root,
        backup_root=value.backup_root,
    )
    if observed != value:
        raise ManifestBuildError("Symile lifecycle roots changed after validation")
    return value


@contextmanager
def symile_prospective_mutation(
    lifecycle: ValidatedSymileLifecycle,
) -> Iterator[ValidatedSymileProspectiveMutation]:
    """Hold the canonical campaign lock while the campaign is proven unopened."""
    roots = require_symile_lifecycle(lifecycle)
    with campaign_lock(
        roots.control_root,
        "Symile",
        trusted_root=roots.repository_root,
    ):
        local, external = prepare_symile_test_open_hierarchy(roots)
        if symile_test_is_open(local, external):
            raise ManifestBuildError(
                "Official Symile test data are already open; prospective scientific mutation "
                "is prohibited"
            )
        authority = object.__new__(ValidatedSymileProspectiveMutation)
        object.__setattr__(authority, "lifecycle", roots)
        _ACTIVE_PROSPECTIVE_MUTATIONS[id(authority)] = authority
        try:
            yield authority
        finally:
            _ACTIVE_PROSPECTIVE_MUTATIONS.pop(id(authority), None)


def symile_test_is_open(local: str | Path, external: str | Path) -> bool:
    """Return whether either durable opening record exists at its canonical coordinate."""
    return any(path.exists() or path.is_symlink() for path in (Path(local), Path(external)))


def require_symile_prospective_mutation(
    value: object,
) -> ValidatedSymileProspectiveMutation:
    """Require a live unopened authority held under the canonical campaign lock."""
    if not isinstance(value, ValidatedSymileProspectiveMutation) or (
        _ACTIVE_PROSPECTIVE_MUTATIONS.get(id(value)) is not value
    ):
        raise PermissionError("Prospective Symile science requires live unopened authority")
    roots = require_symile_lifecycle(value.lifecycle)
    local, external = prepare_symile_test_open_hierarchy(roots)
    if symile_test_is_open(local, external):
        raise PermissionError("Prospective Symile science requires live unopened authority")
    return value


def require_symile_prospective_path(
    value: object,
    path: str | Path,
    context: str,
    *,
    relative: str | Path | None = None,
) -> Path:
    """Bind a prospective mutation destination to its authorized repository."""
    authority = require_symile_prospective_mutation(value)
    candidate = Path(path).absolute().resolve()
    repository = authority.lifecycle.repository_root
    expected = repository / Path(relative) if relative is not None else None
    if (expected is not None and candidate != expected) or not candidate.is_relative_to(repository):
        raise ManifestBuildError(f"{context} is outside the authorized Symile campaign")
    return candidate


def _publish_or_validate_active_execution(control_root: Path, execution_id: str) -> None:
    path = control_root / "active-execution.json"
    expected = {"schema_version": 1, "execution_id": execution_id}
    encoded = canonical_json_bytes(expected)
    if publish_bytes_no_replace(path, encoded) is False and path.read_bytes() != encoded:
        raise ManifestBuildError("Symile campaign is active under another execution")
    _validate_active_execution(control_root, execution_id)


def _validate_active_initialization(control_root: Path, execution_id: str) -> None:
    path = control_root / "active-execution.json"
    if path.exists() or path.is_symlink():
        active = validate_active_symile_execution(control_root)
        if active.execution_id != execution_id:
            raise ManifestBuildError("Symile campaign is active under another execution")
    elif _protected_state_exists(control_root):
        raise ManifestBuildError("Symile protected state lacks an active execution receipt")


def _validate_active_execution(control_root: Path, execution_id: str) -> None:
    path = control_root / "active-execution.json"
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Symile active execution receipt is unavailable") from exc
    expected = {"schema_version": 1, "execution_id": execution_id}
    if path.is_symlink() or document != expected or raw != canonical_json_bytes(expected):
        raise ManifestBuildError("Symile active execution receipt is invalid")


def _protected_state_exists(control_root: Path) -> bool:
    local = any(
        path.exists() or path.is_symlink()
        for path in (control_root / "freezes", control_root / "test-open.json")
    )
    repository = control_root.parents[2] if len(control_root.parents) > 2 else None
    return local or (
        repository is not None
        and any(
            path.exists() or path.is_symlink()
            for path in (
                repository / "reports/symile/development",
                repository / "reports/symile/development-decisions",
                repository / "models/symile/final",
                repository / "private/predictions/symile",
            )
        )
    )


def _validate_execution_directory_membership(root: Path) -> None:
    entries = tuple(root.iterdir())
    if {entry.name for entry in entries} != {"manifest.json"} or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise ManifestBuildError("Symile execution directory membership is invalid")


def _validate_execution_fields(document: Mapping[str, Any]) -> None:
    dataset = document["dataset"]
    plan = document["formal_run_plan"]
    runtime = document["neural_runtime"]
    training = document["training_loader"]
    inference = document["inference_loader"]
    weight = document["pretrained_weight"]
    expected_plan = [
        (spec.family_id, spec.config_relative.as_posix()) for spec in SYMILE_FORMAL_RUN_PLAN
    ]
    if (
        document["study"] != "symile"
        or not isinstance(dataset, dict)
        or set(dataset) != {"bundle_id", "cv_assignment_id"}
        or not _identity(dataset["bundle_id"], "bundle-")
        or not _identity(dataset["cv_assignment_id"], "cv-assignment-")
        or not isinstance(plan, list)
        or len(plan) != len(expected_plan)
        or any(not _valid_run(item) for item in plan)
        or [(item["family_id"], item["config_path"]) for item in plan] != expected_plan
        or not _git_commit(document["git_commit"])
        or not _sha256(document["dependency_lock_sha256"])
        or not isinstance(document["framework_versions"], dict)
        or set(document["framework_versions"])
        != {"python", "torch", "torchvision", "torchxrayvision"}
        or any(
            not isinstance(key, str) or not isinstance(value, str) or not value
            for key, value in document["framework_versions"].items()
        )
        or not _valid_cuda_runtime(runtime)
        or training
        != {
            "lifecycle": "reused",
            "num_workers": 2,
            "pin_memory": True,
            "prefetch_factor": 2,
        }
        or inference
        != {
            "lifecycle": "one_shot",
            "num_workers": 0,
            "pin_memory": True,
            "prefetch_factor": None,
        }
        or document["thread_policy"] != THREAD_POLICY
        or not isinstance(weight, dict)
        or set(weight) != {"declared_name", "stable_identifier", "byte_size", "sha256"}
        or any(
            not isinstance(weight[field], str) or not weight[field]
            for field in ("declared_name", "stable_identifier")
        )
        or type(weight["byte_size"]) is not int
        or weight["byte_size"] <= 0
        or not _sha256(weight["sha256"])
    ):
        raise ManifestBuildError("Symile execution fields are invalid")


def _valid_run(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "family_id",
        "config_path",
        "normalized_config",
    }:
        return False
    path = value["config_path"]
    try:
        parsed = PurePosixPath(path) if isinstance(path, str) else None
    except ValueError:
        return False
    return (
        isinstance(value["family_id"], str)
        and bool(value["family_id"])
        and isinstance(value["normalized_config"], dict)
        and parsed is not None
        and not parsed.is_absolute()
        and parsed.as_posix() == path
        and all(part not in {"", ".", ".."} for part in parsed.parts)
    )


def _valid_cuda_runtime(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    required = {
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
    capability = value.get("gpu_compute_capability")
    return (
        set(value) == required
        and value["device_type"] == "cuda"
        and value["autocast_dtype"] == "float16"
        and value["deterministic_algorithms"] == "enabled_warn_only"
        and value["cudnn_deterministic"] is True
        and value["cudnn_benchmark"] is False
        and isinstance(value["cuda_runtime_version"], str)
        and bool(value["cuda_runtime_version"])
        and type(value["cudnn_version"]) is int
        and isinstance(value["gpu_device_name"], str)
        and bool(value["gpu_device_name"])
        and isinstance(capability, list)
        and len(capability) == 2
        and all(type(item) is int and item >= 0 for item in capability)
    )


def _identity(value: object, prefix: str) -> bool:
    return isinstance(value, str) and value.startswith(prefix) and _sha256(value[len(prefix) :])


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _execution_pretrained_weight(value: Mapping[str, object]) -> dict[str, object]:
    required = {"declared_name", "stable_identifier", "cache_filename", "byte_size", "sha256"}
    if set(value) != required:
        raise ManifestBuildError("Symile pretrained materialization evidence is invalid")
    return {
        "declared_name": value["declared_name"],
        "stable_identifier": value["stable_identifier"],
        "byte_size": value["byte_size"],
        "sha256": value["sha256"],
    }
