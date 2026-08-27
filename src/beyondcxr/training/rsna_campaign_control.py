"""Append-only control state for one exact RSNA formal execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.config import require_runtime_seed
from beyondcxr.training.device import (
    full_precision_neural_runtime_policy,
    neural_inference_runtime_policy,
)
from beyondcxr.training.rsna_evaluation_result import (
    CompletedRsnaEvaluation,
    validate_rsna_evaluation,
    validate_rsna_model_package,
)
from beyondcxr.training.rsna_formal import (
    RSNA_FORMAL_RUN_PLAN,
    ValidatedRsnaPlan,
    require_validated_rsna_plan,
    rsna_run_report_root,
)
from beyondcxr.training.rsna_training_report import training_report_sha256, validate_training_report
from beyondcxr.training.rsna_validation_evidence import load_canonical_validation_cohort
from beyondcxr.utils.publication import publish_bytes_no_replace, validate_path_component

RSNA_EXECUTION_SCHEMA_VERSION = 1
RSNA_PACKAGE_FREEZE_SCHEMA_VERSION = 1
RSNA_EVALUATION_RECORD_SCHEMA_VERSION = 1
EXECUTION_PREFIX = "rsna-execution-"
_EXECUTION_GUARD = object()
_PACKAGE_FREEZE_GUARD = object()


@dataclass(frozen=True)
class ValidatedRsnaExecution:
    execution_id: str
    directory: Path
    manifest: Mapping[str, Any]
    _guard: object


@dataclass(frozen=True)
class FrozenTrainingPackage:
    run_id: str
    model_package_id: str
    model_path: Path
    artifact_directory: Path
    report_sha256: str


@dataclass(frozen=True)
class ValidatedRsnaPackageFreeze:
    """The complete ordered package/report barrier authorizing held-out access."""

    execution: ValidatedRsnaExecution
    packages: tuple[FrozenTrainingPackage, ...]
    model_root: Path
    report_root: Path
    bundle_directory: Path
    _guard: object

    def require(self, package_id: str) -> FrozenTrainingPackage:
        require_validated_package_freeze(self)
        matches = tuple(item for item in self.packages if item.model_package_id == package_id)
        if len(matches) != 1:
            raise ManifestBuildError("RSNA package is not authorized by the complete freeze")
        return matches[0]

    def __iter__(self):
        return iter(self.packages)


def require_validated_package_freeze(value: object) -> ValidatedRsnaPackageFreeze:
    """Reauthenticate one held-out capability against its canonical authorities."""
    if (
        not isinstance(value, ValidatedRsnaPackageFreeze)
        or value._guard is not _PACKAGE_FREEZE_GUARD
    ):
        raise PermissionError("RSNA held-out access requires a validated package freeze")
    canonical = validate_package_freeze(
        value.execution,
        model_root=value.model_root,
        report_root=value.report_root,
        bundle_directory=value.bundle_directory,
    )
    if canonical != value:
        raise ManifestBuildError(
            "RSNA package freeze capability differs from its canonical authority"
        )
    return value


def require_validated_execution(value: object) -> ValidatedRsnaExecution:
    """Reject execution objects that did not originate from canonical validation."""
    if not isinstance(value, ValidatedRsnaExecution) or value._guard is not _EXECUTION_GUARD:
        raise PermissionError("RSNA control access requires a validated execution")
    return value


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
            or config.dataset.bundle_manifest_sha256 != plan.authority.bundle_manifest_sha256
            or config.dataset.split_assignment_id != plan.authority.split_assignment_id
            or config.task.task_id != plan.authority.task_id
            or config.task.label_policy_version != plan.authority.label_policy_version
        ):
            raise ManifestBuildError("RSNA formal plan config agreement is invalid")
    document = _execution_document(plan)
    execution_id = str(document["execution_id"])
    directory = plan.roots.control_root / "executions" / execution_id
    path = directory / "manifest.json"
    encoded = _canonical_bytes(document)
    if publish_bytes_no_replace(path, encoded) is False and path.read_bytes() != encoded:
        raise ManifestBuildError("Existing RSNA execution control differs from current preflight")
    return validate_execution(directory, expected_execution_id=execution_id)


def _execution_document(plan: ValidatedRsnaPlan) -> dict[str, Any]:
    semantic = {
        "dataset": {
            "dataset_id": plan.authority.dataset_id,
            "bundle_id": plan.authority.bundle_id,
            "bundle_manifest_sha256": plan.authority.bundle_manifest_sha256,
            "split_assignment_id": plan.authority.split_assignment_id,
        },
        "task": {
            "task_id": plan.authority.task_id,
            "label_policy_version": plan.authority.label_policy_version,
        },
        "configs": [
            {
                "family_id": config.family.family_id,
                "seed": require_runtime_seed(config),
                "config_source_sha256": config.config_source_sha256,
                "config_semantic_sha256": config.config_semantic_sha256,
            }
            for config in plan.configs
        ],
        "pretrained_weight": plan.pretrained_weight.as_dict(),
        "training_evaluation_runtime": neural_inference_runtime_policy(plan.runtime),
        "localization_runtime": full_precision_neural_runtime_policy(plan.runtime),
        "git_commit": plan.git_commit,
        "dependency_lock_sha256": plan.dependency_lock_sha256,
    }
    execution_id = EXECUTION_PREFIX + hashlib.sha256(_canonical_bytes(semantic)).hexdigest()
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
        "pretrained_weight",
        "training_evaluation_runtime",
        "localization_runtime",
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
        or raw != _canonical_bytes(document)
    ):
        raise ManifestBuildError("RSNA execution control contract is invalid")
    execution_id = document["execution_id"]
    semantic = {
        key: document[key]
        for key in fields
        if key not in {"execution_id", "rsna_execution_schema_version"}
    }
    expected = EXECUTION_PREFIX + hashlib.sha256(_canonical_bytes(semantic)).hexdigest()
    if (
        execution_id != expected
        or root.name != expected
        or (expected_execution_id is not None and execution_id != expected_execution_id)
    ):
        raise ManifestBuildError("RSNA execution control identity is invalid")
    _validate_plan_fields(document)
    return ValidatedRsnaExecution(execution_id, root, document, _EXECUTION_GUARD)


def publish_or_validate_package_freeze(
    *,
    plan: ValidatedRsnaPlan,
    execution: ValidatedRsnaExecution,
    results: Sequence[object],
) -> ValidatedRsnaPackageFreeze:
    """Freeze the exact complete package set before any held-out evaluation."""
    require_validated_rsna_plan(plan)
    _require_execution_for_plan(plan, execution)
    values = tuple(results)
    if len(values) != 8:
        raise ManifestBuildError("RSNA package freeze requires eight packages")
    models = plan.roots.model_root.resolve()
    reports = plan.roots.report_root.resolve()
    cohort = load_canonical_validation_cohort(
        plan.authority.bundle_directory,
        expected_bundle_id=plan.authority.bundle_id,
        expected_manifest_sha256=plan.authority.bundle_manifest_sha256,
        expected_split_assignment_id=plan.authority.split_assignment_id,
        expected_task_id=plan.authority.task_id,
        expected_label_policy_version=plan.authority.label_policy_version,
    )
    members = []
    for result, config in zip(values, plan.configs, strict=True):
        package_id = str(result.model_package_id)
        package = validate_rsna_model_package(models, package_id)
        model_path = _validated_output_path(Path(result.model_path), models, "model")
        artifact_directory = _validated_output_path(
            Path(result.artifact_directory), reports, "report"
        )
        if artifact_directory.parent != rsna_run_report_root(
            reports
        ) or artifact_directory.name != str(result.run_id):
            raise ManifestBuildError("RSNA run report directory does not match its run ID")
        expected_lineage = {
            "bundle_id": plan.authority.bundle_id,
            "bundle_manifest_sha256": plan.authority.bundle_manifest_sha256,
            "split_assignment_id": plan.authority.split_assignment_id,
            "task_id": plan.authority.task_id,
            "label_policy_version": plan.authority.label_policy_version,
            "family_id": config.family.family_id,
            "config_source_sha256": config.config_source_sha256,
            "config_semantic_sha256": config.config_semantic_sha256,
        }
        package_seed = package.get("seed")
        if package_seed is None and isinstance(package.get("training_policy"), Mapping):
            package_seed = package["training_policy"].get("seed")
        model_identity = package.get("model_identity")
        weight_matches = config.neural is None or (
            isinstance(model_identity, Mapping)
            and model_identity.get("pretrained_weight") == plan.pretrained_weight.as_dict()
        )
        runtime_matches = config.neural is None or _package_numerical_runtime(package) == (
            neural_inference_runtime_policy(plan.runtime)
        )
        provenance_matches = _package_provenance(package) == {
            "git_commit": execution.manifest["git_commit"],
            "git_dirty": False,
            "dependency_lock_sha256": execution.manifest["dependency_lock_sha256"],
        }
        if (
            any(package.get(key) != value for key, value in expected_lineage.items())
            or package_seed != require_runtime_seed(config)
            or not weight_matches
            or not runtime_matches
            or not provenance_matches
            or model_path.name not in {"model.skops", "model.pt"}
            or not model_path.is_file()
        ):
            raise ManifestBuildError("RSNA package freeze output is invalid")
        package_directory = models / "packages" / package_id
        validate_training_report(
            artifact_directory,
            run_id=str(result.run_id),
            model_package_id=package_id,
            family_id=config.family.family_id,
            seed=require_runtime_seed(config),
            package=package,
            package_directory=package_directory,
            canonical_validation_cohort=cohort,
        )
        members.append(
            {
                "package_id": package_id,
                "run_id": str(result.run_id),
                "model_relative": model_path.relative_to(models).as_posix(),
                "report_relative": artifact_directory.relative_to(reports).as_posix(),
                "report_sha256": training_report_sha256(artifact_directory),
            }
        )
    if len({item["package_id"] for item in members}) != 8:
        raise ManifestBuildError("RSNA package freeze contains duplicate packages")
    document = {
        "rsna_package_freeze_schema_version": RSNA_PACKAGE_FREEZE_SCHEMA_VERSION,
        "execution_id": execution.execution_id,
        "packages": members,
    }
    path = execution.directory / "package-freeze.json"
    encoded = _canonical_bytes(document)
    if publish_bytes_no_replace(path, encoded) is False and path.read_bytes() != encoded:
        raise ManifestBuildError("Existing RSNA package freeze differs from completed training")
    return validate_package_freeze(
        execution,
        model_root=models,
        report_root=reports,
        bundle_directory=plan.authority.bundle_directory,
    )


def validate_package_freeze(
    execution: ValidatedRsnaExecution,
    *,
    model_root: str | Path,
    report_root: str | Path,
    bundle_directory: str | Path,
) -> ValidatedRsnaPackageFreeze:
    execution = _reauthenticate_execution(execution)
    document = _load_package_freeze_document(execution)
    models = Path(model_root).resolve()
    reports = Path(report_root).resolve()
    bundle = Path(bundle_directory).resolve()
    dataset = execution.manifest["dataset"]
    task = execution.manifest["task"]
    assert isinstance(dataset, dict) and isinstance(task, dict)
    cohort = load_canonical_validation_cohort(
        bundle,
        expected_bundle_id=str(dataset["bundle_id"]),
        expected_manifest_sha256=str(dataset["bundle_manifest_sha256"]),
        expected_split_assignment_id=str(dataset["split_assignment_id"]),
        expected_task_id=str(task["task_id"]),
        expected_label_policy_version=str(task["label_policy_version"]),
    )
    results = []
    plan = execution.manifest["configs"]
    assert isinstance(plan, list)
    for item, configured in zip(document["packages"], plan, strict=True):
        if not isinstance(item, dict) or set(item) != {
            "package_id",
            "run_id",
            "model_relative",
            "report_relative",
            "report_sha256",
        }:
            raise ManifestBuildError("RSNA package freeze member is invalid")
        package_id = validate_path_component(item["package_id"], "RSNA package ID")
        package = validate_rsna_model_package(models, package_id)
        model_path = _controlled_relative(models, item["model_relative"])
        report = _controlled_relative(reports, item["report_relative"])
        package_seed = package.get("seed")
        if package_seed is None and isinstance(package.get("training_policy"), Mapping):
            package_seed = package["training_policy"].get("seed")
        expected_lineage = {
            "bundle_id": dataset["bundle_id"],
            "bundle_manifest_sha256": dataset["bundle_manifest_sha256"],
            "split_assignment_id": dataset["split_assignment_id"],
            "task_id": task["task_id"],
            "label_policy_version": task["label_policy_version"],
            "family_id": configured["family_id"],
            "config_source_sha256": configured["config_source_sha256"],
            "config_semantic_sha256": configured["config_semantic_sha256"],
        }
        package_directory = models / "packages" / package_id
        model_identity = package.get("model_identity")
        weight_matches = configured["family_id"] in {
            "metadata_logistic",
            "metadata_lightgbm",
        } or (
            isinstance(model_identity, Mapping)
            and model_identity.get("pretrained_weight") == execution.manifest["pretrained_weight"]
        )
        runtime_matches = (
            configured["family_id"]
            in {
                "metadata_logistic",
                "metadata_lightgbm",
            }
            or _package_numerical_runtime(package)
            == execution.manifest["training_evaluation_runtime"]
        )
        provenance_matches = _package_provenance(package) == {
            "git_commit": execution.manifest["git_commit"],
            "git_dirty": False,
            "dependency_lock_sha256": execution.manifest["dependency_lock_sha256"],
        }
        if (
            any(package.get(key) != value for key, value in expected_lineage.items())
            or package_seed != configured["seed"]
            or not weight_matches
            or not runtime_matches
            or not provenance_matches
            or model_path.parent.resolve() != package_directory.resolve()
            or model_path.name not in {"model.skops", "model.pt"}
            or not model_path.is_file()
            or report.parent != rsna_run_report_root(reports)
            or report.name != item["run_id"]
            or report.is_symlink()
            or not report.is_dir()
        ):
            raise ManifestBuildError("RSNA package freeze output is unavailable")
        validate_training_report(
            report,
            run_id=str(item["run_id"]),
            model_package_id=package_id,
            family_id=str(configured["family_id"]),
            seed=int(configured["seed"]),
            package=package,
            package_directory=package_directory,
            canonical_validation_cohort=cohort,
        )
        if training_report_sha256(report) != item["report_sha256"]:
            raise ManifestBuildError("RSNA training report differs from the package freeze")
        if not _sha256(item["report_sha256"]):
            raise ManifestBuildError("RSNA training report witness is invalid")
        results.append(
            FrozenTrainingPackage(
                str(item["run_id"]), package_id, model_path, report, item["report_sha256"]
            )
        )
    if len({result.model_package_id for result in results}) != 8:
        raise ManifestBuildError("RSNA package freeze contains duplicate packages")
    return ValidatedRsnaPackageFreeze(
        execution,
        tuple(results),
        models,
        reports,
        bundle,
        _PACKAGE_FREEZE_GUARD,
    )


def _load_package_freeze_document(execution: ValidatedRsnaExecution) -> dict[str, Any]:
    """Load and validate the canonical schema-1 package-freeze document."""
    path = execution.directory / "package-freeze.json"
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("RSNA package freeze is unreadable") from exc
    if (
        path.is_symlink()
        or not isinstance(document, dict)
        or set(document) != {"rsna_package_freeze_schema_version", "execution_id", "packages"}
        or type(document["rsna_package_freeze_schema_version"]) is not int
        or document["rsna_package_freeze_schema_version"] != RSNA_PACKAGE_FREEZE_SCHEMA_VERSION
        or document["execution_id"] != execution.execution_id
        or raw != _canonical_bytes(document)
        or not isinstance(document["packages"], list)
        or len(document["packages"]) != 8
    ):
        raise ManifestBuildError("RSNA package freeze contract is invalid")
    return document


def load_evaluation_record(
    execution: ValidatedRsnaExecution,
    *,
    package_id: str,
    report_root: str | Path,
    private_root: str | Path,
    model_root: str | Path,
) -> CompletedRsnaEvaluation | None:
    """Load one completed package-bound evaluation, if its record exists."""
    require_validated_execution(execution)
    path = execution.directory / "evaluations" / f"{package_id}.json"
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
        or raw != _canonical_bytes(document)
    ):
        raise ManifestBuildError("RSNA evaluation completion record is invalid")
    evaluation_id = validate_path_component(document["evaluation_id"], "RSNA evaluation ID")
    validated = validate_rsna_evaluation(
        Path(report_root) / "rsna" / "evaluations" / evaluation_id,
        private_root=private_root,
        model_root=model_root,
        expected_evaluation_id=evaluation_id,
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
) -> None:
    require_validated_execution(execution)
    document = {
        "rsna_evaluation_record_schema_version": RSNA_EVALUATION_RECORD_SCHEMA_VERSION,
        "execution_id": execution.execution_id,
        "package_id": result.model_package_id,
        "evaluation_id": result.evaluation_id,
        "prediction_id": result.prediction_id,
        "mlflow_run_id": result.mlflow_run_id,
    }
    path = execution.directory / "evaluations" / f"{result.model_package_id}.json"
    encoded = _canonical_bytes(document)
    if publish_bytes_no_replace(path, encoded) is False and path.read_bytes() != encoded:
        raise ManifestBuildError("Existing RSNA evaluation record differs from current result")


def validate_execution_closure(package_freeze: ValidatedRsnaPackageFreeze) -> None:
    """Require the exact completed control-tree membership for one execution."""
    frozen = require_validated_package_freeze(package_freeze)
    execution = frozen.execution
    expected_root = {"manifest.json", "package-freeze.json", "evaluations"}
    entries = tuple(execution.directory.iterdir())
    if {entry.name for entry in entries} != expected_root or any(
        entry.is_symlink() for entry in entries
    ):
        raise ManifestBuildError("RSNA execution control membership is invalid")
    evaluations = execution.directory / "evaluations"
    if not evaluations.is_dir():
        raise ManifestBuildError("RSNA execution records directory is invalid")
    expected_records = {f"{item.model_package_id}.json" for item in frozen.packages}
    records = tuple(evaluations.iterdir())
    if {record.name for record in records} != expected_records or any(
        record.is_symlink() or not record.is_file() for record in records
    ):
        raise ManifestBuildError("RSNA evaluation record membership is invalid")


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
    if (
        not isinstance(dataset, dict)
        or set(dataset)
        != {"dataset_id", "bundle_id", "bundle_manifest_sha256", "split_assignment_id"}
        or not isinstance(task, dict)
        or set(task) != {"task_id", "label_policy_version"}
        or not isinstance(configs, list)
        or len(configs) != 8
        or any(
            not isinstance(item, dict)
            or set(item) != {"family_id", "seed", "config_source_sha256", "config_semantic_sha256"}
            for item in configs
        )
        or not isinstance(weight, dict)
        or set(weight)
        != {"declared_name", "stable_identifier", "cache_filename", "byte_size", "sha256"}
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
    ):
        raise ManifestBuildError("RSNA execution control fields are invalid")
    assert isinstance(dataset, dict) and isinstance(task, dict) and isinstance(configs, list)
    if (
        dataset["dataset_id"] != "rsna"
        or not _identity(dataset["bundle_id"], "bundle-")
        or not _identity(dataset["split_assignment_id"], "split-assignment-")
        or not _sha256(dataset["bundle_manifest_sha256"])
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
        or not isinstance(weight["cache_filename"], str)
        or not weight["cache_filename"]
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
        or not _sha256(item["config_source_sha256"])
        or not _sha256(item["config_semantic_sha256"])
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


def _package_numerical_runtime(package: Mapping[str, Any]) -> dict[str, Any] | None:
    runtime = package.get("runtime_provenance")
    if not isinstance(runtime, Mapping):
        return None
    capability = runtime.get("gpu_compute_capability")
    return {
        "device_type": runtime.get("resolved_device"),
        "autocast_dtype": "float16" if runtime.get("mixed_precision_effective") is True else None,
        "deterministic_algorithms": "enabled_warn_only",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_runtime_version": runtime.get("cuda_runtime_version"),
        "cudnn_version": runtime.get("cudnn_version"),
        "gpu_device_name": runtime.get("gpu_device_name"),
        "gpu_compute_capability": capability,
    }


def _package_provenance(package: Mapping[str, Any]) -> dict[str, Any] | None:
    if package.get("family_id") in {"metadata_logistic", "metadata_lightgbm"}:
        source: Mapping[str, Any] = package
    else:
        value = package.get("source_provenance")
        if not isinstance(value, Mapping):
            return None
        source = value
    return {
        "git_commit": source.get("git_commit"),
        "git_dirty": source.get("git_dirty"),
        "dependency_lock_sha256": source.get("dependency_lock_sha256"),
    }


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _controlled_relative(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ManifestBuildError("RSNA control path is invalid")
    posix_path = PurePosixPath(value)
    if (
        posix_path.as_posix() != value
        or posix_path.is_absolute()
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise ManifestBuildError("RSNA control path is invalid")
    path = Path(*posix_path.parts)
    candidate = root / path
    path_components = (
        root,
        *(root / Path(*path.parts[:index]) for index in range(1, len(path.parts) + 1)),
    )
    if candidate.resolve().is_relative_to(root.resolve()) is False or any(
        current.is_symlink() for current in path_components
    ):
        raise ManifestBuildError("RSNA control path escapes its root")
    return candidate


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


def _canonical_bytes(value: object) -> bytes:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (serialized + "\n").encode()
