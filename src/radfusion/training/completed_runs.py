"""Interpret completed MLflow training and evaluation records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

PROBABILITY_METRIC_NAMES = (
    "average_precision",
    "roc_auc",
    "brier_score",
    "expected_calibration_error",
    "calibration_slope",
    "calibration_intercept",
)
OPERATING_POINT_NAMES = ("youden_j", "target_sensitivity")
OPERATING_PERFORMANCE_NAMES = ("precision", "recall", "specificity", "f1")
CONFUSION_COUNT_NAMES = (
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
)
THRESHOLD_METRIC_NAMES = tuple(f"{policy}_threshold" for policy in OPERATING_POINT_NAMES)
OPERATING_METRIC_NAMES = tuple(
    f"{policy}_{metric}"
    for policy in OPERATING_POINT_NAMES
    for metric in (*OPERATING_PERFORMANCE_NAMES, *CONFUSION_COUNT_NAMES)
)
SCOPED_METRIC_NAMES = (
    *PROBABILITY_METRIC_NAMES,
    *THRESHOLD_METRIC_NAMES,
    *OPERATING_METRIC_NAMES,
    "latency_ms",
)
COMPARISON_SCOPED_METRIC_NAMES = (
    *PROBABILITY_METRIC_NAMES,
    *(
        f"{policy}_{metric}"
        for policy in OPERATING_POINT_NAMES
        for metric in ("threshold", *OPERATING_PERFORMANCE_NAMES)
    ),
    "latency_ms",
)
SEED_SPECIFIC_METRIC_NAMES = (
    *PROBABILITY_METRIC_NAMES,
    *THRESHOLD_METRIC_NAMES,
    *OPERATING_METRIC_NAMES,
    "model_size_mib",
)
SEED_AGGREGATE_METRIC_NAMES = tuple(
    name for name in SEED_SPECIFIC_METRIC_NAMES if name not in THRESHOLD_METRIC_NAMES
)

_COMMON_REQUIRED_TAGS = (
    "experiment_name",
    "model",
    "modality",
    "model_package_id",
    "dataset_bundle_id",
    "split_assignment_id",
    "seed",
    "evaluation_scope",
    "run_kind",
    "task",
)
_BOUNDED_METRICS = frozenset(
    {
        "average_precision",
        "roc_auc",
        "brier_score",
        "expected_calibration_error",
        *THRESHOLD_METRIC_NAMES,
        *(
            f"{policy}_{metric}"
            for policy in OPERATING_POINT_NAMES
            for metric in OPERATING_PERFORMANCE_NAMES
        ),
    }
)
_COUNT_METRICS = frozenset(
    f"{policy}_{metric}" for policy in OPERATING_POINT_NAMES for metric in CONFUSION_COUNT_NAMES
)


@dataclass(frozen=True)
class CompletedRunRecord:
    """Normalized completed-run fields shared by report consumers."""

    run_id: str
    status: str
    run_kind: str
    evaluation_scope: str
    experiment_name: str
    dataset: str
    task: str
    label_policy_version: str
    model: str
    modality: str
    seed: str
    bundle_id: str
    bundle_manifest_sha256: str
    split_assignment_id: str
    model_package_id: str
    source_training_run_id: str
    git_commit: str
    git_dirty: str
    dependency_lock_sha256: str
    config_source_sha256: str
    config_semantic_sha256: str
    local_model_path: str
    local_model_sha256: str
    checkpoint_sha256: str
    source_training_run_parameter: str
    source_cxr_training_run_id: str
    source_cxr_model_package_id: str
    source_cxr_checkpoint_sha256: str
    metrics: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def integer_seed(self) -> int:
        """Return a canonical nonnegative integer seed."""
        if not self.seed.isascii() or not self.seed.isdecimal():
            raise ValueError(f"Run {self.run_id} has an invalid seed tag")
        value = int(self.seed)
        if str(value) != self.seed:
            raise ValueError(f"Run {self.run_id} has a noncanonical seed tag")
        return value


def completed_run_record(run) -> CompletedRunRecord | None:
    """Return a normalized record or ``None`` when a run is ineligible."""
    try:
        return require_completed_run(run)
    except ValueError:
        return None


def require_completed_run(run) -> CompletedRunRecord:
    """Return one normalized record after strict lifecycle validation."""
    if run.info.status != "FINISHED":
        raise ValueError(f"Run {run.info.run_id} is not FINISHED")
    tags = run.data.tags
    if tags.get("run_complete") != "true":
        raise ValueError(f"Run {run.info.run_id} is not complete")
    missing = [
        key
        for key in _COMMON_REQUIRED_TAGS
        if not isinstance(tags.get(key), str) or not tags[key].strip()
    ]
    if missing:
        raise ValueError(f"Run {run.info.run_id} is missing required tags: {missing}")
    scope = tags["evaluation_scope"]
    kind = tags["run_kind"]
    if (kind, scope) not in {
        ("training", "validation"),
        ("test_evaluation", "test"),
    }:
        raise ValueError(f"Run {run.info.run_id} has an invalid lifecycle kind and scope")
    parent = tags.get("source_training_run_id", "")
    if scope == "test" and (not isinstance(parent, str) or not parent.strip()):
        raise ValueError(f"Run {run.info.run_id} has no source training run")
    modality = tags["modality"]
    if modality not in {"metadata", "image", "fusion"}:
        raise ValueError(f"Run {run.info.run_id} has an invalid modality")
    metrics = {name: run.data.metrics.get(f"{scope}_{name}") for name in SCOPED_METRIC_NAMES}
    metrics["model_size_mib"] = run.data.metrics.get("model_size_mib")
    return CompletedRunRecord(
        run_id=run.info.run_id,
        status=run.info.status,
        run_kind=kind,
        evaluation_scope=scope,
        experiment_name=tags["experiment_name"],
        dataset=tags.get("dataset", ""),
        task=tags["task"],
        label_policy_version=tags.get("label_policy_version", ""),
        model=tags["model"],
        modality=modality,
        seed=tags["seed"],
        bundle_id=tags["dataset_bundle_id"],
        bundle_manifest_sha256=run.data.params.get("bundle_manifest_sha256", ""),
        split_assignment_id=tags["split_assignment_id"],
        model_package_id=tags["model_package_id"],
        source_training_run_id=parent,
        git_commit=tags.get("git_commit", ""),
        git_dirty=tags.get("git_dirty", ""),
        dependency_lock_sha256=tags.get("dependency_lock_sha256", ""),
        config_source_sha256=tags.get("config_source_sha256", ""),
        config_semantic_sha256=tags.get("config_semantic_sha256", ""),
        local_model_path=tags.get("local_model_path", ""),
        local_model_sha256=tags.get("local_model_sha256", ""),
        checkpoint_sha256=tags.get("checkpoint_sha256", ""),
        source_training_run_parameter=run.data.params.get("source_training_run_id", ""),
        source_cxr_training_run_id=tags.get("source_cxr_training_run_id", ""),
        source_cxr_model_package_id=tags.get("source_cxr_model_package_id", ""),
        source_cxr_checkpoint_sha256=tags.get("source_cxr_checkpoint_sha256", ""),
        metrics=metrics,
    )


def has_matching_training_parent(
    test_record: CompletedRunRecord,
    training_record: CompletedRunRecord,
) -> bool:
    """Return whether a test record matches its completed training parent."""
    return (
        test_record.run_kind == "test_evaluation"
        and test_record.evaluation_scope == "test"
        and training_record.run_kind == "training"
        and training_record.evaluation_scope == "validation"
        and test_record.source_training_run_id == training_record.run_id
        and all(
            getattr(test_record, field) == getattr(training_record, field)
            for field in (
                "experiment_name",
                "model",
                "modality",
                "task",
                "model_package_id",
                "bundle_id",
                "split_assignment_id",
                "seed",
            )
        )
    )


def comparison_metrics_are_valid(record: CompletedRunRecord) -> bool:
    """Validate the scalar subset required by the comparison views."""
    for name in (*COMPARISON_SCOPED_METRIC_NAMES, "model_size_mib"):
        value = record.metrics[name]
        if name == "latency_ms" and record.modality in {"image", "fusion"} and value is None:
            continue
        if not _finite_number(value):
            return False
        numeric = float(value)
        if name in _BOUNDED_METRICS and not 0.0 <= numeric <= 1.0:
            return False
    latency = record.metrics["latency_ms"]
    size = record.metrics["model_size_mib"]
    return (
        (record.modality in {"image", "fusion"} or float(latency) >= 0.0)
        and _finite_number(size)
        and float(size) > 0.0
    )


def validated_image_test_metrics(record: CompletedRunRecord) -> dict[str, float]:
    """Return the complete finite scalar contract for one image test run."""
    if (
        record.modality != "image"
        or record.run_kind != "test_evaluation"
        or record.evaluation_scope != "test"
    ):
        raise ValueError(f"Run {record.run_id} is not an image test-evaluation run")
    return _validated_seed_metrics(record)


def validated_neural_test_metrics(record: CompletedRunRecord) -> dict[str, float]:
    """Return the complete finite scalar contract for an image or fusion test run."""
    if record.modality not in {"image", "fusion"}:
        raise ValueError(f"Run {record.run_id} is not a neural test-evaluation run")
    if record.run_kind != "test_evaluation" or record.evaluation_scope != "test":
        raise ValueError(f"Run {record.run_id} is not a neural test-evaluation run")
    return _validated_seed_metrics(record)


def _validated_seed_metrics(record: CompletedRunRecord) -> dict[str, float]:
    validated: dict[str, float] = {}
    for name in SEED_SPECIFIC_METRIC_NAMES:
        value = record.metrics[name]
        if not _finite_number(value):
            raise ValueError(f"Run {record.run_id} metric {name} is missing or non-finite")
        numeric = float(value)
        if name in _BOUNDED_METRICS and not 0.0 <= numeric <= 1.0:
            raise ValueError(f"Run {record.run_id} metric {name} is outside [0, 1]")
        if name in _COUNT_METRICS and (numeric < 0.0 or not numeric.is_integer()):
            raise ValueError(f"Run {record.run_id} metric {name} is not a nonnegative count")
        if name == "model_size_mib" and numeric <= 0.0:
            raise ValueError(f"Run {record.run_id} model size is not positive")
        validated[name] = numeric
    return validated


def _finite_number(value: object) -> bool:
    return bool(
        not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)
    )
