"""Shared construction and validation of RSNA training and evaluation reports."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "beyondcxr-matplotlib"))

import numpy as np

from beyondcxr.evaluation.metrics import (
    CALIBRATION_BINNING_STRATEGY,
    OperatingPointMetrics,
    ProbabilityMetrics,
    evaluate_operating_point,
    evaluate_probabilities,
    target_sensitivity_threshold,
    youden_j_threshold,
)
from beyondcxr.evaluation.plots import write_evaluation_plots
from beyondcxr.training.config import ExperimentConfig, load_experiment_config, with_runtime
from beyondcxr.training.rsna_validation_evidence import (
    VALIDATION_EVIDENCE_FILENAME,
    CanonicalValidationCohort,
    validate_cxr_epoch_history,
    validate_validation_evidence,
)

REQUIRED_REPORT_FILENAMES = frozenset(
    {
        "metrics.json",
        "evaluation_report.md",
        "confusion_summary.md",
        "roc_curve.png",
        "precision_recall_curve.png",
        "calibration_curve.png",
        "confusion_matrix_youden_j.png",
        "confusion_matrix_target_sensitivity.png",
    }
)
CXR_REPORT_LIMITATIONS = [
    "The challenge target is radiology-derived.",
    "The shared CXR cache authenticates and decodes all partitions; "
    "test samples are not used for fitting, selection, or threshold derivation.",
]
CXR_REPORT_RUNTIME_FIELDS = (
    "requested_device",
    "resolved_device",
    "cuda_available",
    "mixed_precision_requested",
    "mixed_precision_effective",
    "pin_memory_requested",
    "pin_memory_effective",
    "torch_version",
    "torchvision_version",
    "torchxrayvision_version",
    "cuda_runtime_version",
    "cudnn_version",
    "gpu_device_name",
    "gpu_device_index",
    "gpu_compute_capability",
)


def metrics_document(
    *,
    scope: str,
    calibration_bins: int,
    sensitivity_target: float,
    thresholds: dict[str, float],
    probability: ProbabilityMetrics,
    youden: OperatingPointMetrics,
    target_sensitivity: OperatingPointMetrics,
) -> dict[str, Any]:
    """Build one aggregate metrics document."""
    return {
        "evaluation_scope": scope,
        "calibration": {
            "calibration_bins": calibration_bins,
            "calibration_binning_strategy": CALIBRATION_BINNING_STRATEGY,
        },
        "probability_metrics": probability.as_dict(),
        "operating_points": {
            "youden_j": {
                "threshold": thresholds["youden_j"],
                "metrics": youden.as_dict(),
            },
            "target_sensitivity": {
                "configured_target_sensitivity": sensitivity_target,
                "threshold": thresholds["target_sensitivity"],
                "metrics": target_sensitivity.as_dict(),
            },
        },
    }


def write_run_reports(
    directory: Path,
    *,
    model_name: str,
    targets: np.ndarray,
    probabilities: np.ndarray,
    document: dict[str, Any],
) -> None:
    """Render the aggregate report set for one evaluation scope."""
    directory.mkdir(parents=True, exist_ok=True)
    operating = document["operating_points"]
    write_evaluation_plots(
        targets,
        probabilities,
        youden_j_threshold=operating["youden_j"]["threshold"],
        target_sensitivity_threshold=operating["target_sensitivity"]["threshold"],
        calibration_bins=document["calibration"]["calibration_bins"],
        output_directory=directory,
    )
    (directory / "metrics.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_evaluation_report(directory / "evaluation_report.md", model_name, document)
    _write_confusion_summary(directory / "confusion_summary.md", document)


def validate_report_set(directory: str | Path) -> None:
    """Require the complete aggregate report set and no additional entries."""
    report_directory = Path(directory)
    with os.scandir(report_directory) as entries:
        inspected = list(entries)
    actual = {entry.name for entry in inspected}
    if actual != REQUIRED_REPORT_FILENAMES:
        missing = sorted(REQUIRED_REPORT_FILENAMES - actual)
        unexpected = sorted(actual - REQUIRED_REPORT_FILENAMES)
        raise ValueError(f"Run report set is invalid: missing={missing}, unexpected={unexpected}")
    if any(entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in inspected):
        raise ValueError("Run reports must be regular non-symlink files")


def validate_training_report(
    directory: str | Path,
    *,
    run_id: str,
    model_package_id: str,
    family_id: str,
    seed: int,
    package: Mapping[str, Any],
    package_directory: str | Path,
    canonical_validation_cohort: CanonicalValidationCohort | None = None,
) -> None:
    """Validate exact report structure, rendering, and package-derived claims."""
    validate_report_set(directory)
    report_directory = Path(directory)
    try:
        raw = (report_directory / "metrics.json").read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Training report metrics are unreadable") from exc
    expected = {
        "run_id": run_id,
        "model_package_id": model_package_id,
        "family_id": family_id,
        "seed": seed,
    }
    if not isinstance(document, dict) or document.get("training_package") != expected:
        raise ValueError("Training report package binding is invalid")
    expected_fields = {
        "evaluation_scope",
        "calibration",
        "probability_metrics",
        "operating_points",
        "training_package",
    }
    if family_id == "cxr_densenet":
        expected_fields.add("cxr_training")
    if (
        set(document) != expected_fields
        or raw != (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    ):
        raise ValueError("Training report metrics contract is invalid")
    if any(
        not 0.0 <= document["probability_metrics"][name] <= 1.0
        for name in (
            "average_precision",
            "roc_auc",
            "brier_score",
            "expected_calibration_error",
        )
    ):
        raise ValueError("Training report probability metrics are invalid")
    _validate_training_metrics(document)
    package_path = Path(package_directory)
    config = with_runtime(
        load_experiment_config(package_path / "resolved_config.yaml"),
        seed=seed,
    )
    evidence = validate_validation_evidence(
        package_path / VALIDATION_EVIDENCE_FILENAME,
        package=package,
        config=config,
        expected_cohort=canonical_validation_cohort,
    )
    targets = np.asarray(evidence["targets"], dtype=np.int8)
    probabilities = np.asarray(evidence["probabilities"], dtype=np.float64)
    expected_thresholds = {
        "youden_j": youden_j_threshold(targets, probabilities),
        "target_sensitivity": target_sensitivity_threshold(
            targets,
            probabilities,
            sensitivity=float(evidence["sensitivity_target"]),
        ),
    }
    expected_document = metrics_document(
        scope="validation",
        calibration_bins=int(evidence["calibration_bins"]),
        sensitivity_target=float(evidence["sensitivity_target"]),
        thresholds=expected_thresholds,
        probability=evaluate_probabilities(
            targets,
            probabilities,
            calibration_bins=int(evidence["calibration_bins"]),
        ),
        youden=evaluate_operating_point(
            targets, probabilities, threshold=expected_thresholds["youden_j"]
        ),
        target_sensitivity=evaluate_operating_point(
            targets,
            probabilities,
            threshold=expected_thresholds["target_sensitivity"],
        ),
    )
    for field in ("calibration", "probability_metrics", "operating_points"):
        if document[field] != expected_document[field]:
            raise ValueError("Training report scientific claims differ from validation evidence")
    package_seed = package.get("seed")
    if package_seed is None and isinstance(package.get("training_policy"), Mapping):
        package_seed = package["training_policy"].get("seed")
    threshold_contract = package.get("threshold_contract")
    if (
        package.get("model_package_id") != model_package_id
        or package.get("family_id") != family_id
        or package_seed != seed
        or package.get("thresholds")
        != {
            policy: document["operating_points"][policy]["threshold"]
            for policy in ("youden_j", "target_sensitivity")
        }
        or not isinstance(threshold_contract, Mapping)
        or threshold_contract.get("sensitivity_target")
        != document["operating_points"]["target_sensitivity"]["configured_target_sensitivity"]
    ):
        raise ValueError("Training report claims differ from the model package authority")
    if family_id == "cxr_densenet":
        training = document["cxr_training"]
        _validate_cxr_training_contract(training, config=config, package=package)
        runtime = package.get("runtime_provenance")
        if not isinstance(runtime, Mapping):
            raise ValueError("Training report neural runtime authority is invalid")
        expected_training = {
            "lineage": {
                key: package[key]
                for key in (
                    "bundle_id",
                    "bundle_manifest_sha256",
                    "split_assignment_id",
                    "task_id",
                    "label_policy_version",
                )
            },
            "source_authentication": package.get("source_authentication"),
            "model_identity": package.get("model_identity"),
            "input_contract": package.get("input_contract"),
            "training_transform_contract": package.get("training_transform_contract"),
            "evaluation_transform_contract": package.get("evaluation_transform_contract"),
            "class_weighting": package.get("training_policy", {}).get("class_weight"),
            "runtime": {field: runtime[field] for field in CXR_REPORT_RUNTIME_FIELDS},
            "epoch_history": evidence["epoch_history"],
            "selection": package.get("selection"),
            "limitations": CXR_REPORT_LIMITATIONS,
        }
        if training != expected_training:
            raise ValueError("Training report neural claims differ from the package authority")
    selection = package.get("selection")
    if isinstance(selection, Mapping) and document["probability_metrics"][
        "average_precision"
    ] != selection.get("validation_average_precision"):
        raise ValueError("Training report validation metric differs from the package authority")
    with tempfile.TemporaryDirectory(prefix="beyondcxr-training-report-validation-") as temporary:
        rendered = Path(temporary)
        write_run_reports(
            rendered,
            model_name=family_id,
            targets=targets,
            probabilities=probabilities,
            document=document,
        )
        for filename in REQUIRED_REPORT_FILENAMES:
            if (report_directory / filename).read_bytes() != (rendered / filename).read_bytes():
                raise ValueError("Training report differs from authoritative evidence rendering")


def training_report_sha256(directory: str | Path) -> str:
    """Return one unambiguous byte witness for the complete bounded report set."""
    validate_report_set(directory)
    digest = hashlib.sha256()
    root = Path(directory)
    for filename in sorted(REQUIRED_REPORT_FILENAMES):
        encoded_name = filename.encode()
        content = (root / filename).read_bytes()
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _validate_training_metrics(document: Mapping[str, Any]) -> None:
    probability_fields = {
        "average_precision",
        "roc_auc",
        "brier_score",
        "expected_calibration_error",
        "calibration_slope",
        "calibration_intercept",
    }
    operating_fields = {
        "precision",
        "recall",
        "specificity",
        "f1",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    }
    calibration = document.get("calibration")
    probability = document.get("probability_metrics")
    operating = document.get("operating_points")
    if (
        document.get("evaluation_scope") != "validation"
        or not isinstance(calibration, dict)
        or set(calibration) != {"calibration_bins", "calibration_binning_strategy"}
        or type(calibration["calibration_bins"]) is not int
        or calibration["calibration_bins"] <= 0
        or calibration["calibration_binning_strategy"] != CALIBRATION_BINNING_STRATEGY
        or not isinstance(probability, dict)
        or set(probability) != probability_fields
        or any(
            type(value) not in {int, float} or not math.isfinite(value)
            for value in probability.values()
        )
        or not isinstance(operating, dict)
        or set(operating) != {"youden_j", "target_sensitivity"}
    ):
        raise ValueError("Training report metrics contract is invalid")
    for policy in ("youden_j", "target_sensitivity"):
        value = operating[policy]
        expected = {"threshold", "metrics"}
        if policy == "target_sensitivity":
            expected.add("configured_target_sensitivity")
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or type(value["threshold"]) not in {int, float}
            or not math.isfinite(value["threshold"])
            or not 0.0 <= value["threshold"] <= 1.0
            or not isinstance(value["metrics"], dict)
            or set(value["metrics"]) != operating_fields
            or any(
                type(value["metrics"][name]) not in {int, float}
                or not math.isfinite(value["metrics"][name])
                or not 0.0 <= value["metrics"][name] <= 1.0
                for name in ("precision", "recall", "specificity", "f1")
            )
            or any(
                type(value["metrics"][name]) is not int or value["metrics"][name] < 0
                for name in (
                    "true_negative",
                    "false_positive",
                    "false_negative",
                    "true_positive",
                )
            )
        ):
            raise ValueError("Training report operating-point contract is invalid")
        if policy == "target_sensitivity" and (
            type(value["configured_target_sensitivity"]) not in {int, float}
            or not math.isfinite(value["configured_target_sensitivity"])
            or not 0.0 < value["configured_target_sensitivity"] <= 1.0
        ):
            raise ValueError("Training report sensitivity target is invalid")


def _validate_cxr_training_contract(
    value: object, *, config: ExperimentConfig, package: Mapping[str, Any]
) -> None:
    expected = {
        "lineage",
        "source_authentication",
        "model_identity",
        "input_contract",
        "training_transform_contract",
        "evaluation_transform_contract",
        "class_weighting",
        "runtime",
        "epoch_history",
        "selection",
        "limitations",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or not isinstance(value["epoch_history"], list)
        or not isinstance(value["limitations"], list)
        or any(not isinstance(item, str) or not item for item in value["limitations"])
    ):
        raise ValueError("Training report neural contract is invalid")
    validate_cxr_epoch_history(value["epoch_history"], config=config, package=package)


def mlflow_metrics(
    *,
    scope: str,
    document: dict[str, Any],
    latency_ms: float | None,
    model_size_mib: float,
) -> dict[str, float]:
    """Flatten one aggregate document into stable MLflow metric names."""
    metrics = {
        f"{scope}_{key}": float(value) for key, value in document["probability_metrics"].items()
    }
    for policy, values in document["operating_points"].items():
        metrics[f"{scope}_{policy}_threshold"] = float(values["threshold"])
        metrics.update(
            {f"{scope}_{policy}_{key}": float(value) for key, value in values["metrics"].items()}
        )
    if latency_ms is not None:
        metrics[f"{scope}_latency_ms"] = latency_ms
    metrics["model_size_mib"] = model_size_mib
    return metrics


def _write_evaluation_report(path: Path, model_name: str, document: dict[str, Any]) -> None:
    scope = document["evaluation_scope"]
    probability = document["probability_metrics"]
    lines = [
        f"# {model_name} {scope} evaluation",
        "",
        "| Average Precision | ROC-AUC | Brier | ECE | Calibration slope | Calibration intercept |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {probability['average_precision']:.6f} | {probability['roc_auc']:.6f} | "
        f"{probability['brier_score']:.6f} | "
        f"{probability['expected_calibration_error']:.6f} | "
        f"{probability['calibration_slope']:.6f} | "
        f"{probability['calibration_intercept']:.6f} |",
    ]
    for policy, title in (
        ("youden_j", "Youden-J operating point"),
        ("target_sensitivity", "Target-sensitivity operating point"),
    ):
        values = document["operating_points"][policy]
        metrics = values["metrics"]
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"Validation-derived threshold: `{values['threshold']:.10f}`.",
                "",
                "| Precision | Recall | Specificity | F1 |",
                "| ---: | ---: | ---: | ---: |",
                f"| {metrics['precision']:.6f} | {metrics['recall']:.6f} | "
                f"{metrics['specificity']:.6f} | {metrics['f1']:.6f} |",
            ]
        )
    if cxr_training := document.get("cxr_training"):
        selection = cxr_training["selection"]
        authentication = cxr_training["source_authentication"]
        lines.extend(
            [
                "",
                "## Neural training summary",
                "",
                f"- Selected state: {selection['selected_stage']} epoch "
                f"{selection['selected_epoch']}",
                f"- Authenticated cache files (all partitions): {authentication['file_count']}",
                "- The shared CXR cache includes decoded test images; test samples are not "
                "used for fitting, selection, or threshold derivation.",
            ]
        )
    if cxr_evaluation := document.get("cxr_evaluation"):
        counts = cxr_evaluation["test_counts"]
        lines.extend(
            [
                "",
                "## Verified neural test evaluation",
                "",
                f"- Test samples: {counts['sample_count']}",
                f"- Positive samples: {counts['positive_count']}",
                f"- Negative samples: {counts['negative_count']}",
                "- Package and checkpoint verification completed before test access.",
                "- Operating points were frozen on validation.",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_confusion_summary(path: Path, document: dict[str, Any]) -> None:
    lines = [f"# Aggregate {document['evaluation_scope']} confusion summary", ""]
    for policy, title in (
        ("youden_j", "Youden-J operating point"),
        ("target_sensitivity", "Target-sensitivity operating point"),
    ):
        values = document["operating_points"][policy]["metrics"]
        lines.extend(
            [
                f"## {title}",
                "",
                f"- True negatives: {values['true_negative']:,}",
                f"- False positives: {values['false_positive']:,}",
                f"- False negatives: {values['false_negative']:,}",
                f"- True positives: {values['true_positive']:,}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
