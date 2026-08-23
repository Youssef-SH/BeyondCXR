"""Publish and validate an aggregate result from three RSNA evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beyondcxr.training.rsna_evaluation_result import (
    validate_rsna_evaluation,
    validate_rsna_model_package,
)
from beyondcxr.utils.operational_logging import add_logging_argument, configure_logging
from beyondcxr.utils.package_identity import (
    canonical_scientific_id,
    pretrained_weight_semantic_identity,
)
from beyondcxr.utils.privacy import validate_public_reports
from beyondcxr.utils.publication import (
    install_immutable_directory,
    staging_directory,
    validate_path_component,
)

EXPECTED_SEEDS = (17, 42, 2026)
SEED_SUMMARY_SCHEMA_VERSION = 1
SEED_SUMMARY_PREFIX = "seed-summary-"
MANIFEST_FILENAME = "manifest.json"
SEED_SUMMARY_FILENAMES = frozenset({MANIFEST_FILENAME, "metrics.csv", "summary.md"})
_PROBABILITY_METRICS = (
    "average_precision",
    "roc_auc",
    "brier_score",
    "expected_calibration_error",
    "calibration_slope",
    "calibration_intercept",
)
_OPERATING_POINTS = ("youden_j", "target_sensitivity")
_OPERATING_METRICS = (
    "precision",
    "recall",
    "specificity",
    "f1",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
)
_SUPPORTED_FAMILIES = {
    "cxr_densenet": ["cxr"],
    "cxr_metadata_concat": ["cxr", "metadata"],
}
_POLICY = "rsna-arithmetic-mean-and-sample-standard-deviation-v1"


@dataclass(frozen=True)
class SeedSummaryResult:
    """One validated reusable three-seed scientific result."""

    seed_summary_id: str
    directory: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str


def publish_seed_summary(
    evaluation_ids: Sequence[str],
    *,
    output_directory: str | Path = "reports",
    model_directory: str | Path = "models/rsna",
    private_directory: str | Path = "private",
) -> SeedSummaryResult:
    """Publish the exact validated three-seed aggregate scientific result."""
    members = _validated_members(
        evaluation_ids,
        report_root=output_directory,
        model_root=model_directory,
        private_root=private_directory,
    )
    context = _scientific_context(members)
    aggregate = _aggregate(members)
    semantic = {
        **context,
        "evaluation_ids": [item["evaluation_id"] for item in members],
        "required_seeds": list(EXPECTED_SEEDS),
        "policy": _POLICY,
    }
    summary_id = canonical_scientific_id(SEED_SUMMARY_PREFIX, semantic)
    document = {
        "seed_summary_schema_version": SEED_SUMMARY_SCHEMA_VERSION,
        **semantic,
        "seed_summary_id": summary_id,
        "members": members,
        "aggregate": aggregate,
    }
    destination = Path(output_directory) / "rsna" / "seed-summaries" / summary_id
    stage = staging_directory(destination)
    try:
        _write(stage, document)
        validate_public_reports(stage.iterdir(), forbidden_source_values=())
        install_immutable_directory(
            stage,
            destination,
            lambda path, **kwargs: validate_seed_summary(
                path,
                report_root=output_directory,
                model_root=model_directory,
                private_root=private_directory,
                **kwargs,
            ),
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return validate_seed_summary(
        destination,
        report_root=output_directory,
        model_root=model_directory,
        private_root=private_directory,
        expected_seed_summary_id=summary_id,
    )


def validate_seed_summary(
    directory: str | Path,
    *,
    report_root: str | Path,
    model_root: str | Path,
    private_root: str | Path,
    expected_seed_summary_id: str | None = None,
    enforce_directory_name: bool = True,
) -> SeedSummaryResult:
    """Transitively validate and re-derive a three-seed aggregate result."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Seed summary has an invalid artifact set")
    with os.scandir(root) as entries:
        inspected = list(entries)
    if {entry.name for entry in inspected} != set(SEED_SUMMARY_FILENAMES) or any(
        entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in inspected
    ):
        raise ValueError("Seed summary has an invalid artifact set")
    manifest_bytes = (root / MANIFEST_FILENAME).read_bytes()
    try:
        document = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Seed summary manifest is unreadable") from exc
    fields = {
        "seed_summary_schema_version",
        "seed_summary_id",
        "dataset_id",
        "task_id",
        "family_id",
        "modalities",
        "bundle_id",
        "split_assignment_id",
        "family_scientific_context",
        "evaluation_policy",
        "evaluation_ids",
        "required_seeds",
        "policy",
        "members",
        "aggregate",
    }
    if not isinstance(document, dict) or set(document) != fields:
        raise ValueError("Seed summary manifest has an unexpected field set")
    schema_version = document["seed_summary_schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SEED_SUMMARY_SCHEMA_VERSION
    ):
        raise ValueError("Seed summary schema version is invalid")
    if document["required_seeds"] != list(EXPECTED_SEEDS) or document["policy"] != _POLICY:
        raise ValueError("Seed summary policy is invalid")
    members = _validated_members(
        document["evaluation_ids"],
        report_root=report_root,
        model_root=model_root,
        private_root=private_root,
    )
    if document["members"] != members:
        raise ValueError("Seed summary members differ from evaluation authorities")
    context = _scientific_context(members)
    if any(document[key] != value for key, value in context.items()):
        raise ValueError("Seed summary scientific context is inconsistent")
    aggregate = _aggregate(members)
    if document["aggregate"] != aggregate:
        raise ValueError("Seed summary aggregate differs from re-derived claims")
    semantic = {
        key: document[key]
        for key in fields
        if key
        not in {
            "seed_summary_schema_version",
            "seed_summary_id",
            "members",
            "aggregate",
        }
    }
    summary_id = canonical_scientific_id(SEED_SUMMARY_PREFIX, semantic)
    if document["seed_summary_id"] != summary_id:
        raise ValueError("Seed summary identity differs from its scientific claims")
    if expected_seed_summary_id is not None and summary_id != expected_seed_summary_id:
        raise ValueError("Seed summary differs from the expected identity")
    if enforce_directory_name and root.name != summary_id:
        raise ValueError("Seed summary directory differs from its identity")
    with tempfile.TemporaryDirectory(prefix="beyondcxr-seed-summary-validation-") as workspace:
        temporary = Path(workspace)
        _write(temporary, document)
        for filename in SEED_SUMMARY_FILENAMES - {MANIFEST_FILENAME}:
            if (root / filename).read_bytes() != (temporary / filename).read_bytes():
                raise ValueError("Seed summary rendering differs from its manifest")
    return SeedSummaryResult(
        summary_id,
        root,
        document,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _validated_members(
    evaluation_ids: Sequence[str],
    *,
    report_root: str | Path,
    model_root: str | Path,
    private_root: str | Path,
) -> list[dict[str, Any]]:
    identities = tuple(evaluation_ids)
    if len(identities) != 3 or len(set(identities)) != 3:
        raise ValueError("Seed summary requires exactly three distinct evaluation IDs")
    members = []
    for evaluation_id in identities:
        validate_path_component(evaluation_id, "evaluation ID")
        evaluation = validate_rsna_evaluation(
            Path(report_root) / "rsna" / "evaluations" / evaluation_id,
            private_root=private_root,
            model_root=model_root,
            expected_evaluation_id=evaluation_id,
        )
        package = validate_rsna_model_package(model_root, evaluation.manifest["model_package_id"])
        seed = _package_seed(package)
        if evaluation.manifest["seed"] != seed:
            raise ValueError("Seed summary evaluation and package seed coordinates differ")
        members.append(
            {
                "seed": seed,
                "evaluation_id": evaluation_id,
                "model_package_id": package["model_package_id"],
                "claims": evaluation.manifest["claims"],
                "scientific_context": {
                    key: evaluation.manifest[key]
                    for key in (
                        "dataset_id",
                        "task_id",
                        "family_id",
                        "modalities",
                        "bundle_id",
                        "split_assignment_id",
                    )
                },
                "family_scientific_context": _package_family_context(
                    package, model_root=model_root
                ),
                "evaluation_policy": evaluation.manifest["evaluation_policy"],
            }
        )
    members.sort(key=lambda item: item["seed"])
    if tuple(item["seed"] for item in members) != EXPECTED_SEEDS:
        raise ValueError(f"Seed summary requires seeds {list(EXPECTED_SEEDS)}")
    reference = members[0]
    if any(
        item["scientific_context"] != reference["scientific_context"]
        or item["family_scientific_context"] != reference["family_scientific_context"]
        or item["evaluation_policy"] != reference["evaluation_policy"]
        for item in members[1:]
    ):
        raise ValueError("Seed summary evaluations are not scientifically compatible")
    return members


def _scientific_context(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    member = members[0]
    return {
        **dict(member["scientific_context"]),
        "family_scientific_context": member["family_scientific_context"],
        "evaluation_policy": member["evaluation_policy"],
    }


def _aggregate(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "probability_metrics": {
            metric: _summary_statistics(
                [item["claims"]["probability_metrics"][metric] for item in members]
            )
            for metric in _PROBABILITY_METRICS
        },
        "operating_points": {
            point: {
                "metrics": {
                    metric: _summary_statistics(
                        [
                            item["claims"]["operating_points"][point]["metrics"][metric]
                            for item in members
                        ]
                    )
                    for metric in _OPERATING_METRICS
                }
            }
            for point in _OPERATING_POINTS
        },
    }


def _summary_statistics(values: Sequence[object]) -> dict[str, float]:
    numeric = [float(value) for value in values]
    return {
        "mean": statistics.fmean(numeric),
        "sample_standard_deviation": statistics.stdev(numeric),
    }


def _package_seed(package: Mapping[str, Any]) -> int:
    training_policy = package.get("training_policy")
    value = training_policy.get("seed") if isinstance(training_policy, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Seed summary package seed coordinate is invalid")
    return value


def _package_family_context(
    package: Mapping[str, Any], *, model_root: str | Path
) -> dict[str, Any]:
    family_id = package.get("family_id")
    modalities = package.get("modalities")
    if family_id not in _SUPPORTED_FAMILIES or modalities != _SUPPORTED_FAMILIES[family_id]:
        raise ValueError("Seed summary package family/modalities contract is unsupported")
    training_policy = package.get("training_policy")
    if not isinstance(training_policy, Mapping):
        raise ValueError("Seed summary requires a neural package training policy")
    normalized_training_policy = {
        key: value for key, value in training_policy.items() if key != "seed"
    }
    context = {
        "label_policy_version": package["label_policy_version"],
        "positive_class": package["positive_class"],
        "fit_config": package["fit_config"],
        "model_identity": {
            **package["model_identity"],
            "pretrained_weight": pretrained_weight_semantic_identity(
                package["model_identity"]["pretrained_weight"]
            ),
        },
        "input_contract": package["input_contract"],
        "training_transform_contract": package["training_transform_contract"],
        "evaluation_transform_contract": package["evaluation_transform_contract"],
        "training_policy": normalized_training_policy,
        "threshold_contract": package["threshold_contract"],
    }
    if family_id == "cxr_densenet":
        if package.get("source_package_id") is not None:
            raise ValueError("CXR seed-summary package cannot declare source package lineage")
        return context
    source_package_id = package.get("source_package_id")
    source = validate_rsna_model_package(model_root, source_package_id)
    if (
        source.get("family_id") != "cxr_densenet"
        or source.get("modalities") != ["cxr"]
        or _package_seed(source) != _package_seed(package)
        or any(
            source.get(field) != package.get(field)
            for field in ("dataset_id", "task_id", "bundle_id", "split_assignment_id")
        )
    ):
        raise ValueError("Fusion package source CXR scientific lineage is incompatible")
    context.update(
        {
            "structured_preprocessor_contract": package["structured_preprocessor_contract"],
            "structured_input_conversion": package["structured_input_conversion"],
            "fusion_architecture": package["fusion_architecture"],
            "source_cxr_family_context": _package_family_context(source, model_root=model_root),
        }
    )
    return context


def _write(directory: Path, document: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (directory / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ["record_kind", "seed", "evaluation_id", "section", "policy", "metric", "value"]
        )
        for member in document["members"]:
            for metric in _PROBABILITY_METRICS:
                writer.writerow(
                    [
                        "member",
                        member["seed"],
                        member["evaluation_id"],
                        "probability_metrics",
                        "",
                        metric,
                        f"{member['claims']['probability_metrics'][metric]:.12g}",
                    ]
                )
            for point in _OPERATING_POINTS:
                values = member["claims"]["operating_points"][point]
                writer.writerow(
                    [
                        "member",
                        member["seed"],
                        member["evaluation_id"],
                        "operating_points",
                        point,
                        "threshold",
                        f"{values['threshold']:.12g}",
                    ]
                )
                for metric in _OPERATING_METRICS:
                    writer.writerow(
                        [
                            "member",
                            member["seed"],
                            member["evaluation_id"],
                            "operating_points",
                            point,
                            metric,
                            f"{values['metrics'][metric]:.12g}",
                        ]
                    )
        for metric in _PROBABILITY_METRICS:
            values = document["aggregate"]["probability_metrics"][metric]
            for statistic in ("mean", "sample_standard_deviation"):
                value = values[statistic]
                writer.writerow(
                    [
                        "aggregate",
                        "",
                        "",
                        "probability_metrics",
                        "",
                        f"{metric}.{statistic}",
                        f"{value:.12g}",
                    ]
                )
        for point in _OPERATING_POINTS:
            point_values = document["aggregate"]["operating_points"][point]
            for metric in _OPERATING_METRICS:
                values = point_values["metrics"][metric]
                for statistic in ("mean", "sample_standard_deviation"):
                    value = values[statistic]
                    writer.writerow(
                        [
                            "aggregate",
                            "",
                            "",
                            "operating_points",
                            point,
                            f"{metric}.{statistic}",
                            f"{value:.12g}",
                        ]
                    )
    family_context_sha256 = hashlib.sha256(
        json.dumps(
            document["family_scientific_context"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    evaluation_policy = document["evaluation_policy"]
    threshold_policy = evaluation_policy["threshold_selection"]
    lines = [
        "# RSNA three-seed evaluation summary",
        "",
        f"- Summary ID: `{document['seed_summary_id']}`",
        f"- Dataset: `{document['dataset_id']}`",
        f"- Task: `{document['task_id']}`",
        f"- Family: `{document['family_id']}`",
        f"- Modalities: `{json.dumps(document['modalities'], separators=(',', ':'))}`",
        f"- Bundle: `{document['bundle_id']}`",
        f"- Split assignment: `{document['split_assignment_id']}`",
        f"- Family scientific context SHA-256: `{family_context_sha256}`",
        f"- Label policy: `{document['family_scientific_context']['label_policy_version']}`",
        f"- Positive class: `{document['family_scientific_context']['positive_class']}`",
        f"- Evaluation policy: `{evaluation_policy['policy_version']}`",
        f"- Calibration bins: `{evaluation_policy['calibration_bins']}`",
        f"- Target sensitivity: `{threshold_policy['sensitivity_target']}`",
        f"- Youden-J policy: `{threshold_policy['youden_j_policy_version']}`",
        f"- Target-sensitivity policy: `{threshold_policy['target_sensitivity_policy_version']}`",
        "",
        "## Evaluation members",
        "",
        "| Seed | Evaluation ID | Model package ID |",
        "| ---: | --- | --- |",
    ]
    for member in document["members"]:
        lines.append(
            f"| {member['seed']} | `{member['evaluation_id']}` | `{member['model_package_id']}` |"
        )
    lines.extend(
        [
            "",
            "## Per-seed probability and calibration metrics",
            "",
            "Thresholds are seed-specific validation-derived operating points and are not "
            "averaged.",
            "",
            "| Seed | Average precision | ROC AUC | Brier score | ECE | "
            "Calibration slope | Calibration intercept |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for member in document["members"]:
        metrics = member["claims"]["probability_metrics"]
        lines.append(
            f"| {member['seed']} | "
            f"{metrics['average_precision']:.6f} | {metrics['roc_auc']:.6f} | "
            f"{metrics['brier_score']:.6f} | {metrics['expected_calibration_error']:.6f} | "
            f"{metrics['calibration_slope']:.6f} | {metrics['calibration_intercept']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate probability and calibration metrics",
            "",
            "| Metric | Mean | Sample standard deviation |",
            "| --- | ---: | ---: |",
        ]
    )
    for metric in _PROBABILITY_METRICS:
        values = document["aggregate"]["probability_metrics"][metric]
        lines.append(
            f"| `{metric}` | {values['mean']:.6f} | {values['sample_standard_deviation']:.6f} |"
        )
    for point in _OPERATING_POINTS:
        title = "Youden-J" if point == "youden_j" else "Target sensitivity"
        lines.extend(
            [
                "",
                f"## Per-seed {title} operating point",
                "",
                "| Seed | Threshold | Precision | Recall | Specificity | F1 | TN | FP | FN | TP |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for member in document["members"]:
            values = member["claims"]["operating_points"][point]
            metrics = values["metrics"]
            lines.append(
                f"| {member['seed']} | {values['threshold']:.6f} | "
                f"{metrics['precision']:.6f} | {metrics['recall']:.6f} | "
                f"{metrics['specificity']:.6f} | {metrics['f1']:.6f} | "
                f"{metrics['true_negative']} | {metrics['false_positive']} | "
                f"{metrics['false_negative']} | {metrics['true_positive']} |"
            )
        lines.extend(
            [
                "",
                f"## Aggregate {title} operating-point metrics",
                "",
                "| Metric | Mean | Sample standard deviation |",
                "| --- | ---: | ---: |",
            ]
        )
        for metric in _OPERATING_METRICS:
            values = document["aggregate"]["operating_points"][point]["metrics"][metric]
            lines.append(
                f"| `{metric}` | {values['mean']:.6f} | {values['sample_standard_deviation']:.6f} |"
            )
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-ids", nargs=3, required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("reports"))
    parser.add_argument("--model-directory", type=Path, default=Path("models/rsna"))
    parser.add_argument("--private-directory", type=Path, default=Path("private"))
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        result = publish_seed_summary(
            args.evaluation_ids,
            output_directory=args.output_directory,
            model_directory=args.model_directory,
            private_directory=args.private_directory,
        )
    except (OSError, ValueError) as exc:
        print(f"Seed summary failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps({"seed_summary_id": result.seed_summary_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
