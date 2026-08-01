"""Summarize three explicit compatible RSNA image test-evaluation runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlflow.exceptions import MlflowException
from sqlalchemy.exc import SQLAlchemyError

from radfusion.training.completed_runs import (
    SEED_AGGREGATE_METRIC_NAMES,
    SEED_SPECIFIC_METRIC_NAMES,
    CompletedRunRecord,
    has_matching_training_parent,
    require_completed_run,
    validated_image_test_metrics,
)
from radfusion.training.config import (
    ExperimentConfig,
    image_seed_compatibility_sha256,
    load_experiment_config,
)
from radfusion.utils.mlflow_utils import DEFAULT_TRACKING_URI, configure_mlflow
from radfusion.utils.neural_publication import (
    CONFIG_FILENAME,
    NEURAL_MODEL_FILENAME,
    validate_neural_package_metadata,
)
from radfusion.utils.operational_logging import (
    add_logging_argument,
    configure_logging,
    get_operational_logger,
    timed_phase,
)
from radfusion.utils.privacy import validate_public_reports
from radfusion.utils.publication import publish_directory, staging_directory

EXPECTED_SEEDS = (17, 42, 2026)
THRESHOLD_POLICIES = ("youden_j", "target_sensitivity")
SEED_SUMMARY_SCHEMA_VERSION = 1
SEED_SUMMARY_FILENAMES = frozenset({"summary.json", "metrics.csv", "summary.md"})
_LOGGER = get_operational_logger(__name__)


@dataclass(frozen=True)
class SeedSummaryResult:
    """Published result of one explicit RSNA image three-seed summary."""

    report_id: str
    report_directory: Path
    compatibility_sha256: str
    test_run_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Member:
    seed: int
    test: CompletedRunRecord
    training: CompletedRunRecord
    config: ExperimentConfig
    manifest: Mapping[str, Any]
    metrics: Mapping[str, float]
    compatibility: Mapping[str, Any]


def summarize_seed_runs(
    test_run_ids: Sequence[str],
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    output_directory: str | Path = "reports",
) -> SeedSummaryResult:
    """Validate and summarize exactly three explicit RSNA image test runs."""
    run_ids = tuple(test_run_ids)
    _validate_membership_input(run_ids)
    client = configure_mlflow(tracking_uri=tracking_uri)
    members = sorted(
        (_load_member(client, run_id) for run_id in run_ids),
        key=lambda item: item.seed,
    )
    observed_seeds = tuple(member.seed for member in members)
    if observed_seeds != EXPECTED_SEEDS:
        raise ValueError(
            f"Seed summary requires exactly {list(EXPECTED_SEEDS)}, observed {list(observed_seeds)}"
        )
    compatibility = members[0].compatibility
    for member in members[1:]:
        if member.compatibility != compatibility:
            raise ValueError(
                f"Seed {member.seed} run is not scientifically compatible with seed "
                f"{members[0].seed}"
            )
    compatibility_sha256 = _canonical_sha256(compatibility)
    aggregates = _aggregate_metrics(members)
    report_id = _report_id(members, compatibility_sha256)
    report = _report_document(members, compatibility_sha256, aggregates, report_id)
    destination = Path(output_directory) / members[0].test.dataset / "seed-summaries" / report_id
    if destination.exists():
        raise FileExistsError(f"Seed summary already exists: {destination}")
    stage = staging_directory(destination)
    try:
        _write_reports(stage, report, members, aggregates)
        _validate_report_set(stage)
        validate_public_reports(stage.iterdir(), forbidden_source_values=())
        publish_directory(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return SeedSummaryResult(
        report_id=report_id,
        report_directory=destination,
        compatibility_sha256=compatibility_sha256,
        test_run_ids=tuple(member.test.run_id for member in members),
    )


def _validate_membership_input(run_ids: tuple[str, ...]) -> None:
    if len(run_ids) != len(EXPECTED_SEEDS):
        raise ValueError(f"Seed summary requires exactly {len(EXPECTED_SEEDS)} test run IDs")
    if any(not isinstance(run_id, str) or not run_id.strip() for run_id in run_ids):
        raise ValueError("Seed summary run IDs must be non-empty strings")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Seed summary test run IDs must be distinct")


def _load_member(client, test_run_id: str) -> _Member:
    test_run = client.get_run(test_run_id)
    test = require_completed_run(test_run)
    if test.run_kind != "test_evaluation" or test.evaluation_scope != "test":
        raise ValueError(f"Run {test_run_id} is not a test-evaluation run")
    if test.modality != "image":
        raise ValueError(f"Run {test_run_id} is not an image run")
    seed = test.integer_seed()
    parent_run = client.get_run(test.source_training_run_id)
    training = require_completed_run(parent_run)
    if not has_matching_training_parent(test, training):
        raise ValueError(f"Run {test_run_id} does not match its source training run")
    if not training.local_model_path:
        raise ValueError(f"Training run {training.run_id} has no local model path")
    model_path = Path(training.local_model_path)
    if model_path.name != NEURAL_MODEL_FILENAME:
        raise ValueError(f"Training run {training.run_id} has an invalid neural model path")
    package_directory = model_path.parent
    manifest = validate_neural_package_metadata(package_directory)
    config = load_experiment_config(package_directory / CONFIG_FILENAME)
    metrics = validated_image_test_metrics(test)
    _validate_member_lineage(test_run, parent_run, test, training, config, manifest, seed)
    return _Member(
        seed,
        test,
        training,
        config,
        manifest,
        metrics,
        _compatibility_document(test, config, manifest),
    )


def _validate_member_lineage(
    test_run,
    parent_run,
    test: CompletedRunRecord,
    training: CompletedRunRecord,
    config: ExperimentConfig,
    manifest: Mapping[str, Any],
    seed: int,
) -> None:
    if config.model.modality != "image" or config.image is None:
        raise ValueError("Seed summary package does not contain an image config")
    if test.dataset == "" or training.dataset == "" or test.dataset != training.dataset:
        raise ValueError("Seed summary dataset lineage is missing or inconsistent")
    if (
        test.label_policy_version == ""
        or test.label_policy_version != training.label_policy_version
    ):
        raise ValueError("Seed summary label-policy lineage is missing or inconsistent")
    if (
        test.git_commit == ""
        or test.git_commit != training.git_commit
        or test.dependency_lock_sha256 == ""
        or test.dependency_lock_sha256 != training.dependency_lock_sha256
        or test.git_dirty == ""
        or test.git_dirty != training.git_dirty
    ):
        raise ValueError("Seed summary source provenance is missing or inconsistent")
    if test.source_training_run_parameter != training.run_id:
        raise ValueError("Test run source-training parameter disagrees with its tag")
    expected_manifest = {
        "training_mlflow_run_id": training.run_id,
        "model": training.model,
        "task": training.task,
        "bundle_id": training.bundle_id,
        "split_assignment_id": training.split_assignment_id,
        "label_policy_version": training.label_policy_version,
        "source_config_sha256": training.source_config_sha256,
        "semantic_config_sha256": training.semantic_config_sha256,
        "checkpoint_sha256": training.local_model_sha256,
        "model_package_id": training.model_package_id,
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Neural package {field} disagrees with its training run")
    if training.checkpoint_sha256 != manifest["checkpoint_sha256"]:
        raise ValueError("Training run checkpoint identities disagree")
    source = manifest["source_provenance"]
    if (
        source["git_commit"] != training.git_commit
        or str(source["git_dirty"]).lower() != training.git_dirty
        or source["dependency_lock_sha256"] != training.dependency_lock_sha256
    ):
        raise ValueError("Neural package source provenance disagrees with its training run")
    if manifest["training_policy"].get("seed") != seed or config.training.seed != seed:
        raise ValueError("Neural package seed lineage is inconsistent")
    if (
        config.dataset.registry_key != training.dataset
        or config.dataset.bundle_id != training.bundle_id
        or config.dataset.task_id != training.task
        or config.model.registry_key != training.model
    ):
        raise ValueError("Archived config disagrees with completed-run lineage")
    expected_bundle_manifest = manifest["bundle_manifest_sha256"]
    if any(
        observed != expected_bundle_manifest
        for observed in (
            training.bundle_manifest_sha256,
            test.bundle_manifest_sha256,
        )
    ):
        raise ValueError("Observed bundle-manifest lineage is inconsistent")
    if test.model_package_id != manifest["model_package_id"]:
        raise ValueError("Test run model package disagrees with its manifest")
    for field in ("checkpoint_sha256", "local_model_sha256"):
        if getattr(test, field) != manifest["checkpoint_sha256"]:
            raise ValueError(f"Test run {field} disagrees with its manifest")
    for policy in ("youden_j", "target_sensitivity"):
        threshold = float(manifest["thresholds"][policy])
        for record in (training, test):
            observed = record.metrics[f"{policy}_threshold"]
            if not _same_number(observed, threshold):
                raise ValueError(f"Run {record.run_id} threshold {policy} disagrees with package")
        for run in (parent_run, test_run):
            observed_tag = run.data.tags.get(f"threshold_{policy}")
            try:
                parsed_tag = float(observed_tag)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Run {run.info.run_id} threshold tag {policy} is invalid"
                ) from exc
            if not math.isclose(parsed_tag, threshold, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"Run {run.info.run_id} threshold tag {policy} disagrees with package"
                )


def _compatibility_document(
    test: CompletedRunRecord,
    config: ExperimentConfig,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    training_policy = dict(manifest["training_policy"])
    del training_policy["seed"]
    return {
        "config_compatibility_sha256": image_seed_compatibility_sha256(config),
        "dataset": test.dataset,
        "task": manifest["task"],
        "label_policy_version": manifest["label_policy_version"],
        "model": manifest["model"],
        "modality": manifest["modality"],
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": manifest["bundle_manifest_sha256"],
        "split_assignment_id": manifest["split_assignment_id"],
        "positive_class": manifest["positive_class"],
        "model_package_schema_version": manifest["model_package_schema_version"],
        "source_provenance": {
            "git_commit": test.git_commit,
            "git_dirty": test.git_dirty,
            "dependency_lock_sha256": test.dependency_lock_sha256,
        },
        "model_identity": manifest["model_identity"],
        "input_contract": manifest["input_contract"],
        "training_transform_contract": manifest["training_transform_contract"],
        "evaluation_transform_contract": manifest["evaluation_transform_contract"],
        "training_policy_without_seed": training_policy,
        "threshold_contract": manifest["threshold_contract"],
        "metrics_policy": manifest["metrics_policy"],
        "source_authentication": manifest["source_authentication"],
    }


def _aggregate_metrics(members: Sequence[_Member]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": statistics.fmean(member.metrics[name] for member in members),
            "sample_standard_deviation": statistics.stdev(
                member.metrics[name] for member in members
            ),
        }
        for name in SEED_AGGREGATE_METRIC_NAMES
    }


def _report_id(members: Sequence[_Member], compatibility_sha256: str) -> str:
    payload = {
        "schema_version": SEED_SUMMARY_SCHEMA_VERSION,
        "compatibility_sha256": compatibility_sha256,
        "members": [
            {
                "seed": member.seed,
                "test_run_id": member.test.run_id,
                "training_run_id": member.training.run_id,
                "model_package_id": member.test.model_package_id,
                "metrics": dict(member.metrics),
            }
            for member in members
        ],
    }
    return f"seed-summary-{_canonical_sha256(payload)}"


def _report_document(
    members: Sequence[_Member],
    compatibility_sha256: str,
    aggregates: Mapping[str, Mapping[str, float]],
    report_id: str,
) -> dict[str, Any]:
    first = members[0]
    return {
        "seed_summary_schema_version": SEED_SUMMARY_SCHEMA_VERSION,
        "report_id": report_id,
        "compatibility_sha256": compatibility_sha256,
        "compatibility": dict(first.compatibility),
        "expected_seeds": list(EXPECTED_SEEDS),
        "dataset": first.test.dataset,
        "task": first.test.task,
        "model": first.test.model,
        "modality": first.test.modality,
        "bundle_id": first.test.bundle_id,
        "bundle_manifest_sha256": first.manifest["bundle_manifest_sha256"],
        "split_assignment_id": first.test.split_assignment_id,
        "label_policy_version": first.test.label_policy_version,
        "members": [
            {
                "seed": member.seed,
                "test_run_id": member.test.run_id,
                "training_run_id": member.training.run_id,
                "model_package_id": member.test.model_package_id,
                "source_config_sha256": member.training.source_config_sha256,
                "semantic_config_sha256": member.training.semantic_config_sha256,
                "metrics": dict(member.metrics),
            }
            for member in members
        ],
        "aggregates": {name: dict(values) for name, values in aggregates.items()},
        "thresholds": {
            "aggregation": "seed_specific_only",
            "policies": list(THRESHOLD_POLICIES),
        },
    }


def _write_reports(
    directory: Path,
    report: Mapping[str, Any],
    members: Sequence[_Member],
    aggregates: Mapping[str, Mapping[str, float]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_metrics_csv(directory / "metrics.csv", members, aggregates)
    (directory / "summary.md").write_text(
        _markdown_report(report, members, aggregates),
        encoding="utf-8",
    )


def _write_metrics_csv(
    path: Path,
    members: Sequence[_Member],
    aggregates: Mapping[str, Mapping[str, float]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "record_type",
                "seed",
                "test_run_id",
                "training_run_id",
                "model_package_id",
                "metric",
                "value",
            ]
        )
        for member in members:
            for metric in SEED_SPECIFIC_METRIC_NAMES:
                writer.writerow(
                    [
                        "member",
                        member.seed,
                        member.test.run_id,
                        member.training.run_id,
                        member.test.model_package_id,
                        metric,
                        _format_float(member.metrics[metric]),
                    ]
                )
        for metric in SEED_AGGREGATE_METRIC_NAMES:
            for statistic in ("mean", "sample_standard_deviation"):
                writer.writerow(
                    [
                        statistic,
                        "",
                        "",
                        "",
                        "",
                        metric,
                        _format_float(aggregates[metric][statistic]),
                    ]
                )


def _markdown_report(
    report: Mapping[str, Any],
    members: Sequence[_Member],
    aggregates: Mapping[str, Mapping[str, float]],
) -> str:
    lines = [
        "# RSNA image three-seed summary",
        "",
        f"- Report ID: `{report['report_id']}`",
        f"- Dataset: `{report['dataset']}`",
        f"- Task: `{report['task']}`",
        f"- Model: `{report['model']}`",
        f"- Compatibility SHA-256: `{report['compatibility_sha256']}`",
        "",
        "| Seed | Test run | Training run | Model package |",
        "| ---: | --- | --- | --- |",
    ]
    lines.extend(
        f"| {member.seed} | `{member.test.run_id}` | `{member.training.run_id}` | "
        f"`{member.test.model_package_id}` |"
        for member in members
    )
    lines.extend(
        [
            "",
            "| Metric | Seed 17 | Seed 42 | Seed 2026 | Mean | Sample SD |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in SEED_SPECIFIC_METRIC_NAMES:
        values = [_format_float(member.metrics[metric]) for member in members]
        aggregate = aggregates.get(metric)
        mean = _format_float(aggregate["mean"]) if aggregate is not None else ""
        deviation = (
            _format_float(aggregate["sample_standard_deviation"]) if aggregate is not None else ""
        )
        lines.append(
            f"| `{metric}` | {values[0]} | {values[1]} | {values[2]} | {mean} | {deviation} |"
        )
    lines.extend(
        [
            "",
            "Validation-selected thresholds remain seed-specific. This report does not select a "
            "canonical seed, average model weights, or define a deployment threshold.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_report_set(directory: Path) -> None:
    entries = list(directory.iterdir())
    if {entry.name for entry in entries} != SEED_SUMMARY_FILENAMES:
        raise ValueError("Seed summary report set is incomplete")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("Seed summary reports must be regular non-symlink files")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _same_number(value: object, expected: float) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    )


def _format_float(value: float) -> str:
    return f"{value:.10f}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-run-ids", nargs="+", required=True)
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow SQLite tracking URI",
    )
    parser.add_argument("--output-directory", type=Path, default=Path("reports"))
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate explicit members, publish their summary, and print its identity."""
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        with timed_phase(_LOGGER, "seed_summary_generation"):
            result = summarize_seed_runs(
                args.test_run_ids,
                tracking_uri=args.tracking_uri,
                output_directory=args.output_directory,
            )
    except (MlflowException, SQLAlchemyError, OSError, ValueError, KeyError) as exc:
        print(f"Seed summary failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "report_id": result.report_id,
                "report_directory": result.report_directory.as_posix(),
                "compatibility_sha256": result.compatibility_sha256,
                "test_run_ids": list(result.test_run_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
