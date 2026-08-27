"""Regenerate deterministic RSNA comparison views from evaluation results."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from beyondcxr.training.rsna_evaluation_result import validate_rsna_evaluation
from beyondcxr.utils.operational_logging import add_logging_argument, configure_logging
from beyondcxr.utils.package_identity import canonical_scientific_id
from beyondcxr.utils.publication import (
    install_immutable_directory,
    staging_directory,
    validate_path_component,
)

COMPARISON_COLUMNS = (
    "dataset_id",
    "task_id",
    "family_id",
    "modalities",
    "seed",
    "bundle_id",
    "split_assignment_id",
    "evaluation_id",
    "model_package_id",
    "average_precision",
    "roc_auc",
    "brier_score",
    "expected_calibration_error",
    "calibration_slope",
    "calibration_intercept",
    "youden_j_threshold",
    "youden_j_precision",
    "youden_j_recall",
    "youden_j_specificity",
    "youden_j_f1",
    "target_sensitivity_threshold",
    "target_sensitivity_precision",
    "target_sensitivity_recall",
    "target_sensitivity_specificity",
    "target_sensitivity_f1",
)
COMPARISON_SCHEMA_VERSION = 1
COMPARISON_POLICY_VERSION = 1
COMPARISON_PREFIX = "comparison-"
COMPARISON_FILENAMES = frozenset({"manifest.json", "table.csv", "table.md"})


@dataclass(frozen=True)
class ComparisonResult:
    """One immutable comparison view over explicit evaluation authorities."""

    comparison_id: str
    directory: Path
    csv_path: Path
    markdown_path: Path
    row_count: int
    manifest: Mapping[str, Any]


def regenerate_comparison(
    evaluation_ids: Sequence[str],
    *,
    output_directory: str | Path = "reports",
    private_directory: str | Path = "private",
    model_directory: str | Path = "models/rsna",
) -> ComparisonResult:
    """Publish an immutable comparison over exact validated evaluation identities."""
    identities = tuple(evaluation_ids)
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("Comparison requires unique explicit evaluation IDs")
    output = Path(output_directory)
    records = _comparison_records(
        identities,
        output_directory=output,
        private_directory=private_directory,
        model_directory=model_directory,
    )
    ordered_ids = [str(record["evaluation_id"]) for record in records]
    comparison_id = canonical_scientific_id(
        COMPARISON_PREFIX,
        {
            "comparison_policy_version": COMPARISON_POLICY_VERSION,
            "evaluation_ids": ordered_ids,
            "columns": list(COMPARISON_COLUMNS),
        },
    )
    document = {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "comparison_policy_version": COMPARISON_POLICY_VERSION,
        "comparison_id": comparison_id,
        "evaluation_ids": ordered_ids,
        "columns": list(COMPARISON_COLUMNS),
        "row_count": len(records),
    }
    destination = output / "rsna" / "comparisons" / comparison_id
    stage = staging_directory(destination)
    try:
        _write_comparison(stage, document, records)
        install_immutable_directory(
            stage,
            destination,
            lambda path, **kwargs: validate_comparison(
                path,
                report_root=output,
                private_root=private_directory,
                model_root=model_directory,
                **kwargs,
            ),
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return validate_comparison(
        destination,
        report_root=output,
        private_root=private_directory,
        model_root=model_directory,
        expected_comparison_id=comparison_id,
    )


def validate_comparison(
    directory: str | Path,
    *,
    report_root: str | Path,
    private_root: str | Path,
    model_root: str | Path,
    expected_comparison_id: str | None = None,
    enforce_directory_name: bool = True,
) -> ComparisonResult:
    """Revalidate membership and deterministic renderings of one comparison."""
    root = Path(directory)
    if (
        root.is_symlink()
        or not root.is_dir()
        or {path.name for path in root.iterdir()} != set(COMPARISON_FILENAMES)
    ):
        raise ValueError("RSNA comparison artifact set is invalid")
    if any(path.is_symlink() or not path.is_file() for path in root.iterdir()):
        raise ValueError("RSNA comparison artifact set is invalid")
    raw = (root / "manifest.json").read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("RSNA comparison manifest is unreadable") from exc
    fields = {
        "comparison_schema_version",
        "comparison_policy_version",
        "comparison_id",
        "evaluation_ids",
        "columns",
        "row_count",
    }
    if (
        not isinstance(document, dict)
        or set(document) != fields
        or type(document["comparison_schema_version"]) is not int
        or document["comparison_schema_version"] != COMPARISON_SCHEMA_VERSION
        or type(document["comparison_policy_version"]) is not int
        or document["comparison_policy_version"] != COMPARISON_POLICY_VERSION
        or document["columns"] != list(COMPARISON_COLUMNS)
        or not isinstance(document["evaluation_ids"], list)
        or not document["evaluation_ids"]
        or any(not isinstance(value, str) for value in document["evaluation_ids"])
        or len(document["evaluation_ids"]) != len(set(document["evaluation_ids"]))
        or type(document["row_count"]) is not int
        or document["row_count"] != len(document["evaluation_ids"])
        or raw != (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    ):
        raise ValueError("RSNA comparison manifest contract is invalid")
    records = _comparison_records(
        tuple(document["evaluation_ids"]),
        output_directory=Path(report_root),
        private_directory=private_root,
        model_directory=model_root,
    )
    ordered_ids = [str(record["evaluation_id"]) for record in records]
    expected = canonical_scientific_id(
        COMPARISON_PREFIX,
        {
            "comparison_policy_version": COMPARISON_POLICY_VERSION,
            "evaluation_ids": ordered_ids,
            "columns": list(COMPARISON_COLUMNS),
        },
    )
    if (
        document["evaluation_ids"] != ordered_ids
        or document["comparison_id"] != expected
        or (expected_comparison_id is not None and expected != expected_comparison_id)
        or (enforce_directory_name and root.name != expected)
    ):
        raise ValueError("RSNA comparison identity is invalid")
    with tempfile.TemporaryDirectory(prefix="beyondcxr-comparison-validation-") as temporary:
        rendered = Path(temporary)
        _write_comparison(rendered, document, records)
        if any(
            (root / filename).read_bytes() != (rendered / filename).read_bytes()
            for filename in COMPARISON_FILENAMES
        ):
            raise ValueError("RSNA comparison rendering differs from its authorities")
    return ComparisonResult(
        expected,
        root,
        root / "table.csv",
        root / "table.md",
        len(records),
        document,
    )


def _comparison_records(
    identities: Sequence[str],
    *,
    output_directory: Path,
    private_directory: str | Path,
    model_directory: str | Path,
) -> list[dict[str, object]]:
    records = []
    for evaluation_id in identities:
        validate_path_component(evaluation_id, "evaluation ID")
        result = validate_rsna_evaluation(
            output_directory / "rsna/evaluations" / evaluation_id,
            private_root=private_directory,
            model_root=model_directory,
            expected_evaluation_id=evaluation_id,
        )
        document = result.manifest
        probability = document["claims"]["probability_metrics"]
        operating = document["claims"]["operating_points"]
        records.append(
            {
                "dataset_id": document["dataset_id"],
                "task_id": document["task_id"],
                "family_id": document["family_id"],
                "modalities": ",".join(document["modalities"]),
                "seed": document["seed"],
                "bundle_id": document["bundle_id"],
                "split_assignment_id": document["split_assignment_id"],
                "evaluation_id": evaluation_id,
                "model_package_id": document["model_package_id"],
                **{key: probability[key] for key in COMPARISON_COLUMNS if key in probability},
                **_operating_columns(operating),
            }
        )
    records.sort(key=lambda item: (item["family_id"], item["seed"], item["evaluation_id"]))
    return records


def _write_comparison(
    directory: Path,
    document: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame.from_records(records, columns=COMPARISON_COLUMNS)
    _atomic_csv(table, directory / "table.csv")
    _atomic_text(directory / "table.md", "# Evaluation comparison\n\n" + _markdown_table(table))
    _atomic_text(
        directory / "manifest.json",
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _markdown_table(table: pd.DataFrame) -> str:
    """Render the fixed comparison table without optional dependencies."""
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = []
    for values in table.itertuples(index=False, name=None):
        cells = [
            (f"{value:.10g}" if isinstance(value, float) else str(value)).replace("|", "\\|")
            for value in values
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join((header, separator, *rows)) + "\n"


def _operating_columns(value: dict[str, object]) -> dict[str, object]:
    result = {}
    for policy in ("youden_j", "target_sensitivity"):
        point = value[policy]
        metrics = point["metrics"]
        for metric in ("threshold", "precision", "recall", "specificity", "f1"):
            result[f"{policy}_{metric}"] = (
                point["threshold"] if metric == "threshold" else metrics[metric]
            )
    return result


def _atomic_csv(table: pd.DataFrame, path: Path) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            table.to_csv(stream, index=False, float_format="%.10f", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-ids", nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("reports"))
    parser.add_argument("--private-directory", type=Path, default=Path("private"))
    parser.add_argument("--model-directory", type=Path, default=Path("models/rsna"))
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        result = regenerate_comparison(
            args.evaluation_ids,
            output_directory=args.output_directory,
            private_directory=args.private_directory,
            model_directory=args.model_directory,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Comparison failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"Wrote {result.row_count} rows to {result.csv_path} and {result.markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
