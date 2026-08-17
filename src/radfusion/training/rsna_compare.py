"""Regenerate deterministic RSNA comparison views from evaluation results."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from radfusion.training.rsna_evaluation_result import validate_rsna_evaluation
from radfusion.utils.operational_logging import add_logging_argument, configure_logging
from radfusion.utils.publication import validate_path_component

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


def regenerate_comparison(
    evaluation_ids: Sequence[str],
    *,
    output_directory: str | Path = "reports",
    private_directory: str | Path = "private",
    model_directory: str | Path = "models/rsna",
) -> tuple[Path, Path, int]:
    """Write comparison views for exact validated evaluation identities."""
    identities = tuple(evaluation_ids)
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("Comparison requires unique explicit evaluation IDs")
    output = Path(output_directory)
    records = []
    for evaluation_id in identities:
        validate_path_component(evaluation_id, "evaluation ID")
        result = validate_rsna_evaluation(
            output / "rsna/evaluations" / evaluation_id,
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
    table = pd.DataFrame.from_records(records, columns=COMPARISON_COLUMNS)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "model_comparison_table.csv"
    markdown_path = output / "model_comparison_table.md"
    _atomic_csv(table, csv_path)
    _atomic_text(
        markdown_path,
        "# Evaluation comparison\n\n" + _markdown_table(table),
    )
    return csv_path, markdown_path, len(table)


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
        csv_path, markdown_path, count = regenerate_comparison(
            args.evaluation_ids,
            output_directory=args.output_directory,
            private_directory=args.private_directory,
            model_directory=args.model_directory,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Comparison failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"Wrote {count} rows to {csv_path} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
