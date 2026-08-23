"""Generate aggregate RSNA dataset audit reports from a validated bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.hashing import sha256_file
from beyondcxr.data.rsna_artifacts import resolve_bundle
from beyondcxr.data.rsna_source import RSNA_CLASS_VALUES
from beyondcxr.data.rsna_splitting import SPLIT_NAMES
from beyondcxr.utils.operational_logging import (
    add_logging_argument,
    configure_logging,
    get_operational_logger,
    timed_phase,
)
from beyondcxr.utils.privacy import validate_public_reports
from beyondcxr.utils.publication import publish_directory, staging_directory

REPORT_FILENAMES = (
    "dataset_summary.md",
    "split_summary.md",
    "label_distribution.csv",
    "age_distribution.csv",
    "sex_distribution.csv",
    "view_distribution.csv",
    "image_dimensions.csv",
    "pixel_spacing.csv",
    "bbox_statistics.csv",
    "missingness_report.md",
)
_AGE_LABELS = ("<18", "18-<40", "40-<60", "60-<80", "80-120", ">120")
_LOGGER = get_operational_logger(__name__)


def generate_rsna_audit(
    manifest_directory: str | Path = "data/manifests",
    output_directory: str | Path = "reports/rsna/audit",
    *,
    bundle_id: str | None = None,
) -> dict[str, object]:
    """Generate deterministic aggregate reports for one validated RSNA bundle."""
    with timed_phase(_LOGGER, "audit_generation"):
        return _generate_rsna_audit(manifest_directory, output_directory, bundle_id=bundle_id)


def _generate_rsna_audit(
    manifest_directory: str | Path,
    output_directory: str | Path,
    *,
    bundle_id: str | None,
) -> dict[str, object]:
    with timed_phase(_LOGGER, "audit_bundle_loading"):
        bundle = resolve_bundle(manifest_directory, bundle_id=bundle_id)
        samples = pq.read_table(bundle.samples_path)
        labels = pq.read_table(bundle.labels_path)
        annotations = pq.read_table(bundle.annotations_path)
        splits = pq.read_table(bundle.splits_path)
        metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))

    sample_frame = samples.to_pandas()
    label_frame = labels.to_pandas()
    annotation_frame = annotations.to_pandas()
    split_frame = splits.to_pandas()
    frame = _audit_frame(sample_frame, label_frame, split_frame)

    output = Path(output_directory) / bundle.bundle_id
    stage = staging_directory(output)
    try:
        tables = {
            "label_distribution.csv": _label_distribution(label_frame, split_frame),
            "age_distribution.csv": _age_distribution(frame),
            "sex_distribution.csv": _category_distribution(frame, "sex"),
            "view_distribution.csv": _category_distribution(frame, "view_position"),
            "image_dimensions.csv": _image_dimensions_distribution(frame),
            "pixel_spacing.csv": _pixel_spacing_distribution(frame),
            "bbox_statistics.csv": _bbox_statistics(frame, annotation_frame),
        }
        for filename, table in tables.items():
            table.to_csv(stage / filename, index=False, float_format="%.6f", lineterminator="\n")
        (stage / "dataset_summary.md").write_text(
            _dataset_summary(bundle.bundle_id, metadata, frame, annotation_frame), encoding="utf-8"
        )
        (stage / "split_summary.md").write_text(_split_summary(metadata, frame), encoding="utf-8")
        (stage / "missingness_report.md").write_text(_missingness_report(frame), encoding="utf-8")
        actual_reports = {path.name for path in stage.iterdir() if path.is_file()}
        if actual_reports != set(REPORT_FILENAMES):
            raise ValueError(
                f"RSNA audit staging set mismatch: expected={sorted(REPORT_FILENAMES)}, "
                f"actual={sorted(actual_reports)}"
            )
        validate_public_reports(
            [stage / filename for filename in REPORT_FILENAMES],
            forbidden_source_values=_source_identifiers(sample_frame),
        )
        publish_directory(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    return {
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_sha256": sha256_file(bundle.metadata_path),
        "split_recipe_id": str(metadata["membership"]["split"]["split_recipe_id"]),
        "split_assignment_id": str(metadata["membership"]["split"]["split_assignment_id"]),
        "report_directory": output.as_posix(),
        "reports": list(REPORT_FILENAMES),
    }


def _audit_frame(samples: pd.DataFrame, labels: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    pneumonia = labels.loc[labels["task_id"] == "pneumonia", ["sample_id", "label_value"]]
    pneumonia = pneumonia.rename(columns={"label_value": "target_pneumonia"})
    return (
        samples.merge(splits[["sample_id", "split_name"]], on="sample_id", validate="one_to_one")
        .merge(pneumonia, on="sample_id", validate="one_to_one")
        .sort_values("sample_id", kind="stable")
        .reset_index(drop=True)
    )


def _scopes(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    split_scopes = [(name, frame.loc[frame["split_name"] == name]) for name in SPLIT_NAMES]
    return [("overall", frame), *split_scopes]


def _label_distribution(labels: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    frame = labels.merge(
        splits[["sample_id", "split_name"]], on="sample_id", validate="many_to_one"
    )
    class_names = {value: name for name, value in RSNA_CLASS_VALUES.items()}
    records: list[dict[str, object]] = []
    for scope, scoped in _scopes(frame):
        for (task_id, value), group in scoped.groupby(["task_id", "label_value"], sort=True):
            task_total = int((scoped["task_id"] == task_id).sum())
            if task_id == "pneumonia":
                label_name = "positive" if int(value) == 1 else "negative"
            else:
                label_name = class_names[int(value)]
            records.append(
                {
                    "scope": scope,
                    "task_id": task_id,
                    "label_value": int(value),
                    "label_name": label_name,
                    "count": len(group),
                    "fraction": len(group) / task_total,
                }
            )
    return pd.DataFrame.from_records(records)


def _age_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scope, scoped in _scopes(frame):
        ages = scoped["age_years"]
        counts = {
            "<18": int((ages < 18.0).sum()),
            "18-<40": int(((ages >= 18.0) & (ages < 40.0)).sum()),
            "40-<60": int(((ages >= 40.0) & (ages < 60.0)).sum()),
            "60-<80": int(((ages >= 60.0) & (ages < 80.0)).sum()),
            "80-120": int(((ages >= 80.0) & (ages <= 120.0)).sum()),
            ">120": int((ages > 120.0).sum()),
        }
        for band in _AGE_LABELS:
            records.append({"scope": scope, "age_band": band, "count": counts[band]})
        records.append(
            {
                "scope": scope,
                "age_band": "<missing>",
                "count": int(scoped["age_years"].isna().sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def _category_distribution(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scope, scoped in _scopes(frame):
        values = scoped[column].fillna("<missing>").astype(str)
        for value, count in values.value_counts().sort_index().items():
            records.append({"scope": scope, column: value, "count": int(count)})
    return pd.DataFrame.from_records(records)


def _pixel_spacing_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scope, scoped in _scopes(frame):
        pairs = scoped[["pixel_spacing_row_mm", "pixel_spacing_col_mm"]].copy()
        grouped = pairs.value_counts(dropna=False, sort=False).sort_index()
        for (row_spacing, col_spacing), count in grouped.items():
            records.append(
                {
                    "scope": scope,
                    "pixel_spacing_row_mm": row_spacing,
                    "pixel_spacing_col_mm": col_spacing,
                    "count": int(count),
                }
            )
    return pd.DataFrame.from_records(records)


def _image_dimensions_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for scope, scoped in _scopes(frame):
        grouped = scoped[["image_rows", "image_columns"]].value_counts(sort=False).sort_index()
        for (rows, columns), count in grouped.items():
            records.append(
                {
                    "scope": scope,
                    "image_rows": int(rows),
                    "image_columns": int(columns),
                    "count": int(count),
                }
            )
    return pd.DataFrame.from_records(records)


def _bbox_statistics(frame: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    box_counts = annotations.groupby("sample_id").size().rename("bbox_count")
    with_boxes = frame.join(box_counts, on="sample_id").fillna({"bbox_count": 0})
    with_boxes["bbox_count"] = with_boxes["bbox_count"].astype(int)
    records: list[dict[str, object]] = []
    for scope, scoped in _scopes(with_boxes):
        positive = scoped.loc[scoped["target_pneumonia"] == 1, "bbox_count"]
        has_positive = not positive.empty
        records.append(
            {
                "scope": scope,
                "sample_count": len(scoped),
                "positive_sample_count": len(positive),
                "annotation_count": int(positive.sum()),
                "mean_boxes_per_positive": float(positive.mean()) if has_positive else None,
                "median_boxes_per_positive": float(positive.median()) if has_positive else None,
                "min_boxes_per_positive": int(positive.min()) if has_positive else None,
                "max_boxes_per_positive": int(positive.max()) if has_positive else None,
                "positive_samples_one_box": int((positive == 1).sum()),
                "positive_samples_multiple_boxes": int((positive > 1).sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def _dataset_summary(
    bundle_id: str,
    metadata: dict[str, object],
    frame: pd.DataFrame,
    annotations: pd.DataFrame,
) -> str:
    positive = int(frame["target_pneumonia"].sum())
    lines = [
        "# RSNA dataset summary",
        "",
        f"- Bundle ID: `{bundle_id}`",
        f"- Samples: {len(frame):,}",
        f"- Patients: {frame['patient_id'].nunique():,}",
        f"- Positive samples: {positive:,}",
        f"- Negative samples: {len(frame) - positive:,}",
        f"- Bounding boxes: {len(annotations):,}",
        "- Implausible ages above 120 years: "
        f"{int(metadata['qualification']['age_parsing']['implausible_age_count']):,}",
        "",
        "The benchmark endpoint is the radiology-derived RSNA challenge target.",
        "",
    ]
    return "\n".join(lines)


def _split_summary(metadata: dict[str, object], frame: pd.DataFrame) -> str:
    split_metadata = metadata["membership"]["split"]
    lines = [
        "# RSNA split summary",
        "",
        f"Split recipe ID: `{split_metadata['split_recipe_id']}`",
        f"Split assignment ID: `{split_metadata['split_assignment_id']}`",
        "",
        "| Split | Samples | Patients | Positive | Negative | Positive rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in SPLIT_NAMES:
        scoped = frame.loc[frame["split_name"] == split_name]
        positive = int(scoped["target_pneumonia"].sum())
        lines.append(
            f"| {split_name} | {len(scoped):,} | {scoped['patient_id'].nunique():,} | "
            f"{positive:,} | {len(scoped) - positive:,} | {positive / len(scoped):.6f} |"
        )
    lines.extend(
        [
            "",
            "Validation guarantees:",
            "",
            "- Patient overlap: 0",
            "- Sample overlap: 0",
            f"- Covered samples: {len(frame):,} of {len(frame):,}",
            "",
        ]
    )
    return "\n".join(lines)


def _missingness_report(frame: pd.DataFrame) -> str:
    columns = (
        "age_years",
        "sex",
        "view_position",
        "pixel_spacing_row_mm",
        "pixel_spacing_col_mm",
    )
    lines = [
        "# RSNA metadata missingness",
        "",
        "| Scope | Field | Missing | Total | Missing rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for scope, scoped in _scopes(frame):
        for column in columns:
            missing = int(scoped[column].isna().sum())
            lines.append(
                f"| {scope} | `{column}` | {missing:,} | {len(scoped):,} | "
                f"{missing / len(scoped):.6f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _source_identifiers(samples: pd.DataFrame) -> set[str]:
    identifiers: set[str] = set()
    for column in ("patient_id", "sample_id", "image_id", "image_path"):
        identifiers.update(str(value) for value in samples[column].dropna())
    identifiers.update(Path(value).name for value in samples["image_path"].dropna())
    identifiers.update(Path(value).stem for value in samples["image_path"].dropna())
    return identifiers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-id",
        help="Immutable bundle identifier; omit only to resolve CURRENT interactively",
    )
    parser.add_argument(
        "--manifest-directory",
        type=Path,
        default=Path("data/manifests"),
        help="Root directory containing RSNA bundles",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("reports/rsna/audit"),
        help="Root directory for bundle-specific audit reports",
    )
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate reports and print their bundle lineage."""
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        summary = generate_rsna_audit(
            args.manifest_directory,
            args.output_directory,
            bundle_id=args.bundle_id,
        )
    except (ManifestBuildError, OSError, ValueError, KeyError) as exc:
        print(f"RSNA audit failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
