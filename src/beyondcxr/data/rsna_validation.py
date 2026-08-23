"""Validate relationships across RSNA sample, label, and annotation artifacts."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path, PurePosixPath

import pyarrow as pa

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.rsna_dicom import (
    ALLOWED_SEX,
    ALLOWED_VIEW_POSITIONS,
    validate_spacing_pair,
)
from beyondcxr.data.rsna_schemas import (
    PNEUMONIA_TASK_ID,
    RSNA_ANNOTATION_SCHEMA,
    RSNA_CLASS_TASK_ID,
    RSNA_LABEL_SCHEMA,
    RSNA_SAMPLE_SCHEMA,
    require_exact_schema,
)
from beyondcxr.data.rsna_source import (
    RSNA_CLASS_VALUES,
    canonical_image_path,
    resolve_image_path,
)


def validate_sample_table(samples: pa.Table, dataset_root: str | Path | None = None) -> None:
    """Validate the task-agnostic sample contract and deterministic paths."""
    _require_schema(samples, RSNA_SAMPLE_SCHEMA, "RSNA samples")
    rows = samples.to_pylist()
    if not rows:
        raise ManifestBuildError("RSNA samples artifact must not be empty")
    sample_ids = [row["sample_id"] for row in rows]
    image_ids = [row["image_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ManifestBuildError("sample_id must be unique")
    if len(image_ids) != len(set(image_ids)):
        raise ManifestBuildError("image_id must be unique within the RSNA dataset")
    if sample_ids != sorted(sample_ids):
        raise ManifestBuildError("RSNA samples must be ordered by sample_id")

    root = Path(dataset_root).resolve() if dataset_root is not None else None
    for row in rows:
        if not row["patient_id"] or not row["image_id"]:
            raise ManifestBuildError("patient_id and image_id must be non-null and non-empty")
        if row["sample_id"] != f"rsna:{row['image_id']}":
            raise ManifestBuildError("RSNA sample_id must be 'rsna:<image_id>'")
        if row["image_rows"] <= 0 or row["image_columns"] <= 0:
            raise ManifestBuildError("image_rows and image_columns must be positive")
        age = row["age_years"]
        if age is not None and not math.isfinite(age):
            raise ManifestBuildError("age_years must be finite when present")
        if row["age_is_implausible"] != (age is not None and (age < 0.0 or age > 120.0)):
            raise ManifestBuildError("age_is_implausible does not match age_years")
        if row["sex"] is not None and row["sex"] not in ALLOWED_SEX:
            raise ManifestBuildError(f"Invalid sex value: {row['sex']!r}")
        if row["view_position"] is not None and row["view_position"] not in ALLOWED_VIEW_POSITIONS:
            raise ManifestBuildError(f"Invalid view_position value: {row['view_position']!r}")
        validate_spacing_pair(row["pixel_spacing_row_mm"], row["pixel_spacing_col_mm"])
        relative_path = _validated_relative_path(row["image_path"])
        expected_path = canonical_image_path(row["image_id"])
        if relative_path != expected_path:
            raise ManifestBuildError(
                f"RSNA image_path must match image_id; expected {expected_path.as_posix()!r}"
            )
        if root is not None and not resolve_image_path(root, relative_path).is_file():
            raise ManifestBuildError(f"Sample image does not exist: {row['image_path']}")


def validate_label_table(labels: pa.Table, samples: pa.Table) -> None:
    """Validate task-label cardinality, domains, and source compatibility."""
    _require_schema(labels, RSNA_LABEL_SCHEMA, "RSNA labels")
    sample_ids = set(samples.column("sample_id").to_pylist())
    rows = labels.to_pylist()
    keys = [(row["sample_id"], row["task_id"]) for row in rows]
    if keys != sorted(keys):
        raise ManifestBuildError("RSNA labels must be ordered by sample_id and task_id")
    if len(keys) != len(set(keys)):
        raise ManifestBuildError("Each sample/task pair must have exactly one label row")

    by_sample: dict[str, dict[str, int]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        task_id = row["task_id"]
        if sample_id not in sample_ids:
            raise ManifestBuildError(f"Label references unknown sample_id {sample_id!r}")
        if task_id == PNEUMONIA_TASK_ID:
            values = {0, 1}
        elif task_id == RSNA_CLASS_TASK_ID:
            values = set(RSNA_CLASS_VALUES.values())
        else:
            raise ManifestBuildError(f"Unexpected task_id {task_id!r}")
        if row["label_value"] not in values:
            raise ManifestBuildError(
                f"Invalid label_value {row['label_value']!r} for task {task_id!r}"
            )
        by_sample.setdefault(sample_id, {})[task_id] = row["label_value"]

    if set(by_sample) != sample_ids:
        missing = sorted(sample_ids - set(by_sample))
        raise ManifestBuildError(
            f"Samples without labels: {len(missing)} (examples: {missing[:10]})"
        )
    expected_tasks = {PNEUMONIA_TASK_ID, RSNA_CLASS_TASK_ID}
    for sample_id, tasks in by_sample.items():
        if set(tasks) != expected_tasks:
            raise ManifestBuildError(
                f"Sample {sample_id!r} must have exactly pneumonia and rsna_class labels"
            )
        pneumonia = tasks[PNEUMONIA_TASK_ID]
        rsna_class = tasks[RSNA_CLASS_TASK_ID]
        compatible = (pneumonia == 1 and rsna_class == RSNA_CLASS_VALUES["Lung Opacity"]) or (
            pneumonia == 0 and rsna_class != RSNA_CLASS_VALUES["Lung Opacity"]
        )
        if not compatible:
            raise ManifestBuildError(f"Target/class incompatibility for {sample_id!r}")


def pneumonia_targets(labels: pa.Table) -> dict[str, int]:
    """Return the validated sample-to-pneumonia label mapping."""
    return {
        row["sample_id"]: row["label_value"]
        for row in labels.to_pylist()
        if row["task_id"] == PNEUMONIA_TASK_ID
    }


def validate_annotation_table(
    annotations: pa.Table,
    samples: pa.Table,
    labels: pa.Table,
) -> None:
    """Validate annotation relationships, geometry, identifiers, and ordering."""
    _require_schema(annotations, RSNA_ANNOTATION_SCHEMA, "RSNA annotations")
    sample_rows = samples.to_pylist()
    sample_ids = {row["sample_id"] for row in sample_rows}
    sample_images = {row["sample_id"]: row["image_id"] for row in sample_rows}
    image_dimensions = {
        row["sample_id"]: (row["image_rows"], row["image_columns"]) for row in sample_rows
    }
    targets = pneumonia_targets(labels)
    if set(targets) != sample_ids:
        raise ManifestBuildError(
            "Pneumonia label coverage must match samples before annotation validation"
        )
    positive_ids = {sample_id for sample_id, target in targets.items() if target == 1}
    rows = annotations.to_pylist()
    annotation_ids = [row["annotation_id"] for row in rows]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ManifestBuildError("annotation_id must be unique")
    if annotation_ids != sorted(annotation_ids):
        raise ManifestBuildError("RSNA annotations must be ordered by annotation_id")

    annotated_ids: set[str] = set()
    annotation_indexes: Counter[str] = Counter()
    seen_boxes: set[tuple[str, float, float, float, float]] = set()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in sample_ids:
            raise ManifestBuildError(f"Annotation references unknown sample_id {sample_id!r}")
        if targets[sample_id] != 1:
            raise ManifestBuildError(f"Negative sample {sample_id!r} must not have annotations")
        expected_id = f"rsna:{sample_images[sample_id]}:bbox:{annotation_indexes[sample_id]:04d}"
        if row["annotation_id"] != expected_id:
            raise ManifestBuildError(
                f"Invalid deterministic annotation_id {row['annotation_id']!r}; "
                f"expected {expected_id!r}"
            )
        annotation_indexes[sample_id] += 1
        coordinates = tuple(float(row[name]) for name in ("x", "y", "width", "height"))
        if not all(math.isfinite(value) for value in coordinates):
            raise ManifestBuildError(
                f"Annotation {row['annotation_id']!r} has non-finite coordinates"
            )
        x, y, width, height = coordinates
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ManifestBuildError(f"Annotation {row['annotation_id']!r} has invalid geometry")
        image_rows, image_columns = image_dimensions[sample_id]
        if x + width > image_columns or y + height > image_rows:
            raise ManifestBuildError(f"Annotation {row['annotation_id']!r} exceeds image bounds")
        box_key = (sample_id, x, y, width, height)
        if box_key in seen_boxes:
            raise ManifestBuildError(f"Duplicate bounding box for sample {sample_id!r}")
        seen_boxes.add(box_key)
        annotated_ids.add(sample_id)
    if annotated_ids != positive_ids:
        missing = sorted(positive_ids - annotated_ids)
        raise ManifestBuildError(
            f"Every positive sample must have annotations; missing {len(missing)} "
            f"(examples: {missing[:10]})"
        )


def _require_schema(table: pa.Table, expected: pa.Schema, artifact_name: str) -> None:
    try:
        require_exact_schema(table, expected, artifact_name)
    except ValueError as exc:
        raise ManifestBuildError(str(exc)) from exc


def _validated_relative_path(value: str) -> PurePosixPath:
    if "\\" in value:
        raise ManifestBuildError(f"image_path must use POSIX separators: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ManifestBuildError(f"image_path must be a normalized relative POSIX path: {value!r}")
    return path
