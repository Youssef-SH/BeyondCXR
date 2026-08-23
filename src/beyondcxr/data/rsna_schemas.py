"""Define Arrow contracts for RSNA data artifacts."""

from __future__ import annotations

import pyarrow as pa

DATASET_ID = "rsna"
PNEUMONIA_TASK_ID = "pneumonia"
PNEUMONIA_LABEL_SOURCE = "rsna-stage-2-challenge-target"
PNEUMONIA_LABEL_POLICY_VERSION = "rsna-stage-2-target-v1"
RSNA_CLASS_TASK_ID = "rsna_class"
RSNA_CLASS_LABEL_SOURCE = "rsna-stage-2-detailed-class-info"
RSNA_CLASS_LABEL_POLICY_VERSION = "rsna-stage-2-class-v1"

RSNA_SAMPLE_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("patient_id", pa.string(), nullable=False),
        pa.field("image_id", pa.string(), nullable=False),
        pa.field("image_path", pa.string(), nullable=False),
        pa.field("image_rows", pa.int32(), nullable=False),
        pa.field("image_columns", pa.int32(), nullable=False),
        pa.field("age_years", pa.float64(), nullable=True),
        pa.field("age_is_implausible", pa.bool_(), nullable=False),
        pa.field("sex", pa.string(), nullable=True),
        pa.field("view_position", pa.string(), nullable=True),
        pa.field("pixel_spacing_row_mm", pa.float64(), nullable=True),
        pa.field("pixel_spacing_col_mm", pa.float64(), nullable=True),
    ]
)

RSNA_LABEL_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("label_value", pa.int8(), nullable=False),
    ]
)

RSNA_ANNOTATION_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("annotation_id", pa.string(), nullable=False),
        pa.field("x", pa.float64(), nullable=False),
        pa.field("y", pa.float64(), nullable=False),
        pa.field("width", pa.float64(), nullable=False),
        pa.field("height", pa.float64(), nullable=False),
    ]
)

RSNA_SPLIT_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("split_name", pa.string(), nullable=False),
    ]
)

RSNA_SOURCE_INVENTORY_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("relative_path", pa.string(), nullable=False),
        pa.field("byte_size", pa.int64(), nullable=False),
        pa.field("sha256", pa.string(), nullable=False),
    ]
)


def require_exact_schema(table: pa.Table, expected: pa.Schema, artifact_name: str) -> None:
    """Reject missing, extra, reordered, or physically different Arrow fields."""
    if table.schema != expected:
        raise ValueError(
            f"{artifact_name} schema mismatch. Expected {expected}, received {table.schema}"
        )
