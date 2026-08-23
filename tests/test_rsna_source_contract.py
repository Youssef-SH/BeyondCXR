from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from rsna_manifest_test_support import replace_table_row as _replace_table_row
from rsna_manifest_test_support import tables as _tables
from rsna_manifest_test_support import write_header as _write_header
from rsna_manifest_test_support import write_sources as _write_sources

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.rsna_artifacts import (
    build_rsna_artifacts,
)
from beyondcxr.data.rsna_schemas import (
    PNEUMONIA_TASK_ID,
    RSNA_ANNOTATION_SCHEMA,
    RSNA_CLASS_TASK_ID,
    RSNA_LABEL_SCHEMA,
    RSNA_SAMPLE_SCHEMA,
    RSNA_SOURCE_INVENTORY_SCHEMA,
    RSNA_SPLIT_SCHEMA,
)
from beyondcxr.data.rsna_source import aggregate_labels
from beyondcxr.data.rsna_validation import (
    validate_annotation_table,
    validate_label_table,
    validate_sample_table,
)


def test_happy_path_builds_canonical_samples_and_multiple_annotations(tmp_path: Path) -> None:
    root, result = _tables(tmp_path)

    assert result.samples.schema == RSNA_SAMPLE_SCHEMA
    assert result.labels.schema == RSNA_LABEL_SCHEMA
    assert result.annotations.schema == RSNA_ANNOTATION_SCHEMA
    assert result.splits.schema == RSNA_SPLIT_SCHEMA
    assert result.source_inventory.schema == RSNA_SOURCE_INVENTORY_SCHEMA
    assert result.samples.num_rows == 2
    assert result.labels.num_rows == 4
    assert result.annotations.num_rows == 2
    assert result.splits.num_rows == 2
    assert result.source_inventory.num_rows == 2
    positive = result.samples.to_pylist()[1]
    assert positive["sample_id"] == "rsna:positive"
    assert positive["image_rows"] == 1024
    assert positive["image_columns"] == 1024
    assert positive["age_is_implausible"] is False
    assert positive["image_path"] == "stage_2_train_images/positive.dcm"
    assert not Path(positive["image_path"]).is_absolute()
    validate_sample_table(result.samples, root)
    validate_label_table(result.labels, result.samples)

    positive_labels = {
        row["task_id"]: row["label_value"]
        for row in result.labels.to_pylist()
        if row["sample_id"] == "rsna:positive"
    }
    assert positive_labels == {PNEUMONIA_TASK_ID: 1, RSNA_CLASS_TASK_ID: 2}


def test_final_rsna_schemas_contain_only_row_level_facts() -> None:
    assert RSNA_SAMPLE_SCHEMA.names == [
        "sample_id",
        "patient_id",
        "image_id",
        "image_path",
        "image_rows",
        "image_columns",
        "age_years",
        "age_is_implausible",
        "sex",
        "view_position",
        "pixel_spacing_row_mm",
        "pixel_spacing_col_mm",
    ]
    assert RSNA_LABEL_SCHEMA.names == ["sample_id", "task_id", "label_value"]
    assert RSNA_ANNOTATION_SCHEMA.names == [
        "sample_id",
        "annotation_id",
        "x",
        "y",
        "width",
        "height",
    ]
    assert RSNA_SPLIT_SCHEMA.names == ["sample_id", "split_name"]
    assert RSNA_SOURCE_INVENTORY_SCHEMA.names == [
        "sample_id",
        "relative_path",
        "byte_size",
        "sha256",
    ]


def test_all_negative_fixture_has_empty_typed_annotations(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted", positive=False)
    result = build_rsna_artifacts(root)

    assert result.samples.num_rows == 1
    assert result.labels.num_rows == 2
    assert result.annotations.num_rows == 0
    assert result.annotations.schema == RSNA_ANNOTATION_SCHEMA


@pytest.mark.parametrize("column", ["patientId", "x", "Target"])
def test_missing_label_column_fails(tmp_path: Path, column: str) -> None:
    root = _write_sources(tmp_path / "extracted")
    labels_path = root / "stage_2_train_labels.csv"
    pd.read_csv(labels_path).drop(columns=column).to_csv(labels_path, index=False)

    with pytest.raises(ManifestBuildError):
        build_rsna_artifacts(root)


def test_missing_class_column_fails(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    class_path = root / "stage_2_detailed_class_info.csv"
    pd.read_csv(class_path).drop(columns="class").to_csv(class_path, index=False)

    with pytest.raises(ManifestBuildError):
        build_rsna_artifacts(root)


def test_inconsistent_targets_fail(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    labels_path = root / "stage_2_train_labels.csv"
    labels = pd.read_csv(labels_path)
    labels.loc[labels.index[-1], "Target"] = 0
    labels.to_csv(labels_path, index=False)

    with pytest.raises(ManifestBuildError):
        build_rsna_artifacts(root)


def test_inconsistent_classes_fail(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    class_path = root / "stage_2_detailed_class_info.csv"
    classes = pd.read_csv(class_path)
    classes.loc[classes.index[-1], "class"] = "Normal"
    classes.to_csv(class_path, index=False)

    with pytest.raises(ManifestBuildError):
        build_rsna_artifacts(root)


@pytest.mark.parametrize("missing_from", ["labels", "classes", "dicoms"])
def test_source_identifier_mismatches_fail(tmp_path: Path, missing_from: str) -> None:
    root = _write_sources(tmp_path / "extracted")
    if missing_from == "labels":
        path = root / "stage_2_train_labels.csv"
        frame = pd.read_csv(path)
        frame[frame["patientId"] != "positive"].to_csv(path, index=False)
    elif missing_from == "classes":
        path = root / "stage_2_detailed_class_info.csv"
        frame = pd.read_csv(path)
        frame[frame["patientId"] != "positive"].to_csv(path, index=False)
    else:
        (root / "stage_2_train_images" / "positive.dcm").unlink()

    with pytest.raises(ManifestBuildError):
        build_rsna_artifacts(root)


def test_extra_dicom_without_label_fails(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    _write_header(root / "stage_2_train_images" / "extra.dcm", "extra")

    with pytest.raises(ManifestBuildError):
        build_rsna_artifacts(root)


def test_duplicate_image_identifier_fails(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    _write_header(root / "stage_2_train_images" / "positive.DCM", "positive")

    with pytest.raises(ManifestBuildError):
        build_rsna_artifacts(root)


def test_missing_dicom_patient_id_fails(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    _write_header(root / "stage_2_train_images" / "positive.dcm", None)

    with pytest.raises(ManifestBuildError):
        build_rsna_artifacts(root)


def test_filename_dicom_patient_id_mismatch_fails(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    _write_header(root / "stage_2_train_images" / "positive.dcm", "different")

    with pytest.raises(ManifestBuildError):
        build_rsna_artifacts(root)


def test_unreadable_dicom_fails_clearly(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    (root / "stage_2_train_images" / "positive.dcm").write_bytes(b"not a dicom")

    with pytest.raises(ManifestBuildError):
        build_rsna_artifacts(root)


def test_missing_optional_metadata_is_preserved_as_null(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted", age=None, sex=None, view=None, spacing=None)
    result = build_rsna_artifacts(root)
    positive = result.samples.to_pylist()[1]

    assert positive["age_years"] is None
    assert positive["sex"] is None
    assert positive["view_position"] is None
    assert positive["pixel_spacing_row_mm"] is None
    assert result.metadata["qualification"]["age_parsing"]["status_counts"]["missing"] == 1


def test_malformed_and_implausible_ages_are_reported_in_aggregate(tmp_path: Path) -> None:
    with pytest.warns(UserWarning):
        malformed_root = _write_sources(tmp_path / "malformed", age="BAD")
    malformed = build_rsna_artifacts(malformed_root)
    assert malformed.samples.to_pylist()[1]["age_years"] is None
    assert malformed.samples.to_pylist()[1]["age_is_implausible"] is False
    assert malformed.metadata["qualification"]["age_parsing"]["status_counts"]["malformed"] == 1

    with pytest.warns(UserWarning):
        implausible_root = _write_sources(tmp_path / "implausible", age="155")
    implausible = build_rsna_artifacts(implausible_root)
    assert implausible.samples.to_pylist()[1]["age_years"] == 155.0
    assert implausible.samples.to_pylist()[1]["age_is_implausible"] is True
    assert implausible.metadata["qualification"]["age_parsing"]["implausible_age_count"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("sex", "X"), ("view", "LL")],
)
def test_invalid_categories_fail(tmp_path: Path, field: str, value: str) -> None:
    kwargs = {field: value}
    root = _write_sources(tmp_path / "extracted", **kwargs)

    with pytest.raises(ManifestBuildError):
        build_rsna_artifacts(root)


def test_target_class_incompatibility_fails(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    class_path = root / "stage_2_detailed_class_info.csv"
    classes = pd.read_csv(class_path)
    classes.loc[classes["patientId"] == "positive", "class"] = "Normal"
    classes.to_csv(class_path, index=False)

    with pytest.raises(ManifestBuildError):
        build_rsna_artifacts(root)


def _labels_frame(boxes: list[dict[str, object]], target: int = 1) -> pd.DataFrame:
    return pd.DataFrame([{"patientId": "sample", "Target": target, **box} for box in boxes])


def test_partial_positive_box_fails() -> None:
    frame = _labels_frame([{"x": 1, "y": 2, "width": 3, "height": None}])
    with pytest.raises(ManifestBuildError):
        aggregate_labels(frame)


def test_coordinates_on_negative_fail() -> None:
    frame = _labels_frame([{"x": 1, "y": None, "width": None, "height": None}], target=0)
    with pytest.raises(ManifestBuildError):
        aggregate_labels(frame)


@pytest.mark.parametrize(
    "box",
    [
        {"x": -1, "y": 0, "width": 1, "height": 1},
        {"x": 0, "y": -1, "width": 1, "height": 1},
        {"x": 0, "y": 0, "width": 0, "height": 1},
        {"x": 0, "y": 0, "width": 1, "height": -1},
        {"x": 0, "y": 0, "width": float("inf"), "height": 1},
    ],
)
def test_invalid_box_geometry_fails(box: dict[str, object]) -> None:
    with pytest.raises(ManifestBuildError):
        aggregate_labels(_labels_frame([box]))


def test_duplicate_boxes_fail() -> None:
    box = {"x": 1, "y": 2, "width": 3, "height": 4}
    with pytest.raises(ManifestBuildError):
        aggregate_labels(_labels_frame([box, box]))


def test_bundle_construction_rejects_box_outside_source_dimensions(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    labels_path = root / "stage_2_train_labels.csv"
    labels = pd.read_csv(labels_path)
    labels.loc[labels["patientId"] == "positive", "x"] = 1023
    labels.to_csv(labels_path, index=False)

    with pytest.raises(ManifestBuildError):
        build_rsna_artifacts(root)


def test_portable_annotation_validation_rejects_invalid_geometry(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    invalid = _replace_table_row(result.annotations, 0, width=0.0)

    with pytest.raises(ManifestBuildError):
        validate_annotation_table(invalid, result.samples, result.labels)


def test_portable_annotation_validation_uses_stored_sample_dimensions(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    samples = _replace_table_row(
        result.samples,
        1,
        image_rows=1,
        image_columns=1,
    )

    with pytest.raises(ManifestBuildError):
        validate_annotation_table(result.annotations, samples, result.labels)


def test_nonpositive_image_dimensions_fail(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    _write_header(root / "stage_2_train_images" / "positive.dcm", "positive", rows=0)

    with pytest.raises(ManifestBuildError):
        build_rsna_artifacts(root)


@pytest.mark.parametrize("spacing", [(0.0, 0.1), (-0.1, 0.1), (float("inf"), 0.1)])
def test_invalid_pixel_spacing_fails(tmp_path: Path, spacing: tuple[float, float]) -> None:
    root = _write_sources(tmp_path / "extracted", spacing=spacing)

    with pytest.raises(ManifestBuildError):
        build_rsna_artifacts(root)


def test_sample_schema_rejects_unexpected_column(tmp_path: Path) -> None:
    root, result = _tables(tmp_path)
    invalid = result.samples.append_column("extra", pa.array([1, 2], type=pa.int8()))

    with pytest.raises(ManifestBuildError):
        validate_sample_table(invalid, root)


def test_annotation_schema_rejects_unexpected_column(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    invalid = result.annotations.append_column("extra", pa.array([1, 2], type=pa.int8()))

    with pytest.raises(ManifestBuildError):
        validate_annotation_table(invalid, result.samples, result.labels)


def test_label_schema_rejects_unexpected_column(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    invalid = result.labels.append_column(
        "extra", pa.array([1] * result.labels.num_rows, type=pa.int8())
    )

    with pytest.raises(ManifestBuildError):
        validate_label_table(invalid, result.samples)


def test_exactly_one_pneumonia_label_is_required_per_sample(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    rows = [
        row
        for row in result.labels.to_pylist()
        if not (row["sample_id"] == "rsna:positive" and row["task_id"] == PNEUMONIA_TASK_ID)
    ]
    missing = pa.Table.from_pylist(rows, RSNA_LABEL_SCHEMA)

    with pytest.raises(ManifestBuildError):
        validate_label_table(missing, result.samples)


def test_annotation_relationships_and_identifier_are_enforced(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    negative_annotation = _replace_table_row(
        result.annotations,
        0,
        sample_id="rsna:negative",
        annotation_id="rsna:negative:bbox:0000",
    )
    with pytest.raises(ManifestBuildError):
        validate_annotation_table(
            negative_annotation,
            result.samples,
            result.labels,
        )

    invalid_id = _replace_table_row(result.annotations, 0, annotation_id="arbitrary")
    with pytest.raises(ManifestBuildError):
        validate_annotation_table(invalid_id, result.samples, result.labels)


def test_every_positive_sample_requires_an_annotation(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    empty = pa.Table.from_pylist([], RSNA_ANNOTATION_SCHEMA)

    with pytest.raises(ManifestBuildError):
        validate_annotation_table(empty, result.samples, result.labels)


def test_sample_validation_accepts_multiple_samples_for_one_patient(tmp_path: Path) -> None:
    root, result = _tables(tmp_path)
    rows = result.samples.to_pylist()
    duplicate_patient = {
        **rows[1],
        "sample_id": "rsna:second-sample",
        "image_id": "second-sample",
        "image_path": "stage_2_train_images/second-sample.dcm",
    }
    _write_header(root / duplicate_patient["image_path"], "positive")
    table = pa.Table.from_pylist(
        sorted([*rows, duplicate_patient], key=lambda row: row["sample_id"]),
        RSNA_SAMPLE_SCHEMA,
    )

    validate_sample_table(table, root)


def test_path_traversal_and_absolute_paths_fail(tmp_path: Path) -> None:
    root, result = _tables(tmp_path)
    for invalid_path in ("../positive.dcm", str((root / "positive.dcm").resolve())):
        invalid = _replace_table_row(result.samples, 0, image_path=invalid_path)
        with pytest.raises(ManifestBuildError):
            validate_sample_table(invalid, root)
