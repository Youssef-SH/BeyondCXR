from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from radfusion.data.errors import ManifestBuildError
from radfusion.data.hashing import arrow_ipc_sha256, sha256_file
from radfusion.data.rsna_artifacts import (
    ANNOTATIONS_FILENAME,
    CURRENT_FILENAME,
    LABELS_FILENAME,
    SAMPLES_FILENAME,
    SOURCE_INVENTORY_FILENAME,
    SPLITS_FILENAME,
    BuildResult,
    _bundle_id,
    _bundle_identity_payload,
    build_rsna_artifacts,
    load_current_bundle,
    validate_bundle_directory,
    validate_bundle_reference,
    write_bundle,
)
from radfusion.data.rsna_audit import REPORT_FILENAMES, generate_rsna_audit
from radfusion.data.rsna_manifest import main
from radfusion.data.rsna_schemas import (
    PNEUMONIA_TASK_ID,
    RSNA_ANNOTATION_SCHEMA,
    RSNA_CLASS_TASK_ID,
    RSNA_LABEL_SCHEMA,
    RSNA_SAMPLE_SCHEMA,
    RSNA_SOURCE_INVENTORY_SCHEMA,
    RSNA_SPLIT_SCHEMA,
)
from radfusion.data.rsna_source import aggregate_labels
from radfusion.data.rsna_validation import (
    validate_annotation_table,
    validate_label_table,
    validate_sample_table,
)
from radfusion.training.config import load_experiment_config, with_runtime
from radfusion.training.rsna_datasets import RsnaDataset
from radfusion.utils.privacy import validate_public_reports


def _write_header(
    path: Path,
    patient_id: str | None,
    *,
    age: str | None = "057Y",
    sex: str | None = "F",
    view: str | None = "PA",
    spacing: tuple[float, float] | None = (0.168, 0.168),
    rows: int = 1024,
    columns: int = 1024,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    sop_instance_uid = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    if patient_id is not None:
        dataset.PatientID = patient_id
    if age is not None:
        dataset.PatientAge = age
    if sex is not None:
        dataset.PatientSex = sex
    if view is not None:
        dataset.ViewPosition = view
    if spacing is not None:
        dataset.PixelSpacing = list(spacing)
    dataset.Rows = rows
    dataset.Columns = columns
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.Modality = "CR"
    dataset.BodyPartExamined = "CHEST"
    dataset.save_as(path)


def _write_sources(
    root: Path,
    *,
    positive: bool = True,
    patient_id: str = "positive",
    age: str | None = "057Y",
    sex: str | None = "F",
    view: str | None = "PA",
    spacing: tuple[float, float] | None = (0.168, 0.168),
) -> Path:
    images = root / "stage_2_train_images"
    images.mkdir(parents=True)
    labels = [
        {
            "patientId": "negative",
            "x": None,
            "y": None,
            "width": None,
            "height": None,
            "Target": 0,
        }
    ]
    classes = [{"patientId": "negative", "class": "Normal"}]
    _write_header(images / "negative.dcm", "negative")
    if positive:
        labels.extend(
            [
                {
                    "patientId": patient_id,
                    "x": 1,
                    "y": 2,
                    "width": 3,
                    "height": 4,
                    "Target": 1,
                },
                {
                    "patientId": patient_id,
                    "x": 5,
                    "y": 6,
                    "width": 7,
                    "height": 8,
                    "Target": 1,
                },
            ]
        )
        classes.extend(
            [
                {"patientId": patient_id, "class": "Lung Opacity"},
                {"patientId": patient_id, "class": "Lung Opacity"},
            ]
        )
        _write_header(
            images / f"{patient_id}.dcm",
            patient_id,
            age=age,
            sex=sex,
            view=view,
            spacing=spacing,
        )
    pd.DataFrame(labels).to_csv(root / "stage_2_train_labels.csv", index=False)
    pd.DataFrame(classes).to_csv(root / "stage_2_detailed_class_info.csv", index=False)
    return root


def _tables(tmp_path: Path) -> tuple[Path, BuildResult]:
    root = _write_sources(tmp_path / "extracted")
    return root, build_rsna_artifacts(root)


def _replace_table_row(table: pa.Table, row_index: int, **changes: object) -> pa.Table:
    rows = table.to_pylist()
    rows[row_index].update(changes)
    return pa.Table.from_pylist(rows, schema=table.schema)


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
    assert "target_pneumonia" not in positive
    assert "rsna_class" not in positive
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


def test_parquet_round_trip_is_exact_and_nested_free(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")

    restored_samples = pq.read_table(written.paths.samples_path)
    restored_labels = pq.read_table(written.paths.labels_path)
    restored_annotations = pq.read_table(written.paths.annotations_path)
    restored_inventory = pq.read_table(written.paths.source_inventory_path)
    assert restored_samples.equals(result.samples)
    assert restored_labels.equals(result.labels)
    assert restored_annotations.equals(result.annotations)
    assert restored_inventory.equals(result.source_inventory)
    assert restored_samples.schema == RSNA_SAMPLE_SCHEMA
    assert restored_labels.schema == RSNA_LABEL_SCHEMA
    assert restored_annotations.schema == RSNA_ANNOTATION_SCHEMA
    assert not any(pa.types.is_nested(field.type) for field in restored_annotations.schema)


def test_arrow_ipc_hashes_are_deterministic(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    first = write_bundle(result, tmp_path / "first")
    second = write_bundle(result, tmp_path / "second")

    assert first.logical_arrow_sha256 == second.logical_arrow_sha256
    assert first.paths.bundle_id == second.paths.bundle_id
    assert first.logical_arrow_sha256[SAMPLES_FILENAME] == arrow_ipc_sha256(result.samples)


def test_logical_hash_ignores_null_buffers_and_survives_parquet_round_trip(
    tmp_path: Path,
) -> None:
    null_bitmap = pa.py_buffer(b"\x01")
    first = pa.Array.from_buffers(
        pa.int32(),
        2,
        [null_bitmap, pa.py_buffer((1).to_bytes(4, "little") + (7).to_bytes(4, "little"))],
        null_count=1,
    )
    second = pa.Array.from_buffers(
        pa.int32(),
        2,
        [null_bitmap, pa.py_buffer((1).to_bytes(4, "little") + (99).to_bytes(4, "little"))],
        null_count=1,
    )

    assert first.to_pylist() == second.to_pylist() == [1, None]
    first_table = pa.table({"value": first})
    second_table = pa.table({"value": second})
    path = tmp_path / "nulls.parquet"
    pq.write_table(first_table, path)

    expected = arrow_ipc_sha256(first_table)
    assert expected == arrow_ipc_sha256(second_table)
    assert expected == arrow_ipc_sha256(pq.read_table(path))


def test_semantic_identity_is_independent_of_parquet_encoding(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    zstd = tmp_path / "samples-zstd.parquet"
    snappy = tmp_path / "samples-snappy.parquet"
    pq.write_table(result.samples, zstd, compression="zstd")
    pq.write_table(result.samples, snappy, compression="snappy")

    assert sha256_file(zstd) != sha256_file(snappy)
    zstd_samples = pq.read_table(zstd)
    snappy_samples = pq.read_table(snappy)
    zstd_sample_hash = arrow_ipc_sha256(zstd_samples)
    snappy_sample_hash = arrow_ipc_sha256(snappy_samples)
    assert zstd_sample_hash == snappy_sample_hash

    zstd_logical_hashes = {
        SAMPLES_FILENAME: zstd_sample_hash,
        LABELS_FILENAME: arrow_ipc_sha256(result.labels),
        ANNOTATIONS_FILENAME: arrow_ipc_sha256(result.annotations),
        SPLITS_FILENAME: arrow_ipc_sha256(result.splits),
        SOURCE_INVENTORY_FILENAME: arrow_ipc_sha256(result.source_inventory),
    }
    snappy_logical_hashes = {
        SAMPLES_FILENAME: snappy_sample_hash,
        LABELS_FILENAME: arrow_ipc_sha256(result.labels),
        ANNOTATIONS_FILENAME: arrow_ipc_sha256(result.annotations),
        SPLITS_FILENAME: arrow_ipc_sha256(result.splits),
        SOURCE_INVENTORY_FILENAME: arrow_ipc_sha256(result.source_inventory),
    }
    assert zstd_logical_hashes == snappy_logical_hashes
    assert _bundle_id(zstd_logical_hashes, result.metadata) == _bundle_id(
        snappy_logical_hashes, result.metadata
    )


def test_bundle_id_ignores_nonsemantic_metadata(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    changed = replace(
        result,
        metadata={
            **result.metadata,
            "generation": {"timestamp_utc": "2099-01-01T00:00:00+00:00"},
            "provenance": {
                "tool_versions": {
                    "python": "different",
                    "pandas": "different",
                    "pyarrow": "different",
                    "pydicom": "different",
                }
            },
        },
    )

    first = write_bundle(result, tmp_path / "first")
    second = write_bundle(changed, tmp_path / "second")

    assert first.paths.bundle_id == second.paths.bundle_id


def test_bundle_id_changes_with_semantic_metadata(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    changed = {
        **result.metadata,
        "dataset": {**result.metadata["dataset"], "release": "semantic-change"},
    }
    logical_hashes = {
        SAMPLES_FILENAME: arrow_ipc_sha256(result.samples),
        LABELS_FILENAME: arrow_ipc_sha256(result.labels),
        ANNOTATIONS_FILENAME: arrow_ipc_sha256(result.annotations),
        SPLITS_FILENAME: arrow_ipc_sha256(result.splits),
        SOURCE_INVENTORY_FILENAME: arrow_ipc_sha256(result.source_inventory),
    }

    assert _bundle_id(logical_hashes, result.metadata) != _bundle_id(logical_hashes, changed)


def test_bundle_validation_ignores_provenance_and_generation_changes(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    original_id = written.paths.bundle_id
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["generation"]["timestamp_utc"] = "2099-01-01T00:00:00+00:00"
    metadata["provenance"]["tool_versions"] = {"python": "different"}
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    assert load_current_bundle(output).bundle_id == original_id


def test_bundle_id_uses_semantic_metadata_and_logical_artifact_hashes(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    arrow_hashes = {
        SAMPLES_FILENAME: "a" * 64,
        LABELS_FILENAME: "b" * 64,
        ANNOTATIONS_FILENAME: "c" * 64,
        SPLITS_FILENAME: "d" * 64,
        SOURCE_INVENTORY_FILENAME: "e" * 64,
    }
    original = _bundle_id(arrow_hashes, result.metadata)
    changed_split = {
        **result.metadata,
        "membership": {"split": {**result.metadata["membership"]["split"], "seed": 43}},
    }

    assert _bundle_id(arrow_hashes, changed_split) != original
    assert _bundle_id({**arrow_hashes, SAMPLES_FILENAME: "f" * 64}, result.metadata) != original


def test_rsna_semantic_payload_uses_abstract_artifact_roles(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    logical_hashes = {
        SAMPLES_FILENAME: "a" * 64,
        LABELS_FILENAME: "b" * 64,
        ANNOTATIONS_FILENAME: "c" * 64,
        SPLITS_FILENAME: "d" * 64,
        SOURCE_INVENTORY_FILENAME: "e" * 64,
    }

    payload = _bundle_identity_payload(logical_hashes, result.metadata)

    assert set(payload["artifacts"]) == {
        "samples",
        "labels",
        "annotations",
        "splits",
        "source_inventory",
    }
    assert ".parquet" not in json.dumps(payload, sort_keys=True)


def test_staging_failure_preserves_current_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    first = write_bundle(result, output)
    current_before = first.paths.current_path.read_text(encoding="utf-8")

    def fail_validation(*args: object, **kwargs: object) -> None:
        raise ManifestBuildError("staged validation failed")

    monkeypatch.setattr("radfusion.data.rsna_artifacts.validate_bundle_directory", fail_validation)
    with pytest.raises(ManifestBuildError):
        write_bundle(result, output)
    assert first.paths.current_path.read_text(encoding="utf-8") == current_before
    validate_bundle_directory(
        first.paths.bundle_directory,
        expected_bundle_id=first.paths.bundle_id,
    )


def test_new_bundle_is_validated_once_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result = _tables(tmp_path)
    calls: list[tuple[Path, bool]] = []

    def record_validation(
        directory: str | Path,
        *,
        expected_bundle_id: str | None = None,
        enforce_directory_name: bool = True,
    ) -> dict[str, object]:
        calls.append((Path(directory), enforce_directory_name))
        return validate_bundle_directory(
            directory,
            expected_bundle_id=expected_bundle_id,
            enforce_directory_name=enforce_directory_name,
        )

    monkeypatch.setattr(
        "radfusion.data.rsna_artifacts.validate_bundle_directory",
        record_validation,
    )
    written = write_bundle(result, tmp_path / "manifests")

    assert len(calls) == 1
    assert calls[0][1] is False
    load_current_bundle(tmp_path / "manifests")
    assert calls[-1] == (written.paths.bundle_directory, True)


def test_consumer_rejects_bundle_with_hash_mismatch(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")
    written.paths.labels_path.write_bytes(b"corrupt")

    with pytest.raises(ManifestBuildError):
        load_current_bundle(tmp_path / "manifests")


def test_bundle_reference_validation_does_not_materialize_parquet_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")

    monkeypatch.setattr(
        "radfusion.data.rsna_artifacts.pq.read_table",
        lambda *args, **kwargs: pytest.fail((args, kwargs, "row materialization")),
    )

    validated = validate_bundle_reference(
        written.paths.bundle_directory,
        expected_bundle_id=written.paths.bundle_id,
        expected_manifest_sha256=sha256_file(written.paths.metadata_path),
    )
    assert validated.manifest["bundle"]["bundle_id"] == written.paths.bundle_id
    assert validated.manifest_sha256 == sha256_file(written.paths.metadata_path)


def test_bundle_reference_allows_new_operational_metadata_for_same_identity(
    tmp_path: Path,
) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")
    published_metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    base = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    configured = with_runtime(
        replace(
            base,
            dataset=replace(
                base.dataset,
                bundle_id=written.paths.bundle_id,
                bundle_manifest_sha256=sha256_file(written.paths.metadata_path),
                split_assignment_id=published_metadata["membership"]["split"][
                    "split_assignment_id"
                ],
            ),
        ),
        manifest_directory=tmp_path / "manifests",
    )
    original_lineage = RsnaDataset().load_lineage(configured)
    original = validate_bundle_reference(
        written.paths.bundle_directory,
        expected_bundle_id=written.paths.bundle_id,
    )
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["generation"]["timestamp_utc"] = "2099-01-01T00:00:00+00:00"
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    regenerated = validate_bundle_reference(
        written.paths.bundle_directory,
        expected_bundle_id=written.paths.bundle_id,
    )

    assert regenerated.manifest["bundle"]["bundle_id"] == written.paths.bundle_id
    assert regenerated.manifest_sha256 != original.manifest_sha256
    refreshed_config = replace(
        configured,
        dataset=replace(
            configured.dataset,
            bundle_manifest_sha256=sha256_file(written.paths.metadata_path),
        ),
    )
    assert RsnaDataset().load_lineage(refreshed_config) == original_lineage


def test_bundle_reference_rejects_wrong_selection_and_malformed_manifest(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    selected = write_bundle(result, tmp_path / "selected")
    with pytest.raises(ManifestBuildError):
        validate_bundle_reference(
            selected.paths.bundle_directory,
            expected_bundle_id="bundle-wrong",
        )

    malformed = write_bundle(result, tmp_path / "malformed")
    malformed.paths.metadata_path.write_bytes(b"not-json")
    with pytest.raises(ManifestBuildError):
        validate_bundle_reference(
            malformed.paths.bundle_directory,
            expected_bundle_id=malformed.paths.bundle_id,
        )


def test_image_test_manifest_mismatch_fails_before_partition_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")
    base = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    configured = with_runtime(
        replace(base, dataset=replace(base.dataset, bundle_id=written.paths.bundle_id)),
        manifest_directory=tmp_path / "manifests",
        source_root=tmp_path / "raw",
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_datasets._task_frame",
        lambda *args, **kwargs: pytest.fail((args, kwargs, "test partition access")),
    )

    with pytest.raises(ManifestBuildError):
        RsnaDataset().load_image_test(
            configured,
            expected_manifest_sha256="0" * 64,
        )


def test_bundle_reference_lineage_rejects_manifest_and_coordinated_artifact_tampering(
    tmp_path: Path,
) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")
    expected_manifest_sha256 = sha256_file(written.paths.metadata_path)

    table = pq.read_table(written.paths.labels_path)
    pq.write_table(table, written.paths.labels_path, compression=None)
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"][LABELS_FILENAME]["physical_file_sha256"] = sha256_file(
        written.paths.labels_path
    )
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ManifestBuildError):
        validate_bundle_reference(
            written.paths.bundle_directory,
            expected_bundle_id=written.paths.bundle_id,
            expected_manifest_sha256=expected_manifest_sha256,
        )


def test_bundle_reference_lineage_rejects_artifact_only_and_metadata_only_tampering(
    tmp_path: Path,
) -> None:
    _, result = _tables(tmp_path)
    artifact = write_bundle(result, tmp_path / "artifact")
    artifact_manifest_sha256 = sha256_file(artifact.paths.metadata_path)
    artifact.paths.labels_path.write_bytes(b"tampered")
    with pytest.raises(ManifestBuildError):
        validate_bundle_reference(
            artifact.paths.bundle_directory,
            expected_bundle_id=artifact.paths.bundle_id,
            expected_manifest_sha256=artifact_manifest_sha256,
        )

    manifest = write_bundle(result, tmp_path / "manifest")
    manifest_sha256 = sha256_file(manifest.paths.metadata_path)
    document = json.loads(manifest.paths.metadata_path.read_text(encoding="utf-8"))
    document["source"]["files"][next(iter(document["source"]["files"]))] = "a" * 64
    manifest.paths.metadata_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ManifestBuildError):
        validate_bundle_reference(
            manifest.paths.bundle_directory,
            expected_bundle_id=manifest.paths.bundle_id,
            expected_manifest_sha256=manifest_sha256,
        )


def test_consumer_rejects_declared_artifact_hash_tampering(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"][LABELS_FILENAME]["physical_file_sha256"] = "tampered"
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ManifestBuildError):
        load_current_bundle(output)


@pytest.mark.parametrize("entry_kind", ["file", "directory", "symlink"])
def test_consumer_rejects_unexpected_bundle_entries(tmp_path: Path, entry_kind: str) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    unexpected = written.paths.bundle_directory / "unexpected"
    if entry_kind == "file":
        unexpected.write_text("unexpected\n", encoding="utf-8")
    elif entry_kind == "directory":
        unexpected.mkdir()
    else:
        unexpected.symlink_to(written.paths.samples_path.name)

    with pytest.raises(ManifestBuildError):
        load_current_bundle(output)


@pytest.mark.parametrize("attribute", ["labels_path", "metadata_path"])
def test_consumer_rejects_required_bundle_artifact_symlinks(tmp_path: Path, attribute: str) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    required_path = getattr(written.paths, attribute)
    external = tmp_path / f"external-{required_path.name}"
    required_path.rename(external)
    required_path.symlink_to(external)

    with pytest.raises(ManifestBuildError):
        load_current_bundle(output)


def test_consumer_rejects_extra_declared_artifact(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"]["unexpected.parquet"] = {}
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ManifestBuildError):
        load_current_bundle(output)


def test_consumer_rejects_unsupported_manifest_schema(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["bundle_manifest_schema_version"] = "unsupported"
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ManifestBuildError):
        load_current_bundle(output)


def test_source_inventory_authenticates_every_dicom_and_detects_tampering(tmp_path: Path) -> None:
    root, result = _tables(tmp_path)
    rows = result.source_inventory.to_pylist()
    assert {row["relative_path"] for row in rows} == {
        "stage_2_train_images/negative.dcm",
        "stage_2_train_images/positive.dcm",
    }
    for row in rows:
        source = root / row["relative_path"]
        assert row["byte_size"] == source.stat().st_size
        assert row["sha256"] == sha256_file(source)

    written = write_bundle(result, tmp_path / "manifests")
    written.paths.source_inventory_path.write_bytes(b"tampered")
    with pytest.raises(ManifestBuildError):
        load_current_bundle(tmp_path / "manifests")


def test_portable_bundle_validation_does_not_access_external_dicoms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")

    def reject_external_access(*args: object, **kwargs: object) -> Path:
        raise AssertionError("portable validation accessed the external dataset")

    monkeypatch.setattr(
        "radfusion.data.rsna_artifacts.resolve_image_path",
        reject_external_access,
    )
    validate_bundle_directory(
        written.paths.bundle_directory,
        expected_bundle_id=written.paths.bundle_id,
    )


def test_consumer_rejects_metadata_that_does_not_match_bundle_id(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["membership"]["split"]["seed"] = 43
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ManifestBuildError):
        load_current_bundle(output)


@pytest.mark.parametrize(
    "invalid_form",
    [
        "dictionary-ratios",
        "reordered-ratios",
        "duplicate-split-name",
        "unknown-field",
    ],
)
def test_consumer_rejects_noncanonical_split_metadata(
    tmp_path: Path,
    invalid_form: str,
) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    split = metadata["membership"]["split"]
    if invalid_form == "dictionary-ratios":
        split["ratios"] = {"train": 0.7, "validation": 0.15, "test": 0.15}
    elif invalid_form == "reordered-ratios":
        split["ratios"] = [split["ratios"][0], split["ratios"][2], split["ratios"][1]]
    elif invalid_form == "duplicate-split-name":
        split["ratios"][1]["split_name"] = "train"
    else:
        split["unexpected"] = "value"
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestBuildError):
        load_current_bundle(output)


def test_cli_success_and_failure_exit_codes(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    output = tmp_path / "manifests"

    assert main(["--dataset-root", str(root), "--output-directory", str(output)]) == 0
    current = load_current_bundle(output)
    assert current.current_path.name == CURRENT_FILENAME
    assert current.samples_path.is_file()
    assert current.labels_path.is_file()
    assert current.splits_path.is_file()
    assert current.source_inventory_path.is_file()
    metadata = json.loads(current.metadata_path.read_text(encoding="utf-8"))
    assert metadata["qualification"]["counts"]["samples"] == 2
    assert metadata["qualification"]["counts"]["labels"] == 4
    assert metadata["bundle_manifest_schema_version"] == "1"
    split = metadata["membership"]["split"]
    assert split["patient_hash_algorithm"] == "sha256"
    assert split["patient_hash_input_encoding"] == "utf-8"
    assert split["patient_hash_input_template"] == "<seed>\\0<patient_id>"
    assert split["stratification_target"] == "pneumonia"
    assert split["allocation_rule"] == "feasible-minimum-then-largest-remainder-canonical-tiebreak"
    assert split["seed"] == 42
    assert split["ratios"] == [
        {"split_name": "train", "ratio": 0.7},
        {"split_name": "validation", "ratio": 0.15},
        {"split_name": "test", "ratio": 0.15},
    ]
    assert split["split_recipe_id"].startswith("split-recipe-")
    assert split["split_assignment_id"].startswith("split-assignment-")
    assert set(split) == {
        "split_source",
        "split_recipe_id",
        "split_assignment_id",
        "algorithm_version",
        "patient_grouping_rule",
        "patient_target_consistency_rule",
        "ranking_rule",
        "patient_hash_algorithm",
        "patient_hash_input_encoding",
        "patient_hash_input_template",
        "seed",
        "stratification_target",
        "allocation_rule",
        "split_order",
        "ratios",
    }
    for derived_field in (
        "sex_distribution",
        "view_distribution",
        "pixel_spacing_summary",
        "rsna_class_distribution",
        "image_dimensions_summary",
    ):
        assert derived_field not in metadata
    assert metadata["qualification"]["counts"]["source_inventory"] == 2
    assert SOURCE_INVENTORY_FILENAME in metadata["artifacts"]
    assert metadata["provenance"]["logical_arrow_runtime"]["pyarrow_version"] == pa.__version__
    assert (
        "cross-version stability is not claimed"
        in metadata["provenance"]["logical_arrow_runtime"]["stability_scope"]
    )
    assert main(["--dataset-root", str(tmp_path / "missing")]) == 1


@pytest.fixture(scope="session")
def real_rsna_build() -> BuildResult:
    root = Path("data/raw/rsna/extracted")
    if not (root / "stage_2_train_images").is_dir():
        pytest.skip("Local RSNA data is not available")
    return build_rsna_artifacts(root)


@pytest.mark.integration
def test_real_rsna_aggregate_contract(real_rsna_build: BuildResult) -> None:
    assert real_rsna_build.samples.num_rows == 26_684
    assert real_rsna_build.labels.num_rows == 53_368
    assert real_rsna_build.annotations.num_rows == 9_555
    targets = Counter(
        row["label_value"]
        for row in real_rsna_build.labels.to_pylist()
        if row["task_id"] == PNEUMONIA_TASK_ID
    )
    assert targets == {0: 20_672, 1: 6_012}


@pytest.mark.integration
def test_real_rsna_audit_reports_are_aggregate(
    tmp_path: Path, real_rsna_build: BuildResult
) -> None:
    output = tmp_path / "manifests"
    written = write_bundle(real_rsna_build, output)

    report_directory = tmp_path / "reports" / "rsna"
    summary = generate_rsna_audit(output, report_directory)

    audit_directory = report_directory / written.paths.bundle_id
    assert set(path.name for path in audit_directory.iterdir()) == set(REPORT_FILENAMES)
    assert summary["reports"] == list(REPORT_FILENAMES)
    sample_rows = real_rsna_build.samples.slice(0, 10).to_pylist()
    source_values = {
        str(row[field])
        for row in sample_rows
        for field in ("patient_id", "sample_id", "image_id", "image_path")
    }
    source_values.update(Path(row["image_path"]).name for row in sample_rows)
    source_values.update(Path(row["image_path"]).stem for row in sample_rows)
    validate_public_reports(audit_directory.iterdir(), forbidden_source_values=source_values)
