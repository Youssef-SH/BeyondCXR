"""Minimal synthetic RSNA source builders for data-contract tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from radfusion.data.rsna_artifacts import BuildResult, build_rsna_artifacts


def write_header(
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


def write_sources(
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
    write_header(images / "negative.dcm", "negative")
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
        write_header(
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


def tables(tmp_path: Path) -> tuple[Path, BuildResult]:
    root = write_sources(tmp_path / "extracted")
    return root, build_rsna_artifacts(root)


def replace_table_row(table: pa.Table, row_index: int, **changes: object) -> pa.Table:
    rows = table.to_pylist()
    rows[row_index].update(changes)
    return pa.Table.from_pylist(rows, schema=table.schema)
