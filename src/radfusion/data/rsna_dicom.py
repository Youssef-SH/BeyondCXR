"""Extract RSNA DICOM metadata and accumulate aggregate audit statistics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydicom
from pydicom.errors import InvalidDicomError

from radfusion.data.age import AgeParseResult, parse_dicom_age
from radfusion.data.errors import ManifestBuildError

ALLOWED_SEX = {"F", "M"}
ALLOWED_VIEW_POSITIONS = {"AP", "PA"}

_DICOM_TAGS = (
    "PatientID",
    "PatientAge",
    "PatientSex",
    "ViewPosition",
    "PixelSpacing",
    "Rows",
    "Columns",
    "PhotometricInterpretation",
    "SOPClassUID",
    "SOPInstanceUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SamplesPerPixel",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "LossyImageCompression",
    "LossyImageCompressionMethod",
    "Modality",
    "BodyPartExamined",
    "SpecificCharacterSet",
)


@dataclass(frozen=True)
class DicomMetadata:
    """DICOM metadata used for sample rows and aggregate audit."""

    patient_id: str
    age: AgeParseResult
    sex: str | None
    view_position: str | None
    pixel_spacing_row_mm: float | None
    pixel_spacing_col_mm: float | None
    rows: int | None
    columns: int | None
    sop_instance_uid: str | None
    study_instance_uid: str | None
    series_instance_uid: str | None
    media_storage_sop_instance_uid: str | None
    audit_values: tuple[tuple[str, str | None], ...]


@dataclass
class AuditAccumulator:
    """Accumulate aggregate DICOM audit counters."""

    dicom_values: dict[str, Counter[str]]
    age_status: Counter[str]
    age_source_format: Counter[str]
    age_warnings: Counter[str]
    implausible_age_count: int
    sop_instance_uids: set[str]
    study_instance_uids: set[str]
    series_instance_uids: set[str]
    media_storage_sop_instance_uids: set[str]
    media_sop_matches: int

    @classmethod
    def empty(cls) -> AuditAccumulator:
        """Return an empty audit accumulator."""
        return cls(
            dicom_values={},
            age_status=Counter(),
            age_source_format=Counter(),
            age_warnings=Counter(),
            implausible_age_count=0,
            sop_instance_uids=set(),
            study_instance_uids=set(),
            series_instance_uids=set(),
            media_storage_sop_instance_uids=set(),
            media_sop_matches=0,
        )

    def add(self, metadata: DicomMetadata) -> None:
        """Add one DICOM metadata record to the aggregate audit."""
        for keyword, value in metadata.audit_values:
            counter = self.dicom_values.setdefault(keyword, Counter())
            counter[value if value is not None else "<missing>"] += 1
        self.age_status[metadata.age.status] += 1
        self.age_source_format[metadata.age.source_format] += 1
        if metadata.age.warning:
            self.age_warnings[metadata.age.warning] += 1
        if metadata.age.implausible:
            self.implausible_age_count += 1
        if metadata.sop_instance_uid:
            self.sop_instance_uids.add(metadata.sop_instance_uid)
        if metadata.study_instance_uid:
            self.study_instance_uids.add(metadata.study_instance_uid)
        if metadata.series_instance_uid:
            self.series_instance_uids.add(metadata.series_instance_uid)
        if metadata.media_storage_sop_instance_uid:
            self.media_storage_sop_instance_uids.add(metadata.media_storage_sop_instance_uid)
        if (
            metadata.sop_instance_uid
            and metadata.sop_instance_uid == metadata.media_storage_sop_instance_uid
        ):
            self.media_sop_matches += 1


def read_dicom_metadata(path: Path) -> DicomMetadata:
    """Read selected DICOM header tags with pixel decoding disabled."""
    try:
        dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=list(_DICOM_TAGS))
    except (InvalidDicomError, OSError, EOFError, TypeError, ValueError) as exc:
        raise ManifestBuildError(f"Could not read DICOM metadata from {path}: {exc}") from exc

    patient_id = _optional_text(getattr(dataset, "PatientID", None))
    if patient_id is None:
        raise ManifestBuildError(f"DICOM PatientID is missing from {path}")
    age = parse_dicom_age(_optional_text(getattr(dataset, "PatientAge", None)))
    sex = _normalized_category(getattr(dataset, "PatientSex", None))
    view_position = _normalized_category(getattr(dataset, "ViewPosition", None))
    spacing_row, spacing_col = _parse_pixel_spacing(getattr(dataset, "PixelSpacing", None), path)
    transfer_syntax = _optional_text(getattr(dataset.file_meta, "TransferSyntaxUID", None))
    media_sop_instance = _optional_text(
        getattr(dataset.file_meta, "MediaStorageSOPInstanceUID", None)
    )

    audit_keywords = (
        "PhotometricInterpretation",
        "SOPClassUID",
        "SamplesPerPixel",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "LossyImageCompression",
        "LossyImageCompressionMethod",
        "Modality",
        "BodyPartExamined",
        "SpecificCharacterSet",
    )
    audit_values = tuple(
        (keyword, _audit_text(getattr(dataset, keyword, None))) for keyword in audit_keywords
    ) + (("TransferSyntaxUID", transfer_syntax),)
    metadata = DicomMetadata(
        patient_id=patient_id,
        age=age,
        sex=sex,
        view_position=view_position,
        pixel_spacing_row_mm=spacing_row,
        pixel_spacing_col_mm=spacing_col,
        rows=_optional_int(getattr(dataset, "Rows", None)),
        columns=_optional_int(getattr(dataset, "Columns", None)),
        sop_instance_uid=_optional_text(getattr(dataset, "SOPInstanceUID", None)),
        study_instance_uid=_optional_text(getattr(dataset, "StudyInstanceUID", None)),
        series_instance_uid=_optional_text(getattr(dataset, "SeriesInstanceUID", None)),
        media_storage_sop_instance_uid=media_sop_instance,
        audit_values=audit_values,
    )
    validate_dicom_metadata(metadata, path)
    return metadata


def validate_dicom_metadata(metadata: DicomMetadata, path: Path) -> None:
    """Validate required DICOM metadata and supported categorical values."""
    if metadata.sex is not None and metadata.sex not in ALLOWED_SEX:
        raise ManifestBuildError(f"Unexpected PatientSex {metadata.sex!r} in {path}")
    if metadata.view_position is not None and metadata.view_position not in ALLOWED_VIEW_POSITIONS:
        raise ManifestBuildError(f"Unexpected ViewPosition {metadata.view_position!r} in {path}")
    validate_spacing_pair(metadata.pixel_spacing_row_mm, metadata.pixel_spacing_col_mm)
    if metadata.rows is None or metadata.columns is None:
        raise ManifestBuildError(f"Missing Rows or Columns in {path}")
    if metadata.rows <= 0 or metadata.columns <= 0:
        raise ManifestBuildError(f"Rows and Columns must be positive in {path}")


def validate_spacing_pair(row_spacing: float | None, col_spacing: float | None) -> None:
    """Validate an optional DICOM pixel-spacing pair."""
    if (row_spacing is None) != (col_spacing is None):
        raise ManifestBuildError("Pixel spacing must provide both row and column values")
    if row_spacing is None or col_spacing is None:
        return
    if (
        not math.isfinite(row_spacing)
        or not math.isfinite(col_spacing)
        or row_spacing <= 0
        or col_spacing <= 0
    ):
        raise ManifestBuildError("Pixel spacing values must be finite and strictly positive")


def _parse_pixel_spacing(value: Any, path: Path) -> tuple[float | None, float | None]:
    if value is None or str(value).strip() == "":
        return None, None
    try:
        if len(value) != 2:
            raise ValueError("expected exactly two values")
        row_spacing, col_spacing = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ManifestBuildError(f"Invalid PixelSpacing in {path}: {value!r}") from exc
    validate_spacing_pair(row_spacing, col_spacing)
    return row_spacing, col_spacing


def _normalized_category(value: Any) -> str | None:
    text = _optional_text(value)
    return None if text is None else text.upper()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _audit_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ",".join(str(item) for item in value)
    return _optional_text(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
