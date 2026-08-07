"""Parse RSNA Stage 2 sources, join identifiers, and discover DICOM files."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd

from radfusion.data.errors import ManifestBuildError

LABEL_COLUMNS = ("patientId", "x", "y", "width", "height", "Target")
CLASS_COLUMNS = ("patientId", "class")
ALLOWED_CLASSES = {"Normal", "No Lung Opacity / Not Normal", "Lung Opacity"}
RSNA_CLASS_VALUES = {
    "Normal": 0,
    "No Lung Opacity / Not Normal": 1,
    "Lung Opacity": 2,
}


@dataclass(frozen=True)
class RsnaPaths:
    """Resolved inputs beneath an extracted RSNA dataset root."""

    root: Path
    labels: Path
    class_info: Path
    images: Path

    @classmethod
    def from_root(cls, root: str | Path) -> RsnaPaths:
        resolved_root = Path(root).expanduser().resolve()
        return cls(
            root=resolved_root,
            labels=resolved_root / "stage_2_train_labels.csv",
            class_info=resolved_root / "stage_2_detailed_class_info.csv",
            images=resolved_root / "stage_2_train_images",
        )

    def validate(self) -> None:
        missing = [str(path) for path in (self.labels, self.class_info) if not path.is_file()]
        if not self.images.is_dir():
            missing.append(str(self.images))
        if missing:
            raise ManifestBuildError("Missing required RSNA source paths: " + ", ".join(missing))


@dataclass(frozen=True, order=True)
class BoundingBox:
    """One RSNA bounding box in source-image pixel coordinates."""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class SourceSample:
    """Validated labels and annotations for one RSNA source image."""

    source_id: str
    target: int
    rsna_class: str
    boxes: tuple[BoundingBox, ...]


def load_source_samples(paths: RsnaPaths) -> dict[str, SourceSample]:
    """Join and normalize RSNA challenge targets, classes, and boxes."""
    labels = pd.read_csv(paths.labels, dtype={"patientId": "string"})
    classes = pd.read_csv(paths.class_info, dtype={"patientId": "string"})
    _require_source_columns(labels, LABEL_COLUMNS, paths.labels)
    _require_source_columns(classes, CLASS_COLUMNS, paths.class_info)
    labels_by_id = aggregate_labels(labels)
    classes_by_id = _aggregate_classes(classes)
    validate_identifier_sets(set(labels_by_id), set(classes_by_id), "labels", "classes")

    samples: dict[str, SourceSample] = {}
    for source_id in sorted(labels_by_id):
        target, boxes = labels_by_id[source_id]
        rsna_class = classes_by_id[source_id]
        validate_target_class(target, rsna_class, source_id)
        samples[source_id] = SourceSample(source_id, target, rsna_class, boxes)
    return samples


def aggregate_labels(labels: pd.DataFrame) -> dict[str, tuple[int, tuple[BoundingBox, ...]]]:
    """Aggregate label CSV rows to one binary target and ordered box tuple per source ID."""
    if labels["patientId"].isna().any() or (labels["patientId"].str.strip() == "").any():
        raise ManifestBuildError("Label patientId values must be non-null and non-empty")
    try:
        targets = pd.to_numeric(labels["Target"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ManifestBuildError("Target values must be numeric") from exc
    labels = labels.assign(Target=targets)

    records: dict[str, tuple[int, tuple[BoundingBox, ...]]] = {}
    for patient_id, group in labels.groupby("patientId", sort=False):
        source_id = str(patient_id).strip()
        target_values = set(group["Target"].dropna().tolist())
        if len(target_values) != 1 or not target_values <= {0, 1}:
            raise ManifestBuildError(
                f"Target must be one consistent binary value for patientId {source_id!r}; "
                f"found {sorted(target_values)}"
            )
        target = int(next(iter(target_values)))
        boxes: list[BoundingBox] = []
        for row in group.itertuples(index=False):
            raw_coordinates = (row.x, row.y, row.width, row.height)
            missing = tuple(pd.isna(value) for value in raw_coordinates)
            if target == 0:
                if any(not value for value in missing):
                    raise ManifestBuildError(
                        f"Negative patientId {source_id!r} unexpectedly contains box coordinates"
                    )
                continue
            if any(missing):
                raise ManifestBuildError(
                    f"Positive patientId {source_id!r} has an incomplete bounding box"
                )
            try:
                box = BoundingBox(*(float(value) for value in raw_coordinates))
            except (TypeError, ValueError) as exc:
                raise ManifestBuildError(
                    f"PatientId {source_id!r} has non-numeric bounding-box coordinates"
                ) from exc
            validate_box_values(box, source_id)
            boxes.append(box)
        ordered_boxes = tuple(sorted(boxes))
        if len(ordered_boxes) != len(set(ordered_boxes)):
            raise ManifestBuildError(f"Duplicate bounding box for patientId {source_id!r}")
        if target == 1 and not ordered_boxes:
            raise ManifestBuildError(f"Positive patientId {source_id!r} has no bounding boxes")
        records[source_id] = (target, ordered_boxes)
    return records


def discover_dicoms(image_directory: Path) -> dict[str, Path]:
    """Discover DICOM files and reject case-insensitive identifier collisions."""
    dicoms: dict[str, Path] = {}
    duplicates: list[str] = []
    candidates = sorted(
        path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".dcm"
    )
    for path in candidates:
        resolved = path.resolve()
        if path.stem in dicoms:
            duplicates.append(path.stem)
        dicoms[path.stem] = resolved
    if duplicates:
        raise ManifestBuildError(f"Duplicate DICOM image identifiers: {sorted(duplicates)[:10]}")
    if not dicoms:
        raise ManifestBuildError(f"No .dcm files found in {image_directory}")
    return dicoms


def canonical_image_path(image_id: str) -> PurePosixPath:
    """Return the canonical dataset-relative path for an RSNA image."""
    return PurePosixPath("stage_2_train_images", f"{image_id}.dcm")


def resolve_image_path(root: Path, relative_path: PurePosixPath) -> Path:
    """Resolve a dataset-relative image path within the dataset root."""
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*relative_path.parts)).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ManifestBuildError(f"Image path escapes the dataset root: {relative_path}")
    return resolved


def validate_identifier_sets(
    left: set[str], right: set[str], left_name: str, right_name: str
) -> None:
    """Validate that two source identifier sets match exactly."""
    missing_right = sorted(left - right)
    missing_left = sorted(right - left)
    messages = []
    if missing_right:
        messages.append(
            f"{left_name} missing from {right_name}: {len(missing_right)} "
            f"(examples: {missing_right[:10]})"
        )
    if missing_left:
        messages.append(
            f"{right_name} missing from {left_name}: {len(missing_left)} "
            f"(examples: {missing_left[:10]})"
        )
    if messages:
        raise ManifestBuildError("Source identifier mismatch:\n" + "\n".join(messages))


def validate_target_class(target: int, rsna_class: str, identifier: str) -> None:
    """Validate compatibility between a pneumonia target and RSNA class."""
    compatible = (target == 1 and rsna_class == "Lung Opacity") or (
        target == 0 and rsna_class != "Lung Opacity"
    )
    if not compatible:
        raise ManifestBuildError(
            f"Target/class incompatibility for {identifier!r}: target={target}, "
            f"class={rsna_class!r}"
        )


def validate_box_values(box: BoundingBox, identifier: str) -> None:
    """Validate bounding-box coordinates and extents."""
    values = (box.x, box.y, box.width, box.height)
    if not all(math.isfinite(value) for value in values):
        raise ManifestBuildError(f"Bounding box for {identifier!r} has non-finite coordinates")
    if box.x < 0 or box.y < 0 or box.width <= 0 or box.height <= 0:
        raise ManifestBuildError(f"Bounding box for {identifier!r} has invalid geometry")


def validate_box_bounds(box: BoundingBox, dimensions: tuple[int, int], identifier: str) -> None:
    """Validate that a bounding box lies within image bounds."""
    validate_box_values(box, identifier)
    rows, columns = dimensions
    if box.x + box.width > columns or box.y + box.height > rows:
        raise ManifestBuildError(f"Bounding box for {identifier!r} exceeds image bounds")


def _aggregate_classes(classes: pd.DataFrame) -> dict[str, str]:
    if classes["patientId"].isna().any() or (classes["patientId"].str.strip() == "").any():
        raise ManifestBuildError("Class patientId values must be non-null and non-empty")
    records: dict[str, str] = {}
    for patient_id, group in classes.groupby("patientId", sort=False):
        source_id = str(patient_id).strip()
        values = {str(value).strip() for value in group["class"].dropna() if str(value).strip()}
        if len(values) != 1:
            raise ManifestBuildError(
                f"Class must be one consistent non-empty value for patientId {source_id!r}; "
                f"found {sorted(values)}"
            )
        rsna_class = next(iter(values))
        if rsna_class not in ALLOWED_CLASSES:
            raise ManifestBuildError(f"Unexpected RSNA class {rsna_class!r} for {source_id!r}")
        records[source_id] = rsna_class
    return records


def _require_source_columns(frame: pd.DataFrame, required: tuple[str, ...], path: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ManifestBuildError(f"{path} is missing required columns: {missing}")
