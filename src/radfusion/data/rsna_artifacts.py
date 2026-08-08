"""Construct and publish immutable RSNA artifact bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pydicom

from radfusion.data.artifact_validation import (
    validate_annotation_table,
    validate_label_table,
    validate_sample_table,
)
from radfusion.data.errors import ManifestBuildError
from radfusion.data.hashing import arrow_ipc_sha256, sha256_file
from radfusion.data.rsna_dicom import AuditAccumulator, read_dicom_metadata
from radfusion.data.rsna_source import (
    RSNA_CLASS_VALUES,
    BoundingBox,
    RsnaPaths,
    canonical_image_path,
    discover_dicoms,
    load_source_samples,
    resolve_image_path,
    validate_box_bounds,
    validate_identifier_sets,
)
from radfusion.data.schemas import (
    DATASET_ID,
    MANIFEST_SCHEMA_VERSION,
    PNEUMONIA_LABEL_POLICY_VERSION,
    PNEUMONIA_LABEL_SOURCE,
    PNEUMONIA_TASK_ID,
    RSNA_ANNOTATION_SCHEMA,
    RSNA_CLASS_LABEL_POLICY_VERSION,
    RSNA_CLASS_LABEL_SOURCE,
    RSNA_CLASS_TASK_ID,
    RSNA_LABEL_SCHEMA,
    RSNA_SAMPLE_SCHEMA,
    RSNA_SOURCE_INVENTORY_SCHEMA,
    RSNA_SPLIT_SCHEMA,
    require_exact_schema,
)
from radfusion.data.splitting import (
    PATIENT_GROUPING_RULE,
    PATIENT_HASH_ALGORITHM,
    PATIENT_HASH_INPUT_ENCODING,
    PATIENT_HASH_INPUT_TEMPLATE,
    PATIENT_RANKING_RULE,
    PATIENT_TARGET_CONSISTENCY_RULE,
    SPLIT_ALGORITHM_VERSION,
    SPLIT_ALLOCATION_RULE,
    SPLIT_NAMES,
    SPLIT_SOURCE,
    STRATIFICATION_TARGET,
    SplitConfig,
    create_patient_stratified_splits,
    split_assignment_id,
    validate_split_table,
)
from radfusion.utils.operational_logging import CountProgress, get_operational_logger
from radfusion.utils.publication import update_current_marker

DATASET_VERSION = "rsna-pneumonia-detection-challenge-stage-2"
_LOGGER = get_operational_logger(__name__)
SAMPLES_FILENAME = "rsna_samples.parquet"
LABELS_FILENAME = "rsna_labels.parquet"
ANNOTATIONS_FILENAME = "rsna_annotations.parquet"
SPLITS_FILENAME = "rsna_splits.parquet"
SOURCE_INVENTORY_FILENAME = "rsna_source_inventory.parquet"
METADATA_FILENAME = "rsna_manifest_metadata.json"
CURRENT_FILENAME = "CURRENT"
BUNDLES_DIRECTORY = "builds"
BUNDLE_IDENTITY_POLICY_VERSION = "rsna-bundle-identity-v2"
_SPLIT_METADATA_FIELDS = frozenset(
    {
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
)


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    patient_id: str
    image_id: str
    image_path: str
    image_rows: int
    image_columns: int
    age_years: float | None
    age_is_implausible: bool
    sex: str | None
    view_position: str | None
    pixel_spacing_row_mm: float | None
    pixel_spacing_col_mm: float | None


@dataclass(frozen=True)
class LabelRecord:
    sample_id: str
    task_id: str
    label_value: int


@dataclass(frozen=True)
class AnnotationRecord:
    sample_id: str
    annotation_id: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class SourceInventoryRecord:
    sample_id: str
    relative_path: str
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class BuildResult:
    samples: pa.Table
    labels: pa.Table
    annotations: pa.Table
    splits: pa.Table
    source_inventory: pa.Table
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class BundlePaths:
    bundle_id: str
    bundle_directory: Path
    samples_path: Path
    labels_path: Path
    annotations_path: Path
    splits_path: Path
    source_inventory_path: Path
    metadata_path: Path
    current_path: Path


@dataclass(frozen=True)
class WriteResult:
    paths: BundlePaths
    arrow_ipc_sha256: Mapping[str, str]


@dataclass(frozen=True)
class ValidatedBundleReference:
    """Validated bundle manifest and its exact serialized-byte identity."""

    manifest: Mapping[str, Any]
    manifest_sha256: str


def build_rsna_artifacts(
    dataset_root: str | Path, split_config: SplitConfig | None = None
) -> BuildResult:
    """Construct validated, deterministic in-memory RSNA artifacts."""
    paths = RsnaPaths.from_root(dataset_root)
    paths.validate()
    source_samples = load_source_samples(paths)
    dicom_paths = discover_dicoms(paths.images)
    validate_identifier_sets(set(source_samples), set(dicom_paths), "labels/classes", "DICOMs")

    sample_records: list[SampleRecord] = []
    label_records: list[LabelRecord] = []
    annotation_records: list[AnnotationRecord] = []
    source_inventory_records: list[SourceInventoryRecord] = []
    dimensions: dict[str, tuple[int, int]] = {}
    audit = AuditAccumulator.empty()
    source_count = len(source_samples)
    progress = (
        CountProgress(
            _LOGGER,
            "manifest_construction_progress",
            total=source_count,
            unit="dicom_files",
        )
        if source_count > 0
        else None
    )

    for completed, source_id in enumerate(sorted(source_samples), start=1):
        source = source_samples[source_id]
        relative_image_path = canonical_image_path(source_id)
        image_path = resolve_image_path(paths.root, relative_image_path)
        if image_path != dicom_paths[source_id]:
            raise ManifestBuildError(
                f"Discovered DICOM path differs from canonical path for source ID {source_id!r}"
            )
        dicom = read_dicom_metadata(image_path)
        if dicom.patient_id != source_id:
            raise ManifestBuildError(
                f"DICOM PatientID {dicom.patient_id!r} does not match filename/source "
                f"identifier {source_id!r}"
            )
        audit.add(dicom)

        sample_id = f"{DATASET_ID}:{source_id}"
        source_inventory_records.append(
            SourceInventoryRecord(
                sample_id=sample_id,
                relative_path=relative_image_path.as_posix(),
                byte_size=image_path.stat().st_size,
                sha256=sha256_file(image_path),
            )
        )
        if dicom.rows is None or dicom.columns is None:
            raise ManifestBuildError(f"Missing image dimensions for DICOM {source_id!r}")
        dimensions[sample_id] = (dicom.rows, dicom.columns)
        sample_records.append(
            SampleRecord(
                sample_id=sample_id,
                patient_id=source_id,
                image_id=source_id,
                image_path=relative_image_path.as_posix(),
                image_rows=dicom.rows,
                image_columns=dicom.columns,
                age_years=dicom.age.value_years,
                age_is_implausible=dicom.age.implausible,
                sex=dicom.sex,
                view_position=dicom.view_position,
                pixel_spacing_row_mm=dicom.pixel_spacing_row_mm,
                pixel_spacing_col_mm=dicom.pixel_spacing_col_mm,
            )
        )
        label_records.extend(_label_records(sample_id, source.target, source.rsna_class))
        annotation_records.extend(
            _annotation_records(
                sample_id,
                source_id,
                source.target,
                source.boxes,
                dimensions[sample_id],
            )
        )
        if progress is not None:
            progress.update(completed)

    samples = pa.Table.from_pylist(
        [asdict(record) for record in sample_records], RSNA_SAMPLE_SCHEMA
    )
    labels = pa.Table.from_pylist(
        [asdict(record) for record in sorted(label_records, key=_label_sort_key)],
        RSNA_LABEL_SCHEMA,
    )
    annotations = pa.Table.from_pylist(
        [asdict(record) for record in annotation_records], RSNA_ANNOTATION_SCHEMA
    )
    source_inventory = pa.Table.from_pylist(
        [asdict(record) for record in source_inventory_records],
        RSNA_SOURCE_INVENTORY_SCHEMA,
    )
    validate_sample_table(samples, paths.root)
    validate_label_table(labels, samples)
    validate_annotation_table(annotations, samples, labels)
    _validate_source_inventory(source_inventory, samples, paths.root)
    if len(audit.sop_instance_uids) != samples.num_rows:
        raise ManifestBuildError("Every RSNA sample must have one unique SOP Instance UID")
    resolved_split_config = split_config or SplitConfig()
    splits = create_patient_stratified_splits(samples, labels, resolved_split_config)
    metadata = _build_aggregate_metadata(
        paths,
        samples,
        labels,
        annotations,
        splits,
        source_inventory,
        audit,
        resolved_split_config,
    )
    return BuildResult(samples, labels, annotations, splits, source_inventory, metadata)


def write_bundle(result: BuildResult, output_directory: str | Path) -> WriteResult:
    """Publish an immutable bundle and atomically update its CURRENT marker."""
    output_root = Path(output_directory)
    dataset_bundle_root = output_root / DATASET_ID
    builds_root = dataset_bundle_root / BUNDLES_DIRECTORY
    builds_root.mkdir(parents=True, exist_ok=True)
    current_path = dataset_bundle_root / CURRENT_FILENAME

    arrow_hashes = {
        SAMPLES_FILENAME: arrow_ipc_sha256(result.samples),
        LABELS_FILENAME: arrow_ipc_sha256(result.labels),
        ANNOTATIONS_FILENAME: arrow_ipc_sha256(result.annotations),
        SPLITS_FILENAME: arrow_ipc_sha256(result.splits),
        SOURCE_INVENTORY_FILENAME: arrow_ipc_sha256(result.source_inventory),
    }
    stage_directory = Path(tempfile.mkdtemp(prefix=".staging-", dir=builds_root))

    try:
        staged_paths = _bundle_paths("pending", stage_directory, current_path)
        pq.write_table(result.samples, staged_paths.samples_path, compression="zstd")
        pq.write_table(result.labels, staged_paths.labels_path, compression="zstd")
        pq.write_table(result.annotations, staged_paths.annotations_path, compression="zstd")
        pq.write_table(result.splits, staged_paths.splits_path, compression="zstd")
        pq.write_table(
            result.source_inventory,
            staged_paths.source_inventory_path,
            compression="zstd",
        )
        bundle_id = _bundle_id(arrow_hashes, result.metadata)
        final_directory = builds_root / bundle_id
        staged_paths = _bundle_paths(bundle_id, stage_directory, current_path)
        metadata = _finalize_metadata(result.metadata, bundle_id, staged_paths, arrow_hashes)
        staged_paths.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_bundle_directory(
            stage_directory,
            expected_bundle_id=bundle_id,
            enforce_directory_name=False,
        )

        if final_directory.exists():
            validate_bundle_directory(final_directory, expected_bundle_id=bundle_id)
            shutil.rmtree(stage_directory)
        else:
            os.replace(stage_directory, final_directory)
        final_paths = _bundle_paths(bundle_id, final_directory, current_path)
        update_current_marker(current_path, bundle_id)
    finally:
        if stage_directory.exists():
            shutil.rmtree(stage_directory)

    return WriteResult(final_paths, arrow_hashes)


def build_and_write(
    dataset_root: str | Path,
    output_directory: str | Path,
    split_config: SplitConfig | None = None,
) -> WriteResult:
    """Construct RSNA artifacts and publish the resulting bundle."""
    return write_bundle(build_rsna_artifacts(dataset_root, split_config), output_directory)


def load_current_bundle(output_directory: str | Path) -> BundlePaths:
    """Resolve CURRENT and verify metadata plus every declared artifact hash."""
    dataset_bundle_root = Path(output_directory) / DATASET_ID
    current_path = dataset_bundle_root / CURRENT_FILENAME
    if not current_path.is_file():
        raise ManifestBuildError(f"Missing CURRENT marker: {current_path}")
    bundle_id = current_path.read_text(encoding="utf-8").strip()
    if not bundle_id or Path(bundle_id).name != bundle_id:
        raise ManifestBuildError("CURRENT contains an invalid bundle identifier")
    bundle_directory = dataset_bundle_root / BUNDLES_DIRECTORY / bundle_id
    validate_bundle_directory(bundle_directory, expected_bundle_id=bundle_id)
    return _bundle_paths(bundle_id, bundle_directory, current_path)


def validate_bundle_directory(
    bundle_directory: str | Path,
    *,
    expected_bundle_id: str | None = None,
    enforce_directory_name: bool = True,
) -> dict[str, Any]:
    """Require complete metadata plus matching file and Arrow IPC hashes for a bundle."""
    validated = validate_bundle_reference(
        bundle_directory,
        expected_bundle_id=expected_bundle_id,
        enforce_directory_name=enforce_directory_name,
    )
    metadata = dict(validated.manifest)
    directory = Path(bundle_directory)
    expected_files = {
        SAMPLES_FILENAME: RSNA_SAMPLE_SCHEMA,
        LABELS_FILENAME: RSNA_LABEL_SCHEMA,
        ANNOTATIONS_FILENAME: RSNA_ANNOTATION_SCHEMA,
        SPLITS_FILENAME: RSNA_SPLIT_SCHEMA,
        SOURCE_INVENTORY_FILENAME: RSNA_SOURCE_INVENTORY_SCHEMA,
    }
    bundle_id = metadata.get("bundle", {}).get("bundle_id")
    split_config = _validate_manifest_contract(metadata)
    hashes = metadata.get("generated_artifact_hashes", {})
    actual_arrow_hashes: dict[str, str] = {}
    for filename in expected_files:
        path = directory / filename
        declared = hashes.get(filename)
        table = pq.read_table(path)
        actual_arrow_hash = arrow_ipc_sha256(table)
        actual_arrow_hashes[filename] = actual_arrow_hash
        if actual_arrow_hash != declared.get("arrow_ipc_sha256"):
            raise ManifestBuildError(f"Arrow IPC hash mismatch for {filename}")
    try:
        computed_bundle_id = _bundle_id(actual_arrow_hashes, metadata)
    except (KeyError, TypeError) as exc:
        raise ManifestBuildError("Bundle metadata is missing identity fields") from exc
    if computed_bundle_id != bundle_id:
        raise ManifestBuildError("Bundle ID does not match deterministic bundle content")
    samples = pq.read_table(directory / SAMPLES_FILENAME)
    labels = pq.read_table(directory / LABELS_FILENAME)
    annotations = pq.read_table(directory / ANNOTATIONS_FILENAME)
    splits = pq.read_table(directory / SPLITS_FILENAME)
    source_inventory = pq.read_table(directory / SOURCE_INVENTORY_FILENAME)
    validate_sample_table(samples)
    validate_label_table(labels, samples)
    validate_annotation_table(annotations, samples, labels)
    split_metadata = metadata.get("split")
    if not isinstance(split_metadata, dict):
        raise ManifestBuildError("Bundle metadata is missing split lineage")
    validate_split_table(splits, samples, labels, config=split_config)
    assignments = {row["sample_id"]: row["split_name"] for row in splits.to_pylist()}
    if split_metadata.get("split_recipe_id") != split_config.recipe_id:
        raise ManifestBuildError("Split metadata recipe does not match its policy")
    if split_metadata.get("split_assignment_id") != split_assignment_id(assignments):
        raise ManifestBuildError("Split assignment identity does not match the split artifact")
    _validate_source_inventory(source_inventory, samples)
    _validate_metadata_counts(metadata, samples, labels, annotations, source_inventory)
    return metadata


def validate_bundle_reference(
    bundle_directory: str | Path,
    *,
    expected_bundle_id: str | None = None,
    expected_manifest_sha256: str | None = None,
    enforce_directory_name: bool = True,
) -> ValidatedBundleReference:
    """Validate a bundle reference and return its manifest and exact byte hash."""
    directory = Path(bundle_directory)
    expected_files = {
        SAMPLES_FILENAME: RSNA_SAMPLE_SCHEMA,
        LABELS_FILENAME: RSNA_LABEL_SCHEMA,
        ANNOTATIONS_FILENAME: RSNA_ANNOTATION_SCHEMA,
        SPLITS_FILENAME: RSNA_SPLIT_SCHEMA,
        SOURCE_INVENTORY_FILENAME: RSNA_SOURCE_INVENTORY_SCHEMA,
    }
    _require_exact_regular_entries(directory, {METADATA_FILENAME, *expected_files})
    metadata_path = directory / METADATA_FILENAME
    try:
        metadata_bytes = metadata_path.read_bytes()
    except OSError as exc:
        raise ManifestBuildError(f"Bundle metadata is unreadable: {metadata_path}") from exc
    observed_manifest_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    if expected_manifest_sha256 is not None:
        if not _sha256_text(expected_manifest_sha256):
            raise ManifestBuildError("Expected bundle-manifest SHA-256 is invalid")
        if observed_manifest_sha256 != expected_manifest_sha256:
            raise ManifestBuildError("Bundle-manifest SHA-256 does not match expected lineage")
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError(f"Bundle metadata is unreadable: {metadata_path}") from exc
    bundle_id = metadata.get("bundle", {}).get("bundle_id")
    if expected_bundle_id is not None and bundle_id != expected_bundle_id:
        raise ManifestBuildError(
            f"Bundle metadata ID {bundle_id!r} does not match {expected_bundle_id!r}"
        )
    schema_version = metadata.get("manifest_schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ManifestBuildError(f"Unsupported manifest_schema_version: {schema_version!r}")
    _validate_manifest_contract(metadata)
    hashes = metadata.get("generated_artifact_hashes", {})
    if not isinstance(hashes, dict) or set(hashes) != set(expected_files):
        raise ManifestBuildError("Bundle metadata declares an unexpected artifact set")
    declared_arrow_hashes: dict[str, str] = {}
    for filename, schema in expected_files.items():
        path = directory / filename
        declared = hashes.get(filename)
        if not isinstance(declared, dict):
            raise ManifestBuildError(f"Bundle is incomplete: {filename}")
        if sha256_file(path) != declared.get("file_sha256"):
            raise ManifestBuildError(f"File hash mismatch for {filename}")
        if pq.read_schema(path) != schema:
            raise ManifestBuildError(f"{filename} schema mismatch")
        arrow_hash = declared.get("arrow_ipc_sha256")
        if not _sha256_text(arrow_hash):
            raise ManifestBuildError(f"Arrow IPC hash declaration is invalid for {filename}")
        declared_arrow_hashes[filename] = arrow_hash
    try:
        computed_bundle_id = _bundle_id(declared_arrow_hashes, metadata)
    except (KeyError, TypeError) as exc:
        raise ManifestBuildError("Bundle metadata is missing identity fields") from exc
    if computed_bundle_id != bundle_id:
        raise ManifestBuildError("Bundle ID does not match deterministic bundle content")
    if enforce_directory_name and directory.name != bundle_id:
        raise ManifestBuildError("Bundle directory name does not match manifest identity")
    return ValidatedBundleReference(metadata, observed_manifest_sha256)


def _require_exact_regular_entries(directory: Path, required_names: set[str]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ManifestBuildError(f"Bundle path is not a physical directory: {directory}")
    with os.scandir(directory) as entries:
        inspected = list(entries)
    actual_names = {entry.name for entry in inspected}
    if actual_names != required_names:
        raise ManifestBuildError("Bundle directory contains an unexpected artifact set")
    for entry in inspected:
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise ManifestBuildError(
                f"Bundle entry must be a regular non-symlink file: {entry.name}"
            )


def _validate_manifest_contract(metadata: Mapping[str, Any]) -> SplitConfig:
    tasks = metadata.get("tasks")
    if tasks != _task_definitions():
        raise ManifestBuildError("Bundle metadata declares an unsupported task contract")
    if metadata.get("privacy") != {
        "classification": "protected patient-level data",
        "public_reporting": "aggregate only",
    }:
        raise ManifestBuildError("Bundle metadata declares an unsupported privacy classification")
    split = metadata.get("split")
    if not isinstance(split, dict):
        raise ManifestBuildError("Bundle metadata is missing split lineage")
    if set(split) != _SPLIT_METADATA_FIELDS:
        raise ManifestBuildError("Bundle metadata contains invalid split metadata fields")
    expected_split_policy = {
        "split_source": SPLIT_SOURCE,
        "algorithm_version": SPLIT_ALGORITHM_VERSION,
        "patient_grouping_rule": PATIENT_GROUPING_RULE,
        "patient_target_consistency_rule": PATIENT_TARGET_CONSISTENCY_RULE,
        "ranking_rule": PATIENT_RANKING_RULE,
        "patient_hash_algorithm": PATIENT_HASH_ALGORITHM,
        "patient_hash_input_encoding": PATIENT_HASH_INPUT_ENCODING,
        "patient_hash_input_template": PATIENT_HASH_INPUT_TEMPLATE,
        "allocation_rule": SPLIT_ALLOCATION_RULE,
        "split_order": list(SPLIT_NAMES),
        "stratification_target": STRATIFICATION_TARGET,
    }
    if any(split.get(key) != value for key, value in expected_split_policy.items()):
        raise ManifestBuildError("Bundle metadata contains an unsupported split algorithm")
    return _split_config_from_metadata(split)


def _label_records(sample_id: str, target: int, rsna_class: str) -> list[LabelRecord]:
    return [
        LabelRecord(
            sample_id,
            PNEUMONIA_TASK_ID,
            target,
        ),
        LabelRecord(
            sample_id,
            RSNA_CLASS_TASK_ID,
            RSNA_CLASS_VALUES[rsna_class],
        ),
    ]


def _task_definitions() -> dict[str, dict[str, Any]]:
    return {
        PNEUMONIA_TASK_ID: {
            "label_source": PNEUMONIA_LABEL_SOURCE,
            "label_policy_version": PNEUMONIA_LABEL_POLICY_VERSION,
            "label_values": [0, 1],
            "label_meanings": {"0": "negative", "1": "positive"},
            "status_semantics": "observed challenge label",
            "exclusion_rules": "none within the labeled Stage 2 cohort",
        },
        RSNA_CLASS_TASK_ID: {
            "label_source": RSNA_CLASS_LABEL_SOURCE,
            "label_policy_version": RSNA_CLASS_LABEL_POLICY_VERSION,
            "label_values": sorted(RSNA_CLASS_VALUES.values()),
            "label_meanings": {
                str(value): name for name, value in sorted(RSNA_CLASS_VALUES.items())
            },
            "status_semantics": "observed detailed class label",
            "exclusion_rules": "none within the labeled Stage 2 cohort",
        },
    }


def _annotation_records(
    sample_id: str,
    image_id: str,
    target: int,
    boxes: tuple[BoundingBox, ...],
    dimensions: tuple[int, int],
) -> list[AnnotationRecord]:
    if target == 0:
        if boxes:
            raise ManifestBuildError(f"Negative sample {sample_id!r} must not contain boxes")
        return []
    records: list[AnnotationRecord] = []
    for index, box in enumerate(boxes):
        validate_box_bounds(box, dimensions, sample_id)
        records.append(
            AnnotationRecord(
                sample_id,
                f"{DATASET_ID}:{image_id}:bbox:{index:04d}",
                box.x,
                box.y,
                box.width,
                box.height,
            )
        )
    return records


def _build_aggregate_metadata(
    paths: RsnaPaths,
    samples: pa.Table,
    labels: pa.Table,
    annotations: pa.Table,
    splits: pa.Table,
    source_inventory: pa.Table,
    audit: AuditAccumulator,
    split_config: SplitConfig,
) -> dict[str, Any]:
    label_rows = labels.to_pylist()
    pneumonia = Counter(
        str(row["label_value"]) for row in label_rows if row["task_id"] == PNEUMONIA_TASK_ID
    )
    assignments = {row["sample_id"]: row["split_name"] for row in splits.to_pylist()}
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "tasks": _task_definitions(),
        "sample_count": samples.num_rows,
        "label_count": labels.num_rows,
        "positive_count": pneumonia["1"],
        "negative_count": pneumonia["0"],
        "annotation_count": annotations.num_rows,
        "source_inventory_count": source_inventory.num_rows,
        "split": {
            "split_source": SPLIT_SOURCE,
            "split_recipe_id": split_config.recipe_id,
            "split_assignment_id": split_assignment_id(assignments),
            "algorithm_version": SPLIT_ALGORITHM_VERSION,
            "patient_grouping_rule": PATIENT_GROUPING_RULE,
            "patient_target_consistency_rule": PATIENT_TARGET_CONSISTENCY_RULE,
            "ranking_rule": PATIENT_RANKING_RULE,
            "patient_hash_algorithm": PATIENT_HASH_ALGORITHM,
            "patient_hash_input_encoding": PATIENT_HASH_INPUT_ENCODING,
            "patient_hash_input_template": PATIENT_HASH_INPUT_TEMPLATE,
            "seed": split_config.seed,
            "stratification_target": STRATIFICATION_TARGET,
            "allocation_rule": SPLIT_ALLOCATION_RULE,
            "split_order": list(SPLIT_NAMES),
            "ratios": split_config.recipe_payload["ratios"],
        },
        "age_parsing_summary": {
            "status_counts": _sorted_counter(audit.age_status),
            "source_format_counts": _sorted_counter(audit.age_source_format),
            "warning_counts": _sorted_counter(audit.age_warnings),
        },
        "implausible_age_count": audit.implausible_age_count,
        "dicom_audit": {
            "field_value_counts": {
                keyword: _sorted_counter(counter)
                for keyword, counter in sorted(audit.dicom_values.items())
            },
            "uid_uniqueness": {
                "sop_instance_uid_unique_count": len(audit.sop_instance_uids),
                "study_instance_uid_unique_count": len(audit.study_instance_uids),
                "series_instance_uid_unique_count": len(audit.series_instance_uids),
                "media_storage_sop_instance_uid_unique_count": len(
                    audit.media_storage_sop_instance_uids
                ),
                "media_storage_matches_sop_count": audit.media_sop_matches,
            },
        },
        "source_file_hashes": {
            paths.labels.name: {"algorithm": "sha256", "file_sha256": sha256_file(paths.labels)},
            paths.class_info.name: {
                "algorithm": "sha256",
                "file_sha256": sha256_file(paths.class_info),
            },
        },
        "hash_policy": {
            "source_files": "SHA-256 over source CSV and DICOM bytes",
            "arrow_ipc_sha256": "SHA-256 over ordered Arrow IPC stream including exact schema",
            "artifact_files": "SHA-256 over generated Parquet bytes",
            "dicom_files": "SHA-256 and byte size for every source DICOM",
        },
        "privacy": {
            "classification": "protected patient-level data",
            "public_reporting": "aggregate only",
        },
        "provenance": {
            "tool_versions": {
                "python": platform.python_version(),
                "pydicom": pydicom.__version__,
                "pyarrow": pa.__version__,
                "pandas": pd.__version__,
            },
            "arrow_ipc_runtime": {
                "pyarrow_version": pa.__version__,
                "stability_scope": (
                    "Deterministic only for the recorded PyArrow version; cross-version "
                    "stability is not claimed"
                ),
            },
        },
    }


def _finalize_metadata(
    base: Mapping[str, Any],
    bundle_id: str,
    paths: BundlePaths,
    arrow_hashes: Mapping[str, str],
) -> dict[str, Any]:
    artifact_paths = {
        SAMPLES_FILENAME: paths.samples_path,
        LABELS_FILENAME: paths.labels_path,
        ANNOTATIONS_FILENAME: paths.annotations_path,
        SPLITS_FILENAME: paths.splits_path,
        SOURCE_INVENTORY_FILENAME: paths.source_inventory_path,
    }
    return {
        **base,
        "bundle": {
            "bundle_id": bundle_id,
            "publication_model": "immutable-build-directory-with-atomic-CURRENT-marker",
        },
        "generated_artifact_hashes": {
            filename: {
                "arrow_ipc_sha256": arrow_hashes[filename],
                "file_sha256": sha256_file(path),
            }
            for filename, path in artifact_paths.items()
        },
        "generation": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "command": "python -m radfusion.data.rsna_manifest",
        },
    }


def _bundle_id(
    arrow_hashes: Mapping[str, str],
    deterministic_metadata: Mapping[str, Any],
) -> str:
    identity = _bundle_identity_payload(arrow_hashes, deterministic_metadata)
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"build-{digest}"


def _bundle_identity_payload(
    arrow_hashes: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the explicit semantic identity payload for an RSNA bundle."""
    split = metadata["split"]
    return {
        "identity_policy_version": BUNDLE_IDENTITY_POLICY_VERSION,
        "manifest_schema_version": metadata["manifest_schema_version"],
        "dataset_id": metadata["dataset_id"],
        "dataset_version": metadata["dataset_version"],
        "tasks": metadata["tasks"],
        "split": {
            "split_source": split["split_source"],
            "algorithm_version": split["algorithm_version"],
            "seed": split["seed"],
            "stratification_target": split["stratification_target"],
            "ratios": split["ratios"],
        },
        "source_file_hashes": metadata["source_file_hashes"],
        "arrow_ipc_sha256": dict(sorted(arrow_hashes.items())),
    }


def _bundle_paths(bundle_id: str, directory: Path, current_path: Path) -> BundlePaths:
    return BundlePaths(
        bundle_id,
        directory,
        directory / SAMPLES_FILENAME,
        directory / LABELS_FILENAME,
        directory / ANNOTATIONS_FILENAME,
        directory / SPLITS_FILENAME,
        directory / SOURCE_INVENTORY_FILENAME,
        directory / METADATA_FILENAME,
        current_path,
    )


def _sha256_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _label_sort_key(record: LabelRecord) -> tuple[str, str]:
    return record.sample_id, record.task_id


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _split_config_from_metadata(metadata: Mapping[str, Any]) -> SplitConfig:
    ratios = metadata.get("ratios")
    if not isinstance(ratios, list) or len(ratios) != len(SPLIT_NAMES):
        raise ManifestBuildError("Bundle metadata contains invalid ordered split ratios")
    values: list[float] = []
    for expected_name, entry in zip(SPLIT_NAMES, ratios, strict=True):
        if not isinstance(entry, dict) or set(entry) != {"split_name", "ratio"}:
            raise ManifestBuildError("Bundle metadata contains invalid ordered split ratios")
        if entry.get("split_name") != expected_name:
            raise ManifestBuildError("Bundle metadata contains invalid ordered split ratios")
        ratio = entry.get("ratio")
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, int | float)
            or not math.isfinite(ratio)
        ):
            raise ManifestBuildError("Bundle metadata contains invalid ordered split ratios")
        values.append(ratio)
    try:
        config = SplitConfig(
            seed=metadata["seed"],
            train_ratio=values[0],
            validation_ratio=values[1],
            test_ratio=values[2],
        )
    except (KeyError, TypeError) as exc:
        raise ManifestBuildError("Bundle metadata contains an invalid split policy") from exc
    config.validate()
    return config


def _validate_source_inventory(
    inventory: pa.Table,
    samples: pa.Table,
    dataset_root: Path | None = None,
) -> None:
    try:
        require_exact_schema(inventory, RSNA_SOURCE_INVENTORY_SCHEMA, "RSNA source inventory")
    except ValueError as exc:
        raise ManifestBuildError(str(exc)) from exc
    rows = inventory.to_pylist()
    sample_rows = samples.to_pylist()
    expected = {row["sample_id"]: row["image_path"] for row in sample_rows}
    ids = [row["sample_id"] for row in rows]
    if ids != sorted(ids) or len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ManifestBuildError("Source inventory must provide ordered one-to-one sample coverage")
    for row in rows:
        if row["relative_path"] != expected[row["sample_id"]]:
            raise ManifestBuildError("Source inventory path does not match the sample artifact")
        relative_path = PurePosixPath(row["relative_path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ManifestBuildError("Source inventory contains an unsafe relative path")
        digest = row["sha256"]
        if (
            row["byte_size"] <= 0
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ManifestBuildError("Source inventory contains an invalid size or SHA-256")
        if dataset_root is not None:
            path = resolve_image_path(dataset_root, relative_path)
            if path.stat().st_size != row["byte_size"] or sha256_file(path) != row["sha256"]:
                raise ManifestBuildError(
                    f"Source DICOM authentication failed: {row['relative_path']}"
                )


def _validate_metadata_counts(
    metadata: Mapping[str, Any],
    samples: pa.Table,
    labels: pa.Table,
    annotations: pa.Table,
    source_inventory: pa.Table,
) -> None:
    targets = {
        row["sample_id"]: row["label_value"]
        for row in labels.to_pylist()
        if row["task_id"] == PNEUMONIA_TASK_ID
    }
    expected = {
        "sample_count": samples.num_rows,
        "label_count": labels.num_rows,
        "positive_count": sum(value == 1 for value in targets.values()),
        "negative_count": sum(value == 0 for value in targets.values()),
        "annotation_count": annotations.num_rows,
        "source_inventory_count": source_inventory.num_rows,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ManifestBuildError(f"Bundle metadata {field} does not match artifact content")
