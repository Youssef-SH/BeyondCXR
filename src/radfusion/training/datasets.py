"""Load approved RSNA model inputs from an exact bundle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from radfusion.data.dicom_loader import DicomRecord, read_dicom
from radfusion.data.hashing import sha256_file
from radfusion.data.rsna_artifacts import (
    ANNOTATIONS_FILENAME,
    BUNDLES_DIRECTORY,
    LABELS_FILENAME,
    METADATA_FILENAME,
    SAMPLES_FILENAME,
    SOURCE_INVENTORY_FILENAME,
    SPLITS_FILENAME,
    validate_bundle_directory,
    validate_bundle_reference,
)
from radfusion.data.rsna_source import ManifestBuildError, resolve_image_path
from radfusion.data.tabular_preprocess import SOURCE_FEATURES
from radfusion.evaluation.metrics import validated_binary_targets
from radfusion.training.config import DatasetConfig
from radfusion.training.interfaces import DatasetLineage, DatasetPartition, DatasetRunData
from radfusion.utils.operational_logging import CountProgress, get_operational_logger, log_event

_IMAGE_FRAME_COLUMNS = ("sample_id", "patient_id", "image_path", "split_name", "target")
_FUSION_FRAME_COLUMNS = (
    "sample_id",
    "patient_id",
    "image_path",
    *SOURCE_FEATURES,
    "split_name",
    "target",
)
SOURCE_AUTHENTICATION_POLICY_VERSION = "partition-inventory-sha256-v1"
_LOGGER = get_operational_logger(__name__)


class ImageSample(TypedDict):
    """One lazily decoded image example."""

    image: torch.Tensor
    target: torch.Tensor
    sample_id: str
    patient_id: str


class FusionSample(ImageSample):
    """One lazily decoded image aligned with one transformed metadata row."""

    structured: torch.Tensor


@dataclass(frozen=True)
class _ImageRow:
    sample_id: str
    patient_id: str
    image_path: PurePosixPath
    target: int


@dataclass(frozen=True)
class SourceAuthentication:
    """Deterministic proof that permitted source bytes match the bundle inventory."""

    policy_version: str
    partitions: tuple[str, ...]
    file_count: int
    source_inventory_arrow_sha256: str
    source_inventory_file_sha256: str
    authenticated_rows_sha256: str
    success: bool = True

    def as_dict(self) -> dict[str, object]:
        """Return serializable source-authentication provenance."""
        return {
            "policy_version": self.policy_version,
            "partitions": list(self.partitions),
            "file_count": self.file_count,
            "source_inventory_arrow_sha256": self.source_inventory_arrow_sha256,
            "source_inventory_file_sha256": self.source_inventory_file_sha256,
            "authenticated_rows_sha256": self.authenticated_rows_sha256,
            "success": self.success,
        }


@dataclass(frozen=True)
class ImageRunData:
    """Authenticated train and validation rows for one image-training run."""

    train: pd.DataFrame
    validation: pd.DataFrame
    lineage: DatasetLineage
    bundle_manifest_sha256: str
    authentication: SourceAuthentication


@dataclass(frozen=True)
class ImageTestData:
    """Authenticated test rows for one explicit image-evaluation run."""

    test: pd.DataFrame
    lineage: DatasetLineage
    bundle_manifest_sha256: str
    authentication: SourceAuthentication


@dataclass(frozen=True)
class FusionRunData:
    """Authenticated aligned train and validation rows for fusion training."""

    train: pd.DataFrame
    validation: pd.DataFrame
    lineage: DatasetLineage
    bundle_manifest_sha256: str
    authentication: SourceAuthentication


@dataclass(frozen=True)
class FusionTestData:
    """Authenticated aligned test rows for explicit fusion evaluation."""

    test: pd.DataFrame
    lineage: DatasetLineage
    bundle_manifest_sha256: str
    authentication: SourceAuthentication


@dataclass(frozen=True)
class LocalizationTestData:
    """Authenticated image test rows with positive-case box geometry."""

    images: ImageTestData
    dimensions: pd.DataFrame
    annotations: pd.DataFrame


class RsnaImageDataset(Dataset[ImageSample]):
    """Lazily decode a validated, deterministically ordered RSNA partition."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        dataset_root: str | Path,
        partition: str,
        transform: Callable[[np.ndarray], torch.Tensor],
        decoder: Callable[[str | Path], tuple[np.ndarray, DicomRecord]] = read_dicom,
    ) -> None:
        if tuple(frame.columns) != _IMAGE_FRAME_COLUMNS:
            raise ManifestBuildError(
                f"RSNA image frame columns must be exactly {_IMAGE_FRAME_COLUMNS}"
            )
        if partition not in {"train", "validation", "test"}:
            raise ManifestBuildError(f"Unsupported RSNA image partition: {partition!r}")
        if frame.empty:
            raise ManifestBuildError(f"RSNA image partition {partition!r} is empty")

        self._dataset_root = Path(dataset_root)
        self._transform = transform
        self._decoder = decoder
        rows: list[_ImageRow] = []
        for row in frame.to_dict(orient="records"):
            sample_id = _nonempty_text(row["sample_id"], "sample_id")
            patient_id = _nonempty_text(row["patient_id"], "patient_id")
            if row["split_name"] != partition:
                raise ManifestBuildError("RSNA image frame does not match the requested partition")
            relative_path = _validated_image_path(row["image_path"])
            resolve_image_path(self._dataset_root, relative_path)
            target = row["target"]
            if (
                isinstance(target, bool)
                or not isinstance(target, int | np.integer)
                or target not in {0, 1}
            ):
                raise ManifestBuildError(f"Invalid binary target for sample {sample_id!r}")
            rows.append(_ImageRow(sample_id, patient_id, relative_path, int(target)))
        sample_ids = [row.sample_id for row in rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise ManifestBuildError("RSNA image sample_id values must be unique")
        if sample_ids != sorted(sample_ids):
            raise ManifestBuildError("RSNA image samples must be ordered by sample_id")
        self._rows = tuple(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> ImageSample:
        row = self._rows[index]
        path = resolve_image_path(self._dataset_root, row.image_path)
        pixels, record = self._decoder(path)
        if record.patient_id != row.patient_id:
            raise ManifestBuildError(
                f"Decoded DICOM patient does not match sample {row.sample_id!r}"
            )
        image = self._transform(pixels)
        if (
            not isinstance(image, torch.Tensor)
            or image.dtype != torch.float32
            or image.shape != (1, 224, 224)
            or not torch.isfinite(image).all()
        ):
            raise ManifestBuildError(
                f"Image transform returned an invalid tensor for sample {row.sample_id!r}"
            )
        return {
            "image": image,
            "target": torch.tensor(float(row.target), dtype=torch.float32),
            "sample_id": row.sample_id,
            "patient_id": row.patient_id,
        }


class RsnaFusionDataset(Dataset[FusionSample]):
    """Expose aligned RSNA images and eagerly transformed metadata tensors."""

    def __init__(
        self,
        frame: pd.DataFrame,
        structured: np.ndarray,
        *,
        structured_sample_ids: tuple[str, ...],
        dataset_root: str | Path,
        partition: str,
        transform: Callable[[np.ndarray], torch.Tensor],
        decoder: Callable[[str | Path], tuple[np.ndarray, DicomRecord]] = read_dicom,
    ) -> None:
        if tuple(frame.columns) != _FUSION_FRAME_COLUMNS:
            raise ManifestBuildError(
                f"RSNA fusion frame columns must be exactly {_FUSION_FRAME_COLUMNS}"
            )
        sample_ids = tuple(frame["sample_id"].astype(str))
        if sample_ids != structured_sample_ids:
            raise ManifestBuildError("Structured rows are not aligned with fusion sample IDs")
        try:
            matrix = np.asarray(structured, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ManifestBuildError("Fusion structured values are not numeric") from exc
        if (
            matrix.ndim != 2
            or matrix.shape[0] != len(frame)
            or matrix.shape[1] == 0
            or not np.isfinite(matrix).all()
        ):
            raise ManifestBuildError("Fusion structured matrix must be finite N x D data")
        self._structured = torch.from_numpy(np.ascontiguousarray(matrix)).contiguous()
        self._images = RsnaImageDataset(
            frame.loc[:, _IMAGE_FRAME_COLUMNS],
            dataset_root=dataset_root,
            partition=partition,
            transform=transform,
            decoder=decoder,
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> FusionSample:
        image = self._images[index]
        return {**image, "structured": self._structured[index]}


class RsnaDataset:
    """Expose approved RSNA model partitions from a pinned bundle."""

    def load_train_validation(self, config: DatasetConfig) -> DatasetRunData:
        """Load train and validation without reading the test partition."""
        bundle, metadata = _load_pinned_bundle(config)
        frame = _task_frame(bundle, config.task_id, partitions=("train", "validation"))
        train = _partition(frame, "train")
        validation = _partition(frame, "validation")
        return DatasetRunData(
            train=train,
            validation=validation,
            lineage=_lineage(config, metadata),
        )

    def load_lineage(self, config: DatasetConfig) -> DatasetLineage:
        """Validate the pinned bundle and return lineage without reading partitions."""
        _, metadata = _load_pinned_bundle(config)
        return _lineage(config, metadata)

    def load_test(self, config: DatasetConfig) -> tuple[DatasetPartition, DatasetLineage]:
        """Load the test partition from the same pinned bundle."""
        bundle, metadata = _load_pinned_bundle(config)
        frame = _task_frame(bundle, config.task_id, partitions=("test",))
        test = _partition(frame, "test")
        return test, _lineage(config, metadata)

    def load_image_partition_frame(
        self,
        config: DatasetConfig,
        partition: str,
    ) -> tuple[pd.DataFrame, DatasetLineage]:
        """Load approved image rows without decoding DICOM pixels."""
        if partition not in {"train", "validation", "test"}:
            raise ManifestBuildError(f"Unsupported RSNA image partition: {partition!r}")
        bundle, metadata = _load_pinned_bundle(config, materialize_all_rows=False)
        frame = _task_frame(
            bundle,
            config.task_id,
            partitions=(partition,),
            feature_columns=("image_path",),
        )
        return frame.loc[:, _IMAGE_FRAME_COLUMNS].copy(), _lineage(config, metadata)

    def load_image_train_validation(self, config: DatasetConfig) -> ImageRunData:
        """Load and authenticate train and validation image rows only."""
        bundle, metadata = _load_pinned_bundle(config, materialize_all_rows=False)
        frame = _task_frame(
            bundle,
            config.task_id,
            partitions=("train", "validation"),
            feature_columns=("image_path",),
        )
        authentication = _authenticate_source_rows(
            config,
            bundle,
            metadata,
            frame,
            partitions=("train", "validation"),
        )
        manifest_sha256 = _required_manifest_sha256(bundle)
        return ImageRunData(
            train=_image_partition(frame, "train"),
            validation=_image_partition(frame, "validation"),
            lineage=_lineage(config, metadata),
            bundle_manifest_sha256=manifest_sha256,
            authentication=authentication,
        )

    def load_image_test(
        self,
        config: DatasetConfig,
        *,
        expected_manifest_sha256: str,
    ) -> ImageTestData:
        """Load and authenticate test image rows only."""
        bundle, metadata = _load_pinned_bundle(
            config,
            materialize_all_rows=False,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        frame = _task_frame(
            bundle,
            config.task_id,
            partitions=("test",),
            feature_columns=("image_path",),
        )
        authentication = _authenticate_source_rows(
            config,
            bundle,
            metadata,
            frame,
            partitions=("test",),
        )
        manifest_sha256 = _required_manifest_sha256(bundle)
        return ImageTestData(
            test=_image_partition(frame, "test"),
            lineage=_lineage(config, metadata),
            bundle_manifest_sha256=manifest_sha256,
            authentication=authentication,
        )

    def load_fusion_train_validation(self, config: DatasetConfig) -> FusionRunData:
        """Load and authenticate aligned train and validation fusion rows."""
        bundle, metadata = _load_pinned_bundle(config, materialize_all_rows=False)
        frame = _task_frame(
            bundle,
            config.task_id,
            partitions=("train", "validation"),
            feature_columns=("image_path", *SOURCE_FEATURES),
        )
        authentication = _authenticate_source_rows(
            config,
            bundle,
            metadata,
            frame,
            partitions=("train", "validation"),
        )
        return FusionRunData(
            train=_fusion_partition(frame, "train"),
            validation=_fusion_partition(frame, "validation"),
            lineage=_lineage(config, metadata),
            bundle_manifest_sha256=_required_manifest_sha256(bundle),
            authentication=authentication,
        )

    def load_fusion_test(
        self,
        config: DatasetConfig,
        *,
        expected_manifest_sha256: str,
    ) -> FusionTestData:
        """Load and authenticate aligned test fusion rows."""
        bundle, metadata = _load_pinned_bundle(
            config,
            materialize_all_rows=False,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        frame = _task_frame(
            bundle,
            config.task_id,
            partitions=("test",),
            feature_columns=("image_path", *SOURCE_FEATURES),
        )
        authentication = _authenticate_source_rows(
            config,
            bundle,
            metadata,
            frame,
            partitions=("test",),
        )
        return FusionTestData(
            test=_fusion_partition(frame, "test"),
            lineage=_lineage(config, metadata),
            bundle_manifest_sha256=_required_manifest_sha256(bundle),
            authentication=authentication,
        )

    def load_localization_test(
        self,
        config: DatasetConfig,
        *,
        expected_manifest_sha256: str,
    ) -> LocalizationTestData:
        """Load authenticated test images and validated positive-box geometry."""
        images = self.load_image_test(
            config,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        bundle, _ = _load_pinned_bundle(
            config,
            materialize_all_rows=False,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        sample_ids = images.test["sample_id"].astype(str).tolist()
        dimensions = pq.read_table(
            bundle.samples_path,
            columns=["sample_id", "image_rows", "image_columns"],
            filters=[("sample_id", "in", sample_ids)],
        ).to_pandas()
        annotation_path = bundle.metadata_path.parent / ANNOTATIONS_FILENAME
        annotations = pq.read_table(
            annotation_path,
            columns=["sample_id", "annotation_id", "x", "y", "width", "height"],
            filters=[("sample_id", "in", sample_ids)],
        ).to_pandas()
        dimensions = dimensions.sort_values("sample_id", kind="stable").reset_index(drop=True)
        annotations = annotations.sort_values("annotation_id", kind="stable").reset_index(drop=True)
        if dimensions["sample_id"].astype(str).tolist() != sorted(sample_ids) or len(
            dimensions
        ) != len(sample_ids):
            raise ManifestBuildError("Localization dimensions do not cover the test partition")
        positive_ids = set(images.test.loc[images.test["target"] == 1, "sample_id"].astype(str))
        if set(annotations["sample_id"].astype(str)) != positive_ids:
            raise ManifestBuildError("Localization boxes do not cover every positive test sample")
        return LocalizationTestData(images, dimensions, annotations)


@dataclass(frozen=True)
class _PinnedBundlePaths:
    samples_path: Path
    labels_path: Path
    splits_path: Path
    source_inventory_path: Path
    metadata_path: Path
    manifest_sha256: str | None = None


def _load_pinned_bundle(
    config: DatasetConfig,
    *,
    materialize_all_rows: bool = True,
    expected_manifest_sha256: str | None = None,
) -> tuple[_PinnedBundlePaths, dict[str, object]]:
    dataset_root = config.manifest_directory / config.registry_key
    bundle_directory = dataset_root / BUNDLES_DIRECTORY / config.bundle_id
    if materialize_all_rows:
        metadata = validate_bundle_directory(
            bundle_directory,
            expected_bundle_id=config.bundle_id,
        )
        manifest_sha256 = None
    else:
        validated = validate_bundle_reference(
            bundle_directory,
            expected_bundle_id=config.bundle_id,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        metadata = dict(validated.manifest)
        manifest_sha256 = validated.manifest_sha256
    return (
        _PinnedBundlePaths(
            bundle_directory / SAMPLES_FILENAME,
            bundle_directory / LABELS_FILENAME,
            bundle_directory / SPLITS_FILENAME,
            bundle_directory / SOURCE_INVENTORY_FILENAME,
            bundle_directory / METADATA_FILENAME,
            manifest_sha256,
        ),
        metadata,
    )


def _required_manifest_sha256(bundle: _PinnedBundlePaths) -> str:
    if bundle.manifest_sha256 is None:
        raise ManifestBuildError("Image bundle validation did not return manifest byte identity")
    return bundle.manifest_sha256


def _task_frame(
    bundle: _PinnedBundlePaths,
    task_id: str,
    *,
    partitions: tuple[str, ...],
    feature_columns: tuple[str, ...] = SOURCE_FEATURES,
) -> pd.DataFrame:
    assignments = pq.read_table(
        bundle.splits_path,
        columns=["sample_id", "split_name"],
        filters=[("split_name", "in", list(partitions))],
    ).to_pandas()
    selected_ids = assignments["sample_id"].astype(str).tolist()
    sample_columns = ["sample_id", "patient_id", *feature_columns]
    samples = pq.read_table(
        bundle.samples_path,
        columns=sample_columns,
        filters=[("sample_id", "in", selected_ids)],
    ).to_pandas()
    target = pq.read_table(
        bundle.labels_path,
        columns=["sample_id", "label_value"],
        filters=[
            ("task_id", "=", task_id),
            ("sample_id", "in", selected_ids),
        ],
    ).to_pandas()
    if target.empty:
        raise ManifestBuildError(f"Bundle does not contain configured task {task_id!r}")
    return (
        samples.merge(assignments, on="sample_id", validate="one_to_one")
        .merge(
            target.rename(columns={"label_value": "target"}), on="sample_id", validate="one_to_one"
        )
        .sort_values("sample_id", kind="stable")
        .reset_index(drop=True)
    )


def _partition(frame: pd.DataFrame, name: str) -> DatasetPartition:
    selected = frame.loc[frame["split_name"] == name]
    if selected.empty:
        raise ManifestBuildError(f"Bundle partition {name!r} is empty")
    targets = validated_binary_targets(selected["target"].to_numpy())
    features = selected.loc[:, SOURCE_FEATURES].copy()
    if tuple(features.columns) != SOURCE_FEATURES:
        raise ManifestBuildError("RSNA metadata feature contract is invalid")
    return DatasetPartition(
        features=features,
        targets=targets,
        sample_ids=tuple(selected["sample_id"].astype(str)),
        patient_ids=tuple(selected["patient_id"].astype(str)),
        partition=name,
    )


def _image_partition(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    selected = frame.loc[frame["split_name"] == name, _IMAGE_FRAME_COLUMNS].copy()
    if selected.empty:
        raise ManifestBuildError(f"Bundle partition {name!r} is empty")
    return selected.reset_index(drop=True)


def _fusion_partition(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    selected = frame.loc[frame["split_name"] == name, _FUSION_FRAME_COLUMNS].copy()
    if selected.empty:
        raise ManifestBuildError(f"Bundle partition {name!r} is empty")
    sample_ids = selected["sample_id"].astype(str).tolist()
    patient_ids = selected["patient_id"].astype(str).tolist()
    targets = validated_binary_targets(selected["target"].to_numpy())
    if (
        len(sample_ids) != len(set(sample_ids))
        or sample_ids != sorted(sample_ids)
        or any(not value for value in patient_ids)
        or len(targets) != len(selected)
        or set(selected["split_name"].astype(str)) != {name}
    ):
        raise ManifestBuildError("RSNA fusion row alignment contract is invalid")
    selected["target"] = targets
    return selected.reset_index(drop=True)


def _authenticate_source_rows(
    config: DatasetConfig,
    bundle: _PinnedBundlePaths,
    metadata: dict[str, object],
    frame: pd.DataFrame,
    *,
    partitions: tuple[str, ...],
) -> SourceAuthentication:
    if config.dataset_root is None:
        raise ManifestBuildError("Image source authentication requires dataset.dataset_root")
    actual_partitions = tuple(dict.fromkeys(frame["split_name"].astype(str)))
    if set(actual_partitions) != set(partitions):
        raise ManifestBuildError("Image rows do not match the permitted source partitions")
    sample_ids = frame["sample_id"].astype(str).tolist()
    inventory = pq.read_table(
        bundle.source_inventory_path,
        columns=["sample_id", "relative_path", "byte_size", "sha256"],
        filters=[("sample_id", "in", sample_ids)],
    ).to_pandas()
    inventory_ids = inventory["sample_id"].astype(str).tolist()
    if (
        len(inventory_ids) != len(set(inventory_ids))
        or len(inventory_ids) != len(sample_ids)
        or set(inventory_ids) != set(sample_ids)
    ):
        raise ManifestBuildError("Source inventory does not provide one row per permitted image")
    rows = inventory.sort_values("sample_id", kind="stable").to_dict(orient="records")
    log_event(
        _LOGGER,
        "source_authentication_started",
        partition_count=len(partitions),
        total=len(rows),
        unit="files",
    )
    progress = (
        CountProgress(
            _LOGGER,
            "source_authentication_progress",
            total=len(rows),
            unit="files",
        )
        if rows
        else None
    )
    expected_paths = dict(zip(frame["sample_id"], frame["image_path"], strict=True))
    canonical_rows: list[dict[str, object]] = []
    for completed, row in enumerate(rows, start=1):
        relative = _validated_image_path(row["relative_path"])
        if relative.as_posix() != expected_paths[row["sample_id"]]:
            raise ManifestBuildError("Source inventory path differs from the permitted image row")
        byte_size = row["byte_size"]
        digest = row["sha256"]
        if (
            isinstance(byte_size, bool)
            or not isinstance(byte_size, int)
            or byte_size <= 0
            or not _is_sha256(digest)
        ):
            raise ManifestBuildError("Source inventory contains an invalid size or SHA-256")
        source_path = resolve_image_path(config.dataset_root, relative)
        if not source_path.is_file() or source_path.stat().st_size != byte_size:
            raise ManifestBuildError(f"Source DICOM size authentication failed: {relative}")
        if sha256_file(source_path) != digest:
            raise ManifestBuildError(f"Source DICOM SHA-256 authentication failed: {relative}")
        canonical_rows.append(
            {
                "sample_id": row["sample_id"],
                "relative_path": relative.as_posix(),
                "byte_size": byte_size,
                "sha256": digest,
            }
        )
        if progress is not None:
            progress.update(completed)
    hashes = metadata.get("generated_artifact_hashes")
    declared = hashes.get(SOURCE_INVENTORY_FILENAME) if isinstance(hashes, dict) else None
    if not isinstance(declared, dict):
        raise ManifestBuildError("Bundle metadata is missing source-inventory identity")
    arrow_hash = declared.get("arrow_ipc_sha256")
    file_hash = declared.get("file_sha256")
    if not _is_sha256(arrow_hash) or not _is_sha256(file_hash):
        raise ManifestBuildError("Bundle metadata source-inventory hashes are invalid")
    encoded = json.dumps(
        canonical_rows,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    result = SourceAuthentication(
        policy_version=SOURCE_AUTHENTICATION_POLICY_VERSION,
        partitions=partitions,
        file_count=len(canonical_rows),
        source_inventory_arrow_sha256=arrow_hash,
        source_inventory_file_sha256=file_hash,
        authenticated_rows_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    log_event(
        _LOGGER,
        "source_authentication_completed",
        partition_count=len(partitions),
        total=result.file_count,
        unit="files",
    )
    return result


def _lineage(
    config: DatasetConfig,
    metadata: dict[str, object],
) -> DatasetLineage:
    split = metadata["split"]
    tasks = metadata["tasks"]
    return DatasetLineage(
        bundle_id=config.bundle_id,
        split_assignment_id=str(split["split_assignment_id"]),
        label_policy_version=str(tasks[config.task_id]["label_policy_version"]),
        task_id=config.task_id,
    )


def _nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestBuildError(f"RSNA image {field} must be a non-empty string")
    return value


def _validated_image_path(value: object) -> PurePosixPath:
    text = _nonempty_text(value, "image_path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in text
    ):
        raise ManifestBuildError(f"Invalid normalized relative image path: {text!r}")
    return path


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
