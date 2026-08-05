"""Load approved RSNA model inputs from an exact bundle."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from radfusion.data.cxr_cache import (
    CXR_CACHE_FRAME_COLUMNS,
    CxrCacheIdentity,
    ValidatedCxrCache,
    build_cxr_cache,
    preprocessing_identity,
)
from radfusion.data.cxr_transforms import StandardCxrTransform
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
from radfusion.data.rsna_source import ManifestBuildError
from radfusion.data.tabular_preprocess import SOURCE_FEATURES
from radfusion.evaluation.metrics import validated_binary_targets
from radfusion.training.config import DatasetConfig
from radfusion.training.interfaces import DatasetLineage, DatasetPartition, DatasetRunData

_IMAGE_FRAME_COLUMNS = ("sample_id", "patient_id", "image_path", "split_name", "target")
_FUSION_FRAME_COLUMNS = (
    "sample_id",
    "patient_id",
    "image_path",
    *SOURCE_FEATURES,
    "split_name",
    "target",
)


class ImageSample(TypedDict):
    """One cache-backed image example."""

    image: torch.Tensor
    target: torch.Tensor
    sample_id: str
    patient_id: str


class FusionSample(ImageSample):
    """One cached image aligned with one transformed metadata row."""

    structured: torch.Tensor


class CalibrationSample(TypedDict):
    """One task-label-free cached image used for loader calibration."""

    image: torch.Tensor
    sample_id: str


@dataclass(frozen=True)
class SourceInventoryIdentity:
    """Exact bundle source-inventory identity used by a cache-backed consumer."""

    source_inventory_arrow_sha256: str
    source_inventory_file_sha256: str


@dataclass(frozen=True)
class ImageRunData:
    """Train and validation rows for one cache-backed image-training run."""

    train: pd.DataFrame
    validation: pd.DataFrame
    lineage: DatasetLineage
    bundle_manifest_sha256: str
    source_inventory: SourceInventoryIdentity


@dataclass(frozen=True)
class ImageTestData:
    """Test rows for one explicit cache-backed image-evaluation run."""

    test: pd.DataFrame
    lineage: DatasetLineage
    bundle_manifest_sha256: str
    source_inventory: SourceInventoryIdentity


@dataclass(frozen=True)
class ImageCacheData:
    """Source-inventory-bound rows for one complete deterministic image cache."""

    frame: pd.DataFrame
    lineage: DatasetLineage
    bundle_manifest_sha256: str
    source_inventory: SourceInventoryIdentity


@dataclass(frozen=True)
class FusionRunData:
    """Aligned train and validation rows for cache-backed fusion training."""

    train: pd.DataFrame
    validation: pd.DataFrame
    lineage: DatasetLineage
    bundle_manifest_sha256: str
    source_inventory: SourceInventoryIdentity


@dataclass(frozen=True)
class FusionTestData:
    """Aligned test rows for explicit cache-backed fusion evaluation."""

    test: pd.DataFrame
    lineage: DatasetLineage
    bundle_manifest_sha256: str
    source_inventory: SourceInventoryIdentity


@dataclass(frozen=True)
class LocalizationTestData:
    """Cache-backed image test rows with positive-case box geometry."""

    images: ImageTestData
    dimensions: pd.DataFrame
    annotations: pd.DataFrame


class RsnaCachedImageDataset(Dataset[ImageSample]):
    """Load deterministic RSNA bases from cache and apply live transforms."""

    epoch_tagged_requests = True

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        cache: ValidatedCxrCache,
        expected_cache_identity: CxrCacheIdentity,
        partition: str,
        transform: StandardCxrTransform,
        training_seed: int,
    ) -> None:
        if tuple(frame.columns) != _IMAGE_FRAME_COLUMNS or frame.empty:
            raise ManifestBuildError("RSNA cached image frame has an invalid contract")
        if partition not in {"train", "validation", "test"}:
            raise ManifestBuildError(f"Unsupported RSNA image partition: {partition!r}")
        rows: list[tuple[str, str, int]] = []
        for row in frame.itertuples(index=False):
            sample_id = _nonempty_text(row.sample_id, "sample_id")
            patient_id = _nonempty_text(row.patient_id, "patient_id")
            target = row.target
            if (
                isinstance(target, bool)
                or not isinstance(target, int | np.integer)
                or target not in {0, 1}
            ):
                raise ManifestBuildError(f"Invalid binary target for sample {sample_id!r}")
            if (
                row.split_name != partition
                or sample_id not in cache.sample_rows
                or cache.sample_partitions[sample_id] != partition
            ):
                raise ManifestBuildError(
                    "RSNA cached image frame does not match its partition/cache"
                )
            rows.append((sample_id, patient_id, int(target)))
        sample_ids = tuple(value[0] for value in rows)
        if sample_ids != tuple(sorted(sample_ids)) or len(sample_ids) != len(set(sample_ids)):
            raise ManifestBuildError("RSNA cached image samples must be unique and ordered")
        _validate_training_seed(training_seed)
        if cache.identity != expected_cache_identity:
            raise ManifestBuildError("CXR cache identity does not match the consuming experiment")
        self._rows = tuple(rows)
        self._cache = cache
        self._transform = transform
        self._training_seed = training_seed

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, request: int | tuple[int, int]) -> ImageSample:
        epoch, index = request if isinstance(request, tuple) else (0, request)
        sample_id, patient_id, target = self._rows[index]
        seed = (
            _stable_augmentation_seed(self._training_seed, epoch, sample_id)
            if self._transform.training
            else None
        )
        image = self._transform.from_validated_cache_base(
            self._cache.image(sample_id), augmentation_seed=seed
        )
        return {
            "image": image,
            "target": torch.tensor(float(target), dtype=torch.float32),
            "sample_id": sample_id,
            "patient_id": patient_id,
        }


class RsnaCachedCalibrationDataset(Dataset[CalibrationSample]):
    """Apply the real training transform to cached train images without task data."""

    epoch_tagged_requests = True

    def __init__(
        self,
        sample_ids: tuple[str, ...],
        *,
        cache: ValidatedCxrCache,
        transform: StandardCxrTransform,
        training_seed: int,
    ) -> None:
        if (
            not sample_ids
            or sample_ids != tuple(sorted(sample_ids))
            or len(sample_ids) != len(set(sample_ids))
            or any(cache.sample_partitions.get(sample_id) != "train" for sample_id in sample_ids)
        ):
            raise ManifestBuildError("CXR loader calibration requires ordered train cache samples")
        if not isinstance(transform, StandardCxrTransform) or not transform.training:
            raise ManifestBuildError("CXR loader calibration requires the training transform")
        _validate_training_seed(training_seed)
        self._sample_ids = sample_ids
        self._cache = cache
        self._transform = transform
        self._training_seed = training_seed

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, request: int | tuple[int, int]) -> CalibrationSample:
        epoch, index = request if isinstance(request, tuple) else (0, request)
        sample_id = self._sample_ids[index]
        image = self._transform.from_validated_cache_base(
            self._cache.image(sample_id),
            augmentation_seed=_stable_augmentation_seed(self._training_seed, epoch, sample_id),
        )
        return {"image": image, "sample_id": sample_id}


class RsnaCachedFusionDataset(Dataset[FusionSample]):
    """Expose cached RSNA images aligned with transformed metadata."""

    epoch_tagged_requests = True

    def __init__(
        self,
        frame: pd.DataFrame,
        structured: np.ndarray,
        *,
        structured_sample_ids: tuple[str, ...],
        cache: ValidatedCxrCache,
        expected_cache_identity: CxrCacheIdentity,
        partition: str,
        transform: StandardCxrTransform,
        training_seed: int,
    ) -> None:
        if tuple(frame.columns) != _FUSION_FRAME_COLUMNS:
            raise ManifestBuildError("RSNA cached fusion frame has an invalid contract")
        sample_ids = tuple(frame["sample_id"].astype(str))
        if sample_ids != structured_sample_ids:
            raise ManifestBuildError("Structured rows are not aligned with fusion sample IDs")
        try:
            matrix = np.asarray(structured, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ManifestBuildError("Fusion structured matrix must be float-compatible") from exc
        if (
            matrix.ndim != 2
            or matrix.shape[0] != len(frame)
            or matrix.shape[1] == 0
            or not np.isfinite(matrix).all()
        ):
            raise ManifestBuildError("Fusion structured matrix must be finite N x D data")
        self._structured = torch.from_numpy(np.ascontiguousarray(matrix))
        self._images = RsnaCachedImageDataset(
            frame.loc[:, _IMAGE_FRAME_COLUMNS],
            cache=cache,
            expected_cache_identity=expected_cache_identity,
            partition=partition,
            transform=transform,
            training_seed=training_seed,
        )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, request: int | tuple[int, int]) -> FusionSample:
        index = request[1] if isinstance(request, tuple) else request
        return {**self._images[request], "structured": self._structured[index]}


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
        """Load train and validation rows bound to the pinned source inventory."""
        bundle, metadata = _load_pinned_bundle(config, materialize_all_rows=False)
        frame = _task_frame(
            bundle,
            config.task_id,
            partitions=("train", "validation"),
            feature_columns=("image_path",),
        )
        manifest_sha256 = _required_manifest_sha256(bundle)
        return ImageRunData(
            train=_image_partition(frame, "train"),
            validation=_image_partition(frame, "validation"),
            lineage=_lineage(config, metadata),
            bundle_manifest_sha256=manifest_sha256,
            source_inventory=_source_inventory_identity(metadata),
        )

    def load_image_cache(self, config: DatasetConfig) -> ImageCacheData:
        """Load every inventory-bound image row required by the shared cache."""
        bundle, metadata = _load_pinned_bundle(config, materialize_all_rows=False)
        frame = _image_cache_frame(bundle)
        return ImageCacheData(
            frame=frame,
            lineage=_lineage(config, metadata),
            bundle_manifest_sha256=_required_manifest_sha256(bundle),
            source_inventory=_source_inventory_identity(metadata),
        )

    def load_image_test(
        self,
        config: DatasetConfig,
        *,
        expected_manifest_sha256: str,
    ) -> ImageTestData:
        """Load test image rows bound to the pinned source inventory."""
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
        manifest_sha256 = _required_manifest_sha256(bundle)
        return ImageTestData(
            test=_image_partition(frame, "test"),
            lineage=_lineage(config, metadata),
            bundle_manifest_sha256=manifest_sha256,
            source_inventory=_source_inventory_identity(metadata),
        )

    def load_fusion_train_validation(self, config: DatasetConfig) -> FusionRunData:
        """Load aligned train and validation rows bound to the source inventory."""
        bundle, metadata = _load_pinned_bundle(config, materialize_all_rows=False)
        frame = _task_frame(
            bundle,
            config.task_id,
            partitions=("train", "validation"),
            feature_columns=("image_path", *SOURCE_FEATURES),
        )
        return FusionRunData(
            train=_fusion_partition(frame, "train"),
            validation=_fusion_partition(frame, "validation"),
            lineage=_lineage(config, metadata),
            bundle_manifest_sha256=_required_manifest_sha256(bundle),
            source_inventory=_source_inventory_identity(metadata),
        )

    def load_fusion_test(
        self,
        config: DatasetConfig,
        *,
        expected_manifest_sha256: str,
    ) -> FusionTestData:
        """Load aligned test fusion rows bound to the source inventory."""
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
        return FusionTestData(
            test=_fusion_partition(frame, "test"),
            lineage=_lineage(config, metadata),
            bundle_manifest_sha256=_required_manifest_sha256(bundle),
            source_inventory=_source_inventory_identity(metadata),
        )

    def load_localization_test(
        self,
        config: DatasetConfig,
        *,
        expected_manifest_sha256: str,
    ) -> LocalizationTestData:
        """Load cache-backed test images and validated positive-box geometry."""
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


def prepare_rsna_cxr_cache(
    dataset: RsnaDataset,
    config: DatasetConfig,
    transform: StandardCxrTransform,
    *,
    cache_root: str | Path = "data/cache/rsna",
) -> ValidatedCxrCache:
    """Build or validate the one cache pinned by an RSNA dataset configuration."""
    data = dataset.load_image_cache(config)
    identity = CxrCacheIdentity(
        bundle_id=data.lineage.bundle_id,
        bundle_manifest_sha256=data.bundle_manifest_sha256,
        source_inventory_file_sha256=data.source_inventory.source_inventory_file_sha256,
        source_inventory_arrow_sha256=data.source_inventory.source_inventory_arrow_sha256,
        preprocessing_sha256=preprocessing_identity(transform),
    )
    if config.dataset_root is None:
        raise ValueError("RSNA image cache requires dataset.dataset_root")
    return build_cxr_cache(
        data.frame,
        dataset_root=config.dataset_root,
        cache_root=cache_root,
        identity=identity,
        transform=transform,
    )


def expected_rsna_cxr_cache_identity(
    *,
    lineage: DatasetLineage,
    bundle_manifest_sha256: str,
    source_inventory: SourceInventoryIdentity,
    transform: StandardCxrTransform,
) -> CxrCacheIdentity:
    """Return the exact cache identity expected by one RSNA consumer."""
    return CxrCacheIdentity(
        bundle_id=lineage.bundle_id,
        bundle_manifest_sha256=bundle_manifest_sha256,
        source_inventory_file_sha256=source_inventory.source_inventory_file_sha256,
        source_inventory_arrow_sha256=source_inventory.source_inventory_arrow_sha256,
        preprocessing_sha256=preprocessing_identity(transform),
    )


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


def _image_cache_frame(bundle: _PinnedBundlePaths) -> pd.DataFrame:
    """Load deterministic image source rows without opening task labels."""
    assignments = pq.read_table(
        bundle.splits_path,
        columns=["sample_id", "split_name"],
    ).to_pandas()
    samples = pq.read_table(
        bundle.samples_path,
        columns=["sample_id", "patient_id", "image_path"],
    ).to_pandas()
    inventory = pq.read_table(
        bundle.source_inventory_path,
        columns=["sample_id", "relative_path", "byte_size", "sha256"],
    ).to_pandas()
    frame = (
        samples.merge(assignments, on="sample_id", validate="one_to_one")
        .merge(inventory, on="sample_id", validate="one_to_one")
        .sort_values("sample_id", kind="stable")
        .reset_index(drop=True)
    )
    if not frame["relative_path"].equals(frame["image_path"]):
        raise ManifestBuildError("RSNA source inventory paths differ from sample image paths")
    frame = frame.drop(columns="relative_path")
    if tuple(frame.columns) != CXR_CACHE_FRAME_COLUMNS or set(frame["split_name"]) != {
        "train",
        "validation",
        "test",
    }:
        raise ManifestBuildError("RSNA cache source rows have an invalid partition contract")
    return frame


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


def _source_inventory_identity(metadata: dict[str, object]) -> SourceInventoryIdentity:
    """Read the exact source-inventory artifact identity from bundle metadata."""
    hashes = metadata.get("generated_artifact_hashes")
    declared = hashes.get(SOURCE_INVENTORY_FILENAME) if isinstance(hashes, dict) else None
    if not isinstance(declared, dict):
        raise ManifestBuildError("Bundle metadata is missing source-inventory identity")
    arrow_hash = declared.get("arrow_ipc_sha256")
    file_hash = declared.get("file_sha256")
    if not _is_sha256(arrow_hash) or not _is_sha256(file_hash):
        raise ManifestBuildError("Bundle metadata source-inventory hashes are invalid")
    return SourceInventoryIdentity(
        source_inventory_arrow_sha256=arrow_hash,
        source_inventory_file_sha256=file_hash,
    )


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


def _validate_training_seed(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestBuildError("RSNA cached dataset training seed must be nonnegative")


def _stable_augmentation_seed(training_seed: int, epoch: int, sample_id: str) -> int:
    """Derive worker-independent augmentation randomness for one sample request."""
    if epoch < 0:
        raise ManifestBuildError("Image epoch must be nonnegative")
    payload = f"radfusion-rsna-augmentation\0{training_seed}\0{epoch}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
