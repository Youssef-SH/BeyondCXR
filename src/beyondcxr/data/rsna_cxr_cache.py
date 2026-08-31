"""Build and validate the disposable deterministic RSNA CXR cache."""

from __future__ import annotations

import hashlib
import json
import shutil
from bisect import bisect_left
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from beyondcxr.data.cxr_transforms import STANDARD_CXR_IMAGE_SIZE, StandardCxrTransform
from beyondcxr.data.dicom_loader import read_dicom
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.hashing import sha256_file
from beyondcxr.data.rsna_source import resolve_image_path
from beyondcxr.utils.operational_logging import CountProgress, get_operational_logger, log_event
from beyondcxr.utils.publication import publish_directory, staging_directory
from beyondcxr.utils.semantic_identity import canonical_scientific_id, semantic_sha256

IMAGES_FILENAME = "images.npy"
INDEX_FILENAME = "index.parquet"
METADATA_FILENAME = "metadata.json"
SOURCE_AUTHENTICATION_POLICY_VERSION = "cache-source-inventory-sha256-v1"
_SPLIT_NAMES = ("train", "validation", "test")
CXR_CACHE_FRAME_COLUMNS = (
    "sample_id",
    "patient_id",
    "image_path",
    "split_name",
    "byte_size",
    "sha256",
)
_INDEX_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("row_index", pa.int64(), nullable=False),
    ]
)
_LOGGER = get_operational_logger(__name__)


@dataclass(frozen=True)
class _ImmutableStringMapping[Value](Mapping[str, Value]):
    """Small pickle-safe immutable mapping with deterministic iteration."""

    _keys: tuple[str, ...]
    _values: tuple[Value, ...]

    def __post_init__(self) -> None:
        if (
            len(self._keys) != len(self._values)
            or self._keys != tuple(sorted(self._keys))
            or len(self._keys) != len(set(self._keys))
            or any(not isinstance(key, str) or not key for key in self._keys)
        ):
            raise ValueError("Immutable CXR cache mapping is invalid")

    def __getitem__(self, key: str) -> Value:
        index = bisect_left(self._keys, key)
        if index < len(self._keys) and self._keys[index] == key:
            return self._values[index]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


@dataclass(frozen=True)
class CxrCacheSourceAuthentication:
    """Immutable witness for complete raw-source authentication."""

    policy_version: str
    partitions: tuple[str, ...]
    file_count: int
    source_inventory_semantic_sha256: str

    def __post_init__(self) -> None:
        if (
            self.policy_version != SOURCE_AUTHENTICATION_POLICY_VERSION
            or self.partitions != _SPLIT_NAMES
            or isinstance(self.file_count, bool)
            or not isinstance(self.file_count, int)
            or self.file_count <= 0
            or not _is_sha256(self.source_inventory_semantic_sha256)
        ):
            raise ValueError("CXR cache source-authentication witness is invalid")

    def as_dict(self) -> dict[str, object]:
        """Return the serialized source-authentication witness."""
        return {
            "policy_version": self.policy_version,
            "partitions": list(self.partitions),
            "file_count": self.file_count,
            "source_inventory_semantic_sha256": self.source_inventory_semantic_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> CxrCacheSourceAuthentication:
        """Validate and reconstruct serialized source authentication."""
        fields = {
            "policy_version",
            "partitions",
            "file_count",
            "source_inventory_semantic_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("CXR cache source-authentication witness is invalid")
        try:
            return cls(
                policy_version=cast(str, value["policy_version"]),
                partitions=tuple(cast(tuple[str, ...] | list[str], value["partitions"])),
                file_count=cast(int, value["file_count"]),
                source_inventory_semantic_sha256=cast(
                    str, value["source_inventory_semantic_sha256"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("CXR cache source-authentication witness is invalid") from exc


@dataclass(frozen=True)
class CxrCacheIdentity:
    """Source and deterministic-preprocessing identity for one cache."""

    bundle_id: str
    source_inventory_semantic_sha256: str
    preprocessing_sha256: str
    source_authentication_policy: str = SOURCE_AUTHENTICATION_POLICY_VERSION

    def __post_init__(self) -> None:
        if (
            not self.bundle_id.startswith("bundle-")
            or not _is_sha256(self.bundle_id[7:])
            or not all(
                _is_sha256(value)
                for value in (
                    self.source_inventory_semantic_sha256,
                    self.preprocessing_sha256,
                )
            )
            or self.source_authentication_policy != SOURCE_AUTHENTICATION_POLICY_VERSION
        ):
            raise ValueError("CXR cache identity fields are invalid")

    @property
    def cache_id(self) -> str:
        """Return the identity-addressed cache directory name."""
        return canonical_scientific_id("cache-", self.as_dict(), domain="cache", schema_version=1)

    def as_dict(self) -> dict[str, str]:
        """Return serializable cache identity fields."""
        return {
            "bundle_id": self.bundle_id,
            "source_inventory_semantic_sha256": self.source_inventory_semantic_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
            "source_authentication_policy": self.source_authentication_policy,
        }


class PreparedCxrCache:
    """Opaque operation that can issue only validated exact-scope cache views."""

    __slots__ = ("identity", "source_authentication", "_scope_request")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("Prepared CXR cache handles are issued only by cache validation")

    def __reduce__(self) -> object:
        raise TypeError("Prepared CXR cache handles are process-local")


class ValidatedCxrCache:
    """Opaque exact-scope image reader issued by a validated prepared cache."""

    __slots__ = ("identity", "source_authentication", "_require", "_read", "_serialization")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("Scoped CXR cache views are issued only by validated cache scoping")

    def require_sample(self, sample_id: str, partition: str) -> None:
        """Require one sample to belong to this exact consumer scope."""
        self._require(sample_id, partition)

    def image(self, sample_id: str) -> torch.Tensor:
        """Return one copied tensor only when it belongs to this authorized scope."""
        return self._read(sample_id)

    def __reduce__(self) -> object:
        if self._serialization is None:
            raise TypeError("Held-out CXR cache authority cannot be serialized")
        return self._serialization


def _prepared_cache(
    directory: Path,
    identity: CxrCacheIdentity,
    source_authentication: CxrCacheSourceAuthentication,
    mapping_sha256: str,
    sample_partitions: _ImmutableStringMapping[str],
) -> PreparedCxrCache:
    images_physical_identity = _physical_identity(directory / IMAGES_FILENAME)
    development_partition_sha256 = _ImmutableStringMapping(
        ("train", "validation"),
        tuple(
            _partition_authority_sha256(
                tuple(
                    (sample_id, split)
                    for sample_id in sample_partitions
                    if sample_partitions[sample_id] == split
                )
            )
            for split in ("train", "validation")
        ),
    )
    handle = object.__new__(PreparedCxrCache)
    object.__setattr__(handle, "identity", identity)
    object.__setattr__(handle, "source_authentication", source_authentication)

    def scope_request(
        sample_ids: tuple[str, ...],
        partition: str,
        held_out_authorization: object | None,
        package_id: str | None,
    ) -> ValidatedCxrCache:
        return _scope_cxr_cache_state(
            directory=directory,
            identity=identity,
            source_authentication=source_authentication,
            mapping_sha256=mapping_sha256,
            images_physical_identity=images_physical_identity,
            development_partition_sha256=development_partition_sha256,
            sample_ids=sample_ids,
            partition=partition,
            held_out_authorization=held_out_authorization,
            package_id=package_id,
        )

    object.__setattr__(handle, "_scope_request", scope_request)
    return handle


def scope_cxr_cache(
    prepared: PreparedCxrCache,
    *,
    sample_ids: tuple[str, ...],
    partition: str,
    held_out_authorization: object | None = None,
    package_id: str | None = None,
) -> ValidatedCxrCache:
    """Create one exact consumer view; test scope requires live package authority."""
    if not isinstance(prepared, PreparedCxrCache) or not hasattr(prepared, "_scope_request"):
        raise PermissionError("CXR cache scoping requires a genuine prepared-cache handle")
    return prepared._scope_request(
        sample_ids,
        partition,
        held_out_authorization,
        package_id,
    )


def _scope_cxr_cache_state(
    *,
    directory: Path,
    identity: CxrCacheIdentity,
    source_authentication: CxrCacheSourceAuthentication,
    mapping_sha256: str,
    images_physical_identity: tuple[int, int, int, int, int],
    development_partition_sha256: _ImmutableStringMapping[str],
    sample_ids: tuple[str, ...],
    partition: str,
    held_out_authorization: object | None,
    package_id: str | None,
) -> ValidatedCxrCache:
    if partition not in _SPLIT_NAMES or not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ManifestBuildError("CXR cache consumer scope is invalid")
    if partition == "test":
        from beyondcxr.training.rsna_campaign_control import require_validated_rsna_held_out
        from beyondcxr.training.rsna_validation_evidence import load_canonical_held_out_cohort

        if not isinstance(package_id, str):
            raise PermissionError("Test cache scope requires live package-bound authorization")
        capability = require_validated_rsna_held_out(held_out_authorization)
        cohort = load_canonical_held_out_cohort(
            capability,
            model_package_id=package_id,
        )
        dataset = capability.current_execution.execution.manifest.get("dataset")
        if (
            not isinstance(dataset, Mapping)
            or identity.bundle_id != dataset.get("bundle_id")
            or sample_ids != cohort.sample_ids
        ):
            raise ManifestBuildError(
                "CXR cache or consumer scope differs from the authorized held-out cohort"
            )
    elif held_out_authorization is not None or package_id is not None:
        raise ManifestBuildError("Development cache scopes do not accept held-out authority")
    try:
        metadata = json.loads((directory / METADATA_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestBuildError("CXR cache metadata is unreadable") from exc
    declared_development = (
        metadata.get("development_partition_authority_sha256")
        if isinstance(metadata, dict)
        else None
    )
    if (
        not isinstance(metadata, dict)
        or not isinstance(declared_development, dict)
        or metadata.get("identity") != identity.as_dict()
        or metadata.get("source_authentication") != source_authentication.as_dict()
        or metadata.get("mapping_sha256") != mapping_sha256
        or declared_development != dict(development_partition_sha256)
        or sha256_file(directory / INDEX_FILENAME) != mapping_sha256
        or _physical_identity(directory / IMAGES_FILENAME) != images_physical_identity
    ):
        raise ManifestBuildError("CXR cache prepared authority changed after validation")
    index = pq.read_table(directory / INDEX_FILENAME)
    if index.schema != _INDEX_SCHEMA:
        raise ManifestBuildError("CXR cache physical index schema changed")
    physical = index.to_pandas().set_index("sample_id", verify_integrity=True)["row_index"]
    rows: list[int] = []
    for sample_id in sample_ids:
        try:
            row = int(physical.loc[sample_id])
        except KeyError as exc:
            raise ManifestBuildError(
                "CXR cache does not cover the requested consumer scope"
            ) from exc
        rows.append(row)
    if partition != "test" and development_partition_sha256[partition] != (
        _partition_authority_sha256(tuple((sample_id, partition) for sample_id in sample_ids))
    ):
        raise ManifestBuildError("CXR cache consumer scope crosses a partition boundary")
    try:
        images = np.load(directory / IMAGES_FILENAME, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ManifestBuildError("CXR cache image array is unreadable") from exc
    mapping = _ImmutableStringMapping(sample_ids, tuple(rows))
    scoped = object.__new__(ValidatedCxrCache)
    object.__setattr__(scoped, "identity", identity)
    object.__setattr__(scoped, "source_authentication", source_authentication)

    def require_sample(sample_id: str, requested_partition: str) -> None:
        if requested_partition != partition or sample_id not in mapping:
            raise ManifestBuildError("CXR cache sample is outside the authorized partition scope")

    def read_image(sample_id: str) -> torch.Tensor:
        try:
            row = mapping[sample_id]
        except KeyError as exc:
            raise ManifestBuildError("CXR cache sample is outside the authorized scope") from exc
        return torch.from_numpy(np.array(images[row], dtype=np.float32, copy=True))

    object.__setattr__(scoped, "_require", require_sample)
    object.__setattr__(scoped, "_read", read_image)
    object.__setattr__(
        scoped,
        "_serialization",
        None
        if partition == "test"
        else (
            _restore_development_cxr_cache,
            (
                directory,
                identity,
                source_authentication,
                mapping_sha256,
                images_physical_identity,
                development_partition_sha256,
                sample_ids,
                partition,
            ),
        ),
    )
    return scoped


def _restore_development_cxr_cache(
    directory: Path,
    identity: CxrCacheIdentity,
    source_authentication: CxrCacheSourceAuthentication,
    mapping_sha256: str,
    images_physical_identity: tuple[int, int, int, int, int],
    development_partition_sha256: _ImmutableStringMapping[str],
    sample_ids: tuple[str, ...],
    partition: str,
) -> ValidatedCxrCache:
    """Reauthenticate one development scope against its parent's validated evidence."""
    if partition not in {"train", "validation"}:
        raise PermissionError("Serialized CXR cache scope is not a development partition")
    return _scope_cxr_cache_state(
        directory=directory,
        identity=identity,
        source_authentication=source_authentication,
        mapping_sha256=mapping_sha256,
        images_physical_identity=images_physical_identity,
        development_partition_sha256=development_partition_sha256,
        sample_ids=sample_ids,
        partition=partition,
        held_out_authorization=None,
        package_id=None,
    )


def preprocessing_identity(transform: StandardCxrTransform) -> str:
    """Hash deterministic CXR preprocessing meaning."""
    return semantic_sha256(
        "cxr-preprocessing",
        1,
        deterministic_preprocessing_contract(transform),
    )


def deterministic_preprocessing_contract(
    transform: StandardCxrTransform,
) -> dict[str, object]:
    """Return the exact DICOM-to-cache preprocessing contract."""
    contract = transform.contract()
    return {
        "transform_policy": contract["policy_version"],
        "dicom_decode": "pydicom.pixel_array",
        "decoded_dtype": "float32",
        "monochrome1": "pixel_max_minus_pixels_before_minmax",
        "canonical_intensity": "per_image_minmax_to_0_1_constant_to_zeros",
        "input": contract["input"],
        "output_shape": contract["output"]["shape"],
        "center_crop": contract["center_crop"],
        "resize": contract["resize"],
        "cache_boundary": "after_resize_before_training_augmentation",
        "cache_dtype": "float32",
        "cache_range": [0.0, 1.0],
    }


def build_cxr_cache(
    frame: pd.DataFrame,
    *,
    dataset_root: str | Path,
    cache_root: str | Path,
    identity: CxrCacheIdentity,
    transform: StandardCxrTransform,
) -> PreparedCxrCache:
    """Build and atomically publish one complete deterministic cache."""
    _validate_source_frame(frame)
    destination = Path(cache_root) / identity.cache_id
    if destination.exists():
        return validate_cxr_cache(
            destination,
            identity=identity,
            expected_sample_partitions=_sample_partitions(frame),
        )
    stage = staging_directory(destination)
    validated_stage: PreparedCxrCache | None = None
    images: np.memmap | None = None
    try:
        shape = (len(frame), 1, STANDARD_CXR_IMAGE_SIZE, STANDARD_CXR_IMAGE_SIZE)
        images = np.lib.format.open_memmap(
            stage / IMAGES_FILENAME,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        content = hashlib.sha256()
        index_rows: list[dict[str, object]] = []
        root = Path(dataset_root)
        log_event(
            _LOGGER,
            "source_authentication_started",
            partition_count=3,
            total=len(frame),
            unit="files",
        )
        progress = CountProgress(
            _LOGGER,
            "source_authentication_progress",
            total=len(frame),
            unit="files",
        )
        for row_index, row in enumerate(frame.itertuples(index=False)):
            path = resolve_image_path(root, _relative_image_path(row.image_path))
            try:
                pixels, record = read_dicom(
                    path,
                    expected_byte_size=int(row.byte_size),
                    expected_sha256=str(row.sha256),
                )
            except (OSError, ValueError) as exc:
                raise ManifestBuildError(
                    f"CXR cache source authentication failed for {row.sample_id!r}"
                ) from exc
            if record.patient_id != str(row.patient_id):
                raise ManifestBuildError(f"Decoded DICOM patient does not match {row.sample_id!r}")
            base = transform.deterministic_base(pixels).numpy()
            images[row_index] = base
            content.update(memoryview(np.ascontiguousarray(base)).cast("B"))
            index_rows.append(
                {
                    "sample_id": str(row.sample_id),
                    "row_index": row_index,
                }
            )
            progress.update(row_index + 1)
        log_event(
            _LOGGER,
            "source_authentication_completed",
            partition_count=3,
            total=len(frame),
            unit="files",
        )
        images.flush()
        images._mmap.close()
        images = None
        pq.write_table(
            pa.Table.from_pylist(index_rows, schema=_INDEX_SCHEMA), stage / INDEX_FILENAME
        )
        metadata = {
            "cache_id": identity.cache_id,
            "identity": identity.as_dict(),
            "sample_count": len(frame),
            "dtype": "float32",
            "shape": list(shape),
            "mapping_sha256": sha256_file(stage / INDEX_FILENAME),
            "partition_authority_sha256": _partition_authority_sha256(_sample_partitions(frame)),
            "development_partition_authority_sha256": {
                split: _partition_authority_sha256(
                    tuple(value for value in _sample_partitions(frame) if value[1] == split)
                )
                for split in ("train", "validation")
            },
            "content_sha256": content.hexdigest(),
            "deterministic_preprocessing": deterministic_preprocessing_contract(transform),
            "source_authentication": {
                "policy_version": identity.source_authentication_policy,
                "partitions": list(_SPLIT_NAMES),
                "file_count": len(frame),
                "source_inventory_semantic_sha256": identity.source_inventory_semantic_sha256,
            },
        }
        (stage / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validated_stage = validate_cxr_cache(
            stage,
            identity=identity,
            expected_sample_partitions=_sample_partitions(frame),
            enforce_directory_name=False,
        )
        publish_directory(stage, destination)
    finally:
        if images is not None:
            images._mmap.close()
        if stage.exists():
            shutil.rmtree(stage)
    if validated_stage is None:
        raise ManifestBuildError("CXR cache staging validation did not complete")
    return validate_cxr_cache(
        destination,
        identity=identity,
        expected_sample_partitions=_sample_partitions(frame),
    )


def validate_cxr_source_availability(
    frame: pd.DataFrame,
    *,
    dataset_root: str | Path,
) -> None:
    """Prove that every inventoried DICOM is available before campaign output begins.

    Full byte authentication remains owned by cache construction. This preflight checks the
    complete path and declared-size surface without decoding or duplicating the cache's SHA-256
    pass.
    """
    _validate_source_frame(frame)
    root = Path(dataset_root)
    if root.is_symlink() or not root.is_dir():
        raise ManifestBuildError("RSNA source root is unavailable")
    for row in frame.itertuples(index=False):
        path = resolve_image_path(root, _relative_image_path(row.image_path))
        if path.is_symlink() or not path.is_file():
            raise ManifestBuildError(f"RSNA source DICOM is unavailable for {row.sample_id!r}")
        try:
            observed_size = path.stat().st_size
        except OSError as exc:
            raise ManifestBuildError(
                f"RSNA source DICOM is unavailable for {row.sample_id!r}"
            ) from exc
        if observed_size != int(row.byte_size):
            raise ManifestBuildError(f"RSNA source DICOM size differs for {row.sample_id!r}")


def validate_cxr_cache(
    directory: str | Path,
    *,
    identity: CxrCacheIdentity,
    expected_sample_partitions: tuple[tuple[str, str], ...],
    enforce_directory_name: bool = True,
) -> PreparedCxrCache:
    """Validate the complete cache and return an opaque prepared-cache handle."""
    root = Path(directory)
    if not root.is_dir() or (enforce_directory_name and root.name != identity.cache_id):
        raise ManifestBuildError("CXR cache directory does not match the expected identity")
    expected_files = {IMAGES_FILENAME, INDEX_FILENAME, METADATA_FILENAME}
    if {item.name for item in root.iterdir()} != expected_files:
        raise ManifestBuildError("CXR cache file set is incomplete or unexpected")
    try:
        metadata = json.loads((root / METADATA_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestBuildError("CXR cache metadata is unreadable") from exc
    if not isinstance(metadata, dict) or set(metadata) != {
        "cache_id",
        "identity",
        "sample_count",
        "dtype",
        "shape",
        "mapping_sha256",
        "partition_authority_sha256",
        "development_partition_authority_sha256",
        "content_sha256",
        "deterministic_preprocessing",
        "source_authentication",
    }:
        raise ManifestBuildError("CXR cache metadata field set does not match")
    if (
        not isinstance(expected_sample_partitions, tuple)
        or not expected_sample_partitions
        or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or not isinstance(item[1], str)
            or item[1] not in _SPLIT_NAMES
            for item in expected_sample_partitions
        )
    ):
        raise ManifestBuildError("Expected CXR cache sample partitions are invalid")
    sample_ids = tuple(item[0] for item in expected_sample_partitions)
    expected_splits = tuple(item[1] for item in expected_sample_partitions)
    if (
        sample_ids != tuple(sorted(sample_ids))
        or len(sample_ids) != len(set(sample_ids))
        or set(expected_splits) != set(_SPLIT_NAMES)
    ):
        raise ManifestBuildError("Expected CXR cache sample partitions are incomplete or unordered")
    if metadata.get("partition_authority_sha256") != _partition_authority_sha256(
        expected_sample_partitions
    ):
        raise ManifestBuildError("CXR cache scientific partition authority does not match")
    expected_development_hashes = {
        split: _partition_authority_sha256(
            tuple(value for value in expected_sample_partitions if value[1] == split)
        )
        for split in ("train", "validation")
    }
    if metadata.get("development_partition_authority_sha256") != expected_development_hashes:
        raise ManifestBuildError("CXR cache development authority does not match")
    if (
        metadata.get("cache_id") != identity.cache_id
        or metadata.get("identity") != identity.as_dict()
    ):
        raise ManifestBuildError("CXR cache source or preprocessing identity does not match")
    preprocessing = metadata.get("deterministic_preprocessing")
    if (
        not isinstance(preprocessing, dict)
        or semantic_sha256("cxr-preprocessing", 1, preprocessing) != identity.preprocessing_sha256
    ):
        raise ManifestBuildError("CXR cache deterministic preprocessing contract does not match")
    if metadata.get("sample_count") != len(sample_ids):
        raise ManifestBuildError("CXR cache sample count does not match")
    if metadata.get("dtype") != "float32" or metadata.get("shape") != [
        len(sample_ids),
        1,
        STANDARD_CXR_IMAGE_SIZE,
        STANDARD_CXR_IMAGE_SIZE,
    ]:
        raise ManifestBuildError("CXR cache tensor contract does not match")
    try:
        witness = CxrCacheSourceAuthentication.from_dict(metadata.get("source_authentication"))
    except ValueError as exc:
        raise ManifestBuildError("CXR cache source-authentication witness does not match") from exc
    if (
        witness.file_count != len(sample_ids)
        or witness.source_inventory_semantic_sha256 != identity.source_inventory_semantic_sha256
    ):
        raise ManifestBuildError("CXR cache source-authentication witness does not match")
    if sha256_file(root / INDEX_FILENAME) != metadata.get("mapping_sha256"):
        raise ManifestBuildError("CXR cache mapping integrity does not match")
    index = pq.read_table(root / INDEX_FILENAME)
    if index.schema != _INDEX_SCHEMA:
        raise ManifestBuildError("CXR cache index schema does not match")
    index_frame = index.to_pandas()
    if tuple(index_frame["sample_id"].astype(str)) != sample_ids or tuple(
        index_frame["row_index"].astype(int)
    ) != tuple(range(len(sample_ids))):
        raise ManifestBuildError("CXR cache sample mapping does not match")
    try:
        images = np.load(root / IMAGES_FILENAME, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ManifestBuildError("CXR cache image array is unreadable") from exc
    try:
        if images.dtype != np.float32 or images.shape != (
            len(sample_ids),
            1,
            STANDARD_CXR_IMAGE_SIZE,
            STANDARD_CXR_IMAGE_SIZE,
        ):
            raise ManifestBuildError("CXR cache image array contract does not match")
        digest = hashlib.sha256()
        for start in range(0, len(images), 128):
            chunk = np.ascontiguousarray(images[start : start + 128])
            if not np.isfinite(chunk).all() or float(chunk.min()) < 0.0 or float(chunk.max()) > 1.0:
                raise ManifestBuildError("CXR cache image content is invalid")
            digest.update(memoryview(chunk).cast("B"))
        if digest.hexdigest() != metadata.get("content_sha256"):
            raise ManifestBuildError("CXR cache image content integrity does not match")
        partitions = _ImmutableStringMapping(sample_ids, expected_splits)
        result = _prepared_cache(
            root,
            identity,
            witness,
            cast(str, metadata["mapping_sha256"]),
            partitions,
        )
        return result
    finally:
        images._mmap.close()


def _validate_source_frame(frame: pd.DataFrame) -> None:
    if (
        not isinstance(frame, pd.DataFrame)
        or tuple(frame.columns) != CXR_CACHE_FRAME_COLUMNS
        or frame.empty
    ):
        raise ManifestBuildError("CXR cache source frame is invalid")
    sample_ids = tuple(frame["sample_id"].astype(str))
    if sample_ids != tuple(sorted(sample_ids)) or len(sample_ids) != len(set(sample_ids)):
        raise ManifestBuildError("CXR cache source rows must have unique ordered sample IDs")
    if any(not isinstance(value, str) or not value for value in frame["sample_id"]):
        raise ManifestBuildError("CXR cache sample IDs must be non-empty strings")
    if any(not isinstance(value, str) or not value for value in frame["patient_id"]):
        raise ManifestBuildError("CXR cache patient IDs must be non-empty strings")
    split_names = tuple(frame["split_name"])
    if any(not isinstance(value, str) or value not in _SPLIT_NAMES for value in split_names) or set(
        split_names
    ) != set(_SPLIT_NAMES):
        raise ManifestBuildError("CXR cache source rows must cover every current split")
    for value in frame["image_path"]:
        _relative_image_path(value)
    for row in frame.itertuples(index=False):
        if (
            isinstance(row.byte_size, bool)
            or not isinstance(row.byte_size, int | np.integer)
            or row.byte_size <= 0
            or not _is_sha256(row.sha256)
        ):
            raise ManifestBuildError("CXR cache source inventory row is invalid")


def _sample_partitions(frame: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    return tuple((str(row.sample_id), str(row.split_name)) for row in frame.itertuples(index=False))


def _partition_authority_sha256(value: tuple[tuple[str, str], ...]) -> str:
    return semantic_sha256(
        "rsna-cache-partition-authority",
        1,
        [{"sample_id": sample_id, "split_name": split_name} for sample_id, split_name in value],
    )


def _physical_identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        value = path.stat()
    except OSError as exc:
        raise ManifestBuildError("CXR cache image storage is unavailable") from exc
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _relative_image_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ManifestBuildError("CXR cache image paths must be non-empty strings")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or path.as_posix() != value
    ):
        raise ManifestBuildError(f"Invalid CXR cache image path: {value!r}")
    return path


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
