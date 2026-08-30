"""Build, validate, publish, and access immutable Symile bundles."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from beyondcxr.data.bundle_contract import (
    BUNDLE_MANIFEST_SCHEMA_VERSION,
    BUNDLE_PREFIX,
    BUNDLES_DIRECTORY,
    CURRENT_FILENAME,
    MANIFEST_FILENAME,
    SAMPLES_FILENAME,
    valid_bundle_id,
    validate_common_bundle_envelope,
)
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.hashing import LogicalTableContract, logical_table_sha256, sha256_file
from beyondcxr.data.symile_schemas import (
    DATASET_ID,
    DATASET_RELEASE,
    DEVELOPMENT_SPLITS,
    LAB_ITEM_IDS,
    LAB_NAMES,
    LAB_SCHEMA,
    OFFICIAL_SPLITS,
    SAMPLE_SCHEMA,
    TASK_ID,
    task_contract,
)
from beyondcxr.data.symile_source import (
    EXPECTED_RELEASE_ASSETS,
    _QualifiedSymileSource,
    authenticate_development_source_asset,
)
from beyondcxr.utils.publication import (
    install_immutable_directory,
    staging_directory,
    update_current_marker,
)
from beyondcxr.utils.semantic_identity import (
    canonical_json_bytes,
    semantic_sha256,
)

OPAQUE_SOURCE_FILES = ("symile_mimic_data.csv", "test.csv")
LABS_FILENAME = "labs.parquet"
METADATA_FILENAME = MANIFEST_FILENAME
_EXPECTED_FILES = {SAMPLES_FILENAME: SAMPLE_SCHEMA, LABS_FILENAME: LAB_SCHEMA}
_ARTIFACT_ROLES = MappingProxyType({"samples": SAMPLES_FILENAME, "labs": LABS_FILENAME})
_OFFICIAL_MEMBERSHIP_SOURCE = {
    "train": "train.csv",
    "validation": "val.csv",
    "test": "positive self-query rows of test.csv",
}
_COMMON_MODALITY_LOCATOR = ["official_split", "source_row"]
_CXR_MODALITY_DECLARATIONS = {
    "asset": "data_npy/<split>/cxr_<split>.npy",
    "dtype": "float32",
    "preprocessing": "official resize/crop/ImageNet normalization",
}
_ECG_MODALITY_DECLARATIONS = {
    "asset": "data_npy/<split>/ecg_<split>.npy",
    "dtype": "float32",
    "range": [-1.0, 1.0],
}
_TEMPORAL_DECLARATIONS = {
    "cxr": {
        "source": "symile_mimic_data.csv",
        "semantics": "earliest eligible AP/PA image >24 and <=72 hours after admission",
    },
    "ecg": {
        "source": "symile_mimic_data.csv",
        "semantics": "earliest valid recording within +/-24 hours of admission",
    },
    "laboratories": {
        "source": "code/process_mimic_data.py:get_labs_df",
        "semantics": "earliest values among 50 selected tests within 24 hours of admission",
        "release_scope": "selected values without raw laboratory-event timestamps",
    },
}
_LAB_VALUE_SEMANTICS = "raw float64 source value; null means unobserved"
_LAB_OBSERVED_SEMANTICS = "true exactly when the raw value is non-null"


def _artifact_contracts() -> tuple[tuple[str, str, LogicalTableContract], ...]:
    return (
        (
            "samples",
            SAMPLES_FILENAME,
            LogicalTableContract(
                "symile.samples", 1, SAMPLE_SCHEMA, ("sample_id",), ("sample_id",)
            ),
        ),
        (
            "labs",
            LABS_FILENAME,
            LogicalTableContract("symile.labs", 1, LAB_SCHEMA, ("sample_id",), ("sample_id",)),
        ),
    )


@dataclass(frozen=True)
class _SymileBuildState:
    samples: pa.Table
    labs: pa.Table
    metadata: Mapping[str, Any]
    source_root: Path


@dataclass(frozen=True)
class SymileBundlePaths:
    bundle_id: str
    bundle_directory: Path
    samples_path: Path
    labs_path: Path
    metadata_path: Path
    current_path: Path


@dataclass(frozen=True)
class ValidatedSymileBundleReference:
    """Structurally inspected envelope; logical table values are not rehashed."""

    manifest: Mapping[str, Any]


def _build_symile_artifacts(source: _QualifiedSymileSource) -> _SymileBuildState:
    """Construct exact canonical Arrow artifacts from a qualified source release."""
    samples = pa.Table.from_pandas(source.samples, schema=SAMPLE_SCHEMA, preserve_index=False)
    labs = pa.Table.from_pandas(source.labs, schema=LAB_SCHEMA, preserve_index=False)
    _validate_tables(samples, labs)
    metadata = _base_metadata(source)
    return _SymileBuildState(samples, labs, metadata, source.source_root)


def _write_symile_bundle(
    result: _SymileBuildState,
    output_directory: str | Path = "data/manifests",
) -> SymileBundlePaths:
    """Validate and immutably publish one content-addressed Symile bundle."""
    dataset_root = Path(output_directory) / DATASET_ID
    bundles_root = dataset_root / BUNDLES_DIRECTORY
    bundles_root.mkdir(parents=True, exist_ok=True)
    current_path = dataset_root / CURRENT_FILENAME
    logical_hashes = {
        filename: logical_table_sha256(getattr(result, role), contract)
        for role, filename, contract in _artifact_contracts()
    }
    bundle_id = semantic_bundle_id(result.metadata, logical_hashes)
    destination = bundles_root / bundle_id
    stage = staging_directory(destination)
    try:
        for filename in OPAQUE_SOURCE_FILES:
            shutil.copyfile(result.source_root / filename, stage / filename)
        pq.write_table(result.samples, stage / SAMPLES_FILENAME, compression="zstd")
        pq.write_table(result.labs, stage / LABS_FILENAME, compression="zstd")
        metadata = _finalize_metadata(result.metadata, bundle_id, stage, logical_hashes)
        (stage / METADATA_FILENAME).write_bytes(canonical_json_bytes(metadata))
        install_immutable_directory(
            stage,
            destination,
            partial(validate_symile_bundle, expected_bundle_id=bundle_id),
        )
        update_current_marker(current_path, bundle_id)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return _bundle_paths(dataset_root, bundle_id)


def semantic_bundle_id(metadata: Mapping[str, Any], logical_hashes: Mapping[str, str]) -> str:
    """Return the semantic identity of meaning-bearing Symile bundle content."""
    payload = bundle_identity_payload(metadata, logical_hashes)
    digest = semantic_sha256("symile-bundle", 1, payload)
    return BUNDLE_PREFIX + digest


def bundle_identity_payload(
    metadata: Mapping[str, Any], logical_hashes: Mapping[str, str]
) -> dict[str, object]:
    """Build the canonical semantic bundle identity payload."""
    membership = metadata["membership"]
    source = metadata["source"]
    modalities = metadata["modalities"]
    source_assets = sorted(
        (
            {
                "relative_path": asset["relative_path"],
                "sha256": asset["expected_sha256"],
            }
            for asset in source["source_assets"]
        ),
        key=lambda asset: str(asset["relative_path"]),
    )
    return {
        "dataset": metadata["dataset"],
        "tasks": metadata["tasks"],
        "membership": {key: membership[key] for key in ("source", "test_selector")},
        "source": {
            "release": source["release"],
            "assets": source_assets,
        },
        "modalities": {
            key: modalities[key] for key in ("common_locator", "cxr", "ecg", "labs", "temporal")
        },
        "artifacts": {role: logical_hashes[filename] for role, filename in _ARTIFACT_ROLES.items()},
    }


def resolve_symile_bundle(
    manifest_directory: str | Path = "data/manifests",
    *,
    bundle_id: str | None = None,
    full_validation: bool = False,
) -> SymileBundlePaths:
    """Resolve an explicit immutable bundle or the interactive CURRENT selection."""
    dataset_root = Path(manifest_directory) / DATASET_ID
    current_path = dataset_root / CURRENT_FILENAME
    resolved = bundle_id
    if resolved is None:
        if current_path.is_symlink() or not current_path.is_file():
            raise ManifestBuildError("Symile CURRENT marker is missing")
        resolved = current_path.read_text(encoding="utf-8").strip()
    if not valid_bundle_id(resolved):
        raise ManifestBuildError("Symile bundle identity is invalid")
    paths = _bundle_paths(dataset_root, resolved)
    validator = validate_symile_bundle if full_validation else validate_symile_bundle_reference
    validator(paths.bundle_directory, expected_bundle_id=resolved)
    return paths


def validate_symile_bundle(
    bundle_directory: str | Path,
    *,
    expected_bundle_id: str | None = None,
    enforce_directory_name: bool = True,
) -> dict[str, Any]:
    """Validate development semantics and opaque transport without held-out decoding."""
    reference = validate_symile_bundle_reference(
        bundle_directory,
        expected_bundle_id=expected_bundle_id,
        enforce_directory_name=enforce_directory_name,
    )
    root = Path(bundle_directory)
    samples = pq.read_table(root / SAMPLES_FILENAME)
    labs = pq.read_table(root / LABS_FILENAME)
    _validate_tables(samples, labs)
    if set(samples.column("official_split").to_pylist()) != set(DEVELOPMENT_SPLITS):
        raise ManifestBuildError("Symile bundle tables must be development-only")
    actual = {
        filename: logical_table_sha256(samples if filename == SAMPLES_FILENAME else labs, contract)
        for _, filename, contract in _artifact_contracts()
    }
    declared = reference.manifest["artifacts"]
    for filename, digest in actual.items():
        if digest != declared[filename]["semantic_sha256"]:
            raise ManifestBuildError(f"Logical table hash mismatch: {filename}")
        if declared[filename]["row_count"] != (
            samples.num_rows if filename == SAMPLES_FILENAME else labs.num_rows
        ):
            raise ManifestBuildError(f"Row count mismatch: {filename}")
    if semantic_bundle_id(reference.manifest, actual) != reference.manifest["bundle"]["bundle_id"]:
        raise ManifestBuildError("Symile semantic bundle identity does not match content")
    _validate_manifest_counts(reference.manifest, samples)
    return dict(reference.manifest)


def validate_symile_bundle_reference(
    bundle_directory: str | Path,
    *,
    expected_bundle_id: str | None = None,
    enforce_directory_name: bool = True,
) -> ValidatedSymileBundleReference:
    """Inspect files, schemas, and declarations without validating logical table values."""
    root = Path(bundle_directory)
    _require_exact_regular_files(root, {METADATA_FILENAME, *_EXPECTED_FILES, *OPAQUE_SOURCE_FILES})
    metadata_bytes = (root / METADATA_FILENAME).read_bytes()
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Symile manifest is unreadable") from exc
    if metadata_bytes != canonical_json_bytes(metadata):
        raise ManifestBuildError("Symile manifest serialization is not canonical")
    _validate_manifest(metadata)
    assets = {entry["relative_path"]: entry for entry in metadata["source"]["source_assets"]}
    for filename in OPAQUE_SOURCE_FILES:
        path = root / filename
        if (
            path.stat().st_size != assets[filename]["byte_size"]
            or sha256_file(path) != assets[filename]["expected_sha256"]
        ):
            raise ManifestBuildError(f"Opaque source hash mismatch: {filename}")
    bundle_id = metadata["bundle"]["bundle_id"]
    if expected_bundle_id is not None and bundle_id != expected_bundle_id:
        raise ManifestBuildError("Symile bundle identity differs from the expected identity")
    if enforce_directory_name and root.name != bundle_id:
        raise ManifestBuildError("Symile build directory name differs from its identity")
    logical_hashes: dict[str, str] = {}
    for filename, schema in _EXPECTED_FILES.items():
        declaration = metadata["artifacts"][filename]
        path = root / filename
        try:
            if pq.read_schema(path) != schema:
                raise ManifestBuildError(f"Parquet schema mismatch: {filename}")
            if pq.read_metadata(path).num_rows != declaration["row_count"]:
                raise ManifestBuildError(f"Parquet row count mismatch: {filename}")
        except (OSError, pa.ArrowException) as exc:
            raise ManifestBuildError(f"Symile artifact is unreadable: {filename}") from exc
        logical_hashes[filename] = declaration["semantic_sha256"]
    if semantic_bundle_id(metadata, logical_hashes) != bundle_id:
        raise ManifestBuildError("Symile declared semantic identity is invalid")
    return ValidatedSymileBundleReference(metadata)


def read_symile_samples(
    bundle: SymileBundlePaths,
    *,
    official_splits: Sequence[str],
    sample_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read development sample rows through explicit partition and identity filters."""
    if not official_splits or any(value not in DEVELOPMENT_SPLITS for value in official_splits):
        raise PermissionError("Symile sample access is limited to development splits")
    filters: list[tuple[str, str, object]] = [("official_split", "in", list(official_splits))]
    if sample_ids is not None:
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ManifestBuildError("Symile sample identity filter is invalid")
        filters.append(("sample_id", "in", list(sample_ids)))
    table = pq.read_table(bundle.samples_path, filters=filters)
    frame = table.to_pandas().sort_values("sample_id", kind="stable").reset_index(drop=True)
    if sample_ids is not None and set(frame["sample_id"]) != set(sample_ids):
        raise ManifestBuildError("Symile sample rows do not cover the requested samples")
    return frame


def read_symile_labs(bundle: SymileBundlePaths, *, sample_ids: Sequence[str]) -> pd.DataFrame:
    """Read laboratory rows for an explicit development-only sample set."""
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ManifestBuildError("Symile laboratory identity filter is invalid")
    development = read_symile_samples(
        bundle,
        official_splits=DEVELOPMENT_SPLITS,
    )
    if not set(sample_ids).issubset(set(development["sample_id"].astype(str))):
        raise PermissionError("Symile laboratory access is limited to development samples")
    table = pq.read_table(bundle.labs_path, filters=[("sample_id", "in", list(sample_ids))])
    frame = table.to_pandas().sort_values("sample_id", kind="stable").reset_index(drop=True)
    if set(frame["sample_id"]) != set(sample_ids):
        raise ManifestBuildError("Symile laboratory rows do not cover the requested samples")
    return frame


def strict_pneumonia_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return eligible development rows with a transient strict binary target."""
    if not {"pneumonia_state", "sample_id", "official_split"} <= set(frame):
        raise ManifestBuildError("Symile strict-pneumonia input rows are incomplete")
    if not set(frame["official_split"]).issubset(DEVELOPMENT_SPLITS):
        raise PermissionError("Symile target derivation is limited to development rows")
    eligible = frame.loc[frame["pneumonia_state"].isin([0, 1])].copy()
    eligible["target"] = eligible["pneumonia_state"].astype("int8")
    return eligible.sort_values("sample_id", kind="stable").reset_index(drop=True)


def authenticate_source_asset(
    bundle: SymileBundlePaths,
    source_root: str | Path,
    *,
    official_split: str,
    modality: str,
) -> Path:
    """Authenticate one development modality through the bundle-bound checksum manifest."""
    if official_split not in DEVELOPMENT_SPLITS:
        raise PermissionError("Symile source-asset access is limited to development splits")
    reference = validate_symile_bundle_reference(
        bundle.bundle_directory, expected_bundle_id=bundle.bundle_id
    )
    return authenticate_development_source_asset(
        source_root,
        checksum_manifest_sha256=reference.manifest["source"]["checksum_manifest_sha256"],
        official_split=official_split,
        modality=modality,
    )


def _base_metadata(source: _QualifiedSymileSource) -> dict[str, Any]:
    reconciliation = dict(source.evidence["reconciliation"])
    modality_evidence = source.evidence["modality_qualification"]
    train_evidence = modality_evidence["splits"]["train"]
    return {
        "bundle_manifest_schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
        "dataset": {"dataset_id": DATASET_ID, "release": DATASET_RELEASE},
        "tasks": {TASK_ID: task_contract()},
        "membership": {
            "source": _OFFICIAL_MEMBERSHIP_SOURCE,
            "test_selector": "label == 1 and label_hadm_id == hadm_id",
        },
        "source": {
            "release": DATASET_RELEASE,
            "checksum_manifest": "SHA256SUMS.txt",
            "checksum_manifest_sha256": source.checksum_manifest_sha256,
            "authentication": "passed",
            "source_assets": [asset.as_dict() for asset in source.source_assets],
        },
        "modalities": {
            "common_locator": _COMMON_MODALITY_LOCATOR,
            "cxr": {
                **_CXR_MODALITY_DECLARATIONS,
                "sample_shape": train_evidence["cxr_shape"],
            },
            "ecg": {
                **_ECG_MODALITY_DECLARATIONS,
                "sample_shape": train_evidence["ecg_shape"],
            },
            "temporal": _TEMPORAL_DECLARATIONS,
            "labs": {
                "item_order": list(LAB_ITEM_IDS),
                "item_names": {item: LAB_NAMES[item] for item in LAB_ITEM_IDS},
                "value_semantics": _LAB_VALUE_SEMANTICS,
                "observed_semantics": _LAB_OBSERVED_SEMANTICS,
                "official_percentiles": (
                    "conformance reference for the upstream percentile representation"
                ),
                "labs_means": "official missing-percentile conformance reference",
            },
        },
        "privacy": {
            "classification": "restricted patient-level data",
            "public_reporting": "aggregate only",
        },
        "qualification": {
            "membership_counts": reconciliation,
            "modalities": modality_evidence,
        },
    }


def _finalize_metadata(
    base: Mapping[str, Any],
    bundle_id: str,
    stage: Path,
    logical_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        **base,
        "bundle": {
            "bundle_id": bundle_id,
            "publication_model": "immutable-bundle-directory-with-atomic-CURRENT-marker",
        },
        "artifacts": {
            filename: {
                "semantic_sha256": logical_hashes[filename],
                "row_count": pq.read_metadata(stage / filename).num_rows,
            }
            for filename in sorted(_EXPECTED_FILES)
        },
    }


def _validate_manifest(metadata: object) -> None:
    validate_common_bundle_envelope(metadata, expected_artifacts=set(_EXPECTED_FILES))
    assert isinstance(metadata, dict)
    if metadata["dataset"] != {"dataset_id": DATASET_ID, "release": DATASET_RELEASE} or metadata[
        "tasks"
    ] != {TASK_ID: task_contract()}:
        raise ManifestBuildError("Symile manifest dataset/task contract is invalid")
    if metadata["privacy"] != {
        "classification": "restricted patient-level data",
        "public_reporting": "aggregate only",
    }:
        raise ManifestBuildError("Symile manifest privacy contract is invalid")
    qualification = metadata["qualification"]
    if not isinstance(qualification, Mapping) or set(qualification) != {
        "membership_counts",
        "modalities",
    }:
        raise ManifestBuildError("Symile qualification metadata is invalid")
    laboratories = metadata["modalities"].get("labs")
    if (
        not isinstance(laboratories, dict)
        or laboratories.get("item_order") != list(LAB_ITEM_IDS)
        or laboratories.get("item_names") != {item: LAB_NAMES[item] for item in LAB_ITEM_IDS}
        or laboratories.get("value_semantics") != _LAB_VALUE_SEMANTICS
        or laboratories.get("observed_semantics") != _LAB_OBSERVED_SEMANTICS
    ):
        raise ManifestBuildError("Symile manifest laboratory contract is invalid")
    membership = metadata["membership"]
    if (
        not isinstance(membership, dict)
        or membership.get("source") != _OFFICIAL_MEMBERSHIP_SOURCE
        or membership.get("test_selector") != "label == 1 and label_hadm_id == hadm_id"
        or set(membership) != {"source", "test_selector"}
    ):
        raise ManifestBuildError("Symile official membership contract is invalid")
    modalities = metadata["modalities"]
    cxr = modalities.get("cxr") if isinstance(modalities, dict) else None
    ecg = modalities.get("ecg") if isinstance(modalities, dict) else None
    if (
        not isinstance(cxr, dict)
        or set(cxr) != {"sample_shape", *_CXR_MODALITY_DECLARATIONS}
        or any(cxr.get(key) != value for key, value in _CXR_MODALITY_DECLARATIONS.items())
        or not isinstance(ecg, dict)
        or set(ecg) != {"sample_shape", *_ECG_MODALITY_DECLARATIONS}
        or any(ecg.get(key) != value for key, value in _ECG_MODALITY_DECLARATIONS.items())
        or modalities.get("common_locator") != _COMMON_MODALITY_LOCATOR
        or modalities.get("temporal") != _TEMPORAL_DECLARATIONS
    ):
        raise ManifestBuildError("Symile manifest modality contract is invalid")
    source_release = metadata["source"]
    assets = source_release.get("source_assets")
    if (
        source_release.get("release") != DATASET_RELEASE
        or source_release.get("checksum_manifest") != "SHA256SUMS.txt"
        or source_release.get("authentication") != "passed"
        or not _valid_sha256(source_release.get("checksum_manifest_sha256"))
        or not isinstance(assets, list)
        or len(assets) != len(EXPECTED_RELEASE_ASSETS)
        or [entry.get("relative_path") for entry in assets]
        != sorted(entry.get("relative_path") for entry in assets)
    ):
        raise ManifestBuildError("Symile source asset declarations are invalid")
    if {entry.get("relative_path") for entry in assets} != EXPECTED_RELEASE_ASSETS:
        raise ManifestBuildError("Symile source asset set is invalid")
    for entry in assets:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"relative_path", "byte_size", "expected_sha256", "observed_sha256"}
            or isinstance(entry["byte_size"], bool)
            or not isinstance(entry["byte_size"], int)
            or entry["byte_size"] <= 0
            or not _valid_sha256(entry["expected_sha256"])
            or entry["observed_sha256"] != entry["expected_sha256"]
        ):
            raise ManifestBuildError("Symile source asset identity is invalid")
    bundle_id = metadata["bundle"].get("bundle_id")
    if not valid_bundle_id(bundle_id):
        raise ManifestBuildError("Symile bundle identity declaration is invalid")


def _validate_tables(samples: pa.Table, labs: pa.Table) -> None:
    if (
        samples.schema != SAMPLE_SCHEMA
        or labs.schema != LAB_SCHEMA
        or samples.num_rows == 0
        or samples.num_rows != labs.num_rows
    ):
        raise ManifestBuildError("Symile bundle table contracts are invalid")
    sample_frame = samples.to_pandas()
    lab_frame = labs.to_pandas()
    sample_ids = tuple(sample_frame["sample_id"].astype(str))
    if (
        sample_ids != tuple(sorted(sample_ids))
        or len(sample_ids) != len(set(sample_ids))
        or tuple(lab_frame["sample_id"].astype(str)) != sample_ids
    ):
        raise ManifestBuildError("Symile sample/laboratory ordering or coverage is invalid")
    hadm = sample_frame["hadm_id"].astype("int64")
    if hadm.duplicated().any() or not all(
        sample_id == f"symile:{value}" for sample_id, value in zip(sample_ids, hadm, strict=True)
    ):
        raise ManifestBuildError("Symile admission identity contract is invalid")
    if (
        not set(sample_frame["official_split"]) <= set(OFFICIAL_SPLITS)
        or sample_frame[["official_split", "source_row"]].duplicated().any()
        or (sample_frame["source_row"] < 0).any()
    ):
        raise ManifestBuildError("Symile official split/source-row contract is invalid")
    states = set(sample_frame["pneumonia_state"].dropna().astype(int))
    if (
        not states <= {-1, 0, 1}
        or not set(sample_frame["view_position"]) <= {"AP", "PA"}
        or not set(sample_frame["sex"]) <= {"F", "M"}
    ):
        raise ManifestBuildError("Symile source-state or demographic domain is invalid")
    if sample_frame["age_years"].isna().any() or (sample_frame["age_years"] < 0).any():
        raise ManifestBuildError("Symile age domain is invalid")
    for item_id in LAB_ITEM_IDS:
        values = lab_frame[f"lab_{item_id}_value"]
        observed = lab_frame[f"lab_{item_id}_observed"]
        if (
            not observed.eq(values.notna()).all()
            or not pd.Series(values.dropna()).map(_finite).all()
        ):
            raise ManifestBuildError("Symile raw laboratory observedness contract is invalid")


def _validate_manifest_counts(metadata: Mapping[str, Any], samples: pa.Table) -> None:
    frame = samples.to_pandas()
    qualification = metadata["qualification"]
    if not isinstance(qualification, Mapping) or set(qualification) != {
        "membership_counts",
        "modalities",
    }:
        raise ManifestBuildError("Symile qualification metadata is invalid")
    counts = qualification["membership_counts"]
    observed = {
        f"{split}_admissions": int((frame["official_split"] == split).sum())
        for split in DEVELOPMENT_SPLITS
    }
    if counts != observed:
        raise ManifestBuildError("Symile manifest development membership counts do not match")
    if set(qualification["modalities"].get("splits", {})) != set(DEVELOPMENT_SPLITS):
        raise ManifestBuildError("Symile qualification scope is invalid")


def _require_exact_regular_files(directory: Path, expected: set[str]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ManifestBuildError("Symile bundle path is not a physical directory")
    entries = list(os.scandir(directory))
    if {entry.name for entry in entries} != expected or any(
        entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in entries
    ):
        raise ManifestBuildError("Symile bundle file set is incomplete or unexpected")


def _bundle_paths(dataset_root: Path, bundle_id: str) -> SymileBundlePaths:
    directory = dataset_root / BUNDLES_DIRECTORY / bundle_id
    return SymileBundlePaths(
        bundle_id,
        directory,
        directory / SAMPLES_FILENAME,
        directory / LABS_FILENAME,
        directory / METADATA_FILENAME,
        dataset_root / CURRENT_FILENAME,
    )


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
