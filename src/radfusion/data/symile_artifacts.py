"""Build, validate, publish, and access immutable Symile bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from radfusion.data.errors import ManifestBuildError
from radfusion.data.hashing import arrow_ipc_sha256, sha256_file
from radfusion.data.symile_schemas import (
    DATASET_ID,
    DATASET_RELEASE,
    LAB_ITEM_IDS,
    LAB_NAMES,
    LAB_SCHEMA,
    MANIFEST_SCHEMA_VERSION,
    OFFICIAL_SPLITS,
    SAMPLE_SCHEMA,
    task_contract,
)
from radfusion.data.symile_source import (
    EXPECTED_RELEASE_ASSETS,
    QualifiedSymileSource,
    authenticate_release_asset,
    modality_asset_path,
)
from radfusion.utils.publication import staging_directory, update_current_marker

SAMPLES_FILENAME = "symile_samples.parquet"
LABS_FILENAME = "symile_labs.parquet"
METADATA_FILENAME = "symile_manifest_metadata.json"
CURRENT_FILENAME = "CURRENT"
BUILDS_DIRECTORY = "builds"
_EXPECTED_FILES = {SAMPLES_FILENAME: SAMPLE_SCHEMA, LABS_FILENAME: LAB_SCHEMA}
_MANIFEST_FIELDS = {
    "manifest_schema_version",
    "dataset",
    "task",
    "official_membership",
    "source_release",
    "modalities",
    "laboratories",
    "privacy",
    "bundle",
    "artifacts",
    "provenance",
    "generation",
}
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


@dataclass(frozen=True)
class SymileBuildResult:
    samples: pa.Table
    labs: pa.Table
    metadata: Mapping[str, Any]


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
    manifest: Mapping[str, Any]
    manifest_sha256: str


def official_split_assignment_id(samples: pa.Table | pd.DataFrame) -> str:
    """Hash canonical official sample-to-split assignments."""
    frame = samples.to_pandas() if isinstance(samples, pa.Table) else samples
    if not {"sample_id", "official_split"} <= set(frame.columns):
        raise ManifestBuildError("Official split assignments are incomplete")
    pairs = sorted(
        (str(row.sample_id), str(row.official_split))
        for row in frame[["sample_id", "official_split"]].itertuples(index=False)
    )
    if not pairs or len(pairs) != len({sample_id for sample_id, _ in pairs}):
        raise ManifestBuildError("Official split assignments are invalid")
    payload = json.dumps(pairs, ensure_ascii=True, separators=(",", ":")).encode()
    return "split-assignment-" + hashlib.sha256(payload).hexdigest()


def build_symile_artifacts(source: QualifiedSymileSource) -> SymileBuildResult:
    """Construct exact canonical Arrow artifacts from a qualified source release."""
    samples = pa.Table.from_pandas(source.samples, schema=SAMPLE_SCHEMA, preserve_index=False)
    labs = pa.Table.from_pandas(source.labs, schema=LAB_SCHEMA, preserve_index=False)
    _validate_tables(samples, labs)
    metadata = _base_metadata(source, samples)
    return SymileBuildResult(samples, labs, metadata)


def write_symile_bundle(
    result: SymileBuildResult,
    output_directory: str | Path = "data/manifests",
) -> SymileBundlePaths:
    """Validate and immutably publish one content-addressed Symile bundle."""
    dataset_root = Path(output_directory) / DATASET_ID
    builds_root = dataset_root / BUILDS_DIRECTORY
    builds_root.mkdir(parents=True, exist_ok=True)
    current_path = dataset_root / CURRENT_FILENAME
    logical_hashes = {
        SAMPLES_FILENAME: arrow_ipc_sha256(result.samples),
        LABS_FILENAME: arrow_ipc_sha256(result.labs),
    }
    bundle_id = semantic_bundle_id(result.metadata, logical_hashes)
    destination = builds_root / bundle_id
    stage = staging_directory(destination)
    try:
        pq.write_table(result.samples, stage / SAMPLES_FILENAME, compression="zstd")
        pq.write_table(result.labs, stage / LABS_FILENAME, compression="zstd")
        metadata = _finalize_metadata(result.metadata, bundle_id, stage, logical_hashes)
        (stage / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_symile_bundle(stage, expected_bundle_id=bundle_id, enforce_directory_name=False)
        if destination.exists():
            validate_symile_bundle(destination, expected_bundle_id=bundle_id)
            shutil.rmtree(stage)
        else:
            os.replace(stage, destination)
        update_current_marker(current_path, bundle_id)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return _bundle_paths(dataset_root, bundle_id)


def semantic_bundle_id(metadata: Mapping[str, Any], logical_hashes: Mapping[str, str]) -> str:
    """Return the semantic identity of meaning-bearing Symile bundle content."""
    payload = bundle_identity_payload(metadata, logical_hashes)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return "build-" + digest


def bundle_identity_payload(
    metadata: Mapping[str, Any], logical_hashes: Mapping[str, str]
) -> dict[str, object]:
    """Build the frozen semantic bundle identity payload."""
    return {
        "manifest_schema_version": metadata["manifest_schema_version"],
        "dataset": metadata["dataset"],
        "task": metadata["task"],
        "official_membership": {
            key: metadata["official_membership"][key] for key in ("source", "test_selector")
        },
        "source_release": {
            "release": metadata["source_release"]["release"],
            "checksum_manifest_sha256": metadata["source_release"]["checksum_manifest_sha256"],
        },
        "modalities": {
            key: metadata["modalities"][key] for key in ("common_locator", "cxr", "ecg", "temporal")
        },
        "laboratories": {
            key: metadata["laboratories"][key]
            for key in ("item_order", "item_names", "value_semantics", "observed_semantics")
        },
        "logical_arrow_hashes": dict(sorted(logical_hashes.items())),
    }


def resolve_symile_bundle(
    manifest_directory: str | Path = "data/manifests",
    *,
    bundle_id: str | None = None,
    full_validation: bool = False,
) -> SymileBundlePaths:
    """Resolve CURRENT once or use one explicit immutable bundle identity."""
    dataset_root = Path(manifest_directory) / DATASET_ID
    current_path = dataset_root / CURRENT_FILENAME
    resolved = bundle_id
    if resolved is None:
        if current_path.is_symlink() or not current_path.is_file():
            raise ManifestBuildError("Symile CURRENT marker is missing")
        resolved = current_path.read_text(encoding="utf-8").strip()
    if not _valid_identity(resolved, "build-"):
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
    """Perform portable full validation without reopening restricted source assets."""
    reference = validate_symile_bundle_reference(
        bundle_directory,
        expected_bundle_id=expected_bundle_id,
        enforce_directory_name=enforce_directory_name,
    )
    root = Path(bundle_directory)
    samples = pq.read_table(root / SAMPLES_FILENAME)
    labs = pq.read_table(root / LABS_FILENAME)
    _validate_tables(samples, labs)
    actual = {
        SAMPLES_FILENAME: arrow_ipc_sha256(samples),
        LABS_FILENAME: arrow_ipc_sha256(labs),
    }
    declared = reference.manifest["artifacts"]
    for filename, digest in actual.items():
        if digest != declared[filename]["logical_arrow_sha256"]:
            raise ManifestBuildError(f"Logical Arrow hash mismatch: {filename}")
    if semantic_bundle_id(reference.manifest, actual) != reference.manifest["bundle"]["bundle_id"]:
        raise ManifestBuildError("Symile semantic bundle identity does not match content")
    _validate_manifest_counts(reference.manifest, samples)
    return dict(reference.manifest)


def validate_symile_bundle_reference(
    bundle_directory: str | Path,
    *,
    expected_bundle_id: str | None = None,
    expected_manifest_sha256: str | None = None,
    enforce_directory_name: bool = True,
) -> ValidatedSymileBundleReference:
    """Lightly validate exact files, hashes, schemas, and semantic declarations."""
    root = Path(bundle_directory)
    _require_exact_regular_files(root, {METADATA_FILENAME, *_EXPECTED_FILES})
    metadata_bytes = (root / METADATA_FILENAME).read_bytes()
    manifest_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ManifestBuildError("Symile bundle-manifest SHA-256 does not match")
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Symile manifest is unreadable") from exc
    _validate_manifest(metadata)
    bundle_id = metadata["bundle"]["bundle_id"]
    if expected_bundle_id is not None and bundle_id != expected_bundle_id:
        raise ManifestBuildError("Symile bundle identity differs from the expected identity")
    if enforce_directory_name and root.name != bundle_id:
        raise ManifestBuildError("Symile build directory name differs from its identity")
    logical_hashes: dict[str, str] = {}
    for filename, schema in _EXPECTED_FILES.items():
        declaration = metadata["artifacts"][filename]
        path = root / filename
        if sha256_file(path) != declaration["physical_file_sha256"]:
            raise ManifestBuildError(f"Physical file hash mismatch: {filename}")
        if pq.read_schema(path) != schema:
            raise ManifestBuildError(f"Parquet schema mismatch: {filename}")
        logical_hashes[filename] = declaration["logical_arrow_sha256"]
    if semantic_bundle_id(metadata, logical_hashes) != bundle_id:
        raise ManifestBuildError("Symile declared semantic identity is invalid")
    return ValidatedSymileBundleReference(metadata, manifest_sha256)


def read_symile_samples(
    bundle: SymileBundlePaths,
    *,
    official_splits: Sequence[str] | None = None,
    sample_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read validated sample rows through explicit partition and identity filters."""
    filters: list[tuple[str, str, object]] = []
    if official_splits is not None:
        if not official_splits or any(value not in OFFICIAL_SPLITS for value in official_splits):
            raise ManifestBuildError("Symile sample split filter is invalid")
        filters.append(("official_split", "in", list(official_splits)))
    if sample_ids is not None:
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ManifestBuildError("Symile sample identity filter is invalid")
        filters.append(("sample_id", "in", list(sample_ids)))
    table = pq.read_table(bundle.samples_path, filters=filters or None)
    frame = table.to_pandas().sort_values("sample_id", kind="stable").reset_index(drop=True)
    if sample_ids is not None and set(frame["sample_id"]) != set(sample_ids):
        raise ManifestBuildError("Symile sample rows do not cover the requested samples")
    return frame


def read_symile_labs(bundle: SymileBundlePaths, *, sample_ids: Sequence[str]) -> pd.DataFrame:
    """Read laboratory rows for an explicit unique sample set."""
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ManifestBuildError("Symile laboratory identity filter is invalid")
    table = pq.read_table(bundle.labs_path, filters=[("sample_id", "in", list(sample_ids))])
    frame = table.to_pandas().sort_values("sample_id", kind="stable").reset_index(drop=True)
    if set(frame["sample_id"]) != set(sample_ids):
        raise ManifestBuildError("Symile laboratory rows do not cover the requested samples")
    return frame


def strict_pneumonia_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return eligible rows with a transient strict binary target."""
    if "pneumonia_state" not in frame or "sample_id" not in frame:
        raise ManifestBuildError("Symile strict-pneumonia input rows are incomplete")
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
    """Authenticate a source modality through the bundle-bound official checksum manifest."""
    reference = validate_symile_bundle_reference(
        bundle.bundle_directory, expected_bundle_id=bundle.bundle_id
    )
    path = modality_asset_path(source_root, official_split, modality)
    relative = path.relative_to(Path(source_root)).as_posix()
    return authenticate_release_asset(
        source_root,
        checksum_manifest_sha256=reference.manifest["source_release"]["checksum_manifest_sha256"],
        relative_path=relative,
    )


def _base_metadata(source: QualifiedSymileSource, samples: pa.Table) -> dict[str, Any]:
    reconciliation = dict(source.evidence["reconciliation"])
    modality_evidence = source.evidence["modality_qualification"]
    train_evidence = modality_evidence["splits"]["train"]
    counts = _strict_counts(samples.to_pandas())
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": {"dataset_id": DATASET_ID, "release": DATASET_RELEASE},
        "task": task_contract(),
        "official_membership": {
            "source": _OFFICIAL_MEMBERSHIP_SOURCE,
            "test_selector": "label == 1 and label_hadm_id == hadm_id",
            "official_split_assignment_id": official_split_assignment_id(samples),
            "counts": reconciliation,
            "strict_pneumonia_counts": counts,
        },
        "source_release": {
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
            "qualification": modality_evidence,
            "temporal": _TEMPORAL_DECLARATIONS,
        },
        "laboratories": {
            "item_order": list(LAB_ITEM_IDS),
            "item_names": {item: LAB_NAMES[item] for item in LAB_ITEM_IDS},
            "value_semantics": _LAB_VALUE_SEMANTICS,
            "observed_semantics": _LAB_OBSERVED_SEMANTICS,
            "official_percentiles": (
                "conformance reference for the upstream percentile representation"
            ),
            "labs_means": "official missing-percentile conformance reference",
        },
        "privacy": {
            "classification": "restricted patient-level data",
            "public_reporting": "aggregate only",
        },
        "provenance": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "numpy": __import__("numpy").__version__,
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
            "publication_model": "immutable-build-directory-with-atomic-CURRENT-marker",
        },
        "artifacts": {
            filename: {
                "logical_arrow_sha256": logical_hashes[filename],
                "physical_file_sha256": sha256_file(stage / filename),
            }
            for filename in sorted(_EXPECTED_FILES)
        },
        "generation": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "command": "python -m radfusion.data.symile_manifest",
        },
    }


def _validate_manifest(metadata: object) -> None:
    if not isinstance(metadata, dict) or set(metadata) != _MANIFEST_FIELDS:
        raise ManifestBuildError("Symile manifest field set is invalid")
    if (
        metadata["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION
        or metadata["dataset"] != {"dataset_id": DATASET_ID, "release": DATASET_RELEASE}
        or metadata["task"] != task_contract()
    ):
        raise ManifestBuildError("Symile manifest dataset/task contract is invalid")
    if metadata["privacy"] != {
        "classification": "restricted patient-level data",
        "public_reporting": "aggregate only",
    }:
        raise ManifestBuildError("Symile manifest privacy contract is invalid")
    if set(metadata["artifacts"]) != set(_EXPECTED_FILES):
        raise ManifestBuildError("Symile manifest artifact set is invalid")
    laboratories = metadata["laboratories"]
    if (
        not isinstance(laboratories, dict)
        or laboratories.get("item_order") != list(LAB_ITEM_IDS)
        or laboratories.get("item_names") != {item: LAB_NAMES[item] for item in LAB_ITEM_IDS}
        or laboratories.get("value_semantics") != _LAB_VALUE_SEMANTICS
        or laboratories.get("observed_semantics") != _LAB_OBSERVED_SEMANTICS
    ):
        raise ManifestBuildError("Symile manifest laboratory contract is invalid")
    membership = metadata["official_membership"]
    if (
        not isinstance(membership, dict)
        or membership.get("source") != _OFFICIAL_MEMBERSHIP_SOURCE
        or membership.get("test_selector") != "label == 1 and label_hadm_id == hadm_id"
        or not _valid_identity(membership.get("official_split_assignment_id"), "split-assignment-")
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
    source_release = metadata["source_release"]
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
    if not _valid_identity(bundle_id, "build-"):
        raise ManifestBuildError("Symile bundle identity declaration is invalid")
    for declaration in metadata["artifacts"].values():
        if (
            not isinstance(declaration, dict)
            or set(declaration) != {"logical_arrow_sha256", "physical_file_sha256"}
            or not all(_valid_sha256(value) for value in declaration.values())
        ):
            raise ManifestBuildError("Symile artifact hash declaration is invalid")


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
        set(sample_frame["official_split"]) != set(OFFICIAL_SPLITS)
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
    counts = metadata["official_membership"]["counts"]
    observed = {
        "official_admissions": len(frame),
        "train_admissions": int((frame["official_split"] == "train").sum()),
        "validation_admissions": int((frame["official_split"] == "validation").sum()),
        "test_admissions": int((frame["official_split"] == "test").sum()),
    }
    if any(counts.get(key) != value for key, value in observed.items()):
        raise ManifestBuildError("Symile manifest official membership counts do not match")
    if (
        counts.get("full_admissions")
        != counts.get("official_admissions") + counts.get("excluded_admissions")
        or counts.get("excluded_admissions")
        != counts.get("train_subject_exclusions") + counts.get("validation_subject_exclusions")
        or counts.get("unexplained_exclusions") != 0
        or counts.get("patient_overlap") != 0
        or counts.get("admission_overlap") != 0
    ):
        raise ManifestBuildError("Symile manifest reconciliation counts are invalid")
    if metadata["official_membership"]["strict_pneumonia_counts"] != _strict_counts(frame):
        raise ManifestBuildError("Symile strict-pneumonia counts do not match")
    if metadata["official_membership"][
        "official_split_assignment_id"
    ] != official_split_assignment_id(frame):
        raise ManifestBuildError("Symile official split assignment identity does not match")


def _strict_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in OFFICIAL_SPLITS:
        scoped = frame.loc[
            (frame["official_split"] == split) & frame["pneumonia_state"].isin([0, 1])
        ]
        positive = int((scoped["pneumonia_state"] == 1).sum())
        result[split] = {
            "eligible": len(scoped),
            "positive": positive,
            "negative": len(scoped) - positive,
        }
    development = {
        key: result["train"][key] + result["validation"][key]
        for key in ("eligible", "positive", "negative")
    }
    result["development"] = development
    return result


def _require_exact_regular_files(directory: Path, expected: set[str]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ManifestBuildError("Symile bundle path is not a physical directory")
    entries = list(os.scandir(directory))
    if {entry.name for entry in entries} != expected or any(
        entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in entries
    ):
        raise ManifestBuildError("Symile bundle file set is incomplete or unexpected")


def _bundle_paths(dataset_root: Path, bundle_id: str) -> SymileBundlePaths:
    directory = dataset_root / BUILDS_DIRECTORY / bundle_id
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


def _valid_identity(value: object, prefix: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(prefix)
        and _valid_sha256(value.removeprefix(prefix))
    )


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
