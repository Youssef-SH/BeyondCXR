"""Generate and publish immutable repeated patient-grouped CV assignments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from radfusion.data.bundle_contract import BUNDLE_PREFIX, valid_bundle_id
from radfusion.data.errors import ManifestBuildError
from radfusion.data.hashing import logical_arrow_sha256, sha256_file
from radfusion.data.symile_artifacts import (
    SymileBundlePaths,
    read_symile_samples,
    resolve_symile_bundle,
    strict_pneumonia_rows,
)
from radfusion.data.symile_schemas import (
    CV_SCHEMA,
    DEVELOPMENT_SPLITS,
    LABEL_POLICY_VERSION,
    OUTER_FOLDS,
    REPEAT_SEEDS,
    TASK_ID,
)
from radfusion.utils.operational_logging import (
    add_logging_argument,
    configure_logging,
    get_operational_logger,
    timed_phase,
)
from radfusion.utils.publication import staging_directory

CV_DIRECTORY = "cv"
CV_ASSIGNMENTS_FILENAME = "assignments.parquet"
CV_MANIFEST_FILENAME = "manifest.json"
_EXPECTED_FILES = {CV_ASSIGNMENTS_FILENAME, CV_MANIFEST_FILENAME}
_LOGGER = get_operational_logger(__name__)


@dataclass(frozen=True)
class ValidatedSymileCvReference:
    """Validated CV declaration and assignments without bundle sample access."""

    manifest: dict[str, Any]
    manifest_sha256: str
    assignments: pa.Table


def generate_cv_assignments(samples: pd.DataFrame) -> pa.Table:
    """Generate the exact frozen 3x5 patient-grouped assignments."""
    eligible = strict_pneumonia_rows(samples)
    eligible = eligible.loc[eligible["official_split"].isin(DEVELOPMENT_SPLITS)]
    eligible = eligible.sort_values("sample_id", kind="stable").reset_index(drop=True)
    if eligible.empty:
        raise ManifestBuildError("Symile development cohort is empty")
    records: list[dict[str, object]] = []
    x = np.arange(len(eligible), dtype=np.int64).reshape(-1, 1)
    y = eligible["target"].to_numpy(dtype=np.int8)
    groups = eligible["subject_id"].to_numpy(dtype=np.int64)
    for seed in REPEAT_SEEDS:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        assigned = np.full(len(eligible), -1, dtype=np.int8)
        for fold, (_, holdout_indices) in enumerate(splitter.split(x, y, groups)):
            assigned[holdout_indices] = fold
        if np.any(assigned < 0):
            raise ManifestBuildError("StratifiedGroupKFold left samples unassigned")
        records.extend(
            {
                "sample_id": sample_id,
                "repeat_seed": seed,
                "outer_fold": int(fold),
            }
            for sample_id, fold in zip(eligible["sample_id"], assigned, strict=True)
        )
    table = pa.Table.from_pylist(records, schema=CV_SCHEMA).sort_by(
        [("repeat_seed", "ascending"), ("sample_id", "ascending")]
    )
    validate_cv_table(table, samples)
    return table


def validate_cv_table(assignments: pa.Table, samples: pd.DataFrame) -> None:
    """Validate deterministic coverage, grouping, class presence, and domains."""
    if assignments.schema != CV_SCHEMA or assignments.num_rows == 0:
        raise ManifestBuildError("Symile CV assignment schema is invalid")
    frame = assignments.to_pandas()
    ordered = frame.sort_values(["repeat_seed", "sample_id"], kind="stable").reset_index(drop=True)
    if not frame.reset_index(drop=True).equals(ordered):
        raise ManifestBuildError("Symile CV assignments are not canonically ordered")
    if (
        frame[["sample_id", "repeat_seed"]].duplicated().any()
        or set(frame["repeat_seed"]) != set(REPEAT_SEEDS)
        or set(frame["outer_fold"]) != set(OUTER_FOLDS)
    ):
        raise ManifestBuildError("Symile CV assignment key or domain is invalid")
    eligible = strict_pneumonia_rows(samples)
    eligible = eligible.loc[eligible["official_split"].isin(DEVELOPMENT_SPLITS)]
    expected_ids = set(eligible["sample_id"])
    merged = frame.merge(
        eligible[["sample_id", "subject_id", "target"]], on="sample_id", validate="many_to_one"
    )
    if len(merged) != len(frame):
        raise ManifestBuildError("Symile CV assignments contain ineligible samples")
    for seed in REPEAT_SEEDS:
        scoped = merged.loc[merged["repeat_seed"] == seed]
        if set(scoped["sample_id"]) != expected_ids or len(scoped) != len(expected_ids):
            raise ManifestBuildError("Symile CV repeat does not cover every eligible sample once")
        patient_folds = scoped.groupby("subject_id")["outer_fold"].nunique()
        if not patient_folds.eq(1).all():
            raise ManifestBuildError("Symile CV repeat splits a patient across folds")
        for fold in OUTER_FOLDS:
            fold_rows = scoped.loc[scoped["outer_fold"] == fold]
            if fold_rows.empty or set(fold_rows["target"]) != {0, 1}:
                raise ManifestBuildError("Symile CV fold is empty or lacks one target class")


def cv_assignment_id(
    bundle_id: str,
    assignment_logical_hash: str,
) -> str:
    """Hash the complete meaning-bearing CV design and assignment content."""
    payload = _cv_identity_payload(bundle_id, assignment_logical_hash)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return "cv-assignment-" + digest


def publish_symile_cv(
    bundle: SymileBundlePaths,
    *,
    manifest_directory: str | Path = "data/manifests",
) -> tuple[str, Path]:
    """Generate, validate, and immutably publish the bundle-bound CV artifact."""
    samples = read_symile_samples(bundle)
    assignments = generate_cv_assignments(samples)
    logical_hash = logical_arrow_sha256(assignments)
    assignment_id = cv_assignment_id(bundle.bundle_id, logical_hash)
    destination = Path(manifest_directory) / "symile" / CV_DIRECTORY / assignment_id
    stage = staging_directory(destination)
    try:
        pq.write_table(assignments, stage / CV_ASSIGNMENTS_FILENAME, compression="zstd")
        manifest = _cv_manifest(bundle.bundle_id, assignment_id, assignments, logical_hash, stage)
        (stage / CV_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_symile_cv(stage, bundle=bundle, enforce_directory_name=False)
        if destination.exists():
            validate_symile_cv(destination, bundle=bundle)
            shutil.rmtree(stage)
        else:
            os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return assignment_id, destination


def validate_symile_cv(
    directory: str | Path,
    *,
    bundle: SymileBundlePaths,
    expected_assignment_id: str | None = None,
    enforce_directory_name: bool = True,
) -> dict[str, Any]:
    """Validate an immutable CV artifact against its exact source bundle."""
    reference = validate_symile_cv_reference(
        directory,
        bundle_id=bundle.bundle_id,
        expected_assignment_id=expected_assignment_id,
        enforce_directory_name=enforce_directory_name,
    )
    samples = read_symile_samples(bundle)
    validate_cv_table(reference.assignments, samples)
    return reference.manifest


def validate_symile_cv_reference(
    directory: str | Path,
    *,
    bundle_id: str,
    expected_assignment_id: str | None = None,
    expected_manifest_sha256: str | None = None,
    enforce_directory_name: bool = True,
) -> ValidatedSymileCvReference:
    """Validate one immutable CV artifact without reading bundle sample rows."""
    root = Path(directory)
    _require_exact_files(root)
    try:
        manifest_bytes = (root / CV_MANIFEST_FILENAME).read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ManifestBuildError("Symile CV manifest is unreadable") from exc
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ManifestBuildError("Symile CV manifest SHA-256 differs from expected identity")
    _validate_cv_manifest(manifest)
    assignment_id = manifest["cv_assignment_id"]
    if expected_assignment_id is not None and assignment_id != expected_assignment_id:
        raise ManifestBuildError("Symile CV identity differs from expected identity")
    if enforce_directory_name and root.name != assignment_id:
        raise ManifestBuildError("Symile CV directory differs from its identity")
    if manifest["bundle_id"] != bundle_id:
        raise ManifestBuildError("Symile CV artifact is bound to a different bundle")
    artifact = manifest["artifact"]
    path = root / CV_ASSIGNMENTS_FILENAME
    if sha256_file(path) != artifact["physical_file_sha256"] or pq.read_schema(path) != CV_SCHEMA:
        raise ManifestBuildError("Symile CV artifact physical integrity is invalid")
    assignments = pq.read_table(path)
    if artifact["row_count"] != assignments.num_rows:
        raise ManifestBuildError("Symile CV artifact row count does not match")
    logical_hash = logical_arrow_sha256(assignments)
    if (
        logical_hash != artifact["logical_arrow_sha256"]
        or cv_assignment_id(bundle_id, logical_hash) != assignment_id
    ):
        raise ManifestBuildError("Symile CV semantic identity is invalid")
    return ValidatedSymileCvReference(dict(manifest), manifest_sha256, assignments)


def _cv_identity_payload(bundle_id: str, logical_hash: str) -> dict[str, object]:
    return {
        "bundle_id": bundle_id,
        "task_id": TASK_ID,
        "label_policy_version": LABEL_POLICY_VERSION,
        "eligible_official_splits": list(DEVELOPMENT_SPLITS),
        "group_field": "subject_id",
        "stratification_target": TASK_ID,
        "algorithm": "sklearn.model_selection.StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "repeat_seeds": list(REPEAT_SEEDS),
        "artifacts": {"assignments": logical_hash},
    }


def _cv_manifest(
    bundle_id: str,
    assignment_id: str,
    assignments: pa.Table,
    logical_hash: str,
    stage: Path,
) -> dict[str, object]:
    return {
        "cv_assignment_id": assignment_id,
        "bundle_id": bundle_id,
        **{
            key: value
            for key, value in _cv_identity_payload(bundle_id, logical_hash).items()
            if key != "bundle_id"
        },
        "artifact": {
            "filename": CV_ASSIGNMENTS_FILENAME,
            "logical_arrow_sha256": logical_hash,
            "physical_file_sha256": sha256_file(stage / CV_ASSIGNMENTS_FILENAME),
            "row_count": assignments.num_rows,
        },
        "provenance": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pyarrow": pa.__version__,
        },
        "generation": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "command": "python -m radfusion.data.symile_cv",
        },
    }


def _validate_cv_manifest(manifest: object) -> None:
    identity_fields = _cv_identity_payload(BUNDLE_PREFIX + "0" * 64, "0" * 64)
    expected = {
        "cv_assignment_id",
        *identity_fields,
        "artifact",
        "provenance",
        "generation",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise ManifestBuildError("Symile CV manifest field set is invalid")
    bundle_id = manifest.get("bundle_id")
    semantic_artifacts = manifest.get("artifacts")
    logical_hash = (
        semantic_artifacts.get("assignments") if isinstance(semantic_artifacts, dict) else None
    )
    if (
        not valid_bundle_id(bundle_id)
        or not isinstance(logical_hash, str)
        or len(logical_hash) != 64
        or not all(character in "0123456789abcdef" for character in logical_hash)
    ):
        raise ManifestBuildError("Symile CV source identity declarations are invalid")
    design = {
        key: value
        for key, value in _cv_identity_payload(bundle_id, logical_hash).items()
        if key != "bundle_id"
    }
    if any(manifest.get(key) != value for key, value in design.items()):
        raise ManifestBuildError("Symile CV design contract is invalid")
    if not _identity(manifest.get("cv_assignment_id")):
        raise ManifestBuildError("Symile CV identity declaration is invalid")
    artifact = manifest.get("artifact")
    if (
        not isinstance(artifact, dict)
        or set(artifact)
        != {
            "filename",
            "logical_arrow_sha256",
            "physical_file_sha256",
            "row_count",
        }
        or artifact.get("filename") != CV_ASSIGNMENTS_FILENAME
        or artifact.get("logical_arrow_sha256") != logical_hash
        or not isinstance(artifact.get("physical_file_sha256"), str)
        or len(artifact["physical_file_sha256"]) != 64
        or isinstance(artifact.get("row_count"), bool)
        or not isinstance(artifact.get("row_count"), int)
        or artifact["row_count"] <= 0
    ):
        raise ManifestBuildError("Symile CV artifact declaration is invalid")


def _require_exact_files(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ManifestBuildError("Symile CV path is not a physical directory")
    entries = list(os.scandir(root))
    if {entry.name for entry in entries} != _EXPECTED_FILES or any(
        entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in entries
    ):
        raise ManifestBuildError("Symile CV file set is incomplete or unexpected")


def _identity(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("cv-assignment-")
        and len(value) == 78
        and all(character in "0123456789abcdef" for character in value[14:])
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-directory", type=Path, default=Path("data/manifests"))
    parser.add_argument("--bundle-id", default=None)
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        with timed_phase(_LOGGER, "bundle_resolution"):
            bundle = resolve_symile_bundle(
                args.manifest_directory,
                bundle_id=args.bundle_id,
                full_validation=True,
            )
        with timed_phase(_LOGGER, "cv_publication"):
            assignment_id, directory = publish_symile_cv(
                bundle,
                manifest_directory=args.manifest_directory,
            )
    except (ManifestBuildError, OSError, ValueError, KeyError) as exc:
        print(f"Symile CV generation failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "bundle_id": bundle.bundle_id,
                "cv_assignment_id": assignment_id,
                "cv_directory": directory.as_posix(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
