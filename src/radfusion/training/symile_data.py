"""Expose only the frozen Symile development cohort and required CXR tensors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.errors import ManifestBuildError
from radfusion.data.symile_artifacts import (
    SymileBundlePaths,
    authenticate_source_asset,
    read_symile_labs,
    read_symile_samples,
    resolve_symile_bundle,
    strict_pneumonia_rows,
    validate_symile_bundle_reference,
)
from radfusion.data.symile_cv import (
    CV_DIRECTORY,
    ValidatedSymileCvReference,
    validate_cv_table,
    validate_symile_cv_reference,
)
from radfusion.data.symile_preprocess import LAB_FEATURE_COLUMNS
from radfusion.data.symile_schemas import DEVELOPMENT_SPLITS, OUTER_FOLDS, REPEAT_SEEDS
from radfusion.training.config import ExperimentConfig

DEVELOPMENT_COUNT = 2_368
DEVELOPMENT_POSITIVES = 1_104
DEVELOPMENT_NEGATIVES = 1_264
INNER_SPLIT_POLICY_VERSION = "symile-inner-stratified-group-five-fold-v1"
_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True)
class SymileDevelopmentData:
    """Own validated development rows and bind their data-foundation lineage."""

    bundle: SymileBundlePaths
    bundle_manifest_sha256: str
    cv_reference: ValidatedSymileCvReference
    frame: pd.DataFrame


@dataclass(frozen=True)
class SymileOuterFold:
    """Own the model-facing frames for one fixed outer-training/OOF partition."""

    repeat_seed: int
    outer_fold: int
    training: pd.DataFrame
    holdout: pd.DataFrame


@dataclass(frozen=True)
class SymileInnerSplit:
    """Compact deterministic inner assignment for one outer fold."""

    inner_seed: int
    inner_split_id: str
    policy: Mapping[str, object]
    training_indices: np.ndarray
    validation_indices: np.ndarray


def load_symile_development(
    config: ExperimentConfig,
    *,
    enforce_production_counts: bool = True,
) -> SymileDevelopmentData:
    """Load official train/validation strict rows without touching official test samples."""
    bundle = resolve_symile_bundle(
        config.runtime.manifest_directory,
        bundle_id=config.dataset.bundle_id,
        full_validation=False,
    )
    bundle_reference = validate_symile_bundle_reference(
        bundle.bundle_directory,
        expected_bundle_id=config.dataset.bundle_id,
        expected_manifest_sha256=config.dataset.bundle_manifest_sha256,
    )
    split_id = bundle_reference.manifest["membership"]["split_assignment_id"]
    if split_id != config.dataset.split_assignment_id:
        raise ManifestBuildError("Symile official split assignment differs from configuration")
    cv_directory = (
        config.runtime.manifest_directory
        / "symile"
        / CV_DIRECTORY
        / config.dataset.cv_assignment_id
    )
    cv_reference = validate_symile_cv_reference(
        cv_directory,
        bundle_id=bundle.bundle_id,
        expected_assignment_id=config.dataset.cv_assignment_id,
    )
    samples = strict_pneumonia_rows(read_symile_samples(bundle, official_splits=DEVELOPMENT_SPLITS))
    validate_cv_table(cv_reference.assignments, samples)
    sample_ids = samples["sample_id"].astype(str).tolist()
    labs = read_symile_labs(bundle, sample_ids=sample_ids)
    frame = samples.merge(labs, on="sample_id", validate="one_to_one")
    frame = frame.sort_values("sample_id", kind="stable").reset_index(drop=True)
    if tuple(frame.loc[:, LAB_FEATURE_COLUMNS].columns) != LAB_FEATURE_COLUMNS:
        raise ManifestBuildError("Symile development laboratory order is invalid")
    positives = int((frame["target"] == 1).sum())
    if enforce_production_counts and (
        len(frame) != DEVELOPMENT_COUNT
        or positives != DEVELOPMENT_POSITIVES
        or len(frame) - positives != DEVELOPMENT_NEGATIVES
    ):
        raise ManifestBuildError(
            "Symile development cohort differs from the frozen core-development contract"
        )
    return SymileDevelopmentData(
        bundle,
        bundle_reference.manifest_sha256,
        cv_reference,
        frame,
    )


def materialize_outer_fold(
    data: SymileDevelopmentData, *, repeat_seed: int, outer_fold: int
) -> SymileOuterFold:
    """Materialize one exact frozen outer fold from the experimental-design assignment."""
    if repeat_seed not in REPEAT_SEEDS or outer_fold not in OUTER_FOLDS:
        raise ManifestBuildError("Symile outer-fold coordinates are invalid")
    assignments = data.cv_reference.assignments.to_pandas()
    scoped = assignments.loc[assignments["repeat_seed"] == repeat_seed, ["sample_id", "outer_fold"]]
    merged = data.frame.merge(scoped, on="sample_id", validate="one_to_one")
    merged = merged.sort_values("sample_id", kind="stable").reset_index(drop=True)
    if len(merged) != len(data.frame):
        raise ManifestBuildError("Symile outer fold does not cover the development cohort")
    holdout = merged.loc[merged["outer_fold"] == outer_fold].copy()
    training = merged.loc[merged["outer_fold"] != outer_fold].copy()
    if holdout.empty or training.empty:
        raise ManifestBuildError("Symile outer fold contains an empty partition")
    if set(holdout["subject_id"]) & set(training["subject_id"]):
        raise ManifestBuildError("Symile outer fold splits a patient")
    return SymileOuterFold(
        repeat_seed,
        outer_fold,
        training.reset_index(drop=True),
        holdout.reset_index(drop=True),
    )


def derive_inner_split(outer: SymileOuterFold) -> SymileInnerSplit:
    """Derive the frozen deterministic patient-grouped inner split."""
    frame = outer.training.sort_values("sample_id", kind="stable").reset_index(drop=True)
    inner_seed = derive_inner_seed(outer.repeat_seed, outer.outer_fold)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=inner_seed)
    generated = list(
        splitter.split(
            np.arange(len(frame), dtype=np.int64).reshape(-1, 1),
            frame["target"].to_numpy(dtype=np.int8),
            frame["subject_id"].to_numpy(dtype=np.int64),
        )
    )
    training_indices, validation_indices = generated[0]
    training_indices = np.asarray(training_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    if set(frame.iloc[training_indices]["subject_id"]) & set(
        frame.iloc[validation_indices]["subject_id"]
    ):
        raise ManifestBuildError("Symile inner split divides a patient")
    if set(frame.iloc[training_indices]["target"]) != {0, 1}:
        raise ManifestBuildError("Symile inner training lacks one target class")
    if set(frame.iloc[validation_indices]["target"]) != {0, 1}:
        raise ManifestBuildError("Symile inner validation lacks one target class")
    roles = np.full(len(frame), "inner_training", dtype=object)
    roles[validation_indices] = "inner_validation"
    policy: dict[str, object] = {
        "policy_version": INNER_SPLIT_POLICY_VERSION,
        "algorithm": "sklearn.model_selection.StratifiedGroupKFold",
        "n_splits": 5,
        "shuffle": True,
        "generated_validation_fold": 0,
        "group_field": "subject_id",
        "stratification_target": "pneumonia_strict",
        "repeat_seed": outer.repeat_seed,
        "outer_fold": outer.outer_fold,
        "inner_seed": inner_seed,
    }
    identity_payload = {
        "policy": policy,
        "assignments": sorted(zip(frame["sample_id"].astype(str), roles, strict=True)),
    }
    digest = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SymileInnerSplit(
        inner_seed,
        "inner-split-" + digest,
        policy,
        training_indices,
        validation_indices,
    )


def derive_inner_seed(repeat_seed: int, outer_fold: int) -> int:
    """Derive the canonical inner-split seed for one frozen outer coordinate."""
    if repeat_seed not in REPEAT_SEEDS or outer_fold not in OUTER_FOLDS:
        raise ManifestBuildError("Symile inner-split coordinates are invalid")
    payload = f"symile-inner-split\0{repeat_seed}\0{outer_fold}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


class SymileCxrStore:
    """Authenticate and expose read-only official train/validation CXR tensors."""

    def __init__(self, bundle: SymileBundlePaths, source_root: str | Path) -> None:
        self._arrays: dict[str, np.memmap] = {}
        for split in DEVELOPMENT_SPLITS:
            path = authenticate_source_asset(
                bundle,
                source_root,
                official_split=split,
                modality="cxr",
            )
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                not isinstance(array, np.memmap)
                or array.dtype != np.float32
                or array.ndim != 4
                or array.shape[1:] != (3, 320, 320)
            ):
                raise ManifestBuildError("Symile development CXR tensor header is invalid")
            self._arrays[split] = array

    def canonical_image(self, official_split: str, source_row: int) -> np.ndarray:
        """Reconstruct one finite repeated-grayscale CXR in [0, 1]."""
        if official_split not in DEVELOPMENT_SPLITS:
            raise ManifestBuildError("CXR access is limited to Symile development splits")
        array = self._arrays[official_split]
        if (
            isinstance(source_row, bool)
            or not isinstance(source_row, int | np.integer)
            or not 0 <= int(source_row) < len(array)
        ):
            raise ManifestBuildError("Symile CXR source row is invalid")
        normalized = np.asarray(array[int(source_row)], dtype=np.float32)
        if not np.isfinite(normalized).all():
            raise ManifestBuildError("Symile CXR contains non-finite values")
        restored = normalized * _IMAGENET_STD + _IMAGENET_MEAN
        if restored.min() < -2e-6 or restored.max() > 1.000002:
            raise ManifestBuildError("Symile CXR inverse normalization is outside [0, 1]")
        if not (
            np.allclose(restored[0], restored[1], atol=2e-6, rtol=0.0)
            and np.allclose(restored[0], restored[2], atol=2e-6, rtol=0.0)
        ):
            raise ManifestBuildError("Symile CXR channels are not repeated grayscale")
        return np.clip(restored[0], 0.0, 1.0).astype(np.float32, copy=False)


class SymileNeuralDataset(Dataset[dict[str, object]]):
    """Read aligned development CXR and optional transformed labs for one fold view."""

    epoch_tagged_requests = True

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        cxr_store: SymileCxrStore,
        transform: StandardCxrTransform,
        repeat_seed: int,
        labs: np.ndarray | None = None,
    ) -> None:
        required = {"sample_id", "subject_id", "official_split", "source_row", "target"}
        if frame.empty or not required <= set(frame.columns):
            raise ManifestBuildError("Symile neural frame is incomplete")
        ordered = frame.sort_values("sample_id", kind="stable").reset_index(drop=True)
        if not frame.reset_index(drop=True).equals(ordered):
            raise ManifestBuildError("Symile neural frame must be canonically ordered")
        matrix = None
        if labs is not None:
            matrix = np.asarray(labs, dtype=np.float32)
            if matrix.shape != (len(frame), 100) or not np.isfinite(matrix).all():
                raise ManifestBuildError("Symile neural laboratory matrix is invalid")
        self._frame = frame.reset_index(drop=True)
        self._cxr_store = cxr_store
        self._transform = transform
        self._repeat_seed = repeat_seed
        self._labs = matrix

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, request: int | tuple[int, int]) -> dict[str, object]:
        epoch, index = request if isinstance(request, tuple) else (0, request)
        row = self._frame.iloc[index]
        sample_id = str(row["sample_id"])
        canonical = self._cxr_store.canonical_image(
            str(row["official_split"]), int(row["source_row"])
        )
        base = self._transform.deterministic_base(canonical)
        augmentation_seed = (
            _augmentation_seed(self._repeat_seed, epoch, sample_id)
            if self._transform.training
            else None
        )
        result: dict[str, object] = {
            "image": self._transform.from_deterministic_base(
                base, augmentation_seed=augmentation_seed
            ),
            "target": torch.tensor(float(row["target"]), dtype=torch.float32),
            "sample_id": sample_id,
            "patient_id": str(int(row["subject_id"])),
        }
        if self._labs is not None:
            result["structured"] = torch.from_numpy(self._labs[index].copy())
        return result


def _augmentation_seed(repeat_seed: int, epoch: int, sample_id: str) -> int:
    if repeat_seed not in REPEAT_SEEDS or epoch < 0:
        raise ManifestBuildError("Symile augmentation seed inputs are invalid")
    payload = f"radfusion-symile-augmentation\0{repeat_seed}\0{epoch}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)
