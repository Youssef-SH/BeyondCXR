"""Narrow ECG source access and composed three-modality Symile datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_artifacts import (
    SymileBundlePaths,
    validate_symile_bundle_reference,
)
from beyondcxr.data.symile_schemas import DEVELOPMENT_SPLITS
from beyondcxr.data.symile_source import (
    AuthenticatedReleaseAsset,
    establish_authenticated_release_asset,
    modality_asset_path,
    reopen_authenticated_release_memmap,
)
from beyondcxr.training.symile_data import (
    SymileCxrStore,
    derive_augmentation_seed,
    validated_symile_lab_matrix,
)


class SymileEcgStore:
    """Authenticate and expose only development ECG tensor rows."""

    def __init__(self, bundle: SymileBundlePaths, source_root: str | Path) -> None:
        reference = validate_symile_bundle_reference(
            bundle.bundle_directory, expected_bundle_id=bundle.bundle_id
        )
        self._source_root = Path(source_root).absolute()
        self._checksum_sha256 = reference.manifest["source"]["checksum_manifest_sha256"]
        self._arrays: dict[str, np.memmap] = {}
        self._authorities: dict[str, AuthenticatedReleaseAsset] = {}
        for split in DEVELOPMENT_SPLITS:
            path = modality_asset_path(self._source_root, split, "ecg")
            authority = establish_authenticated_release_asset(
                self._source_root,
                checksum_manifest_sha256=self._checksum_sha256,
                relative_path=path.absolute().relative_to(self._source_root).as_posix(),
            )
            array = reopen_authenticated_release_memmap(authority)
            if (
                not isinstance(array, np.memmap)
                or array.dtype != np.float32
                or array.ndim != 4
                or array.shape[1:] != (1, 5000, 12)
            ):
                raise ManifestBuildError("Symile ECG tensor header is invalid")
            self._arrays[split] = array
            self._authorities[split] = authority

    def __getstate__(self) -> dict[str, object]:
        """Send authenticated file references, not tensor contents, to spawned workers."""
        return {
            "source_root": self._source_root,
            "checksum_sha256": self._checksum_sha256,
            "arrays": {
                split: (self._authorities[split], array.shape, array.dtype.str)
                for split, array in self._arrays.items()
            },
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self._source_root = state["source_root"]
        self._checksum_sha256 = state["checksum_sha256"]
        self._arrays = {}
        self._authorities = {}
        for split, (authority, shape, dtype) in state["arrays"].items():
            array = reopen_authenticated_release_memmap(authority)
            if not isinstance(array, np.memmap) or array.shape != shape or array.dtype.str != dtype:
                raise ManifestBuildError("Symile ECG tensor header changed before worker access")
            self._arrays[split] = array
            self._authorities[split] = authority

    def signal(self, official_split: str, source_row: int) -> np.ndarray:
        """Return a finite nonzero float32 ECG in model order 12 x 5000."""
        if official_split not in DEVELOPMENT_SPLITS:
            raise ManifestBuildError("Symile ECG access is limited to Symile development splits")
        if isinstance(source_row, bool) or not isinstance(source_row, int | np.integer):
            raise ManifestBuildError("Symile ECG source row is invalid")
        array = self._arrays[official_split]
        if not 0 <= int(source_row) < len(array):
            raise ManifestBuildError("Symile ECG source row is out of range")
        value = np.asarray(array[int(source_row)], dtype=np.float32)
        if (
            value.shape != (1, 5000, 12)
            or not np.isfinite(value).all()
            or value.min() < -1.0
            or value.max() > 1.0
            or bool(np.all(value == 0.0))
        ):
            raise ManifestBuildError("Symile ECG source value violates the frozen contract")
        return np.ascontiguousarray(value[0].T, dtype=np.float32)


class SymileTriModalDataset(Dataset[dict[str, object]]):
    """Compose existing CXR/lab inputs with the separate Symile ECG source view."""

    epoch_tagged_requests = True

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        cxr_store: SymileCxrStore,
        ecg_store: SymileEcgStore,
        transform: StandardCxrTransform,
        training_seed: int,
        labs: np.ndarray,
    ) -> None:
        required = {"sample_id", "subject_id", "official_split", "source_row", "target"}
        ordered = frame.sort_values("sample_id", kind="stable").reset_index(drop=True)
        matrix = validated_symile_lab_matrix(labs, rows=len(frame))
        if (
            frame.empty
            or not required <= set(frame.columns)
            or not frame.reset_index(drop=True).equals(ordered)
        ):
            raise ManifestBuildError("Symile tri-modal dataset inputs are invalid")
        self._frame = ordered
        self._cxr_store = cxr_store
        self._ecg_store = ecg_store
        self._transform = transform
        self._training_seed = training_seed
        self._labs = matrix

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, request: int | tuple[int, int]) -> dict[str, object]:
        epoch, index = request if isinstance(request, tuple) else (0, request)
        row = self._frame.iloc[index]
        sample_id = str(row["sample_id"])
        image = self._cxr_store.canonical_image(str(row["official_split"]), int(row["source_row"]))
        base = self._transform.deterministic_base(image)
        augmentation_seed = (
            derive_augmentation_seed(self._training_seed, epoch, sample_id)
            if self._transform.training
            else None
        )
        return {
            "image": self._transform.from_deterministic_base(
                base,
                augmentation_seed=augmentation_seed,
            ),
            "structured": torch.from_numpy(self._labs[index].copy()),
            "ecg": torch.from_numpy(
                self._ecg_store.signal(str(row["official_split"]), int(row["source_row"]))
            ),
            "target": torch.tensor(float(row["target"]), dtype=torch.float32),
            "sample_id": sample_id,
            "patient_id": str(int(row["subject_id"])),
        }
