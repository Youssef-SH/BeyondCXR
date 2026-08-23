"""Capability-gated official Symile test views used only after the pre-test freeze."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_artifacts import (
    SymileBundlePaths,
    authenticate_source_asset,
    read_symile_labs,
    read_symile_samples,
    resolve_symile_bundle,
    strict_pneumonia_rows,
    validate_symile_bundle_reference,
)
from beyondcxr.data.symile_preprocess import LAB_FEATURE_COLUMNS
from beyondcxr.data.symile_schemas import LABEL_POLICY_VERSION, TASK_ID
from beyondcxr.training.symile_campaign_control import (
    ValidatedPretestFreeze,
    ValidatedTestOpenRecord,
    materialize_official_test,
    validated_pretest_freeze_manifest,
)
from beyondcxr.training.symile_data import validated_symile_lab_matrix
from beyondcxr.training.symile_final_packages import FinalPackageConfig
from beyondcxr.utils.private_predictions import (
    SYMILE_TEST_INFERENCE_POLICY,
    ValidatedPredictionEvidence,
)

_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)
STRICT_PNEUMONIA_TEST_ADMISSIONS = 110
_HELD_OUT_REFERENCE_COLUMNS = (
    "sample_id",
    "target",
    "subject_id",
)


@dataclass(frozen=True, eq=False)
class HeldOutEvaluationProjection:
    """Ordinary validated projection of the materialized official-test cohort."""

    dataset_id: str
    bundle_id: str
    split_assignment_id: str
    task_id: str
    label_policy_version: str
    scope: str
    inference_policy: Mapping[str, object]
    freeze_id: str
    _frame: pd.DataFrame

    def __post_init__(self) -> None:
        frame = self._frame.copy(deep=True)
        _validate_held_out_projection(frame)
        object.__setattr__(self, "_frame", frame)

    def frame(self) -> pd.DataFrame:
        """Return an isolated copy of the canonical private held-out projection."""
        return self._frame.copy(deep=True)

    def prediction_targets(self) -> pd.DataFrame:
        """Return the exact ordered sample/target projection for prediction validation."""
        return self._frame.loc[:, ["sample_id", "target"]].copy()


class FrozenSymileTestData:
    """A test-only bundle view that cannot be created without the freeze capability."""

    def __init__(
        self,
        capability: ValidatedPretestFreeze,
        test_open_record: ValidatedTestOpenRecord,
        config: FinalPackageConfig,
        *,
        manifest_root: str | Path,
    ) -> None:
        materialize_official_test(capability, test_open_record, lambda: None)
        freeze_manifest = validated_pretest_freeze_manifest(capability)
        frozen_bundle = freeze_manifest.get("bundle")
        frozen_task = freeze_manifest.get("task")
        if (
            not isinstance(frozen_bundle, Mapping)
            or not isinstance(frozen_task, Mapping)
            or config.dataset_id != "symile"
            or config.bundle_id != frozen_bundle.get("bundle_id")
            or config.bundle_manifest_sha256 != frozen_bundle.get("bundle_manifest_sha256")
            or config.split_assignment_id != frozen_bundle.get("split_assignment_id")
            or config.task_id != frozen_task.get("task_id")
            or config.label_policy_version != frozen_task.get("label_policy_version")
        ):
            raise ManifestBuildError("Frozen test configuration differs from pre-test authority")
        bundle = resolve_symile_bundle(
            manifest_root,
            bundle_id=config.bundle_id,
            full_validation=False,
        )
        reference = validate_symile_bundle_reference(
            bundle.bundle_directory,
            expected_bundle_id=config.bundle_id,
            expected_manifest_sha256=config.bundle_manifest_sha256,
        )
        if reference.manifest["membership"]["split_assignment_id"] != config.split_assignment_id:
            raise ManifestBuildError("Frozen test bundle split lineage differs from configuration")
        samples = strict_pneumonia_rows(read_symile_samples(bundle, official_splits=("test",)))
        frame = (
            samples.merge(
                read_symile_labs(bundle, sample_ids=samples["sample_id"].astype(str).tolist()),
                on="sample_id",
                validate="one_to_one",
            )
            .sort_values("sample_id", kind="stable")
            .reset_index(drop=True)
        )
        if len(frame) != STRICT_PNEUMONIA_TEST_ADMISSIONS or set(frame["target"].tolist()) != {
            0,
            1,
        }:
            raise ManifestBuildError("Frozen strict-pneumonia test cohort is invalid")
        self.bundle = bundle
        self.frame = frame
        self.split_assignment_id = config.split_assignment_id
        self._capability = capability

    def evaluation_projection(self) -> HeldOutEvaluationProjection:
        """Return validated private rows used by prediction and result validation."""
        projection = (
            self.frame.loc[:, ["sample_id", "target", "subject_id"]]
            .loc[:, _HELD_OUT_REFERENCE_COLUMNS]
            .sort_values("sample_id", kind="stable")
            .reset_index(drop=True)
        )
        return HeldOutEvaluationProjection(
            dataset_id="symile",
            bundle_id=self.bundle.bundle_id,
            split_assignment_id=self.split_assignment_id,
            task_id=TASK_ID,
            label_policy_version=LABEL_POLICY_VERSION,
            scope="test",
            inference_policy=dict(SYMILE_TEST_INFERENCE_POLICY),
            freeze_id=self._capability.freeze_id,
            _frame=projection,
        )


class FrozenSymileTestCxrStore:
    """Authenticate and expose only official test CXR tensors after authorization."""

    def __init__(
        self,
        capability: ValidatedPretestFreeze,
        test_open_record: ValidatedTestOpenRecord,
        bundle: SymileBundlePaths,
        source_root: str | Path,
    ) -> None:
        materialize_official_test(capability, test_open_record, lambda: None)
        path = authenticate_source_asset(bundle, source_root, official_split="test", modality="cxr")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            not isinstance(array, np.memmap)
            or array.dtype != np.float32
            or array.ndim != 4
            or array.shape[1:] != (3, 320, 320)
        ):
            raise ManifestBuildError("Frozen test CXR tensor header is invalid")
        self._array = array

    def canonical_image(self, source_row: int) -> np.ndarray:
        value = self._tensor(source_row)
        restored = value * _IMAGENET_STD + _IMAGENET_MEAN
        if (
            restored.min() < -2e-6
            or restored.max() > 1.000002
            or not np.allclose(restored[0], restored[1], atol=2e-6, rtol=0.0)
            or not np.allclose(restored[0], restored[2], atol=2e-6, rtol=0.0)
        ):
            raise ManifestBuildError("Frozen test CXR representation violates the source contract")
        return np.clip(restored[0], 0.0, 1.0).astype(np.float32, copy=False)

    def _tensor(self, source_row: int) -> np.ndarray:
        if isinstance(source_row, bool) or not isinstance(source_row, int | np.integer):
            raise ManifestBuildError("Frozen test CXR source row is invalid")
        if not 0 <= int(source_row) < len(self._array):
            raise ManifestBuildError("Frozen test CXR source row is out of range")
        value = np.asarray(self._array[int(source_row)], dtype=np.float32)
        if not np.isfinite(value).all():
            raise ManifestBuildError("Frozen test CXR tensor contains non-finite values")
        return value


class FrozenSymileTestEcgStore:
    """Authenticate and expose only official test ECG tensors after authorization."""

    def __init__(
        self,
        capability: ValidatedPretestFreeze,
        test_open_record: ValidatedTestOpenRecord,
        bundle: SymileBundlePaths,
        source_root: str | Path,
    ) -> None:
        materialize_official_test(capability, test_open_record, lambda: None)
        path = authenticate_source_asset(bundle, source_root, official_split="test", modality="ecg")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            not isinstance(array, np.memmap)
            or array.dtype != np.float32
            or array.ndim != 4
            or array.shape[1:] != (1, 5000, 12)
        ):
            raise ManifestBuildError("Frozen test ECG tensor header is invalid")
        self._array = array

    def signal(self, source_row: int) -> np.ndarray:
        if isinstance(source_row, bool) or not isinstance(source_row, int | np.integer):
            raise ManifestBuildError("Frozen test ECG source row is invalid")
        if not 0 <= int(source_row) < len(self._array):
            raise ManifestBuildError("Frozen test ECG source row is out of range")
        value = np.asarray(self._array[int(source_row)], dtype=np.float32)
        if (
            value.shape != (1, 5000, 12)
            or not np.isfinite(value).all()
            or value.min() < -1.0
            or value.max() > 1.0
            or bool(np.all(value == 0.0))
        ):
            raise ManifestBuildError("Frozen test ECG tensor violates the source contract")
        return np.ascontiguousarray(value[0].T, dtype=np.float32)


class FrozenSymileNeuralTestDataset(Dataset[dict[str, object]]):
    """A test-only neural view with optional fitted labs and ECG inputs."""

    def __init__(
        self,
        capability: ValidatedPretestFreeze,
        test_open_record: ValidatedTestOpenRecord,
        frame: pd.DataFrame,
        *,
        cxr_store: FrozenSymileTestCxrStore,
        transform: StandardCxrTransform,
        labs: np.ndarray | None = None,
        ecg_store: FrozenSymileTestEcgStore | None = None,
    ) -> None:
        materialize_official_test(capability, test_open_record, lambda: None)
        required = {"sample_id", "subject_id", "source_row", "target"}
        ordered = frame.sort_values("sample_id", kind="stable").reset_index(drop=True)
        matrix = None if labs is None else validated_symile_lab_matrix(labs, rows=len(frame))
        if (
            frame.empty
            or not required <= set(frame.columns)
            or not frame.reset_index(drop=True).equals(ordered)
        ):
            raise ManifestBuildError("Frozen neural test dataset is invalid")
        self._frame = ordered
        self._cxr = cxr_store
        self._transform = transform
        self._labs = matrix
        self._ecg = ecg_store

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self._frame.iloc[index]
        image = self._transform.from_deterministic_base(
            self._transform.deterministic_base(self._cxr.canonical_image(int(row["source_row"])))
        )
        output: dict[str, object] = {
            "image": image,
            "target": torch.tensor(float(row["target"]), dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
            "patient_id": str(row["subject_id"]),
        }
        if self._labs is not None:
            output["structured"] = torch.from_numpy(self._labs[index].copy())
        if self._ecg is not None:
            output["ecg"] = torch.from_numpy(self._ecg.signal(int(row["source_row"])))
        return output


def test_laboratory_frame(data: FrozenSymileTestData) -> pd.DataFrame:
    """Return only the fixed raw-lab columns required by fitted final preprocessors."""
    return data.frame.loc[:, LAB_FEATURE_COLUMNS].copy()


def validate_prediction_against_test_projection(
    evidence: ValidatedPredictionEvidence,
    *,
    capability: ValidatedPretestFreeze,
    projection: HeldOutEvaluationProjection,
    expected_package_ids: Sequence[str],
) -> ValidatedPredictionEvidence:
    """Validate one immutable prediction object against materialized official-test rows."""
    if not isinstance(evidence, ValidatedPredictionEvidence) or not isinstance(
        projection, HeldOutEvaluationProjection
    ):
        raise ManifestBuildError("Official-test prediction projection is invalid")
    _validate_held_out_projection(projection.frame())
    freeze_manifest = validated_pretest_freeze_manifest(capability)
    frozen_bundle = freeze_manifest.get("bundle")
    frozen_task = freeze_manifest.get("task")
    expected_packages = set(expected_package_ids)
    if (
        not isinstance(frozen_bundle, Mapping)
        or not isinstance(frozen_task, Mapping)
        or not expected_packages
        or len(expected_packages) != len(tuple(expected_package_ids))
        or projection.freeze_id != capability.freeze_id
        or evidence.manifest.get("dataset_id") != projection.dataset_id
        or evidence.manifest.get("bundle_id") != projection.bundle_id
        or evidence.manifest.get("bundle_id") != frozen_bundle.get("bundle_id")
        or evidence.manifest.get("split_assignment_id") != projection.split_assignment_id
        or evidence.manifest.get("split_assignment_id") != frozen_bundle.get("split_assignment_id")
        or evidence.manifest.get("task_id") != projection.task_id
        or evidence.manifest.get("task_id") != frozen_task.get("task_id")
        or evidence.manifest.get("label_policy_version") != projection.label_policy_version
        or evidence.manifest.get("label_policy_version") != frozen_task.get("label_policy_version")
        or evidence.manifest.get("scope") != projection.scope
        or evidence.manifest.get("inference_policy") != projection.inference_policy
        or evidence.manifest.get("authorized_by_pretest_freeze_id") != capability.freeze_id
        or evidence.manifest.get("model_package_id") not in expected_packages
    ):
        raise ManifestBuildError("Official-test prediction lineage differs from test projection")
    observed = evidence.predictions.select(["sample_id", "target"]).to_pandas()
    expected = projection.prediction_targets()
    if tuple(observed["sample_id"].astype(str)) != tuple(
        expected["sample_id"].astype(str)
    ) or not np.array_equal(
        observed["target"].to_numpy(dtype=np.int8),
        expected["target"].to_numpy(dtype=np.int8),
    ):
        raise ManifestBuildError("Official-test prediction targets differ from test projection")
    return evidence


def _validate_held_out_projection(frame: pd.DataFrame) -> None:
    if (
        not isinstance(frame, pd.DataFrame)
        or tuple(frame.columns) != _HELD_OUT_REFERENCE_COLUMNS
        or len(frame) != STRICT_PNEUMONIA_TEST_ADMISSIONS
        or frame.isna().any().any()
        or frame["sample_id"].duplicated().any()
        or tuple(frame["sample_id"].astype(str)) != tuple(sorted(frame["sample_id"].astype(str)))
        or set(frame["target"].tolist()) != {0, 1}
        or not pd.api.types.is_integer_dtype(frame["target"].dtype)
        or not pd.api.types.is_integer_dtype(frame["subject_id"].dtype)
        or bool((frame["subject_id"] <= 0).any())
    ):
        raise ManifestBuildError("Held-out evaluation projection is invalid")
