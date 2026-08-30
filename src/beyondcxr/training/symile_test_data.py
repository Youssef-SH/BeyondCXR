"""Official Symile test views gated by execution, freeze, and opening receipts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.hashing import sha256_file
from beyondcxr.data.symile_artifacts import (
    SymileBundlePaths,
    _validate_tables,
    resolve_symile_bundle,
    validate_symile_bundle_reference,
)
from beyondcxr.data.symile_preprocess import LAB_FEATURE_COLUMNS
from beyondcxr.data.symile_schemas import (
    LAB_ITEM_IDS,
    LAB_SCHEMA,
    LABEL_POLICY_VERSION,
    SAMPLE_SCHEMA,
    TASK_ID,
)
from beyondcxr.data.symile_source import (
    CHECKSUM_FILENAME,
    FULL_SOURCE_FILENAME,
    _canonical_frames,
    _load_lab_contract,
    _qualify_modalities,
    _validate_membership,
    _validate_retrieval_frame,
    _validate_source_index,
    _validate_temporal,
    _validate_val_retrieval_headers,
    authenticate_release,
    parse_sha256sums,
)
from beyondcxr.training.symile_campaign_control import (
    ValidatedHeldOutCapability,
    ValidatedPretestFreeze,
    ValidatedTestOpenEvidence,
    materialize_official_test,
    require_validated_held_out_capability,
    require_validated_test_open_evidence,
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
    """Defensive value projection of one canonical materialized test cohort."""

    dataset_id: str
    bundle_id: str
    task_id: str
    label_policy_version: str
    scope: str
    inference_policy: Mapping[str, object]
    freeze_id: str
    _frame: pd.DataFrame
    _test_data: FrozenSymileTestData

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
    """A test-only bundle view requiring the complete held-out capability."""

    def __init__(
        self,
        capability: ValidatedHeldOutCapability,
        config: FinalPackageConfig,
        *,
        manifest_root: str | Path,
        source_root: str | Path,
    ) -> None:
        def initialize() -> None:
            _initialize_frozen_test_data(
                self, capability, capability.freeze, config, manifest_root=manifest_root
            )
            _qualify_held_out_source(capability, self, source_root)

        materialize_official_test(capability, initialize)

    @property
    def bundle(self) -> SymileBundlePaths:
        """Return the exact bundle selected by the frozen campaign."""
        return self._bundle

    @property
    def frame(self) -> pd.DataFrame:
        """Return a defensive copy of the canonical official-test cohort."""
        return self._canonical_frame.copy(deep=True)

    @property
    def source_samples(self) -> pd.DataFrame:
        """Return the authorized complete classification test sample evidence."""
        return self._source_samples.copy(deep=True)

    @property
    def source_reconciliation(self) -> dict[str, int]:
        """Return full source reconciliation established during authorized reconstruction."""
        return dict(self._source_reconciliation)

    def evaluation_projection(self) -> HeldOutEvaluationProjection:
        """Return validated private rows used by prediction and result validation."""
        projection = (
            self._canonical_frame.loc[:, ["sample_id", "target", "subject_id"]]
            .loc[:, _HELD_OUT_REFERENCE_COLUMNS]
            .sort_values("sample_id", kind="stable")
            .reset_index(drop=True)
        )
        return HeldOutEvaluationProjection(
            dataset_id="symile",
            bundle_id=self.bundle.bundle_id,
            task_id=TASK_ID,
            label_policy_version=LABEL_POLICY_VERSION,
            scope="test",
            inference_policy=dict(SYMILE_TEST_INFERENCE_POLICY),
            freeze_id=self._capability.freeze_id,
            _frame=projection,
            _test_data=self,
        )


class RestoredSymileTestData(FrozenSymileTestData):
    """Historical preserved test evidence that cannot authorize live source access."""

    def __init__(
        self,
        evidence: ValidatedTestOpenEvidence,
        freeze: ValidatedPretestFreeze,
        config: FinalPackageConfig,
        *,
        manifest_root: str | Path,
    ) -> None:
        require_validated_test_open_evidence(evidence, freeze=freeze)
        if (
            Path(manifest_root).resolve() != evidence.restoration.root / "data" / "manifests"
            or config.bundle_id != evidence.bundle_id
        ):
            raise PermissionError("Restored test data is outside its authenticated campaign")
        _initialize_frozen_test_data(
            self,
            evidence,
            freeze,
            config,
            manifest_root=manifest_root,
        )


def _read_test_source(
    authority: ValidatedHeldOutCapability | ValidatedTestOpenEvidence,
    freeze: ValidatedPretestFreeze,
    bundle: SymileBundlePaths,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Reconstruct held-out rows only inside an authenticated opening scope."""
    _require_test_materialization(authority, freeze)
    if bundle.bundle_id != validated_pretest_freeze_manifest(freeze)["bundle_id"]:
        raise PermissionError("Official-test source bundle differs from frozen campaign")
    if isinstance(authority, ValidatedTestOpenEvidence) and bundle.bundle_directory.resolve() != (
        authority.restoration.root / "data/manifests/symile/bundles" / bundle.bundle_id
    ):
        raise PermissionError("Restored source evidence is outside authenticated campaign")
    validate_symile_bundle_reference(bundle.bundle_directory, expected_bundle_id=bundle.bundle_id)
    full = pd.read_csv(bundle.bundle_directory / FULL_SOURCE_FILENAME)
    _validate_source_index(full)
    test = pd.read_csv(bundle.bundle_directory / "test.csv")
    _validate_retrieval_frame(test, "test")
    selected = test.loc[(test["label"] == 1) & (test["label_hadm_id"] == test["hadm_id"])].copy()
    if selected.empty or selected["hadm_id"].duplicated().any():
        raise ManifestBuildError("Official test positive-query selection is invalid")
    development = pq.read_table(bundle.samples_path).to_pandas()
    official = {
        split: development.loc[development["official_split"] == split]
        for split in ("train", "validation")
    }
    official["test"] = selected
    reconciliation = _validate_membership(full, official, True)
    joined = full.set_index("hadm_id").loc[selected["hadm_id"]]
    for column in ("subject_id", "cxr_path", "ecg_path"):
        if not np.array_equal(joined[column].to_numpy(), selected[column].to_numpy()):
            raise ManifestBuildError("Held-out source alignment is invalid")
    _validate_temporal(joined, "test")
    samples, labs = _canonical_frames({"test": selected}, full)
    full_labs = joined[list(LAB_ITEM_IDS)].to_numpy(dtype=float)
    if not np.allclose(
        full_labs,
        selected[list(LAB_ITEM_IDS)].to_numpy(dtype=float),
        equal_nan=True,
        rtol=0,
        atol=0,
    ):
        raise ManifestBuildError("Held-out raw laboratory values differ from full source")
    _validate_tables(
        pa.Table.from_pandas(samples, schema=SAMPLE_SCHEMA, preserve_index=False),
        pa.Table.from_pandas(labs, schema=LAB_SCHEMA, preserve_index=False),
    )
    return samples, labs, reconciliation


def _qualify_held_out_source(
    capability: ValidatedHeldOutCapability,
    data: FrozenSymileTestData,
    source_root: str | Path,
) -> None:
    """Validate all held-out modality and retrieval semantics under live authority."""
    validated = require_validated_held_out_capability(capability)
    if data._capability is not validated.freeze:
        raise PermissionError("Held-out source qualification differs from frozen campaign")
    root = Path(source_root)
    _, checksum = authenticate_release(root)
    reference = validate_symile_bundle_reference(
        data.bundle.bundle_directory, expected_bundle_id=data.bundle.bundle_id
    )
    if checksum != reference.manifest["source"]["checksum_manifest_sha256"]:
        raise ManifestBuildError("Held-out source differs from authenticated release")
    full = pd.read_csv(root / FULL_SOURCE_FILENAME)
    test = pd.read_csv(root / "test.csv")
    validation_retrieval = pd.read_csv(root / "val_retrieval.csv")
    _validate_retrieval_frame(test, "test")
    _validate_retrieval_frame(validation_retrieval, "val_retrieval")
    selected = test.loc[(test["label"] == 1) & (test["label_hadm_id"] == test["hadm_id"])]

    def load_tensor(split: str, field: str) -> np.memmap:
        require_validated_held_out_capability(capability)
        array = np.load(
            root / "data_npy" / split / f"{field}_{split}.npy", mmap_mode="r", allow_pickle=False
        )
        if not isinstance(array, np.memmap):
            raise ManifestBuildError("Held-out source tensor is not a memory map")
        return array

    evidence = _qualify_modalities(
        {"test": test},
        {"test": selected},
        full,
        _load_lab_contract(root),
        load_npy=load_tensor,
        exact_contract=True,
    )
    for modality in ("cxr", "ecg"):
        if (
            evidence["splits"]["test"][f"{modality}_shape"]
            != reference.manifest["modalities"][modality]["sample_shape"]
        ):
            raise ManifestBuildError("Held-out modality shape differs from development")
    _validate_val_retrieval_headers(validation_retrieval, load_npy=load_tensor, exact_contract=True)


def _strict_test_pneumonia_rows(
    authority: ValidatedHeldOutCapability | ValidatedTestOpenEvidence,
    freeze: ValidatedPretestFreeze,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    _require_test_materialization(authority, freeze)
    if not {"pneumonia_state", "sample_id", "official_split"} <= set(frame) or set(
        frame["official_split"]
    ) != {"test"}:
        raise ManifestBuildError("Symile held-out sample rows are invalid")
    eligible = frame.loc[frame["pneumonia_state"].isin([0, 1])].copy()
    eligible["target"] = eligible["pneumonia_state"].astype("int8")
    return eligible.sort_values("sample_id", kind="stable").reset_index(drop=True)


def _authenticate_test_source_asset(
    authority: ValidatedHeldOutCapability,
    freeze: ValidatedPretestFreeze,
    bundle: SymileBundlePaths,
    source_root: str | Path,
    *,
    modality: str,
) -> Path:
    validated = require_validated_held_out_capability(authority)
    if validated.freeze != freeze:
        raise PermissionError("Official-test source access freeze authority is invalid")
    frozen_bundle_id = validated_pretest_freeze_manifest(validated.freeze).get("bundle_id")
    if bundle.bundle_id != frozen_bundle_id:
        raise PermissionError("Official-test source bundle differs from the frozen campaign")
    reference = validate_symile_bundle_reference(
        bundle.bundle_directory, expected_bundle_id=bundle.bundle_id
    )
    if modality not in {"cxr", "ecg"}:
        raise ManifestBuildError("Official-test modality is invalid")
    root = Path(source_root)
    checksum_path = root / CHECKSUM_FILENAME
    expected_checksum = reference.manifest["source"]["checksum_manifest_sha256"]
    if (
        root.is_symlink()
        or not root.is_dir()
        or checksum_path.is_symlink()
        or not checksum_path.is_file()
        or sha256_file(checksum_path) != expected_checksum
    ):
        raise ManifestBuildError("Official-test checksum authority is invalid")
    try:
        expected = parse_sha256sums(checksum_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestBuildError("Official-test checksum authority is unreadable") from exc
    relative = f"data_npy/test/{modality}_test.npy"
    path = root / "data_npy" / "test" / f"{modality}_test.npy"
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected.get(relative):
        raise ManifestBuildError("Official-test source asset does not match SHA256SUMS")
    return path


def _initialize_frozen_test_data(
    value: FrozenSymileTestData,
    authority: ValidatedHeldOutCapability | ValidatedTestOpenEvidence,
    freeze: ValidatedPretestFreeze,
    config: FinalPackageConfig,
    *,
    manifest_root: str | Path,
) -> None:
    _require_test_materialization(authority, freeze)
    freeze_manifest = validated_pretest_freeze_manifest(freeze)
    frozen_task = freeze_manifest.get("task")
    if (
        not isinstance(frozen_task, Mapping)
        or config.dataset_id != "symile"
        or config.bundle_id != freeze_manifest.get("bundle_id")
        or config.task_id != frozen_task.get("task_id")
        or config.label_policy_version != frozen_task.get("label_policy_version")
    ):
        raise ManifestBuildError("Frozen test configuration differs from pre-test authority")
    bundle = resolve_symile_bundle(
        manifest_root,
        bundle_id=config.bundle_id,
        full_validation=True,
    )
    all_samples, all_labs, reconciliation = _read_test_source(authority, freeze, bundle)
    samples = _strict_test_pneumonia_rows(authority, freeze, all_samples)
    frame = (
        samples.merge(all_labs, on="sample_id", validate="one_to_one")
        .sort_values("sample_id", kind="stable")
        .reset_index(drop=True)
    )
    value._source_samples = all_samples.copy(deep=True)
    value._source_reconciliation = dict(reconciliation)
    if len(frame) != STRICT_PNEUMONIA_TEST_ADMISSIONS or set(frame["target"].tolist()) != {0, 1}:
        raise ManifestBuildError("Frozen strict-pneumonia test cohort is invalid")
    value._bundle = bundle
    value._canonical_frame = frame.copy(deep=True)
    value._capability = freeze


def _require_test_materialization(
    authority: ValidatedHeldOutCapability | ValidatedTestOpenEvidence,
    freeze: ValidatedPretestFreeze,
) -> ValidatedPretestFreeze:
    if isinstance(authority, ValidatedHeldOutCapability):
        validated = require_validated_held_out_capability(authority)
        if validated.freeze != freeze:
            raise PermissionError("Official-test materialization freeze authority is invalid")
        return validated.freeze
    if isinstance(authority, ValidatedTestOpenEvidence):
        require_validated_test_open_evidence(authority, freeze=freeze)
        return freeze
    raise PermissionError("Official-test materialization requires validated opening authority")


class FrozenSymileTestCxrStore:
    """Authenticate and expose only official test CXR tensors after authorization."""

    def __init__(
        self,
        capability: ValidatedHeldOutCapability,
        bundle: SymileBundlePaths,
        source_root: str | Path,
    ) -> None:
        path = materialize_official_test(
            capability,
            lambda: _authenticate_test_source_asset(
                capability,
                capability.freeze,
                bundle,
                source_root,
                modality="cxr",
            ),
        )
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            not isinstance(array, np.memmap)
            or array.dtype != np.float32
            or array.ndim != 4
            or array.shape[1:] != (3, 320, 320)
        ):
            raise ManifestBuildError("Frozen test CXR tensor header is invalid")
        self._array = array
        self._freeze = capability.freeze
        self._bundle_id = bundle.bundle_id

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
        capability: ValidatedHeldOutCapability,
        bundle: SymileBundlePaths,
        source_root: str | Path,
    ) -> None:
        path = materialize_official_test(
            capability,
            lambda: _authenticate_test_source_asset(
                capability,
                capability.freeze,
                bundle,
                source_root,
                modality="ecg",
            ),
        )
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            not isinstance(array, np.memmap)
            or array.dtype != np.float32
            or array.ndim != 4
            or array.shape[1:] != (1, 5000, 12)
        ):
            raise ManifestBuildError("Frozen test ECG tensor header is invalid")
        self._array = array
        self._freeze = capability.freeze
        self._bundle_id = bundle.bundle_id

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
        capability: ValidatedHeldOutCapability,
        data: FrozenSymileTestData,
        *,
        cxr_store: FrozenSymileTestCxrStore,
        transform: StandardCxrTransform,
        labs: np.ndarray | None = None,
        ecg_store: FrozenSymileTestEcgStore | None = None,
    ) -> None:
        materialize_official_test(capability, lambda: None)
        projection = canonical_held_out_evaluation_projection(data, freeze=capability.freeze)
        if (
            not isinstance(cxr_store, FrozenSymileTestCxrStore)
            or cxr_store._freeze is not capability.freeze
            or cxr_store._bundle_id != projection.bundle_id
            or (
                ecg_store is not None
                and (
                    not isinstance(ecg_store, FrozenSymileTestEcgStore)
                    or ecg_store._freeze is not capability.freeze
                    or ecg_store._bundle_id != projection.bundle_id
                )
            )
        ):
            raise PermissionError("Frozen neural test inputs belong to another campaign")
        frame = data.frame
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
    if not isinstance(evidence, ValidatedPredictionEvidence):
        raise ManifestBuildError("Official-test prediction projection is invalid")
    projection = require_held_out_evaluation_projection(projection, freeze=capability)
    _validate_held_out_projection(projection.frame())
    freeze_manifest = validated_pretest_freeze_manifest(capability)
    frozen_task = freeze_manifest.get("task")
    expected_packages = set(expected_package_ids)
    if (
        not isinstance(frozen_task, Mapping)
        or not expected_packages
        or len(expected_packages) != len(tuple(expected_package_ids))
        or projection.freeze_id != capability.freeze_id
        or evidence.manifest.get("dataset_id") != projection.dataset_id
        or evidence.manifest.get("bundle_id") != projection.bundle_id
        or evidence.manifest.get("bundle_id") != freeze_manifest.get("bundle_id")
        or evidence.manifest.get("task_id") != projection.task_id
        or evidence.manifest.get("task_id") != frozen_task.get("task_id")
        or evidence.manifest.get("label_policy_version") != projection.label_policy_version
        or evidence.manifest.get("label_policy_version") != frozen_task.get("label_policy_version")
        or evidence.manifest.get("scope") != projection.scope
        or evidence.manifest.get("inference_policy") != projection.inference_policy
        or evidence.manifest.get("authorized_by_freeze_id") != capability.freeze_id
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


def canonical_held_out_evaluation_projection(
    test_data: object,
    *,
    freeze: ValidatedPretestFreeze,
) -> HeldOutEvaluationProjection:
    """Derive the cohort projection from test data owned by the exact freeze."""
    manifest = validated_pretest_freeze_manifest(freeze)
    if (
        not isinstance(test_data, FrozenSymileTestData)
        or getattr(test_data, "_capability", None) is not freeze
        or test_data.bundle.bundle_id != manifest.get("bundle_id")
    ):
        raise PermissionError("Held-out test data authority is invalid")
    projection = test_data.evaluation_projection()
    _validate_held_out_projection(projection.frame())
    return projection


def require_held_out_evaluation_projection(
    projection: object,
    *,
    freeze: ValidatedPretestFreeze,
) -> HeldOutEvaluationProjection:
    """Replace a supplied projection with its canonically rederived equivalent."""
    if not isinstance(projection, HeldOutEvaluationProjection):
        raise PermissionError("Held-out evaluation projection authority is invalid")
    canonical = canonical_held_out_evaluation_projection(projection._test_data, freeze=freeze)
    if (
        projection.dataset_id != canonical.dataset_id
        or projection.bundle_id != canonical.bundle_id
        or projection.task_id != canonical.task_id
        or projection.label_policy_version != canonical.label_policy_version
        or projection.scope != canonical.scope
        or projection.inference_policy != canonical.inference_policy
        or projection.freeze_id != canonical.freeze_id
        or not projection.frame().equals(canonical.frame())
    ):
        raise PermissionError("Held-out evaluation projection differs from canonical test data")
    return canonical


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
