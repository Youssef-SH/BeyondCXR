"""Authenticate and qualify the official Symile-MIMIC 1.0.0 source release."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from radfusion.data.errors import ManifestBuildError
from radfusion.data.hashing import sha256_file
from radfusion.data.symile_schemas import LAB_ITEM_IDS, LAB_NAMES, OFFICIAL_SPLITS
from radfusion.utils.operational_logging import CountProgress, get_operational_logger

CHECKSUM_FILENAME = "SHA256SUMS.txt"
FULL_SOURCE_FILENAME = "symile_mimic_data.csv"
EXPECTED_RELEASE_ASSETS = frozenset(
    {
        "LICENSE.txt",
        "code/README.md",
        "code/args.py",
        "code/constants.py",
        "code/create_dataset_splits.py",
        "code/environment.yml",
        "code/process_and_save_tensors.py",
        "code/process_mimic_data.py",
        "code/requirements.txt",
        "data_npy/test/cxr_test.npy",
        "data_npy/test/ecg_test.npy",
        "data_npy/test/hadm_id_test.npy",
        "data_npy/test/label_hadm_id_test.npy",
        "data_npy/test/label_test.npy",
        "data_npy/test/labs_missingness_test.npy",
        "data_npy/test/labs_percentiles_test.npy",
        "data_npy/train/cxr_train.npy",
        "data_npy/train/ecg_train.npy",
        "data_npy/train/hadm_id_train.npy",
        "data_npy/train/labs_missingness_train.npy",
        "data_npy/train/labs_percentiles_train.npy",
        "data_npy/val/cxr_val.npy",
        "data_npy/val/ecg_val.npy",
        "data_npy/val/hadm_id_val.npy",
        "data_npy/val/labs_missingness_val.npy",
        "data_npy/val/labs_percentiles_val.npy",
        "data_npy/val_retrieval/cxr_val_retrieval.npy",
        "data_npy/val_retrieval/ecg_val_retrieval.npy",
        "data_npy/val_retrieval/hadm_id_val_retrieval.npy",
        "data_npy/val_retrieval/label_hadm_id_val_retrieval.npy",
        "data_npy/val_retrieval/label_val_retrieval.npy",
        "data_npy/val_retrieval/labs_missingness_val_retrieval.npy",
        "data_npy/val_retrieval/labs_percentiles_val_retrieval.npy",
        "labs_means.json",
        "symile_mimic_data.csv",
        "symile_mimic_model.ckpt",
        "test.csv",
        "train.csv",
        "val.csv",
        "val_retrieval.csv",
    }
)
_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64}) ([^\r\n]+)$")
_IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
_IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)
_SPLIT_SOURCE_NAMES = {"train": "train", "validation": "val", "test": "test"}
_LOGGER = get_operational_logger(__name__)
_AUTHENTICATED_RELEASE_ASSETS: dict[tuple[str, str, str], AuthenticatedReleaseAsset] = {}


@dataclass(frozen=True)
class SourceAsset:
    """Authenticated identity of one release-relative source asset."""

    relative_path: str
    byte_size: int
    expected_sha256: str
    observed_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "byte_size": self.byte_size,
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.observed_sha256,
        }


@dataclass(frozen=True)
class AuthenticatedReleaseAsset:
    """Process-transferable capability for one byte-authenticated physical asset."""

    source_root: Path
    checksum_manifest_sha256: str
    relative_path: str
    expected_sha256: str
    physical_identity: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class QualifiedSymileSource:
    """Qualified classification spine plus aggregate source evidence."""

    samples: pd.DataFrame
    labs: pd.DataFrame
    source_assets: tuple[SourceAsset, ...]
    checksum_manifest_sha256: str
    evidence: Mapping[str, object]


def parse_sha256sums(text: str) -> dict[str, str]:
    """Strictly parse the official one-space SHA256SUMS format."""
    if not isinstance(text, str) or not text.endswith("\n"):
        raise ManifestBuildError("SHA256SUMS must be newline-terminated UTF-8 text")
    result: dict[str, str] = {}
    for line in text.splitlines():
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ManifestBuildError("SHA256SUMS contains a malformed line")
        digest, raw_path = match.groups()
        path = PurePosixPath(raw_path)
        if (
            path.is_absolute()
            or path.as_posix() != raw_path
            or any(part in {"", ".", ".."} for part in raw_path.split("/"))
            or raw_path in result
        ):
            raise ManifestBuildError("SHA256SUMS contains an invalid or duplicate path")
        result[raw_path] = digest
    if set(result) != EXPECTED_RELEASE_ASSETS:
        raise ManifestBuildError("SHA256SUMS asset set does not match Symile-MIMIC 1.0.0")
    return result


def authenticate_release(source_root: str | Path) -> tuple[tuple[SourceAsset, ...], str]:
    """Authenticate every official source asset exactly once."""
    root = Path(source_root)
    checksum_path = root / CHECKSUM_FILENAME
    if (
        root.is_symlink()
        or not root.is_dir()
        or checksum_path.is_symlink()
        or not checksum_path.is_file()
    ):
        raise ManifestBuildError("Symile source-release root is incomplete")
    expected = _authenticated_checksum_map(root, sha256_file(checksum_path))
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if actual_files != EXPECTED_RELEASE_ASSETS:
        raise ManifestBuildError("Symile source-release file set is incomplete or unexpected")
    progress = CountProgress(
        _LOGGER,
        "source_authentication_progress",
        total=len(expected),
        unit="files",
        count_interval=1,
    )
    assets: list[SourceAsset] = []
    for completed, relative_path in enumerate(sorted(expected), start=1):
        path = root / relative_path
        if path.is_symlink() or not path.is_file():
            raise ManifestBuildError(f"Source asset is not a regular file: {relative_path}")
        observed = sha256_file(path)
        if observed != expected[relative_path]:
            raise ManifestBuildError(f"Source asset SHA-256 mismatch: {relative_path}")
        assets.append(
            SourceAsset(relative_path, path.stat().st_size, expected[relative_path], observed)
        )
        progress.update(completed)
    return tuple(assets), sha256_file(checksum_path)


def authenticate_release_asset(
    source_root: str | Path,
    *,
    checksum_manifest_sha256: str,
    relative_path: str,
) -> Path:
    """Authenticate one physical release asset through the bundle-bound checksum manifest."""
    root = Path(source_root)
    expected = _authenticated_checksum_map(root, checksum_manifest_sha256)
    if relative_path not in expected:
        raise ManifestBuildError("Requested source asset is absent from SHA256SUMS")
    path = root / Path(*PurePosixPath(relative_path).parts)
    if path.is_symlink() or not path.is_file():
        raise ManifestBuildError("Requested source asset is not a physical regular file")
    if sha256_file(path) != expected[relative_path]:
        raise ManifestBuildError("Requested source asset does not match SHA256SUMS")
    return path


def establish_authenticated_release_asset(
    source_root: str | Path,
    *,
    checksum_manifest_sha256: str,
    relative_path: str,
) -> AuthenticatedReleaseAsset:
    """Authenticate expensive asset bytes once and return their physical authority."""
    root = Path(source_root).absolute()
    expected = _authenticated_checksum_map(root, checksum_manifest_sha256)
    if relative_path not in expected:
        raise ManifestBuildError("Requested source asset is absent from SHA256SUMS")
    path = root / Path(*PurePosixPath(relative_path).parts)
    key = (root.as_posix(), checksum_manifest_sha256, relative_path)
    cached = _AUTHENTICATED_RELEASE_ASSETS.get(key)
    if cached is not None:
        descriptor = _open_authenticated_asset(cached)
        os.close(descriptor)
        return cached
    descriptor = _open_physical_asset(path)
    try:
        before = _physical_identity(os.fstat(descriptor))
        observed = _sha256_descriptor(descriptor)
        after = _physical_identity(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    if before != after or observed != expected[relative_path]:
        raise ManifestBuildError("Requested source asset does not match SHA256SUMS")
    authority = AuthenticatedReleaseAsset(
        source_root=root,
        checksum_manifest_sha256=checksum_manifest_sha256,
        relative_path=relative_path,
        expected_sha256=expected[relative_path],
        physical_identity=after,
    )
    _AUTHENTICATED_RELEASE_ASSETS[key] = authority
    return authority


def reopen_authenticated_release_memmap(authority: AuthenticatedReleaseAsset) -> np.memmap:
    """Reopen a previously byte-authenticated asset without hashing it again."""
    descriptor = _open_authenticated_asset(authority)
    try:
        array = np.load(f"/proc/self/fd/{descriptor}", mmap_mode="r", allow_pickle=False)
        if _physical_identity(os.fstat(descriptor)) != authority.physical_identity:
            raise ManifestBuildError("Authenticated source asset changed before worker access")
        if not isinstance(array, np.memmap) or array.flags.writeable:
            raise ManifestBuildError("Authenticated source asset did not open read-only")
        array.filename = str(
            authority.source_root / Path(*PurePosixPath(authority.relative_path).parts)
        )
        return array
    finally:
        os.close(descriptor)


def _open_authenticated_asset(authority: AuthenticatedReleaseAsset) -> int:
    if (
        not isinstance(authority, AuthenticatedReleaseAsset)
        or not _is_sha256(authority.checksum_manifest_sha256)
        or not _is_sha256(authority.expected_sha256)
        or PurePosixPath(authority.relative_path).as_posix() != authority.relative_path
        or len(authority.physical_identity) != 5
    ):
        raise ManifestBuildError("Authenticated source authority is invalid")
    path = authority.source_root / Path(*PurePosixPath(authority.relative_path).parts)
    descriptor = _open_physical_asset(path)
    if _physical_identity(os.fstat(descriptor)) != authority.physical_identity:
        os.close(descriptor)
        raise ManifestBuildError("Authenticated source asset changed before worker access")
    return descriptor


def _open_physical_asset(path: Path) -> int:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise ManifestBuildError("Requested source asset is not a physical regular file") from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ManifestBuildError("Requested source asset is not a physical regular file")
    return descriptor


def _physical_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _authenticated_checksum_map(root: Path, expected_sha256: str) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir() or not _is_sha256(expected_sha256):
        raise ManifestBuildError("Symile source authentication inputs are invalid")
    checksum_path = root / CHECKSUM_FILENAME
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise ManifestBuildError("SHA256SUMS must be a physical regular file")
    try:
        checksum_bytes = checksum_path.read_bytes()
        checksum_text = checksum_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ManifestBuildError("SHA256SUMS is unreadable") from exc
    if hashlib.sha256(checksum_bytes).hexdigest() != expected_sha256:
        raise ManifestBuildError("SHA256SUMS does not match the bundle-bound identity")
    return parse_sha256sums(checksum_text)


def select_test_queries(frame: pd.DataFrame) -> pd.DataFrame:
    """Select only positive self-query rows while retaining original source-row indices."""
    _require_columns(frame, {"hadm_id", "label_hadm_id", "label"}, "test.csv")
    selected = frame.loc[(frame["label"] == 1) & (frame["label_hadm_id"] == frame["hadm_id"])]
    if selected.empty or selected["hadm_id"].duplicated().any():
        raise ManifestBuildError("Official test positive-query selection is invalid")
    return selected.copy()


def strict_pneumonia_target(value: object) -> int | None:
    """Derive the strict binary target from one preserved source state."""
    if value is None or pd.isna(value) or float(value) == -1.0:
        return None
    if float(value) in {0.0, 1.0}:
        return int(float(value))
    raise ManifestBuildError("Pneumonia source state is outside {-1, 0, 1, null}")


def qualify_symile_source(
    source_root: str | Path,
    *,
    enforce_production_counts: bool = True,
) -> QualifiedSymileSource:
    """Authenticate the release and strongly qualify the official modeling spine."""
    root = Path(source_root)
    assets, checksum_sha256 = authenticate_release(root)
    lab_means = _load_lab_contract(root)
    full = _load_full_source(root)
    split_frames = _load_split_frames(root)
    selected_test = select_test_queries(split_frames["test"])
    official_frames = {
        "train": split_frames["train"],
        "validation": split_frames["val"],
        "test": selected_test,
    }
    reconciliation = _validate_membership(full, official_frames, enforce_production_counts)
    modality_evidence = _qualify_modalities(
        root,
        split_frames,
        official_frames,
        full,
        lab_means,
        exact_contract=enforce_production_counts,
    )
    samples, labs = _canonical_frames(official_frames, full)
    evidence: dict[str, object] = {
        "reconciliation": reconciliation,
        "modality_qualification": modality_evidence,
    }
    return QualifiedSymileSource(samples, labs, assets, checksum_sha256, evidence)


def modality_asset_path(source_root: str | Path, official_split: str, modality: str) -> Path:
    """Resolve one authenticated split-local modality asset without opening it."""
    if official_split not in OFFICIAL_SPLITS or modality not in {
        "cxr",
        "ecg",
        "labs_percentiles",
        "labs_missingness",
        "hadm_id",
    }:
        raise ManifestBuildError("Symile source-row locator arguments are invalid")
    split = _SPLIT_SOURCE_NAMES[official_split]
    return Path(source_root) / "data_npy" / split / f"{modality}_{split}.npy"


def _load_lab_contract(root: Path) -> dict[str, float]:
    try:
        raw = json.loads((root / "labs_means.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestBuildError("labs_means.json is unreadable") from exc
    expected_keys = [f"{item_id}_percentile" for item_id in LAB_ITEM_IDS]
    if not isinstance(raw, dict) or sorted(raw) != expected_keys:
        raise ManifestBuildError("Official laboratory percentile order does not match")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
        for value in raw.values()
    ):
        raise ManifestBuildError("labs_means.json contains invalid values")
    source = (root / "code/constants.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    labs_node = next(
        (
            node.value
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "LABS" for target in node.targets)
        ),
        None,
    )
    upstream_labs = ast.literal_eval(labs_node) if labs_node is not None else None
    if upstream_labs != LAB_NAMES:
        raise ManifestBuildError(
            "Approved laboratory item/name mapping differs from reference code"
        )
    return {key: float(raw[key]) for key in expected_keys}


def _load_full_source(root: Path) -> pd.DataFrame:
    required = {
        "subject_id",
        "hadm_id",
        "admittime",
        "age",
        "gender",
        "cxr_dicom_id",
        "cxr_study_id",
        "cxr_ViewPosition",
        "cxr_StudyDateTime",
        "cxr_path",
        "Pneumonia",
        "ecg_study_id",
        "ecg_time",
        "ecg_path",
        "labs_all_nan",
        *LAB_ITEM_IDS,
    }
    frame = pd.read_csv(root / FULL_SOURCE_FILENAME, usecols=lambda column: column in required)
    _require_columns(frame, required, FULL_SOURCE_FILENAME)
    if frame.empty or frame["hadm_id"].duplicated().any() or frame["subject_id"].isna().any():
        raise ManifestBuildError("Full Symile source index is invalid")
    if set(frame["cxr_ViewPosition"]) != {"AP", "PA"} or not frame["labs_all_nan"].eq(0).all():
        raise ManifestBuildError("Full Symile modality eligibility is invalid")
    return frame


def _load_split_frames(root: Path) -> dict[str, pd.DataFrame]:
    frames = {
        name: pd.read_csv(root / f"{name}.csv")
        for name in ("train", "val", "test", "val_retrieval")
    }
    required = {
        "subject_id",
        "hadm_id",
        "cxr_path",
        "ecg_path",
        *LAB_ITEM_IDS,
        *(f"{item}_percentile" for item in LAB_ITEM_IDS),
    }
    for name, frame in frames.items():
        _require_columns(frame, required, f"{name}.csv")
        if name in {"test", "val_retrieval"}:
            _require_columns(frame, {"label_hadm_id", "label"}, f"{name}.csv")
    if frames["train"]["hadm_id"].duplicated().any() or frames["val"]["hadm_id"].duplicated().any():
        raise ManifestBuildError("Official train/validation admissions are not unique")
    _validate_retrieval_frame(frames["test"], "test")
    _validate_retrieval_frame(frames["val_retrieval"], "val_retrieval")
    return frames


def _validate_retrieval_frame(frame: pd.DataFrame, name: str) -> None:
    if set(frame["label"].unique()) != {0, 1}:
        raise ManifestBuildError(f"{name} retrieval labels are invalid")
    counts = frame.groupby("label_hadm_id", sort=False).agg(
        rows=("label", "size"), positives=("label", "sum")
    )
    if counts.empty or not counts["rows"].eq(10).all() or not counts["positives"].eq(1).all():
        raise ManifestBuildError(f"{name} retrieval candidate structure is invalid")
    positive = frame.loc[frame["label"] == 1]
    if not positive["hadm_id"].eq(positive["label_hadm_id"]).all():
        raise ManifestBuildError(f"{name} positive retrieval rows are not self-queries")


def _validate_membership(
    full: pd.DataFrame,
    official: Mapping[str, pd.DataFrame],
    enforce: bool,
) -> dict[str, int]:
    full_ids = set(full["hadm_id"].astype(int))
    split_ids = {name: set(frame["hadm_id"].astype(int)) for name, frame in official.items()}
    if any(not ids <= full_ids for ids in split_ids.values()):
        raise ManifestBuildError("Official split admission is missing from full source index")
    if any(
        split_ids[left] & split_ids[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ManifestBuildError("Official split admissions overlap")
    full_by_hadm = full.set_index("hadm_id", verify_integrity=True)
    split_subjects = {
        name: set(full_by_hadm.loc[sorted(ids), "subject_id"].astype(int))
        for name, ids in split_ids.items()
    }
    if any(
        split_subjects[left] & split_subjects[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ManifestBuildError("Official split patients overlap")
    official_ids = set().union(*split_ids.values())
    excluded = full.loc[~full["hadm_id"].isin(official_ids)]
    train_excluded = int(excluded["subject_id"].isin(split_subjects["train"]).sum())
    validation_excluded = int(excluded["subject_id"].isin(split_subjects["validation"]).sum())
    unexplained = len(excluded) - train_excluded - validation_excluded
    result = {
        "full_admissions": len(full),
        "official_admissions": len(official_ids),
        "train_admissions": len(split_ids["train"]),
        "validation_admissions": len(split_ids["validation"]),
        "test_admissions": len(split_ids["test"]),
        "excluded_admissions": len(excluded),
        "train_subject_exclusions": train_excluded,
        "validation_subject_exclusions": validation_excluded,
        "unexplained_exclusions": unexplained,
        "patient_overlap": 0,
        "admission_overlap": 0,
    }
    if unexplained:
        raise ManifestBuildError("Full-source reconciliation contains unexplained exclusions")
    if enforce and result != {
        "full_admissions": 11622,
        "official_admissions": 11214,
        "train_admissions": 10000,
        "validation_admissions": 750,
        "test_admissions": 464,
        "excluded_admissions": 408,
        "train_subject_exclusions": 395,
        "validation_subject_exclusions": 13,
        "unexplained_exclusions": 0,
        "patient_overlap": 0,
        "admission_overlap": 0,
    }:
        raise ManifestBuildError(
            "Official Symile membership differs from the frozen release contract"
        )
    return result


def _qualify_modalities(
    root: Path,
    split_frames: Mapping[str, pd.DataFrame],
    official_frames: Mapping[str, pd.DataFrame],
    full: pd.DataFrame,
    lab_means: Mapping[str, float],
    *,
    exact_contract: bool,
) -> dict[str, object]:
    full_lookup = full.set_index("hadm_id", verify_integrity=True)
    split_evidence: dict[str, object] = {}
    for official_name, source_name in _SPLIT_SOURCE_NAMES.items():
        complete = split_frames[source_name]
        selected = official_frames[official_name]
        selected_rows = selected.index.to_numpy(dtype=np.int64)
        hadm = _load_npy(root, source_name, "hadm_id")
        if (
            hadm.dtype != np.int64
            or hadm.shape != (len(complete),)
            or not np.array_equal(hadm, complete["hadm_id"].to_numpy(dtype=np.int64))
        ):
            raise ManifestBuildError(f"{source_name} identifier tensor is not row-aligned")
        if source_name == "test":
            for field in ("label", "label_hadm_id"):
                array = _load_npy(root, source_name, field)
                if (
                    array.dtype != np.int64
                    or array.shape != (len(complete),)
                    or not np.array_equal(array, complete[field].to_numpy(dtype=np.int64))
                ):
                    raise ManifestBuildError(f"test {field} tensor is not row-aligned")
        joined = full_lookup.loc[selected["hadm_id"].astype(int)]
        if not np.array_equal(
            joined["subject_id"].to_numpy(dtype=np.int64),
            selected["subject_id"].to_numpy(dtype=np.int64),
        ):
            raise ManifestBuildError(f"{official_name} subject alignment differs from full source")
        for path_column in ("cxr_path", "ecg_path"):
            if not np.array_equal(
                joined[path_column].astype(str).to_numpy(),
                selected[path_column].astype(str).to_numpy(),
            ):
                raise ManifestBuildError(
                    f"{official_name} modality provenance differs from full source"
                )
        _validate_temporal(joined, official_name)
        cxr = _load_npy(root, source_name, "cxr")
        ecg = _load_npy(root, source_name, "ecg")
        _validate_cxr(cxr, len(complete), selected_rows, exact_contract=exact_contract)
        _validate_ecg(ecg, len(complete), selected_rows, exact_contract=exact_contract)
        _validate_labs(root, source_name, complete, selected_rows, lab_means, full_lookup)
        split_evidence[official_name] = {
            "source_rows": len(complete),
            "qualified_rows": len(selected),
            "row_alignment": "passed",
            "cxr": "passed",
            "cxr_shape": list(cxr.shape[1:]),
            "ecg": "passed",
            "ecg_shape": list(ecg.shape[1:]),
            "labs": "passed",
            "temporal": "passed",
        }
    _validate_val_retrieval_headers(
        root,
        split_frames["val_retrieval"],
        exact_contract=exact_contract,
    )
    if (
        len({tuple(value["cxr_shape"]) for value in split_evidence.values()}) != 1
        or len({tuple(value["ecg_shape"]) for value in split_evidence.values()}) != 1
    ):
        raise ManifestBuildError("Classification modality shapes differ across splits")
    return {"splits": split_evidence, "val_retrieval_structure": "passed"}


def _load_npy(root: Path, split: str, field: str) -> np.memmap:
    try:
        array = np.load(
            root / "data_npy" / split / f"{field}_{split}.npy", mmap_mode="r", allow_pickle=False
        )
    except (OSError, ValueError) as exc:
        raise ManifestBuildError(f"Unreadable Symile tensor: {field}_{split}.npy") from exc
    if not isinstance(array, np.memmap):
        raise ManifestBuildError("Symile tensor did not open as a memory map")
    return array


def _validate_cxr(
    array: np.memmap,
    source_count: int,
    selected_rows: np.ndarray,
    *,
    exact_contract: bool,
) -> None:
    valid_shape = (
        array.ndim == 4
        and array.shape[0] == source_count
        and array.shape[1] == 3
        and array.shape[2] == array.shape[3]
        and array.shape[2] > 0
    )
    if (
        array.dtype != np.float32
        or not valid_shape
        or (exact_contract and array.shape != (source_count, 3, 320, 320))
    ):
        raise ManifestBuildError("CXR tensor header contract does not match")
    for rows in _row_chunks(selected_rows, 32):
        values = np.asarray(array[rows], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ManifestBuildError("CXR tensor contains non-finite values")
        restored = values * _IMAGENET_STD + _IMAGENET_MEAN
        if restored.min() < -2e-6 or restored.max() > 1.000002:
            raise ManifestBuildError("CXR inverse-normalized values are outside [0, 1]")
        if not (
            np.allclose(restored[:, 0], restored[:, 1], atol=2e-6, rtol=0.0)
            and np.allclose(restored[:, 0], restored[:, 2], atol=2e-6, rtol=0.0)
        ):
            raise ManifestBuildError("CXR tensor channels do not reconstruct repeated grayscale")


def _validate_ecg(
    array: np.memmap,
    source_count: int,
    selected_rows: np.ndarray,
    *,
    exact_contract: bool,
) -> None:
    valid_shape = (
        array.ndim == 4
        and array.shape[0] == source_count
        and array.shape[1] == 1
        and array.shape[2] > 0
        and array.shape[3] == 12
    )
    if (
        array.dtype != np.float32
        or not valid_shape
        or (exact_contract and array.shape != (source_count, 1, 5000, 12))
    ):
        raise ManifestBuildError("ECG tensor header contract does not match")
    for rows in _row_chunks(selected_rows, 64):
        values = np.asarray(array[rows], dtype=np.float32)
        if not np.isfinite(values).all() or values.min() < -1.000001 or values.max() > 1.000001:
            raise ManifestBuildError("ECG tensor values are invalid")
        if np.any(np.all(values == 0.0, axis=(1, 2, 3))):
            raise ManifestBuildError("ECG tensor contains an all-zero signal")


def _validate_labs(
    root: Path,
    split: str,
    frame: pd.DataFrame,
    rows: np.ndarray,
    lab_means: Mapping[str, float],
    full_lookup: pd.DataFrame,
) -> None:
    percentiles = _load_npy(root, split, "labs_percentiles")
    missingness = _load_npy(root, split, "labs_missingness")
    if (
        percentiles.dtype != np.float32
        or percentiles.shape != (len(frame), 50)
        or missingness.dtype != np.int64
        or missingness.shape != (len(frame), 50)
    ):
        raise ManifestBuildError("Laboratory tensor header contract does not match")
    selected = frame.iloc[rows]
    raw = selected[list(LAB_ITEM_IDS)].to_numpy(dtype=np.float64)
    full_raw = full_lookup.loc[selected["hadm_id"].astype(int), list(LAB_ITEM_IDS)].to_numpy(
        dtype=np.float64
    )
    if not np.allclose(raw, full_raw, equal_nan=True, atol=0.0, rtol=0.0):
        raise ManifestBuildError("Split raw laboratory values differ from full source")
    expected_mask = (~np.isnan(raw)).astype(np.int64)
    observed_mask = np.asarray(missingness[rows], dtype=np.int64)
    if not np.array_equal(observed_mask, expected_mask):
        raise ManifestBuildError("Official laboratory missingness differs from raw nullity")
    observed_percentiles = np.asarray(percentiles[rows], dtype=np.float32)
    csv_percentiles = selected[[f"{item}_percentile" for item in LAB_ITEM_IDS]].to_numpy(
        dtype=np.float64
    )
    expected = np.where(
        expected_mask == 1,
        csv_percentiles,
        np.array([lab_means[f"{item}_percentile"] for item in LAB_ITEM_IDS]),
    )
    if not np.isfinite(observed_percentiles).all() or not np.allclose(
        observed_percentiles, expected, atol=1e-6, rtol=1e-6
    ):
        raise ManifestBuildError("Official laboratory percentile/imputation conformance failed")


def _validate_temporal(rows: pd.DataFrame, split: str) -> None:
    admission = pd.to_datetime(rows["admittime"], errors="raise")
    cxr_hours = (
        pd.to_datetime(rows["cxr_StudyDateTime"], errors="raise") - admission
    ).dt.total_seconds() / 3600.0
    ecg_hours = (
        pd.to_datetime(rows["ecg_time"], errors="raise") - admission
    ).dt.total_seconds() / 3600.0
    if not ((cxr_hours > 24.0) & (cxr_hours <= 72.0)).all():
        raise ManifestBuildError(f"{split} CXR temporal qualification failed")
    if not ((ecg_hours >= -24.0) & (ecg_hours <= 24.0)).all():
        raise ManifestBuildError(f"{split} ECG temporal qualification failed")


def _validate_val_retrieval_headers(
    root: Path,
    frame: pd.DataFrame,
    *,
    exact_contract: bool,
) -> None:
    count = len(frame)
    expected = {
        "hadm_id": (np.dtype("int64"), (count,)),
        "label_hadm_id": (np.dtype("int64"), (count,)),
        "label": (np.dtype("int64"), (count,)),
        "labs_missingness": (np.dtype("int64"), (count, 50)),
        "labs_percentiles": (np.dtype("float32"), (count, 50)),
    }
    cxr = _load_npy(root, "val_retrieval", "cxr")
    ecg = _load_npy(root, "val_retrieval", "ecg")
    _validate_cxr(cxr, count, np.array([], dtype=np.int64), exact_contract=exact_contract)
    _validate_ecg(ecg, count, np.array([], dtype=np.int64), exact_contract=exact_contract)
    for field, (dtype, shape) in expected.items():
        array = _load_npy(root, "val_retrieval", field)
        if array.dtype != dtype or array.shape != shape:
            raise ManifestBuildError("Validation retrieval tensor header contract does not match")
    for field in ("hadm_id", "label_hadm_id", "label"):
        if not np.array_equal(
            _load_npy(root, "val_retrieval", field), frame[field].to_numpy(dtype=np.int64)
        ):
            raise ManifestBuildError("Validation retrieval identifier structure is not aligned")


def _canonical_frames(
    official: Mapping[str, pd.DataFrame], full: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_lookup = full.set_index("hadm_id", verify_integrity=True)
    sample_rows: list[dict[str, object]] = []
    lab_rows: list[dict[str, object]] = []
    for split in OFFICIAL_SPLITS:
        frame = official[split]
        for source_row, row in frame.iterrows():
            hadm_id = int(row["hadm_id"])
            source = full_lookup.loc[hadm_id]
            sample_id = f"symile:{hadm_id}"
            state = source["Pneumonia"]
            pneumonia_state = None if pd.isna(state) else int(state)
            strict_pneumonia_target(pneumonia_state)
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "subject_id": int(source["subject_id"]),
                    "hadm_id": hadm_id,
                    "official_split": split,
                    "source_row": int(source_row),
                    "pneumonia_state": pneumonia_state,
                    "age_years": int(source["age"]),
                    "sex": str(source["gender"]),
                    "view_position": str(source["cxr_ViewPosition"]),
                }
            )
            lab_record: dict[str, object] = {"sample_id": sample_id}
            for item_id in LAB_ITEM_IDS:
                value = row[item_id]
                lab_record[f"lab_{item_id}_value"] = None if pd.isna(value) else float(value)
            for item_id in LAB_ITEM_IDS:
                lab_record[f"lab_{item_id}_observed"] = not pd.isna(row[item_id])
            lab_rows.append(lab_record)
    return (
        pd.DataFrame(sample_rows).sort_values("sample_id", kind="stable").reset_index(drop=True),
        pd.DataFrame(lab_rows).sort_values("sample_id", kind="stable").reset_index(drop=True),
    )


def _row_chunks(rows: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(rows), chunk_size):
        yield rows[start : start + chunk_size]


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ManifestBuildError(f"{name} is missing required columns")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
