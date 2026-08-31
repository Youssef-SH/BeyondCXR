"""Authenticate and qualify the official Symile-MIMIC 1.0.0 source release."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import os
import re
import stat
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.hashing import sha256_file
from beyondcxr.data.symile_schemas import LAB_ITEM_IDS, LAB_NAMES
from beyondcxr.utils.operational_logging import CountProgress, get_operational_logger

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
class DevelopmentReleaseAsset:
    """Process-transferable integrity witness restricted to development tensors."""

    source_root: Path
    checksum_manifest_sha256: str
    relative_path: str
    expected_sha256: str
    physical_identity: tuple[int, int, int, int, int]

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (
            len(path.parts) != 3
            or path.parts[0] != "data_npy"
            or path.parts[1] not in {"train", "val"}
            or not path.name.endswith(f"_{path.parts[1]}.npy")
        ):
            raise ValueError("Development source assets cannot reference held-out material")


@dataclass(frozen=True)
class _QualifiedSymileSource:
    """Qualified classification spine plus aggregate source evidence."""

    samples: pd.DataFrame
    labs: pd.DataFrame
    source_assets: tuple[SourceAsset, ...]
    checksum_manifest_sha256: str
    evidence: Mapping[str, object]
    source_root: Path


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


def authenticate_development_source_asset(
    source_root: str | Path,
    *,
    checksum_manifest_sha256: str,
    official_split: str,
    modality: str,
) -> Path:
    """Authenticate one train/validation tensor through the bundle checksum manifest."""
    relative_path = _development_modality_relative_path(official_split, modality)
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


def establish_development_release_asset(
    source_root: str | Path,
    *,
    checksum_manifest_sha256: str,
    official_split: str,
    modality: str,
) -> DevelopmentReleaseAsset:
    """Authenticate one development tensor once for safe worker reopening."""
    relative_path = _development_modality_relative_path(official_split, modality)
    root = Path(source_root).absolute()
    expected = _authenticated_checksum_map(root, checksum_manifest_sha256)
    if relative_path not in expected:
        raise ManifestBuildError("Requested source asset is absent from SHA256SUMS")
    path = root / Path(*PurePosixPath(relative_path).parts)
    descriptor = _open_physical_asset(path)
    try:
        before = _physical_identity(os.fstat(descriptor))
        observed = _sha256_descriptor(descriptor)
        after = _physical_identity(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    if before != after or observed != expected[relative_path]:
        raise ManifestBuildError("Requested source asset does not match SHA256SUMS")
    return DevelopmentReleaseAsset(
        source_root=root,
        checksum_manifest_sha256=checksum_manifest_sha256,
        relative_path=relative_path,
        expected_sha256=expected[relative_path],
        physical_identity=after,
    )


def establish_development_release_assets(
    source_root: str | Path,
    *,
    checksum_manifest_sha256: str,
) -> tuple[DevelopmentReleaseAsset, ...]:
    """Authenticate the four development tensors for one explicit operation."""
    return tuple(
        establish_development_release_asset(
            source_root,
            checksum_manifest_sha256=checksum_manifest_sha256,
            official_split=split,
            modality=modality,
        )
        for modality in ("cxr", "ecg")
        for split in ("train", "validation")
    )


def select_development_release_asset(
    authorities: tuple[DevelopmentReleaseAsset, ...],
    *,
    source_root: str | Path,
    checksum_manifest_sha256: str,
    official_split: str,
    modality: str,
) -> DevelopmentReleaseAsset:
    """Select one exact witness from an operation-owned development set."""
    relative_path = _development_modality_relative_path(official_split, modality)
    root = Path(source_root).absolute()
    matches = tuple(
        authority
        for authority in authorities
        if isinstance(authority, DevelopmentReleaseAsset)
        and authority.source_root == root
        and authority.checksum_manifest_sha256 == checksum_manifest_sha256
        and authority.relative_path == relative_path
    )
    if len(matches) != 1:
        raise ManifestBuildError("Development source witness set is incomplete or incompatible")
    descriptor = _open_authenticated_asset(matches[0])
    os.close(descriptor)
    return matches[0]


def reopen_development_release_memmap(authority: DevelopmentReleaseAsset) -> np.memmap:
    """Reopen a byte-authenticated development tensor without re-hashing it."""
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


def _open_authenticated_asset(authority: DevelopmentReleaseAsset) -> int:
    if (
        not isinstance(authority, DevelopmentReleaseAsset)
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


def _development_modality_relative_path(official_split: str, modality: str) -> str:
    if official_split not in {"train", "validation"} or modality not in {"cxr", "ecg"}:
        raise PermissionError("Symile source access is limited to development modalities")
    split = _SPLIT_SOURCE_NAMES[official_split]
    return f"data_npy/{split}/{modality}_{split}.npy"


def development_modality_asset_path(
    source_root: str | Path, official_split: str, modality: str
) -> Path:
    """Resolve a development-only modality path without opening its bytes."""
    relative = _development_modality_relative_path(official_split, modality)
    return Path(source_root) / Path(*PurePosixPath(relative).parts)


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


def strict_pneumonia_target(value: object) -> int | None:
    """Derive the strict binary target from one preserved source state."""
    if value is None or pd.isna(value) or float(value) == -1.0:
        return None
    if float(value) in {0.0, 1.0}:
        return int(float(value))
    raise ManifestBuildError("Pneumonia source state is outside {-1, 0, 1, null}")


_SOURCE_COLUMNS = {
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


def _validate_source_index(frame: pd.DataFrame) -> None:
    _require_columns(frame, _SOURCE_COLUMNS, FULL_SOURCE_FILENAME)
    if frame.empty or frame["hadm_id"].duplicated().any() or frame["subject_id"].isna().any():
        raise ManifestBuildError("Symile source index is invalid")
    if not set(frame["cxr_ViewPosition"]) <= {"AP", "PA"} or not frame["labs_all_nan"].eq(0).all():
        raise ManifestBuildError("Symile modality eligibility is invalid")


def _load_development_tensor(root: Path, split: str, field: str) -> np.memmap:
    if split not in {"train", "val"}:
        raise PermissionError("Source qualification is limited to development tensors")
    try:
        array = np.load(
            root / "data_npy" / split / f"{field}_{split}.npy", mmap_mode="r", allow_pickle=False
        )
    except (OSError, ValueError) as exc:
        raise ManifestBuildError(f"Unreadable Symile tensor: {field}_{split}.npy") from exc
    if not isinstance(array, np.memmap):
        raise ManifestBuildError("Symile tensor did not open as a memory map")
    return array


def publish_symile_source_bundle(
    source_root: str | Path,
    *,
    output_directory: str | Path = "data/manifests",
    enforce_production_counts: bool = True,
) -> tuple[object, int]:
    """Authenticate transport and publish qualified development scientific evidence."""
    root = Path(source_root)
    assets, checksum_sha256 = authenticate_release(root)
    lab_means = _load_lab_contract(root)
    splits = {name: pd.read_csv(root / f"{name}.csv") for name in ("train", "val")}
    required = {
        "subject_id",
        "hadm_id",
        "cxr_path",
        "ecg_path",
        *LAB_ITEM_IDS,
        *(f"{item}_percentile" for item in LAB_ITEM_IDS),
    }
    for name, frame in splits.items():
        _require_columns(frame, required, f"{name}.csv")
        if frame.empty or frame["hadm_id"].duplicated().any():
            raise ManifestBuildError("Official development admissions are not unique")
    official = {"train": splits["train"], "validation": splits["val"]}
    for column in ("hadm_id", "subject_id"):
        if set(splits["train"][column]) & set(splits["val"][column]):
            raise ManifestBuildError("Official development patients or admissions overlap")
    counts = {f"{name}_admissions": len(frame) for name, frame in official.items()}
    if enforce_production_counts and counts != {
        "train_admissions": 10000,
        "validation_admissions": 750,
    }:
        raise ManifestBuildError(
            "Official Symile development membership differs from release contract"
        )

    def _read_development_index(path: Path, admission_ids: set[int]) -> pd.DataFrame:
        """Decode only records selected by known development keys from a mixed CSV.

        Record framing and key comparison operate on opaque bytes. Neither unselected keys
        nor their scientific fields are decoded, converted to values, retained or summarized.
        The source dialect has one physical record per line and decimal admission keys.
        """
        keys = {str(value).encode("ascii") for value in admission_ids}
        with path.open("rb") as stream:
            header = stream.readline()
            columns = header.rstrip(b"\r\n").split(b",")
            if columns.count(b"hadm_id") != 1:
                raise ManifestBuildError("Source index admission column is invalid")
            position = columns.index(b"hadm_id")
            selected = bytearray(header)
            # Quoted fields may contain commas and escaped quotes, but not record separators.
            field = re.compile(rb'(?:"(?:[^"\r\n]|"")*"|[^,"\r\n]*)(?:,|$)')
            for record in stream:
                raw = record.rstrip(b"\r\n")
                fields = []
                cursor = 0
                while cursor < len(raw):
                    match = field.match(raw, cursor)
                    if match is None or match.end() == cursor:
                        raise ManifestBuildError("Source index CSV framing is invalid")
                    fields.append(match.group().removesuffix(b","))
                    cursor = match.end()
                if raw.endswith(b","):
                    fields.append(b"")
                if len(fields) != len(columns):
                    raise ManifestBuildError("Source index CSV framing is invalid")
                if fields[position] in keys:
                    selected.extend(record)
        frame = pd.read_csv(io.BytesIO(selected), usecols=lambda name: name in _SOURCE_COLUMNS)
        _validate_source_index(frame)
        if set(frame["hadm_id"].astype(int)) != admission_ids:
            raise ManifestBuildError("Development admissions are missing from source index")
        return frame

    full = _read_development_index(
        root / FULL_SOURCE_FILENAME,
        set().union(*(set(frame["hadm_id"].astype(int)) for frame in official.values())),
    )
    modality_evidence = _qualify_modalities(
        splits,
        official,
        full,
        lab_means,
        load_npy=lambda split, field: _load_development_tensor(root, split, field),
        exact_contract=enforce_production_counts,
    )
    samples, labs = _canonical_frames(official, full)
    source = _QualifiedSymileSource(
        samples,
        labs,
        assets,
        checksum_sha256,
        {"reconciliation": counts, "modality_qualification": modality_evidence},
        root,
    )
    from beyondcxr.data.symile_artifacts import _build_symile_artifacts, _write_symile_bundle

    result = _build_symile_artifacts(source)
    paths = _write_symile_bundle(result, output_directory)
    return paths, result.samples.num_rows


def _canonical_frames(
    official: Mapping[str, pd.DataFrame], full: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_lookup = full.set_index("hadm_id", verify_integrity=True)
    sample_rows: list[dict[str, object]] = []
    lab_rows: list[dict[str, object]] = []
    for split in official:
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
    split_frames: Mapping[str, pd.DataFrame],
    official_frames: Mapping[str, pd.DataFrame],
    full: pd.DataFrame,
    lab_means: Mapping[str, float],
    *,
    load_npy: Callable[[str, str], np.memmap],
    exact_contract: bool,
) -> dict[str, object]:
    full_lookup = full.set_index("hadm_id", verify_integrity=True)
    split_evidence: dict[str, object] = {}
    for official_name in official_frames:
        source_name = _SPLIT_SOURCE_NAMES[official_name]
        complete = split_frames[source_name]
        selected = official_frames[official_name]
        selected_rows = selected.index.to_numpy(dtype=np.int64)
        hadm = load_npy(source_name, "hadm_id")
        if (
            hadm.dtype != np.int64
            or hadm.shape != (len(complete),)
            or not np.array_equal(hadm, complete["hadm_id"].to_numpy(dtype=np.int64))
        ):
            raise ManifestBuildError(f"{source_name} identifier tensor is not row-aligned")
        if source_name == "test":
            for field in ("label", "label_hadm_id"):
                array = load_npy(source_name, field)
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
        cxr = load_npy(source_name, "cxr")
        ecg = load_npy(source_name, "ecg")
        _validate_cxr(cxr, len(complete), selected_rows, exact_contract=exact_contract)
        _validate_ecg(ecg, len(complete), selected_rows, exact_contract=exact_contract)
        _validate_labs(
            source_name,
            complete,
            selected_rows,
            lab_means,
            full_lookup,
            load_npy=load_npy,
        )
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
    if (
        len({tuple(value["cxr_shape"]) for value in split_evidence.values()}) != 1
        or len({tuple(value["ecg_shape"]) for value in split_evidence.values()}) != 1
    ):
        raise ManifestBuildError("Classification modality shapes differ across splits")
    return {"splits": split_evidence}


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
    split: str,
    frame: pd.DataFrame,
    rows: np.ndarray,
    lab_means: Mapping[str, float],
    full_lookup: pd.DataFrame,
    *,
    load_npy: Callable[[str, str], np.memmap],
) -> None:
    percentiles = load_npy(split, "labs_percentiles")
    missingness = load_npy(split, "labs_missingness")
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
    frame: pd.DataFrame,
    *,
    load_npy: Callable[[str, str], np.memmap],
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
    cxr = load_npy("val_retrieval", "cxr")
    ecg = load_npy("val_retrieval", "ecg")
    _validate_cxr(cxr, count, np.array([], dtype=np.int64), exact_contract=exact_contract)
    _validate_ecg(ecg, count, np.array([], dtype=np.int64), exact_contract=exact_contract)
    for field, (dtype, shape) in expected.items():
        array = load_npy("val_retrieval", field)
        if array.dtype != dtype or array.shape != shape:
            raise ManifestBuildError("Validation retrieval tensor header contract does not match")
    for field in ("hadm_id", "label_hadm_id", "label"):
        if not np.array_equal(
            load_npy("val_retrieval", field), frame[field].to_numpy(dtype=np.int64)
        ):
            raise ManifestBuildError("Validation retrieval identifier structure is not aligned")


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
