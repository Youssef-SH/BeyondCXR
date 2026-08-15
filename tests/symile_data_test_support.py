"""Mechanical synthetic Symile fixture builders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from radfusion.data.symile_artifacts import (
    SymileBundlePaths,
    build_symile_artifacts,
    write_symile_bundle,
)
from radfusion.data.symile_schemas import (
    LAB_ITEM_IDS,
    LAB_NAMES,
)
from radfusion.data.symile_source import (
    EXPECTED_RELEASE_ASSETS,
    qualify_symile_source,
)


def _split_rows(
    admissions: list[tuple[int, int]],
    *,
    retrieval: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, (subject_id, hadm_id) in enumerate(admissions):
        row: dict[str, object] = {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "cxr_path": f"cxr/{hadm_id}.jpg",
            "ecg_path": f"ecg/{hadm_id}",
        }
        for lab_index, item_id in enumerate(LAB_ITEM_IDS):
            missing = lab_index == 0 and index % 3 == 0
            row[item_id] = np.nan if missing else float(index + lab_index + 1)
            row[f"{item_id}_percentile"] = np.nan if missing else 0.5
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not retrieval:
        return frame
    candidates: list[pd.DataFrame] = [frame.assign(label_hadm_id=frame["hadm_id"], label=1)]
    for query_index, (_, query) in enumerate(frame.iterrows()):
        negative_indices = [(query_index + offset) % len(frame) for offset in range(1, 10)]
        candidates.append(
            frame.iloc[negative_indices].assign(label_hadm_id=int(query["hadm_id"]), label=0)
        )
    return pd.concat(candidates, ignore_index=True)


def _full_rows(
    all_admissions: list[tuple[int, int]],
    states: dict[int, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, (subject_id, hadm_id) in enumerate(all_admissions):
        row: dict[str, object] = {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "admittime": "2020-01-01 00:00:00",
            "gender": "F" if index % 2 else "M",
            "age": 50 + index % 20,
            "cxr_dicom_id": f"dicom-{hadm_id}",
            "cxr_study_id": hadm_id + 1000,
            "cxr_ViewPosition": "AP" if index % 2 else "PA",
            "cxr_StudyDateTime": "2020-01-02 12:00:00",
            "cxr_path": f"cxr/{hadm_id}.jpg",
            "Pneumonia": states[hadm_id],
            "ecg_study_id": hadm_id + 2000,
            "ecg_time": "2020-01-01 01:00:00",
            "ecg_path": f"ecg/{hadm_id}",
            "labs_all_nan": 0,
        }
        for lab_index, item_id in enumerate(LAB_ITEM_IDS):
            row[item_id] = (
                np.nan if lab_index == 0 and index % 3 == 0 else float(index + lab_index + 1)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _modality_arrays(root: Path, split: str, frame: pd.DataFrame) -> None:
    directory = root / "data_npy" / split
    directory.mkdir(parents=True, exist_ok=True)
    count = len(frame)
    raw = np.linspace(0.2, 0.8, num=max(count, 1), dtype=np.float32)[:count]
    grayscale = np.broadcast_to(raw[:, None, None, None], (count, 3, 8, 8)).copy()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    np.save(directory / f"cxr_{split}.npy", ((grayscale - mean) / std).astype(np.float32))
    signal = np.linspace(-1.0, 1.0, 20 * 12, dtype=np.float32).reshape(1, 1, 20, 12)
    np.save(directory / f"ecg_{split}.npy", np.repeat(signal, count, axis=0))
    np.save(directory / f"hadm_id_{split}.npy", frame["hadm_id"].to_numpy(dtype=np.int64))
    raw_labs = frame[list(LAB_ITEM_IDS)].to_numpy(dtype=np.float64)
    mask = (~np.isnan(raw_labs)).astype(np.int64)
    means = np.full((1, len(LAB_ITEM_IDS)), 0.25, dtype=np.float64)
    percentiles = np.where(mask == 1, 0.5, means).astype(np.float32)
    np.save(directory / f"labs_missingness_{split}.npy", mask)
    np.save(directory / f"labs_percentiles_{split}.npy", percentiles)
    if split in {"test", "val_retrieval"}:
        np.save(
            directory / f"label_hadm_id_{split}.npy",
            frame["label_hadm_id"].to_numpy(dtype=np.int64),
        )
        np.save(directory / f"label_{split}.npy", frame["label"].to_numpy(dtype=np.int64))


def _refresh_checksums(root: Path) -> None:
    lines = []
    for relative in sorted(EXPECTED_RELEASE_ASSETS):
        path = root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest} {relative}")
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _synthetic_release(tmp_path: Path) -> Path:
    root = tmp_path / "symile"
    train_admissions = [(910_000, 810_000), (910_000, 810_001)] + [
        (910_000 + index, 810_000 + index) for index in range(2, 12)
    ]
    validation_admissions = [(920_000 + index, 820_000 + index) for index in range(10)]
    test_admissions = [(930_000 + index, 830_000 + index) for index in range(10)]
    excluded = [(910_000, 890_001), (920_000, 890_002)]
    all_admissions = [*train_admissions, *validation_admissions, *test_admissions, *excluded]
    states = {hadm_id: float(index % 2) for index, (_, hadm_id) in enumerate(all_admissions)}
    states[train_admissions[2][1]] = -1.0
    states[validation_admissions[0][1]] = np.nan
    full = _full_rows(all_admissions, states)
    train = _split_rows(train_admissions)
    validation = _split_rows(validation_admissions)
    test = _split_rows(test_admissions, retrieval=True)
    val_retrieval = _split_rows(validation_admissions, retrieval=True)
    full_labs = full.set_index("hadm_id")[list(LAB_ITEM_IDS)]
    for frame in (train, validation, test, val_retrieval):
        aligned = full_labs.loc[frame["hadm_id"].astype(int)].reset_index(drop=True)
        for item_id in LAB_ITEM_IDS:
            frame[item_id] = aligned[item_id].to_numpy()
            frame[f"{item_id}_percentile"] = np.where(frame[item_id].isna(), np.nan, 0.5)
    root.mkdir(parents=True)
    full.to_csv(root / "symile_mimic_data.csv", index=False)
    train.to_csv(root / "train.csv", index=False)
    validation.to_csv(root / "val.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    val_retrieval.to_csv(root / "val_retrieval.csv", index=False)
    (root / "labs_means.json").write_text(
        json.dumps({f"{item}_percentile": 0.25 for item in LAB_ITEM_IDS}),
        encoding="utf-8",
    )
    (root / "code").mkdir()
    (root / "code/constants.py").write_text(f"LABS = {LAB_NAMES!r}\n", encoding="utf-8")
    (root / "code/process_mimic_data.py").write_text(
        "def get_labs_df():\n    pass\n", encoding="utf-8"
    )
    _modality_arrays(root, "train", train)
    _modality_arrays(root, "val", validation)
    _modality_arrays(root, "test", test)
    _modality_arrays(root, "val_retrieval", val_retrieval)
    for relative in EXPECTED_RELEASE_ASSETS:
        path = root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic fixture\n")
    _refresh_checksums(root)
    return root


def _published_synthetic_release(tmp_path: Path) -> tuple[Path, SymileBundlePaths]:
    source_root = _synthetic_release(tmp_path)
    source = qualify_symile_source(source_root, enforce_production_counts=False)
    bundle = write_symile_bundle(build_symile_artifacts(source), tmp_path / "manifests")
    return source_root, bundle
