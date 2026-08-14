from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import radfusion.data.symile_artifacts as symile_artifacts
from radfusion.data.errors import ManifestBuildError
from radfusion.data.hashing import logical_arrow_sha256
from radfusion.data.symile_artifacts import (
    LABS_FILENAME,
    METADATA_FILENAME,
    SAMPLES_FILENAME,
    SymileBundlePaths,
    authenticate_source_asset,
    build_symile_artifacts,
    bundle_identity_payload,
    official_split_assignment_id,
    read_symile_samples,
    resolve_symile_bundle,
    semantic_bundle_id,
    strict_pneumonia_rows,
    validate_symile_bundle,
    validate_symile_bundle_reference,
    write_symile_bundle,
)
from radfusion.data.symile_audit import REPORT_FILENAMES, generate_symile_audit
from radfusion.data.symile_cv import (
    CV_ASSIGNMENTS_FILENAME,
    _cv_identity_payload,
    cv_assignment_id,
    generate_cv_assignments,
    publish_symile_cv,
    validate_symile_cv,
    validate_symile_cv_reference,
)
from radfusion.data.symile_manifest import main as symile_manifest_main
from radfusion.data.symile_schemas import (
    CV_SCHEMA,
    LAB_ITEM_IDS,
    LAB_NAMES,
    LAB_SCHEMA,
    SAMPLE_SCHEMA,
)
from radfusion.data.symile_source import (
    EXPECTED_RELEASE_ASSETS,
    parse_sha256sums,
    qualify_symile_source,
    select_test_queries,
    strict_pneumonia_target,
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


def test_exact_symile_schemas_and_lab_order() -> None:
    assert SAMPLE_SCHEMA == pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("subject_id", pa.int64(), nullable=False),
            pa.field("hadm_id", pa.int64(), nullable=False),
            pa.field("official_split", pa.string(), nullable=False),
            pa.field("source_row", pa.int64(), nullable=False),
            pa.field("pneumonia_state", pa.int8(), nullable=True),
            pa.field("age_years", pa.int16(), nullable=False),
            pa.field("sex", pa.string(), nullable=False),
            pa.field("view_position", pa.string(), nullable=False),
        ]
    )
    assert LAB_SCHEMA == pa.schema(
        [pa.field("sample_id", pa.string(), nullable=False)]
        + [pa.field(f"lab_{item}_value", pa.float64(), nullable=True) for item in LAB_ITEM_IDS]
        + [pa.field(f"lab_{item}_observed", pa.bool_(), nullable=False) for item in LAB_ITEM_IDS]
    )
    assert CV_SCHEMA == pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("repeat_seed", pa.int32(), nullable=False),
            pa.field("outer_fold", pa.int8(), nullable=False),
        ]
    )
    assert tuple(sorted(LAB_ITEM_IDS)) == LAB_ITEM_IDS


@pytest.mark.parametrize(
    ("state", "target"), [(1, 1), (0, 0), (-1, None), (None, None), (np.nan, None)]
)
def test_strict_endpoint_derivation(state: object, target: int | None) -> None:
    assert strict_pneumonia_target(state) == target


def test_positive_self_query_selector_retains_original_rows() -> None:
    frame = pd.DataFrame(
        {"hadm_id": [10, 11, 12], "label_hadm_id": [10, 10, 12], "label": [1, 0, 1]},
        index=[4, 7, 9],
    )
    selected = select_test_queries(frame)
    assert selected.index.tolist() == [4, 9]
    assert selected["hadm_id"].tolist() == [10, 12]


def _identity_metadata() -> dict[str, object]:
    return {
        "bundle_manifest_schema_version": 1,
        "dataset": {"dataset_id": "symile", "release": "1.0.0"},
        "tasks": {
            "pneumonia_strict": {
                "task_id": "pneumonia_strict",
                "label_source": "symile_mimic_data.csv:Pneumonia",
                "label_policy_version": "symile-pneumonia-strict-v1",
                "positive": "pneumonia_state == 1",
                "negative": "pneumonia_state == 0",
                "excluded": ["pneumonia_state == -1", "pneumonia_state is null"],
            }
        },
        "membership": {
            "source": {
                "train": "train.csv",
                "validation": "val.csv",
                "test": "positive self-query rows of test.csv",
            },
            "test_selector": "label == 1 and label_hadm_id == hadm_id",
            "split_assignment_id": "split-assignment-" + "0" * 64,
        },
        "source": {
            "release": "1.0.0",
            "checksum_manifest_sha256": "a" * 64,
            "source_assets": [
                {
                    "relative_path": "train.csv",
                    "byte_size": 1,
                    "expected_sha256": "b" * 64,
                    "observed_sha256": "b" * 64,
                }
            ],
        },
        "modalities": {
            "common_locator": ["official_split", "source_row"],
            "cxr": {
                "asset": "data_npy/<split>/cxr_<split>.npy",
                "dtype": "float32",
                "sample_shape": [3, 320, 320],
                "preprocessing": "official resize/crop/ImageNet normalization",
            },
            "ecg": {
                "asset": "data_npy/<split>/ecg_<split>.npy",
                "dtype": "float32",
                "sample_shape": [1, 5000, 12],
                "range": [-1.0, 1.0],
            },
            "temporal": {
                "cxr": {
                    "source": "symile_mimic_data.csv",
                    "semantics": (
                        "earliest eligible AP/PA image >24 and <=72 hours after admission"
                    ),
                },
                "ecg": {
                    "source": "symile_mimic_data.csv",
                    "semantics": "earliest valid recording within +/-24 hours of admission",
                },
                "laboratories": {
                    "source": "code/process_mimic_data.py:get_labs_df",
                    "semantics": (
                        "earliest values among 50 selected tests within 24 hours of admission"
                    ),
                    "release_scope": "selected values without raw laboratory-event timestamps",
                },
            },
            "labs": {
                "item_order": list(LAB_ITEM_IDS),
                "item_names": LAB_NAMES,
                "value_semantics": "raw float64 source value; null means unobserved",
                "observed_semantics": "true exactly when the raw value is non-null",
                "official_percentiles": "diagnostic",
            },
        },
        "qualification": {"diagnostic": "ignored"},
        "generation": {"timestamp": "ignored"},
        "artifacts": {"physical_file_sha256": "ignored"},
    }


def test_sha256sums_parser_is_strict() -> None:
    text = "\n".join(f"{'0' * 64} {name}" for name in sorted(EXPECTED_RELEASE_ASSETS)) + "\n"
    assert set(parse_sha256sums(text)) == EXPECTED_RELEASE_ASSETS
    with pytest.raises(ManifestBuildError):
        parse_sha256sums(text.replace(" LICENSE.txt", "\tLICENSE.txt", 1))


def test_semantic_bundle_identity_includes_only_meaning_and_logical_content() -> None:
    metadata = _identity_metadata()
    logical = {SAMPLES_FILENAME: "b" * 64, LABS_FILENAME: "c" * 64}
    identity = semantic_bundle_id(metadata, logical)
    for change in ("diagnostics", "generation", "physical", "checksum_manifest"):
        changed = deepcopy(metadata)
        if change == "diagnostics":
            changed["qualification"] = {"diagnostic": "changed"}
        elif change == "generation":
            changed["generation"] = {"timestamp": "different"}
        elif change == "physical":
            changed["artifacts"] = {"physical_file_sha256": "different encoding"}
        else:
            changed["source"]["checksum_manifest_sha256"] = "d" * 64
        assert semantic_bundle_id(changed, logical) == identity

    changes = []
    task_changed = deepcopy(metadata)
    task_changed["tasks"]["pneumonia_strict"]["positive"] = "different"
    changes.append((task_changed, logical))
    split_changed = deepcopy(metadata)
    split_changed["membership"]["split_assignment_id"] = "split-assignment-" + "1" * 64
    changes.append((split_changed, logical))
    source_changed = deepcopy(metadata)
    source_changed["source"]["source_assets"][0]["expected_sha256"] = "e" * 64
    changes.append((source_changed, logical))
    modality_changed = deepcopy(metadata)
    modality_changed["modalities"]["cxr"]["dtype"] = "float64"
    changes.append((modality_changed, logical))
    labs_changed = deepcopy(metadata)
    labs_changed["modalities"]["labs"]["observed_semantics"] = "different"
    changes.append((labs_changed, logical))
    changes.append((metadata, {**logical, SAMPLES_FILENAME: "d" * 64}))
    changes.append((metadata, {**logical, LABS_FILENAME: "e" * 64}))
    assert all(semantic_bundle_id(changed, hashes) != identity for changed, hashes in changes)


def test_semantic_bundle_identity_is_independent_of_parquet_encoding(tmp_path: Path) -> None:
    table = pa.table({"value": pa.array([1, 2], type=pa.int64())})
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(table, first, compression="zstd")
    pq.write_table(table, second, compression=None)
    assert first.read_bytes() != second.read_bytes()
    first_hash = logical_arrow_sha256(pq.read_table(first))
    second_hash = logical_arrow_sha256(pq.read_table(second))
    assert first_hash == second_hash
    metadata = _identity_metadata()
    assert semantic_bundle_id(
        metadata, {SAMPLES_FILENAME: first_hash, LABS_FILENAME: "c" * 64}
    ) == semantic_bundle_id(metadata, {SAMPLES_FILENAME: second_hash, LABS_FILENAME: "c" * 64})


def test_semantic_bundle_identity_is_independent_of_source_root_location(tmp_path: Path) -> None:
    first_source = qualify_symile_source(
        _synthetic_release(tmp_path / "first"), enforce_production_counts=False
    )
    second_source = qualify_symile_source(
        _synthetic_release(tmp_path / "second"), enforce_production_counts=False
    )

    first = build_symile_artifacts(first_source)
    second = build_symile_artifacts(second_source)
    assert semantic_bundle_id(
        first.metadata,
        {
            SAMPLES_FILENAME: logical_arrow_sha256(first.samples),
            LABS_FILENAME: logical_arrow_sha256(first.labs),
        },
    ) == semantic_bundle_id(
        second.metadata,
        {
            SAMPLES_FILENAME: logical_arrow_sha256(second.samples),
            LABS_FILENAME: logical_arrow_sha256(second.labs),
        },
    )


def test_symile_semantic_payload_uses_abstract_roles_and_canonical_source_assets() -> None:
    metadata = _identity_metadata()
    metadata["source"]["source_assets"] = list(reversed(metadata["source"]["source_assets"]))
    payload = bundle_identity_payload(
        metadata, {SAMPLES_FILENAME: "b" * 64, LABS_FILENAME: "c" * 64}
    )

    assert set(payload["artifacts"]) == {"samples", "labs"}
    assert ".parquet" not in json.dumps(payload, sort_keys=True)


def test_synthetic_source_bundle_audit_and_access_workflow(tmp_path: Path) -> None:
    source_root = _synthetic_release(tmp_path)
    source = qualify_symile_source(source_root, enforce_production_counts=False)
    assert source.evidence["reconciliation"]["excluded_admissions"] == 2
    assert source.evidence["reconciliation"]["train_subject_exclusions"] == 1
    assert source.evidence["reconciliation"]["validation_subject_exclusions"] == 1
    result = build_symile_artifacts(source)
    manifest_root = tmp_path / "manifests"
    paths = write_symile_bundle(result, manifest_root)

    assert {path.name for path in paths.bundle_directory.iterdir()} == {
        SAMPLES_FILENAME,
        LABS_FILENAME,
        METADATA_FILENAME,
    }
    validate_symile_bundle(paths.bundle_directory, expected_bundle_id=paths.bundle_id)
    reference = validate_symile_bundle_reference(
        paths.bundle_directory, expected_bundle_id=paths.bundle_id
    )
    assert len(reference.manifest_sha256) == 64
    assert resolve_symile_bundle(manifest_root).bundle_id == paths.bundle_id
    assert (
        resolve_symile_bundle(manifest_root, bundle_id=paths.bundle_id).bundle_id == paths.bundle_id
    )
    samples = read_symile_samples(paths, official_splits=["train", "validation"])
    assert set(samples["official_split"]) == {"train", "validation"}
    assert "target" not in samples
    assert set(strict_pneumonia_rows(samples)["target"]) == {0, 1}
    with pytest.raises(ManifestBuildError):
        read_symile_samples(paths, sample_ids=[samples.iloc[0]["sample_id"], "symile:missing"])
    assert (
        authenticate_source_asset(paths, source_root, official_split="train", modality="cxr").name
        == "cxr_train.npy"
    )

    second = write_symile_bundle(result, manifest_root)
    assert second.bundle_id == paths.bundle_id
    audit = generate_symile_audit(
        manifest_root,
        tmp_path / "reports" / "symile" / "audit",
        bundle_id=paths.bundle_id,
    )
    audit_directory = Path(audit["report_directory"])
    assert {path.name for path in audit_directory.iterdir()} == set(REPORT_FILENAMES)
    assert "symile:" not in (audit_directory / "symile_audit.md").read_text(encoding="utf-8")
    default_audit = generate_symile_audit(manifest_root, tmp_path / "default-reports")
    assert default_audit["bundle_id"] == paths.bundle_id


def test_manifest_validator_rejects_changed_fixed_declaration(tmp_path: Path) -> None:
    _, bundle = _published_synthetic_release(tmp_path)
    manifest = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
    manifest["modalities"]["common_locator"] = ["source_row", "official_split"]
    logical_hashes = {
        filename: manifest["artifacts"][filename]["logical_arrow_sha256"]
        for filename in (SAMPLES_FILENAME, LABS_FILENAME)
    }
    changed_bundle_id = semantic_bundle_id(manifest, logical_hashes)
    manifest["bundle"]["bundle_id"] = changed_bundle_id
    bundle.metadata_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ManifestBuildError):
        validate_symile_bundle_reference(
            bundle.bundle_directory,
            expected_bundle_id=changed_bundle_id,
            enforce_directory_name=False,
        )


def test_manifest_cli_reports_complete_immutable_lineage(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = _synthetic_release(tmp_path)
    source = qualify_symile_source(source_root, enforce_production_counts=False)
    monkeypatch.setattr("radfusion.data.symile_manifest.qualify_symile_source", lambda _: source)
    assert (
        symile_manifest_main(
            [
                "--source-root",
                str(source_root),
                "--output-directory",
                str(tmp_path / "manifests"),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["bundle_id"].startswith("bundle-")
    assert len(output["bundle_manifest_sha256"]) == 64
    assert output["split_assignment_id"].startswith("split-assignment-")


@pytest.mark.parametrize("identifier_column", ["sample_id", "subject_id", "hadm_id"])
def test_symile_audit_rejects_patient_or_admission_identifiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identifier_column: str,
) -> None:
    source_root, bundle = _published_synthetic_release(tmp_path)
    del source_root
    samples = read_symile_samples(bundle)
    leaked_value = str(samples.iloc[0][identifier_column])
    monkeypatch.setattr(
        "radfusion.data.symile_audit._audit_markdown",
        lambda *args: f"# Aggregate audit\n\n{leaked_value}\n",
    )

    with pytest.raises(ValueError):
        generate_symile_audit(
            tmp_path / "manifests",
            tmp_path / "reports",
            bundle_id=bundle.bundle_id,
        )


def test_source_qualification_and_bundle_validation_reject_tampering(tmp_path: Path) -> None:
    source_root = _synthetic_release(tmp_path)
    (source_root / "LICENSE.txt").write_bytes(b"tampered")
    with pytest.raises(ManifestBuildError):
        qualify_symile_source(source_root, enforce_production_counts=False)

    source_root = _synthetic_release(tmp_path / "header")
    cxr_path = source_root / "data_npy/train/cxr_train.npy"
    np.save(cxr_path, np.load(cxr_path, allow_pickle=False).astype(np.float64))
    _refresh_checksums(source_root)
    with pytest.raises(ManifestBuildError):
        qualify_symile_source(source_root, enforce_production_counts=False)

    source_root = _synthetic_release(tmp_path / "second")
    mask_path = source_root / "data_npy/train/labs_missingness_train.npy"
    mask = np.load(mask_path, allow_pickle=False)
    mask[0, 0] = 1 - mask[0, 0]
    np.save(mask_path, mask)
    _refresh_checksums(source_root)
    with pytest.raises(ManifestBuildError):
        qualify_symile_source(source_root, enforce_production_counts=False)

    source_root = _synthetic_release(tmp_path / "third")
    source = qualify_symile_source(source_root, enforce_production_counts=False)
    paths = write_symile_bundle(build_symile_artifacts(source), tmp_path / "manifests")
    with paths.samples_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ManifestBuildError):
        validate_symile_bundle_reference(paths.bundle_directory)


def _published_synthetic_release(tmp_path: Path) -> tuple[Path, SymileBundlePaths]:
    source_root = _synthetic_release(tmp_path)
    source = qualify_symile_source(source_root, enforce_production_counts=False)
    bundle = write_symile_bundle(build_symile_artifacts(source), tmp_path / "manifests")
    return source_root, bundle


def test_source_asset_authentication_rejects_tampered_bytes(tmp_path: Path) -> None:
    source_root, bundle = _published_synthetic_release(tmp_path)
    asset = source_root / "data_npy/train/cxr_train.npy"
    asset.write_bytes(asset.read_bytes() + b"tampered")

    with pytest.raises(ManifestBuildError):
        authenticate_source_asset(bundle, source_root, official_split="train", modality="cxr")


def test_source_asset_authentication_does_not_trust_changed_manifest_asset_hash(
    tmp_path: Path,
) -> None:
    source_root, bundle = _published_synthetic_release(tmp_path)
    asset = source_root / "data_npy/train/cxr_train.npy"
    asset.write_bytes(asset.read_bytes() + b"tampered")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    manifest = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
    entry = next(
        value
        for value in manifest["source"]["source_assets"]
        if value["relative_path"] == "data_npy/train/cxr_train.npy"
    )
    entry["byte_size"] = asset.stat().st_size
    entry["expected_sha256"] = digest
    entry["observed_sha256"] = digest
    bundle.metadata_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ManifestBuildError):
        authenticate_source_asset(bundle, source_root, official_split="train", modality="cxr")


def test_source_asset_authentication_rejects_changed_checksum_manifest(tmp_path: Path) -> None:
    source_root, bundle = _published_synthetic_release(tmp_path)
    checksum = source_root / "SHA256SUMS.txt"
    checksum.write_bytes(checksum.read_bytes() + b"\n")

    with pytest.raises(ManifestBuildError):
        authenticate_source_asset(bundle, source_root, official_split="train", modality="cxr")


@pytest.mark.parametrize("symlink_kind", ["checksum", "asset"])
def test_source_asset_authentication_rejects_symlinks(tmp_path: Path, symlink_kind: str) -> None:
    source_root, bundle = _published_synthetic_release(tmp_path)
    path = (
        source_root / "SHA256SUMS.txt"
        if symlink_kind == "checksum"
        else source_root / "data_npy/train/cxr_train.npy"
    )
    physical = tmp_path / f"physical-{path.name}"
    path.replace(physical)
    path.symlink_to(physical)

    with pytest.raises(ManifestBuildError):
        authenticate_source_asset(bundle, source_root, official_split="train", modality="cxr")


def test_failed_bundle_publication_preserves_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = _synthetic_release(tmp_path)
    source = qualify_symile_source(source_root, enforce_production_counts=False)
    result = build_symile_artifacts(source)
    manifest_root = tmp_path / "manifests"
    paths = write_symile_bundle(result, manifest_root)
    previous = paths.current_path.read_text(encoding="utf-8")

    def fail_validation(*args: object, **kwargs: object) -> None:
        raise ManifestBuildError("invalid stage")

    monkeypatch.setattr(symile_artifacts, "validate_symile_bundle", fail_validation)
    with pytest.raises(ManifestBuildError):
        write_symile_bundle(result, manifest_root)
    assert paths.current_path.read_text(encoding="utf-8") == previous
    assert not list(paths.bundle_directory.parent.glob(".*-staging-*"))


def test_cv_is_deterministic_grouped_and_bundle_bound(tmp_path: Path) -> None:
    source_root = _synthetic_release(tmp_path)
    source = qualify_symile_source(source_root, enforce_production_counts=False)
    manifest_root = tmp_path / "manifests"
    bundle = write_symile_bundle(build_symile_artifacts(source), manifest_root)
    samples = read_symile_samples(bundle)
    first = generate_cv_assignments(samples)
    second = generate_cv_assignments(samples)
    assert first.equals(second)
    assignment_id, directory = publish_symile_cv(bundle, manifest_directory=manifest_root)
    manifest = validate_symile_cv(
        directory,
        bundle=bundle,
        expected_assignment_id=assignment_id,
    )
    assignments = pq.read_table(directory / CV_ASSIGNMENTS_FILENAME).to_pandas()
    assert (
        assignments.shape[0]
        == len(
            strict_pneumonia_rows(
                samples.loc[samples["official_split"].isin(["train", "validation"])]
            )
        )
        * 3
    )
    assert set(assignments["repeat_seed"]) == {17, 42, 2026}
    assert set(assignments["outer_fold"]) == {0, 1, 2, 3, 4}
    grouped = assignments.loc[assignments["sample_id"].isin(["symile:810000", "symile:810001"])]
    for seed in (17, 42, 2026):
        assert grouped.loc[grouped["repeat_seed"] == seed, "outer_fold"].nunique() == 1
    assert manifest["bundle_id"] == bundle.bundle_id
    assert not (manifest_root / "symile" / "cv" / "CURRENT").exists()


def test_split_assignment_identity_is_order_independent() -> None:
    frame = pd.DataFrame(
        {"sample_id": ["symile:2", "symile:1"], "official_split": ["test", "train"]}
    )
    assert official_split_assignment_id(frame) == official_split_assignment_id(frame.iloc[::-1])


def test_cv_identity_changes_with_bundle_or_logical_assignment() -> None:
    original = cv_assignment_id("bundle-" + "a" * 64, "b" * 64)
    assert cv_assignment_id("bundle-" + "a" * 64, "b" * 64) == original
    assert cv_assignment_id("bundle-" + "c" * 64, "b" * 64) != original
    assert cv_assignment_id("bundle-" + "a" * 64, "d" * 64) != original


def test_cv_semantic_payload_uses_abstract_assignment_role() -> None:
    payload = _cv_identity_payload("bundle-" + "a" * 64, "b" * 64)

    assert payload["artifacts"] == {"assignments": "b" * 64}


def test_cv_validation_rejects_incorrect_declared_row_count(tmp_path: Path) -> None:
    source_root, bundle = _published_synthetic_release(tmp_path)
    del source_root
    assignment_id, directory = publish_symile_cv(bundle, manifest_directory=tmp_path / "manifests")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact"]["row_count"] += 1
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ManifestBuildError):
        validate_symile_cv(
            directory,
            bundle=bundle,
            expected_assignment_id=assignment_id,
        )


def test_cv_reference_validation_does_not_load_bundle_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle = _published_synthetic_release(tmp_path)
    assignment_id, directory = publish_symile_cv(bundle, manifest_directory=tmp_path / "manifests")

    def reject_bundle_sample_access(*args: object, **kwargs: object) -> None:
        raise AssertionError("reference validation accessed bundle samples")

    monkeypatch.setattr("radfusion.data.symile_cv.read_symile_samples", reject_bundle_sample_access)
    reference = validate_symile_cv_reference(
        directory,
        bundle_id=bundle.bundle_id,
        expected_assignment_id=assignment_id,
    )

    assert reference.assignments.num_rows > 0
    assert reference.manifest["task_id"] == "pneumonia_strict"


@pytest.mark.parametrize(
    ("value", "observed"),
    [(None, True), (1.0, False)],
)
def test_laboratory_validator_rejects_inconsistent_observedness(
    tmp_path: Path, value: float | None, observed: bool
) -> None:
    source_root = _synthetic_release(tmp_path)
    source = qualify_symile_source(source_root, enforce_production_counts=False)
    result = build_symile_artifacts(source)
    rows = result.labs.to_pylist()
    rows[0]["lab_50802_value"] = value
    rows[0]["lab_50802_observed"] = observed
    invalid_labs = pa.Table.from_pylist(rows, schema=LAB_SCHEMA)

    with pytest.raises(ManifestBuildError):
        symile_artifacts._validate_tables(result.samples, invalid_labs)
