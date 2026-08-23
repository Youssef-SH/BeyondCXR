from __future__ import annotations

from collections import Counter, defaultdict

import pyarrow as pa
import pytest

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.rsna_schemas import RSNA_LABEL_SCHEMA, RSNA_SAMPLE_SCHEMA, RSNA_SPLIT_SCHEMA
from beyondcxr.data.rsna_splitting import (
    SPLIT_ALGORITHM_VERSION,
    SPLIT_NAMES,
    SplitConfig,
    create_patient_stratified_splits,
    split_assignment_id,
    validate_split_table,
)


def _cohort(patient_count: int = 20) -> tuple[pa.Table, pa.Table]:
    samples: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for target in (0, 1):
        for index in range(patient_count):
            patient_id = f"patient-{target}-{index:02d}"
            sample_count = 2 if index < 3 else 1
            for sample_index in range(sample_count):
                sample_id = f"rsna:{patient_id}-{sample_index}"
                samples.append(
                    {
                        "sample_id": sample_id,
                        "patient_id": patient_id,
                        "image_id": f"{patient_id}-{sample_index}",
                        "image_path": f"stage_2_train_images/{patient_id}-{sample_index}.dcm",
                        "image_rows": 1024,
                        "image_columns": 1024,
                        "age_years": 50.0,
                        "age_is_implausible": False,
                        "sex": "F",
                        "view_position": "PA",
                        "pixel_spacing_row_mm": 0.168,
                        "pixel_spacing_col_mm": 0.168,
                    }
                )
                labels.append(
                    {
                        "sample_id": sample_id,
                        "task_id": "pneumonia",
                        "label_value": target,
                    }
                )
    return (
        pa.Table.from_pylist(sorted(samples, key=lambda row: row["sample_id"]), RSNA_SAMPLE_SCHEMA),
        pa.Table.from_pylist(sorted(labels, key=lambda row: row["sample_id"]), RSNA_LABEL_SCHEMA),
    )


def test_split_is_deterministic_disjoint_stratified_and_complete() -> None:
    samples, labels = _cohort()
    first = create_patient_stratified_splits(samples, labels)
    second = create_patient_stratified_splits(samples, labels)

    assert first.schema == RSNA_SPLIT_SCHEMA
    assert first.equals(second)
    assert first.num_rows == samples.num_rows
    assignments = {row["sample_id"]: row["split_name"] for row in first.to_pylist()}
    patients: dict[str, set[str]] = defaultdict(set)
    patient_targets: dict[str, int] = {}
    targets = {row["sample_id"]: row["label_value"] for row in labels.to_pylist()}
    for row in samples.to_pylist():
        patients[row["patient_id"]].add(assignments[row["sample_id"]])
        patient_targets[row["patient_id"]] = targets[row["sample_id"]]
    assert all(len(names) == 1 for names in patients.values())
    counts = Counter(
        (next(iter(names)), patient_targets[patient_id]) for patient_id, names in patients.items()
    )
    assert counts == {
        ("train", 0): 14,
        ("validation", 0): 3,
        ("test", 0): 3,
        ("train", 1): 14,
        ("validation", 1): 3,
        ("test", 1): 3,
    }


def test_split_seed_changes_assignments_and_recipe_identity() -> None:
    samples, labels = _cohort()
    first = create_patient_stratified_splits(samples, labels, SplitConfig(seed=42))
    second = create_patient_stratified_splits(samples, labels, SplitConfig(seed=43))

    assert SplitConfig(seed=42).recipe_id != SplitConfig(seed=43).recipe_id
    first_assignments = {row["sample_id"]: row["split_name"] for row in first.to_pylist()}
    second_assignments = {row["sample_id"]: row["split_name"] for row in second.to_pylist()}
    assert split_assignment_id(first_assignments) != split_assignment_id(second_assignments)
    assert first.column("split_name").to_pylist() != second.column("split_name").to_pylist()


def test_split_recipe_payload_contains_only_compact_identity_fields() -> None:
    assert SplitConfig().recipe_payload == {
        "algorithm_version": SPLIT_ALGORITHM_VERSION,
        "seed": 42,
        "stratification_target": "pneumonia",
        "ratios": [
            {"split_name": "train", "ratio": 0.7},
            {"split_name": "validation", "ratio": 0.15},
            {"split_name": "test", "ratio": 0.15},
        ],
    }


@pytest.mark.parametrize(
    "config",
    [
        SplitConfig(train_ratio=0.0, validation_ratio=0.5, test_ratio=0.5),
        SplitConfig(train_ratio=0.7, validation_ratio=0.2, test_ratio=0.2),
    ],
)
def test_invalid_split_ratios_fail(config: SplitConfig) -> None:
    samples, labels = _cohort()
    with pytest.raises(ManifestBuildError):
        create_patient_stratified_splits(samples, labels, config)


def test_split_validation_rejects_patient_overlap() -> None:
    samples, labels = _cohort()
    splits = create_patient_stratified_splits(samples, labels)
    sample_rows = samples.to_pylist()
    duplicate_patient = sample_rows[0]["patient_id"]
    patient_sample_ids = [
        row["sample_id"] for row in sample_rows if row["patient_id"] == duplicate_patient
    ]
    assert len(patient_sample_ids) == 2
    rows = splits.to_pylist()
    for row in rows:
        if row["sample_id"] == patient_sample_ids[1]:
            current = row["split_name"]
            row["split_name"] = next(name for name in SPLIT_NAMES if name != current)
    invalid = pa.Table.from_pylist(rows, RSNA_SPLIT_SCHEMA)

    with pytest.raises(ManifestBuildError):
        validate_split_table(invalid, samples, labels)


@pytest.mark.parametrize(
    ("total", "expected"),
    [(1, (1, 0, 0)), (2, (1, 1, 0)), (3, (1, 1, 1)), (4, (2, 1, 1)), (20, (14, 3, 3))],
)
def test_split_allocation_is_feasible_and_deterministic(total: int, expected) -> None:
    samples, labels = _cohort(total)
    splits = create_patient_stratified_splits(samples, labels)
    assignments = {row["sample_id"]: row["split_name"] for row in splits.to_pylist()}
    patient_assignments: dict[tuple[int, str], set[str]] = defaultdict(set)
    targets = {row["sample_id"]: row["label_value"] for row in labels.to_pylist()}
    for row in samples.to_pylist():
        patient_assignments[(targets[row["sample_id"]], assignments[row["sample_id"]])].add(
            row["patient_id"]
        )

    for target in (0, 1):
        assert tuple(len(patient_assignments[(target, name)]) for name in SPLIT_NAMES) == expected


def test_row_reordering_preserves_assignments() -> None:
    samples, labels = _cohort()
    reordered_samples = pa.Table.from_pylist(
        list(reversed(samples.to_pylist())), RSNA_SAMPLE_SCHEMA
    )
    reordered_labels = pa.Table.from_pylist(list(reversed(labels.to_pylist())), RSNA_LABEL_SCHEMA)
    first = create_patient_stratified_splits(samples, labels)
    second = create_patient_stratified_splits(reordered_samples, reordered_labels)
    assert first.equals(second)


def test_different_cohorts_change_assignment_identity() -> None:
    samples, labels = _cohort()
    sample_rows = samples.to_pylist()[1:]
    retained = {row["sample_id"] for row in sample_rows}
    label_rows = [row for row in labels.to_pylist() if row["sample_id"] in retained]
    smaller_samples = pa.Table.from_pylist(sample_rows, RSNA_SAMPLE_SCHEMA)
    smaller_labels = pa.Table.from_pylist(label_rows, RSNA_LABEL_SCHEMA)
    first = create_patient_stratified_splits(samples, labels)
    second = create_patient_stratified_splits(smaller_samples, smaller_labels)
    first_assignments = {row["sample_id"]: row["split_name"] for row in first.to_pylist()}
    second_assignments = {row["sample_id"]: row["split_name"] for row in second.to_pylist()}
    assert split_assignment_id(first_assignments) != split_assignment_id(second_assignments)


def test_assignment_tampering_is_detected_against_declared_recipe() -> None:
    samples, labels = _cohort()
    splits = create_patient_stratified_splits(samples, labels)
    rows = splits.to_pylist()
    rows[0]["split_name"] = "test" if rows[0]["split_name"] != "test" else "train"
    with pytest.raises(ManifestBuildError):
        validate_split_table(
            pa.Table.from_pylist(rows, RSNA_SPLIT_SCHEMA),
            samples,
            labels,
            config=SplitConfig(),
        )
    with pytest.raises(ManifestBuildError):
        validate_split_table(splits, samples, labels, config=SplitConfig(seed=43))
