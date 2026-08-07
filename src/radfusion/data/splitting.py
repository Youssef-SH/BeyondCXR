"""Create and validate deterministic patient-level data splits."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from radfusion.data.artifact_validation import pneumonia_targets
from radfusion.data.errors import ManifestBuildError
from radfusion.data.schemas import PNEUMONIA_TASK_ID, RSNA_SPLIT_SCHEMA, require_exact_schema

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_ALGORITHM = "patient-stratified-sha256"
SPLIT_ALGORITHM_POLICY_VERSION = "2"
SPLIT_ALGORITHM_VERSION = f"{SPLIT_ALGORITHM}-v{SPLIT_ALGORITHM_POLICY_VERSION}"
SPLIT_SOURCE = "generated:patient-stratified-pneumonia"
PATIENT_GROUPING_RULE = "group-all-samples-by-patient-id"
PATIENT_TARGET_CONSISTENCY_RULE = "all-patient-samples-share-one-binary-target"
PATIENT_RANKING_RULE = "sha256-utf8-seed-nul-patient-id-with-patient-id-tiebreak"
SPLIT_ALLOCATION_RULE = "feasible-minimum-then-largest-remainder-canonical-tiebreak"
PATIENT_HASH_ALGORITHM = "sha256"
PATIENT_HASH_INPUT_ENCODING = "utf-8"
PATIENT_HASH_INPUT_TEMPLATE = "<seed>\\0<patient_id>"
STRATIFICATION_TARGET = PNEUMONIA_TASK_ID


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for deterministic patient-level stratification."""

    seed: int = 42
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    def validate(self) -> None:
        """Validate the seed and split ratios."""
        ratios = self.ratios
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ManifestBuildError("Split seed must be an integer")
        if not all(math.isfinite(value) and value > 0 for value in ratios):
            raise ManifestBuildError("Split ratios must be finite and strictly positive")
        if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ManifestBuildError("Split ratios must sum to 1.0")

    @property
    def ratios(self) -> tuple[float, float, float]:
        """Return ratios in canonical split order."""
        return self.train_ratio, self.validation_ratio, self.test_ratio

    @property
    def recipe_payload(self) -> dict[str, Any]:
        """Return the compact split recipe identity payload."""
        return {
            "algorithm_version": SPLIT_ALGORITHM_VERSION,
            "seed": self.seed,
            "stratification_target": STRATIFICATION_TARGET,
            "ratios": [
                {"split_name": name, "ratio": ratio}
                for name, ratio in zip(SPLIT_NAMES, self.ratios, strict=True)
            ],
        }

    @property
    def recipe_id(self) -> str:
        """Return the SHA-256 identity of the complete split policy."""
        self.validate()
        return _identity("split-recipe", self.recipe_payload)


def split_assignment_id(assignments: dict[str, str]) -> str:
    """Fingerprint canonical sample-to-split assignments."""
    payload = [[sample_id, assignments[sample_id]] for sample_id in sorted(assignments)]
    return _identity("split-assignment", payload)


def create_patient_stratified_splits(
    samples: pa.Table,
    labels: pa.Table,
    config: SplitConfig | None = None,
) -> pa.Table:
    """Assign every sample to a deterministic patient-level stratified split."""
    resolved = config or SplitConfig()
    resolved.validate()
    assignments = _create_assignments(samples, labels, resolved)
    records = [
        {
            "sample_id": sample_id,
            "split_name": assignments[sample_id],
        }
        for sample_id in sorted(assignments)
    ]
    splits = pa.Table.from_pylist(records, schema=RSNA_SPLIT_SCHEMA)
    validate_split_table(splits, samples, labels, config=resolved)
    return splits


def validate_split_table(
    splits: pa.Table,
    samples: pa.Table,
    labels: pa.Table,
    *,
    config: SplitConfig | None = None,
) -> None:
    """Validate split coverage, ordering, declared recipe, and patient isolation."""
    try:
        require_exact_schema(splits, RSNA_SPLIT_SCHEMA, "RSNA splits")
    except ValueError as exc:
        raise ManifestBuildError(str(exc)) from exc
    sample_rows = samples.to_pylist()
    split_rows = splits.to_pylist()
    expected_ids = {row["sample_id"] for row in sample_rows}
    split_ids = [row["sample_id"] for row in split_rows]
    if split_ids != sorted(split_ids):
        raise ManifestBuildError("RSNA splits must be ordered by sample_id")
    if len(split_ids) != len(set(split_ids)):
        raise ManifestBuildError("Each sample must have exactly one split assignment")
    if set(split_ids) != expected_ids:
        raise ManifestBuildError(
            f"Split coverage mismatch: missing={len(expected_ids - set(split_ids))}, "
            f"extra={len(set(split_ids) - expected_ids)}"
        )
    assignments = {row["sample_id"]: row["split_name"] for row in split_rows}
    if config is not None:
        config.validate()
        expected_assignments = _create_assignments(samples, labels, config)
        if assignments != expected_assignments:
            raise ManifestBuildError("Split rows do not match the declared split recipe")

    sample_patients = {row["sample_id"]: row["patient_id"] for row in sample_rows}
    patient_splits: dict[str, set[str]] = defaultdict(set)
    split_samples: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    for sample_id, split_name in assignments.items():
        if split_name not in SPLIT_NAMES:
            raise ManifestBuildError(f"Unexpected split_name {split_name!r}")
        patient_splits[sample_patients[sample_id]].add(split_name)
        split_samples[split_name].add(sample_id)
        split_counts[split_name] += 1
    overlap = sorted(patient for patient, names in patient_splits.items() if len(names) != 1)
    if overlap:
        raise ManifestBuildError(
            f"Patients assigned to multiple splits: {len(overlap)} (examples: {overlap[:10]})"
        )
    targets = pneumonia_targets(labels)
    patient_target_sets: dict[str, set[int]] = defaultdict(set)
    for row in sample_rows:
        patient_target_sets[row["patient_id"]].add(targets[row["sample_id"]])
    patient_class_counts = Counter(next(iter(values)) for values in patient_target_sets.values())
    if all(patient_class_counts[target] >= len(SPLIT_NAMES) for target in (0, 1)):
        if set(split_counts) != set(SPLIT_NAMES):
            raise ManifestBuildError("Every split must contain samples when stratification permits")
        for split_name in SPLIT_NAMES:
            classes = {targets[sample_id] for sample_id in split_samples[split_name]}
            if classes != {0, 1}:
                raise ManifestBuildError(f"Split {split_name!r} must contain both target classes")


def split_summary(splits: pa.Table, samples: pa.Table, labels: pa.Table) -> list[dict[str, object]]:
    """Return aggregate sample, patient, and target counts for each split."""
    assignments = {row["sample_id"]: row["split_name"] for row in splits.to_pylist()}
    targets = pneumonia_targets(labels)
    patients: dict[str, set[str]] = defaultdict(set)
    counts: Counter[tuple[str, int]] = Counter()
    for row in samples.to_pylist():
        split_name = assignments[row["sample_id"]]
        patients[split_name].add(row["patient_id"])
        counts[(split_name, targets[row["sample_id"]])] += 1
    return [
        {
            "split_name": name,
            "sample_count": counts[(name, 0)] + counts[(name, 1)],
            "patient_count": len(patients[name]),
            "negative_count": counts[(name, 0)],
            "positive_count": counts[(name, 1)],
        }
        for name in SPLIT_NAMES
    ]


def _create_assignments(samples: pa.Table, labels: pa.Table, config: SplitConfig) -> dict[str, str]:
    sample_rows = samples.to_pylist()
    if not sample_rows:
        raise ManifestBuildError("Cannot split an empty sample table")
    targets = pneumonia_targets(labels)
    patient_samples: dict[str, list[str]] = defaultdict(list)
    patient_targets: dict[str, set[int]] = defaultdict(set)
    for row in sample_rows:
        sample_id = row["sample_id"]
        if sample_id not in targets:
            raise ManifestBuildError(f"Sample {sample_id!r} has no pneumonia target")
        patient_samples[row["patient_id"]].append(sample_id)
        patient_targets[row["patient_id"]].add(targets[sample_id])
    strata: dict[int, list[str]] = defaultdict(list)
    for patient_id, values in patient_targets.items():
        if len(values) != 1:
            raise ManifestBuildError(
                f"Patient {patient_id!r} has inconsistent pneumonia targets: {sorted(values)}"
            )
        strata[next(iter(values))].append(patient_id)
    patient_assignments: dict[str, str] = {}
    for target in sorted(strata):
        ranked = sorted(
            strata[target],
            key=lambda patient_id: (_patient_rank(config.seed, patient_id), patient_id),
        )
        counts = _allocate_counts(len(ranked), config.ratios)
        offset = 0
        for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
            for patient_id in ranked[offset : offset + count]:
                patient_assignments[patient_id] = split_name
            offset += count
    return {
        sample_id: patient_assignments[patient_id]
        for patient_id, sample_ids in patient_samples.items()
        for sample_id in sample_ids
    }


def _patient_rank(seed: int, patient_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{patient_id}".encode(PATIENT_HASH_INPUT_ENCODING)).digest()


def _allocate_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if total < 0:
        raise ManifestBuildError("Split stratum size must be nonnegative")
    destinations = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if total < len(destinations):
        counts = [0] * len(ratios)
        ranked_destinations = sorted(destinations, key=lambda index: (-ratios[index], index))
        for index in ranked_destinations[:total]:
            counts[index] = 1
        return counts[0], counts[1], counts[2]
    raw = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    if total >= len(destinations) and any(counts[index] == 0 for index in destinations):
        counts = [1 if index in destinations else 0 for index in range(len(ratios))]
        for _ in range(total - sum(counts)):
            index = min(
                destinations,
                key=lambda candidate: (-(raw[candidate] - counts[candidate]), candidate),
            )
            counts[index] += 1
    return counts[0], counts[1], counts[2]


def _identity(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{prefix}-{digest}"
