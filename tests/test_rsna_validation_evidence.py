from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from rsna_validation_evidence_test_support import write_synthetic_validation_evidence

from beyondcxr.training.config import load_experiment_config, with_runtime
from beyondcxr.training.rsna_validation_evidence import (
    CanonicalValidationCohort,
    evidence_semantic_sha256,
    validate_cxr_epoch_history,
    validate_validation_evidence,
)


@pytest.mark.parametrize(
    "attack",
    ("omit", "substitute", "duplicate", "wrong-target", "reorder-ids-only"),
)
def test_repaired_validation_evidence_cannot_change_canonical_cohort(
    tmp_path: Path,
    attack: str,
) -> None:
    config = with_runtime(
        load_experiment_config("configs/rsna_metadata_logistic.yaml"),
        seed=42,
    )
    path = write_synthetic_validation_evidence(
        tmp_path / "validation-evidence.json",
        Path("configs/rsna_metadata_logistic.yaml"),
        seed=42,
    )
    original = json.loads(path.read_bytes())
    cohort = CanonicalValidationCohort(
        tuple(original["sample_ids"]),
        tuple(original["targets"]),
    )
    document = deepcopy(original)
    if attack == "omit":
        for field in ("sample_ids", "targets", "probabilities"):
            document[field].pop(0)
    elif attack == "substitute":
        document["sample_ids"][0] = "rsna:" + "-".join(
            ("00000000", "0000", "4000", "8000", "000000000000")
        )
    elif attack == "duplicate":
        document["sample_ids"][0] = document["sample_ids"][1]
    elif attack == "wrong-target":
        document["targets"][0] = 1
    else:
        document["sample_ids"][0], document["sample_ids"][1] = (
            document["sample_ids"][1],
            document["sample_ids"][0],
        )
    path.write_bytes(
        (
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    )
    package = {
        key: document[key]
        for key in (
            "dataset_id",
            "bundle_id",
            "bundle_manifest_sha256",
            "split_assignment_id",
            "task_id",
            "label_policy_version",
            "family_id",
            "config_source_sha256",
            "config_semantic_sha256",
            "seed",
        )
    }
    package["validation_evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    package["validation_evidence_semantic_sha256"] = evidence_semantic_sha256(document)

    with pytest.raises(ValueError, match="canonical validation cohort"):
        validate_validation_evidence(
            path,
            package=package,
            config=config,
            expected_cohort=cohort,
        )


def _cxr_evidence(tmp_path: Path):
    config = with_runtime(load_experiment_config("configs/rsna_cxr_densenet.yaml"), seed=42)
    path = write_synthetic_validation_evidence(
        tmp_path / "validation-evidence.json",
        Path("configs/rsna_cxr_densenet.yaml"),
        seed=42,
        selected_epoch=3,
        selected_stage="fine_tune",
        selected_validation_average_precision=0.8,
    )
    document = json.loads(path.read_bytes())
    package = {
        key: document[key]
        for key in (
            "dataset_id",
            "bundle_id",
            "bundle_manifest_sha256",
            "split_assignment_id",
            "task_id",
            "label_policy_version",
            "family_id",
            "config_source_sha256",
            "config_semantic_sha256",
            "seed",
        )
    }
    package["selection"] = {
        "selected_epoch": 3,
        "selected_stage": "fine_tune",
        "validation_average_precision": 0.8,
    }
    _repair_evidence_identity(path, document, package)
    return config, path, document, package


def _repair_evidence_identity(
    path: Path, document: dict[str, object], package: dict[str, object]
) -> None:
    encoded = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    path.write_bytes(encoded)
    package["validation_evidence_sha256"] = hashlib.sha256(encoded).hexdigest()
    package["validation_evidence_semantic_sha256"] = evidence_semantic_sha256(document)


@pytest.mark.parametrize(
    "attack",
    (
        "empty",
        "missing-field",
        "extra-field",
        "wrong-type",
        "invalid-ap",
        "negative-epoch",
        "negative-counter",
        "negative-lr",
        "invalid-stage",
        "skipped-global-epoch",
        "nonmonotonic-global-epoch",
        "invalid-stage-epoch",
        "illegal-stage-transition",
        "inconsistent-selected-best",
        "truncated-history",
        "continue-after-early-stop",
    ),
)
def test_cxr_epoch_history_rejects_repaired_malformed_contracts(
    tmp_path: Path, attack: str
) -> None:
    config, path, document, package = _cxr_evidence(tmp_path)
    history = document["epoch_history"]
    if attack == "empty":
        history.clear()
    elif attack == "missing-field":
        history[0].pop("training_loss")
    elif attack == "extra-field":
        history[0]["unexpected"] = 1
    elif attack == "wrong-type":
        history[0]["global_epoch"] = True
    elif attack == "invalid-ap":
        history[0]["validation_average_precision"] = 1.1
    elif attack == "negative-epoch":
        history[0]["global_epoch"] = -1
    elif attack == "negative-counter":
        history[2]["no_improvement_count"] = -1
    elif attack == "negative-lr":
        history[2]["encoder_learning_rate"] = -0.1
    elif attack == "invalid-stage":
        history[0]["stage"] = "selection"
    elif attack == "skipped-global-epoch":
        history[1]["global_epoch"] = 3
    elif attack == "nonmonotonic-global-epoch":
        history[1]["global_epoch"] = 1
    elif attack == "invalid-stage-epoch":
        history[1]["stage_epoch"] = 3
    elif attack == "illegal-stage-transition":
        history[1].update(
            {
                "stage": "fine_tune",
                "stage_epoch": 1,
                "encoder_learning_rate": 0.00001,
                "scheduler_last_epoch": 1,
            }
        )
    elif attack == "inconsistent-selected-best":
        history[1]["selected_best"] = True
    elif attack == "truncated-history":
        history.pop()
    else:
        final = deepcopy(history[-1])
        final["global_epoch"] += 1
        final["stage_epoch"] += 1
        final["scheduler_last_epoch"] += 1
        final["no_improvement_count"] += 1
        history.append(final)
    _repair_evidence_identity(path, document, package)

    with pytest.raises(ValueError, match="CXR epoch history"):
        validate_validation_evidence(path, package=package, config=config)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_cxr_epoch_history_rejects_nonfinite_numbers(tmp_path: Path, value: float) -> None:
    config, _path, document, package = _cxr_evidence(tmp_path)
    document["epoch_history"][0]["training_loss"] = value

    with pytest.raises(ValueError, match="CXR epoch history row values"):
        validate_cxr_epoch_history(document["epoch_history"], config=config, package=package)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("selected_epoch", 2),
        ("selected_stage", "warmup"),
        ("validation_average_precision", 0.7),
    ),
)
def test_cxr_epoch_history_must_match_package_selection(
    tmp_path: Path, field: str, value: object
) -> None:
    config, path, _document, package = _cxr_evidence(tmp_path)
    package["selection"][field] = value

    with pytest.raises(ValueError, match="package checkpoint selection"):
        validate_validation_evidence(path, package=package, config=config)


def test_cxr_epoch_history_allows_established_final_ap_tolerance(tmp_path: Path) -> None:
    config, path, _document, package = _cxr_evidence(tmp_path)
    package["selection"]["validation_average_precision"] += 5e-13

    validate_validation_evidence(path, package=package, config=config)


def test_cxr_epoch_history_accepts_zero_patience_early_stop(tmp_path: Path) -> None:
    config, path, document, package = _cxr_evidence(tmp_path)
    assert config.neural is not None
    config = replace(config, neural=replace(config.neural, early_stopping_patience=0))
    document["epoch_history"] = document["epoch_history"][:4]
    assert document["epoch_history"][-1]["no_improvement_count"] == 1
    _repair_evidence_identity(path, document, package)

    validate_validation_evidence(path, package=package, config=config)


def test_non_cxr_validation_evidence_forbids_epoch_history(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="validation evidence values"):
        write_synthetic_validation_evidence(
            tmp_path / "validation-evidence.json",
            Path("configs/rsna_metadata_logistic.yaml"),
            seed=42,
            epoch_history=(
                {
                    "global_epoch": 1,
                    "stage_epoch": 1,
                    "stage": "warmup",
                    "training_loss": 0.1,
                    "validation_average_precision": 0.8,
                    "selected_best": True,
                    "encoder_learning_rate": None,
                    "head_learning_rate": 0.001,
                    "scheduler_last_epoch": None,
                    "no_improvement_count": 0,
                },
            ),
        )
