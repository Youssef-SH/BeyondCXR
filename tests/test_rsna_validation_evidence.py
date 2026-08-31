from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from rsna_manifest_test_support import write_header, write_sources
from rsna_validation_evidence_test_support import write_synthetic_validation_evidence

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.rsna_artifacts import build_and_write
from beyondcxr.training.config import load_experiment_config, with_runtime
from beyondcxr.training.rsna_validation_evidence import (
    CanonicalValidationCohort,
    _derive_canonical_held_out_cohort,
    validate_cxr_epoch_history,
    validate_validation_evidence,
    validation_evidence_id,
)


def test_canonical_held_out_cohort_has_one_validated_derivation_owner(tmp_path: Path) -> None:
    source = write_sources(tmp_path / "source")
    labels_path = source / "stage_2_train_labels.csv"
    classes_path = source / "stage_2_detailed_class_info.csv"
    labels = pd.read_csv(labels_path)
    classes = pd.read_csv(classes_path)
    added_labels: list[dict[str, object]] = []
    added_classes: list[dict[str, object]] = []
    for target in (0, 1):
        for index in (1, 2):
            patient_id = f"target-{target}-{index}"
            write_header(source / "stage_2_train_images" / f"{patient_id}.dcm", patient_id)
            added_labels.append(
                {
                    "patientId": patient_id,
                    "x": 1 if target else None,
                    "y": 2 if target else None,
                    "width": 3 if target else None,
                    "height": 4 if target else None,
                    "Target": target,
                }
            )
            added_classes.append(
                {
                    "patientId": patient_id,
                    "class": "Lung Opacity" if target else "Normal",
                }
            )
    pd.concat([labels, pd.DataFrame(added_labels)], ignore_index=True).to_csv(
        labels_path, index=False
    )
    pd.concat([classes, pd.DataFrame(added_classes)], ignore_index=True).to_csv(
        classes_path, index=False
    )
    written = build_and_write(source, tmp_path / "manifests")
    package = {
        "bundle_id": written.paths.bundle_id,
        "task_id": "pneumonia",
        "label_policy_version": "rsna-stage-2-target-v1",
    }

    cohort = _derive_canonical_held_out_cohort(written.paths.bundle_directory, package)

    assert cohort.sample_ids == tuple(sorted(cohort.sample_ids))
    assert len(cohort.sample_ids) == 2
    assert set(cohort.targets) == {0, 1}
    with pytest.raises(ManifestBuildError, match="held-out cohort authority"):
        _derive_canonical_held_out_cohort(
            written.paths.bundle_directory,
            {**package, "task_id": "other-task"},
        )


def test_validation_evidence_identity_excludes_calibration_rendering_policy(
    tmp_path: Path,
) -> None:
    path = write_synthetic_validation_evidence(
        tmp_path / "validation-evidence.json",
        Path("configs/rsna_metadata_logistic.yaml"),
        seed=42,
    )
    document = json.loads(path.read_bytes())
    baseline = document["validation_evidence_id"]
    assert baseline == validation_evidence_id(document)

    changed_bins = deepcopy(document)
    changed_bins["calibration_bins"] += 1
    assert validation_evidence_id(changed_bins) == baseline

    changed_target = deepcopy(document)
    changed_target["sensitivity_target"] = 0.8
    assert validation_evidence_id(changed_target) != baseline


@pytest.mark.parametrize(
    "attack",
    ("omit", "substitute", "duplicate", "wrong-target", "reorder-ids-only"),
)
def test_repaired_validation_evidence_cannot_change_canonical_cohort(
    tmp_path: Path,
    attack: str,
) -> None:
    config = with_runtime(
        load_experiment_config(
            "configs/rsna_metadata_logistic.yaml", bundle_id="bundle-" + "a" * 64
        ),
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
    document["validation_evidence_id"] = validation_evidence_id(document)
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
            "task_id",
            "label_policy_version",
            "family_id",
            "seed",
        )
    }
    package["validation_evidence_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    package["validation_evidence_id"] = validation_evidence_id(document)

    with pytest.raises(ValueError, match="canonical validation cohort"):
        validate_validation_evidence(
            path,
            package=package,
            config=config,
            expected_cohort=cohort,
        )


def _cxr_evidence(tmp_path: Path):
    config = with_runtime(
        load_experiment_config("configs/rsna_cxr_densenet.yaml", bundle_id="bundle-" + "a" * 64),
        seed=42,
    )
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
            "task_id",
            "label_policy_version",
            "family_id",
            "seed",
        )
    }
    package["thresholds"] = {"youden_j": 0.7, "target_sensitivity": 0.7}
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
    document["validation_evidence_id"] = validation_evidence_id(document)
    encoded = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    path.write_bytes(encoded)
    package["validation_evidence_sha256"] = hashlib.sha256(encoded).hexdigest()
    package["validation_evidence_id"] = validation_evidence_id(document)


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
