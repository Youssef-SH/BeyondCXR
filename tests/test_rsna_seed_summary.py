from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from radfusion.training import rsna_seed_summary
from radfusion.training.rsna_seed_summary import (
    EXPECTED_SEEDS,
    MANIFEST_FILENAME,
    SEED_SUMMARY_FILENAMES,
    publish_seed_summary,
    validate_seed_summary,
)

_EVALUATION_IDS = {
    seed: "evaluation-" + f"{index:064x}" for index, seed in enumerate(EXPECTED_SEEDS, start=1)
}
_PACKAGE_IDS = {seed: "model-package-" + f"{seed:064x}" for seed in EXPECTED_SEEDS}


def _threshold_contract() -> dict[str, object]:
    return {
        "youden_j_policy_version": "youden-j-all-roc-highest-finite-tie-v1",
        "target_sensitivity_policy_version": "target-sensitivity-all-roc-highest-finite-v1",
        "sensitivity_target": 0.9,
        "positive_class": 1,
    }


def _evaluation_policy() -> dict[str, object]:
    return {
        "policy_version": "rsna-held-out-evaluation-v1",
        "calibration_bins": 15,
        "threshold_selection": _threshold_contract(),
    }


def _claims(seed: int) -> dict[str, object]:
    index = EXPECTED_SEEDS.index(seed)
    offset = index / 10
    operating = {
        "precision": 0.50 + offset,
        "recall": 0.60 + offset,
        "specificity": 0.70 + offset,
        "f1": 0.55 + offset,
        "true_negative": 10 + index,
        "false_positive": 4 - index,
        "false_negative": 3 - index,
        "true_positive": 11 + index,
    }
    return {
        "evaluation_scope": "test",
        "calibration": {
            "calibration_bins": 15,
            "calibration_binning_strategy": "uniform",
        },
        "probability_metrics": {
            "average_precision": 0.1 + offset,
            "roc_auc": 0.6 + offset,
            "brier_score": 0.3 - offset,
            "expected_calibration_error": 0.2 - offset / 2,
            "calibration_slope": 0.8 + offset,
            "calibration_intercept": -0.1 + offset,
        },
        "operating_points": {
            "youden_j": {"threshold": 0.4 + offset, "metrics": operating},
            "target_sensitivity": {
                "configured_target_sensitivity": 0.9,
                "threshold": 0.2 + offset,
                "metrics": {**operating, "recall": 0.9},
            },
        },
    }


def _package(seed: int) -> dict[str, object]:
    return {
        "model_package_id": _PACKAGE_IDS[seed],
        "dataset_id": "rsna",
        "task_id": "pneumonia",
        "family_id": "cxr_densenet",
        "modalities": ["cxr"],
        "bundle_id": "bundle-" + "a" * 64,
        "split_assignment_id": "split-assignment-" + "b" * 64,
        "label_policy_version": "rsna-pneumonia-v1",
        "positive_class": 1,
        "fit_config": {"family": {"parameters": {"dropout": 0.2}}},
        "model_identity": {
            "architecture": "DenseNet121",
            "pretrained_weight": {
                "declared_name": "densenet121-res224-chex",
                "stable_identifier": "https://example.invalid/weights.pt",
                "cache_filename": "weights.pt",
                "byte_size": 100,
                "sha256": "c" * 64,
            },
        },
        "input_contract": {"shape": [1, 224, 224]},
        "training_transform_contract": {"augmentation": "fixed"},
        "evaluation_transform_contract": {"deterministic": True},
        "training_policy": {"seed": seed, "selection": "validation_ap"},
        "threshold_contract": _threshold_contract(),
        "source_package_id": None,
    }


def _evaluation(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        manifest={
            "evaluation_id": _EVALUATION_IDS[seed],
            "dataset_id": "rsna",
            "task_id": "pneumonia",
            "family_id": "cxr_densenet",
            "modalities": ["cxr"],
            "seed": seed,
            "bundle_id": "bundle-" + "a" * 64,
            "split_assignment_id": "split-assignment-" + "b" * 64,
            "model_package_id": _PACKAGE_IDS[seed],
            "evaluation_policy": _evaluation_policy(),
            "claims": _claims(seed),
        }
    )


@pytest.fixture
def authority(monkeypatch: pytest.MonkeyPatch):
    evaluations = {identity: _evaluation(seed) for seed, identity in _EVALUATION_IDS.items()}
    packages = {_PACKAGE_IDS[seed]: _package(seed) for seed in EXPECTED_SEEDS}

    def validate_evaluation(path: Path, **kwargs):
        assert kwargs["expected_evaluation_id"] == path.name
        return evaluations[path.name]

    def validate_package(model_root: Path, package_id: str):
        del model_root
        return packages[package_id]

    monkeypatch.setattr(
        "radfusion.training.rsna_seed_summary.validate_rsna_evaluation", validate_evaluation
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_seed_summary.validate_rsna_model_package", validate_package
    )
    return evaluations, packages


def _summarize(tmp_path: Path, ids: list[str]):
    return publish_seed_summary(
        ids,
        output_directory=tmp_path / "reports",
        model_directory=tmp_path / "models/rsna",
        private_directory=tmp_path / "private",
    )


def test_summary_is_authoritative_deterministic_complete_and_idempotent(
    tmp_path: Path, authority
) -> None:
    ids = [_EVALUATION_IDS[2026], _EVALUATION_IDS[17], _EVALUATION_IDS[42]]
    first = _summarize(tmp_path, ids)
    second = _summarize(tmp_path, list(reversed(ids)))
    document = json.loads((first.directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert first.seed_summary_id == second.seed_summary_id
    assert first.directory == second.directory
    assert {path.name for path in first.directory.iterdir()} == SEED_SUMMARY_FILENAMES
    assert document["evaluation_ids"] == [_EVALUATION_IDS[seed] for seed in EXPECTED_SEEDS]
    assert set(document["aggregate"]["probability_metrics"]) == {
        "average_precision",
        "roc_auc",
        "brier_score",
        "expected_calibration_error",
        "calibration_slope",
        "calibration_intercept",
    }
    assert document["aggregate"]["probability_metrics"]["average_precision"] == pytest.approx(
        {"mean": 0.2, "sample_standard_deviation": 0.1}
    )
    assert set(document["aggregate"]["operating_points"]) == {
        "youden_j",
        "target_sensitivity",
    }
    assert (
        document["aggregate"]["operating_points"]["youden_j"]["metrics"]["true_positive"]["mean"]
        == 12
    )
    assert all(
        "threshold" not in point for point in document["aggregate"]["operating_points"].values()
    )
    assert [
        member["claims"]["operating_points"]["youden_j"]["threshold"]
        for member in document["members"]
    ] == pytest.approx([0.4, 0.5, 0.6])
    markdown = (first.directory / "summary.md").read_text(encoding="utf-8")
    for identity in (*_EVALUATION_IDS.values(), *_PACKAGE_IDS.values()):
        assert identity in markdown
    for metric in (
        "Average precision",
        "ROC AUC",
        "Brier score",
        "Calibration slope",
        "Calibration intercept",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "TN",
        "FP",
        "FN",
        "TP",
        "Sample standard deviation",
    ):
        assert metric in markdown
    assert "Per-seed Youden-J operating point" in markdown
    assert "Per-seed Target sensitivity operating point" in markdown


def test_summary_validation_renders_only_outside_immutable_result(
    tmp_path: Path, authority, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _summarize(tmp_path, list(_EVALUATION_IDS.values()))
    original_write = rsna_seed_summary._write
    workspaces: list[Path] = []

    def tracked_write(directory: Path, document: dict[str, object]) -> None:
        assert directory != result.directory
        assert result.directory not in directory.parents
        workspaces.append(directory)
        original_write(directory, document)

    monkeypatch.setattr(rsna_seed_summary, "_write", tracked_write)
    validate_seed_summary(
        result.directory,
        report_root=tmp_path / "reports",
        model_root=tmp_path / "models/rsna",
        private_root=tmp_path / "private",
    )
    assert len(workspaces) == 1
    assert not workspaces[0].exists()


@pytest.mark.parametrize("filename", sorted(SEED_SUMMARY_FILENAMES))
def test_summary_validation_rejects_child_symlinks(
    tmp_path: Path, authority, filename: str
) -> None:
    result = _summarize(tmp_path, list(_EVALUATION_IDS.values()))
    path = result.directory / filename
    target = tmp_path / f"physical-{filename.replace('.', '-')}"
    path.rename(target)
    path.symlink_to(target)

    with pytest.raises(ValueError, match="artifact set"):
        validate_seed_summary(
            result.directory,
            report_root=tmp_path / "reports",
            model_root=tmp_path / "models/rsna",
            private_root=tmp_path / "private",
        )


_MISSING_SCHEMA_VERSION = object()


@pytest.mark.parametrize("value", [True, 1.0, "1", None, 0, _MISSING_SCHEMA_VERSION])
def test_summary_requires_integer_schema_version_one(
    tmp_path: Path, authority, value: object
) -> None:
    result = _summarize(tmp_path, list(_EVALUATION_IDS.values()))
    manifest_path = result.directory / MANIFEST_FILENAME
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value is _MISSING_SCHEMA_VERSION:
        document.pop("seed_summary_schema_version")
    else:
        document["seed_summary_schema_version"] = value
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_seed_summary(
            result.directory,
            report_root=tmp_path / "reports",
            model_root=tmp_path / "models/rsna",
            private_root=tmp_path / "private",
        )


@pytest.mark.parametrize(
    "ids",
    [
        [],
        [_EVALUATION_IDS[17]],
        [_EVALUATION_IDS[17], _EVALUATION_IDS[42]],
        [_EVALUATION_IDS[17], _EVALUATION_IDS[17], _EVALUATION_IDS[2026]],
    ],
)
def test_summary_requires_exact_distinct_authorities(
    tmp_path: Path, authority, ids: list[str]
) -> None:
    with pytest.raises(ValueError):
        _summarize(tmp_path, ids)


def test_summary_rejects_incompatible_family(tmp_path: Path, authority) -> None:
    evaluations, _ = authority
    evaluations[_EVALUATION_IDS[42]].manifest["family_id"] = "cxr_metadata_concat"
    with pytest.raises(ValueError, match="compatible"):
        _summarize(tmp_path, list(_EVALUATION_IDS.values()))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("fit_config", {"family": {"parameters": {"dropout": 0.4}}}),
        (
            "model_identity",
            {
                "architecture": "DifferentNet",
                "pretrained_weight": {
                    "declared_name": "densenet121-res224-chex",
                    "stable_identifier": "https://example.invalid/weights.pt",
                    "cache_filename": "weights.pt",
                    "byte_size": 100,
                    "sha256": "c" * 64,
                },
            },
        ),
        ("training_policy", {"seed": 42, "selection": "different"}),
        ("threshold_contract", {**_threshold_contract(), "sensitivity_target": 0.8}),
        ("evaluation_transform_contract", {"deterministic": False}),
    ],
)
def test_summary_rejects_package_family_definition_drift(
    tmp_path: Path, authority, field: str, replacement: object
) -> None:
    _, packages = authority
    packages[_PACKAGE_IDS[42]][field] = replacement
    with pytest.raises(ValueError, match="compatible"):
        _summarize(tmp_path, list(_EVALUATION_IDS.values()))


def test_summary_rejects_evaluation_policy_drift(tmp_path: Path, authority) -> None:
    evaluations, _ = authority
    evaluations[_EVALUATION_IDS[42]].manifest["evaluation_policy"] = {
        **_evaluation_policy(),
        "calibration_bins": 10,
    }
    with pytest.raises(ValueError, match="compatible"):
        _summarize(tmp_path, list(_EVALUATION_IDS.values()))


def test_summary_rejects_wrong_seed_set(tmp_path: Path, authority) -> None:
    _, packages = authority
    packages[_PACKAGE_IDS[17]]["training_policy"]["seed"] = 18
    with pytest.raises(ValueError, match="seed"):
        _summarize(tmp_path, list(_EVALUATION_IDS.values()))


def test_summary_rejects_unsupported_three_seed_family(tmp_path: Path, authority) -> None:
    evaluations, packages = authority
    for seed in EXPECTED_SEEDS:
        packages[_PACKAGE_IDS[seed]].update(
            {"family_id": "metadata_logistic", "modalities": ["metadata"]}
        )
        evaluations[_EVALUATION_IDS[seed]].manifest.update(
            {"family_id": "metadata_logistic", "modalities": ["metadata"]}
        )
    with pytest.raises(ValueError, match="unsupported"):
        _summarize(tmp_path, list(_EVALUATION_IDS.values()))


def test_summary_ignores_operational_and_fitted_member_differences(
    tmp_path: Path, authority
) -> None:
    _, packages = authority
    for index, seed in enumerate(EXPECTED_SEEDS):
        packages[_PACKAGE_IDS[seed]]["runtime_provenance"] = {"device": f"cuda:{index}"}
        packages[_PACKAGE_IDS[seed]]["model_state_sha256"] = f"{index + 1:064x}"
        packages[_PACKAGE_IDS[seed]]["model_identity"]["pretrained_weight"].update(
            {"cache_filename": f"weights-{index}.pt", "byte_size": 100 + index}
        )
        packages[_PACKAGE_IDS[seed]]["thresholds"] = {
            "youden_j": 0.4 + index / 10,
            "target_sensitivity": 0.2 + index / 10,
        }
    result = _summarize(tmp_path, list(_EVALUATION_IDS.values()))
    assert result.seed_summary_id.startswith("seed-summary-")


def test_summary_rejects_fusion_source_with_different_seed(tmp_path: Path, authority) -> None:
    evaluations, packages = authority
    source_ids: dict[int, str] = {}
    for seed in EXPECTED_SEEDS:
        source_id = "model-package-" + f"{seed + 10_000:064x}"
        source_ids[seed] = source_id
        source = deepcopy(_package(seed))
        source["model_package_id"] = source_id
        packages[source_id] = source
        packages[_PACKAGE_IDS[seed]].update(
            {
                "family_id": "cxr_metadata_concat",
                "modalities": ["cxr", "metadata"],
                "source_package_id": source_id,
                "structured_preprocessor_contract": {"features": ["age"]},
                "structured_input_conversion": {"dtype": "float32"},
                "fusion_architecture": {"kind": "concat"},
            }
        )
        evaluations[_EVALUATION_IDS[seed]].manifest.update(
            {"family_id": "cxr_metadata_concat", "modalities": ["cxr", "metadata"]}
        )
    packages[source_ids[42]]["training_policy"]["seed"] = 17

    with pytest.raises(ValueError, match="source CXR"):
        _summarize(tmp_path, list(_EVALUATION_IDS.values()))


def test_summary_validator_rederives_aggregate(tmp_path: Path, authority) -> None:
    result = _summarize(tmp_path, list(_EVALUATION_IDS.values()))
    manifest_path = result.directory / MANIFEST_FILENAME
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["aggregate"]["probability_metrics"]["roc_auc"]["mean"] = 0.0
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="aggregate"):
        validate_seed_summary(
            result.directory,
            report_root=tmp_path / "reports",
            model_root=tmp_path / "models/rsna",
            private_root=tmp_path / "private",
        )


def test_changed_evaluation_authority_changes_summary_identity(tmp_path: Path, authority) -> None:
    evaluations, _ = authority
    first = _summarize(tmp_path / "first", list(_EVALUATION_IDS.values()))
    changed = deepcopy(evaluations[_EVALUATION_IDS[2026]])
    replacement = "evaluation-" + "f" * 64
    changed.manifest["evaluation_id"] = replacement
    evaluations[replacement] = changed
    second = _summarize(
        tmp_path / "second",
        [_EVALUATION_IDS[17], _EVALUATION_IDS[42], replacement],
    )
    assert second.seed_summary_id != first.seed_summary_id
