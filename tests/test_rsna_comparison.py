from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from beyondcxr.data.rsna_metadata_preprocess import metadata_input_contract
from beyondcxr.training import rsna_evaluation_result
from beyondcxr.training.config import load_experiment_config
from beyondcxr.training.rsna_compare import COMPARISON_COLUMNS, regenerate_comparison
from beyondcxr.training.rsna_evaluation_result import (
    publish_rsna_evaluation,
    validate_rsna_evaluation,
)
from beyondcxr.utils.private_predictions import publish_prediction_evidence
from beyondcxr.utils.rsna_model_publication import publish_model_package, threshold_contract
from beyondcxr.utils.skops_io import save_skops


def _package(tmp_path: Path, *, variant: int = 0):
    config_path = Path("configs/rsna_metadata_logistic.yaml")
    config = load_experiment_config(config_path)
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    targets = np.asarray([0, variant % 2, 1, 1])
    pipeline = Pipeline(
        [
            ("preprocess", "passthrough"),
            ("classifier", LogisticRegression(random_state=42).fit(features, targets)),
        ]
    )
    serialized = save_skops(pipeline, tmp_path / f"source-{variant}.skops")
    config_bytes = config_path.read_bytes()
    return publish_model_package(
        model_root=tmp_path / "models/rsna",
        serialized_model_path=serialized,
        source_config_bytes=config_bytes,
        manifest={
            "bundle_id": config.dataset.bundle_id,
            "split_assignment_id": config.dataset.split_assignment_id,
            "task_id": config.task.task_id,
            "positive_class": 1,
            "family_id": config.family.family_id,
            "config_source_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "config_semantic_sha256": config.config_semantic_sha256,
            "seed": 42,
            "git_commit": "test",
            "git_dirty": False,
            "dependency_lock_sha256": "a" * 64,
            "best_iteration": None,
            "thresholds": {"youden_j": 0.5, "target_sensitivity": 0.3},
            "threshold_contract": threshold_contract(sensitivity_target=0.9),
            "input_contract": metadata_input_contract(),
        },
    )


def _evaluation(
    tmp_path: Path,
    *,
    logits: tuple[float, ...],
    variant: int = 0,
):
    package = _package(tmp_path, variant=variant)
    package_manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
    sample_ids = tuple(f"rsna:sample-{index}" for index in range(len(logits)))
    targets = tuple(index % 2 for index in range(len(logits)))
    evidence = publish_prediction_evidence(
        private_root=tmp_path / "private",
        dataset_id="rsna",
        model_package_id=package.model_package_id,
        task_id="pneumonia",
        bundle_id=package_manifest["bundle_id"],
        split_assignment_id=package_manifest["split_assignment_id"],
        scope="test",
        sample_ids=sample_ids,
        targets=targets,
        logits=logits,
    )
    return publish_rsna_evaluation(
        report_root=tmp_path / "reports",
        private_root=tmp_path / "private",
        model_root=tmp_path / "models/rsna",
        evidence=evidence,
        evaluation_policy={
            "policy_version": "rsna-held-out-evaluation-v1",
            "calibration_bins": 15,
            "threshold_selection": threshold_contract(sensitivity_target=0.9),
        },
        forbidden_source_values=sample_ids,
    )


def test_comparison_consumes_explicit_evaluation_identities(tmp_path: Path) -> None:
    first = _evaluation(
        tmp_path,
        logits=(-2.0, 1.0, -1.0, 2.0),
    )
    second = _evaluation(
        tmp_path,
        logits=(-1.5, 0.5, -0.5, 1.5),
        variant=1,
    )

    csv_path, markdown_path, count = regenerate_comparison(
        [second.manifest["evaluation_id"], first.manifest["evaluation_id"]],
        output_directory=tmp_path / "reports",
        private_directory=tmp_path / "private",
        model_directory=tmp_path / "models/rsna",
    )
    frame = pd.read_csv(csv_path)

    assert count == 2
    assert tuple(frame.columns) == COMPARISON_COLUMNS
    assert frame["evaluation_id"].tolist() == sorted(
        [first.manifest["evaluation_id"], second.manifest["evaluation_id"]]
    )
    assert set(frame["model_package_id"]) == {
        first.manifest["model_package_id"],
        second.manifest["model_package_id"],
    }
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# Evaluation comparison\n\n| dataset_id |"
    )


def test_comparison_regeneration_is_deterministic(tmp_path: Path) -> None:
    result = _evaluation(
        tmp_path,
        logits=(-2.0, 1.0, -1.0, 2.0),
    )
    arguments = {
        "output_directory": tmp_path / "reports",
        "private_directory": tmp_path / "private",
        "model_directory": tmp_path / "models/rsna",
    }
    csv_path, markdown_path, _ = regenerate_comparison(
        [result.manifest["evaluation_id"]], **arguments
    )
    first_csv = csv_path.read_bytes()
    first_markdown = markdown_path.read_bytes()
    regenerate_comparison([result.manifest["evaluation_id"]], **arguments)
    assert csv_path.read_bytes() == first_csv
    assert markdown_path.read_bytes() == first_markdown


@pytest.mark.parametrize(
    "evaluation_ids",
    [[], ["evaluation-a", "evaluation-a"]],
)
def test_comparison_rejects_missing_or_duplicate_identities(
    tmp_path: Path, evaluation_ids: list[str]
) -> None:
    with pytest.raises(ValueError):
        regenerate_comparison(
            evaluation_ids,
            output_directory=tmp_path / "reports",
            private_directory=tmp_path / "private",
        )


def test_comparison_rejects_tampered_evaluation_claims(tmp_path: Path) -> None:
    result = _evaluation(
        tmp_path,
        logits=(-2.0, 1.0, -1.0, 2.0),
    )
    manifest_path = result.directory / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["claims"]["probability_metrics"]["average_precision"] = 0.0
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        regenerate_comparison(
            [result.manifest["evaluation_id"]],
            output_directory=tmp_path / "reports",
            private_directory=tmp_path / "private",
            model_directory=tmp_path / "models/rsna",
        )


def test_evaluation_validation_rejects_tampered_derivative(tmp_path: Path) -> None:
    result = _evaluation(
        tmp_path,
        logits=(-2.0, 1.0, -1.0, 2.0),
    )
    derivative = result.directory / "derivatives" / "metrics.json"
    derivative.write_bytes(derivative.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="derivative"):
        validate_rsna_evaluation(
            result.directory,
            private_root=tmp_path / "private",
            model_root=tmp_path / "models/rsna",
        )


@pytest.mark.parametrize("child", ["manifest.json", "derivatives"])
def test_evaluation_validation_rejects_child_symlinks(tmp_path: Path, child: str) -> None:
    result = _evaluation(tmp_path, logits=(-2.0, 1.0, -1.0, 2.0))
    path = result.directory / child
    target = tmp_path / f"physical-{child.replace('.', '-')}"
    path.rename(target)
    path.symlink_to(target, target_is_directory=target.is_dir())

    with pytest.raises(ValueError, match="artifact set"):
        validate_rsna_evaluation(
            result.directory,
            private_root=tmp_path / "private",
            model_root=tmp_path / "models/rsna",
        )


_MISSING_SCHEMA_VERSION = object()


@pytest.mark.parametrize("value", [True, 1.0, "1", None, 0, _MISSING_SCHEMA_VERSION])
def test_evaluation_requires_integer_schema_version_one(tmp_path: Path, value: object) -> None:
    result = _evaluation(tmp_path, logits=(-2.0, 1.0, -1.0, 2.0))
    manifest_path = result.directory / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value is _MISSING_SCHEMA_VERSION:
        document.pop("evaluation_schema_version")
    else:
        document["evaluation_schema_version"] = value
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_rsna_evaluation(
            result.directory,
            private_root=tmp_path / "private",
            model_root=tmp_path / "models/rsna",
        )


def test_evaluation_validation_renders_only_outside_immutable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _evaluation(tmp_path, logits=(-2.0, 1.0, -1.0, 2.0))
    original_write = rsna_evaluation_result.write_run_reports
    workspaces: list[Path] = []

    def tracked_write(directory: Path, **kwargs: object) -> None:
        assert directory != result.directory
        assert result.directory not in directory.parents
        workspaces.append(directory)
        original_write(directory, **kwargs)

    monkeypatch.setattr(rsna_evaluation_result, "write_run_reports", tracked_write)
    validate_rsna_evaluation(
        result.directory,
        private_root=tmp_path / "private",
        model_root=tmp_path / "models/rsna",
    )
    assert len(workspaces) == 1
    assert not workspaces[0].exists()


def test_evaluation_validation_requires_the_actual_model_package(tmp_path: Path) -> None:
    result = _evaluation(tmp_path, logits=(-2.0, 1.0, -1.0, 2.0))
    package_id = result.manifest["model_package_id"]
    package = tmp_path / "models/rsna/packages" / package_id
    package.rename(tmp_path / "unavailable-package")

    with pytest.raises(ValueError):
        validate_rsna_evaluation(
            result.directory,
            private_root=tmp_path / "private",
            model_root=tmp_path / "models/rsna",
        )


def test_evaluation_rederives_claims_from_identified_evidence(tmp_path: Path) -> None:
    original = _evaluation(
        tmp_path,
        logits=(-2.0, 1.0, -1.0, 2.0),
    )
    repeated = _evaluation(
        tmp_path,
        logits=(-2.0, 1.0, -1.0, 2.0),
    )
    assert original.created is True
    assert repeated.created is False
    assert repeated.manifest["evaluation_id"] == original.manifest["evaluation_id"]

    manifest_path = original.directory / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = document["evaluation_id"]
    document["claims"]["probability_metrics"]["roc_auc"] = 0.25
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_rsna_evaluation(
            original.directory,
            private_root=tmp_path / "private",
            model_root=tmp_path / "models/rsna",
            expected_evaluation_id=identity,
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("evaluation_policy", "calibration_bins", True),
        ("evaluation_policy", "calibration_bins", "15"),
        ("evaluation_policy", "calibration_bins", 1),
        ("threshold_selection", "sensitivity_target", 1),
        ("threshold_selection", "sensitivity_target", float("nan")),
        ("threshold_selection", "sensitivity_target", float("inf")),
        ("threshold_selection", "youden_j_policy_version", "changed-policy"),
        ("threshold_selection", "positive_class", 0),
        ("thresholds", "youden_j", True),
        ("thresholds", "youden_j", -0.1),
        ("thresholds", "target_sensitivity", float("inf")),
    ],
)
def test_evaluation_validation_rejects_malformed_scientific_policy(
    tmp_path: Path, section: str, field: str, value: object
) -> None:
    result = _evaluation(tmp_path, logits=(-2.0, 1.0, -1.0, 2.0))
    manifest_path = result.directory / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if section == "threshold_selection":
        document["evaluation_policy"][section][field] = value
    else:
        document[section][field] = value
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_rsna_evaluation(
            result.directory,
            private_root=tmp_path / "private",
            model_root=tmp_path / "models/rsna",
        )


def test_comparison_cli_consumes_evaluation_ids(tmp_path: Path) -> None:
    result = _evaluation(
        tmp_path,
        logits=(-2.0, 1.0, -1.0, 2.0),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "beyondcxr.training.rsna_compare",
            "--evaluation-ids",
            result.manifest["evaluation_id"],
            "--output-directory",
            str(tmp_path / "reports"),
            "--private-directory",
            str(tmp_path / "private"),
            "--model-directory",
            str(tmp_path / "models/rsna"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Wrote 1 rows" in completed.stdout
