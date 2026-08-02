from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from radfusion.training.compare import COMPARISON_COLUMNS, regenerate_comparison
from radfusion.training.completed_runs import require_completed_run
from radfusion.utils.mlflow_utils import configure_mlflow


def _tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def _run(
    *,
    scope: str,
    complete: bool = True,
    status: str = "FINISHED",
    omit_metric: str | None = None,
    parent_training_run_id: str | None = None,
    metric_override: tuple[str, float] | None = None,
    tag_override: tuple[str, str] | None = None,
    modality: str | None = "metadata",
) -> str:
    kind = "training" if scope == "validation" else "test_evaluation"
    parent = (
        parent_training_run_id
        if parent_training_run_id is not None
        else ("parent" if scope == "test" else "")
    )
    tags = {
        "run_kind": kind,
        "run_complete": str(complete).lower(),
        "evaluation_scope": scope,
        "experiment_name": "experiment",
        "model": "metadata_logistic",
        "task": "pneumonia",
        "model_package_id": "model-package-test",
        "dataset_bundle_id": "bundle",
        "split_assignment_id": "assignment",
        "seed": "42",
        "source_training_run_id": parent,
    }
    if modality is not None:
        tags["modality"] = modality
    if tag_override is not None:
        tags[tag_override[0]] = tag_override[1]
    with mlflow.start_run(tags=tags) as run:
        metrics = {
            f"{scope}_average_precision": 0.4,
            f"{scope}_roc_auc": 0.7,
            f"{scope}_brier_score": 0.2,
            f"{scope}_expected_calibration_error": 0.1,
            f"{scope}_calibration_slope": 1.0,
            f"{scope}_calibration_intercept": 0.0,
            f"{scope}_youden_j_threshold": 0.5,
            f"{scope}_youden_j_precision": 0.6,
            f"{scope}_youden_j_recall": 0.7,
            f"{scope}_youden_j_specificity": 0.8,
            f"{scope}_youden_j_f1": 0.65,
            f"{scope}_target_sensitivity_threshold": 0.3,
            f"{scope}_target_sensitivity_precision": 0.4,
            f"{scope}_target_sensitivity_recall": 0.9,
            f"{scope}_target_sensitivity_specificity": 0.3,
            f"{scope}_target_sensitivity_f1": 0.55,
            f"{scope}_latency_ms": 1.0,
            "model_size_mib": 2.0,
        }
        if omit_metric is not None:
            metrics.pop(omit_metric)
        if metric_override is not None:
            metrics[metric_override[0]] = metric_override[1]
        mlflow.log_metrics(metrics)
        run_id = run.info.run_id
        if status == "FAILED":
            mlflow.end_run(status="FAILED")
    return run_id


def test_image_comparison_uses_canonical_test_evaluation_run_kind(tmp_path: Path) -> None:
    tracking_uri = _tracking_uri(tmp_path)
    configure_mlflow(experiment_name="comparison-test", tracking_uri=tracking_uri)
    training_id = _run(scope="validation", modality="image")
    test_id = _run(scope="test", parent_training_run_id=training_id, modality="image")

    csv_path, _, count = regenerate_comparison(
        tracking_uri=tracking_uri,
        output_directory=tmp_path / "reports",
    )
    table = pd.read_csv(csv_path)

    assert count == 1
    assert table["run_id"].tolist() == [test_id]


def test_completed_metadata_run_requires_the_current_explicit_contract(tmp_path: Path) -> None:
    tracking_uri = _tracking_uri(tmp_path)
    client = configure_mlflow(experiment_name="comparison-test", tracking_uri=tracking_uri)
    run_id = _run(scope="validation")

    record = require_completed_run(client.get_run(run_id))

    assert record.run_id == run_id
    assert record.run_kind == "training"
    assert record.evaluation_scope == "validation"
    assert record.modality == "metadata"
    assert record.integer_seed() == 42
    assert record.metrics["average_precision"] == 0.4
    assert record.metrics["model_size_mib"] == 2.0


def test_completed_run_rejects_missing_modality(tmp_path: Path) -> None:
    tracking_uri = _tracking_uri(tmp_path)
    client = configure_mlflow(experiment_name="comparison-test", tracking_uri=tracking_uri)
    run_id = _run(scope="validation", modality=None)

    with pytest.raises(ValueError):
        require_completed_run(client.get_run(run_id))


def test_comparison_is_regenerated_from_complete_mlflow_runs(tmp_path: Path) -> None:
    tracking_uri = _tracking_uri(tmp_path)
    configure_mlflow(experiment_name="comparison-test", tracking_uri=tracking_uri)
    validation_id = _run(scope="validation")
    test_id = _run(scope="test", parent_training_run_id=validation_id)

    csv_path, markdown_path, count = regenerate_comparison(
        tracking_uri=tracking_uri,
        output_directory=tmp_path / "reports",
    )

    table = pd.read_csv(csv_path)
    assert count == 2
    assert list(table.columns) == list(COMPARISON_COLUMNS)
    assert set(table["run_id"]) == {validation_id, test_id}
    assert table.loc[table["run_id"] == test_id, "parent_training_run_id"].iloc[0] == validation_id
    assert set(table["model_package_id"]) == {"model-package-test"}
    assert "Experiment comparison" in markdown_path.read_text(encoding="utf-8")


def test_comparison_excludes_failed_and_incomplete_runs(tmp_path: Path) -> None:
    tracking_uri = _tracking_uri(tmp_path)
    configure_mlflow(experiment_name="comparison-test", tracking_uri=tracking_uri)
    included = _run(scope="validation")
    _run(scope="validation", complete=False)
    _run(scope="test", status="FAILED", parent_training_run_id=included)
    _run(scope="validation", omit_metric="validation_brier_score")
    _run(scope="test", parent_training_run_id="")
    _run(scope="test", parent_training_run_id="missing")
    _run(
        scope="test",
        parent_training_run_id=included,
        tag_override=("dataset_bundle_id", "different"),
    )
    _run(
        scope="validation",
        metric_override=("validation_average_precision", math.nan),
    )

    csv_path, _, count = regenerate_comparison(
        tracking_uri=tracking_uri,
        output_directory=tmp_path / "reports",
    )

    table = pd.read_csv(csv_path)
    assert count == 1
    assert table["run_id"].tolist() == [included]


def test_comparison_regeneration_is_deterministic(tmp_path: Path) -> None:
    tracking_uri = _tracking_uri(tmp_path)
    configure_mlflow(experiment_name="comparison-test", tracking_uri=tracking_uri)
    validation_id = _run(scope="validation")
    _run(scope="test", parent_training_run_id=validation_id)
    output = tmp_path / "reports"

    csv_path, _, _ = regenerate_comparison(
        tracking_uri=tracking_uri,
        output_directory=output,
    )
    first = csv_path.read_bytes()
    regenerate_comparison(
        tracking_uri=tracking_uri,
        output_directory=output,
    )

    table = pd.read_csv(csv_path)
    assert table["evaluation_scope"].tolist() == ["test", "validation"]
    assert first == csv_path.read_bytes()


def test_comparison_cli_initializes_sqlite_in_a_fresh_process(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"
    output = tmp_path / "reports"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "radfusion.training.compare",
            "--tracking-uri",
            f"sqlite:///{database.as_posix()}",
            "--output-directory",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Generated 0 rows" in completed.stdout
    assert database.is_file()
    assert (output / "model_comparison_table.csv").is_file()
