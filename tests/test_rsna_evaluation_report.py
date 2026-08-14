from __future__ import annotations

from pathlib import Path

from radfusion.evaluation.metrics import (
    evaluate_operating_point,
    evaluate_probabilities,
    target_sensitivity_threshold,
)
from radfusion.training.config import load_experiment_config
from radfusion.training.rsna_train_metadata import (
    _write_evaluation_report,
    metrics_document,
)


def test_report_renders_configured_sensitivity_target_and_exact_metric_names(
    tmp_path: Path,
) -> None:
    targets = [0, 0, 1, 1]
    probabilities = [0.1, 0.3, 0.6, 0.9]
    target_threshold = target_sensitivity_threshold(targets, probabilities, sensitivity=0.9)
    probability = evaluate_probabilities(targets, probabilities, calibration_bins=15)
    metrics = evaluate_operating_point(targets, probabilities, threshold=0.5)
    target_metrics = evaluate_operating_point(targets, probabilities, threshold=target_threshold)
    config = load_experiment_config("configs/rsna_metadata_logistic.yaml")
    document = metrics_document(
        scope="validation",
        calibration_bins=15,
        sensitivity_target=config.evaluation.sensitivity_target,
        thresholds={"youden_j": 0.5, "target_sensitivity": target_threshold},
        probability=probability,
        youden=metrics,
        target_sensitivity=target_metrics,
    )
    path = tmp_path / "evaluation_report.md"

    _write_evaluation_report(path, "test_model", document)

    report = path.read_text(encoding="utf-8")
    assert "Average Precision" in report
    assert "PR-AUC" not in report
    assert "Youden-J operating point" in report
    assert "Target-sensitivity operating point" in report
    assert "validation evaluation" in report
