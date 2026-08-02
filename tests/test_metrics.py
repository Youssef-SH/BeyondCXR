from __future__ import annotations

import numpy as np
import pytest

from radfusion.evaluation.metrics import (
    calibration_coefficients,
    evaluate_binary,
    expected_calibration_error,
    target_sensitivity_threshold,
    validated_binary_arrays,
    youden_j_threshold,
)


def test_binary_metrics_group_probability_and_operating_point_values() -> None:
    metrics = evaluate_binary([0, 0, 1, 1], [0.1, 0.7, 0.4, 0.9], threshold=0.5)

    assert metrics.operating_point.as_dict() == {
        "precision": 0.5,
        "recall": 0.5,
        "specificity": 0.5,
        "f1": 0.5,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }
    assert 0.0 <= metrics.probability.average_precision <= 1.0
    assert 0.0 <= metrics.probability.roc_auc <= 1.0
    assert "expected_calibration_error" in metrics.probability.as_dict()


def test_thresholds_are_derived_from_validation_probabilities() -> None:
    targets = [0, 0, 1, 1]
    probabilities = [0.1, 0.3, 0.6, 0.9]

    assert youden_j_threshold(targets, probabilities) == pytest.approx(0.6)
    assert target_sensitivity_threshold(targets, probabilities, sensitivity=0.9) == pytest.approx(
        0.6
    )


def test_threshold_selection_enumerates_all_roc_candidates(monkeypatch) -> None:
    calls: list[bool] = []

    def fake_roc_curve(truth, scores, *, drop_intermediate):
        calls.append(drop_intermediate)
        return (
            np.asarray([0.0, 0.2, 0.4]),
            np.asarray([0.0, 0.8, 1.0]),
            np.asarray([float("inf"), 0.7, 0.4]),
        )

    monkeypatch.setattr("radfusion.evaluation.metrics.roc_curve", fake_roc_curve)
    assert youden_j_threshold([0, 1], [0.1, 0.9]) == pytest.approx(0.7)
    assert target_sensitivity_threshold([0, 1], [0.1, 0.9], sensitivity=0.8) == pytest.approx(0.7)
    assert calls == [False, False]


def test_youden_ties_select_the_highest_finite_threshold(monkeypatch) -> None:
    monkeypatch.setattr(
        "radfusion.evaluation.metrics.roc_curve",
        lambda *args, **kwargs: (
            np.asarray([0.0, 0.25, 0.5]),
            np.asarray([0.0, 0.75, 1.0]),
            np.asarray([float("inf"), 0.8, 0.6]),
        ),
    )
    assert youden_j_threshold([0, 1], [0.1, 0.9]) == pytest.approx(0.8)


@pytest.mark.parametrize(
    "targets",
    [
        [0, 0],
        [0.0, 0.5, 1.0],
        [0, 256, 1],
        [0, float("nan"), 1],
        [0, float("inf"), 1],
        ["0", "1"],
    ],
)
def test_binary_target_validation_rejects_malformed_values(targets: list[object]) -> None:
    with pytest.raises(ValueError):
        validated_binary_arrays(targets, np.linspace(0.1, 0.9, len(targets)))


@pytest.mark.parametrize("targets", [[False, True], [0, 1], [0.0, 1.0]])
def test_binary_target_validation_accepts_exact_boolean_and_numeric_values(targets) -> None:
    truth, scores = validated_binary_arrays(targets, [0.1, 0.9])
    np.testing.assert_array_equal(truth, [0, 1])
    np.testing.assert_array_equal(scores, [0.1, 0.9])


def test_binary_target_validation_rejects_shapes_lengths_and_probability_bounds() -> None:
    with pytest.raises(ValueError):
        validated_binary_arrays([[0], [1]], [0.1, 0.9])
    with pytest.raises(ValueError):
        validated_binary_arrays([0, 1], [[0.1], [0.9]])
    with pytest.raises(ValueError):
        validated_binary_arrays([0, 1], [0.1])
    with pytest.raises(ValueError):
        validated_binary_arrays([0, 1], [-0.1, 1.1])
    with pytest.raises(ValueError):
        validated_binary_arrays([0, 1], [0.1, float("nan")])


def test_ece_equal_width_reference_includes_edges_and_skips_empty_bins() -> None:
    assert expected_calibration_error([0, 0, 1, 1], [0.0, 0.2, 0.8, 1.0], bins=5) == pytest.approx(
        0.1
    )
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [0.1, 0.9], bins=0)


def test_calibration_coefficients_reference_and_clipping() -> None:
    slope, intercept = calibration_coefficients([0, 0, 1, 1], [0.0, 0.2, 0.8, 1.0])
    assert slope == pytest.approx(6.498093334322061)
    assert intercept == pytest.approx(0.0, abs=1e-12)


def test_configurable_calibration_bins_keep_generic_metric_name() -> None:
    metrics = evaluate_binary(
        [0, 0, 1, 1], [0.1, 0.3, 0.6, 0.9], threshold=0.5, calibration_bins=10
    )
    assert metrics.probability.expected_calibration_error == pytest.approx(0.225)
