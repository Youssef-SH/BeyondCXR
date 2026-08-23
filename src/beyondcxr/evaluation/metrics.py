"""Compute reusable binary classification metrics and operating thresholds."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

CALIBRATION_BINNING_STRATEGY = "uniform"
CALIBRATION_LOGIT_CLIP = 1e-6


@dataclass(frozen=True)
class ProbabilityMetrics:
    """Threshold-independent metrics for binary probabilities."""

    average_precision: float
    roc_auc: float
    brier_score: float
    expected_calibration_error: float
    calibration_slope: float
    calibration_intercept: float

    def as_dict(self) -> dict[str, float]:
        """Return metrics as a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class OperatingPointMetrics:
    """Metrics evaluated at one explicit decision threshold."""

    precision: float
    recall: float
    specificity: float
    f1: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def as_dict(self) -> dict[str, float | int]:
        """Return metrics as a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class BinaryMetrics:
    """Probability and operating-point metrics for one binary evaluation."""

    probability: ProbabilityMetrics
    operating_point: OperatingPointMetrics


def evaluate_binary(
    targets: ArrayLike,
    probabilities: ArrayLike,
    *,
    threshold: float,
    calibration_bins: int = 15,
) -> BinaryMetrics:
    """Evaluate binary probabilities at one fixed decision threshold."""
    truth, scores = validated_binary_arrays(targets, probabilities)
    if not math_is_probability(threshold):
        raise ValueError("threshold must be finite and between 0 and 1")
    return BinaryMetrics(
        probability=evaluate_probabilities(truth, scores, calibration_bins=calibration_bins),
        operating_point=evaluate_operating_point(truth, scores, threshold=threshold),
    )


def evaluate_probabilities(
    targets: ArrayLike,
    probabilities: ArrayLike,
    *,
    calibration_bins: int = 15,
) -> ProbabilityMetrics:
    """Compute threshold-independent probability metrics once."""
    truth, scores = validated_binary_arrays(targets, probabilities)
    slope, intercept = calibration_coefficients(truth, scores)
    return ProbabilityMetrics(
        average_precision=float(average_precision_score(truth, scores)),
        roc_auc=float(roc_auc_score(truth, scores)),
        brier_score=float(brier_score_loss(truth, scores)),
        expected_calibration_error=expected_calibration_error(truth, scores, bins=calibration_bins),
        calibration_slope=slope,
        calibration_intercept=intercept,
    )


def evaluate_operating_point(
    targets: ArrayLike,
    probabilities: ArrayLike,
    *,
    threshold: float,
) -> OperatingPointMetrics:
    """Compute confusion-derived metrics at one fixed threshold."""
    truth, scores = validated_binary_arrays(targets, probabilities)
    if not math_is_probability(threshold):
        raise ValueError("threshold must be finite and between 0 and 1")
    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    return OperatingPointMetrics(
        precision=float(precision_score(truth, predictions, zero_division=0)),
        recall=float(recall_score(truth, predictions, zero_division=0)),
        specificity=float(tn / (tn + fp)) if tn + fp else 0.0,
        f1=float(f1_score(truth, predictions, zero_division=0)),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )


def youden_j_threshold(targets: ArrayLike, probabilities: ArrayLike) -> float:
    """Select the threshold maximizing validation Youden's J statistic."""
    truth, scores = validated_binary_arrays(targets, probabilities)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        truth, scores, drop_intermediate=False
    )
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("No finite ROC thresholds are available")
    statistic = true_positive_rate[finite] - false_positive_rate[finite]
    candidates = thresholds[finite]
    maximum = statistic.max()
    return float(candidates[statistic == maximum].max())


def target_sensitivity_threshold(
    targets: ArrayLike, probabilities: ArrayLike, *, sensitivity: float = 0.90
) -> float:
    """Select the highest validation threshold meeting a sensitivity target."""
    if not 0 < sensitivity <= 1:
        raise ValueError("sensitivity must be in (0, 1]")
    truth, scores = validated_binary_arrays(targets, probabilities)
    _, true_positive_rate, thresholds = roc_curve(truth, scores, drop_intermediate=False)
    feasible = thresholds[(true_positive_rate >= sensitivity) & np.isfinite(thresholds)]
    if not len(feasible):
        raise ValueError(f"No threshold reaches sensitivity {sensitivity:.3f}")
    return float(feasible.max())


def expected_calibration_error(
    targets: ArrayLike, probabilities: ArrayLike, *, bins: int = 15
) -> float:
    """Compute expected calibration error with equal-width probability bins."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    truth, scores = validated_binary_arrays(targets, probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            error += selected.mean() * abs(scores[selected].mean() - truth[selected].mean())
    return float(error)


def calibration_coefficients(targets: ArrayLike, probabilities: ArrayLike) -> tuple[float, float]:
    """Fit calibration slope and intercept on clipped probability logits."""
    truth, scores = validated_binary_arrays(targets, probabilities)
    clipped = np.clip(scores, CALIBRATION_LOGIT_CLIP, 1.0 - CALIBRATION_LOGIT_CLIP)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2_000)
    calibration.fit(logits, truth)
    return float(calibration.coef_[0, 0]), float(calibration.intercept_[0])


def validated_binary_arrays(
    targets: ArrayLike, probabilities: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Validate exact binary targets and aligned one-dimensional probabilities."""
    truth = validated_binary_targets(targets)
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("targets and probabilities must be one-dimensional")
    if len(truth) != len(scores):
        raise ValueError("targets and probabilities must have equal non-zero length")
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("probabilities must be finite and between 0 and 1")
    return truth, scores


def validated_binary_targets(targets: ArrayLike) -> np.ndarray:
    """Return exact one-dimensional binary targets without narrowing invalid values."""
    raw_truth = np.asarray(targets)
    if raw_truth.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if not len(raw_truth):
        raise ValueError("targets must have non-zero length")
    if not (
        np.issubdtype(raw_truth.dtype, np.bool_)
        or np.issubdtype(raw_truth.dtype, np.integer)
        or np.issubdtype(raw_truth.dtype, np.floating)
    ):
        raise ValueError("targets must be numeric or Boolean binary values")
    numeric_truth = raw_truth.astype(np.float64, copy=False)
    if not np.isfinite(numeric_truth).all():
        raise ValueError("targets must be finite")
    if not np.equal(numeric_truth, np.floor(numeric_truth)).all():
        raise ValueError("targets must not contain fractional values")
    if set(np.unique(numeric_truth).tolist()) != {0.0, 1.0}:
        raise ValueError("targets must contain exactly both binary classes {0, 1}")
    return numeric_truth.astype(np.int8)


def math_is_probability(value: float) -> bool:
    """Return whether a scalar is a finite probability."""
    return bool(np.isfinite(value) and 0.0 <= value <= 1.0)
