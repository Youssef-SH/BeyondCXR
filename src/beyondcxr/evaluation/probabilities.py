"""Validate fitted class contracts and extract positive-class probabilities."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class ProbabilityEstimator(Protocol):
    """Fitted estimator that exposes class labels and class probabilities."""

    classes_: Any

    def predict_proba(self, features: Any) -> Any:
        """Return class probabilities."""


def positive_class_probabilities(
    estimator: ProbabilityEstimator,
    features: Any,
    *,
    best_iteration: int | None = None,
) -> np.ndarray:
    """Return validated probabilities for the exact positive label ``1``."""
    classes = np.asarray(getattr(estimator, "classes_", None))
    if classes.ndim != 1 or len(classes) != 2 or len(np.unique(classes)) != 2:
        raise ValueError("Estimator must expose two unique one-dimensional class labels")
    if set(classes.tolist()) != {0, 1}:
        raise ValueError("Estimator classes must be exactly {0, 1}")
    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) != 1:
        raise ValueError("Estimator must expose the positive class label 1 exactly once")
    if best_iteration is None:
        raw_probabilities = estimator.predict_proba(features)
    elif hasattr(estimator, "named_steps"):
        transformed = estimator.named_steps["preprocess"].transform(features)
        classifier = estimator.named_steps["classifier"]
        if isinstance(classifier, LGBMClassifier):
            positive = np.asarray(
                classifier.booster_.predict(
                    transformed,
                    num_iteration=best_iteration,
                ),
                dtype=np.float64,
            )
            raw_probabilities = np.column_stack((1.0 - positive, positive))
        else:
            raw_probabilities = classifier.predict_proba(
                transformed,
                num_iteration=best_iteration,
            )
    else:
        raw_probabilities = estimator.predict_proba(features, num_iteration=best_iteration)
    probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] == 0
        or probabilities.shape[1] != len(classes)
    ):
        raise ValueError("predict_proba returned an invalid class-probability matrix")
    try:
        feature_count = len(features)
    except TypeError:
        feature_count = None
    if feature_count is not None and probabilities.shape[0] != feature_count:
        raise ValueError("Class-probability row count does not match the input")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Class probabilities must be finite and within [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Class-probability rows must sum to 1")
    positive = probabilities[:, int(positive_columns[0])]
    return positive


def canonical_binary_raw_scores(
    estimator: object,
    features: Any,
    *,
    best_iteration: int | None = None,
) -> np.ndarray:
    """Return true fitted-model margins and verify their probability interpretation."""
    if not isinstance(estimator, Pipeline):
        raise TypeError("Canonical tabular raw scores require a fitted pipeline")
    classifier = estimator.named_steps.get("classifier")
    if isinstance(classifier, LogisticRegression):
        scores = estimator.decision_function(features)
    elif isinstance(classifier, LGBMClassifier):
        if best_iteration is None:
            raise ValueError("LightGBM raw scores require the selected best iteration")
        transformed = estimator.named_steps["preprocess"].transform(features)
        scores = classifier.booster_.predict(
            transformed,
            raw_score=True,
            num_iteration=best_iteration,
        )
    else:
        raise TypeError("Fitted tabular classifier does not expose a supported raw score")
    result = np.asarray(scores, dtype=np.float64).reshape(-1)
    if result.shape != (len(features),) or not np.isfinite(result).all():
        raise ValueError("Fitted tabular classifier produced invalid raw scores")
    derived = _stable_sigmoid(result)
    probabilities = positive_class_probabilities(
        estimator,
        features,
        best_iteration=best_iteration,
    )
    if not np.allclose(derived, probabilities, rtol=1e-12, atol=1e-15):
        raise ValueError("Raw scores disagree with fitted classifier probabilities")
    return result


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result
