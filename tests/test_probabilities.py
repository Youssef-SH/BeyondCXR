from __future__ import annotations

import numpy as np
import pytest
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from beyondcxr.evaluation.probabilities import (
    canonical_binary_raw_scores,
    positive_class_probabilities,
)


class _Estimator:
    def __init__(self, classes, probabilities) -> None:
        self.classes_ = np.asarray(classes)
        self._probabilities = probabilities

    def predict_proba(self, features):
        return self._probabilities


def test_positive_probability_uses_the_column_labeled_one() -> None:
    estimator = _Estimator([1, 0], [[0.8, 0.2], [0.3, 0.7]])
    np.testing.assert_array_equal(positive_class_probabilities(estimator, ["a", "b"]), [0.8, 0.3])


def test_positive_probability_forwards_lightgbm_best_iteration() -> None:
    class IterationEstimator(_Estimator):
        def __init__(self):
            super().__init__([0, 1], [[0.2, 0.8]])
            self.iteration = None

        def predict_proba(self, features, *, num_iteration=None):
            self.iteration = num_iteration
            return self._probabilities

    estimator = IterationEstimator()
    positive_class_probabilities(estimator, ["a"], best_iteration=17)
    assert estimator.iteration == 17


def test_logistic_raw_scores_are_true_margins_at_probability_boundaries() -> None:
    features = np.asarray([[-1.0], [0.0], [1.0]])
    classifier = LogisticRegression().fit(features, [0, 0, 1])
    classifier.coef_[:] = 1000.0
    classifier.intercept_[:] = 0.0
    pipeline = Pipeline([("preprocess", "passthrough"), ("classifier", classifier)])

    scores = canonical_binary_raw_scores(pipeline, features)

    np.testing.assert_array_equal(scores, [-1000.0, 0.0, 1000.0])
    probabilities = positive_class_probabilities(pipeline, features)
    assert probabilities[0] == 0.0
    assert probabilities[1] == 0.5
    assert probabilities[2] == 1.0


def test_lightgbm_raw_scores_match_selected_iteration_probabilities() -> None:
    features = np.arange(16, dtype=np.float64).reshape(-1, 1)
    pipeline = Pipeline(
        [
            ("preprocess", StandardScaler()),
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=5,
                    min_child_samples=1,
                    num_leaves=4,
                    random_state=42,
                    verbosity=-1,
                    n_jobs=1,
                ),
            ),
        ]
    ).fit(features, np.asarray([0] * 8 + [1] * 8))
    classifier = pipeline.named_steps["classifier"]

    scores = canonical_binary_raw_scores(pipeline, features, best_iteration=3)

    transformed = pipeline.named_steps["preprocess"].transform(features)
    expected = classifier.booster_.predict(transformed, raw_score=True, num_iteration=3)
    np.testing.assert_allclose(scores, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        1.0 / (1.0 + np.exp(-scores)),
        positive_class_probabilities(pipeline, features, best_iteration=3),
        rtol=1e-12,
        atol=1e-15,
    )
    with pytest.raises(ValueError, match="selected best iteration"):
        canonical_binary_raw_scores(pipeline, features)


@pytest.mark.parametrize("classes", [[0], [0, 2], [0, 0], [[0, 1]]])
def test_positive_probability_rejects_invalid_class_contracts(classes) -> None:
    with pytest.raises(ValueError):
        positive_class_probabilities(_Estimator(classes, [[0.5, 0.5]]), ["a"])


@pytest.mark.parametrize(
    "probabilities",
    [
        [[0.5]],
        [0.5, 0.5],
        [],
        [[float("nan"), 0.5]],
        [[-0.1, 1.1]],
        [[0.4, 0.5]],
        [[0.5, 0.5], [0.5, 0.5]],
    ],
)
def test_positive_probability_rejects_invalid_probability_outputs(probabilities) -> None:
    with pytest.raises(ValueError):
        positive_class_probabilities(_Estimator([0, 1], probabilities), ["a"])
