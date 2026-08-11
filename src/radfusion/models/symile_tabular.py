"""Fit the two frozen unweighted Symile laboratory baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from radfusion.data.symile_preprocess import SymileLabEcdfTransformer
from radfusion.evaluation.metrics import validated_binary_targets


@dataclass(frozen=True)
class SymileTabularFit:
    """Safely serializable fitted pipeline and optional iteration witness."""

    pipeline: Pipeline
    best_iteration: int | None


def fit_symile_labs_logistic(
    outer_training_features: pd.DataFrame,
    outer_training_targets: np.ndarray,
    *,
    parameters: Mapping[str, object],
    repeat_seed: int,
) -> SymileTabularFit:
    """Fit the frozen LR path on complete outer training."""
    targets = validated_binary_targets(outer_training_targets)
    pipeline = Pipeline(
        [
            ("preprocess", SymileLabEcdfTransformer()),
            (
                "classifier",
                LogisticRegression(
                    l1_ratio=float(parameters["l1_ratio"]),
                    solver=str(parameters["solver"]),
                    C=float(parameters["C"]),
                    max_iter=int(parameters["max_iter"]),
                    class_weight=parameters["class_weight"],
                    random_state=repeat_seed,
                ),
            ),
        ]
    )
    pipeline.fit(outer_training_features, targets)
    return SymileTabularFit(pipeline, None)


def fit_symile_labs_lightgbm(
    outer_training_features: pd.DataFrame,
    outer_training_targets: np.ndarray,
    *,
    parameters: Mapping[str, object],
    inner_training_indices: np.ndarray,
    inner_validation_indices: np.ndarray,
    repeat_seed: int,
) -> SymileTabularFit:
    """Fit the frozen LightGBM path with inner-validation AUROC stopping."""
    targets = validated_binary_targets(outer_training_targets)
    train_indices = _validated_partition_indices(
        inner_training_indices, len(targets), "inner training"
    )
    validation_indices = _validated_partition_indices(
        inner_validation_indices, len(targets), "inner validation"
    )
    if set(train_indices) & set(validation_indices) or set(train_indices) | set(
        validation_indices
    ) != set(range(len(targets))):
        raise ValueError("Inner LightGBM partitions must divide complete outer training")
    preprocessor = SymileLabEcdfTransformer().fit(outer_training_features)
    transformed = preprocessor.transform(outer_training_features)
    classifier = LGBMClassifier(
        objective=str(parameters["objective"]),
        n_estimators=int(parameters["n_estimators"]),
        learning_rate=float(parameters["learning_rate"]),
        num_leaves=int(parameters["num_leaves"]),
        min_child_samples=int(parameters["min_child_samples"]),
        subsample=float(parameters["subsample"]),
        subsample_freq=int(parameters["subsample_freq"]),
        colsample_bytree=float(parameters["colsample_bytree"]),
        reg_lambda=float(parameters["reg_lambda"]),
        class_weight=parameters["class_weight"],
        random_state=repeat_seed,
        bagging_seed=repeat_seed,
        feature_fraction_seed=repeat_seed,
        data_random_seed=repeat_seed,
        drop_seed=repeat_seed,
        extra_seed=repeat_seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        metric="None",
        verbosity=-1,
    )
    classifier.fit(
        transformed[train_indices],
        targets[train_indices],
        eval_set=[(transformed[validation_indices], targets[validation_indices])],
        eval_names=["inner_validation"],
        eval_metric=_roc_auc_metric,
        callbacks=[
            lgb.early_stopping(
                int(parameters["early_stopping_rounds"]),
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = classifier.best_iteration_
    if (
        isinstance(best_iteration, bool)
        or not isinstance(best_iteration, int)
        or best_iteration <= 0
    ):
        raise ValueError("Symile LightGBM did not produce a valid best_iteration")
    return SymileTabularFit(
        Pipeline([("preprocess", preprocessor), ("classifier", classifier)]),
        best_iteration,
    )


def symile_tabular_logits(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return one finite raw logit per row from either fitted lab pipeline."""
    classifier = pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocess")
    if isinstance(classifier, LogisticRegression):
        logits = pipeline.decision_function(features)
    elif isinstance(classifier, LGBMClassifier) and isinstance(
        preprocessor, SymileLabEcdfTransformer
    ):
        transformed = preprocessor.transform(features)
        feature_names = getattr(classifier, "feature_name_", None)
        model_input: np.ndarray | pd.DataFrame = transformed
        if isinstance(feature_names, list) and len(feature_names) == transformed.shape[1]:
            model_input = pd.DataFrame(transformed, columns=feature_names)
        logits = classifier.predict(
            model_input,
            raw_score=True,
            num_iteration=classifier.best_iteration_,
        )
    else:
        raise TypeError("Symile tabular pipeline has an unexpected fitted structure")
    result = np.asarray(logits, dtype=np.float64)
    if result.shape != (len(features),) or not np.isfinite(result).all():
        raise ValueError("Symile tabular model produced invalid logits")
    return result


def _validated_partition_indices(values: object, size: int, name: str) -> np.ndarray:
    indices = np.asarray(values)
    if (
        indices.ndim != 1
        or not len(indices)
        or not np.issubdtype(indices.dtype, np.integer)
        or len(indices) != len(np.unique(indices))
        or (indices < 0).any()
        or (indices >= size).any()
    ):
        raise ValueError(f"Symile {name} indices are invalid")
    return indices.astype(np.int64)


def _roc_auc_metric(targets: np.ndarray, probabilities: np.ndarray) -> tuple[str, float, bool]:
    truth = validated_binary_targets(targets)
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.shape != truth.shape or not np.isfinite(scores).all():
        raise ValueError("Symile LightGBM validation probabilities are invalid")
    return "roc_auc", float(roc_auc_score(truth, scores)), True
