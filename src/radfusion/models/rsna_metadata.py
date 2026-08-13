"""Define fixed tabular baseline estimators."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from radfusion.data.rsna_metadata_preprocess import build_rsna_preprocessor
from radfusion.evaluation.metrics import validated_binary_targets
from radfusion.training.config import ExperimentConfig
from radfusion.training.rsna_interfaces import ModelFitResult


class MetadataLogisticModel:
    """Fit a configured Logistic Regression metadata pipeline."""

    def fit(
        self,
        config: ExperimentConfig,
        training_seed: int,
        train_features: pd.DataFrame,
        train_targets: np.ndarray,
        validation_features: pd.DataFrame,
        validation_targets: np.ndarray,
    ) -> ModelFitResult:
        """Fit preprocessing and Logistic Regression on training data."""
        del validation_features, validation_targets
        parameters = config.training.parameters
        expected = {"l1_ratio", "solver", "C", "max_iter", "class_weight"}
        if set(parameters) != expected:
            raise ValueError("Logistic Regression parameter field set is invalid")
        validated_targets = validated_binary_targets(train_targets)
        pipeline = Pipeline(
            [
                (
                    "preprocess",
                    build_rsna_preprocessor(str(config.preprocessing["metadata_policy"])),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        **dict(parameters),
                        random_state=training_seed,
                    ),
                ),
            ]
        )
        pipeline.fit(train_features, validated_targets)
        return ModelFitResult(
            pipeline,
            {
                "estimator_random_state": training_seed,
                "resolved_class_weight": parameters["class_weight"],
            },
        )


class MetadataLightgbmModel:
    """Fit a configured LightGBM metadata pipeline with validation stopping."""

    def fit(
        self,
        config: ExperimentConfig,
        training_seed: int,
        train_features: pd.DataFrame,
        train_targets: np.ndarray,
        validation_features: pd.DataFrame,
        validation_targets: np.ndarray,
    ) -> ModelFitResult:
        """Fit train-only preprocessing and validation-monitored LightGBM."""
        validated_train_targets = validated_binary_targets(train_targets)
        validated_validation_targets = validated_binary_targets(validation_targets)
        preprocessor = build_rsna_preprocessor(str(config.preprocessing["metadata_policy"]))
        transformed_train = preprocessor.fit_transform(train_features, validated_train_targets)
        transformed_validation = preprocessor.transform(validation_features)
        negatives = int((validated_train_targets == 0).sum())
        positives = int((validated_train_targets == 1).sum())
        scale_pos_weight = negatives / positives
        if not np.isfinite(scale_pos_weight) or scale_pos_weight <= 0:
            raise ValueError("Derived LightGBM positive-class weight must be finite and positive")
        parameters = {**dict(config.family.parameters), **dict(config.training.parameters)}
        expected = {
            "objective",
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "min_child_samples",
            "subsample",
            "subsample_freq",
            "colsample_bytree",
            "reg_lambda",
            "class_weighting",
            "early_stopping_rounds",
        }
        if set(parameters) != expected:
            raise ValueError("LightGBM parameter field set is invalid")
        early_stopping_rounds = int(parameters.pop("early_stopping_rounds"))
        class_weighting = str(parameters.pop("class_weighting"))
        if class_weighting != "train_neg_pos_ratio":
            raise ValueError("LightGBM requires train_neg_pos_ratio class weighting")
        classifier = LGBMClassifier(
            **parameters,
            random_state=training_seed,
            bagging_seed=training_seed,
            feature_fraction_seed=training_seed,
            data_random_seed=training_seed,
            drop_seed=training_seed,
            extra_seed=training_seed,
            scale_pos_weight=scale_pos_weight,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            metric="None",
            verbosity=-1,
        )
        if config.training.selection_metric != "average_precision":
            raise ValueError("RSNA LightGBM selection requires average_precision")
        classifier.fit(
            transformed_train,
            validated_train_targets,
            eval_set=[(transformed_validation, validated_validation_targets)],
            eval_names=["validation"],
            eval_metric=_average_precision_metric,
            callbacks=[
                lgb.early_stopping(
                    early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                ),
                lgb.log_evaluation(period=0),
            ],
        )
        pipeline = Pipeline([("preprocess", preprocessor), ("classifier", classifier)])
        direct_probabilities = classifier.predict_proba(
            transformed_validation, num_iteration=classifier.best_iteration_
        )
        assembled_probabilities = pipeline.predict_proba(
            validation_features, num_iteration=classifier.best_iteration_
        )
        if not np.array_equal(direct_probabilities, assembled_probabilities):
            raise RuntimeError("Assembled LightGBM pipeline changed fitted probabilities")
        return ModelFitResult(
            pipeline,
            {
                "scale_pos_weight": scale_pos_weight,
                "early_stopping_rounds": early_stopping_rounds,
                "early_stopping_metric": config.training.selection_metric,
                "early_stopping_dataset": "validation",
                "early_stopping_greater_is_better": True,
                "best_iteration": classifier.best_iteration_,
                "estimator_random_state": training_seed,
                "lightgbm_deterministic": True,
                "lightgbm_force_col_wise": True,
                "lightgbm_n_jobs": 1,
                "lightgbm_verbosity": -1,
            },
        )


def _average_precision_metric(
    targets: np.ndarray, probabilities: np.ndarray
) -> tuple[str, float, bool]:
    """Return LightGBM's sole validation metric with an explicit optimization direction."""
    truth = validated_binary_targets(targets)
    scores = np.asarray(probabilities, dtype=np.float64)
    if (
        scores.ndim != 1
        or len(scores) != len(truth)
        or not np.isfinite(scores).all()
        or ((scores < 0.0) | (scores > 1.0)).any()
    ):
        raise ValueError("LightGBM validation probabilities are invalid")
    return "average_precision", float(average_precision_score(truth, scores)), True
