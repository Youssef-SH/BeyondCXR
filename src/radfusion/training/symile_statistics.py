"""Frozen raw-probability statistics for the Symile development and test lifecycles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from radfusion.data.symile_schemas import REPEAT_SEEDS
from radfusion.evaluation.metrics import calibration_coefficients
from radfusion.training.symile_families import FINAL_NEURAL_MEMBER_SEEDS

METRIC_POLICY = {
    "primary_comparison": {"candidate": "cxr_labs_gated", "comparator": "cxr_densenet"},
    "primary_metric": "roc_auc",
    "secondary_metrics": ["average_precision", "brier_score"],
    "effect": "candidate_minus_comparator",
    "metric_direction": {
        "roc_auc": "higher_is_better",
        "average_precision": "higher_is_better",
        "brier_score": "lower_is_better",
    },
}
BOOTSTRAP_POLICY = {"resamples": 2_000, "seed": 2026, "maximum_attempts": 20_000, "ci": 0.95}
SUBGROUP_POLICY = {
    "metric_policy": METRIC_POLICY,
    "minimum_samples": 100,
    "minimum_positives": 20,
    "minimum_negatives": 20,
    "age": [(18, 49), (50, 64), (65, 79), (80, None)],
    "sex": ["F", "M"],
    "view_position": ["AP", "PA"],
    "observed_lab_count": [(1, 29), (30, 34), (35, 40), (41, 50)],
}

ERROR_REVIEW_POLICY = {
    "case_types": ["false_positive", "false_negative"],
    "limit_per_type": 10,
    "false_positive_order": ["probability_descending", "sample_id_ascending"],
    "false_negative_order": ["probability_ascending", "sample_id_ascending"],
}


def headline_metric_names() -> tuple[str, ...]:
    """Derive headline metric membership from the single canonical policy."""
    return (
        str(METRIC_POLICY["primary_metric"]),
        *(str(name) for name in METRIC_POLICY["secondary_metrics"]),
    )


def sigmoid(logits: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Logits must be one-dimensional and finite")
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1 + negative_exp)
    return result


def raw_probability_metrics(
    targets: Sequence[int], probabilities: Sequence[float]
) -> dict[str, float]:
    truth, scores = _validated(targets, probabilities)
    slope, intercept = calibration_coefficients(truth, scores)
    return {
        "roc_auc": float(roc_auc_score(truth, scores)),
        "average_precision": float(average_precision_score(truth, scores)),
        "brier_score": float(brier_score_loss(truth, scores)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def metrics(targets: Sequence[int], probabilities: Sequence[float]) -> dict[str, float]:
    """Return the three headline raw metrics."""
    truth, scores = _validated(targets, probabilities)
    return {
        "roc_auc": float(roc_auc_score(truth, scores)),
        "average_precision": float(average_precision_score(truth, scores)),
        "brier_score": float(brier_score_loss(truth, scores)),
    }


def development_mean_logit_ensemble(frame: pd.DataFrame) -> pd.DataFrame:
    return _mean_logit(frame, "repeat_seed", REPEAT_SEEDS)


def final_member_mean_logit_ensemble(frame: pd.DataFrame) -> pd.DataFrame:
    return _mean_logit(frame, "member_seed", FINAL_NEURAL_MEMBER_SEEDS)


def _mean_logit(
    frame: pd.DataFrame, coordinate: str, expected_seeds: Sequence[int]
) -> pd.DataFrame:
    required = {"sample_id", "target", "logit", coordinate}
    if set(frame.columns) != required:
        raise ValueError("Mean-logit input schema is invalid")
    expected = set(expected_seeds)
    if set(frame[coordinate].unique()) != expected:
        raise ValueError(f"Mean-logit ensemble requires seeds {tuple(expected_seeds)}")
    targets = frame.groupby("sample_id")["target"].nunique()
    counts = frame.groupby("sample_id")[coordinate].nunique()
    if (targets != 1).any() or (counts != len(expected_seeds)).any():
        raise ValueError("Mean-logit members are not exactly aligned")
    result = (
        frame.groupby("sample_id", sort=True)
        .agg(target=("target", "first"), logit=("logit", "mean"))
        .reset_index()
    )
    result["probability"] = sigmoid(result["logit"].to_numpy())
    return result


def youden_threshold(targets: Sequence[int], probabilities: Sequence[float]) -> float:
    truth, scores = _validated(targets, probabilities)
    fpr, tpr, thresholds = roc_curve(truth, scores, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    best = np.max((tpr - fpr)[finite])
    return float(np.max(thresholds[finite & np.isclose(tpr - fpr, best, rtol=0, atol=0)]))


def sensitivity_threshold(
    targets: Sequence[int], probabilities: Sequence[float], target: float = 0.9
) -> float:
    truth, scores = _validated(targets, probabilities)
    _, tpr, thresholds = roc_curve(truth, scores, drop_intermediate=False)
    eligible = np.isfinite(thresholds) & (tpr >= target)
    if not eligible.any():
        raise ValueError("Target sensitivity is infeasible")
    return float(np.max(thresholds[eligible]))


def operating_point_metrics(
    targets: Sequence[int], probabilities: Sequence[float], threshold: float
) -> dict[str, float | int]:
    truth, scores = _validated(targets, probabilities)
    predicted = scores >= threshold
    tp = int(((truth == 1) & predicted).sum())
    tn = int(((truth == 0) & ~predicted).sum())
    fp = int(((truth == 0) & predicted).sum())
    fn = int(((truth == 1) & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": 2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def cluster_bootstrap_effect(
    frame: pd.DataFrame,
    *,
    candidate: str,
    comparator: str,
    metric: str,
    policy: Mapping[str, object] = BOOTSTRAP_POLICY,
) -> dict[str, float | int]:
    resamples, seed, maximum_attempts, ci = _validated_bootstrap_policy(policy)
    required = {"sample_id", "subject_id", "target", candidate, comparator}
    if not required <= set(frame):
        raise ValueError("Bootstrap input schema is invalid")
    subjects = np.asarray(sorted(frame["subject_id"].unique()))
    rng = np.random.default_rng(seed)
    accepted: list[float] = []
    attempts = 0
    while len(accepted) < resamples and attempts < maximum_attempts:
        attempts += 1
        draw = rng.choice(subjects, len(subjects), replace=True)
        sampled = pd.concat(
            [frame.loc[frame["subject_id"] == subject] for subject in draw], ignore_index=True
        )
        if sampled["target"].nunique() < 2:
            continue
        left = _metric(sampled["target"], sampled[candidate], metric)
        right = _metric(sampled["target"], sampled[comparator], metric)
        accepted.append(left - right)
    if len(accepted) != resamples:
        raise ValueError("Bootstrap could not obtain the required accepted draws")
    values = np.asarray(accepted)
    point = _metric(frame["target"], frame[candidate], metric) - _metric(
        frame["target"], frame[comparator], metric
    )
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha], method="linear")
    return {
        "point": point,
        "lower": float(low),
        "upper": float(high),
        "accepted": len(accepted),
        "attempts": attempts,
    }


def _validated_bootstrap_policy(policy: Mapping[str, object]) -> tuple[int, int, int, float]:
    if not isinstance(policy, Mapping) or set(policy) != {
        "resamples",
        "seed",
        "maximum_attempts",
        "ci",
    }:
        raise ValueError("Bootstrap policy is invalid")
    resamples = policy["resamples"]
    seed = policy["seed"]
    maximum_attempts = policy["maximum_attempts"]
    ci = policy["ci"]
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2**32 - 1
        or isinstance(maximum_attempts, bool)
        or not isinstance(maximum_attempts, int)
        or maximum_attempts < resamples
        or isinstance(ci, bool)
        or not isinstance(ci, Real)
        or not np.isfinite(ci)
        or not 0.0 < float(ci) < 1.0
    ):
        raise ValueError("Bootstrap policy is invalid")
    return resamples, seed, maximum_attempts, float(ci)


def deterministic_error_cases(frame: pd.DataFrame, threshold: float) -> dict[str, list[str]]:
    required = {"sample_id", "target", "probability"}
    if set(frame.columns) != required:
        raise ValueError("Error-review input schema is invalid")
    predicted = frame["probability"] >= threshold
    selections = {
        "false_positive": frame[(frame["target"] == 0) & predicted],
        "false_negative": frame[(frame["target"] == 1) & ~predicted],
    }
    limit = int(ERROR_REVIEW_POLICY["limit_per_type"])
    return {
        case_type: selections[case_type]
        .sort_values(
            ["probability", "sample_id"],
            ascending=[
                ERROR_REVIEW_POLICY[f"{case_type}_order"][0] == "probability_ascending",
                ERROR_REVIEW_POLICY[f"{case_type}_order"][1] == "sample_id_ascending",
            ],
        )["sample_id"]
        .head(limit)
        .tolist()
        for case_type in ERROR_REVIEW_POLICY["case_types"]
    }


def focused_development_subgroups(
    cxr: pd.DataFrame,
    gated: pd.DataFrame,
    attributes: pd.DataFrame,
) -> dict[str, object]:
    """Derive repeat-specific and mean-logit CXR-versus-gated development strata."""
    required = {"sample_id", "age_years", "sex", "view_position", "observed_lab_count"}
    if set(attributes.columns) != required:
        raise ValueError("Development subgroup attributes are invalid")
    prediction_fields = {"sample_id", "target", "logit", "repeat_seed"}
    if set(cxr.columns) != prediction_fields or set(gated.columns) != prediction_fields:
        raise ValueError("Development subgroup predictions must contain exact repeat OOF logits")
    paired_repeats = cxr.merge(
        gated,
        on=["sample_id", "target", "repeat_seed"],
        suffixes=("_cxr", "_gated"),
        validate="one_to_one",
    )
    expected_seeds = set(REPEAT_SEEDS)
    if (
        set(paired_repeats["repeat_seed"].unique()) != expected_seeds
        or paired_repeats.groupby("sample_id")["repeat_seed"].nunique().ne(len(REPEAT_SEEDS)).any()
        or paired_repeats.groupby("sample_id")["target"].nunique().ne(1).any()
    ):
        raise ValueError("Development subgroup repeat predictions are not exactly aligned")
    cxr_ensemble = development_mean_logit_ensemble(cxr)
    gated_ensemble = development_mean_logit_ensemble(gated)
    merged = (
        cxr_ensemble.merge(
            gated_ensemble,
            on=["sample_id", "target"],
            suffixes=("_cxr", "_gated"),
            validate="one_to_one",
        )
        .merge(attributes, on="sample_id", validate="one_to_one")
        .sort_values("sample_id", kind="stable")
        .reset_index(drop=True)
    )
    masks: dict[str, pd.Series] = {}
    for low, high in SUBGROUP_POLICY["age"]:
        mask = merged["age_years"] >= low
        if high is not None:
            mask &= merged["age_years"] <= high
        masks[f"age:{low}-{high or 'plus'}"] = mask
    for value in SUBGROUP_POLICY["sex"]:
        masks[f"sex:{value}"] = merged["sex"] == value
    for value in SUBGROUP_POLICY["view_position"]:
        masks[f"view:{value}"] = merged["view_position"] == value
    for low, high in SUBGROUP_POLICY["observed_lab_count"]:
        masks[f"observed_labs:{low}-{high}"] = merged["observed_lab_count"].between(low, high)
    result = {}
    for name, mask in masks.items():
        group = merged.loc[mask]
        positives = int(group["target"].sum())
        negatives = len(group) - positives
        if (
            len(group) < int(SUBGROUP_POLICY["minimum_samples"])
            or positives < int(SUBGROUP_POLICY["minimum_positives"])
            or negatives < int(SUBGROUP_POLICY["minimum_negatives"])
        ):
            result[name] = {"supported": False, "n": len(group)}
            continue
        result[name] = {
            "supported": True,
            "n": len(group),
            "positives": positives,
            "negatives": negatives,
            "repeat_metrics": {
                str(seed): _subgroup_repeat_metrics(
                    paired_repeats,
                    sample_ids=set(group["sample_id"]),
                    repeat_seed=seed,
                )
                for seed in REPEAT_SEEDS
            },
            "mean_logit_ensemble": {
                "cxr": metrics(group["target"], group["probability_cxr"]),
                "gated": metrics(group["target"], group["probability_gated"]),
            },
        }
    return {"policy": SUBGROUP_POLICY, "strata": result}


def _subgroup_repeat_metrics(
    paired: pd.DataFrame, *, sample_ids: set[str], repeat_seed: int
) -> dict[str, object]:
    rows = paired.loc[(paired["repeat_seed"] == repeat_seed) & paired["sample_id"].isin(sample_ids)]
    if len(rows) != len(sample_ids):
        raise ValueError("Development subgroup repeat predictions are incomplete")
    cxr = metrics(rows["target"], sigmoid(rows["logit_cxr"]))
    gated = metrics(rows["target"], sigmoid(rows["logit_gated"]))
    return {
        "cxr": cxr,
        "gated": gated,
        "gated_minus_cxr": {name: gated[name] - cxr[name] for name in headline_metric_names()},
    }


def _metric(targets: Sequence[int], scores: Sequence[float], name: str) -> float:
    truth, probabilities = _validated(targets, scores)
    if name == "roc_auc":
        return float(roc_auc_score(truth, probabilities))
    if name == "average_precision":
        return float(average_precision_score(truth, probabilities))
    if name == "brier_score":
        return float(brier_score_loss(truth, probabilities))
    raise ValueError(f"Unknown bootstrap metric: {name}")


def _validated(
    targets: Sequence[int], probabilities: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(targets, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if (
        truth.ndim != 1
        or scores.shape != truth.shape
        or set(np.unique(truth)) != {0, 1}
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("Targets and probabilities are invalid")
    return truth, scores
