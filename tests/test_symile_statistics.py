from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import beyondcxr.training.symile_campaign_control as campaign_control
import beyondcxr.training.symile_ecg_extension_result as extension_result
import beyondcxr.training.symile_statistics as symile_statistics
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.symile_statistics import (
    cluster_bootstrap_effect,
    deterministic_error_cases,
    focused_development_subgroups,
    raw_probability_metrics,
    sensitivity_threshold,
    youden_threshold,
)


def test_subgroup_publication_is_atomic_idempotent_and_rejects_conflicts(tmp_path) -> None:
    summary = {"policy": {}, "strata": {}}

    def publish():
        return extension_result.publish_focused_subgroup_derivative(
            report_root=tmp_path, summary=summary
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        identities = list(executor.map(lambda _: publish(), range(4)))
    assert len(set(identities)) == 1
    path = tmp_path / "development-subgroups" / f"{identities[0]}.json"
    assert extension_result.validate_focused_subgroup_derivative(path) == {
        "focused_subgroup_derivative_schema_version": 1,
        **summary,
    }
    inode = path.stat().st_ino
    assert publish() == identities[0]
    assert path.stat().st_ino == inode
    path.write_bytes(b"conflicting content")
    with pytest.raises(ManifestBuildError, match="conflicts"):
        publish()
    assert path.read_bytes() == b"conflicting content"
    assert path.stat().st_ino == inode
    assert list(path.parent.iterdir()) == [path]


def test_sigmoid_handles_extreme_finite_logits_without_changing_ordinary_results() -> None:
    ordinary = np.linspace(-20, 20, 101)
    expected = np.where(
        ordinary >= 0, 1 / (1 + np.exp(-ordinary)), np.exp(ordinary) / (1 + np.exp(ordinary))
    )
    np.testing.assert_array_equal(symile_statistics.sigmoid(ordinary), expected)
    np.testing.assert_array_equal(symile_statistics.sigmoid([-1000, 0, 1000]), [0, 0.5, 1])


def test_raw_metrics_threshold_ties_and_primary_error_ranking() -> None:
    target = [0, 0, 1, 1]
    probability = [0.1, 0.8, 0.4, 0.9]
    assert set(raw_probability_metrics(target, probability)) == {
        "roc_auc",
        "average_precision",
        "brier_score",
        "calibration_slope",
        "calibration_intercept",
    }
    assert youden_threshold(target, probability) == 0.9
    assert sensitivity_threshold(target, probability, target=0.5) == 0.9
    cases = deterministic_error_cases(
        pd.DataFrame(
            {
                "sample_id": ["a", "b", "c", "d"],
                "target": target,
                "probability": probability,
            }
        ),
        0.5,
    )
    assert cases == {"false_positive": ["b"], "false_negative": ["c"]}


def test_error_case_selection_consumes_published_ranking_policy(monkeypatch) -> None:
    monkeypatch.setitem(symile_statistics.ERROR_REVIEW_POLICY, "limit_per_type", 1)
    monkeypatch.setitem(
        symile_statistics.ERROR_REVIEW_POLICY,
        "false_positive_order",
        ["probability_ascending", "sample_id_ascending"],
    )
    frame = pd.DataFrame(
        {
            "sample_id": ["high", "low", "miss"],
            "target": [0, 0, 1],
            "probability": [0.9, 0.6, 0.1],
        }
    )

    assert deterministic_error_cases(frame, 0.5) == {
        "false_positive": ["low"],
        "false_negative": ["miss"],
    }


def test_subject_cluster_bootstrap_uses_accepted_draw_count() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "subject_id": [1, 1, 2, 3],
            "target": [0, 0, 1, 1],
            "candidate": [0.1, 0.2, 0.8, 0.9],
            "comparator": [0.2, 0.3, 0.6, 0.7],
        }
    )
    result = cluster_bootstrap_effect(
        frame,
        candidate="candidate",
        comparator="comparator",
        metric="brier_score",
        policy={"resamples": 25, "seed": 2026, "maximum_attempts": 250, "ci": 0.95},
    )
    assert result["accepted"] == 25
    assert result["attempts"] >= 25


def test_subject_cluster_bootstrap_does_not_fit_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        symile_statistics,
        "calibration_coefficients",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bootstrap attempted calibration fitting")
        ),
    )
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "subject_id": [1, 2, 3, 4],
            "target": [0, 0, 1, 1],
            "candidate": [0.1, 0.2, 0.8, 0.9],
            "comparator": [0.2, 0.3, 0.7, 0.8],
        }
    )
    for metric in symile_statistics.headline_metric_names():
        cluster_bootstrap_effect(
            frame,
            candidate="candidate",
            comparator="comparator",
            metric=metric,
            policy={"resamples": 10, "seed": 2026, "maximum_attempts": 100, "ci": 0.95},
        )


def test_bootstrap_policy_drives_linear_ci_and_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "subject_id": [1, 2, 3, 4],
            "target": [0, 0, 1, 1],
            "candidate": [0.1, 0.2, 0.8, 0.9],
            "comparator": [0.2, 0.3, 0.7, 0.8],
        }
    )
    observed: dict[str, object] = {}
    quantile = np.quantile

    def capture(values: np.ndarray, q: list[float], *, method: str) -> np.ndarray:
        observed.update(q=q, method=method)
        return quantile(values, q, method=method)

    monkeypatch.setattr(symile_statistics.np, "quantile", capture)
    policy = {"resamples": 5, "seed": 2026, "maximum_attempts": 50, "ci": 0.95}
    result = cluster_bootstrap_effect(
        frame,
        candidate="candidate",
        comparator="comparator",
        metric="brier_score",
        policy=policy,
    )
    assert observed == {"q": pytest.approx([0.025, 0.975]), "method": "linear"}
    assert result["accepted"] == 5

    invalid = (
        {**policy, "ci": 0.0},
        {**policy, "ci": 1.0},
        {**policy, "ci": float("nan")},
        {**policy, "ci": True},
        {**policy, "resamples": True},
        {**policy, "seed": True},
        {**policy, "maximum_attempts": True},
        {**policy, "maximum_attempts": 4},
    )
    for candidate_policy in invalid:
        with pytest.raises(ValueError, match="Bootstrap policy is invalid"):
            cluster_bootstrap_effect(
                frame,
                candidate="candidate",
                comparator="comparator",
                metric="brier_score",
                policy=candidate_policy,
            )


def test_focused_subgroups_keep_three_repeat_estimates_separate_from_ensemble() -> None:
    sample_ids = [f"symile:{index}" for index in range(100)]
    targets = [index % 2 for index in range(100)]
    cxr_rows = []
    gated_rows = []
    for seed, offset in ((17, -0.2), (42, 0.0), (2026, 0.2)):
        for sample_id, target in zip(sample_ids, targets, strict=True):
            base = 1.0 if target else -1.0
            cxr_rows.append(
                {
                    "sample_id": sample_id,
                    "target": target,
                    "logit": base + offset,
                    "repeat_seed": seed,
                }
            )
            gated_rows.append(
                {
                    "sample_id": sample_id,
                    "target": target,
                    "logit": 1.5 * base + offset,
                    "repeat_seed": seed,
                }
            )
    attributes = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "age_years": [30] * 100,
            "sex": ["F"] * 50 + ["M"] * 50,
            "view_position": ["AP"] * 50 + ["PA"] * 50,
            "observed_lab_count": [25] * 100,
        }
    )

    result = focused_development_subgroups(
        pd.DataFrame(cxr_rows), pd.DataFrame(gated_rows), attributes
    )

    supported = result["strata"]["age:18-49"]
    assert supported["supported"] is True
    assert set(supported["repeat_metrics"]) == {"17", "42", "2026"}
    assert set(supported["mean_logit_ensemble"]) == {"cxr", "gated"}
    assert supported["n"] == 100


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_focused_subgroups_require_exact_attribute_membership(mutation: str) -> None:
    sample_ids = [f"symile:{index}" for index in range(2)]
    rows = [
        {"sample_id": sample_id, "target": index, "logit": float(index), "repeat_seed": seed}
        for seed in (17, 42, 2026)
        for index, sample_id in enumerate(sample_ids)
    ]
    attributes = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "age_years": [30, 60],
            "sex": ["F", "M"],
            "view_position": ["AP", "PA"],
            "observed_lab_count": [25, 35],
        }
    )
    if mutation == "missing":
        attributes = attributes.iloc[:1]
    elif mutation == "extra":
        attributes = pd.concat(
            [attributes, attributes.iloc[[0]].assign(sample_id="symile:extra")],
            ignore_index=True,
        )
    else:
        attributes = pd.concat([attributes, attributes.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="not exactly aligned"):
        focused_development_subgroups(pd.DataFrame(rows), pd.DataFrame(rows), attributes)


def test_global_reliability_plot_is_raw_deterministic_and_exact_six() -> None:
    targets = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.1, 0.3, 0.7, 0.9])
    curve = campaign_control._reliability_curve(targets, probabilities)
    curves = {
        family: curve
        for family in (
            "labs_logistic",
            "labs_lightgbm",
            "cxr_densenet",
            "cxr_labs_concat",
            "cxr_labs_gated",
            "cxr_labs_ecg_gated",
        )
    }

    first = campaign_control._reliability_svg(curves)
    second = campaign_control._reliability_svg(curves)

    assert first == second
    assert first.count("<polyline") == 6
    assert "Raw reliability curves (descriptive)" in first
    with pytest.raises(ManifestBuildError, match="membership"):
        campaign_control._reliability_svg({"labs_logistic": curve})


def test_reliability_computation_consumes_published_policy(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def calibration(targets, probabilities, *, n_bins, strategy):
        del targets, probabilities
        observed.update(n_bins=n_bins, strategy=strategy)
        return np.asarray([0.25]), np.asarray([0.2])

    monkeypatch.setitem(campaign_control.RELIABILITY_POLICY, "bins", 7)
    monkeypatch.setitem(campaign_control.RELIABILITY_POLICY, "strategy", "quantile")
    monkeypatch.setattr(campaign_control, "calibration_curve", calibration)

    campaign_control._reliability_curve(pd.Series([0, 1]), pd.Series([0.2, 0.8]))

    assert observed == {"n_bins": 7, "strategy": "quantile"}
    assert campaign_control._global_evaluation_policy()["reliability_plot"] == {
        "bins": 7,
        "strategy": "quantile",
        "role": "descriptive",
    }


def test_global_effect_computation_and_metadata_share_comparison_authority(monkeypatch) -> None:
    rows = pd.DataFrame({"sample_id": ["a", "b"], "target": [0, 1], "probability": [0.2, 0.8]})
    candidate = rows.assign(probability=[0.1, 0.9])
    comparator = rows.assign(probability=[0.4, 0.6])
    views = {
        "candidate": candidate,
        "comparator": comparator,
        "cxr_labs_gated": rows,
        "labs_logistic": rows,
    }
    monkeypatch.setattr(
        campaign_control,
        "GLOBAL_EFFECT_COMPARISONS",
        (("only_effect", "candidate", "comparator"),),
    )
    monkeypatch.setattr(campaign_control, "_predictor_views", lambda packages, predictions: views)
    monkeypatch.setattr(
        campaign_control,
        "_validated_projection_subjects",
        lambda capability, projection, reference: pd.DataFrame(
            {"sample_id": ["a", "b"], "subject_id": [1, 2]}
        ),
    )
    captured: list[tuple[float, float]] = []

    def effect(frame, **kwargs):
        del kwargs
        captured.append((float(frame["candidate"].iloc[0]), float(frame["comparator"].iloc[0])))
        return {"point": 0.0, "lower": 0.0, "upper": 0.0, "accepted": 1, "attempts": 1}

    monkeypatch.setattr(campaign_control, "cluster_bootstrap_effect", effect)
    monkeypatch.setattr(campaign_control, "raw_probability_metrics", lambda *args: {})
    monkeypatch.setattr(campaign_control, "_reliability_curve", lambda *args: {})
    monkeypatch.setattr(campaign_control, "operating_point_metrics", lambda *args: {})

    claims = campaign_control._derive_global_claims(
        SimpleNamespace(manifest={"primary_thresholds": {"youden_j": 0.5}}), (), (), object()
    )

    assert campaign_control._global_evaluation_policy()["paired_effects"] == ["only_effect"]
    assert set(claims["paired_effects"]) == {"only_effect"}
    assert captured == [(0.1, 0.4)] * 3
