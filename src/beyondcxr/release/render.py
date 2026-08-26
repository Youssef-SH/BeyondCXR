"""Deterministic aggregate public tables and scientific SVG figures."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_schemas import REPEAT_SEEDS
from beyondcxr.release.projection import RELIABILITY_FAMILIES, PublicResultProjection
from beyondcxr.training.symile_families import (
    FINAL_PACKAGE_POLICY,
    SYMILE_CORE_DEVELOPMENT_FAMILIES,
    SYMILE_ECG_GATED_FAMILY,
)
from beyondcxr.utils.privacy import validate_public_reports

RESULT_START = "<!-- BEYONDCXR_RESULTS_START -->"
RESULT_END = "<!-- BEYONDCXR_RESULTS_END -->"
DISPLAY = {
    "labs_logistic": "Labs Logistic Regression",
    "labs_lightgbm": "Labs LightGBM",
    "cxr_densenet": "CXR",
    "cxr_labs_concat": "CXR + labs concat",
    "cxr_labs_gated": "CXR + labs gated",
    "cxr_labs_gated_no_observedness": "CXR + labs gated without observedness",
    "cxr_labs_ecg_gated": "CXR + labs + ECG gated",
}
EFFECT_ORDER = ("concat_vs_cxr", "primary", "gated_vs_concat", "ecg_vs_gated")
EFFECT_DISPLAY = {
    "concat_vs_cxr": "CXR + labs concat − CXR",
    "primary": "CXR + labs gated − CXR — PRIMARY",
    "gated_vs_concat": "CXR + labs gated − CXR + labs concat",
    "ecg_vs_gated": "CXR + labs + ECG gated − CXR + labs gated",
}
COLORS = dict(
    zip(
        FINAL_PACKAGE_POLICY,
        ("#0072B2", "#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00"),
        strict=True,
    )
)
PUBLIC_RESULT_TABLES = (
    "cohort",
    "development_performance",
    "heldout_performance",
    "probability_operating_points",
    "observedness_subgroups",
)
PUBLIC_RESULT_FIGURES = (
    "incremental_effects.svg",
    "discrimination.svg",
    "reliability.svg",
)


def public_result_surface() -> frozenset[str]:
    """Return the canonical files rendered beneath the public Symile result root."""
    return frozenset(
        {
            *(f"data/{name}.csv" for name in PUBLIC_RESULT_TABLES),
            *(f"tables/{name}.md" for name in PUBLIC_RESULT_TABLES),
            *(f"figures/{name}" for name in PUBLIC_RESULT_FIGURES),
        }
    )


@dataclass(frozen=True)
class _Table:
    caption: str
    note: str
    columns: tuple[tuple[str, str], ...]
    rows: tuple[Mapping[str, Any], ...]


def render_public_results(projection: PublicResultProjection, output: str | Path) -> dict[str, str]:
    """Render the public aggregate result tables and figures."""
    root = Path(output)
    if root.is_symlink() or (root.exists() and any(root.iterdir())):
        raise ManifestBuildError("Public result destination must be absent or empty")
    for name in ("data", "tables", "figures"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for name, table in zip(PUBLIC_RESULT_TABLES, _tables(projection), strict=True):
        _write_csv(root / "data" / f"{name}.csv", table)
        (root / "tables" / f"{name}.md").write_text(_markdown_table(table), encoding="utf-8")
    with matplotlib.rc_context(
        {"svg.hashsalt": "beyondcxr-release-v1", "font.family": "DejaVu Sans"}
    ):
        _incremental_effects(projection, root / "figures" / PUBLIC_RESULT_FIGURES[0])
        _discrimination(projection, root / "figures" / PUBLIC_RESULT_FIGURES[1])
        _reliability(projection, root / "figures" / PUBLIC_RESULT_FIGURES[2])
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    if {path.relative_to(root).as_posix() for path in paths} != public_result_surface():
        raise ManifestBuildError("Rendered public result membership is invalid")
    validate_public_reports(paths, forbidden_source_values=())
    return {path.relative_to(root).as_posix(): _sha256(path) for path in paths}


def replace_result_region(text: str, replacement: str) -> str:
    before, _, after = parse_result_region(text)
    return f"{before}{RESULT_START}\n{replacement.rstrip()}\n{RESULT_END}{after}"


def parse_result_region(text: str) -> tuple[str, str, str]:
    """Return the one ordered bounded public-result region."""
    if (
        text.count(RESULT_START) != 1
        or text.count(RESULT_END) != 1
        or text.index(RESULT_START) > text.index(RESULT_END)
    ):
        raise ManifestBuildError("Document must contain exactly one result region")
    before, remainder = text.split(RESULT_START)
    body, after = remainder.split(RESULT_END)
    return before, body, after


def _tables(p: PublicResultProjection) -> tuple[_Table, ...]:
    return (
        _cohort_table(p),
        _development_performance_table(p),
        _heldout_performance_table(p),
        _probability_operating_points_table(p),
        _observedness_subgroups_table(p),
    )


def _cohort_table(p: PublicResultProjection) -> _Table:
    cohort = [
        {
            "section": "Strict endpoint",
            "cohort": row["scope"].title(),
            "admissions": row["eligible"],
            "positive": row["positive"],
            "negative": row["negative"],
            "status": "Eligible Pneumonia = 1 versus 0",
        }
        for row in p.cohort["rows"]
    ]
    labels = (
        ("full_admissions", "Full source index", "Authenticated source admissions"),
        (
            "official_admissions",
            "Official classification membership",
            "Train + validation + held-out test",
        ),
        ("excluded_admissions", "Audited excluded admissions", "Audited and excluded"),
    )
    cohort.extend(
        {
            "section": "Source reconciliation",
            "cohort": label,
            "admissions": p.cohort["source_reconciliation"][key],
            "status": status,
        }
        for key, label, status in labels
    )
    cohort.extend(
        {
            "section": "Cohort construction",
            "cohort": row["item"],
            "status": row["status"],
        }
        for row in p.cohort["construction"]
    )
    cohort.extend(
        {
            "section": "Aggregate description",
            "cohort": row["scope"].title(),
            "attribute": row["attribute"],
            "category": row["category"],
            "admissions": row["count"],
            "status": "Aggregate count",
        }
        for row in p.cohort["descriptions"]
    )
    endpoint = p.cohort["endpoint"]
    cohort.extend(
        (
            {
                "section": "Endpoint provenance",
                "cohort": "Endpoint",
                "status": endpoint["task_id"],
            },
            {
                "section": "Endpoint provenance",
                "cohort": "Policy version",
                "status": endpoint["label_policy_version"],
            },
            {
                "section": "Endpoint provenance",
                "cohort": "Label source",
                "status": endpoint["label_source"],
            },
            {
                "section": "Endpoint provenance",
                "cohort": "State interpretation",
                "status": (
                    f"Positive: {endpoint['positive']}; negative: {endpoint['negative']}; "
                    f"excluded: {', '.join(endpoint['excluded'])}"
                ),
            },
            {
                "section": "Endpoint provenance",
                "cohort": "Prospective selection",
                "status": endpoint["selection_provenance"],
            },
        )
    )
    return _Table(
        "Symile cohort construction and strict-label eligibility.",
        "Admissions are the analytic unit; subject identity defines grouping.",
        (
            ("section", "Section"),
            ("cohort", "Cohort"),
            ("attribute", "Attribute"),
            ("category", "Category"),
            ("admissions", "Admissions"),
            ("positive", "Positive"),
            ("negative", "Negative"),
            ("status", "Definition / status"),
        ),
        tuple(cohort),
    )


def _development_performance_table(p: PublicResultProjection) -> _Table:
    metrics = (
        ("roc_auc", "AUROC"),
        ("average_precision", "Average Precision"),
        ("brier_score", "Brier score"),
    )
    development: list[dict[str, Any]] = []
    for family in SYMILE_CORE_DEVELOPMENT_FAMILIES:
        development.extend(
            {
                "estimate": "OOF repeat",
                "model": DISPLAY[family],
                "repeat": seed,
                **p.development["repeat_metrics"][family][str(seed)],
            }
            for seed in REPEAT_SEEDS
        )
        ensemble = p.development["ensemble_metrics"].get(family)
        if ensemble is not None:
            development.append(
                {
                    "estimate": "Mean-logit OOF ensemble",
                    "model": DISPLAY[family],
                    **ensemble,
                }
            )
    development.extend(
        {
            "estimate": "ECG extension OOF repeat",
            "model": DISPLAY[SYMILE_ECG_GATED_FAMILY],
            "repeat": seed,
            **p.development["ecg_repeat_metrics"][str(seed)],
        }
        for seed in REPEAT_SEEDS
    )
    development.append(
        {
            "estimate": "ECG extension mean-logit OOF ensemble",
            "model": DISPLAY[SYMILE_ECG_GATED_FAMILY],
            **p.development["ecg_ensemble_metrics"],
        }
    )
    development_effects = {
        "concat_vs_cxr": p.development["paired_effects"]["concat_minus_cxr"],
        "primary": p.development["paired_effects"]["gated_minus_cxr"],
        "gated_vs_concat": p.development["paired_effects"]["gated_minus_concat"],
        "ecg_vs_gated": p.development["ecg_vs_gated_repeat_effects"],
    }
    ensembles = p.development["ensemble_metrics"]
    ensemble_effects = {
        "concat_vs_cxr": _metric_difference(
            ensembles["cxr_labs_concat"], ensembles["cxr_densenet"]
        ),
        "primary": _metric_difference(ensembles["cxr_labs_gated"], ensembles["cxr_densenet"]),
        "gated_vs_concat": _metric_difference(
            ensembles["cxr_labs_gated"], ensembles["cxr_labs_concat"]
        ),
        "ecg_vs_gated": p.development["ecg_vs_gated_ensemble_effect"],
    }
    for effect_name in EFFECT_ORDER:
        for seed in REPEAT_SEEDS:
            development.append(
                {
                    "estimate": "OOF repeat paired effect",
                    "model": EFFECT_DISPLAY[effect_name],
                    "repeat": seed,
                    **{
                        f"delta_{metric}": value
                        for metric, value in development_effects[effect_name][str(seed)].items()
                    },
                }
            )
        development.append(
            {
                "estimate": "Mean-logit OOF paired effect",
                "model": EFFECT_DISPLAY[effect_name],
                **{
                    f"delta_{metric}": value
                    for metric, value in ensemble_effects[effect_name].items()
                },
            }
        )
    return _Table(
        "Repeated-CV development performance.",
        "Point estimates only; no repeated-CV confidence intervals are defined.",
        (
            ("estimate", "Evidence"),
            ("model", "Model / comparison"),
            ("repeat", "Repeat seed"),
            *metrics,
            ("delta_roc_auc", "ΔAUROC"),
            ("delta_average_precision", "ΔAP"),
            ("delta_brier_score", "ΔBrier"),
        ),
        tuple(development),
    )


def _heldout_performance_table(p: PublicResultProjection) -> _Table:
    metrics = (
        ("roc_auc", "AUROC"),
        ("average_precision", "Average Precision"),
        ("brier_score", "Brier score"),
    )
    held = [
        {"evidence": "Predictor", "name": DISPLAY[family], **p.held_out["predictor_views"][family]}
        for family in FINAL_PACKAGE_POLICY
    ]
    for comparison in EFFECT_ORDER:
        row: dict[str, Any] = {"evidence": "Paired effect", "name": EFFECT_DISPLAY[comparison]}
        for metric, result in p.held_out["paired_effects"][comparison].items():
            row.update(
                {
                    f"{metric}_effect": result["point"],
                    f"{metric}_lower": result["lower"],
                    f"{metric}_upper": result["upper"],
                }
            )
        held.append(row)
    return _Table(
        "Held-out test performance and paired effects.",
        "Effects are candidate minus comparator. Intervals are paired subject-level bootstrap "
        "95% CIs; negative ΔBrier favors the candidate.",
        (
            ("evidence", "Evidence"),
            ("name", "Model / comparison"),
            *metrics,
            ("roc_auc_effect", "ΔAUROC"),
            ("roc_auc_lower", "ΔAUROC 95% CI lower"),
            ("roc_auc_upper", "ΔAUROC 95% CI upper"),
            ("average_precision_effect", "ΔAP"),
            ("average_precision_lower", "ΔAP 95% CI lower"),
            ("average_precision_upper", "ΔAP 95% CI upper"),
            ("brier_score_effect", "ΔBrier"),
            ("brier_score_lower", "ΔBrier 95% CI lower"),
            ("brier_score_upper", "ΔBrier 95% CI upper"),
        ),
        tuple(held),
    )


def _probability_operating_points_table(p: PublicResultProjection) -> _Table:
    probability = []
    for family in RELIABILITY_FAMILIES:
        probability.append(
            {
                "section": "Development OOF raw probability",
                "model_or_point": DISPLAY[family],
                **{
                    key: p.development["calibration"][family][key]
                    for key in ("brier_score", "calibration_slope", "calibration_intercept")
                },
            }
        )
        probability.append(
            {
                "section": "Held-out descriptive raw probability",
                "model_or_point": DISPLAY[family],
                **{
                    key: p.held_out["predictor_views"][family][key]
                    for key in ("brier_score", "calibration_slope", "calibration_intercept")
                },
            }
        )
    for name in ("youden_j", "target_sensitivity"):
        probability.append(
            {
                "section": "Held-out behavior at development-derived operating point",
                "model_or_point": "Youden-J" if name == "youden_j" else "90% sensitivity target",
                "threshold": p.thresholds[name],
                **p.held_out["primary_operating_points"][name],
            }
        )
    return _Table(
        "Raw probability quality and primary gated operating points.",
        "Thresholds were derived from development OOF evidence and applied unchanged to "
        "held-out probabilities. No post-hoc recalibration; /predict emits no thresholded "
        "decision.",
        (
            ("section", "Section"),
            ("model_or_point", "Model / operating point"),
            ("brier_score", "Brier score"),
            ("calibration_slope", "Calibration slope"),
            ("calibration_intercept", "Calibration intercept"),
            ("threshold", "Threshold"),
            ("precision", "Precision"),
            ("sensitivity", "Sensitivity"),
            ("specificity", "Specificity"),
            ("f1", "F1"),
            ("tn", "TN"),
            ("fp", "FP"),
            ("fn", "FN"),
            ("tp", "TP"),
        ),
        tuple(probability),
    )


def _observedness_subgroups_table(p: PublicResultProjection) -> _Table:
    metrics = (
        ("roc_auc", "AUROC"),
        ("average_precision", "Average Precision"),
        ("brier_score", "Brier score"),
    )
    ablation_authority = p.development["observedness_ablation"]
    subgroups = [
        {
            "analysis": "Observedness ablation",
            "stratum": "All development",
            "estimate": "OOF repeat effect",
            "repeat": seed,
            "status": "Reported",
            **{
                f"effect_{metric}": ablation_authority["repeat_effects"][str(seed)][metric]
                for metric, _ in metrics
            },
        }
        for seed in REPEAT_SEEDS
    ]
    subgroups.append(
        {
            "analysis": "Observedness ablation",
            "stratum": "All development",
            "estimate": "Mean-logit OOF ensemble effect",
            "status": "Reported",
            **{
                f"effect_{metric}": ablation_authority["ensemble_effect"][metric]
                for metric, _ in metrics
            },
        }
    )
    policy = p.subgroups["policy"]
    unsupported = (
        f"Unavailable — n ≥ {policy['minimum_samples']}, "
        f"positives ≥ {policy['minimum_positives']}, "
        f"negatives ≥ {policy['minimum_negatives']} required"
    )
    for item in p.subgroups["rows"]:
        subgroups.append(
            {
                "analysis": "Focused development subgroup",
                "status": "Reported" if item["supported"] else unsupported,
                **item,
            }
        )
    return _Table(
        "Observedness ablation and focused development subgroups.",
        "Observedness effects are gated − gated-no-observedness; subgroup effects are gated − "
        "CXR. Negative ΔBrier favors the first (candidate) model in each contrast. No subgroup "
        "confidence intervals are defined.",
        (
            ("analysis", "Analysis"),
            ("stratum", "Stratum"),
            ("estimate", "Evidence"),
            ("repeat", "Repeat seed"),
            ("status", "Status"),
            ("n", "N"),
            ("positives", "Positive"),
            ("negatives", "Negative"),
            ("cxr_roc_auc", "CXR AUROC"),
            ("gated_roc_auc", "Gated AUROC"),
            ("effect_roc_auc", "ΔAUROC"),
            ("cxr_average_precision", "CXR AP"),
            ("gated_average_precision", "Gated AP"),
            ("effect_average_precision", "ΔAP"),
            ("cxr_brier_score", "CXR Brier"),
            ("gated_brier_score", "Gated Brier"),
            ("effect_brier_score", "ΔBrier"),
        ),
        tuple(subgroups),
    )


def _metric_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {
        metric: float(left[metric]) - float(right[metric])
        for metric in ("roc_auc", "average_precision", "brier_score")
    }


def _write_csv(path: Path, table: _Table) -> None:
    fields = [key for key, _ in table.columns]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(
            {field: "" if row.get(field) is None else row.get(field, "") for field in fields}
            for row in table.rows
        )


def _markdown_table(table: _Table) -> str:
    lines = [
        table.caption,
        "",
        "| " + " | ".join(label for _, label in table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_format(row.get(key)) for key, _ in table.columns) + " |"
        for row in table.rows
    )
    return "\n".join((*lines, "", f"*{table.note}*", ""))


def _format(value: object) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|")


def _incremental_effects(p: PublicResultProjection, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    for axis, metric, title in zip(
        axes,
        ("roc_auc", "average_precision", "brier_score"),
        ("ΔAUROC", "ΔAverage Precision", "ΔBrier"),
        strict=True,
    ):
        axis.axvline(0, color="#666", linestyle="--", lw=1)
        for y, name in enumerate(EFFECT_ORDER):
            effect = p.held_out["paired_effects"][name][metric]
            color = "#D55E00" if name == "primary" else "#0072B2"
            axis.hlines(y, effect["lower"], effect["upper"], color=color, linewidth=2)
            axis.plot(effect["point"], y, marker="o", color=color, linestyle="none")
        axis.set_yticks(
            range(4),
            [EFFECT_DISPLAY[name] for name in EFFECT_ORDER] if axis is axes[0] else [""] * 4,
        )
        axis.set_xlabel(title)
        axis.invert_yaxis()
    fig.suptitle(
        "Held-out paired incremental effects (95% subject-bootstrap intervals)\n"
        "Negative ΔBrier favors the candidate"
    )
    _save_svg(fig, path)


def _discrimination(p: PublicResultProjection, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    axes[0].plot([0, 1], [0, 1], color="#888", linestyle="--", lw=1)
    for family in FINAL_PACKAGE_POLICY:
        curve = p.held_out["discrimination_curves"][family]
        axes[0].plot(
            curve["roc"]["false_positive_rate"],
            curve["roc"]["true_positive_rate"],
            color=COLORS[family],
            label=DISPLAY[family],
        )
        axes[1].plot(
            curve["precision_recall"]["recall"],
            curve["precision_recall"]["precision"],
            color=COLORS[family],
            label=DISPLAY[family],
        )
    axes[0].set(title="Held-out ROC", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].set(title="Held-out precision–recall", xlabel="Recall", ylabel="Precision")
    axes[1].legend(fontsize=7)
    _save_svg(fig, path)


def _reliability(p: PublicResultProjection, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for axis, curves, title in (
        (axes[0], p.development["reliability_curves"], "Development OOF"),
        (axes[1], p.held_out["reliability_curves"], "Held-out test"),
    ):
        axis.plot([0, 1], [0, 1], color="#777", linestyle="--")
        for family in RELIABILITY_FAMILIES:
            curve = curves[family]
            axis.plot(
                curve["mean_predicted_probability"],
                curve["observed_positive_fraction"],
                color=COLORS[family],
                label=DISPLAY[family],
            )
        axis.set(
            title=title, xlabel="Mean predicted probability", ylabel="Observed positive fraction"
        )
    axes[1].legend(fontsize=8)
    fig.suptitle("Raw reliability (descriptive; no post-hoc recalibration)")
    _save_svg(fig, path)


def _save_svg(figure: Any, path: Path) -> None:
    figure.savefig(path, format="svg", metadata={"Date": None})
    plt.close(figure)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
