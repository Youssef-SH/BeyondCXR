"""Publish explicit aggregate cross-family Symile M5 development analysis."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from radfusion.data.errors import ManifestBuildError
from radfusion.training.config import SYMILE_M5_FAMILIES
from radfusion.training.symile_data import DEVELOPMENT_COUNT
from radfusion.utils.operational_logging import add_logging_argument, configure_logging
from radfusion.utils.symile_publication import (
    FOLD_PREFIX,
    ValidatedDevelopmentResult,
    publish_analysis_result,
    validate_development_result,
    validate_fold_package,
)

NEURAL_ANALYSIS_FAMILIES = ("cxr", "concat", "gated", "gated_no_observedness")
PAIRED_COMPARISONS = {
    "concat_minus_cxr": ("concat", "cxr"),
    "gated_minus_cxr": ("gated", "cxr"),
    "gated_minus_concat": ("gated", "concat"),
}
ANALYSIS_POLICY = {
    "policy_version": "symile-m5-development-analysis-v1",
    "alignment": ["sample_id", "repeat_seed"],
    "metrics": ["roc_auc", "average_precision", "brier_score"],
    "paired_comparisons": PAIRED_COMPARISONS,
    "neural_ensemble": "align three repeat logits; arithmetic mean logits; sigmoid once",
    "uncertainty": "none; folds are not treated as independent replicates",
    "official_test_access": "closed",
}


def analyze_symile_development(
    development_ids: Sequence[str],
    *,
    report_root: str | Path = "reports/symile/development",
    model_root: str | Path = "models/symile/development",
) -> tuple[str, Path]:
    """Validate six explicit family authorities and publish aggregate M5 evidence."""
    if len(development_ids) != 6 or len(set(development_ids)) != 6:
        raise ManifestBuildError("Symile analysis requires six unique development IDs")
    developments = _resolve_developments(development_ids, report_root, model_root)
    if set(developments) != set(SYMILE_M5_FAMILIES):
        raise ManifestBuildError("Symile analysis development IDs do not cover exact M5 families")
    frames = {
        family: _load_family_oof(result, model_root) for family, result in developments.items()
    }
    repeat_metrics = {
        family: {
            str(seed): _metrics(scoped["target"], scoped["probability"])
            for seed, scoped in frame.groupby("repeat_seed", sort=True)
        }
        for family, frame in frames.items()
    }
    paired_effects = {
        comparison: _paired_repeat_effects(frames[left], frames[right])
        for comparison, (left, right) in PAIRED_COMPARISONS.items()
    }
    ensemble_metrics = {
        family: _metrics(*_mean_logit_ensemble(frames[family])[1:])
        for family in NEURAL_ANALYSIS_FAMILIES
    }
    observedness_ablation = {
        "repeat_effects": _paired_repeat_effects(frames["gated"], frames["gated_no_observedness"]),
        "ensemble_effect": _ensemble_effect(frames["gated"], frames["gated_no_observedness"]),
    }
    return publish_analysis_result(
        report_root=report_root,
        model_root=model_root,
        family_development_ids={
            family: str(result.manifest["development_id"])
            for family, result in developments.items()
        },
        repeat_metrics=repeat_metrics,
        paired_effects=paired_effects,
        ensemble_metrics=ensemble_metrics,
        observedness_ablation=observedness_ablation,
        policy=ANALYSIS_POLICY,
    )


def _resolve_developments(
    development_ids: Sequence[str], report_root: str | Path, model_root: str | Path
) -> dict[str, ValidatedDevelopmentResult]:
    root = Path(report_root) / "families"
    result: dict[str, ValidatedDevelopmentResult] = {}
    for development_id in development_ids:
        reference = validate_development_result(
            root / development_id,
            model_root=model_root,
            expected_development_id=development_id,
        )
        family = str(reference.manifest["family"])
        if family in result:
            raise ManifestBuildError("Symile analysis received duplicate family authorities")
        result[family] = reference
    return result


def _load_family_oof(
    development: ValidatedDevelopmentResult,
    model_root: str | Path,
) -> pd.DataFrame:
    folds_root = Path(model_root) / "folds"
    frames: list[pd.DataFrame] = []
    for reference in development.manifest["fold_packages"]:
        fold_id = reference["fold_package_id"]
        if not isinstance(fold_id, str) or not fold_id.startswith(FOLD_PREFIX):
            raise ManifestBuildError("Symile development contains an invalid fold reference")
        fold = validate_fold_package(folds_root / fold_id, expected_fold_package_id=fold_id)
        if (
            fold.manifest_sha256 != reference["fold_manifest_sha256"]
            or fold.manifest["family"] != development.manifest["family"]
            or fold.manifest["repeat_seed"] != reference["repeat_seed"]
            or fold.manifest["outer_fold"] != reference["outer_fold"]
        ):
            raise ManifestBuildError("Symile family authority and fold package differ")
        frame = fold.oof.to_pandas()
        frame["repeat_seed"] = int(reference["repeat_seed"])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    for seed in (17, 42, 2026):
        scoped = combined.loc[combined["repeat_seed"] == seed]
        if len(scoped) != DEVELOPMENT_COUNT or scoped["sample_id"].duplicated().any():
            raise ManifestBuildError("Symile family repeat OOF coverage is incomplete")
    return combined.sort_values(["repeat_seed", "sample_id"], kind="stable").reset_index(drop=True)


def _paired_repeat_effects(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for seed in (17, 42, 2026):
        left_scoped = left.loc[left["repeat_seed"] == seed]
        right_scoped = right.loc[right["repeat_seed"] == seed]
        aligned = _align(left_scoped, right_scoped)
        left_metrics = _metrics(aligned["target"], aligned["left_probability"])
        right_metrics = _metrics(aligned["target"], aligned["right_probability"])
        result[str(seed)] = {
            metric: left_metrics[metric] - right_metrics[metric]
            for metric in ("roc_auc", "average_precision", "brier_score")
        }
    return result


def _mean_logit_ensemble(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = frame.pivot(index="sample_id", columns="repeat_seed", values="logit")
    if list(pivot.columns) != [17, 42, 2026] or pivot.isna().any().any():
        raise ManifestBuildError("Symile neural repeat logits are not exactly aligned")
    targets = frame.pivot(index="sample_id", columns="repeat_seed", values="target")
    if list(targets.columns) != [17, 42, 2026] or not targets.nunique(axis=1).eq(1).all():
        raise ManifestBuildError("Symile neural repeat targets differ")
    mean_logits = pivot.to_numpy(dtype=np.float64).mean(axis=1)
    probabilities = _sigmoid(mean_logits)
    return mean_logits, targets.iloc[:, 0].to_numpy(dtype=np.int8), probabilities


def _ensemble_effect(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float]:
    _, left_targets, left_probabilities = _mean_logit_ensemble(left)
    _, right_targets, right_probabilities = _mean_logit_ensemble(right)
    if not np.array_equal(left_targets, right_targets):
        raise ManifestBuildError("Symile ensemble targets differ across families")
    left_metrics = _metrics(left_targets, left_probabilities)
    right_metrics = _metrics(right_targets, right_probabilities)
    return {key: left_metrics[key] - right_metrics[key] for key in left_metrics}


def _align(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    aligned = left[["sample_id", "target", "probability"]].merge(
        right[["sample_id", "target", "probability"]],
        on="sample_id",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if len(aligned) != len(left) or len(aligned) != len(right):
        raise ManifestBuildError("Symile paired family samples differ")
    if not aligned["target_left"].eq(aligned["target_right"]).all():
        raise ManifestBuildError("Symile paired family targets differ")
    return aligned.rename(
        columns={
            "target_left": "target",
            "probability_left": "left_probability",
            "probability_right": "right_probability",
        }
    ).drop(columns="target_right")


def _metrics(targets: object, probabilities: object) -> dict[str, float]:
    truth = np.asarray(targets, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.shape != scores.shape or truth.ndim != 1 or set(truth.tolist()) != {0, 1}:
        raise ManifestBuildError("Symile analysis metric inputs are invalid")
    return {
        "roc_auc": float(roc_auc_score(truth, scores)),
        "average_precision": float(average_precision_score(truth, scores)),
        "brier_score": float(brier_score_loss(truth, scores)),
    }


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-ids", nargs=6, required=True)
    parser.add_argument("--report-root", type=Path, default=Path("reports/symile/development"))
    parser.add_argument("--model-root", type=Path, default=Path("models/symile/development"))
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        analysis_id, directory = analyze_symile_development(
            args.development_ids,
            report_root=args.report_root,
            model_root=args.model_root,
        )
    except (ManifestBuildError, OSError, ValueError) as exc:
        print(f"Symile development analysis failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps({"analysis_id": analysis_id, "report_directory": directory.as_posix()}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
