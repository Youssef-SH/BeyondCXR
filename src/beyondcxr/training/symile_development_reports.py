"""Render deterministic aggregate views of validated Symile development evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_preprocess import (
    LAB_OBSERVED_COLUMNS,
)
from beyondcxr.data.symile_schemas import (
    OUTER_FOLDS,
    REPEAT_SEEDS,
    TASK_ID,
)
from beyondcxr.training.config import load_symile_development_config, with_runtime
from beyondcxr.training.symile_data import (
    DEVELOPMENT_COUNT,
    load_symile_development_cohort,
)
from beyondcxr.training.symile_families import (
    SYMILE_CORE_DEVELOPMENT_FAMILIES,
    SYMILE_FUSION_FAMILIES,
)
from beyondcxr.training.symile_statistics import (
    METRIC_POLICY,
    focused_development_subgroups,
    headline_metric_names,
)
from beyondcxr.training.symile_statistics import (
    metrics as symile_headline_metrics,
)
from beyondcxr.utils.operational_logging import add_logging_argument, configure_logging
from beyondcxr.utils.privacy import validate_public_reports
from beyondcxr.utils.publication import (
    install_regenerable_directory,
    publish_regenerable_bytes,
    staging_directory,
    validate_path_component,
)
from beyondcxr.utils.symile_publication import (
    CONFIG_FILENAME,
    DEVELOPMENT_PREFIX,
    SUMMARY_FILENAME,
    ValidatedDevelopmentResult,
    ValidatedFoldPackage,
    _finite,
    _identity,
    _require_exact_regular_files,
    development_repeat_oof,
    validate_development_result,
    validated_development_repeat_oof,
)


def analyze_symile_development(
    development_ids: Sequence[str],
    *,
    report_root: str | Path = "reports/symile/development",
    model_root: str | Path = "models/symile/development",
    prediction_root: str | Path = "private",
    manifest_root: str | Path = "data/manifests",
) -> Path:
    """Validate six family authorities and publish their aggregate analysis derivative."""
    if len(development_ids) != 6 or len(set(development_ids)) != 6:
        raise ManifestBuildError("Symile analysis requires six unique development IDs")
    developments = _resolve_developments(
        development_ids,
        report_root,
        model_root,
        prediction_root,
        manifest_root,
    )
    if set(developments) != set(SYMILE_CORE_DEVELOPMENT_FAMILIES):
        raise ManifestBuildError(
            "Symile analysis development IDs do not cover exact core-development families"
        )
    family_ids = {
        family: str(result.manifest["development_id"]) for family, result in developments.items()
    }
    return _publish_analysis_document(
        report_root, family_ids, _analysis_from_developments(developments, family_ids)
    )


def _resolve_developments(
    development_ids: Sequence[str],
    report_root: str | Path,
    model_root: str | Path,
    prediction_root: str | Path,
    manifest_root: str | Path,
) -> dict[str, ValidatedDevelopmentResult]:
    root = Path(report_root) / "families"
    result: dict[str, ValidatedDevelopmentResult] = {}
    for development_id in development_ids:
        validate_path_component(development_id, "development ID")
        reference = validate_development_result(
            root / development_id,
            model_root=model_root,
            prediction_root=prediction_root,
            manifest_root=manifest_root,
            expected_development_id=development_id,
        )
        family = str(reference.manifest["family_id"])
        if family in result:
            raise ManifestBuildError("Symile analysis received duplicate family authorities")
        result[family] = reference
    return result


FOCUSED_SUBGROUP_DERIVATIVE_SCHEMA_VERSION = 1
FOCUSED_SUBGROUP_FILENAME = "focused.json"


def publish_focused_subgroup_derivative(
    *, report_root: str | Path, summary: dict[str, object]
) -> Path:
    """Publish the one canonical reconstructible focused-subgroup view."""
    encoded = _focused_subgroup_bytes(summary)
    destination = Path(report_root) / "development-subgroups" / FOCUSED_SUBGROUP_FILENAME
    publish_regenerable_bytes(destination, encoded)
    if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != encoded:
        raise ManifestBuildError("Focused subgroup derivative conflicts with existing content")
    return destination


def validate_focused_subgroup_derivative(path: str | Path) -> dict[str, object]:
    """Validate the canonical aggregate subgroup derivative."""
    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.name != FOCUSED_SUBGROUP_FILENAME:
        raise ManifestBuildError("Focused subgroup derivative is unavailable")
    raw = source.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestBuildError("Focused subgroup derivative is unreadable") from exc
    schema_version = (
        document.get("focused_subgroup_derivative_schema_version")
        if isinstance(document, dict)
        else None
    )
    if (
        not isinstance(document, dict)
        or set(document) != {"focused_subgroup_derivative_schema_version", "policy", "strata"}
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != FOCUSED_SUBGROUP_DERIVATIVE_SCHEMA_VERSION
    ):
        raise ManifestBuildError("Focused subgroup derivative contract is invalid")
    return document


def derive_focused_subgroup_derivative(
    *,
    cxr_development_id: str,
    primary_repeats: pd.DataFrame,
    report_root: Path,
    model_root: str | Path,
    prediction_root: str | Path,
    manifest_root: str | Path,
) -> dict[str, object]:
    development_root = report_root / "development"
    cxr = validate_development_result(
        development_root / "families" / cxr_development_id,
        model_root=model_root,
        prediction_root=prediction_root,
        manifest_root=manifest_root,
        expected_development_id=cxr_development_id,
    )
    cxr_repeats = validated_development_repeat_oof(cxr, prediction_root)
    return _focused_subgroups_from_development(
        cxr, cxr_repeats, primary_repeats, model_root=model_root, manifest_root=manifest_root
    )


def _focused_subgroups_from_development(
    cxr, cxr_repeats, primary_repeats, *, model_root, manifest_root
):
    first_fold = cxr.manifest["fold_packages"][0]["fold_package_id"]
    config = with_runtime(
        load_symile_development_config(
            Path(model_root) / "packages" / first_fold / CONFIG_FILENAME,
            bundle_id=cxr.manifest["scientific_context"]["bundle_id"],
            cv_assignment_id=cxr.manifest["scientific_context"]["cv_assignment_id"],
        ),
        manifest_directory=manifest_root,
    )
    cohort = load_symile_development_cohort(config).frame
    attributes = cohort.loc[:, ["sample_id", "age_years", "sex", "view_position"]].copy()
    attributes["observed_lab_count"] = cohort.loc[:, LAB_OBSERVED_COLUMNS].sum(axis=1)
    summary = focused_development_subgroups(cxr_repeats, primary_repeats, attributes)
    _focused_subgroup_bytes(summary)
    return summary


def publish_operation_subgroups(capability, *, report_root, model_root, manifest_root):
    """Render subgroups from the active operation's authenticated OOF evidence."""
    from beyondcxr.training.symile_campaign_control import require_validated_held_out_capability

    capability = require_validated_held_out_capability(capability)
    if capability._operation is None:
        raise PermissionError("Subgroups require an active held-out operation")
    families = {
        item.manifest["family_id"]: item for item in capability.frozen_campaign.developments
    }
    cxr = families["cxr_densenet"]
    primary = families["cxr_labs_gated"]
    summary = _focused_subgroups_from_development(
        cxr,
        development_repeat_oof(cxr, cxr.predictions),
        development_repeat_oof(primary, primary.predictions),
        model_root=Path(model_root) / "development",
        manifest_root=manifest_root,
    )
    return publish_focused_subgroup_derivative(report_root=report_root, summary=summary)


def _focused_subgroup_bytes(summary: dict[str, object]) -> bytes:
    if not isinstance(summary, dict) or set(summary) != {"policy", "strata"}:
        raise ManifestBuildError("Focused subgroup derivative summary is invalid")
    document = {
        "focused_subgroup_derivative_schema_version": (FOCUSED_SUBGROUP_DERIVATIVE_SCHEMA_VERSION),
        **summary,
    }
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


ANALYSIS_SCHEMA_VERSION = 1


ANALYSIS_MANIFEST_FILENAME = "manifest.json"


SYMILE_DEVELOPMENT_COMPARISON_POLICY = {
    "policy_version": "symile-core-development-analysis-v1",
    "alignment": ["sample_id", "repeat_seed"],
    "metric_policy": METRIC_POLICY,
    "paired_comparisons": {
        "concat_minus_cxr": ["cxr_labs_concat", "cxr_densenet"],
        "gated_minus_cxr": ["cxr_labs_gated", "cxr_densenet"],
        "gated_minus_concat": ["cxr_labs_gated", "cxr_labs_concat"],
    },
    "neural_ensemble": "align three repeat logits; arithmetic mean logits; sigmoid once",
    "uncertainty": "none; folds are not treated as independent replicates",
    "official_test_access": "closed",
}


def _publish_analysis_document(report_root, family_development_ids, derived) -> Path:
    payload = {
        "dataset_id": "symile",
        "task_id": TASK_ID,
        "family_development_ids": dict(sorted(family_development_ids.items())),
        "policy": SYMILE_DEVELOPMENT_COMPARISON_POLICY,
        **derived,
    }
    document = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        **payload,
    }
    destination = Path(report_root) / "analyses" / "core"
    stage = staging_directory(destination)
    try:
        (stage / ANALYSIS_MANIFEST_FILENAME).write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (stage / SUMMARY_FILENAME).write_text(_analysis_summary(document), encoding="utf-8")
        validate_public_reports([stage / SUMMARY_FILENAME], forbidden_source_values=())
        expected = {
            name: (stage / name).read_bytes()
            for name in (ANALYSIS_MANIFEST_FILENAME, SUMMARY_FILENAME)
        }

        def validate_rendering(path: Path, **_kwargs: object) -> None:
            _require_exact_regular_files(path, set(expected))
            if any((path / name).read_bytes() != content for name, content in expected.items()):
                raise ManifestBuildError(
                    "Symile analysis rendering differs from authenticated evidence"
                )

        install_regenerable_directory(stage, destination, validate_rendering)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return destination


def publish_operation_analysis(capability, *, report_root: Path) -> Path:
    """Render the authenticated frozen development graph within its locked operation."""
    from beyondcxr.training.symile_campaign_control import require_validated_held_out_capability

    capability = require_validated_held_out_capability(capability)
    if capability._operation is None:
        raise PermissionError("Analysis requires an active held-out operation")
    developments = {
        item.manifest["family_id"]: item
        for item in capability.frozen_campaign.developments
        if item.manifest["family_id"] in SYMILE_CORE_DEVELOPMENT_FAMILIES
    }
    if set(developments) != set(SYMILE_CORE_DEVELOPMENT_FAMILIES):
        raise ManifestBuildError("Frozen analysis family membership is incomplete")
    family_ids = {family: item.manifest["development_id"] for family, item in developments.items()}
    return _publish_analysis_document(
        report_root, family_ids, _analysis_from_developments(developments, family_ids)
    )


def validate_analysis_result(
    directory: str | Path,
    *,
    report_root: str | Path = "reports/symile/development",
    model_root: str | Path = "models/symile/development",
    prediction_root: str | Path = "private",
    manifest_root: str | Path = "data/manifests",
    enforce_directory_name: bool = True,
) -> dict[str, Any]:
    """Validate one deterministic cross-family development analysis derivative."""
    root = Path(directory)
    _require_exact_regular_files(root, {ANALYSIS_MANIFEST_FILENAME, SUMMARY_FILENAME})
    try:
        document = json.loads((root / ANALYSIS_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestBuildError("Symile analysis manifest is unreadable") from exc
    expected = {
        "analysis_schema_version",
        "dataset_id",
        "task_id",
        "family_development_ids",
        "repeat_metrics",
        "paired_effects",
        "ensemble_metrics",
        "observedness_ablation",
        "policy",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise ManifestBuildError("Symile analysis manifest fields are invalid")
    schema_version = document["analysis_schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != ANALYSIS_SCHEMA_VERSION
    ):
        raise ManifestBuildError("Symile analysis schema version is invalid")
    if (
        document["dataset_id"] != "symile"
        or document["task_id"] != TASK_ID
        or set(document["family_development_ids"]) != set(SYMILE_CORE_DEVELOPMENT_FAMILIES)
        or any(
            not _identity(value, DEVELOPMENT_PREFIX)
            for value in document["family_development_ids"].values()
        )
    ):
        raise ManifestBuildError("Symile analysis lineage is invalid")
    if document["policy"] != SYMILE_DEVELOPMENT_COMPARISON_POLICY:
        raise ManifestBuildError("Symile analysis policy is invalid")
    if enforce_directory_name and root.name != "core":
        raise ManifestBuildError("Symile analysis directory is not canonical")
    derived = _derive_analysis(
        document["family_development_ids"],
        report_root,
        model_root,
        prediction_root,
        manifest_root,
    )
    for field in (
        "repeat_metrics",
        "paired_effects",
        "ensemble_metrics",
        "observedness_ablation",
    ):
        if not _nested_numeric_close(document[field], derived[field]):
            raise ManifestBuildError(f"Symile analysis {field} differs from family evidence")
    if (root / SUMMARY_FILENAME).read_bytes() != _analysis_summary(document).encode("utf-8"):
        raise ManifestBuildError("Symile analysis summary differs from its manifest")
    validate_public_reports([root / SUMMARY_FILENAME], forbidden_source_values=())
    return document


def _analysis_summary(document: Mapping[str, Any]) -> str:
    lines = [
        "# Symile cross-family development analysis",
        "",
        "- Evidence: three-repeat patient-grouped OOF development predictions",
        "- Official test access: closed",
        "- Uncertainty: no fold-independence interval or pooled OOF bootstrap",
        "",
        "## Neural mean-logit ensemble",
        "",
        "| Family | AUROC | Average Precision | Brier |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family, metrics in sorted(document["ensemble_metrics"].items()):
        lines.append(
            f"| {family} | {metrics['roc_auc']:.6f} | {metrics['average_precision']:.6f} | "
            f"{metrics['brier_score']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Repeat-level paired effects",
            "",
            "| Comparison | Repeat | Delta AUROC | Delta AP | Delta Brier |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparison, repeats in sorted(document["paired_effects"].items()):
        for seed, metrics in sorted(repeats.items(), key=lambda item: int(item[0])):
            lines.append(
                f"| {comparison} | {seed} | {metrics['roc_auc']:.6f} | "
                f"{metrics['average_precision']:.6f} | {metrics['brier_score']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Observedness ablation",
            "",
            "The manifest records gated minus gated-without-observedness effects for every "
            "repeat and for the mean-logit development ensemble.",
            "",
            "Repeat-to-repeat values are stability diagnostics. They are not independent-fold "
            "confidence intervals.",
        ]
    )
    return "\n".join(lines) + "\n"


def _derive_analysis(
    family_ids: Mapping[str, str],
    report_root: str | Path,
    model_root: str | Path,
    prediction_root: str | Path,
    manifest_root: str | Path,
) -> dict[str, object]:
    developments = {
        family: validate_development_result(
            Path(report_root) / "families" / development_id,
            model_root=model_root,
            prediction_root=prediction_root,
            manifest_root=manifest_root,
            expected_development_id=development_id,
        )
        for family, development_id in family_ids.items()
    }
    return _analysis_from_developments(developments, family_ids)


def _analysis_from_developments(developments, family_ids) -> dict[str, object]:
    frames: dict[str, pd.DataFrame] = {}
    family_folds: dict[str, dict[tuple[int, int], ValidatedFoldPackage]] = {}
    for family, result in developments.items():
        if result.manifest["family_id"] != family:
            raise ManifestBuildError("Symile analysis family authority is mislabeled")
        fold_frames = []
        resolved = result.folds
        family_folds[family] = {
            (fold.manifest["repeat_seed"], fold.manifest["outer_fold"]): fold for fold in resolved
        }
        predictions = result.predictions
        for evidence in predictions:
            frame = evidence.predictions.to_pandas()
            frame["repeat_seed"] = evidence.manifest["repeat_seed"]
            fold_frames.append(frame)
        frames[family] = pd.concat(fold_frames, ignore_index=True).sort_values(
            ["repeat_seed", "sample_id"], kind="stable"
        )
    _validate_cross_family_fold_lineage(family_folds, family_ids)
    _validate_cross_family_oof(frames)
    repeat_metrics = {
        family: {
            str(seed): _metrics(frame.loc[frame["repeat_seed"] == seed]) for seed in REPEAT_SEEDS
        }
        for family, frame in frames.items()
    }
    comparisons = SYMILE_DEVELOPMENT_COMPARISON_POLICY["paired_comparisons"]
    paired = {
        name: _paired_effects(frames[left], frames[right])
        for name, (left, right) in comparisons.items()
    }
    neural = (
        "cxr_densenet",
        "cxr_labs_concat",
        "cxr_labs_gated",
        "cxr_labs_gated_no_observedness",
    )
    ensemble = {family: _ensemble_metrics(frames[family]) for family in neural}
    ablation_repeats = _paired_effects(
        frames["cxr_labs_gated"], frames["cxr_labs_gated_no_observedness"]
    )
    ablation_ensemble = {
        metric: ensemble["cxr_labs_gated"][metric]
        - ensemble["cxr_labs_gated_no_observedness"][metric]
        for metric in headline_metric_names()
    }
    return {
        "repeat_metrics": repeat_metrics,
        "paired_effects": paired,
        "ensemble_metrics": ensemble,
        "observedness_ablation": {
            "repeat_effects": ablation_repeats,
            "ensemble_effect": ablation_ensemble,
        },
    }


def _validate_cross_family_fold_lineage(
    families: Mapping[str, Mapping[tuple[int, int], ValidatedFoldPackage]],
    family_ids: Mapping[str, str],
) -> None:
    coordinates = {(seed, fold) for seed in REPEAT_SEEDS for fold in OUTER_FOLDS}
    for coordinate in coordinates:
        _validate_cross_family_coordinate_lineage(families, family_ids, coordinate)


def _validate_cross_family_coordinate_lineage(
    families: Mapping[str, Mapping[tuple[int, int], ValidatedFoldPackage]],
    family_ids: Mapping[str, str],
    coordinate: tuple[int, int],
) -> None:
    """Validate cross-family lineage at one coordinate without requiring a campaign graph."""
    try:
        packages = [families[family][coordinate] for family in SYMILE_CORE_DEVELOPMENT_FAMILIES]
    except KeyError as exc:
        raise ManifestBuildError("Symile analysis family coordinates are incomplete") from exc
    data_fields = ("bundle_id", "cv_assignment_id", "task_id")
    selection_packages = [
        item for item in packages if item.manifest["family_id"] != "labs_logistic"
    ]
    if len({item.manifest["inner_split"]["inner_split_id"] for item in selection_packages}) != 1:
        raise ManifestBuildError("Symile analysis families use different inner splits")
    if any(
        len({item.manifest["lineage"][field] for item in packages}) != 1 for field in data_fields
    ):
        raise ManifestBuildError("Symile analysis families use different frozen data lineage")
    cxr = families["cxr_densenet"][coordinate]
    for family in SYMILE_FUSION_FAMILIES:
        source = families[family][coordinate].manifest["source_cxr"]
        if (
            source["development_id"] != family_ids["cxr_densenet"]
            or source["fold_package_id"] != cxr.manifest["fold_package_id"]
        ):
            raise ManifestBuildError("Symile fusion analysis uses incompatible CXR lineage")


def _validate_cross_family_oof(frames: Mapping[str, pd.DataFrame]) -> None:
    for seed in REPEAT_SEEDS:
        reference: pd.Series | None = None
        for family in SYMILE_CORE_DEVELOPMENT_FAMILIES:
            scoped = frames[family].loc[frames[family]["repeat_seed"] == seed]
            targets = scoped.set_index("sample_id")["target"].sort_index()
            if reference is None:
                reference = targets
            elif not targets.equals(reference):
                raise ManifestBuildError("Symile analysis OOF targets differ across families")


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    truth = frame["target"].to_numpy(dtype=np.int8)
    scores = frame["probability"].to_numpy(dtype=np.float64)
    if len(frame) != DEVELOPMENT_COUNT or set(truth.tolist()) != {0, 1}:
        raise ManifestBuildError("Symile analysis family evidence is incomplete")
    return symile_headline_metrics(truth, scores)


def _paired_effects(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for seed in REPEAT_SEEDS:
        left_seed = left.loc[left["repeat_seed"] == seed]
        right_seed = right.loc[right["repeat_seed"] == seed]
        aligned = left_seed.merge(
            right_seed,
            on=["sample_id", "repeat_seed"],
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        if (
            len(aligned) != DEVELOPMENT_COUNT
            or not aligned["target_left"].eq(aligned["target_right"]).all()
        ):
            raise ManifestBuildError("Symile analysis paired evidence is misaligned")
        left_metrics = _metrics(
            aligned.rename(columns={"target_left": "target", "probability_left": "probability"})
        )
        right_metrics = _metrics(
            aligned.rename(columns={"target_right": "target", "probability_right": "probability"})
        )
        result[str(seed)] = {
            metric: left_metrics[metric] - right_metrics[metric]
            for metric in headline_metric_names()
        }
    return result


def _ensemble_metrics(frame: pd.DataFrame) -> dict[str, float]:
    logits = frame.pivot(index="sample_id", columns="repeat_seed", values="logit")
    targets = frame.pivot(index="sample_id", columns="repeat_seed", values="target")
    if (
        list(logits.columns) != list(REPEAT_SEEDS)
        or len(logits) != DEVELOPMENT_COUNT
        or logits.isna().any().any()
        or list(targets.columns) != list(REPEAT_SEEDS)
        or not targets.nunique(axis=1).eq(1).all()
    ):
        raise ManifestBuildError("Symile analysis neural repeat evidence is misaligned")
    probabilities = _sigmoid(logits.to_numpy(dtype=np.float64).mean(axis=1))
    evidence = pd.DataFrame({"target": targets.iloc[:, 0], "probability": probabilities})
    return _metrics(evidence)


def _nested_numeric_close(left: object, right: object) -> bool:
    if isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and set(left) == set(right)
            and all(_nested_numeric_close(left[key], value) for key, value in right.items())
        )
    return (
        _finite(left)
        and _finite(right)
        and np.isclose(float(left), float(right), rtol=0.0, atol=1e-15)
    )


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-ids", nargs=6, required=True)
    parser.add_argument("--report-root", type=Path, default=Path("reports/symile/development"))
    parser.add_argument("--model-root", type=Path, default=Path("models/symile/development"))
    parser.add_argument("--prediction-root", type=Path, default=Path("private"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        directory = analyze_symile_development(
            args.development_ids,
            report_root=args.report_root,
            model_root=args.model_root,
            prediction_root=args.prediction_root,
            manifest_root=args.manifest_root,
        )
    except (ManifestBuildError, OSError, ValueError) as exc:
        print(f"Symile development analysis failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps({"report_directory": directory.as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
