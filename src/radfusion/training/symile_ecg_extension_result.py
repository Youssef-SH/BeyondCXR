"""Compact ECG extension result derived exclusively from validated development evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from radfusion.data.errors import ManifestBuildError
from radfusion.data.symile_preprocess import LAB_OBSERVED_COLUMNS
from radfusion.data.symile_schemas import REPEAT_SEEDS
from radfusion.training.config import load_symile_development_config, with_runtime
from radfusion.training.symile_data import load_symile_development_cohort
from radfusion.training.symile_families import FINAL_PACKAGE_POLICY, SYMILE_ECG_GATED_FAMILY
from radfusion.training.symile_final_packages import final_input_projection_from_development
from radfusion.training.symile_statistics import (
    METRIC_POLICY,
    development_mean_logit_ensemble,
    focused_development_subgroups,
    headline_metric_names,
    metrics,
    sensitivity_threshold,
    sigmoid,
    youden_threshold,
)
from radfusion.utils.package_identity import (
    canonical_scientific_id,
    pretrained_weight_semantic_identity,
)
from radfusion.utils.publication import (
    install_immutable_directory,
    publish_bytes_no_replace,
    staging_directory,
)
from radfusion.utils.symile_publication import (
    CONFIG_FILENAME,
    ValidatedDevelopmentResult,
    validate_analysis_result,
    validate_development_result,
    validate_fold_package,
    validated_development_repeat_oof,
)

ECG_EXTENSION_RESULT_PREFIX = "ecg-extension-result-"
FOCUSED_SUBGROUP_DERIVATIVE_SCHEMA_VERSION = 1
_VALIDATION_GUARD = object()


@dataclass(frozen=True)
class ValidatedEcgExtensionResult:
    directory: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    claims_sha256: str
    _guard: object

    @property
    def result_id(self) -> str:
        return str(self.manifest["ecg_extension_result_id"])


def publish_focused_subgroup_derivative(
    *, report_root: str | Path, summary: dict[str, object]
) -> str:
    """Cache one content-addressed aggregate derivative without creating an authority class."""
    encoded = _focused_subgroup_bytes(summary)
    derivative_id = _focused_subgroup_id(encoded)
    destination = Path(report_root) / "development-subgroups" / f"{derivative_id}.json"
    publish_bytes_no_replace(destination, encoded)
    if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != encoded:
        raise ManifestBuildError("Focused subgroup derivative conflicts with existing content")
    return derivative_id


def validate_focused_subgroup_derivative(
    path: str | Path, *, expected_id: str | None = None
) -> dict[str, object]:
    """Validate the cached aggregate subgroup derivative by its content identity."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ManifestBuildError("Focused subgroup derivative is unavailable")
    raw = source.read_bytes()
    derivative_id = f"focused-subgroup-{hashlib.sha256(raw).hexdigest()}"
    if source.name != f"{derivative_id}.json" or (
        expected_id is not None and derivative_id != expected_id
    ):
        raise ManifestBuildError("Focused subgroup derivative identity is invalid")
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


def publish_ecg_extension_result(
    *,
    report_root: str | Path,
    core_analysis_directory: str | Path,
    ecg_development: ValidatedDevelopmentResult,
    focused_subgroup_derivative: str,
    model_root: str | Path = "models/symile/development",
    prediction_root: str | Path = "private",
    manifest_root: str | Path = "data/manifests",
) -> ValidatedEcgExtensionResult:
    report_root = Path(report_root)
    core_analysis = validate_analysis_result(
        core_analysis_directory,
        report_root=Path(core_analysis_directory).parent.parent,
        model_root=model_root,
        prediction_root=prediction_root,
        manifest_root=manifest_root,
    )
    ecg = validate_development_result(
        ecg_development.directory,
        model_root=model_root,
        prediction_root=prediction_root,
        manifest_root=manifest_root,
    )
    authorities = _resolve_development_authorities(
        core_analysis_id=str(core_analysis["analysis_id"]),
        ecg_development_id=str(ecg.manifest["development_id"]),
        report_root=report_root,
        model_root=model_root,
        prediction_root=prediction_root,
        manifest_root=manifest_root,
    )
    primary_repeats, ecg_repeats, final_family_authorities = authorities
    expected_subgroup = _expected_focused_subgroup_derivative(
        core_analysis_id=str(core_analysis["analysis_id"]),
        primary_repeats=primary_repeats,
        report_root=report_root,
        model_root=model_root,
        prediction_root=prediction_root,
        manifest_root=manifest_root,
    )
    if focused_subgroup_derivative != expected_subgroup:
        raise ManifestBuildError(
            "Focused subgroup derivative does not rederive from development evidence"
        )
    validate_focused_subgroup_derivative(
        report_root / "development-subgroups" / f"{expected_subgroup}.json",
        expected_id=expected_subgroup,
    )
    thresholds, claims = _derive_extension_claims(primary_repeats, ecg_repeats)
    semantic = {
        "core_analysis_id": core_analysis["analysis_id"],
        "ecg_development_id": ecg.manifest["development_id"],
        "primary_thresholds": thresholds,
        "final_family_authorities": final_family_authorities,
        "focused_subgroup_derivative": focused_subgroup_derivative,
    }
    result_id = canonical_scientific_id(ECG_EXTENSION_RESULT_PREFIX, semantic)
    document = {
        "ecg_extension_result_schema_version": 1,
        "ecg_extension_result_id": result_id,
        **semantic,
    }
    destination = report_root / "ecg-extension-results" / result_id
    stage = staging_directory(destination)
    try:
        (stage / "manifest.json").write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (stage / "claims.json").write_text(
            json.dumps(claims, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )

        def validator(path: Path, **kwargs: object) -> ValidatedEcgExtensionResult:
            return validate_ecg_extension_result(
                path,
                report_root=report_root,
                model_root=model_root,
                prediction_root=prediction_root,
                manifest_root=manifest_root,
                **kwargs,
            )

        install_immutable_directory(stage, destination, validator)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return validator(destination)


def validate_ecg_extension_result(
    directory: str | Path,
    *,
    report_root: str | Path | None = None,
    model_root: str | Path = "models/symile/development",
    prediction_root: str | Path = "private",
    manifest_root: str | Path = "data/manifests",
    enforce_directory_name: bool = True,
) -> ValidatedEcgExtensionResult:
    root = Path(directory)
    if (
        root.is_symlink()
        or not root.is_dir()
        or {path.name for path in root.iterdir()} != {"manifest.json", "claims.json"}
    ):
        raise ManifestBuildError("ECG extension result directory is invalid")
    raw = (root / "manifest.json").read_bytes()
    document = json.loads(raw)
    required = {
        "ecg_extension_result_schema_version",
        "ecg_extension_result_id",
        "core_analysis_id",
        "ecg_development_id",
        "primary_thresholds",
        "final_family_authorities",
        "focused_subgroup_derivative",
    }
    schema_version = (
        document.get("ecg_extension_result_schema_version") if isinstance(document, dict) else None
    )
    if (
        not isinstance(document, dict)
        or set(document) != required
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ManifestBuildError("ECG extension result contract is invalid")
    semantic = {
        key: document[key]
        for key in required - {"ecg_extension_result_schema_version", "ecg_extension_result_id"}
    }
    expected = canonical_scientific_id(ECG_EXTENSION_RESULT_PREFIX, semantic)
    if document["ecg_extension_result_id"] != expected or (
        enforce_directory_name and root.name != expected
    ):
        raise ManifestBuildError("ECG extension result identity is invalid")
    if set(document["primary_thresholds"]) != {"youden_j", "target_sensitivity"}:
        raise ManifestBuildError("ECG extension result must own exactly two thresholds")
    _validate_final_family_authorities(document["final_family_authorities"])
    result_root = Path(report_root) if report_root is not None else root.parent.parent
    authorities = _resolve_development_authorities(
        core_analysis_id=str(document["core_analysis_id"]),
        ecg_development_id=str(document["ecg_development_id"]),
        report_root=result_root,
        model_root=model_root,
        prediction_root=prediction_root,
        manifest_root=manifest_root,
    )
    primary_repeats, ecg_repeats, expected_final_authorities = authorities
    expected_subgroup = _expected_focused_subgroup_derivative(
        core_analysis_id=str(document["core_analysis_id"]),
        primary_repeats=primary_repeats,
        report_root=result_root,
        model_root=model_root,
        prediction_root=prediction_root,
        manifest_root=manifest_root,
    )
    if document["focused_subgroup_derivative"] != expected_subgroup:
        raise ManifestBuildError(
            "Focused subgroup derivative does not rederive from development evidence"
        )
    validate_focused_subgroup_derivative(
        result_root / "development-subgroups" / f"{expected_subgroup}.json",
        expected_id=expected_subgroup,
    )
    expected_thresholds, expected_claims = _derive_extension_claims(primary_repeats, ecg_repeats)
    if document["final_family_authorities"] != expected_final_authorities:
        raise ManifestBuildError(
            "Final family authorities do not rederive from development evidence"
        )
    if document["primary_thresholds"] != expected_thresholds:
        raise ManifestBuildError("Primary thresholds do not rederive from gated OOF evidence")
    if json.loads((root / "claims.json").read_bytes()) != expected_claims:
        raise ManifestBuildError("ECG extension claims do not rederive from OOF evidence")
    return ValidatedEcgExtensionResult(
        root,
        document,
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256((root / "claims.json").read_bytes()).hexdigest(),
        _VALIDATION_GUARD,
    )


def require_validated_ecg_extension_result(
    value: ValidatedEcgExtensionResult,
) -> ValidatedEcgExtensionResult:
    if not isinstance(value, ValidatedEcgExtensionResult) or value._guard is not _VALIDATION_GUARD:
        raise ManifestBuildError("A fully validated ECG extension result is required")
    if (
        value.directory.is_symlink()
        or not value.directory.is_dir()
        or {path.name for path in value.directory.iterdir()} != {"manifest.json", "claims.json"}
        or hashlib.sha256((value.directory / "manifest.json").read_bytes()).hexdigest()
        != value.manifest_sha256
        or hashlib.sha256((value.directory / "claims.json").read_bytes()).hexdigest()
        != value.claims_sha256
        or json.loads((value.directory / "manifest.json").read_bytes()) != value.manifest
    ):
        raise ManifestBuildError("Validated ECG extension result changed after validation")
    return value


def _validate_aggregate_oof(frame: pd.DataFrame, family: str) -> None:
    if (
        not isinstance(frame, pd.DataFrame)
        or tuple(frame.columns) != ("sample_id", "target", "logit", "probability")
        or frame.empty
        or frame.isna().any().any()
        or frame["sample_id"].duplicated().any()
        or tuple(frame["sample_id"].astype(str)) != tuple(sorted(frame["sample_id"].astype(str)))
        or set(frame["target"].tolist()) != {0, 1}
        or not np.isfinite(frame[["logit", "probability"]].to_numpy(dtype=np.float64)).all()
        or not np.array_equal(
            frame["probability"].to_numpy(dtype=np.float64),
            sigmoid(frame["logit"].to_numpy(dtype=np.float64)),
        )
    ):
        raise ManifestBuildError(f"{family} aggregate OOF evidence is invalid")


def _expected_focused_subgroup_derivative(
    *,
    core_analysis_id: str,
    primary_repeats: pd.DataFrame,
    report_root: Path,
    model_root: str | Path,
    prediction_root: str | Path,
    manifest_root: str | Path,
) -> str:
    development_root = report_root / "development"
    core = validate_analysis_result(
        development_root / "analyses" / core_analysis_id,
        report_root=development_root,
        model_root=model_root,
        prediction_root=prediction_root,
        manifest_root=manifest_root,
        expected_analysis_id=core_analysis_id,
    )
    cxr_id = core["family_development_ids"].get("cxr_densenet")
    if not isinstance(cxr_id, str):
        raise ManifestBuildError("Focused subgroup derivative lacks CXR development authority")
    cxr = validate_development_result(
        development_root / "families" / cxr_id,
        model_root=model_root,
        prediction_root=prediction_root,
        manifest_root=manifest_root,
        expected_development_id=cxr_id,
    )
    cxr_repeats = validated_development_repeat_oof(cxr, prediction_root)
    first_fold = cxr.manifest["fold_packages"][0]["fold_package_id"]
    config = with_runtime(
        load_symile_development_config(
            Path(model_root) / "packages" / first_fold / CONFIG_FILENAME
        ),
        manifest_directory=manifest_root,
    )
    cohort = load_symile_development_cohort(config).frame
    attributes = cohort.loc[:, ["sample_id", "age_years", "sex", "view_position"]].copy()
    attributes["observed_lab_count"] = cohort.loc[:, LAB_OBSERVED_COLUMNS].sum(axis=1)
    summary = focused_development_subgroups(cxr_repeats, primary_repeats, attributes)
    return _focused_subgroup_id(_focused_subgroup_bytes(summary))


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


def _focused_subgroup_id(encoded: bytes) -> str:
    return f"focused-subgroup-{hashlib.sha256(encoded).hexdigest()}"


def _resolve_development_authorities(
    *,
    core_analysis_id: str,
    ecg_development_id: str,
    report_root: Path,
    model_root: str | Path,
    prediction_root: str | Path,
    manifest_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    development_root = report_root / "development"
    core = validate_analysis_result(
        development_root / "analyses" / core_analysis_id,
        report_root=development_root,
        model_root=model_root,
        prediction_root=prediction_root,
        manifest_root=manifest_root,
        expected_analysis_id=core_analysis_id,
    )
    development_ids = dict(core["family_development_ids"])
    development_ids[SYMILE_ECG_GATED_FAMILY] = ecg_development_id
    resolved: dict[str, ValidatedDevelopmentResult] = {}
    final_family_authorities: dict[str, dict[str, object]] = {}
    for family in FINAL_PACKAGE_POLICY:
        development_id = development_ids.get(family)
        if not isinstance(development_id, str):
            raise ManifestBuildError("ECG extension lacks a final-family development authority")
        development = validate_development_result(
            development_root / "families" / development_id,
            model_root=model_root,
            prediction_root=prediction_root,
            manifest_root=manifest_root,
        )
        if (
            development.manifest.get("development_id") != development_id
            or development.manifest.get("family_id") != family
        ):
            raise ManifestBuildError("ECG extension development authority is invalid")
        resolved[family] = development
        final_family_authorities[family] = {
            "development_id": None if family == "labs_logistic" else development_id,
            "final_training_budget": development.manifest["final_training_budget"],
            "final_input": final_input_projection_from_development(
                development.manifest["scientific_context"]["fit_config"]
            ),
            "pretrained_scientific_identity": _development_pretrained_identity(
                development, family=family, model_root=Path(model_root)
            ),
        }
    _validate_final_family_authorities(final_family_authorities)
    primary = resolved["cxr_labs_gated"]
    ecg = resolved[SYMILE_ECG_GATED_FAMILY]
    primary_repeats = validated_development_repeat_oof(primary, prediction_root)
    ecg_repeats = validated_development_repeat_oof(ecg, prediction_root)
    if not primary_repeats[["sample_id", "target", "repeat_seed"]].equals(
        ecg_repeats[["sample_id", "target", "repeat_seed"]]
    ):
        raise ManifestBuildError("ECG extension repeat OOF evidence is not exactly aligned")
    return primary_repeats, ecg_repeats, final_family_authorities


def _validate_final_family_authorities(value: object) -> None:
    if not isinstance(value, dict) or set(value) != set(FINAL_PACKAGE_POLICY):
        raise ManifestBuildError("ECG extension final-family authority membership is invalid")
    for family, authority in value.items():
        if (
            not isinstance(authority, dict)
            or set(authority)
            != {
                "development_id",
                "final_training_budget",
                "final_input",
                "pretrained_scientific_identity",
            }
            or not isinstance(authority["final_input"], dict)
            or authority["final_input"].get("family", {}).get("family_id") != family
        ):
            raise ManifestBuildError("ECG extension final-family authority is invalid")
        development_id = authority["development_id"]
        budget = authority["final_training_budget"]
        pretrained = authority["pretrained_scientific_identity"]
        if (family == "cxr_densenet") != isinstance(pretrained, dict):
            raise ManifestBuildError("Final family pretrained authority is invalid")
        if family == "cxr_densenet" and (
            set(pretrained) != {"declared_name", "stable_identifier", "sha256"}
            or not isinstance(pretrained["declared_name"], str)
            or not pretrained["declared_name"]
            or not isinstance(pretrained["stable_identifier"], str)
            or not pretrained["stable_identifier"]
            or not isinstance(pretrained["sha256"], str)
            or len(pretrained["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in pretrained["sha256"])
        ):
            raise ManifestBuildError("Final family pretrained authority is invalid")
        if family == "labs_logistic":
            if development_id is not None or budget is not None:
                raise ManifestBuildError("Final logistic authority must not own a budget")
        elif (
            not isinstance(development_id, str)
            or not development_id.startswith("development-")
            or len(development_id) != len("development-") + 64
            or isinstance(budget, bool)
            or not isinstance(budget, int)
            or budget <= 0
        ):
            raise ManifestBuildError("Final family budget authority is invalid")


def _development_pretrained_identity(
    development: ValidatedDevelopmentResult, *, family: str, model_root: Path
) -> dict[str, object] | None:
    identities = []
    for reference in development.manifest["fold_packages"]:
        package = validate_fold_package(
            model_root / "packages" / reference["fold_package_id"],
            expected_fold_package_id=reference["fold_package_id"],
        )
        fingerprint = package.manifest["lineage"]["pretrained_weight"]
        if fingerprint is not None:
            identities.append(pretrained_weight_semantic_identity(fingerprint))
    if family != "cxr_densenet":
        if identities:
            raise ManifestBuildError("Non-CXR development declares pretrained weight lineage")
        return None
    if len(identities) != len(development.manifest["fold_packages"]) or any(
        identity != identities[0] for identity in identities[1:]
    ):
        raise ManifestBuildError("Development CXR folds disagree on pretrained identity")
    return identities[0]


def _derive_extension_claims(
    primary_repeats: pd.DataFrame, ecg_repeats: pd.DataFrame
) -> tuple[dict[str, float], dict[str, object]]:
    primary_ensemble = development_mean_logit_ensemble(primary_repeats)
    ecg_ensemble = development_mean_logit_ensemble(ecg_repeats)
    _validate_aggregate_oof(primary_ensemble, "Primary gated")
    _validate_aggregate_oof(ecg_ensemble, "ECG gated")
    thresholds = {
        "youden_j": youden_threshold(primary_ensemble["target"], primary_ensemble["probability"]),
        "target_sensitivity": sensitivity_threshold(
            primary_ensemble["target"], primary_ensemble["probability"]
        ),
    }
    ecg_repeat_metrics: dict[str, object] = {}
    repeat_effects: dict[str, object] = {}
    for seed in REPEAT_SEEDS:
        primary = primary_repeats.loc[primary_repeats["repeat_seed"] == seed]
        ecg = ecg_repeats.loc[ecg_repeats["repeat_seed"] == seed]
        primary_metrics = metrics(primary["target"], sigmoid(primary["logit"]))
        ecg_metrics = metrics(ecg["target"], sigmoid(ecg["logit"]))
        ecg_repeat_metrics[str(seed)] = ecg_metrics
        repeat_effects[str(seed)] = {
            name: ecg_metrics[name] - primary_metrics[name] for name in headline_metric_names()
        }
    primary_metrics = metrics(primary_ensemble["target"], primary_ensemble["probability"])
    ecg_metrics = metrics(ecg_ensemble["target"], ecg_ensemble["probability"])
    claims = {
        "metric_policy": METRIC_POLICY,
        "ecg_repeat_metrics": ecg_repeat_metrics,
        "ecg_vs_gated_repeat_effects": repeat_effects,
        "ecg_ensemble_metrics": ecg_metrics,
        "ecg_vs_gated_ensemble_effect": {
            name: ecg_metrics[name] - primary_metrics[name] for name in headline_metric_names()
        },
    }
    return thresholds, claims
