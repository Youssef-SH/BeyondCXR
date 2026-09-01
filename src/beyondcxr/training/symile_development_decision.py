"""Unified scientific decision authority for completed Symile development."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_schemas import LABEL_POLICY_VERSION, REPEAT_SEEDS, TASK_ID
from beyondcxr.training.symile_development_reports import SYMILE_DEVELOPMENT_COMPARISON_POLICY
from beyondcxr.training.symile_execution import (
    ValidatedSymileProspectiveMutation,
    require_symile_prospective_mutation,
    require_symile_prospective_path,
)
from beyondcxr.training.symile_families import (
    DEVELOPMENT_FAMILY_ORDER,
    FINAL_PACKAGE_POLICY,
    SYMILE_ECG_GATED_FAMILY,
)
from beyondcxr.training.symile_final_packages import final_input_projection_from_development
from beyondcxr.training.symile_statistics import (
    METRIC_POLICY,
    development_mean_logit_ensemble,
    headline_metric_names,
    metrics,
    sensitivity_threshold,
    sigmoid,
    youden_threshold,
)
from beyondcxr.utils.package_identity import pretrained_weight_semantic_identity
from beyondcxr.utils.publication import install_immutable_directory, staging_directory
from beyondcxr.utils.semantic_identity import canonical_scientific_id
from beyondcxr.utils.symile_publication import (
    ValidatedDevelopmentResult,
    validate_development_result,
    validate_fold_package,
    validated_development_repeat_oof,
)
from beyondcxr.utils.validation_context import memoized_directory_validation

DEVELOPMENT_DECISION_SCHEMA_VERSION = 1
DEVELOPMENT_DECISION_PREFIX = "development-decision-"
DEVELOPMENT_DECISION_POLICY = {
    "policy_version": "symile-development-decision-v1",
    "family_order": list(DEVELOPMENT_FAMILY_ORDER),
    "comparison_policy": SYMILE_DEVELOPMENT_COMPARISON_POLICY,
    "selected_families": list(FINAL_PACKAGE_POLICY),
    "threshold_derivation": {
        "source_family": "cxr_labs_gated",
        "ensemble": "mean repeat logits then sigmoid",
        "youden_j": "all ROC thresholds; highest finite threshold on ties",
        "target_sensitivity": "highest finite threshold meeting prespecified sensitivity",
        "target_sensitivity_value": 0.9,
    },
}
_VALIDATION_GUARD = object()


@dataclass(frozen=True)
class ValidatedDevelopmentDecision:
    directory: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    claims_sha256: str
    report_root: Path
    model_root: Path
    prediction_root: Path
    manifest_root: Path
    _guard: object

    @property
    def decision_id(self) -> str:
        return str(self.manifest["development_decision_id"])


def publish_development_decision(
    *,
    prospective_mutation: ValidatedSymileProspectiveMutation,
    report_root: str | Path,
    family_developments: Mapping[str, ValidatedDevelopmentResult],
    model_root: str | Path = "models/symile/development",
    prediction_root: str | Path = "private",
    manifest_root: str | Path = "data/manifests",
) -> ValidatedDevelopmentDecision:
    """Publish the sole cross-family scientific conclusion of development."""
    require_symile_prospective_mutation(prospective_mutation)
    for path, context in (
        (report_root, "Symile decision report root"),
        (model_root, "Symile decision model root"),
        (prediction_root, "Symile decision prediction root"),
        (manifest_root, "Symile decision manifest root"),
    ):
        require_symile_prospective_path(prospective_mutation, path, context)
    root = Path(report_root)
    if tuple(family_developments) != DEVELOPMENT_FAMILY_ORDER:
        raise ManifestBuildError("Symile development decision family order is invalid")
    family_ids: dict[str, str] = {}
    for family, supplied in family_developments.items():
        if not isinstance(supplied, ValidatedDevelopmentResult):
            raise ManifestBuildError("Symile development decision family authority is invalid")
        development_id = str(supplied.manifest.get("development_id"))
        validated = validate_development_result(
            supplied.directory,
            model_root=model_root,
            prediction_root=prediction_root,
            manifest_root=manifest_root,
            expected_development_id=development_id,
        )
        if validated.manifest.get("family_id") != family:
            raise ManifestBuildError("Symile development decision family authority is invalid")
        family_ids[family] = development_id
    document, claims = _decision_document(
        family_ids=family_ids,
        report_root=root,
        model_root=model_root,
        prediction_root=prediction_root,
        manifest_root=manifest_root,
    )
    destination = root / "development-decisions" / document["development_decision_id"]
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

        def validator(path: Path, **kwargs: object) -> ValidatedDevelopmentDecision:
            return validate_development_decision(
                path,
                report_root=root,
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


@memoized_directory_validation("symile-development-decision")
def validate_development_decision(
    directory: str | Path,
    *,
    report_root: str | Path | None = None,
    model_root: str | Path = "models/symile/development",
    prediction_root: str | Path = "private",
    manifest_root: str | Path = "data/manifests",
    enforce_directory_name: bool = True,
) -> ValidatedDevelopmentDecision:
    """Reconstruct and validate the unified decision from family authorities."""
    root = Path(directory)
    if (
        root.is_symlink()
        or not root.is_dir()
        or {path.name for path in root.iterdir()} != {"manifest.json", "claims.json"}
        or any(path.is_symlink() or not path.is_file() for path in root.iterdir())
    ):
        raise ManifestBuildError("Symile development decision directory is invalid")
    raw = (root / "manifest.json").read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Symile development decision is unreadable") from exc
    fields = {
        "development_decision_schema_version",
        "development_decision_id",
        "dataset_id",
        "bundle_id",
        "cv_assignment_id",
        "task",
        "family_development_ids",
        "decision_policy",
        "selected_family_authorities",
        "primary_thresholds",
    }
    if (
        not isinstance(document, dict)
        or set(document) != fields
        or type(document["development_decision_schema_version"]) is not int
        or document["development_decision_schema_version"] != DEVELOPMENT_DECISION_SCHEMA_VERSION
        or document["dataset_id"] != "symile"
        or document["task"] != {"task_id": TASK_ID, "label_policy_version": LABEL_POLICY_VERSION}
        or document["decision_policy"] != DEVELOPMENT_DECISION_POLICY
    ):
        raise ManifestBuildError("Symile development decision contract is invalid")
    family_ids = _validate_family_development_ids(document["family_development_ids"])
    result_root = Path(report_root) if report_root is not None else root.parent.parent
    expected, claims = _decision_document(
        family_ids=family_ids,
        report_root=result_root,
        model_root=model_root,
        prediction_root=prediction_root,
        manifest_root=manifest_root,
    )
    if document != expected or (
        enforce_directory_name and root.name != document["development_decision_id"]
    ):
        raise ManifestBuildError("Symile development decision differs from its authorities")
    claims_raw = (root / "claims.json").read_bytes()
    if (
        claims_raw
        != (
            json.dumps(claims, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    ):
        raise ManifestBuildError("Symile development claims differ from family evidence")
    return ValidatedDevelopmentDecision(
        root.resolve(),
        document,
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(claims_raw).hexdigest(),
        result_root.resolve(),
        Path(model_root).resolve(),
        Path(prediction_root).resolve(),
        Path(manifest_root).resolve(),
        _VALIDATION_GUARD,
    )


def require_validated_development_decision(
    value: ValidatedDevelopmentDecision,
) -> ValidatedDevelopmentDecision:
    if not isinstance(value, ValidatedDevelopmentDecision) or value._guard is not _VALIDATION_GUARD:
        raise ManifestBuildError("A fully validated Symile development decision is required")
    observed = validate_development_decision(
        value.directory,
        report_root=value.report_root,
        model_root=value.model_root,
        prediction_root=value.prediction_root,
        manifest_root=value.manifest_root,
        enforce_directory_name=False,
    )
    if (
        observed.manifest != value.manifest
        or observed.manifest_sha256 != value.manifest_sha256
        or observed.claims_sha256 != value.claims_sha256
    ):
        raise ManifestBuildError("Validated Symile development decision changed after validation")
    return value


def _decision_document(
    *,
    family_ids: dict[str, str],
    report_root: Path,
    model_root: str | Path,
    prediction_root: str | Path,
    manifest_root: str | Path,
) -> tuple[dict[str, Any], dict[str, object]]:
    if tuple(family_ids) != DEVELOPMENT_FAMILY_ORDER:
        raise ManifestBuildError("Symile development decision family order is invalid")
    development_root = report_root / "development"
    developments = {
        family: validate_development_result(
            development_root / "families" / family_ids[family],
            model_root=model_root,
            prediction_root=prediction_root,
            manifest_root=manifest_root,
            expected_development_id=family_ids[family],
        )
        for family in DEVELOPMENT_FAMILY_ORDER
    }
    if any(
        developments[family].manifest["family_id"] != family for family in DEVELOPMENT_FAMILY_ORDER
    ):
        raise ManifestBuildError("Symile development decision family authority is invalid")
    contexts = [item.manifest["scientific_context"] for item in developments.values()]
    coordinates = {
        (context["bundle_id"], context["cv_assignment_id"], context["task_id"])
        for context in contexts
    }
    if len(coordinates) != 1:
        raise ManifestBuildError("Symile development authorities disagree on scientific context")
    bundle_id, cv_assignment_id, task_id = coordinates.pop()
    if task_id != TASK_ID:
        raise ManifestBuildError("Symile development decision task is invalid")
    primary, ecg, selected = _resolve_development_authorities(
        developments=developments,
        model_root=model_root,
        prediction_root=prediction_root,
    )
    _validate_selected_family_authorities(selected)
    thresholds, claims = _derive_decision_claims(primary, ecg)
    semantic = {
        "dataset_id": "symile",
        "bundle_id": bundle_id,
        "cv_assignment_id": cv_assignment_id,
        "task": {"task_id": TASK_ID, "label_policy_version": LABEL_POLICY_VERSION},
        "family_development_ids": [
            {"family_id": family, "development_id": family_ids[family]}
            for family in DEVELOPMENT_FAMILY_ORDER
        ],
        "decision_policy": DEVELOPMENT_DECISION_POLICY,
        "selected_family_authorities": selected,
        "primary_thresholds": thresholds,
    }
    decision_id = canonical_scientific_id(
        DEVELOPMENT_DECISION_PREFIX,
        semantic,
        domain="development-decision",
        schema_version=1,
    )
    return (
        {
            "development_decision_schema_version": DEVELOPMENT_DECISION_SCHEMA_VERSION,
            "development_decision_id": decision_id,
            **semantic,
        },
        claims,
    )


def _validate_family_development_ids(value: object) -> dict[str, str]:
    """Validate exact ordered seven-family decision membership using metadata only."""
    if (
        not isinstance(value, list)
        or any(
            not isinstance(item, dict)
            or set(item) != {"family_id", "development_id"}
            or not isinstance(item["development_id"], str)
            or not item["development_id"].startswith("development-")
            or len(item["development_id"]) != len("development-") + 64
            or any(
                character not in "0123456789abcdef"
                for character in item["development_id"][len("development-") :]
            )
            for item in value
        )
        or [item["family_id"] for item in value] != list(DEVELOPMENT_FAMILY_ORDER)
    ):
        raise ManifestBuildError("Symile development decision family membership is invalid")
    return {item["family_id"]: item["development_id"] for item in value}


def _resolve_development_authorities(
    *,
    developments: Mapping[str, ValidatedDevelopmentResult],
    model_root: str | Path,
    prediction_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    resolved: dict[str, ValidatedDevelopmentResult] = {}
    selected_family_authorities: dict[str, dict[str, object]] = {}
    for family in FINAL_PACKAGE_POLICY:
        development = developments.get(family)
        if not isinstance(development, ValidatedDevelopmentResult):
            raise ManifestBuildError("Development decision lacks a selected-family authority")
        development_id = development.manifest.get("development_id")
        if not isinstance(development_id, str) or development.manifest.get("family_id") != family:
            raise ManifestBuildError("Development decision family authority is invalid")
        resolved[family] = development
        selected_family_authorities[family] = {
            "development_id": development_id,
            "final_training_budget": development.manifest["final_training_budget"],
            "final_input": final_input_projection_from_development(
                development.manifest["scientific_context"]["fit_config"]
            ),
            "pretrained_scientific_identity": _development_pretrained_identity(
                development, family=family, model_root=Path(model_root)
            ),
        }
    _validate_selected_family_authorities(selected_family_authorities)
    primary = resolved["cxr_labs_gated"]
    ecg = resolved[SYMILE_ECG_GATED_FAMILY]
    primary_repeats = validated_development_repeat_oof(primary, prediction_root)
    ecg_repeats = validated_development_repeat_oof(ecg, prediction_root)
    if not primary_repeats[["sample_id", "target", "repeat_seed"]].equals(
        ecg_repeats[["sample_id", "target", "repeat_seed"]]
    ):
        raise ManifestBuildError("Development decision repeat OOF evidence is not aligned")
    return primary_repeats, ecg_repeats, selected_family_authorities


def _validate_selected_family_authorities(value: object) -> None:
    if not isinstance(value, dict) or set(value) != set(FINAL_PACKAGE_POLICY):
        raise ManifestBuildError("Development decision selected-family membership is invalid")
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
            raise ManifestBuildError("Development decision selected-family authority is invalid")
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
            if (
                not isinstance(development_id, str)
                or not development_id.startswith("development-")
                or len(development_id) != len("development-") + 64
                or budget is not None
            ):
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


def _derive_decision_claims(
    primary_repeats: pd.DataFrame, ecg_repeats: pd.DataFrame
) -> tuple[dict[str, float], dict[str, object]]:
    primary_ensemble = development_mean_logit_ensemble(primary_repeats)
    ecg_ensemble = development_mean_logit_ensemble(ecg_repeats)
    _validate_aggregate_oof(primary_ensemble, "Primary gated")
    _validate_aggregate_oof(ecg_ensemble, "ECG gated")
    thresholds = {
        "youden_j": youden_threshold(primary_ensemble["target"], primary_ensemble["probability"]),
        "target_sensitivity": sensitivity_threshold(
            primary_ensemble["target"], primary_ensemble["probability"], target=0.9
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
