"""Derive binding and aggregate projection from validated restored scientific campaign state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_schemas import REPEAT_SEEDS
from beyondcxr.release.binding import ResultBinding, validate_result_binding
from beyondcxr.release.projection import (
    RELIABILITY_FAMILIES,
    PublicResultProjection,
    project_public_results,
)
from beyondcxr.training.symile_campaign_control import predictor_views
from beyondcxr.training.symile_families import (
    FINAL_PACKAGE_POLICY,
    SYMILE_CORE_DEVELOPMENT_FAMILIES,
    SYMILE_ECG_GATED_FAMILY,
)
from beyondcxr.training.symile_statistics import SUBGROUP_POLICY, raw_probability_metrics, sigmoid
from beyondcxr.utils.symile_publication import validated_development_repeat_oof

if TYPE_CHECKING:
    from beyondcxr.training.symile_campaign import RestoredCampaignClosure


def derive_result_binding(campaign: RestoredCampaignClosure) -> ResultBinding:
    """Derive canonical coordinates exclusively from a validated campaign closure."""
    frozen = campaign.freeze.manifest
    analysis = campaign.analysis
    extension = campaign.extension.manifest
    bundle = frozen["bundle"]
    cv_ids = {
        development.manifest["scientific_context"]["cv_assignment_id"]
        for development in campaign.developments
    }
    if len(cv_ids) != 1:
        raise ManifestBuildError("Validated campaign development has inconsistent CV lineage")
    document = {
        "result_binding_schema_version": 1,
        "science_execution": {
            "git_commit": frozen["science_git_commit"],
        },
        "data": {
            "bundle_id": bundle["bundle_id"],
            "split_assignment_id": bundle["split_assignment_id"],
            "cv_assignment_id": next(iter(cv_ids)),
        },
        "development": {
            "core_analysis_id": analysis["analysis_id"],
            "family_development_ids": analysis["family_development_ids"],
            "ecg_development_id": extension["ecg_development_id"],
            "ecg_extension_result_id": extension["ecg_extension_result_id"],
        },
        "held_out": {
            "pretest_freeze_id": campaign.freeze.freeze_id,
            "final_package_ids": [package.package_id for package in campaign.packages],
            "prediction_ids": [prediction.prediction_id for prediction in campaign.predictions],
            "global_result_id": campaign.global_result.result_id,
        },
    }
    return validate_result_binding(document)


def validate_release_inputs(
    binding: ResultBinding,
    campaign: RestoredCampaignClosure,
    artifact_root: str | Path,
    *,
    export_manifest_sha256: str,
) -> PublicResultProjection:
    """Validate one result binding and aggregate public projection against a restored campaign."""
    root = Path(artifact_root)
    frozen = campaign.freeze.manifest
    bundle = frozen["bundle"]
    expected = derive_result_binding(campaign)
    if binding != expected:
        raise ManifestBuildError("Result binding differs from validated campaign state")
    bundle_manifest = _json(
        root
        / "data"
        / "manifests"
        / "symile"
        / "bundles"
        / binding.data.bundle_id
        / "manifest.json"
    )
    extension_claims = _json(campaign.extension.directory / "claims.json")
    global_claims = _json(campaign.global_result.directory / "claims.json")
    global_claims["discrimination_curves"] = _held_out_curves(campaign)
    development_reliability = _development_reliability(campaign, root / "private")
    cxr_family, primary_gated_family = RELIABILITY_FAMILIES
    cxr_packages = [
        package
        for package in campaign.packages
        if package.manifest["input"]["family"]["family_id"] == cxr_family
    ]
    projection = project_public_results(
        cohort=_cohort_projection(
            bundle_manifest,
            root
            / "data"
            / "manifests"
            / "symile"
            / "bundles"
            / binding.data.bundle_id
            / "samples.parquet",
        ),
        core_analysis=campaign.analysis,
        extension_claims=extension_claims,
        global_claims=global_claims,
        development_reliability=development_reliability,
        thresholds=campaign.extension.manifest["primary_thresholds"],
        subgroups=_subgroup_projection(campaign.subgroup),
        provenance={
            "science_git_commit": binding.science_execution.git_commit,
            "science_dependency_lock_sha256": frozen["dependency_lock_sha256"],
            "bundle_id": binding.data.bundle_id,
            "bundle_manifest_sha256": bundle["bundle_manifest_sha256"],
            "split_assignment_id": binding.data.split_assignment_id,
            "cv_assignment_id": binding.data.cv_assignment_id,
            "pretest_freeze_id": binding.held_out.pretest_freeze_id,
            "global_result_id": binding.held_out.global_result_id,
            "export_manifest_sha256": export_manifest_sha256,
            "primary_package_ids": [
                package.package_id
                for package in campaign.packages
                if package.manifest["input"]["family"]["family_id"] == primary_gated_family
            ],
            "pretrained_scientific_identity": cxr_packages[0].manifest[
                "pretrained_scientific_identity"
            ],
            "pretrained_weight_materializations": [
                {
                    "seed": package.manifest["seed_policy"],
                    **package.manifest["pretrained_weight_fingerprint"],
                }
                for package in cxr_packages
            ],
        },
    )
    return projection


def _cohort_projection(manifest: dict[str, Any], samples_path: Path) -> dict[str, Any]:
    qualification = manifest["qualification"]
    strict = qualification["strict_pneumonia_counts"]
    task = manifest["tasks"]["pneumonia_strict"]
    membership = manifest["membership"]
    samples = pq.read_table(
        samples_path,
        columns=["official_split", "pneumonia_state", "age_years", "sex", "view_position"],
    ).to_pandas()
    eligible = samples.loc[samples["pneumonia_state"].isin((0, 1))]
    scopes = {
        "train": eligible.loc[eligible["official_split"] == "train"],
        "validation": eligible.loc[eligible["official_split"] == "validation"],
        "development": eligible.loc[eligible["official_split"].isin(("train", "validation"))],
        "test": eligible.loc[eligible["official_split"] == "test"],
    }
    descriptions = []
    for scope, frame in scopes.items():
        for low, high in SUBGROUP_POLICY["age"]:
            mask = frame["age_years"] >= low
            if high is not None:
                mask &= frame["age_years"] <= high
            descriptions.append(
                {
                    "scope": scope,
                    "attribute": "age",
                    "category": f"{low}-{high or 'plus'}",
                    "count": int(mask.sum()),
                }
            )
        for attribute, categories in (
            ("sex", SUBGROUP_POLICY["sex"]),
            ("view_position", SUBGROUP_POLICY["view_position"]),
        ):
            descriptions.extend(
                {
                    "scope": scope,
                    "attribute": attribute,
                    "category": category,
                    "count": int((frame[attribute] == category).sum()),
                }
                for category in categories
            )
    return {
        "rows": [
            {"scope": scope, **strict[scope]}
            for scope in ("train", "validation", "development", "test")
        ],
        "source_reconciliation": {
            key: qualification["membership_counts"][key]
            for key in ("full_admissions", "official_admissions", "excluded_admissions")
        },
        "construction": [
            {
                "item": "Train inclusion",
                "status": f"All unique admissions in {membership['source']['train']}",
            },
            {
                "item": "Validation inclusion",
                "status": f"All unique admissions in {membership['source']['validation']}",
            },
            {
                "item": "Held-out test inclusion",
                "status": (
                    f"{membership['source']['test']}; selector: {membership['test_selector']}"
                ),
            },
            {
                "item": "Classification exclusions",
                "status": (
                    "Retrieval-negative candidates and val_retrieval.csv are excluded from "
                    "classification"
                ),
            },
        ],
        "endpoint": {
            key: task[key]
            for key in (
                "task_id",
                "label_policy_version",
                "label_source",
                "positive",
                "negative",
                "excluded",
            )
        }
        | {
            "selection_provenance": (
                "Broader and alternative label policies were audited prospectively before "
                "supervised modeling; their larger eligible cohorts resulted from changed target "
                "semantics and/or negative-class spectrum. The study therefore froze explicit "
                "Pneumonia = 1 versus explicit Pneumonia = 0 and excludes uncertain or "
                "unmentioned labels."
            )
        },
        "descriptions": descriptions,
    }


def _subgroup_projection(document: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, value in document["strata"].items():
        base: dict[str, Any] = {
            "stratum": name,
            "supported": value["supported"],
            "n": value["n"],
            "positives": value.get("positives"),
            "negatives": value.get("negatives"),
        }
        if not value["supported"]:
            rows.append({**base, "estimate": "Unavailable", "repeat": None})
            continue
        estimates = [
            ("OOF repeat", seed, value["repeat_metrics"][str(seed)]) for seed in REPEAT_SEEDS
        ]
        estimates.append(("Mean-logit OOF ensemble", None, value["mean_logit_ensemble"]))
        for estimate, repeat, result in estimates:
            row = {**base, "estimate": estimate, "repeat": repeat}
            for metric in ("roc_auc", "average_precision", "brier_score"):
                row[f"cxr_{metric}"] = result["cxr"][metric]
                row[f"gated_{metric}"] = result["gated"][metric]
                row[f"effect_{metric}"] = result["gated"][metric] - result["cxr"][metric]
            rows.append(row)
    return {"rows": rows, "policy": document["policy"]}


def _held_out_curves(campaign: RestoredCampaignClosure) -> dict[str, Any]:
    views = predictor_views(campaign.packages, campaign.predictions)
    if tuple(views) != tuple(FINAL_PACKAGE_POLICY):
        raise ManifestBuildError("Release curve membership differs from fixed predictor views")
    return {
        family: _curves(frame["target"], frame["probability"]) for family, frame in views.items()
    }


def _development_reliability(
    campaign: RestoredCampaignClosure, private_root: Path
) -> dict[str, Any]:
    by_family = {item.manifest["family_id"]: item for item in campaign.developments}
    expected = set(SYMILE_CORE_DEVELOPMENT_FAMILIES) | {SYMILE_ECG_GATED_FAMILY}
    if set(by_family) != expected:
        raise ManifestBuildError("Restored development membership is incomplete")
    curves: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    for family in RELIABILITY_FAMILIES:
        frame = validated_development_repeat_oof(by_family[family], private_root)
        wide = frame.pivot(index=["sample_id", "target"], columns="repeat_seed", values="logit")
        probabilities = sigmoid(wide.to_numpy(dtype=np.float64).mean(axis=1))
        targets = wide.index.get_level_values("target").to_numpy(dtype=np.int8)
        observed, predicted = calibration_curve(
            targets, probabilities, n_bins=10, strategy="uniform"
        )
        curves[family] = {
            "mean_predicted_probability": predicted.tolist(),
            "observed_positive_fraction": observed.tolist(),
        }
        calibration[family] = raw_probability_metrics(targets, probabilities)
    return {"curves": curves, "calibration": calibration}


def _curves(targets: Any, probabilities: Any) -> dict[str, Any]:
    false_positive, true_positive, _ = roc_curve(targets, probabilities)
    precision, recall, _ = precision_recall_curve(targets, probabilities)
    return {
        "roc": {
            "false_positive_rate": false_positive.tolist(),
            "true_positive_rate": true_positive.tolist(),
        },
        "precision_recall": {"recall": recall.tolist(), "precision": precision.tolist()},
    }


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Validated release input became unreadable") from exc
    if not isinstance(value, dict):
        raise ManifestBuildError("Validated release input must be a JSON object")
    return value


def export_manifest_sha256(path: str | Path) -> str:
    """Hash canonical export-manifest bytes without extracting restricted members."""
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            raw = archive.read("export-manifest.json")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ManifestBuildError("Campaign export manifest is unavailable") from exc
    return hashlib.sha256(raw).hexdigest()
