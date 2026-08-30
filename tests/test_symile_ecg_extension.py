from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

import beyondcxr.training.symile_development_decision as decision
import beyondcxr.training.symile_development_reports as derivatives
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.symile_families import FINAL_PACKAGE_POLICY
from beyondcxr.training.symile_statistics import (
    metrics,
    sensitivity_threshold,
    sigmoid,
    youden_threshold,
)
from beyondcxr.utils.semantic_identity import canonical_json_bytes
from beyondcxr.utils.symile_publication import ValidatedDevelopmentResult


def test_focused_subgroup_derivative_requires_exact_v1_contract(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    path = derivatives.publish_focused_subgroup_derivative(
        report_root=report_root, summary={"policy": {}, "strata": {}}
    )
    document = json.loads(path.read_bytes())
    assert document == {
        "focused_subgroup_derivative_schema_version": 1,
        "policy": {},
        "strata": {},
    }
    assert derivatives.validate_focused_subgroup_derivative(path) == document

    for mutation in ("missing", "wrong", "boolean", "float", "extra"):
        altered = dict(document)
        if mutation == "missing":
            altered.pop("focused_subgroup_derivative_schema_version")
        elif mutation == "wrong":
            altered["focused_subgroup_derivative_schema_version"] = 2
        elif mutation == "boolean":
            altered["focused_subgroup_derivative_schema_version"] = True
        elif mutation == "float":
            altered["focused_subgroup_derivative_schema_version"] = 1.0
        else:
            altered["unexpected"] = True
        path.write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(ManifestBuildError, match="contract"):
            derivatives.validate_focused_subgroup_derivative(path)


def _family_references() -> list[dict[str, str]]:
    return [
        {"family_id": family, "development_id": "development-" + f"{index:x}" * 64}
        for index, family in enumerate(decision.DEVELOPMENT_FAMILY_ORDER, start=1)
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "duplicate",
        "wrong-family",
        "wrong-order",
        "short-id",
        "nonhex-id",
    ],
)
def test_development_decision_requires_exact_seven_family_membership(mutation: str) -> None:
    references = _family_references()
    if mutation == "missing":
        references.pop()
    elif mutation == "extra":
        references.append({"family_id": "fabricated", "development_id": "development-" + "f" * 64})
    elif mutation == "duplicate":
        references[1] = dict(references[0])
    elif mutation == "wrong-family":
        references[0]["family_id"] = "fabricated"
    elif mutation == "wrong-order":
        references.reverse()
    elif mutation == "short-id":
        references[0]["development_id"] = "development-short"
    else:
        references[0]["development_id"] = "development-" + "z" * 64

    with pytest.raises(ManifestBuildError, match="family membership"):
        decision._validate_family_development_ids(references)


def test_development_decision_accepts_exact_ordered_seven_family_metadata() -> None:
    references = _family_references()
    assert decision._validate_family_development_ids(references) == {
        item["family_id"]: item["development_id"] for item in references
    }


def test_selected_final_family_authorities_require_exact_membership_and_budgets() -> None:
    authorities = {
        family: {
            "development_id": "development-" + f"{index:x}" * 64,
            "final_training_budget": None if family == "labs_logistic" else index,
            "final_input": {"family": {"family_id": family}},
            "pretrained_scientific_identity": (
                {
                    "declared_name": "densenet121-res224-chex",
                    "stable_identifier": "weight-v1",
                    "sha256": "a" * 64,
                }
                if family == "cxr_densenet"
                else None
            ),
        }
        for index, family in enumerate(FINAL_PACKAGE_POLICY, start=1)
    }
    decision._validate_selected_family_authorities(authorities)

    missing = dict(authorities)
    missing.pop(next(iter(missing)))
    with pytest.raises(ManifestBuildError, match="membership"):
        decision._validate_selected_family_authorities(missing)

    wrong_budget = deepcopy(authorities)
    wrong_budget["cxr_labs_ecg_gated"]["final_training_budget"] = 0
    with pytest.raises(ManifestBuildError, match="budget"):
        decision._validate_selected_family_authorities(wrong_budget)


def _repeat_oof(offset: float) -> pd.DataFrame:
    rows = []
    for seed in (17, 42, 2026):
        for sample_id, target, logit in zip(
            ("a", "b", "c", "d"), (0, 0, 1, 1), (-2.0, -0.5, 0.5, 2.0), strict=True
        ):
            rows.append(
                {
                    "sample_id": sample_id,
                    "target": target,
                    "logit": logit + offset,
                    "repeat_seed": seed,
                }
            )
    return pd.DataFrame(rows).sort_values(["repeat_seed", "sample_id"]).reset_index(drop=True)


def test_development_decision_thresholds_and_ecg_claims_use_tiny_repeat_evidence() -> None:
    primary = _repeat_oof(0.0)
    ecg = _repeat_oof(0.1)
    ecg.loc[ecg["repeat_seed"] == 42, "logit"] *= -1
    thresholds, claims = decision._derive_decision_claims(primary, ecg)
    ensemble = decision.development_mean_logit_ensemble(primary)

    assert thresholds == {
        "youden_j": youden_threshold(ensemble["target"], ensemble["probability"]),
        "target_sensitivity": sensitivity_threshold(
            ensemble["target"], ensemble["probability"], target=0.9
        ),
    }
    assert set(claims["ecg_repeat_metrics"]) == {"17", "42", "2026"}
    assert set(claims["ecg_vs_gated_repeat_effects"]) == {"17", "42", "2026"}
    for seed in (17, 42, 2026):
        primary_repeat = primary.loc[primary["repeat_seed"] == seed]
        ecg_repeat = ecg.loc[ecg["repeat_seed"] == seed]
        primary_metrics = metrics(primary_repeat["target"], sigmoid(primary_repeat["logit"]))
        ecg_metrics = metrics(ecg_repeat["target"], sigmoid(ecg_repeat["logit"]))
        assert claims["ecg_repeat_metrics"][str(seed)] == ecg_metrics
        assert claims["ecg_vs_gated_repeat_effects"][str(seed)] == {
            name: ecg_metrics[name] - primary_metrics[name] for name in primary_metrics
        }
    assert claims["ecg_vs_gated_repeat_effects"]["42"]["roc_auc"] < 0
    changed_thresholds, changed_claims = decision._derive_decision_claims(primary, _repeat_oof(0.1))
    assert changed_thresholds == thresholds
    assert changed_claims != claims


def test_development_decision_requires_strict_v1_schema(tmp_path: Path) -> None:
    base = {
        "development_decision_schema_version": 1,
        "development_decision_id": "development-decision-" + "a" * 64,
        "dataset_id": "symile",
        "bundle_id": "bundle-" + "b" * 64,
        "cv_assignment_id": "cv-assignment-" + "c" * 64,
        "task": {
            "task_id": "pneumonia_strict",
            "label_policy_version": "symile-pneumonia-strict-v1",
        },
        "family_development_ids": _family_references(),
        "decision_policy": decision.DEVELOPMENT_DECISION_POLICY,
        "selected_family_authorities": {},
        "primary_thresholds": {},
    }
    mutations = {
        "missing": lambda document: document.pop("development_decision_schema_version"),
        "boolean": lambda document: document.__setitem__(
            "development_decision_schema_version", True
        ),
        "float": lambda document: document.__setitem__("development_decision_schema_version", 1.0),
        "wrong": lambda document: document.__setitem__("development_decision_schema_version", 2),
        "extra": lambda document: document.__setitem__("unexpected", True),
    }
    for name, mutation in mutations.items():
        document = deepcopy(base)
        mutation(document)
        directory = tmp_path / name
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
        (directory / "claims.json").write_text("{}\n", encoding="utf-8")
        with pytest.raises(ManifestBuildError, match="contract"):
            decision.validate_development_decision(directory)


def test_development_decision_revalidates_persisted_claims_and_family_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    family_ids = {item["family_id"]: item["development_id"] for item in _family_references()}
    developments = {
        identity: ValidatedDevelopmentResult(
            tmp_path / identity,
            {
                "development_id": identity,
                "family_id": family,
                "final_training_budget": None if family == "labs_logistic" else 3,
                "scientific_context": {
                    "bundle_id": "bundle-" + "b" * 64,
                    "cv_assignment_id": "cv-assignment-" + "c" * 64,
                    "task_id": decision.TASK_ID,
                    "fit_config": {"family": {"family_id": family}},
                },
            },
        )
        for family, identity in family_ids.items()
    }

    def validate_family(directory: Path, **kwargs: object) -> ValidatedDevelopmentResult:
        identity = kwargs["expected_development_id"]
        if identity not in developments or Path(directory).name != identity:
            raise ManifestBuildError("Selected family evidence is unavailable")
        return developments[identity]

    monkeypatch.setattr(decision, "validate_development_result", validate_family)
    monkeypatch.setattr(decision, "final_input_projection_from_development", lambda config: config)
    monkeypatch.setattr(
        decision,
        "_development_pretrained_identity",
        lambda development, *, family, model_root: (
            {"declared_name": "fixture", "stable_identifier": "fixture", "sha256": "a" * 64}
            if family == "cxr_densenet"
            else None
        ),
    )
    primary, ecg = _repeat_oof(0.0), _repeat_oof(0.1)
    monkeypatch.setattr(
        decision,
        "validated_development_repeat_oof",
        lambda development, root: (
            ecg if development.manifest["family_id"] == "cxr_labs_ecg_gated" else primary
        ),
    )
    document, claims = decision._decision_document(
        family_ids=family_ids,
        report_root=tmp_path,
        model_root=tmp_path / "models",
        prediction_root=tmp_path / "private",
        manifest_root=tmp_path / "manifests",
    )
    directory = tmp_path / document["development_decision_id"]
    directory.mkdir()
    manifest_path, claims_path = directory / "manifest.json", directory / "claims.json"
    manifest_path.write_bytes(canonical_json_bytes(document))
    claims_path.write_bytes(canonical_json_bytes(claims))
    assert decision.validate_development_decision(directory).decision_id == directory.name

    claims_path.write_bytes(canonical_json_bytes({**claims, "fabricated": True}))
    with pytest.raises(ManifestBuildError, match="claims differ from family evidence"):
        decision.validate_development_decision(directory)
    claims_path.write_bytes(canonical_json_bytes(claims))

    changed = deepcopy(document)
    changed["primary_thresholds"]["youden_j"] += 0.01
    manifest_path.write_bytes(canonical_json_bytes(changed))
    with pytest.raises(ManifestBuildError, match="differs from its authorities"):
        decision.validate_development_decision(directory)

    changed = deepcopy(document)
    changed["family_development_ids"][-1]["development_id"] = "development-" + "f" * 64
    manifest_path.write_bytes(canonical_json_bytes(changed))
    with pytest.raises(ManifestBuildError, match="Selected family evidence is unavailable"):
        decision.validate_development_decision(directory)

    manifest_path.write_bytes(canonical_json_bytes(document))
    ecg.loc[ecg["repeat_seed"] == 42, "logit"] *= -1
    with pytest.raises(ManifestBuildError, match="claims differ from family evidence"):
        decision.validate_development_decision(directory)
