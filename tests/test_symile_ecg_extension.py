from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from symile_campaign_test_support import (
    _repeat_oof,
    _synthetic_final_family_authorities,
)

import beyondcxr.training.symile_ecg_extension_result as extension_result
import beyondcxr.training.symile_statistics as symile_statistics
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.utils.package_identity import canonical_scientific_id
from beyondcxr.utils.symile_publication import ValidatedDevelopmentResult


def test_focused_subgroup_derivative_requires_exact_v1_contract(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    derivative_id = extension_result.publish_focused_subgroup_derivative(
        report_root=report_root, summary={"policy": {}, "strata": {}}
    )
    path = report_root / "development-subgroups" / f"{derivative_id}.json"
    document = json.loads(path.read_bytes())
    assert document == {
        "focused_subgroup_derivative_schema_version": 1,
        "policy": {},
        "strata": {},
    }
    assert extension_result.validate_focused_subgroup_derivative(path) == document

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
            altered["legacy"] = True
        encoded = (json.dumps(altered, sort_keys=True, separators=(",", ":")) + "\n").encode()
        altered_id = "focused-subgroup-" + hashlib.sha256(encoded).hexdigest()
        altered_path = path.parent / f"{altered_id}.json"
        altered_path.write_bytes(encoded)
        with pytest.raises(ManifestBuildError, match="contract"):
            extension_result.validate_focused_subgroup_derivative(altered_path)


def test_ecg_extension_claims_and_thresholds_are_rederived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core_analysis = {
        "analysis_id": "analysis-" + "a" * 64,
        "family_development_ids": {
            family: "development-" + f"{index:064x}"
            for index, family in enumerate(
                (
                    "labs_logistic",
                    "labs_lightgbm",
                    "cxr_densenet",
                    "cxr_labs_concat",
                    "cxr_labs_gated",
                    "cxr_labs_gated_no_observedness",
                )
            )
        },
    }
    ecg = ValidatedDevelopmentResult(
        tmp_path / "ecg",
        {
            "family_id": "cxr_labs_ecg_gated",
            "development_id": "development-" + "e" * 64,
            "repeat_metrics": {"17": {"roc_auc": 0.5}},
        },
        "f" * 64,
    )
    monkeypatch.setattr(
        extension_result, "validate_analysis_result", lambda *args, **kwargs: core_analysis
    )
    monkeypatch.setattr(
        extension_result, "validate_development_result", lambda *args, **kwargs: ecg
    )
    primary = _repeat_oof([-2.0, 1.0, -1.0, 2.0])
    ecg_oof = _repeat_oof([-3.0, 0.0, 0.5, 3.0])
    development_ids = dict(core_analysis["family_development_ids"])
    development_ids["cxr_labs_ecg_gated"] = ecg.manifest["development_id"]
    final_family_authorities = _synthetic_final_family_authorities(development_ids)
    monkeypatch.setattr(
        extension_result,
        "_resolve_development_authorities",
        lambda **kwargs: (
            primary.copy(),
            ecg_oof.copy(),
            final_family_authorities,
        ),
    )
    subgroup_id = extension_result.publish_focused_subgroup_derivative(
        report_root=tmp_path / "reports",
        summary={"policy": {}, "strata": {}},
    )
    monkeypatch.setattr(
        extension_result,
        "_expected_focused_subgroup_derivative",
        lambda **kwargs: subgroup_id,
    )
    result = extension_result.publish_ecg_extension_result(
        report_root=tmp_path / "reports",
        core_analysis_directory=tmp_path / "analysis",
        ecg_development=ecg,
        focused_subgroup_derivative=subgroup_id,
    )
    for value in (True, 1.0):
        altered = tmp_path / f"altered-extension-{value!r}"
        shutil.copytree(result.directory, altered)
        manifest_path = altered / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["ecg_extension_result_schema_version"] = value
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ManifestBuildError, match="contract"):
            extension_result.validate_ecg_extension_result(
                altered,
                report_root=tmp_path / "reports",
                enforce_directory_name=False,
            )
    assert "ecg_ensemble_metrics" not in result.manifest
    claims = json.loads((result.directory / "claims.json").read_bytes())
    assert set(claims["ecg_vs_gated_repeat_effects"]) == {"17", "42", "2026"}
    assert all(
        set(effect) == {"roc_auc", "average_precision", "brier_score"}
        for effect in claims["ecg_vs_gated_repeat_effects"].values()
    )
    for seed in (17, 42, 2026):
        gated = primary.loc[primary["repeat_seed"] == seed]
        candidate = ecg_oof.loc[ecg_oof["repeat_seed"] == seed]
        gated_metrics = symile_statistics.metrics(
            gated["target"], symile_statistics.sigmoid(gated["logit"])
        )
        candidate_metrics = symile_statistics.metrics(
            candidate["target"], symile_statistics.sigmoid(candidate["logit"])
        )
        assert claims["ecg_vs_gated_repeat_effects"][str(seed)] == {
            metric: candidate_metrics[metric] - gated_metrics[metric]
            for metric in symile_statistics.headline_metric_names()
        }

    fabricated_id = extension_result.publish_focused_subgroup_derivative(
        report_root=tmp_path / "reports",
        summary={"policy": {"fabricated": True}, "strata": {}},
    )
    fabricated = tmp_path / "fabricated-extension"
    shutil.copytree(result.directory, fabricated)
    fabricated_manifest = json.loads((fabricated / "manifest.json").read_bytes())
    fabricated_manifest["focused_subgroup_derivative"] = fabricated_id
    semantic = {
        key: value
        for key, value in fabricated_manifest.items()
        if key not in {"ecg_extension_result_schema_version", "ecg_extension_result_id"}
    }
    fabricated_manifest["ecg_extension_result_id"] = canonical_scientific_id(
        extension_result.ECG_EXTENSION_RESULT_PREFIX, semantic
    )
    (fabricated / "manifest.json").write_text(
        json.dumps(fabricated_manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestBuildError, match="does not rederive"):
        extension_result.validate_ecg_extension_result(
            fabricated,
            report_root=tmp_path / "reports",
            enforce_directory_name=False,
        )

    other_ecg_oof = ecg_oof.assign(logit=ecg_oof["logit"] + 0.75)
    monkeypatch.setattr(
        extension_result,
        "_resolve_development_authorities",
        lambda **kwargs: (
            primary.copy(),
            other_ecg_oof.copy(),
            final_family_authorities,
        ),
    )
    with pytest.raises(ManifestBuildError, match="do not rederive"):
        extension_result.validate_ecg_extension_result(
            result.directory,
            report_root=tmp_path / "reports",
        )


def test_ecg_extension_authority_resolution_rejects_recorded_id_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core_id = "analysis-" + "a" * 64
    family_ids = {
        family: "development-" + f"{index + 1:064x}"
        for index, family in enumerate(
            (
                "labs_logistic",
                "labs_lightgbm",
                "cxr_densenet",
                "cxr_labs_concat",
                "cxr_labs_gated",
                "cxr_labs_gated_no_observedness",
            )
        )
    }
    ecg_id = "development-" + "c" * 64
    core = {
        "analysis_id": core_id,
        "family_development_ids": family_ids,
    }

    def validate_analysis(directory: Path, **kwargs: object) -> dict[str, object]:
        if Path(directory).name != core_id or kwargs["expected_analysis_id"] != core_id:
            raise ManifestBuildError("core analysis identity tampering")
        return core

    def validate_development(directory: Path, **kwargs: object) -> ValidatedDevelopmentResult:
        del kwargs
        identity = Path(directory).name
        if identity == ecg_id:
            family = "cxr_labs_ecg_gated"
        elif identity in family_ids.values():
            family = next(name for name, value in family_ids.items() if value == identity)
        else:
            raise ManifestBuildError("ECG development identity tampering")
        return ValidatedDevelopmentResult(
            Path(directory),
            {
                "development_id": identity,
                "family_id": family,
                "final_training_budget": None if family == "labs_logistic" else 3,
                "scientific_context": {"fit_config": {"family": {"family_id": family}}},
            },
            "d" * 64,
        )

    monkeypatch.setattr(extension_result, "validate_analysis_result", validate_analysis)
    monkeypatch.setattr(extension_result, "validate_development_result", validate_development)
    monkeypatch.setattr(
        extension_result,
        "final_input_projection_from_development",
        lambda fit_config: {"family": {"family_id": fit_config["family"]["family_id"]}},
    )
    monkeypatch.setattr(
        extension_result,
        "_development_pretrained_identity",
        lambda development, family, model_root: (
            {
                "declared_name": "densenet121-res224-chex",
                "stable_identifier": "synthetic-weight",
                "sha256": "a" * 64,
            }
            if family == "cxr_densenet"
            else None
        ),
    )
    monkeypatch.setattr(
        extension_result,
        "validated_development_repeat_oof",
        lambda development, prediction_root: _repeat_oof(
            [-2.0, 1.0, -1.0, 2.0]
            if development.manifest["family_id"] == "cxr_labs_gated"
            else [-3.0, 0.0, 0.5, 3.0]
        ),
    )
    arguments = {
        "report_root": tmp_path / "reports",
        "model_root": tmp_path / "models",
        "prediction_root": tmp_path / "private",
        "manifest_root": tmp_path / "manifests",
    }
    extension_result._resolve_development_authorities(
        core_analysis_id=core_id, ecg_development_id=ecg_id, **arguments
    )
    with pytest.raises(ManifestBuildError, match="core analysis identity tampering"):
        extension_result._resolve_development_authorities(
            core_analysis_id="analysis-" + "9" * 64,
            ecg_development_id=ecg_id,
            **arguments,
        )
    with pytest.raises(ManifestBuildError, match="ECG development identity tampering"):
        extension_result._resolve_development_authorities(
            core_analysis_id=core_id,
            ecg_development_id="development-" + "9" * 64,
            **arguments,
        )
