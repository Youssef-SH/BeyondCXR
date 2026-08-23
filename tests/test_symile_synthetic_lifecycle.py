from __future__ import annotations

from pathlib import Path

import pytest
from symile_campaign_test_support import (
    _freeze,
    _repeat_oof,
    _synthetic_final_family_authorities,
    _synthetic_final_packages,
)

import beyondcxr.training.symile_campaign_control as campaign_control
import beyondcxr.training.symile_ecg_extension_result as extension_result
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.symile_final_packages import (
    ValidatedFinalPackage,
)
from beyondcxr.utils.symile_publication import ValidatedDevelopmentResult


def test_complete_synthetic_pretest_authority_lifecycle_stops_before_test_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = _freeze(tmp_path / "template")
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
    ecg_id = "development-" + "e" * 64
    core = {"analysis_id": core_id, "family_development_ids": family_ids}
    all_development_ids = {**family_ids, "cxr_labs_ecg_gated": ecg_id}
    final_family_authorities = _synthetic_final_family_authorities(all_development_ids, template)
    ecg = ValidatedDevelopmentResult(
        tmp_path / "ecg-development",
        {
            "development_id": ecg_id,
            "family_id": "cxr_labs_ecg_gated",
            "final_training_budget": 3,
            "scientific_context": {"fit_config": {"family": {"family_id": "cxr_labs_ecg_gated"}}},
        },
        "f" * 64,
    )

    def validate_analysis(directory: Path, **kwargs: object) -> dict[str, object]:
        expected = kwargs.get("expected_analysis_id")
        if expected is not None and (expected != core_id or Path(directory).name != core_id):
            raise ManifestBuildError("invalid core development analysis")
        return core

    def validate_development(directory: Path, **kwargs: object) -> ValidatedDevelopmentResult:
        del kwargs
        identity = Path(directory).name
        if Path(directory) == ecg.directory or identity == ecg_id:
            return ecg
        if identity in family_ids.values():
            family = next(name for name, value in family_ids.items() if value == identity)
            return ValidatedDevelopmentResult(
                Path(directory),
                {
                    "development_id": identity,
                    "family_id": family,
                    "final_training_budget": None if family == "labs_logistic" else 3,
                    "scientific_context": {"fit_config": {"family": {"family_id": family}}},
                },
                "a" * 64,
            )
        raise ManifestBuildError("invalid development authority")

    monkeypatch.setattr(extension_result, "validate_analysis_result", validate_analysis)
    monkeypatch.setattr(extension_result, "validate_development_result", validate_development)
    monkeypatch.setattr(
        extension_result,
        "final_input_projection_from_development",
        lambda fit_config: final_family_authorities[fit_config["family"]["family_id"]][
            "final_input"
        ],
    )
    monkeypatch.setattr(
        extension_result,
        "_development_pretrained_identity",
        lambda development, family, model_root: final_family_authorities[family][
            "pretrained_scientific_identity"
        ],
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
    report_root = tmp_path / "reports"
    subgroup_id = extension_result.publish_focused_subgroup_derivative(
        report_root=report_root, summary={"policy": {}, "strata": {}}
    )
    monkeypatch.setattr(
        extension_result,
        "_expected_focused_subgroup_derivative",
        lambda **kwargs: subgroup_id,
    )
    development = extension_result.publish_ecg_extension_result(
        report_root=report_root,
        core_analysis_directory=tmp_path / "core-analysis",
        ecg_development=ecg,
        focused_subgroup_derivative=subgroup_id,
    )
    packages = _synthetic_final_packages(
        tmp_path / "packages",
        template,
        family_authorities=dict(development.manifest["final_family_authorities"]),
    )
    package_by_path = {package.directory: package for package in packages}
    monkeypatch.setattr(
        campaign_control,
        "validate_final_package",
        lambda directory, **kwargs: package_by_path[Path(directory)],
    )
    freeze = campaign_control.publish_pretest_freeze(
        control_root=tmp_path / "control",
        bundle=template.manifest["bundle"],
        task=template.manifest["task"],
        ecg_extension_result=development,
        final_packages=packages,
        neural_inference_runtime=template.manifest["held_out_policy"]["neural_inference_runtime"],
        science_git_commit=template.manifest["science_git_commit"],
        dependency_lock_sha256=template.manifest["dependency_lock_sha256"],
    )
    restored = campaign_control.validate_pretest_freeze(
        freeze.directory,
        ecg_extension_result=development,
        final_packages=packages,
    )
    assert restored.freeze_id == freeze.freeze_id
    assert len(restored.manifest["final_packages"]) == 14
    assert not (tmp_path / "control" / "test-open.json").exists()

    def assert_freeze_rejects(index: int, manifest: dict[str, object], message: str) -> None:
        altered = ValidatedFinalPackage(
            packages[index].directory,
            manifest,
            "9" * 64,
        )
        candidate = (*packages[:index], altered, *packages[index + 1 :])
        package_by_path[altered.directory] = altered
        with pytest.raises(ManifestBuildError, match=message):
            campaign_control.publish_pretest_freeze(
                control_root=tmp_path / f"rejected-{index}",
                bundle=template.manifest["bundle"],
                task=template.manifest["task"],
                ecg_extension_result=development,
                final_packages=candidate,
                neural_inference_runtime=template.manifest["held_out_policy"][
                    "neural_inference_runtime"
                ],
                science_git_commit=template.manifest["science_git_commit"],
                dependency_lock_sha256=template.manifest["dependency_lock_sha256"],
            )
        with pytest.raises(ManifestBuildError, match=message):
            campaign_control.validate_pretest_freeze(
                freeze.directory,
                ecg_extension_result=development,
                final_packages=candidate,
            )
        package_by_path[altered.directory] = packages[index]

    wrong_budget = {
        **packages[1].manifest,
        "final_package_id": "final-package-" + "7" * 64,
        "final_training_budget": 4,
    }
    assert_freeze_rejects(1, wrong_budget, "development-derived family authority")

    wrong_input = {
        **packages[5].manifest,
        "final_package_id": "final-package-" + "8" * 64,
        "input": {
            **packages[5].manifest["input"],
            "preprocessing": {
                **packages[5].manifest["input"]["preprocessing"],
                "lab_policy": "altered-fit-input",
            },
        },
    }
    assert_freeze_rejects(5, wrong_input, "development-derived family authority")

    wrong_ancestry = {
        **packages[5].manifest,
        "final_package_id": "final-package-" + "6" * 64,
        "source_cxr_package_id": packages[3].package_id,
    }
    assert_freeze_rejects(5, wrong_ancestry, "same-seed CXR ancestry")
