from __future__ import annotations

import copy
import csv
import io
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from uuid import UUID

import pytest

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_schemas import REPEAT_SEEDS
from beyondcxr.release.binding import validate_result_binding
from beyondcxr.release.checks import (
    _PRIVATE_PATH,
    _check_package_metadata,
    _check_release_delta,
    _container_serving_command,
    _docker_verification_command,
    check_repository,
    inspect_distribution_archives,
    run_final_acceptance,
)
from beyondcxr.release.container_acceptance import _stop_process
from beyondcxr.release.inputs import _subgroup_projection
from beyondcxr.release.projection import RELIABILITY_FAMILIES, project_public_results
from beyondcxr.release.render import (
    DISPLAY,
    RESULT_END,
    RESULT_START,
    parse_result_region,
    public_result_surface,
    render_public_results,
    replace_result_region,
)
from beyondcxr.release.reproduction import (
    _install_with_rollback,
    _model_card_fragment,
    _readme_fragment,
    _Regenerated,
    reproduce_results,
)
from beyondcxr.release.serving import clean_release_provenance, validate_serving_api_responses
from beyondcxr.serving.authority import RESEARCH_WARNING
from beyondcxr.training.symile_families import (
    FINAL_PACKAGE_POLICY,
    SYMILE_CORE_DEVELOPMENT_FAMILIES,
    SYMILE_ECG_GATED_FAMILY,
)


def _identity(prefix: str, index: int) -> str:
    return prefix + f"{index:064x}"


def fixture_text(*parts: str) -> str:
    return "".join(parts)


def _symile_source_sample_id() -> str:
    return f"symile:{23_456_789}"


def _mimic_cxr_source_dicom_id() -> str:
    groups = (0x02AA804E, 0xBDE0AFDD, 0x112C0B34, 0x7BC16630, 0x4E384014)
    return "-".join(f"{group:08x}" for group in groups)


def _rsna_source_identifier() -> str:
    return str(UUID(int=0x123E4567E89B42D3A456426614174000))


def _binding() -> dict[str, object]:
    return {
        "result_binding_schema_version": 1,
        "science_execution": {"git_commit": "a" * 40},
        "data": {
            "bundle_id": _identity("bundle-", 1),
            "split_assignment_id": _identity("split-assignment-", 2),
            "cv_assignment_id": _identity("cv-assignment-", 3),
        },
        "development": {
            "core_analysis_id": _identity("analysis-", 4),
            "family_development_ids": {
                family: _identity("development-", index + 10)
                for index, family in enumerate(SYMILE_CORE_DEVELOPMENT_FAMILIES)
            },
            "ecg_development_id": _identity("development-", 30),
            "ecg_extension_result_id": _identity("ecg-extension-result-", 31),
        },
        "held_out": {
            "pretest_freeze_id": _identity("pretest-freeze-", 40),
            "final_package_ids": [_identity("final-package-", 50 + i) for i in range(14)],
            "prediction_ids": [_identity("prediction-", 70 + i) for i in range(14)],
            "global_result_id": _identity("global-result-", 90),
        },
    }


def _metrics(value: float) -> dict[str, float]:
    return {"roc_auc": value, "average_precision": value, "brier_score": 1.0 - value}


def _projection(*, point_outside: bool = False):
    repeats = {
        family: {str(seed): _metrics(0.5 + index / 100) for seed in REPEAT_SEEDS}
        for index, family in enumerate(SYMILE_CORE_DEVELOPMENT_FAMILIES)
    }
    views = {
        family: {**_metrics(0.6), "calibration_slope": 1.0, "calibration_intercept": 0.0}
        for family in FINAL_PACKAGE_POLICY
    }
    curves = {
        family: {"mean_predicted_probability": [0.2, 0.8], "observed_positive_fraction": [0.1, 0.9]}
        for family in FINAL_PACKAGE_POLICY
    }
    discrimination = {
        family: {
            "roc": {"false_positive_rate": [0.0, 1.0], "true_positive_rate": [0.0, 1.0]},
            "precision_recall": {"recall": [1.0, 0.0], "precision": [0.5, 1.0]},
        }
        for family in FINAL_PACKAGE_POLICY
    }
    effect = {
        metric: {"point": 0.1, "lower": 0.01, "upper": 0.2, "accepted": 2000, "attempts": 2000}
        for metric in ("roc_auc", "average_precision", "brier_score")
    }
    effects = {
        name: copy.deepcopy(effect)
        for name in ("concat_vs_cxr", "primary", "gated_vs_concat", "ecg_vs_gated")
    }
    if point_outside:
        effects["primary"]["roc_auc"].update(point=0.3, lower=0.01, upper=0.2)
    return project_public_results(
        cohort={
            "rows": [{"scope": "development", "eligible": 20, "positive": 10, "negative": 10}],
            "source_reconciliation": {
                "full_admissions": 22,
                "official_admissions": 20,
                "excluded_admissions": 2,
            },
            "construction": [
                {"item": "Train inclusion", "status": "All unique admissions in train.csv"},
                {
                    "item": "Classification exclusions",
                    "status": "Retrieval candidates excluded",
                },
            ],
            "endpoint": {
                "task_id": "pneumonia_strict",
                "label_policy_version": "symile-pneumonia-strict-v1",
                "label_source": "symile_mimic_data.csv:Pneumonia",
                "positive": "pneumonia_state == 1",
                "negative": "pneumonia_state == 0",
                "excluded": ["pneumonia_state == -1", "pneumonia_state is null"],
                "selection_provenance": (
                    "Broader and alternative label policies were audited prospectively before "
                    "supervised modeling; their larger eligible cohorts resulted from changed "
                    "target semantics and/or negative-class spectrum. The study therefore froze "
                    "explicit Pneumonia = 1 versus explicit Pneumonia = 0 and excludes uncertain "
                    "or unmentioned labels."
                ),
            },
            "descriptions": [
                {"scope": "development", "attribute": "age", "category": "18-49", "count": 8},
                {"scope": "development", "attribute": "sex", "category": "F", "count": 11},
                {
                    "scope": "development",
                    "attribute": "view_position",
                    "category": "AP",
                    "count": 12,
                },
            ],
        },
        core_analysis={
            "repeat_metrics": repeats,
            "paired_effects": {
                name: {str(seed): _metrics(0.01) for seed in REPEAT_SEEDS}
                for name in ("concat_minus_cxr", "gated_minus_cxr", "gated_minus_concat")
            },
            "ensemble_metrics": {
                family: _metrics(0.6 + index / 100)
                for index, family in enumerate(
                    (
                        "cxr_densenet",
                        "cxr_labs_concat",
                        "cxr_labs_gated",
                        "cxr_labs_gated_no_observedness",
                    )
                )
            },
            "observedness_ablation": {
                "repeat_effects": {str(seed): _metrics(0.01) for seed in REPEAT_SEEDS},
                "ensemble_effect": _metrics(0.01),
            },
        },
        extension_claims={
            "ecg_repeat_metrics": {str(seed): _metrics(0.6) for seed in REPEAT_SEEDS},
            "ecg_vs_gated_repeat_effects": {str(seed): _metrics(0.01) for seed in REPEAT_SEEDS},
            "ecg_ensemble_metrics": _metrics(0.6),
            "ecg_vs_gated_ensemble_effect": _metrics(0.01),
        },
        global_claims={
            "predictor_views": views,
            "discrimination_curves": discrimination,
            "reliability_curves": curves,
            "paired_effects": effects,
            "primary_operating_points": {
                "youden_j": {
                    "precision": 0.6,
                    "sensitivity": 0.7,
                    "specificity": 0.8,
                    "f1": 0.65,
                    "tn": 8,
                    "fp": 2,
                    "fn": 3,
                    "tp": 7,
                },
                "target_sensitivity": {
                    "precision": 0.5,
                    "sensitivity": 0.9,
                    "specificity": 0.4,
                    "f1": 0.64,
                    "tn": 4,
                    "fp": 6,
                    "fn": 1,
                    "tp": 9,
                },
            },
        },
        development_reliability={
            "curves": {family: curves[family] for family in RELIABILITY_FAMILIES},
            "calibration": {family: views[family] for family in RELIABILITY_FAMILIES},
        },
        thresholds={"youden_j": 0.55, "target_sensitivity": 0.25},
        subgroups={
            "policy": {
                "minimum_samples": 100,
                "minimum_positives": 20,
                "minimum_negatives": 20,
            },
            "rows": [
                {
                    "stratum": "age:18-49",
                    "supported": True,
                    "n": 120,
                    "positives": 60,
                    "negatives": 60,
                    "estimate": estimate,
                    "repeat": repeat,
                    "cxr_roc_auc": 0.60,
                    "gated_roc_auc": 0.62,
                    "effect_roc_auc": 0.02,
                    "cxr_average_precision": 0.61,
                    "gated_average_precision": 0.62,
                    "effect_average_precision": 0.01,
                    "cxr_brier_score": 0.22,
                    "gated_brier_score": 0.21,
                    "effect_brier_score": -0.01,
                }
                for estimate, repeat in (
                    ("OOF repeat", 17),
                    ("OOF repeat", 42),
                    ("OOF repeat", 2026),
                    ("Mean-logit OOF ensemble", None),
                )
            ]
            + [
                {
                    "stratum": "age:80-plus",
                    "supported": False,
                    "n": 30,
                    "positives": None,
                    "negatives": None,
                    "estimate": "Unavailable",
                    "repeat": None,
                }
            ],
        },
        provenance={
            "global_result_id": _identity("global-result-", 90),
            "primary_package_ids": [_identity("final-package-", index) for index in range(3)],
            "science_git_commit": "a" * 40,
            "bundle_id": _identity("bundle-", 1),
            "split_assignment_id": _identity("split-assignment-", 2),
            "cv_assignment_id": _identity("cv-assignment-", 3),
            "pretest_freeze_id": _identity("pretest-freeze-", 4),
            "science_dependency_lock_sha256": "b" * 64,
            "bundle_manifest_sha256": "c" * 64,
            "export_manifest_sha256": "d" * 64,
            "pretrained_scientific_identity": {
                "declared_name": "densenet121-res224-chex",
                "stable_identifier": "https://example.invalid/weights",
                "sha256": "e" * 64,
            },
            "pretrained_weight_materializations": [
                {
                    "seed": seed,
                    "declared_name": "densenet121-res224-chex",
                    "stable_identifier": "https://example.invalid/weights",
                    "cache_filename": "weights.pt",
                    "byte_size": 123,
                    "sha256": "e" * 64,
                }
                for seed in REPEAT_SEEDS
            ],
        },
    )


def test_result_binding_is_strict_and_path_neutral() -> None:
    binding = validate_result_binding(_binding())
    assert binding.canonical_bytes()
    with pytest.raises(AttributeError):
        binding.data.bundle_id = "changed"
    with pytest.raises(TypeError):
        binding.development.family_development_ids[0][1] = "changed"
    invalid = copy.deepcopy(_binding())
    invalid["artifact_root"] = "/private/path"
    with pytest.raises(ManifestBuildError):
        validate_result_binding(invalid)
    duplicate = copy.deepcopy(_binding())
    duplicate["held_out"]["prediction_ids"][1] = duplicate["held_out"]["prediction_ids"][0]
    with pytest.raises(ManifestBuildError, match="duplicated"):
        validate_result_binding(duplicate)
    for section, field in (("science_execution", "git_commit"), ("data", "bundle_id")):
        malformed = copy.deepcopy(_binding())
        malformed[section][field] = 123
        with pytest.raises(ManifestBuildError):
            validate_result_binding(malformed)
    invalid = copy.deepcopy(_binding())
    invalid["data"]["unexpected"] = "value"
    with pytest.raises(ManifestBuildError):
        validate_result_binding(invalid)
    invalid = copy.deepcopy(_binding())
    invalid["result_binding_schema_version"] = 1.0
    with pytest.raises(ManifestBuildError, match="contract is invalid"):
        validate_result_binding(invalid)
    for section, field, value in (
        ("data", "bundle_id", _identity("bundle-extra-", 1)),
        (
            "held_out",
            "final_package_ids",
            [_identity("final-package-extra-", i) for i in range(14)],
        ),
    ):
        invalid = copy.deepcopy(_binding())
        invalid[section][field] = value
        with pytest.raises(ManifestBuildError, match="identity is invalid"):
            validate_result_binding(invalid)


def test_public_projection_rejects_patient_level_fields() -> None:
    with pytest.raises(ManifestBuildError, match="private or row-level"):
        project_public_results(
            cohort={fixture_text("sample", "_id"): fixture_text("symile", ":1")},
            core_analysis={
                "repeat_metrics": {},
                "paired_effects": {},
                "ensemble_metrics": {},
                "observedness_ablation": {},
            },
            extension_claims={
                "ecg_repeat_metrics": {},
                "ecg_vs_gated_repeat_effects": {},
                "ecg_ensemble_metrics": {},
                "ecg_vs_gated_ensemble_effect": {},
            },
            global_claims={
                "predictor_views": {},
                "discrimination_curves": {},
                "reliability_curves": {"cxr_densenet": {}, "cxr_labs_gated": {}},
                "paired_effects": {},
                "primary_operating_points": {},
            },
            development_reliability={"curves": {}, "calibration": {}},
            thresholds={},
            subgroups={},
            provenance={},
        )


def test_public_projection_is_recursively_immutable() -> None:
    projection = _projection()
    with pytest.raises(TypeError):
        projection.held_out["predictor_views"]["cxr_densenet"]["roc_auc"] = 0.9
    with pytest.raises(TypeError):
        projection.held_out["discrimination_curves"]["cxr_densenet"]["roc"]["false_positive_rate"][
            0
        ] = 0.1
    invalid = copy.deepcopy(_binding())
    invalid["nested"] = {1: "not coerced"}
    with pytest.raises(ManifestBuildError, match="keys must be strings"):
        project_public_results(
            cohort=invalid,
            core_analysis={
                "repeat_metrics": {},
                "paired_effects": {},
                "ensemble_metrics": {},
                "observedness_ablation": {},
            },
            extension_claims={
                "ecg_repeat_metrics": {},
                "ecg_vs_gated_repeat_effects": {},
                "ecg_ensemble_metrics": {},
                "ecg_vs_gated_ensemble_effect": {},
            },
            global_claims={
                "predictor_views": {},
                "discrimination_curves": {},
                "reliability_curves": {"cxr_densenet": {}, "cxr_labs_gated": {}},
                "paired_effects": {},
                "primary_operating_points": {},
            },
            development_reliability={"curves": {}, "calibration": {}},
            thresholds={},
            subgroups={},
            provenance={},
        )


def test_subgroup_projection_preserves_repeat_and_ensemble_evidence() -> None:
    metric = {"roc_auc": 0.60, "average_precision": 0.61, "brier_score": 0.22}
    gated = {"roc_auc": 0.62, "average_precision": 0.63, "brier_score": 0.21}
    projected = _subgroup_projection(
        {
            "policy": {
                "minimum_samples": 100,
                "minimum_positives": 20,
                "minimum_negatives": 20,
            },
            "strata": {
                "age:18-49": {
                    "supported": True,
                    "n": 120,
                    "positives": 60,
                    "negatives": 60,
                    "repeat_metrics": {
                        str(seed): {"cxr": metric, "gated": gated} for seed in REPEAT_SEEDS
                    },
                    "mean_logit_ensemble": {"cxr": metric, "gated": gated},
                },
                "age:80-plus": {"supported": False, "n": 30},
            },
        }
    )
    supported = [row for row in projected["rows"] if row["stratum"] == "age:18-49"]
    assert [row["repeat"] for row in supported] == [17, 42, 2026, None]
    assert all(row["effect_roc_auc"] == pytest.approx(0.02) for row in supported)
    unsupported = projected["rows"][-1]
    assert unsupported == {
        "stratum": "age:80-plus",
        "supported": False,
        "n": 30,
        "positives": None,
        "negatives": None,
        "estimate": "Unavailable",
        "repeat": None,
    }


def test_all_public_tables_and_figures_render_deterministically(tmp_path: Path) -> None:
    import matplotlib

    projection = _projection()
    original_hashsalt = matplotlib.rcParams["svg.hashsalt"]
    original_family = matplotlib.rcParams["font.family"]
    first = render_public_results(projection, tmp_path / "first")
    second = render_public_results(projection, tmp_path / "second")
    assert matplotlib.rcParams["svg.hashsalt"] == original_hashsalt
    assert matplotlib.rcParams["font.family"] == original_family
    assert first == second
    assert set(first) == public_result_surface()
    table4 = (tmp_path / "first/tables/probability_operating_points.md").read_text(encoding="utf-8")
    assert DISPLAY["cxr_densenet"] in table4
    assert DISPLAY["cxr_labs_gated"] in table4
    assert DISPLAY["labs_logistic"] not in table4
    assert "Development OOF raw probability" in table4
    assert "Held-out descriptive raw probability" in table4
    assert "Held-out behavior at development-derived operating point" in table4
    assert (
        "Thresholds were derived from development OOF evidence and applied unchanged to held-out "
        "probabilities."
    ) in table4
    table3 = (tmp_path / "first/tables/heldout_performance.md").read_text(encoding="utf-8")
    assert "CXR + labs gated − CXR — PRIMARY" in table3
    assert "concat_vs_cxr" not in table3
    table2 = (tmp_path / "first/tables/development_performance.md").read_text(encoding="utf-8")
    assert "ΔAUROC" in table2 and "Mean-logit OOF paired effect" in table2
    assert all(str(seed) in table2 for seed in REPEAT_SEEDS)
    assert "CXR + labs + ECG gated − CXR + labs gated" in table2
    assert DISPLAY[SYMILE_ECG_GATED_FAMILY] in table2
    assert DISPLAY["cxr_labs_gated_no_observedness"] in table2
    with (tmp_path / "first/data/development_performance.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        development_rows = tuple(csv.DictReader(stream))
    development_models = {
        row["model"] for row in development_rows if row["estimate"] == "OOF repeat"
    }
    assert development_models == {DISPLAY[family] for family in SYMILE_CORE_DEVELOPMENT_FAMILIES}
    assert DISPLAY[SYMILE_ECG_GATED_FAMILY] not in development_models
    ecg_rows = [row for row in development_rows if row["model"] == DISPLAY[SYMILE_ECG_GATED_FAMILY]]
    assert [row["repeat"] for row in ecg_rows if row["estimate"] == "ECG extension OOF repeat"] == [
        str(seed) for seed in REPEAT_SEEDS
    ]
    assert sum(row["estimate"] == "ECG extension mean-logit OOF ensemble" for row in ecg_rows) == 1
    table5 = (tmp_path / "first/tables/observedness_subgroups.md").read_text(encoding="utf-8")
    assert "CXR AUROC" in table5 and "Gated AUROC" in table5
    assert all(str(seed) in table5 for seed in REPEAT_SEEDS)
    assert "None" not in table4 + table5
    cohort = (tmp_path / "first/tables/cohort.md").read_text(encoding="utf-8")
    assert "pneumonia_strict" in cohort
    assert "symile-pneumonia-strict-v1" in cohort
    assert "Aggregate description" in cohort
    assert "Cohort construction" in cohort
    assert "larger eligible cohorts resulted from changed target semantics" in cohort
    assert "negative-class spectrum" in cohort
    readme = _readme_fragment(projection)
    assert "https://github.com/Youssef-SH/BeyondCXR/tree/main/results/symile/" in readme
    model_card = _model_card_fragment(projection)
    assert "Pretrained scientific weight" in model_card
    assert "Seed 17 materialization" in model_card
    assert "Development subgroup characterization" in model_card
    assert "age:80-plus: unavailable under the prespecified support rule" in model_card
    assert set(projection.development) == {
        "repeat_metrics",
        "ensemble_metrics",
        "paired_effects",
        "observedness_ablation",
        "ecg_repeat_metrics",
        "ecg_ensemble_metrics",
        "ecg_vs_gated_repeat_effects",
        "ecg_vs_gated_ensemble_effect",
        "reliability_curves",
        "calibration",
    }
    assert set(projection.held_out["reliability_curves"]) == {
        "cxr_densenet",
        "cxr_labs_gated",
    }
    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(ManifestBuildError, match="absent or empty"):
        render_public_results(projection, broken)


def test_incremental_effect_interval_need_not_contain_point(tmp_path: Path) -> None:
    projection = _projection(point_outside=True)
    effect = projection.held_out["paired_effects"]["primary"]["roc_auc"]
    assert effect["point"] > effect["upper"]
    render_public_results(projection, tmp_path / "rendered")
    assert (tmp_path / "rendered/figures/incremental_effects.svg").is_file()


def test_docker_acceptance_command_checks_brand_notices_and_private_roots() -> None:
    command = _docker_verification_command()
    assert "import beyondcxr" in command
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        assert name in command
    for name in ("private", "models", "reports", "outbox", "data"):
        assert f"test ! -e /app/{name}" in command
    container = _container_serving_command(
        image="beyondcxr:test",
        authority=Path("/approved/authority"),
        package_root=Path("/restored/packages"),
    )
    assert container[:4] == ["docker", "run", "--rm", "--network"]
    assert sum(item.endswith(",readonly") for item in container) == 2
    assert "type=bind,src=/approved/authority,dst=/artifacts/authority,readonly" in container
    assert "type=bind,src=/restored/packages,dst=/artifacts/packages,readonly" in container
    assert "beyondcxr.release.container_acceptance" in container


def test_container_acceptance_kills_server_after_termination_timeout() -> None:
    calls: list[str] = []

    class Process:
        def terminate(self) -> None:
            calls.append("terminate")

        def wait(self, timeout: int | None = None) -> None:
            calls.append(f"wait:{timeout}")
            if timeout is not None:
                raise subprocess.TimeoutExpired("server", timeout)

        def kill(self) -> None:
            calls.append("kill")

    _stop_process(Process())  # type: ignore[arg-type]
    assert calls == ["terminate", "wait:15", "kill", "wait:None"]


def test_release_api_acceptance_validates_current_contract() -> None:
    authority_id = _identity("serving-authority-", 1)
    missing = ["lab_50868", "lab_50882"]
    model_info = {
        "task": "pneumonia_strict",
        "positive_class": {"value": 1, "meaning": "report-derived Pneumonia finding present"},
        "serving_authority_id": authority_id,
        "model_package_ids": [_identity("final-package-", index) for index in (1, 2, 3)],
        "seeds": [17, 42, 2026],
        "family": "cxr_labs_gated",
        "ensemble_policy": "ordered-seed-17-42-2026-mean-logit-then-sigmoid-v1",
        "input_contract": {"transport_policy": "test-input"},
        "preprocessing": {"spatial_policy": "test-spatial"},
        "operating_thresholds": {"youden_j": 0.5, "target_sensitivity": 0.2},
        "global_result_id": _identity("global-result-", 4),
        "science_git_commit": "a" * 40,
        "serving_release_git_commit": "b" * 40,
        "warning": RESEARCH_WARNING,
    }
    health = {"status": "ready", "serving_authority_id": authority_id}
    prediction = {
        "task": "pneumonia_strict",
        "probability": 0.61,
        "missing_labs": missing,
        "serving_authority_id": authority_id,
        "warning": RESEARCH_WARNING,
    }

    def validate(observed_model: dict[str, object]) -> None:
        validate_serving_api_responses(
            200,
            health,
            200,
            observed_model,
            200,
            prediction,
            authority_id,
            expected_model_info=model_info,
            expected_missing_labs=missing,
        )

    validate(model_info)
    incomplete = dict(model_info)
    incomplete.pop("family")
    wrong_order = copy.deepcopy(model_info)
    wrong_order["model_package_ids"].reverse()
    changed_threshold = copy.deepcopy(model_info)
    changed_threshold["operating_thresholds"]["youden_j"] = 0.51
    for invalid in (incomplete, wrong_order, changed_threshold):
        with pytest.raises(ManifestBuildError, match="model-info response"):
            validate(invalid)
    invalid_probability = dict(prediction, probability=True)
    with pytest.raises(ManifestBuildError, match="probability is invalid"):
        validate_serving_api_responses(
            200,
            health,
            200,
            model_info,
            200,
            invalid_probability,
            authority_id,
            expected_model_info=model_info,
            expected_missing_labs=missing,
        )


def test_bounded_result_region_requires_exactly_one_pair() -> None:
    source = f"before\n{RESULT_START}\npending\n{RESULT_END}\nafter\n"
    assert parse_result_region(source) == ("before\n", "\npending\n", "\nafter\n")
    assert "bound" in replace_result_region(source, "bound")
    invalid = (
        "no markers",
        f"{RESULT_START}missing end",
        f"missing start{RESULT_END}",
        f"{RESULT_START}{RESULT_START}{RESULT_END}",
        f"{RESULT_START}{RESULT_END}{RESULT_END}",
        f"{RESULT_END}{RESULT_START}",
    )
    for document in invalid:
        with pytest.raises(ManifestBuildError):
            parse_result_region(document)


def test_distribution_inspection_rejects_restricted_members(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    package_members = tuple(
        path.relative_to(repository_root / "src/beyondcxr").as_posix()
        for path in sorted(repository_root.glob("src/beyondcxr/**/*.py"))
    )
    representative_members = (
        "__init__.py",
        "data/errors.py",
        "models/cxr_baseline.py",
        "release/cli.py",
        "serving/api.py",
        "training/symile_campaign.py",
    )

    def add_wheel_package(archive: zipfile.ZipFile) -> None:
        for name in package_members:
            archive.writestr(f"beyondcxr/{name}", "")

    def add_wheel_release(archive: zipfile.ZipFile) -> None:
        add_wheel_package(archive)
        archive.writestr("beyondcxr-1.0.0.dist-info/METADATA", "Name: beyondcxr\nVersion: 1.0.0\n")
        archive.writestr("beyondcxr-1.0.0.dist-info/licenses/LICENSE", "license")
        archive.writestr("beyondcxr-1.0.0.dist-info/licenses/THIRD_PARTY_NOTICES.md", "notices")

    wheel = tmp_path / "beyondcxr.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        add_wheel_release(archive)
    payload = tmp_path / "module.py"
    payload.write_text("release", encoding="utf-8")
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nname = "beyondcxr"\nversion = "1.0.0"\n', encoding="utf-8")

    def add_sdist_release(archive: tarfile.TarFile) -> None:
        for name in package_members:
            archive.add(payload, arcname=f"beyondcxr-1.0.0/src/beyondcxr/{name}")
        archive.add(project, arcname="beyondcxr-1.0.0/pyproject.toml")
        for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CITATION.cff"):
            archive.add(payload, arcname=f"beyondcxr-1.0.0/{name}")
        archive.add(payload, arcname="beyondcxr-1.0.0/PKG-INFO")

    def add_tar_bytes(archive: tarfile.TarFile, name: str, content: bytes = b"data") -> None:
        member = tarfile.TarInfo(name)
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))

    sdist = tmp_path / "beyondcxr.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        add_sdist_release(archive)
    inspect_distribution_archives([wheel, sdist], repository_root=repository_root)

    unsafe = tmp_path / "unsafe.whl"
    with zipfile.ZipFile(unsafe, "w") as archive:
        add_wheel_package(archive)
        archive.writestr("beyondcxr-1.0.0.dist-info/METADATA", "Name: beyondcxr\nVersion: 1.0.0\n")
        archive.writestr("beyondcxr-1.0.0.dist-info/licenses/LICENSE", "license")
        archive.writestr("beyondcxr-1.0.0.dist-info/licenses/THIRD_PARTY_NOTICES.md", "notices")
        archive.writestr("private/predictions.parquet", "restricted")
    with pytest.raises(ManifestBuildError, match="unexpected package namespace"):
        inspect_distribution_archives([unsafe, sdist], repository_root=repository_root)

    mixed = tmp_path / "mixed.tar.gz"
    with tarfile.open(mixed, "w:gz") as archive:
        archive.add(payload, arcname="beyondcxr-0.1.0/src/beyondcxr/__init__.py")
        archive.add(project, arcname="beyondcxr-1.0.0/pyproject.toml")
    with pytest.raises(ManifestBuildError, match="top-level root"):
        inspect_distribution_archives([wheel, mixed], repository_root=repository_root)

    unexpected_namespace = tmp_path / "unexpected_namespace.whl"
    with zipfile.ZipFile(unexpected_namespace, "w") as archive:
        add_wheel_package(archive)
        archive.writestr("other_package/resource.json", "{}")
        archive.writestr("beyondcxr-1.0.0.dist-info/METADATA", "Name: beyondcxr\nVersion: 1.0.0\n")
        archive.writestr("beyondcxr-1.0.0.dist-info/licenses/LICENSE", "license")
        archive.writestr("beyondcxr-1.0.0.dist-info/licenses/THIRD_PARTY_NOTICES.md", "notices")
    with pytest.raises(ManifestBuildError, match="unexpected package namespace"):
        inspect_distribution_archives(
            [unexpected_namespace, sdist], repository_root=repository_root
        )

    unexpected_sdist_namespace = tmp_path / "unexpected_sdist_namespace.tar.gz"
    with tarfile.open(unexpected_sdist_namespace, "w:gz") as archive:
        for name in package_members:
            archive.add(payload, arcname=f"beyondcxr-1.0.0/src/beyondcxr/{name}")
        archive.add(payload, arcname="beyondcxr-1.0.0/src/other_package/resource.json")
        archive.add(project, arcname="beyondcxr-1.0.0/pyproject.toml")
        for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CITATION.cff"):
            archive.add(payload, arcname=f"beyondcxr-1.0.0/{name}")
    with pytest.raises(ManifestBuildError, match="unexpected package namespace"):
        inspect_distribution_archives(
            [wheel, unexpected_sdist_namespace], repository_root=repository_root
        )

    init_only = tmp_path / "init_only.whl"
    with zipfile.ZipFile(init_only, "w") as archive:
        archive.writestr("beyondcxr/__init__.py", "")
        archive.writestr("beyondcxr-1.0.0.dist-info/METADATA", "Name: beyondcxr\nVersion: 1.0.0\n")
        archive.writestr("beyondcxr-1.0.0.dist-info/licenses/LICENSE", "license")
        archive.writestr("beyondcxr-1.0.0.dist-info/licenses/THIRD_PARTY_NOTICES.md", "notices")
    with pytest.raises(ManifestBuildError, match="does not contain the beyondcxr package"):
        inspect_distribution_archives([init_only, sdist], repository_root=repository_root)

    sdist_resource_only = tmp_path / "sdist_resource_only.tar.gz"
    with tarfile.open(sdist_resource_only, "w:gz") as archive:
        archive.add(payload, arcname="beyondcxr-1.0.0/src/beyondcxr/__init__.py")
        archive.add(project, arcname="beyondcxr-1.0.0/pyproject.toml")
        for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CITATION.cff"):
            archive.add(payload, arcname=f"beyondcxr-1.0.0/{name}")
        archive.add(payload, arcname="beyondcxr-1.0.0/PKG-INFO")
    with pytest.raises(ManifestBuildError, match="does not contain src/beyondcxr"):
        inspect_distribution_archives([wheel, sdist_resource_only], repository_root=repository_root)

    partial_wheel = tmp_path / "partial.whl"
    with zipfile.ZipFile(partial_wheel, "w") as archive:
        for name in representative_members:
            archive.writestr(f"beyondcxr/{name}", "")
        archive.writestr("beyondcxr-1.0.0.dist-info/METADATA", "Name: beyondcxr\nVersion: 1.0.0\n")
        archive.writestr("beyondcxr-1.0.0.dist-info/licenses/LICENSE", "license")
        archive.writestr("beyondcxr-1.0.0.dist-info/licenses/THIRD_PARTY_NOTICES.md", "notices")
    with pytest.raises(ManifestBuildError, match="does not contain the beyondcxr package"):
        inspect_distribution_archives([partial_wheel, sdist], repository_root=repository_root)

    partial_sdist = tmp_path / "partial.tar.gz"
    with tarfile.open(partial_sdist, "w:gz") as archive:
        for name in representative_members:
            archive.add(payload, arcname=f"beyondcxr-1.0.0/src/beyondcxr/{name}")
        archive.add(project, arcname="beyondcxr-1.0.0/pyproject.toml")
        for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CITATION.cff"):
            archive.add(payload, arcname=f"beyondcxr-1.0.0/{name}")
    with pytest.raises(ManifestBuildError, match="does not contain src/beyondcxr"):
        inspect_distribution_archives([wheel, partial_sdist], repository_root=repository_root)

    for label, member_name, message in (
        ("wheel_traversal", "beyondcxr/../private/predictions.csv", "malformed member path"),
        ("wheel_backslash", "beyondcxr\\private\\predictions.csv", "malformed member path"),
    ):
        candidate = tmp_path / f"{label}.whl"
        with zipfile.ZipFile(candidate, "w") as archive:
            add_wheel_release(archive)
            archive.writestr(member_name, "restricted")
        with pytest.raises(ManifestBuildError, match=message):
            inspect_distribution_archives([candidate, sdist], repository_root=repository_root)

    duplicate_wheel = tmp_path / "wheel_duplicate.whl"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate_wheel, "w") as archive:
            add_wheel_release(archive)
            archive.writestr("beyondcxr/__init__.py", "duplicate")
    with pytest.raises(ManifestBuildError, match="duplicate member path"):
        inspect_distribution_archives([duplicate_wheel, sdist], repository_root=repository_root)

    for label, member_name in (
        ("sdist_traversal", "beyondcxr-1.0.0/src/beyondcxr/../../private/rows.csv"),
        ("sdist_backslash", "beyondcxr-1.0.0\\private\\rows.csv"),
    ):
        candidate = tmp_path / f"{label}.tar.gz"
        with tarfile.open(candidate, "w:gz") as archive:
            add_sdist_release(archive)
            add_tar_bytes(archive, member_name)
        with pytest.raises(ManifestBuildError, match="malformed member path"):
            inspect_distribution_archives([wheel, candidate], repository_root=repository_root)

    duplicate_sdist = tmp_path / "sdist_duplicate.tar.gz"
    with tarfile.open(duplicate_sdist, "w:gz") as archive:
        add_sdist_release(archive)
        add_tar_bytes(archive, "beyondcxr-1.0.0/pyproject.toml")
    with pytest.raises(ManifestBuildError, match="duplicate member path"):
        inspect_distribution_archives([wheel, duplicate_sdist], repository_root=repository_root)

    for label, member_type in (
        ("sdist_symlink", tarfile.SYMTYPE),
        ("sdist_hardlink", tarfile.LNKTYPE),
        ("sdist_device", tarfile.CHRTYPE),
        ("sdist_fifo", tarfile.FIFOTYPE),
    ):
        candidate = tmp_path / f"{label}.tar.gz"
        with tarfile.open(candidate, "w:gz") as archive:
            add_sdist_release(archive)
            member = tarfile.TarInfo(f"beyondcxr-1.0.0/{label}")
            member.type = member_type
            if member_type in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
                member.linkname = "beyondcxr-1.0.0/README.md"
            archive.addfile(member)
        with pytest.raises(ManifestBuildError, match="special member"):
            inspect_distribution_archives([wheel, candidate], repository_root=repository_root)


def _commit(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=BeyondCXR Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "-m",
            message,
        ],
        cwd=root,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _release_delta_repository(root: Path) -> tuple[Path, str]:
    (root / "src/beyondcxr/training").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "docs").mkdir()
    (root / "src/beyondcxr/training/science.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "configs/study.yaml").write_text("seed: 17\n", encoding="utf-8")
    bounded = f"Before\n{RESULT_START}\npending\n{RESULT_END}\nAfter\n"
    (root / "README.md").write_text(bounded, encoding="utf-8")
    (root / "docs/model_card.md").write_text(bounded, encoding="utf-8")
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## Unreleased\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "beyondcxr"\nversion = "0.1.0"\ndependencies = ["numpy"]\n',
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text("version: 0.1.0\n", encoding="utf-8")
    (root / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "beyondcxr"\nversion = "0.1.0"\n'
        'dependencies = [{ name = "numpy" }]\n\n[[package]]\nname = "numpy"\nversion = "2.0.0"\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    science = _commit(root, "science")
    return root, science


def _install_test_binding(root: Path, science: str) -> None:
    document = _binding()
    document["science_execution"]["git_commit"] = science
    destination = root / "results/symile/binding.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(validate_result_binding(document).canonical_bytes())


def _prepare_valid_release_delta(root: Path, science: str) -> None:
    _install_test_binding(root, science)
    for relative in ("README.md", "docs/model_card.md"):
        path = root / relative
        path.write_text(
            replace_result_region(path.read_text(encoding="utf-8"), "bound evidence"),
            encoding="utf-8",
        )
    for relative, old, new in (
        ("pyproject.toml", 'version = "0.1.0"', 'version = "1.0.0"'),
        ("CITATION.cff", "version: 0.1.0", "version: 1.0.0"),
        ("uv.lock", 'version = "0.1.0"', 'version = "1.0.0"'),
    ):
        path = root / relative
        path.write_text(path.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## 1.0.0 — 2026-09-17\n", encoding="utf-8")


def test_release_delta_accepts_only_mechanical_release_changes(tmp_path: Path) -> None:
    root, science = _release_delta_repository(tmp_path / "allowed")
    _prepare_valid_release_delta(root, science)
    _commit(root, "release")
    _check_release_delta(root)


def test_release_delta_rejects_forbidden_source_renamed_into_results(tmp_path: Path) -> None:
    root, science = _release_delta_repository(tmp_path / "forbidden-rename")
    _prepare_valid_release_delta(root, science)
    destination = root / "results/symile/science.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "mv", "src/beyondcxr/training/science.py", destination.relative_to(root)],
        cwd=root,
        check=True,
    )
    _commit(root, "forbidden rename")
    with pytest.raises(ManifestBuildError, match="Unauthorized post-science change"):
        _check_release_delta(root)


def test_release_delta_allows_result_to_result_rename(tmp_path: Path) -> None:
    root, _ = _release_delta_repository(tmp_path / "allowed-rename")
    original = root / "results/symile/old.csv"
    original.parent.mkdir(parents=True)
    original.write_text("metric,value\nauroc,0.5\n", encoding="utf-8")
    science = _commit(root, "add science result surface")
    _prepare_valid_release_delta(root, science)
    subprocess.run(
        ["git", "mv", "results/symile/old.csv", "results/symile/new.csv"],
        cwd=root,
        check=True,
    )
    _commit(root, "allowed result rename")
    _check_release_delta(root)


def test_release_delta_rejects_science_config_prose_and_dependency_drift(tmp_path: Path) -> None:
    cases = (
        ("science", "src/beyondcxr/training/science.py", "VALUE = 2\n", "Unauthorized"),
        ("config", "configs/study.yaml", "seed: 42\n", "Unauthorized"),
        (
            "prose",
            "README.md",
            f"Changed\n{RESULT_START}\npending\n{RESULT_END}\nAfter\n",
            "outside result region",
        ),
        (
            "dependency",
            "uv.lock",
            'version = 1\n\n[[package]]\nname = "beyondcxr"\nversion = "1.0.0"\n'
            'dependencies = [{ name = "pandas" }]\n',
            "not version-only",
        ),
        (
            "result_index",
            "results/README.md",
            "changed result documentation\n",
            "Unauthorized",
        ),
        (
            "changelog",
            "CHANGELOG.md",
            "# Changelog\n\n## 1.0.0 — 2026-09-17\n\nChanged body\n",
            "heading-only",
        ),
        (
            "invalid_date",
            "CHANGELOG.md",
            "# Changelog\n\n## 1.0.0 — 2026-99-99\n",
            "date is invalid",
        ),
    )
    for name, relative, content, message in cases:
        root, science = _release_delta_repository(tmp_path / name)
        _install_test_binding(root, science)
        if relative != "CHANGELOG.md":
            (root / "CHANGELOG.md").write_text(
                "# Changelog\n\n## 1.0.0 — 2026-09-17\n", encoding="utf-8"
            )
        (root / relative).write_text(content, encoding="utf-8")
        _commit(root, "invalid release")
        with pytest.raises(ManifestBuildError, match=message):
            _check_release_delta(root)


def test_release_delta_rejects_unchanged_unreleased_changelog(tmp_path: Path) -> None:
    root, science = _release_delta_repository(tmp_path / "unchanged")
    _install_test_binding(root, science)
    _commit(root, "result binding only")
    with pytest.raises(ManifestBuildError, match="heading-only"):
        _check_release_delta(root)


def test_final_metadata_requires_v1_while_nonfinal_accepts_development_version(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "beyondcxr"\nversion = "0.1.0"\nlicense = "Apache-2.0"\n'
        'license-files = ["LICENSE", "THIRD_PARTY_NOTICES.md"]\n',
        encoding="utf-8",
    )
    (tmp_path / "CITATION.cff").write_text(
        'title: "BeyondCXR"\ntype: software\nversion: 0.1.0\nlicense: Apache-2.0\n'
        'repository-code: "https://github.com/Youssef-SH/BeyondCXR"\n',
        encoding="utf-8",
    )
    (tmp_path / "LICENSE").write_text("license\n", encoding="utf-8")
    (tmp_path / "THIRD_PARTY_NOTICES.md").write_text("notices\n", encoding="utf-8")
    _check_package_metadata(tmp_path)
    with pytest.raises(ManifestBuildError, match="1.0.0"):
        _check_package_metadata(tmp_path, final=True)
    for relative, old, new in (
        ("pyproject.toml", 'version = "0.1.0"', 'version = "1.0.0"'),
        ("CITATION.cff", "version: 0.1.0", "version: 1.0.0"),
    ):
        path = tmp_path / relative
        path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    _check_package_metadata(tmp_path, final=True)
    (tmp_path / "THIRD_PARTY_NOTICES.md").unlink()
    with pytest.raises(ManifestBuildError, match="release contract"):
        _check_package_metadata(tmp_path, final=True)


def _release_check_repository(root: Path) -> None:
    (root / "docs").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "README.md").write_text(
        f"{RESULT_START}\nAwaiting formal Symile execution\n{RESULT_END}\n", encoding="utf-8"
    )
    (root / "docs/model_card.md").write_text(
        f"{RESULT_START}\nAwaiting formal Symile execution\n{RESULT_END}\n", encoding="utf-8"
    )
    (root / "results/README.md").write_text("# Results\n", encoding="utf-8")
    (root / "Makefile").write_text("release-check:\n\t@true\n", encoding="utf-8")
    (root / ".dockerignore").write_text(
        "*\n!pyproject.toml\n!uv.lock\n!README.md\n!LICENSE\n"
        "!THIRD_PARTY_NOTICES.md\n!src/\n!src/**\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "beyondcxr"\nversion = "0.1.0"\n'
        'license = "Apache-2.0"\n'
        'license-files = ["LICENSE", "THIRD_PARTY_NOTICES.md"]\n',
        encoding="utf-8",
    )
    for name in ("uv.lock", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        (root / name).write_text("public\n", encoding="utf-8")
    (root / "CITATION.cff").write_text(
        'title: "BeyondCXR"\ntype: software\nversion: 0.1.0\nlicense: Apache-2.0\n'
        'repository-code: "https://github.com/Youssef-SH/BeyondCXR"\n',
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("failure", "destination"),
    (
        (OSError("ordinary failure"), "model-card"),
        (KeyboardInterrupt(), "readme"),
        (KeyboardInterrupt(), "model-card"),
    ),
)
def test_result_install_rolls_back_all_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    destination: str,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "old.txt").write_text("old results", encoding="utf-8")
    readme = tmp_path / "README.md"
    model_card = tmp_path / "model-card.md"
    readme.write_bytes(b"old readme")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "new.txt").write_text("new results", encoding="utf-8")

    import beyondcxr.release.reproduction as reproduction

    real_replace = reproduction.os.replace
    injected = False

    failure_target = readme if destination == "readme" else model_card

    def fail_install(source: Path, destination: Path) -> None:
        nonlocal injected
        if not injected and destination == failure_target and source.parent.name != "backups":
            injected = True
            raise failure
        real_replace(source, destination)

    monkeypatch.setattr(reproduction.os, "replace", fail_install)
    with pytest.raises(type(failure)):
        _install_with_rollback(
            candidate_results=candidate,
            output_root=results,
            readme_path=readme,
            readme=b"new readme",
            model_card_path=model_card,
            model_card=b"new model card",
        )
    assert (results / "old.txt").read_text(encoding="utf-8") == "old results"
    assert readme.read_bytes() == b"old readme"
    assert not model_card.exists()


def test_result_install_surfaces_transient_rollback_failure_after_restoration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "old.txt").write_text("old results", encoding="utf-8")
    readme = tmp_path / "README.md"
    model_card = tmp_path / "model-card.md"
    readme.write_bytes(b"old readme")
    model_card.write_bytes(b"old model card")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "new.txt").write_text("new results", encoding="utf-8")

    import beyondcxr.release.reproduction as reproduction

    real_replace = reproduction.os.replace
    install_failed = False
    rollback_failed = False

    def fail_install_and_rollback(source: Path, destination: Path) -> None:
        nonlocal install_failed, rollback_failed
        if not install_failed and destination == model_card and source.parent.name != "backups":
            install_failed = True
            raise OSError("installation failure")
        if install_failed and not rollback_failed and source.parent.name == "backups":
            rollback_failed = True
            raise OSError("rollback failure")
        real_replace(source, destination)

    monkeypatch.setattr(reproduction.os, "replace", fail_install_and_rollback)
    with pytest.raises(BaseExceptionGroup, match="rollback encountered failures") as caught:
        _install_with_rollback(
            candidate_results=candidate,
            output_root=results,
            readme_path=readme,
            readme=b"new readme",
            model_card_path=model_card,
            model_card=b"new model card",
        )
    messages = {str(error) for error in caught.value.exceptions}
    assert {"installation failure", "rollback failure"} <= messages
    assert (results / "old.txt").read_text(encoding="utf-8") == "old results"
    assert readme.read_bytes() == b"old readme"
    assert model_card.read_bytes() == b"old model card"


def test_serving_release_provenance_normalizes_git_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_git(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise subprocess.CalledProcessError(128, ["git", "status"])

    monkeypatch.setattr(subprocess, "run", fail_git)
    with pytest.raises(ManifestBuildError, match="valid Git checkout"):
        clean_release_provenance(tmp_path)


def test_failed_serving_smoke_leaves_authority_root_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import beyondcxr.release.serving as serving

    destination = tmp_path / "authorities"
    destination.mkdir()
    (destination / "existing.txt").write_text("preserved", encoding="utf-8")
    package_directory = tmp_path / "packages" / "final-package-test"
    package_directory.mkdir(parents=True)
    campaign = SimpleNamespace(
        freeze=object(),
        global_result=object(),
        predictions=(),
        test_data=object(),
        packages=(SimpleNamespace(directory=package_directory),),
    )

    monkeypatch.setattr(serving, "clean_release_provenance", lambda _: {})
    monkeypatch.setattr(serving, "validate_restored_campaign", lambda _: campaign)

    def restore_with_real_lifetime(_: Path, restoration_validator: object) -> object:
        with tempfile.TemporaryDirectory() as restored:
            return restoration_validator(Path(restored))

    monkeypatch.setattr(serving, "restore_and_validate_export", restore_with_real_lifetime)

    def publish_candidate(*, authority_root: Path, **_: object) -> SimpleNamespace:
        candidate = authority_root / "serving-authority-test"
        candidate.mkdir(parents=True)
        return SimpleNamespace(directory=candidate, authority_id=candidate.name)

    monkeypatch.setattr(serving, "publish_serving_authority", publish_candidate)
    monkeypatch.setattr(
        serving.SymileServingPredictor,
        "load",
        lambda *_args, **_kwargs: SimpleNamespace(
            authority=SimpleNamespace(authority_id="serving-authority-test"),
            model_info=lambda: {},
        ),
    )

    async def fail_smoke(*_: object, **__: object) -> None:
        raise ManifestBuildError("injected smoke failure")

    monkeypatch.setattr(serving, "_exercise_api", fail_smoke)
    with pytest.raises(ManifestBuildError, match="injected smoke failure"):
        serving.publish_and_smoke_test_serving_authority(
            artifact_root=tmp_path / "export.zip",
            authority_root=destination,
            repository_root=tmp_path / "checkout",
        )
    assert {path.name for path in destination.iterdir()} == {"existing.txt"}

    symlink = tmp_path / "authority-link"
    symlink.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ManifestBuildError, match="must not be a symlink"):
        serving.publish_and_smoke_test_serving_authority(
            artifact_root=tmp_path / "export.zip",
            authority_root=symlink,
            repository_root=tmp_path / "checkout",
        )


@pytest.mark.parametrize("divergent", (False, True))
def test_existing_serving_authority_is_idempotent_only_when_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, divergent: bool
) -> None:
    import beyondcxr.release.serving as serving

    destination = tmp_path / "authorities"
    existing = destination / "serving-authority-test"
    existing.mkdir(parents=True)
    (existing / "manifest.json").write_bytes(b"different" if divergent else b"authority")
    monkeypatch.setattr(serving, "clean_release_provenance", lambda _: {})

    def restored_campaign(root: Path) -> SimpleNamespace:
        package_directory = root / "packages/final-package-test"
        package_directory.mkdir(parents=True)
        return SimpleNamespace(
            freeze=object(),
            global_result=object(),
            predictions=(),
            test_data=object(),
            packages=(SimpleNamespace(directory=package_directory),),
        )

    monkeypatch.setattr(serving, "validate_restored_campaign", restored_campaign)

    def restore_with_real_lifetime(_: Path, restoration_validator: object) -> object:
        with tempfile.TemporaryDirectory() as restored:
            result = restoration_validator(Path(restored))
        assert not Path(restored).exists()
        return result

    monkeypatch.setattr(serving, "restore_and_validate_export", restore_with_real_lifetime)

    def publish_candidate(*, authority_root: Path, **_: object) -> SimpleNamespace:
        candidate = authority_root / "serving-authority-test"
        candidate.mkdir(parents=True)
        (candidate / "manifest.json").write_bytes(b"authority")
        return SimpleNamespace(directory=candidate, authority_id=candidate.name)

    monkeypatch.setattr(serving, "publish_serving_authority", publish_candidate)
    monkeypatch.setattr(
        serving,
        "validate_serving_authority",
        lambda *_args, **_kwargs: SimpleNamespace(authority_id="serving-authority-test"),
    )
    monkeypatch.setattr(
        serving.SymileServingPredictor,
        "load",
        lambda *_args, **_kwargs: SimpleNamespace(
            authority=SimpleNamespace(authority_id="serving-authority-test"),
            model_info=lambda: {},
        ),
    )

    async def pass_smoke(*_: object, **__: object) -> None:
        return None

    monkeypatch.setattr(serving, "_exercise_api", pass_smoke)

    def operation() -> Path:
        return serving.publish_and_smoke_test_serving_authority(
            artifact_root=tmp_path / "export.zip",
            authority_root=destination,
            repository_root=tmp_path / "checkout",
        )

    if divergent:
        with pytest.raises(ManifestBuildError, match="differs from candidate"):
            operation()
        assert (existing / "manifest.json").read_bytes() == b"different"
    else:
        assert operation() == existing
        assert (existing / "manifest.json").read_bytes() == b"authority"


def _stub_final_acceptance(monkeypatch: pytest.MonkeyPatch, *, fail_after_smoke: bool) -> None:
    import beyondcxr.release.checks as checks

    monkeypatch.setattr(checks, "check_repository", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(checks, "inspect_distribution_archives", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(checks, "_run_container_serving_acceptance", lambda **_kwargs: None)
    monkeypatch.setattr(checks.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(checks.shutil, "which", lambda command: command)

    def run(command: list[str], cwd: Path) -> None:
        if command[:2] == ["git", "clone"]:
            Path(command[-1]).mkdir(parents=True)
        if command[:2] == ["uv", "build"]:
            (cwd / "dist").mkdir()
        if fail_after_smoke and command[:2] == ["docker", "build"]:
            raise ManifestBuildError("injected later acceptance failure")

    def capture(command: list[str], cwd: Path) -> str:
        del cwd
        authority_root = Path(command[command.index("--authority-root") + 1])
        candidate = authority_root / "serving-authority-test"
        candidate.mkdir(parents=True)
        (candidate / "manifest.json").write_text("accepted", encoding="utf-8")
        return candidate.name

    monkeypatch.setattr(checks, "_run", run)
    monkeypatch.setattr(checks, "_run_capture", capture)


def test_final_acceptance_publishes_authority_only_after_later_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_final_acceptance(monkeypatch, fail_after_smoke=False)
    authority_root = tmp_path / "authorities"
    run_final_acceptance(
        tmp_path / "repository",
        artifact_root=tmp_path / "artifact.zip",
        authority_root=authority_root,
    )
    assert (authority_root / "serving-authority-test/manifest.json").read_text() == "accepted"


def test_final_acceptance_later_failure_leaves_no_new_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_final_acceptance(monkeypatch, fail_after_smoke=True)
    authority_root = tmp_path / "authorities"
    with pytest.raises(ManifestBuildError, match="later acceptance failure"):
        run_final_acceptance(
            tmp_path / "repository",
            artifact_root=tmp_path / "artifact.zip",
            authority_root=authority_root,
        )
    assert not authority_root.exists()


def test_final_acceptance_failure_preserves_existing_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_final_acceptance(monkeypatch, fail_after_smoke=True)
    authority_root = tmp_path / "authorities"
    existing = authority_root / "serving-authority-test"
    existing.mkdir(parents=True)
    (existing / "manifest.json").write_text("existing", encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="later acceptance failure"):
        run_final_acceptance(
            tmp_path / "repository",
            artifact_root=tmp_path / "artifact.zip",
            authority_root=authority_root,
        )
    assert (existing / "manifest.json").read_text() == "existing"


def test_final_acceptance_reports_missing_docker_without_cleanup_masking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import beyondcxr.release.checks as checks

    _stub_final_acceptance(monkeypatch, fail_after_smoke=False)
    monkeypatch.setattr(
        checks.shutil, "which", lambda command: None if command == "docker" else command
    )
    with pytest.raises(ManifestBuildError, match="requires docker"):
        run_final_acceptance(
            tmp_path / "repository",
            artifact_root=tmp_path / "artifact.zip",
            authority_root=tmp_path / "authorities",
        )


def test_reproduction_rejects_extra_and_stale_public_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import beyondcxr.release.reproduction as reproduction

    binding = validate_result_binding(_binding())
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(binding.canonical_bytes())
    results = tmp_path / "results"
    results.mkdir()
    (results / "README.md").write_bytes(b"result index")
    (results / "symile").mkdir()
    (results / "symile/binding.json").write_bytes(binding.canonical_bytes())
    (results / "symile/data").mkdir()
    ecg_table = b"estimate,model,repeat,roc_auc\nECG extension OOF repeat,ECG gated,17,0.6\n"
    (results / "symile/data/development_performance.csv").write_bytes(ecg_table)
    readme = tmp_path / "README.md"
    card = tmp_path / "model-card.md"
    bounded_readme = f"before\n{RESULT_START}\nbound readme\n{RESULT_END}\nafter\n".encode()
    bounded_card = f"before\n{RESULT_START}\nbound model card\n{RESULT_END}\nafter\n".encode()
    readme.write_bytes(bounded_readme)
    card.write_bytes(bounded_card)

    def regenerated(**_: object) -> _Regenerated:
        workspace = tempfile.TemporaryDirectory()
        root = Path(workspace.name) / "results"
        root.mkdir()
        (root / "README.md").write_bytes(b"result index")
        (root / "symile").mkdir()
        (root / "symile/binding.json").write_bytes(binding.canonical_bytes())
        (root / "symile/data").mkdir()
        (root / "symile/data/development_performance.csv").write_bytes(ecg_table)
        return _Regenerated(root, bounded_readme, bounded_card, workspace)

    monkeypatch.setattr(reproduction, "_regenerate", regenerated)
    arguments = {
        "binding_path": binding_path,
        "artifact_root": tmp_path / "preserved.zip",
        "output_root": results,
        "readme_path": readme,
        "model_card_path": card,
    }
    reproduce_results(**arguments)
    development_table = results / "symile/data/development_performance.csv"
    development_table.unlink()
    with pytest.raises(ManifestBuildError, match="do not reproduce exactly"):
        reproduce_results(**arguments)
    development_table.write_bytes(ecg_table)
    unexpected = results / "unexpected.txt"
    unexpected.write_text("extra", encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="do not reproduce exactly"):
        reproduce_results(**arguments)
    unexpected.unlink()
    readme.write_bytes(f"before\n{RESULT_START}\nstale\n{RESULT_END}\nafter\n".encode())
    with pytest.raises(ManifestBuildError, match="README result region"):
        reproduce_results(**arguments)
    readme.write_bytes(bounded_readme)
    card.write_bytes(f"before\n{RESULT_START}\nstale\n{RESULT_END}\nafter\n".encode())
    with pytest.raises(ManifestBuildError, match="Model-card result region"):
        reproduce_results(**arguments)


@pytest.mark.parametrize(
    ("relative", "content", "message"),
    (
        (
            "docs/private.md",
            fixture_text("machine: /", "home/researcher/data\n"),
            "private material",
        ),
        (
            "docs/secret.md",
            fixture_text('"api', '_key": "abcdefghijklmnop"\n'),
            "private material",
        ),
        ("docs/token.md", fixture_text("to", "ken: abcdefghijklmnop\n"), "private material"),
        (
            "private/patients.csv",
            fixture_text("subject", f"_id,value\n{12_345_678},4\n"),
            "row-level material",
        ),
        (
            "data/raw/patients.tsv",
            fixture_text("sample", f"_id\ttarget\n{_symile_source_sample_id()}\t1\n"),
            "row-level material",
        ),
        (
            "scripts/credentials.txt",
            fixture_text("pass", "word=abcdefghijklmnop\n"),
            "private material",
        ),
        (
            "scripts/credentials.sh",
            fixture_text("export API", "_KEY=abcdefghijklmnop\n"),
            "private material",
        ),
        ("config/runtime.ini", fixture_text("to", "ken=abcdefghijklmnop\n"), "private material"),
        ("config/.env", fixture_text("SEC", "RET=abcdefghijklmnop\n"), "private material"),
        ("DEPLOYMENT", fixture_text("pass", "word=abcdefghijklmnop\n"), "private material"),
        (
            "tools/path.txt",
            fixture_text("source: /", "home/researcher/private\n"),
            "private material",
        ),
        (
            "src/beyondcxr/leaked_config.py",
            fixture_text("API", '_KEY = "abcdefghijklmnop', 'REALVALUE"\n'),
            "private material",
        ),
        (
            "src/beyondcxr/leaked_path.py",
            fixture_text('SOURCE_ROOT = "/', 'home/user/private/mimic"\n'),
            "private material",
        ),
        (
            "src/beyondcxr/leaked_macos_path.py",
            fixture_text('SOURCE_ROOT = "/', 'Users/user/private/mimic"\n'),
            "private material",
        ),
        (
            "src/beyondcxr/leaked_windows_path.py",
            fixture_text('SOURCE_ROOT = r"C:\\', 'Users\\user\\private\\mimic"\n'),
            "private material",
        ),
        (
            "src/beyondcxr/exception_secret.py",
            fixture_text('raise RuntimeError("to', 'ken=abcdefghijklmnop")\n'),
            "private material",
        ),
        (
            "src/beyondcxr/exception_path.py",
            fixture_text('raise OSError("/', 'home/user/private/file")\n'),
            "private material",
        ),
        (
            "src/beyondcxr/comment_path.py",
            fixture_text("# local source: /", "home/user/private/data\n"),
            "private material",
        ),
        (
            "src/beyondcxr/comment_secret.py",
            fixture_text("# to", "ken=abcdefghijklmnop\n"),
            "private material",
        ),
        (
            "scratch/rows.json",
            fixture_text('{"subject', f'_id": {12_345_678}}}\n'),
            "row-level material",
        ),
        ("reports/summary.txt", "aggregate only\n", "Generated or private repository state"),
        ("docs/broken.md", "[missing](absent.md)\n", "Broken documentation link"),
        ("docs/command.md", "```bash\nmake nonexistent-target\n```\n", "Unknown Make command"),
        (
            "results/rows.json",
            fixture_text('"subject', f'_id": {12_345_678}\n'),
            "row-level material",
        ),
        (
            "results/rows.txt",
            fixture_text("subject", f"_id={12_345_678}\n"),
            "row-level material",
        ),
        (
            "src/beyondcxr/rows.csv",
            fixture_text("subject", f"_id,value\n{12_345_678},4\n"),
            "row-level material",
        ),
        (
            "src/beyondcxr/string_rows.csv",
            fixture_text("sample", f"_id,target,probability\n{_symile_source_sample_id()},1,0.8\n"),
            "row-level material",
        ),
        (
            "src/beyondcxr/last_id.csv",
            fixture_text("target,sample", f"_id\n1,{_symile_source_sample_id()}\n"),
            "row-level material",
        ),
        (
            "src/beyondcxr/only_id.csv",
            fixture_text("sample", f"_id\n{_symile_source_sample_id()}\n"),
            "row-level material",
        ),
        (
            "src/beyondcxr/quoted_rows.csv",
            fixture_text('"target","sample', f'_id"\n1,"{_symile_source_sample_id()}"\n'),
            "row-level material",
        ),
        (
            "docs/rows.md",
            fixture_text(
                "| target | sample",
                f"_id |\n| --- | --- |\n| 1 | {_symile_source_sample_id()} |\n",
            ),
            "row-level material",
        ),
        (
            "src/beyondcxr/rows.tsv",
            fixture_text("target\tsample", f"_id\n1\t{_symile_source_sample_id()}\n"),
            "row-level material",
        ),
        (
            "results/rows.txt",
            fixture_text("sample", f"_id target\n{_symile_source_sample_id()} 1\n"),
            "row-level material",
        ),
        (
            "docs/natural-id.md",
            fixture_text("sample", f"_id: {_symile_source_sample_id()}\n"),
            "row-level material",
        ),
        (
            "docs/shell.md",
            "```bash\nmake release-check BUNDLE_ID=bundle-<sha256>\n```\n",
            "Shell-invalid documented Bash command",
        ),
        ("unexpected_root_file", "unexpected\n", "Unexpected tracked root file"),
        ("model.pt", "restricted\n", "Restricted artifact"),
    ),
)
def test_repository_check_rejects_public_release_leaks(
    tmp_path: Path, relative: str, content: str, message: str
) -> None:
    root = tmp_path / "repository"
    _release_check_repository(root)
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    with pytest.raises(ManifestBuildError, match=message):
        check_repository(root)


@pytest.mark.parametrize("broken", (False, True))
def test_repository_check_rejects_tracked_symlinks(tmp_path: Path, broken: bool) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    target = root / "target.txt"
    if not broken:
        target.write_text("public\n", encoding="utf-8")
    link = root / "tracked-link.txt"
    link.symlink_to(target.name)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "tracked-link.txt"], cwd=root, check=True)
    with pytest.raises(ManifestBuildError, match="Tracked symlink"):
        check_repository(root)


def test_repository_check_rejects_non_utf8_text_candidate(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    _release_check_repository(root)
    candidate = root / "config/settings.conf"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"public=true\n\xff")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    with pytest.raises(ManifestBuildError, match="not UTF-8"):
        check_repository(root)


def test_repository_check_allows_synthetic_identifiers_across_text_formats(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    _release_check_repository(root)
    fixtures = {
        "tests/fixtures/rows.py": (
            '{"sample_id": "rsna:a", "patient_id": "patient-positive", '
            '"image_id": "synthetic-image"}\n'
        ),
        "tests/fixtures/rows.json": '{"patient_id": "patient-positive"}\n',
        "tests/fixtures/rows.csv": "sample_id,target\nvalidation-negative,0\n",
    }
    for relative, content in fixtures.items():
        fixture = root / relative
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    check_repository(root)


@pytest.mark.parametrize(
    ("relative", "content"),
    (
        ("src/beyondcxr/row.py", f'row = {{"sample_id": {_symile_source_sample_id()!r}}}\n'),
        ("tests/fixtures/row.json", f'{{"hadm_id": {23_456_789}}}\n'),
        ("tests/fixtures/row.yaml", f"cxr_study_id: {50_123_456}\n"),
        ("tests/fixtures/row.csv", f"hadm_id,target\n{23_456_789},1\n"),
    ),
)
def test_repository_check_rejects_source_identifiers_across_text_formats(
    tmp_path: Path, relative: str, content: str
) -> None:
    root = tmp_path / "repository"
    _release_check_repository(root)
    fixture = root / relative
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    with pytest.raises(ManifestBuildError, match="row-level material"):
        check_repository(root)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sample_id", _symile_source_sample_id()),
        ("subject_id", 12_345_678),
        ("hadm_id", 23_456_789),
        ("dicom_id", _mimic_cxr_source_dicom_id()),
        ("cxr_dicom_id", _mimic_cxr_source_dicom_id()),
        ("cxr_24_72_hr", _mimic_cxr_source_dicom_id()),
        ("cxr_study_id", 50_123_456),
        ("ecg_study_id", 40_123_456),
        ("ecg_adm", 40_123_456.0),
        ("ecg_file_name", "40123456"),
    ),
)
def test_repository_check_rejects_current_source_identifiers(
    tmp_path: Path, field: str, value: object
) -> None:
    root = tmp_path / "repository"
    _release_check_repository(root)
    fixture = root / "src/beyondcxr/rows.py"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(f"row = {{{field!r}: {value!r}}}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    with pytest.raises(ManifestBuildError, match="row-level material"):
        check_repository(root)


@pytest.mark.parametrize("raw_literal", (False, True))
def test_repository_check_rejects_decoded_python_windows_private_path(
    tmp_path: Path, raw_literal: bool
) -> None:
    root = tmp_path / "repository"
    _release_check_repository(root)
    fixture = root / "src/beyondcxr/settings.py"
    fixture.parent.mkdir(parents=True)
    private_path = str(PureWindowsPath("C:\\", "Users", "researcher", "private"))
    source = (
        f'SOURCE_ROOT = r"{private_path}"\n' if raw_literal else f"SOURCE_ROOT = {private_path!r}\n"
    )
    fixture.write_text(source, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    with pytest.raises(ManifestBuildError, match="private material"):
        check_repository(root)


@pytest.mark.parametrize(
    "value",
    (
        str(Path("/", "home", "researcher", "private")),
        f"token={'a' * 16}",
    ),
)
def test_repository_check_rejects_private_material_in_decoded_python_string(
    tmp_path: Path, value: str
) -> None:
    root = tmp_path / "repository"
    _release_check_repository(root)
    fixture = root / "src/beyondcxr/errors.py"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(f"raise RuntimeError({value!r})\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    with pytest.raises(ManifestBuildError, match="private material"):
        check_repository(root)


@pytest.mark.parametrize(
    "source",
    (
        f'# "hadm_id": {23_456_789}\n',
        f'DEBUG_ROW = \'{{"patient_id": "{_rsna_source_identifier()}"}}\'\n',
        f'DEBUG_ROW = \'{{"cxr_dicom_id": "{_mimic_cxr_source_dicom_id()}"}}\'\n',
    ),
)
def test_repository_check_rejects_quoted_source_identifier_in_python_text(
    tmp_path: Path, source: str
) -> None:
    root = tmp_path / "repository"
    _release_check_repository(root)
    fixture = root / "src/beyondcxr/debug.py"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(source, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    with pytest.raises(ManifestBuildError, match="row-level material"):
        check_repository(root)


@pytest.mark.parametrize(
    "identifier",
    (_mimic_cxr_source_dicom_id(), _rsna_source_identifier(), _symile_source_sample_id()),
)
def test_repository_check_rejects_source_identifier_in_tracked_path(
    tmp_path: Path, identifier: str
) -> None:
    root = tmp_path / "repository"
    _release_check_repository(root)
    fixture = root / "docs" / f"{identifier}.txt"
    fixture.write_text("aggregate documentation\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    with pytest.raises(ManifestBuildError, match="Tracked path contains row-level material"):
        check_repository(root)


def test_private_path_detector_is_naturally_self_hosting() -> None:
    detector_source = (
        '_PATH_SEPARATOR = "/"\n'
        '_HOME_DIRECTORY = "home"\n'
        '_USERS_DIRECTORY = "Users"\n'
        '_WINDOWS_SEPARATOR = "\\\\"\n'
    )
    assert _PRIVATE_PATH.search(detector_source) is None
    private_paths = (
        fixture_text("/", "home/researcher/data"),
        fixture_text("/", "Users/researcher/data"),
        fixture_text("C:\\", "Users\\researcher\\data"),
    )
    assert all(_PRIVATE_PATH.search(path) is not None for path in private_paths)


def test_repository_check_rejects_opaque_tracked_binary(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    _release_check_repository(root)
    candidate = root / "docs/image.png"
    candidate.write_bytes(b"\x89PNG\r\n\x1a\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    with pytest.raises(ManifestBuildError, match="Opaque tracked binary"):
        check_repository(root)


def test_generic_release_commands_do_not_import_serving_dependencies() -> None:
    script = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'fastapi', 'httpx', 'uvicorn'}:
        raise AssertionError(f'unexpected serving dependency: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import beyondcxr.release.cli as cli
import beyondcxr.release.reproduction as reproduction
cli.check_repository = lambda _root: None
reproduction.publish_results = lambda **_kwargs: None
reproduction.reproduce_results = lambda **_kwargs: None
assert cli.main(['check', '--root', '.']) == 0
assert cli.main(['results', '--artifact-root', 'artifact.zip']) == 0
assert cli.main(['reproduce', '--artifact-root', 'artifact.zip']) == 0
"""
    subprocess.run([sys.executable, "-c", script], check=True)
