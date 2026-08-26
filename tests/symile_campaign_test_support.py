from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from symile_data_test_support import _published_synthetic_release
from symile_development_test_support import _development_frame, _publish_family_folds

import beyondcxr.training.symile_campaign_control as campaign_control
import beyondcxr.training.symile_data as symile_data
import beyondcxr.training.symile_ecg_extension_result as extension_result
import beyondcxr.training.symile_final_packages as final_packages
import beyondcxr.training.symile_statistics as symile_statistics
import beyondcxr.training.symile_test_data as symile_test_data
import beyondcxr.utils.symile_publication as symile_publication
from beyondcxr.data.symile_artifacts import validate_symile_bundle_reference
from beyondcxr.data.symile_cv import publish_symile_cv
from beyondcxr.data.symile_preprocess import LAB_FEATURE_COLUMNS, LAB_OBSERVED_COLUMNS
from beyondcxr.models.symile_tabular import (
    fit_final_symile_labs_lightgbm,
    fit_symile_labs_logistic,
)
from beyondcxr.training.config import load_symile_development_config, with_runtime
from beyondcxr.training.symile_campaign_control import (
    PRETEST_FREEZE_PREFIX,
    ValidatedGlobalResult,
    ValidatedPretestFreeze,
)
from beyondcxr.training.symile_families import (
    FINAL_NEURAL_MEMBER_SEEDS,
    FINAL_PACKAGE_COUNT,
    FINAL_PACKAGE_POLICY,
    FINAL_TABULAR_SEED,
    SYMILE_CORE_DEVELOPMENT_FAMILIES,
    SYMILE_ECG_GATED_FAMILY,
)
from beyondcxr.training.symile_final_packages import (
    ValidatedFinalPackage,
    final_training_plan,
    publish_final_neural_package,
    publish_final_tabular_package,
)
from beyondcxr.training.symile_statistics import focused_development_subgroups
from beyondcxr.training.symile_test_data import (
    HeldOutEvaluationProjection,
)
from beyondcxr.utils.package_identity import canonical_scientific_id
from beyondcxr.utils.private_predictions import (
    SYMILE_TEST_INFERENCE_POLICY,
    ValidatedPredictionEvidence,
    build_prediction_table,
)
from beyondcxr.utils.symile_publication import (
    ValidatedDevelopmentResult,
    publish_analysis_result,
    publish_development_result,
)


def _final_provenance(*, neural=False):
    result = {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "dependency_lock_sha256": "b" * 64,
        "mlflow_run_id": "synthetic-final-run",
    }
    if neural:
        result.update(
            runtime_provenance={"pin_memory_effective": False},
            loader_execution={
                "lifecycle": "reused",
                "num_workers": 0,
                "pin_memory": False,
                "batch_size": 2,
                "drop_last": False,
                "shuffle": False,
                "sampler": "epoch_permutation",
                "persistent_workers": False,
                "prefetch_factor": None,
                "multiprocessing_context": None,
            },
        )
    return result


def _freeze(
    tmp_path: Path, suffix: str = "a", *, neural_runtime: dict[str, object] | None = None
) -> ValidatedPretestFreeze:
    neural_runtime = neural_runtime or {
        "device_type": "cpu",
        "autocast_dtype": None,
        "cuda_runtime_version": None,
        "cudnn_version": None,
        "gpu_device_name": None,
        "gpu_compute_capability": None,
    }
    semantic = {
        "bundle": {
            "bundle_id": "bundle-" + suffix * 64,
            "bundle_manifest_sha256": "b" * 64,
            "split_assignment_id": "split-assignment-" + "c" * 64,
        },
        "task": {
            "task_id": "pneumonia_strict",
            "label_policy_version": "symile-pneumonia-strict-v1",
        },
        "ecg_extension_result_id": "ecg-extension-result-" + "d" * 64,
        "final_packages": [
            {
                "package_id": "final-package-" + f"{index:064x}",
                "manifest_sha256": "e" * 64,
            }
            for index in range(FINAL_PACKAGE_COUNT)
        ],
        "held_out_policy": {
            "metric_policy": symile_statistics.METRIC_POLICY,
            "bootstrap": dict(symile_statistics.BOOTSTRAP_POLICY),
            "neural_inference_runtime": neural_runtime,
        },
        "primary_thresholds": {"youden_j": 0.5, "target_sensitivity": 0.25},
        "science_git_commit": "f" * 40,
        "dependency_lock_sha256": "0" * 64,
    }
    freeze_id = canonical_scientific_id(PRETEST_FREEZE_PREFIX, semantic)
    directory = tmp_path / freeze_id
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "pretest_freeze_schema_version": 1,
                "pretest_freeze_id": freeze_id,
                **semantic,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    raw = (directory / "manifest.json").read_bytes()
    return ValidatedPretestFreeze(
        directory,
        json.loads(raw),
        hashlib.sha256(raw).hexdigest(),
        campaign_control._CAPABILITY_GUARD,
    )


def _synthetic_final_packages(
    tmp_path: Path,
    freeze: ValidatedPretestFreeze,
    *,
    family_authorities: dict[str, dict[str, object]] | None = None,
) -> tuple[ValidatedFinalPackage, ...]:
    family_members = tuple(
        (
            family,
            (FINAL_TABULAR_SEED,) if int(policy["members"]) == 1 else FINAL_NEURAL_MEMBER_SEEDS,
        )
        for family, policy in FINAL_PACKAGE_POLICY.items()
    )
    flattened = [(family, seed) for family, seeds in family_members for seed in seeds]
    cxr_by_seed = {
        seed: freeze.manifest["final_packages"][index]["package_id"]
        for index, (family, seed) in enumerate(flattened)
        if family == "cxr_densenet"
    }
    packages = []
    index = 0
    for family, seeds in family_members:
        for seed in seeds:
            frozen = freeze.manifest["final_packages"][index]
            packages.append(
                ValidatedFinalPackage(
                    tmp_path / f"package-{index}",
                    {
                        "final_package_id": frozen["package_id"],
                        "package_kind": "tabular" if family.startswith("labs_") else "neural",
                        "seed_policy": seed,
                        "input": (
                            family_authorities[family]["final_input"]
                            if family_authorities is not None
                            else {
                                "dataset": {
                                    "dataset_id": "symile",
                                    "bundle_id": freeze.manifest["bundle"]["bundle_id"],
                                    "split_assignment_id": freeze.manifest["bundle"][
                                        "split_assignment_id"
                                    ],
                                    "cohort": ("official_train_plus_validation_strict_pneumonia"),
                                },
                                "task": dict(freeze.manifest["task"]),
                                "family": {"family_id": family},
                            }
                        ),
                        "bundle_manifest_sha256": freeze.manifest["bundle"][
                            "bundle_manifest_sha256"
                        ],
                        "family_development_id": (
                            family_authorities[family]["development_id"]
                            if family_authorities is not None
                            else (
                                None
                                if family == "labs_logistic"
                                else "development-" + f"{index + 1:064x}"
                            )
                        ),
                        "final_training_budget": (
                            family_authorities[family]["final_training_budget"]
                            if family_authorities is not None
                            else (None if family == "labs_logistic" else 3)
                        ),
                        "pretrained_scientific_identity": (
                            family_authorities[family]["pretrained_scientific_identity"]
                            if family_authorities is not None
                            else (
                                {
                                    "declared_name": "densenet121-res224-chex",
                                    "stable_identifier": "synthetic-weight",
                                    "sha256": "a" * 64,
                                }
                                if family == "cxr_densenet"
                                else None
                            )
                        ),
                        "source_cxr_package_id": (
                            cxr_by_seed[seed]
                            if family not in {"labs_logistic", "labs_lightgbm", "cxr_densenet"}
                            else None
                        ),
                        "execution_provenance": {
                            "git_commit": freeze.manifest["science_git_commit"],
                            "git_dirty": False,
                            "dependency_lock_sha256": freeze.manifest["dependency_lock_sha256"],
                        },
                    },
                    frozen["manifest_sha256"],
                )
            )
            index += 1
    return tuple(packages)


def _synthetic_final_family_authorities(
    development_ids: dict[str, str],
    freeze: ValidatedPretestFreeze | None = None,
) -> dict[str, dict[str, object]]:
    authorities = {
        family: {
            "development_id": None if family == "labs_logistic" else development_ids[family],
            "final_training_budget": None if family == "labs_logistic" else 3,
            "final_input": final_packages._final_input_projection(
                load_symile_development_config(Path("configs") / f"symile_{family}.yaml")
            ),
            "pretrained_scientific_identity": (
                {
                    "declared_name": "densenet121-res224-chex",
                    "stable_identifier": "synthetic-weight",
                    "sha256": "a" * 64,
                }
                if family == "cxr_densenet"
                else None
            ),
        }
        for family in FINAL_PACKAGE_POLICY
    }
    if freeze is not None:
        for authority in authorities.values():
            final_input = authority["final_input"]
            final_input["dataset"] = {
                "dataset_id": "symile",
                "bundle_id": freeze.manifest["bundle"]["bundle_id"],
                "split_assignment_id": freeze.manifest["bundle"]["split_assignment_id"],
                "cohort": "official_train_plus_validation_strict_pneumonia",
            }
            final_input["task"] = dict(freeze.manifest["task"])
    return authorities


def _synthetic_test_predictions(
    tmp_path: Path,
    freeze: ValidatedPretestFreeze,
    packages: tuple[ValidatedFinalPackage, ...],
) -> tuple[ValidatedPredictionEvidence, ...]:
    sample_ids = [f"sample-{index:03d}" for index in range(110)]
    targets = np.asarray([0] * 55 + [1] * 55, dtype=np.int8)
    return tuple(
        ValidatedPredictionEvidence(
            tmp_path / f"evidence-{index}",
            {
                "prediction_id": "prediction-" + f"{index:064x}",
                "model_package_id": package.package_id,
                "authorized_by_pretest_freeze_id": freeze.freeze_id,
                "dataset_id": "symile",
                "bundle_id": freeze.manifest["bundle"]["bundle_id"],
                "split_assignment_id": freeze.manifest["bundle"]["split_assignment_id"],
                "task_id": freeze.manifest["task"]["task_id"],
                "label_policy_version": freeze.manifest["task"]["label_policy_version"],
                "scope": "test",
                "inference_policy": dict(SYMILE_TEST_INFERENCE_POLICY),
            },
            "7" * 64,
            build_prediction_table(
                sample_ids,
                targets,
                np.linspace(-2.0 + index / 100, 2.0 + index / 100, 110),
            ),
        )
        for index, package in enumerate(packages)
    )


def _projection_for_freeze(
    freeze: ValidatedPretestFreeze, *, grouped: bool
) -> HeldOutEvaluationProjection:
    sample_ids = [f"sample-{index:03d}" for index in range(110)]
    targets = np.asarray([0] * 55 + [1] * 55, dtype=np.int8)
    subjects = np.repeat(np.arange(1, 56), 2) if grouped else np.arange(1, 111)
    return HeldOutEvaluationProjection(
        dataset_id="symile",
        bundle_id=str(freeze.manifest["bundle"]["bundle_id"]),
        split_assignment_id=str(freeze.manifest["bundle"]["split_assignment_id"]),
        task_id=str(freeze.manifest["task"]["task_id"]),
        label_policy_version=str(freeze.manifest["task"]["label_policy_version"]),
        scope="test",
        inference_policy=dict(SYMILE_TEST_INFERENCE_POLICY),
        freeze_id=freeze.freeze_id,
        _frame=pd.DataFrame({"sample_id": sample_ids, "target": targets, "subject_id": subjects}),
    )


def _synthetic_opened_development_graph(
    *,
    model_root: Path,
    report_root: Path,
    private_root: Path,
    freeze: ValidatedPretestFreeze,
) -> SimpleNamespace:
    """Materialize the development authorities selected by an opened campaign."""
    family_ids = {
        family: "development-" + f"{index + 1:064x}"
        for index, family in enumerate(SYMILE_CORE_DEVELOPMENT_FAMILIES)
    }
    ecg_id = "development-" + "e" * 64
    analysis_id = "analysis-" + "a" * 64
    cv_id = "cv-assignment-" + "c" * 64
    developments = {}
    for index, (family, development_id) in enumerate(
        [*family_ids.items(), (SYMILE_ECG_GATED_FAMILY, ecg_id)]
    ):
        fold_id = "fold-package-" + f"{index:064x}"
        prediction_id = "prediction-" + f"{index:064x}"
        directory = report_root / "development" / "families" / development_id
        directory.mkdir(parents=True)
        (directory / "authority.json").write_text("{}\n", encoding="utf-8")
        for path in (
            model_root / "development" / "packages" / fold_id,
            private_root / "predictions" / "symile" / "oof" / prediction_id,
        ):
            path.mkdir(parents=True)
            (path / "authority.json").write_text("{}\n", encoding="utf-8")
        developments[development_id] = ValidatedDevelopmentResult(
            directory,
            {
                "development_id": development_id,
                "family_id": family,
                "scientific_context": {
                    "bundle_id": freeze.manifest["bundle"]["bundle_id"],
                    "cv_assignment_id": cv_id,
                },
                "fold_packages": [{"fold_package_id": fold_id, "prediction_id": prediction_id}],
            },
            "1" * 64,
        )
    analysis_directory = report_root / "development" / "analyses" / analysis_id
    analysis_directory.mkdir(parents=True)
    (analysis_directory / "authority.json").write_text("{}\n", encoding="utf-8")
    subgroup_id = "focused-subgroup-" + "3" * 64
    subgroup = report_root / "development-subgroups" / f"{subgroup_id}.json"
    subgroup.parent.mkdir(parents=True)
    subgroup.write_text("{}\n", encoding="utf-8")
    extension_directory = (
        report_root / "ecg-extension-results" / freeze.manifest["ecg_extension_result_id"]
    )
    extension_directory.mkdir(parents=True)
    (extension_directory / "authority.json").write_text("{}\n", encoding="utf-8")
    extension = extension_result.ValidatedEcgExtensionResult(
        extension_directory,
        {
            "ecg_extension_result_id": freeze.manifest["ecg_extension_result_id"],
            "core_analysis_id": analysis_id,
            "ecg_development_id": ecg_id,
            "focused_subgroup_derivative": subgroup_id,
        },
        "4" * 64,
        "5" * 64,
        extension_result._VALIDATION_GUARD,
    )
    return SimpleNamespace(
        cv_id=cv_id,
        analysis={"analysis_id": analysis_id, "family_development_ids": family_ids},
        developments=developments,
        extension=extension,
    )


def _synthetic_opened_heldout_graph(
    tmp_path: Path,
    *,
    manifest_root: Path,
    model_root: Path,
    report_root: Path,
    private_root: Path,
    freeze: ValidatedPretestFreeze,
    cv_id: str,
) -> SimpleNamespace:
    """Materialize the held-out authorities selected by an opened campaign."""
    packages = []
    for package in _synthetic_final_packages(tmp_path / "unused-packages", freeze):
        directory = model_root / "final" / "packages" / package.package_id
        directory.mkdir(parents=True)
        (directory / "authority.json").write_text("{}\n", encoding="utf-8")
        packages.append(type(package)(directory, package.manifest, package.manifest_sha256))
    predictions = []
    for index, package in enumerate(packages):
        prediction_id = "prediction-" + f"{index + 100:064x}"
        directory = private_root / "predictions" / "symile" / "test" / prediction_id
        directory.mkdir(parents=True)
        (directory / "authority.json").write_text("{}\n", encoding="utf-8")
        predictions.append(
            ValidatedPredictionEvidence(
                directory,
                {"prediction_id": prediction_id, "model_package_id": package.package_id},
                "6" * 64,
                SimpleNamespace(),
            )
        )
    global_id = "global-result-" + "7" * 64
    global_directory = report_root / "global-results" / global_id
    global_directory.mkdir(parents=True)
    (global_directory / "authority.json").write_text("{}\n", encoding="utf-8")
    error_review = private_root / "error-review" / "symile" / ("error-review-" + "8" * 64 + ".json")
    error_review.parent.mkdir(parents=True)
    error_review.write_text("{}\n", encoding="utf-8")
    for path in (
        manifest_root / "symile" / "bundles" / freeze.manifest["bundle"]["bundle_id"],
        manifest_root / "symile" / "cv" / cv_id,
    ):
        path.mkdir(parents=True)
        (path / "authority.json").write_text("{}\n", encoding="utf-8")
    return SimpleNamespace(
        packages=tuple(packages),
        predictions=tuple(predictions),
        global_result=ValidatedGlobalResult(global_directory, {"global_result_id": global_id}),
        error_review=error_review,
    )


def _published_preserved_campaign_foundation(
    tmp_path: Path,
    manifest_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Publish the bundle, CV, and pinned configs used by preserved-campaign evidence."""
    _, bundle = _published_synthetic_release(tmp_path / "release")
    source_manifest_root = bundle.bundle_directory.parent.parent.parent
    shutil.copytree(source_manifest_root / "symile", manifest_root / "symile")
    bundle = type(bundle)(
        bundle.bundle_id,
        manifest_root / "symile/bundles" / bundle.bundle_id,
        manifest_root / "symile/bundles" / bundle.bundle_id / "samples.parquet",
        manifest_root / "symile/bundles" / bundle.bundle_id / "labs.parquet",
        manifest_root / "symile/bundles" / bundle.bundle_id / "manifest.json",
        manifest_root / "symile/CURRENT",
    )
    cv_id, cv_directory = publish_symile_cv(bundle, manifest_directory=manifest_root)
    bundle_reference = validate_symile_bundle_reference(bundle.bundle_directory)
    bundle_manifest = bundle_reference.manifest
    cv_hash = hashlib.sha256((cv_directory / "manifest.json").read_bytes()).hexdigest()
    config_directory = tmp_path / "configs"
    config_directory.mkdir()
    configs = {}
    for family in (*SYMILE_CORE_DEVELOPMENT_FAMILIES, SYMILE_ECG_GATED_FAMILY):
        source = Path("configs") / f"symile_{family}.yaml"
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
        document["dataset"].update(
            {
                "bundle_id": bundle.bundle_id,
                "bundle_manifest_sha256": bundle_reference.manifest_sha256,
                "split_assignment_id": bundle_manifest["membership"]["split_assignment_id"],
                "cv_assignment_id": cv_id,
            }
        )
        destination = config_directory / source.name
        destination.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        configs[family] = load_symile_development_config(destination)
    strict_counts = bundle_manifest["qualification"]["strict_pneumonia_counts"]
    monkeypatch.setattr(
        symile_publication, "DEVELOPMENT_COUNT", strict_counts["development"]["eligible"]
    )
    monkeypatch.setattr(symile_data, "DEVELOPMENT_COUNT", strict_counts["development"]["eligible"])
    monkeypatch.setattr(
        symile_data, "DEVELOPMENT_POSITIVES", strict_counts["development"]["positive"]
    )
    monkeypatch.setattr(
        symile_data, "DEVELOPMENT_NEGATIVES", strict_counts["development"]["negative"]
    )
    monkeypatch.setattr(
        symile_test_data, "STRICT_PNEUMONIA_TEST_ADMISSIONS", strict_counts["test"]["eligible"]
    )
    config = configs["labs_logistic"]
    authority = symile_publication._load_cv_authority(
        {
            "bundle_id": config.dataset.bundle_id,
            "bundle_manifest_sha256": config.dataset.bundle_manifest_sha256,
            "cv_assignment_id": config.dataset.cv_assignment_id,
            "cv_manifest_sha256": cv_hash,
        },
        manifest_root,
    )
    return SimpleNamespace(
        configs=configs,
        config=config,
        authority=authority,
        cv_hash=cv_hash,
    )


class _TinyFinalNeural(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1, bias=False)

    def forward(self, image: torch.Tensor, structured: torch.Tensor) -> torch.Tensor:
        del image
        return self.encoder(structured[:, :1]).squeeze(1)


def _publish_preserved_family_developments(
    tmp_path: Path,
    *,
    foundation: SimpleNamespace,
    manifest_root: Path,
    model_root: Path,
    report_root: Path,
    private_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, ValidatedDevelopmentResult]:
    """Publish every core and ECG family development authority."""
    monkeypatch.setattr(symile_publication, "_validate_neural_reconstruction", lambda *args: None)
    monkeypatch.setattr(final_packages, "_reconstruct_neural", lambda config: _TinyFinalNeural())
    developments = {}
    folds_by_family = {}
    offsets = {
        "labs_logistic": 0.00,
        "labs_lightgbm": 0.02,
        "cxr_densenet": 0.04,
        "cxr_labs_concat": 0.06,
        "cxr_labs_gated": 0.08,
        "cxr_labs_gated_no_observedness": 0.10,
        "cxr_labs_ecg_gated": 0.12,
    }
    for family in (*SYMILE_CORE_DEVELOPMENT_FAMILIES, SYMILE_ECG_GATED_FAMILY):
        fusion = family in {
            "cxr_labs_concat",
            "cxr_labs_gated",
            "cxr_labs_gated_no_observedness",
            "cxr_labs_ecg_gated",
        }
        folds, semantic_hash = _publish_family_folds(
            tmp_path,
            family,
            offset=offsets[family],
            source_cxr_folds=folds_by_family.get("cxr_densenet") if fusion else None,
            source_cxr_development_id=(
                developments["cxr_densenet"].manifest["development_id"] if fusion else None
            ),
            authority=foundation.authority,
            cv_manifest_sha256=foundation.cv_hash,
            config=foundation.configs[family],
        )
        folds_by_family[family] = folds
        developments[family] = publish_development_result(
            report_root=report_root / "development",
            model_root=model_root / "development",
            prediction_root=private_root,
            manifest_root=manifest_root,
            family=family,
            config_semantic_sha256=semantic_hash,
            folds=[item.package for item in folds],
            predictions=[item.prediction for item in folds],
        )
    return developments


def _publish_preserved_analysis_and_extension(
    *,
    developments: dict[str, ValidatedDevelopmentResult],
    configs: dict[str, object],
    manifest_root: Path,
    model_root: Path,
    report_root: Path,
    private_root: Path,
) -> SimpleNamespace:
    """Publish the core analysis, focused subgroups, and ECG extension."""
    family_ids = {
        family: developments[family].manifest["development_id"]
        for family in SYMILE_CORE_DEVELOPMENT_FAMILIES
    }
    analysis_id, analysis_directory = publish_analysis_result(
        report_root=report_root / "development",
        model_root=model_root / "development",
        prediction_root=private_root,
        manifest_root=manifest_root,
        family_development_ids=family_ids,
    )
    primary = symile_publication.validated_development_repeat_oof(
        developments["cxr_labs_gated"], private_root
    )
    cxr = symile_publication.validated_development_repeat_oof(
        developments["cxr_densenet"], private_root
    )
    primary_config = with_runtime(configs["cxr_labs_gated"], manifest_directory=manifest_root)
    cohort = symile_data.load_symile_development_cohort(primary_config).frame
    attributes = cohort.loc[:, ["sample_id", "age_years", "sex", "view_position"]].copy()
    attributes["observed_lab_count"] = cohort.loc[:, LAB_OBSERVED_COLUMNS].sum(axis=1)
    subgroup_id = extension_result.publish_focused_subgroup_derivative(
        report_root=report_root,
        summary=focused_development_subgroups(cxr, primary, attributes),
    )
    extension = extension_result.publish_ecg_extension_result(
        report_root=report_root,
        core_analysis_directory=analysis_directory,
        ecg_development=developments[SYMILE_ECG_GATED_FAMILY],
        focused_subgroup_derivative=subgroup_id,
        model_root=model_root / "development",
        prediction_root=private_root,
        manifest_root=manifest_root,
    )
    return SimpleNamespace(
        analysis_id=analysis_id,
        extension=extension,
    )


def _publish_preserved_campaign_development(
    tmp_path: Path,
    *,
    foundation: SimpleNamespace,
    manifest_root: Path,
    model_root: Path,
    report_root: Path,
    private_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Publish complete core and ECG development evidence with real validators."""
    developments = _publish_preserved_family_developments(
        tmp_path,
        foundation=foundation,
        manifest_root=manifest_root,
        model_root=model_root,
        report_root=report_root,
        private_root=private_root,
        monkeypatch=monkeypatch,
    )
    analysis = _publish_preserved_analysis_and_extension(
        developments=developments,
        configs=foundation.configs,
        manifest_root=manifest_root,
        model_root=model_root,
        report_root=report_root,
        private_root=private_root,
    )
    return SimpleNamespace(
        developments=developments,
        analysis_id=analysis.analysis_id,
        extension=analysis.extension,
    )


def _publish_preserved_tabular_packages(
    *,
    model_root: Path,
    configs: dict[str, object],
    developments: dict[str, ValidatedDevelopmentResult],
    extension: extension_result.ValidatedEcgExtensionResult,
    families: tuple[str, ...],
) -> tuple[ValidatedFinalPackage, ...]:
    """Publish the terminal tabular packages used by the preserved campaign."""
    feature_frame = _development_frame(20)[list(LAB_FEATURE_COLUMNS)]
    targets = np.asarray([index % 2 for index in range(20)], dtype=np.int8)
    operational = {
        "git_commit": "f" * 40,
        "git_dirty": False,
        "dependency_lock_sha256": "1" * 64,
        "mlflow_run_id": "synthetic-final-run",
    }
    packages = []
    for family in families:
        family_config = configs[family]
        budget = extension.manifest["final_family_authorities"][family]["final_training_budget"]
        if family == "labs_logistic":
            fitted = fit_symile_labs_logistic(
                feature_frame,
                targets,
                parameters=family_config.training.parameters,
                selection_metric=family_config.training.selection_metric,
                lab_policy=str(family_config.preprocessing["lab_policy"]),
                training_seed=42,
            ).pipeline
        else:
            fitted = fit_final_symile_labs_lightgbm(
                feature_frame,
                targets,
                parameters={
                    **family_config.family.parameters,
                    **family_config.training.parameters,
                },
                n_estimators=budget,
                lab_policy=str(family_config.preprocessing["lab_policy"]),
                training_seed=42,
            )
        packages.append(
            publish_final_tabular_package(
                model_root=model_root,
                config=family_config,
                family_development_id=(
                    None
                    if family == "labs_logistic"
                    else developments[family].manifest["development_id"]
                ),
                plan=final_training_plan(family, budget),
                model=fitted,
                operational=operational,
            )
        )
    return tuple(packages)


def _publish_preserved_neural_packages(
    *,
    model_root: Path,
    configs: dict[str, object],
    developments: dict[str, ValidatedDevelopmentResult],
    extension: extension_result.ValidatedEcgExtensionResult,
    families: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[ValidatedFinalPackage, ...]:
    """Publish the terminal neural packages used by the preserved campaign."""
    feature_frame = _development_frame(20)[list(LAB_FEATURE_COLUMNS)]
    operational = {
        "git_commit": "f" * 40,
        "git_dirty": False,
        "dependency_lock_sha256": "1" * 64,
        "mlflow_run_id": "synthetic-final-run",
    }
    preprocessor = symile_publication.SymileLabEcdfTransformer().fit(feature_frame)
    packages = []
    cxr_by_seed = {}
    pretrained = {
        "declared_name": "densenet121-res224-chex",
        "stable_identifier": "https://example.test/weights.pt",
        "cache_filename": "weights.pt",
        "byte_size": 1,
        "sha256": "2" * 64,
    }
    neural_operational = {
        **operational,
        "runtime_provenance": {"pin_memory_effective": False},
        "loader_execution": {
            "lifecycle": "reused",
            "num_workers": 0,
            "pin_memory": False,
            "batch_size": 2,
            "drop_last": False,
            "shuffle": False,
            "sampler": "epoch_permutation",
            "persistent_workers": False,
            "prefetch_factor": None,
            "multiprocessing_context": None,
        },
    }
    for family in families:
        family_config = configs[family]
        budget = extension.manifest["final_family_authorities"][family]["final_training_budget"]
        for seed in seeds:
            package = publish_final_neural_package(
                model_root=model_root,
                config=family_config,
                family_development_id=developments[family].manifest["development_id"],
                plan=final_training_plan(family, budget),
                seed=seed,
                state_dict={"encoder.weight": torch.ones((1, 1))},
                lab_preprocessor=None if family == "cxr_densenet" else preprocessor,
                source_cxr_package_id=(None if family == "cxr_densenet" else cxr_by_seed[seed]),
                pretrained_weight=pretrained if family == "cxr_densenet" else None,
                operational=neural_operational,
            )
            packages.append(package)
            if family == "cxr_densenet":
                cxr_by_seed[seed] = package.package_id
    return tuple(packages)


def _publish_preserved_campaign_final_packages(
    *,
    model_root: Path,
    configs: dict[str, object],
    developments: dict[str, ValidatedDevelopmentResult],
    extension: extension_result.ValidatedEcgExtensionResult,
    tabular_families: tuple[str, ...],
    neural_families: tuple[str, ...],
    neural_seeds: tuple[int, ...],
) -> tuple[ValidatedFinalPackage, ...]:
    """Publish the exact terminal package membership used by the preserved campaign."""
    return (
        *_publish_preserved_tabular_packages(
            model_root=model_root,
            configs=configs,
            developments=developments,
            extension=extension,
            families=tabular_families,
        ),
        *_publish_preserved_neural_packages(
            model_root=model_root,
            configs=configs,
            developments=developments,
            extension=extension,
            families=neural_families,
            seeds=neural_seeds,
        ),
    )


def _repeat_oof(logits: list[float]) -> pd.DataFrame:
    rows = []
    for seed in FINAL_NEURAL_MEMBER_SEEDS:
        for sample_id, target, logit in zip(
            ("a", "b", "c", "d"), (0, 0, 1, 1), logits, strict=True
        ):
            rows.append(
                {
                    "sample_id": sample_id,
                    "target": target,
                    "logit": logit + seed / 100_000,
                    "repeat_seed": seed,
                }
            )
    return pd.DataFrame(rows).sort_values(["sample_id", "repeat_seed"]).reset_index(drop=True)
