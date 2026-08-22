from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from neural_test_support import cpu_runtime
from symile_campaign_test_support import (
    _freeze,
    _projection_for_freeze,
    _synthetic_final_packages,
)

import radfusion.training.symile_campaign as symile_campaign
import radfusion.training.symile_campaign_control as campaign_control
import radfusion.training.symile_data as symile_data
import radfusion.training.symile_ecg_extension_result as extension_result
import radfusion.training.symile_final_packages as final_packages
import radfusion.training.symile_test_data as symile_test_data
import radfusion.training.symile_test_inference as test_inference
import radfusion.utils.symile_publication as symile_publication
from radfusion.data.errors import ManifestBuildError
from radfusion.data.symile_artifacts import validate_symile_bundle_reference
from radfusion.data.symile_cv import publish_symile_cv
from radfusion.data.symile_preprocess import LAB_FEATURE_COLUMNS, LAB_OBSERVED_COLUMNS
from radfusion.models.symile_tabular import (
    fit_final_symile_labs_lightgbm,
    fit_symile_labs_logistic,
)
from radfusion.training.config import load_symile_development_config, with_runtime
from radfusion.training.symile_campaign_control import (
    ValidatedGlobalResult,
    create_or_validate_test_open_record,
)
from radfusion.training.symile_data import load_symile_development_cohort
from radfusion.training.symile_export import export_and_verify, restore_and_validate_symile_export
from radfusion.training.symile_final_packages import (
    final_training_plan,
    publish_final_neural_package,
    publish_final_tabular_package,
)
from radfusion.training.symile_statistics import focused_development_subgroups
from radfusion.training.symile_test_data import FrozenSymileTestData
from radfusion.utils.private_predictions import (
    SYMILE_TEST_INFERENCE_POLICY,
    ValidatedPredictionEvidence,
    publish_prediction_evidence,
)
from radfusion.utils.symile_publication import (
    ValidatedDevelopmentResult,
    publish_analysis_result,
    publish_development_result,
)


def _forbid_campaign_training(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("opened campaign attempted development or final fitting")

    for name in (
        "run_symile_development",
        "analyze_symile_development",
        "fit_final_tabular_family",
        "fit_final_neural_member",
        "_tracked_final_fit",
    ):
        monkeypatch.setattr(symile_campaign, name, forbidden)


def test_campaign_cli_requires_and_forwards_backup_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, object]] = []

    def campaign(**kwargs: object) -> dict[str, str]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(symile_campaign, "run_symile_campaign", campaign)
    with pytest.raises(SystemExit) as missing:
        symile_campaign.main(["--source-root", "source"])
    assert missing.value.code == 2
    assert "--backup-root" in capsys.readouterr().err
    assert not calls
    backup = tmp_path / "persistent backup"
    assert symile_campaign.main(["--source-root", "source", "--backup-root", str(backup)]) == 0
    assert len(calls) == 1
    assert calls[0] == {
        "source_root": Path("source"),
        "backup_root": backup,
        "device": "auto",
        "workers": 2,
    }
    with pytest.raises(SystemExit) as unsupported:
        symile_campaign.main(
            [
                "--source-root",
                "source",
                "--backup-root",
                str(backup),
                "--report-root",
                "elsewhere",
            ]
        )
    assert unsupported.value.code == 2


@pytest.mark.parametrize(
    "backup", ["backup", "models/backup", "reports/anything", ".", "x/../backup"]
)
def test_campaign_rejects_unsafe_backup_before_provenance_or_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backup: str
) -> None:
    monkeypatch.setattr(symile_campaign, "discover_repository_root", lambda: tmp_path)

    def unexpected(*args, **kwargs):
        raise AssertionError("Unsafe paths must fail before campaign execution")

    monkeypatch.setattr(symile_campaign, "git_revision", unexpected)
    with pytest.raises(ManifestBuildError, match="outside the repository"):
        symile_campaign.run_symile_campaign(source_root=tmp_path / "source", backup_root=backup)
    assert not tuple(tmp_path.iterdir())


def test_campaign_rejects_backup_symlink_into_repository(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(repository, target_is_directory=True)
    monkeypatch.setattr(symile_campaign, "discover_repository_root", lambda: repository)
    with pytest.raises(ManifestBuildError, match="outside the repository"):
        symile_campaign.run_symile_campaign(
            source_root=tmp_path / "source", backup_root=alias / "backup"
        )


def test_campaign_rejects_backup_parent_that_contains_repository(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(symile_campaign, "discover_repository_root", lambda: repository)
    with pytest.raises(ManifestBuildError, match="contain campaign source state"):
        symile_campaign.run_symile_campaign(source_root=tmp_path / "source", backup_root=tmp_path)


def test_canonical_formal_layout_accepts_normal_checkout_roots(tmp_path: Path) -> None:
    for path in (
        tmp_path / "data/manifests",
        tmp_path / "models/symile",
        tmp_path / "reports/symile",
        tmp_path / "private",
        tmp_path / "outbox",
        tmp_path / "mlartifacts",
    ):
        path.mkdir(parents=True, exist_ok=True)
    for name in ("mlflow.db", "mlflow.db-wal", "mlflow.db-shm"):
        (tmp_path / name).write_bytes(b"")
    symile_campaign._validate_canonical_formal_layout(tmp_path)


@pytest.mark.parametrize(
    "redirect",
    [
        "private-outside",
        "private-to-reports",
        "models-symile",
        "models-final",
        "private-control",
        "outbox",
        "mlartifacts",
        "mlflow.db",
    ],
)
def test_canonical_formal_layout_rejects_symlink_redirects(tmp_path: Path, redirect: str) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = repository / redirect
    link_target = outside
    if redirect == "private-outside":
        target = repository / "private"
    elif redirect == "private-to-reports":
        target = repository / "private"
        link_target = repository / "reports/symile"
        link_target.mkdir(parents=True)
    elif redirect == "models-symile":
        target = repository / "models/symile"
        target.parent.mkdir()
    elif redirect == "models-final":
        target = repository / "models/symile/final"
        target.parent.mkdir(parents=True)
    elif redirect == "private-control":
        target = repository / "private/control"
        target.parent.mkdir()
    elif redirect == "mlflow.db":
        link_target = outside / "database"
        link_target.write_bytes(b"")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(link_target, target_is_directory=redirect != "mlflow.db")
    with pytest.raises(ManifestBuildError, match="symlink redirect"):
        symile_campaign._validate_canonical_formal_layout(repository)


@pytest.mark.parametrize("redirect", ["models/symile/final", "private/control"])
def test_nested_canonical_layout_failure_precedes_provenance_and_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, redirect: str
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = repository / redirect
    target.parent.mkdir(parents=True)
    target.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(symile_campaign, "discover_repository_root", lambda: repository)

    def unexpected(*args, **kwargs):
        raise AssertionError("canonical preflight must run first")

    monkeypatch.setattr(symile_campaign, "git_revision", unexpected)
    with pytest.raises(ManifestBuildError, match="symlink redirect"):
        symile_campaign.run_symile_campaign(
            source_root=tmp_path / "source", backup_root=tmp_path / "backup"
        )


def test_campaign_resumes_exact_opened_freeze_without_any_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "private"
    control_root = private_root / "control" / "symile"
    freeze = _freeze(control_root / "freezes")
    record = create_or_validate_test_open_record(control_root=control_root, capability=freeze)
    packages = _synthetic_final_packages(tmp_path / "packages", freeze)
    package_by_id = {package.package_id: package for package in packages}
    extension = extension_result.ValidatedEcgExtensionResult(
        tmp_path / "extension",
        {
            "ecg_extension_result_id": freeze.manifest["ecg_extension_result_id"],
            "core_analysis_id": "analysis-" + "1" * 64,
            "ecg_development_id": "development-" + "2" * 64,
            "focused_subgroup_derivative": "focused-subgroup-" + "3" * 64,
        },
        "4" * 64,
        "5" * 64,
        extension_result._VALIDATION_GUARD,
    )
    config = load_symile_development_config("configs/symile_labs_logistic.yaml")
    monkeypatch.setattr(symile_campaign, "discover_repository_root", lambda: tmp_path)
    monkeypatch.setattr(symile_campaign, "git_revision", lambda root: ("f" * 40, False))
    monkeypatch.setattr(symile_campaign, "uv_lock_sha256", lambda path: "0" * 64)
    neural_config = load_symile_development_config("configs/symile_cxr_densenet.yaml")
    monkeypatch.setattr(
        symile_campaign,
        "load_symile_development_config",
        lambda path: neural_config if Path(path).stem == "symile_cxr_densenet" else config,
    )
    monkeypatch.setattr(
        symile_campaign,
        "validate_ecg_extension_result",
        lambda *args, **kwargs: extension,
    )
    monkeypatch.setattr(
        symile_campaign,
        "validate_final_package",
        lambda path, **kwargs: package_by_id[Path(path).name],
    )
    monkeypatch.setattr(
        symile_campaign,
        "validate_pretest_freeze",
        lambda *args, **kwargs: freeze,
    )
    runtime = cpu_runtime()
    monkeypatch.setattr(
        symile_campaign, "resolve_held_out_neural_runtime", lambda packages, device: runtime
    )

    _forbid_campaign_training(monkeypatch)

    def complete(**kwargs: object) -> dict[str, str]:
        assert kwargs["freeze"] is freeze
        assert kwargs["record"] == record
        assert kwargs["ecg_extension"] is extension
        assert tuple(kwargs["packages"]) == packages
        assert kwargs["runtime"] is runtime
        return {"pretest_freeze_id": freeze.freeze_id}

    monkeypatch.setattr(symile_campaign, "_complete_opened_campaign", complete)
    arguments = {
        "source_root": tmp_path / "source",
        "backup_root": tmp_path.parent / f"{tmp_path.name}-backup",
    }
    assert symile_campaign.run_symile_campaign(**arguments) == {
        "pretest_freeze_id": freeze.freeze_id
    }

    (control_root / "test-open.json").write_bytes(b'{"test_open_schema_version":')
    with pytest.raises(ManifestBuildError, match="malformed"):
        symile_campaign.run_symile_campaign(**arguments)


def test_campaign_exact_export_and_interrupted_opened_resume_rehearsal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_root = tmp_path / "data" / "manifests"
    model_root = tmp_path / "models" / "symile"
    report_root = tmp_path / "reports" / "symile"
    private_root = tmp_path / "private"
    freeze = _freeze(private_root / "control" / "symile" / "freezes")
    record = create_or_validate_test_open_record(
        control_root=private_root / "control" / "symile", capability=freeze
    )
    family_ids = {
        family: "development-" + f"{index + 1:064x}"
        for index, family in enumerate(symile_campaign.SYMILE_CORE_DEVELOPMENT_FAMILIES)
    }
    ecg_id = "development-" + "e" * 64
    analysis_id = "analysis-" + "a" * 64
    cv_id = "cv-assignment-" + "c" * 64
    developments = {}
    for index, (family, development_id) in enumerate(
        [*family_ids.items(), (symile_campaign.SYMILE_ECG_GATED_FAMILY, ecg_id)]
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
    packages = _synthetic_final_packages(tmp_path / "unused-packages", freeze)
    canonical_packages = []
    for package in packages:
        directory = model_root / "final" / "packages" / package.package_id
        directory.mkdir(parents=True)
        (directory / "authority.json").write_text("{}\n", encoding="utf-8")
        canonical_packages.append(
            type(package)(directory, package.manifest, package.manifest_sha256)
        )
    packages = tuple(canonical_packages)
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
    predictions = tuple(predictions)
    global_id = "global-result-" + "7" * 64
    global_directory = report_root / "global-results" / global_id
    global_directory.mkdir(parents=True)
    (global_directory / "authority.json").write_text("{}\n", encoding="utf-8")
    global_result = ValidatedGlobalResult(global_directory, {"global_result_id": global_id})
    error_review = private_root / "error-review" / "symile" / ("error-review-" + "8" * 64 + ".json")
    error_review.parent.mkdir(parents=True)
    error_review.write_text("{}\n", encoding="utf-8")
    for path in (
        manifest_root / "symile" / "bundles" / freeze.manifest["bundle"]["bundle_id"],
        manifest_root / "symile" / "cv" / cv_id,
    ):
        path.mkdir(parents=True)
        (path / "authority.json").write_text("{}\n", encoding="utf-8")

    analysis = {"analysis_id": analysis_id, "family_development_ids": family_ids}
    package_by_id = {package.package_id: package for package in packages}
    prediction_by_id = {prediction.prediction_id: prediction for prediction in predictions}
    monkeypatch.setattr(symile_campaign, "validate_analysis_result", lambda *a, **k: analysis)
    monkeypatch.setattr(
        symile_campaign,
        "validate_development_result",
        lambda path, **kwargs: developments[Path(path).name],
    )
    monkeypatch.setattr(symile_campaign, "validate_ecg_extension_result", lambda *a, **k: extension)
    monkeypatch.setattr(
        symile_campaign,
        "validate_final_package",
        lambda path, **kwargs: package_by_id[Path(path).name],
    )
    monkeypatch.setattr(symile_campaign, "validate_pretest_freeze", lambda *a, **k: freeze)
    monkeypatch.setattr(symile_campaign, "validate_global_result", lambda *a, **k: global_result)
    monkeypatch.setattr(symile_campaign, "validate_error_review", lambda *a, **k: None)
    monkeypatch.setattr(
        symile_campaign,
        "validate_prediction_evidence",
        lambda path, **kwargs: prediction_by_id[Path(path).name],
    )
    members = symile_campaign._campaign_export_members(
        freeze=freeze,
        record=record,
        ecg_extension=extension,
        packages=packages,
        predictions=predictions,
        global_result=global_result,
        error_review=error_review,
        test_data=SimpleNamespace(),
        manifest_root=manifest_root,
        model_root=model_root,
        report_root=report_root,
        private_root=private_root,
    )
    selected = {member.path.resolve() for member in members}
    unrelated = report_root / "old-global-result"
    unrelated.mkdir()
    (unrelated / "stale.tmp").write_text("old\n", encoding="utf-8")
    abandoned = private_root / "predictions" / "symile" / "test" / ".result.staging-dead"
    abandoned.mkdir()
    (abandoned / "partial.tmp").write_text("partial\n", encoding="utf-8")
    assert unrelated.resolve() not in selected
    assert abandoned.resolve() not in selected
    arguments = {
        "members": members,
        "export_root": tmp_path / "outbox",
        "backup_root": tmp_path / "backup",
        "export_name": global_id,
    }
    archive = export_and_verify(**arguments)
    before = archive.read_bytes()
    (unrelated / "stale.tmp").write_text("changed\n", encoding="utf-8")
    (abandoned / "partial.tmp").write_text("changed\n", encoding="utf-8")
    assert export_and_verify(**arguments).read_bytes() == before
    for prediction in predictions:
        shutil.rmtree(prediction.directory)
    shutil.rmtree(abandoned)
    error_review.unlink()

    package_by_path = {package.directory: package for package in packages}
    monkeypatch.setattr(
        test_inference,
        "validate_final_package",
        lambda path, **kwargs: package_by_path[Path(path)],
    )
    monkeypatch.setattr(
        campaign_control,
        "validate_final_package",
        lambda path, **kwargs: package_by_id[Path(path).name],
    )
    all_families = tuple(package.manifest["input"]["family"]["family_id"] for package in packages)
    monkeypatch.setattr(test_inference, "FINAL_TABULAR_FAMILIES", all_families)
    family_by_id = {
        package.package_id: package.manifest["input"]["family"]["family_id"] for package in packages
    }

    def package_config(package):
        return SimpleNamespace(
            family_id=family_by_id[package.package_id], modalities=(), mixed_precision=True
        )

    monkeypatch.setattr(test_inference, "load_final_package_config", package_config)
    monkeypatch.setattr(symile_campaign, "load_final_package_config", package_config)
    monkeypatch.setattr(test_inference, "_validate_neural_inference_runtime", lambda *a: None)
    monkeypatch.setattr(test_inference, "load_final_tabular_model", lambda package: package)
    monkeypatch.setattr(test_inference, "test_laboratory_frame", lambda data: data.frame)
    monkeypatch.setattr(
        test_inference,
        "symile_tabular_logits",
        lambda model, frame: np.linspace(-2.0, 2.0, len(frame)),
    )

    def frozen_test_data(capability, record, config, *, manifest_root):
        del record, config, manifest_root
        projection = _projection_for_freeze(capability, grouped=False)
        data = object.__new__(FrozenSymileTestData)
        data.bundle = SimpleNamespace(bundle_id=projection.bundle_id)
        data.frame = projection.frame()
        data.frame["sample_id"] = [f"symile:{index}" for index in range(len(data.frame))]
        data.split_assignment_id = projection.split_assignment_id
        data._capability = capability
        return data

    monkeypatch.setattr(test_inference, "FrozenSymileTestData", frozen_test_data)
    monkeypatch.setattr(symile_campaign, "FrozenSymileTestData", frozen_test_data)
    real_publish = test_inference._publish
    published_before_interruption: list[str] = []

    def publish_then_interrupt(package_id, data, logits, private_root, freeze_id):
        evidence = real_publish(package_id, data, logits, private_root, freeze_id)
        published_before_interruption.append(evidence.prediction_id)
        raise RuntimeError("synthetic interruption after one durable prediction")

    monkeypatch.setattr(test_inference, "_publish", publish_then_interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        test_inference.publish_frozen_test_predictions(
            capability=freeze,
            test_open_record=record,
            final_packages=packages,
            source_root=tmp_path / "source",
            manifest_root=manifest_root,
            private_root=private_root,
            runtime=cpu_runtime(),
        )
    assert len(published_before_interruption) == 1
    monkeypatch.setattr(test_inference, "_publish", real_publish)
    monkeypatch.setattr(
        campaign_control,
        "validate_prediction_evidence",
        lambda path, **kwargs: test_inference.validate_prediction_evidence(path, **kwargs),
    )
    monkeypatch.setattr(
        campaign_control,
        "cluster_bootstrap_effect",
        lambda frame, **kwargs: {"subject_count": int(frame["subject_id"].nunique())},
    )
    monkeypatch.setattr(
        symile_campaign,
        "publish_frozen_test_predictions",
        test_inference.publish_frozen_test_predictions,
    )
    monkeypatch.setattr(
        symile_campaign, "publish_global_result", campaign_control.publish_global_result
    )
    monkeypatch.setattr(
        symile_campaign, "publish_error_review", campaign_control.publish_error_review
    )
    monkeypatch.setattr(
        symile_campaign, "validate_global_result", campaign_control.validate_global_result
    )
    monkeypatch.setattr(
        symile_campaign, "validate_error_review", campaign_control.validate_error_review
    )
    monkeypatch.setattr(
        symile_campaign,
        "validate_prediction_evidence",
        test_inference.validate_prediction_evidence,
    )
    monkeypatch.setattr(symile_campaign, "validate_focused_subgroup_derivative", lambda *a, **k: {})
    monkeypatch.setattr(symile_campaign, "discover_repository_root", lambda: tmp_path)
    monkeypatch.setattr(symile_campaign, "git_revision", lambda root: ("f" * 40, False))
    monkeypatch.setattr(symile_campaign, "uv_lock_sha256", lambda path: "0" * 64)
    base_config = load_symile_development_config("configs/symile_labs_logistic.yaml")
    monkeypatch.setattr(symile_campaign, "load_symile_development_config", lambda path: base_config)
    monkeypatch.setattr(
        symile_campaign,
        "resolve_held_out_neural_runtime",
        lambda packages, device: cpu_runtime(),
    )

    _forbid_campaign_training(monkeypatch)
    final_backup = tmp_path.parent / f"{tmp_path.name}-rehearsal-backup"
    result = symile_campaign.run_symile_campaign(
        source_root=tmp_path / "source",
        backup_root=final_backup,
        device="cpu",
        workers=0,
    )
    assert result["pretest_freeze_id"] == freeze.freeze_id
    assert result["global_result_id"].startswith("global-result-")
    assert Path(result["export"]).is_file()
    assert (final_backup / Path(result["export"]).name).is_file()
    claims = json.loads(
        (report_root / "global-results" / result["global_result_id"] / "claims.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(claims["predictor_views"]) == 6
    assert len(tuple((private_root / "error-review" / "symile").glob("error-review-*.json"))) == 1
    assert (
        len(
            [
                path
                for path in (private_root / "predictions" / "symile" / "test").iterdir()
                if not path.name.startswith(".")
            ]
        )
        == 14
    )


def test_real_campaign_closure_restores_after_original_authorities_are_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Certify the archive with real recursive validators and bounded synthetic fitted state."""
    from symile_data_test_support import _published_synthetic_release
    from symile_development_test_support import _development_frame, _publish_family_folds

    manifest_root = tmp_path / "data/manifests"
    model_root = tmp_path / "models/symile"
    report_root = tmp_path / "reports/symile"
    private_root = tmp_path / "private"
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
    for family in (
        *symile_campaign.SYMILE_CORE_DEVELOPMENT_FAMILIES,
        "cxr_labs_ecg_gated",
    ):
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
    config = configs["labs_logistic"]
    strict_counts = bundle_manifest["qualification"]["strict_pneumonia_counts"]
    development_count = strict_counts["development"]["eligible"]
    test_count = strict_counts["test"]["eligible"]
    monkeypatch.setattr(symile_publication, "DEVELOPMENT_COUNT", development_count)
    monkeypatch.setattr(symile_data, "DEVELOPMENT_COUNT", development_count)
    monkeypatch.setattr(
        symile_data, "DEVELOPMENT_POSITIVES", strict_counts["development"]["positive"]
    )
    monkeypatch.setattr(
        symile_data, "DEVELOPMENT_NEGATIVES", strict_counts["development"]["negative"]
    )
    monkeypatch.setattr(symile_test_data, "STRICT_PNEUMONIA_TEST_ADMISSIONS", test_count)
    authority = symile_publication._load_cv_authority(
        {
            "bundle_id": config.dataset.bundle_id,
            "bundle_manifest_sha256": config.dataset.bundle_manifest_sha256,
            "cv_assignment_id": config.dataset.cv_assignment_id,
            "cv_manifest_sha256": cv_hash,
        },
        manifest_root,
    )

    class TinyNeural(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1, bias=False)

    monkeypatch.setattr(symile_publication, "_validate_neural_reconstruction", lambda *args: None)
    monkeypatch.setattr(final_packages, "_reconstruct_neural", lambda config: TinyNeural())
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
    for family in (*symile_campaign.SYMILE_CORE_DEVELOPMENT_FAMILIES, "cxr_labs_ecg_gated"):
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
            authority=authority,
            cv_manifest_sha256=cv_hash,
            config=configs[family],
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
    family_ids = {
        family: developments[family].manifest["development_id"]
        for family in symile_campaign.SYMILE_CORE_DEVELOPMENT_FAMILIES
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
    primary_config = with_runtime(
        configs["cxr_labs_gated"],
        manifest_directory=manifest_root,
    )
    cohort = load_symile_development_cohort(primary_config).frame
    attributes = cohort.loc[:, ["sample_id", "age_years", "sex", "view_position"]].copy()
    attributes["observed_lab_count"] = cohort.loc[:, LAB_OBSERVED_COLUMNS].sum(axis=1)
    subgroup_id = extension_result.publish_focused_subgroup_derivative(
        report_root=report_root,
        summary=focused_development_subgroups(cxr, primary, attributes),
    )
    extension = extension_result.publish_ecg_extension_result(
        report_root=report_root,
        core_analysis_directory=analysis_directory,
        ecg_development=developments["cxr_labs_ecg_gated"],
        focused_subgroup_derivative=subgroup_id,
        model_root=model_root / "development",
        prediction_root=private_root,
        manifest_root=manifest_root,
    )

    feature_frame = _development_frame(20)[list(LAB_FEATURE_COLUMNS)]
    targets = np.asarray([index % 2 for index in range(20)], dtype=np.int8)
    operational = {
        "git_commit": "f" * 40,
        "git_dirty": False,
        "dependency_lock_sha256": "1" * 64,
        "mlflow_run_id": "synthetic-final-run",
    }
    packages = []
    for family in ("labs_logistic", "labs_lightgbm"):
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
    preprocessor = symile_publication.SymileLabEcdfTransformer().fit(feature_frame)
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
    for family in ("cxr_densenet", "cxr_labs_concat", "cxr_labs_gated", "cxr_labs_ecg_gated"):
        family_config = configs[family]
        budget = extension.manifest["final_family_authorities"][family]["final_training_budget"]
        for seed in (17, 42, 2026):
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
    freeze = campaign_control.publish_pretest_freeze(
        control_root=private_root / "control/symile",
        bundle={
            "bundle_id": config.dataset.bundle_id,
            "bundle_manifest_sha256": config.dataset.bundle_manifest_sha256,
            "split_assignment_id": config.dataset.split_assignment_id,
        },
        task={
            "task_id": config.task.task_id,
            "label_policy_version": config.task.label_policy_version,
        },
        ecg_extension_result=extension,
        final_packages=packages,
        neural_inference_runtime={
            "device_type": "cpu",
            "autocast_dtype": None,
            "cuda_runtime_version": None,
            "cudnn_version": None,
            "gpu_device_name": None,
            "gpu_compute_capability": None,
        },
        science_git_commit="f" * 40,
        dependency_lock_sha256="1" * 64,
    )
    record = create_or_validate_test_open_record(
        control_root=private_root / "control/symile", capability=freeze
    )
    test_data = FrozenSymileTestData(
        freeze,
        record,
        final_packages.load_final_package_config(packages[0]),
        manifest_root=manifest_root,
    )
    projection = test_data.evaluation_projection().frame()
    predictions = tuple(
        publish_prediction_evidence(
            private_root=private_root,
            dataset_id="symile",
            model_package_id=package.package_id,
            task_id=config.task.task_id,
            bundle_id=config.dataset.bundle_id,
            split_assignment_id=config.dataset.split_assignment_id,
            scope="test",
            sample_ids=projection["sample_id"].tolist(),
            targets=projection["target"].tolist(),
            logits=np.linspace(-2.0 + index / 100, 2.0 + index / 100, len(projection)),
            label_policy_version=config.task.label_policy_version,
            inference_policy=SYMILE_TEST_INFERENCE_POLICY,
            authorized_by_pretest_freeze_id=freeze.freeze_id,
        )
        for index, package in enumerate(packages)
    )
    monkeypatch.setattr(
        campaign_control,
        "cluster_bootstrap_effect",
        lambda frame, **kwargs: {"subject_count": int(frame["subject_id"].nunique())},
    )
    global_result = campaign_control.publish_global_result(
        report_root=report_root,
        capability=freeze,
        predictions=predictions,
        final_packages=packages,
        test_data=test_data,
    )
    error_review = campaign_control.publish_error_review(
        private_root=private_root,
        capability=freeze,
        predictions=predictions,
        final_packages=packages,
        test_data=test_data,
    )
    members = symile_campaign._campaign_export_members(
        freeze=freeze,
        record=record,
        ecg_extension=extension,
        packages=packages,
        predictions=predictions,
        global_result=global_result,
        error_review=error_review,
        test_data=test_data,
        manifest_root=manifest_root,
        model_root=model_root,
        report_root=report_root,
        private_root=private_root,
    )
    backup_root = tmp_path / "backup"
    export_and_verify(
        members=members,
        export_root=tmp_path / "outbox",
        backup_root=backup_root,
        export_name=global_result.result_id,
        restoration_validator=symile_campaign._validate_restored_campaign,
    )
    backup = backup_root / f"{global_result.result_id}.zip"
    for member in members:
        shutil.rmtree(member.path) if member.path.is_dir() else member.path.unlink()
    assert not any(member.path.exists() for member in members)
    restore_and_validate_symile_export(
        backup, restoration_validator=symile_campaign._validate_restored_campaign
    )
    assert analysis_id == extension.manifest["core_analysis_id"]
