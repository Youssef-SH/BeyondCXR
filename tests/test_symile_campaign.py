from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from neural_test_support import cpu_runtime
from symile_campaign_test_support import (
    _freeze,
    _projection_for_freeze,
    _publish_preserved_campaign_development,
    _publish_preserved_campaign_final_packages,
    _published_preserved_campaign_foundation,
    _synthetic_final_packages,
    _synthetic_opened_development_graph,
    _synthetic_opened_heldout_graph,
)

import beyondcxr.training.symile_campaign as symile_campaign
import beyondcxr.training.symile_campaign_control as campaign_control
import beyondcxr.training.symile_ecg_extension_result as extension_result
import beyondcxr.training.symile_final_packages as final_packages
import beyondcxr.training.symile_test_inference as test_inference
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.release import serving as release_serving
from beyondcxr.release.render import RESULT_END, RESULT_START
from beyondcxr.release.reproduction import publish_results, reproduce_results
from beyondcxr.training.config import load_symile_development_config
from beyondcxr.training.symile_campaign_control import create_or_validate_test_open_record
from beyondcxr.training.symile_export import export_and_verify, restore_and_validate_symile_export
from beyondcxr.training.symile_test_data import FrozenSymileTestData
from beyondcxr.utils.private_predictions import (
    SYMILE_TEST_INFERENCE_POLICY,
    publish_prediction_evidence,
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
    development = _synthetic_opened_development_graph(
        model_root=model_root,
        report_root=report_root,
        private_root=private_root,
        freeze=freeze,
    )
    heldout = _synthetic_opened_heldout_graph(
        tmp_path,
        manifest_root=manifest_root,
        model_root=model_root,
        report_root=report_root,
        private_root=private_root,
        freeze=freeze,
        cv_id=development.cv_id,
    )
    developments = development.developments
    extension = development.extension
    packages = heldout.packages
    predictions = heldout.predictions
    global_result = heldout.global_result
    error_review = heldout.error_review
    global_id = global_result.result_id
    package_by_id = {package.package_id: package for package in packages}
    prediction_by_id = {prediction.prediction_id: prediction for prediction in predictions}
    monkeypatch.setattr(
        symile_campaign, "validate_analysis_result", lambda *a, **k: development.analysis
    )
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
        lambda frame, **kwargs: {
            "point": 0.0,
            "lower": -0.1,
            "upper": 0.1,
            "accepted": 2000,
            "attempts": 2000,
        },
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


def test_preserved_campaign_drives_downstream_release_and_serving_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Certify the archive with real recursive validators and bounded synthetic fitted state."""
    manifest_root = tmp_path / "data/manifests"
    model_root = tmp_path / "models/symile"
    report_root = tmp_path / "reports/symile"
    private_root = tmp_path / "private"
    foundation = _published_preserved_campaign_foundation(tmp_path, manifest_root, monkeypatch)
    development = _publish_preserved_campaign_development(
        tmp_path,
        foundation=foundation,
        manifest_root=manifest_root,
        model_root=model_root,
        report_root=report_root,
        private_root=private_root,
        monkeypatch=monkeypatch,
    )
    config = foundation.config
    configs = foundation.configs
    developments = development.developments
    extension = development.extension
    analysis_id = development.analysis_id
    packages = _publish_preserved_campaign_final_packages(
        model_root=model_root,
        configs=configs,
        developments=developments,
        extension=extension,
        tabular_families=("labs_logistic", "labs_lightgbm"),
        neural_families=(
            "cxr_densenet",
            "cxr_labs_concat",
            "cxr_labs_gated",
            "cxr_labs_ecg_gated",
        ),
        neural_seeds=(17, 42, 2026),
    )

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
        lambda frame, **kwargs: {
            "point": 0.0,
            "lower": -0.1,
            "upper": 0.1,
            "accepted": 2000,
            "attempts": 2000,
        },
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
    archive = export_and_verify(
        members=members,
        export_root=tmp_path / "outbox",
        backup_root=backup_root,
        export_name=global_result.result_id,
        restoration_validator=symile_campaign.validate_restored_campaign,
    )
    readme = tmp_path / "README.md"
    model_card = tmp_path / "model-card.md"
    bounded = f"Before\n{RESULT_START}\nAwaiting formal Symile execution\n{RESULT_END}\nAfter\n"
    readme.write_text(bounded, encoding="utf-8")
    model_card.write_text(bounded, encoding="utf-8")
    public_results = tmp_path / "public-results"
    public_results.mkdir()
    (public_results / "README.md").write_text("# Aggregate results\n", encoding="utf-8")
    publish_results(
        artifact_root=archive,
        output_root=public_results,
        readme_path=readme,
        model_card_path=model_card,
    )
    assert (public_results / "symile/tables/observedness_subgroups.md").is_file()
    assert (public_results / "symile/figures/reliability.svg").is_file()
    generated_readme = readme.read_text(encoding="utf-8")
    generated_model_card = model_card.read_text(encoding="utf-8")
    for generated_document in (generated_readme, generated_model_card):
        assert "Development OOF evidence" in generated_document
        assert "Held-out test evidence" in generated_document
    assert (public_results / "symile/binding.json").is_file()
    reproduce_results(
        binding_path=public_results / "symile/binding.json",
        artifact_root=archive,
        output_root=public_results,
        readme_path=readme,
        model_card_path=model_card,
    )
    monkeypatch.setattr(
        release_serving,
        "clean_release_provenance",
        lambda root: {
            "git_commit": "e" * 40,
            "dependency_lock_sha256": "2" * 64,
        },
    )
    authority = release_serving.publish_and_smoke_test_serving_authority(
        artifact_root=archive,
        authority_root=tmp_path / "authorities",
        repository_root=tmp_path / "checkout",
    )
    assert authority.name.startswith("serving-authority-")
    authority_bytes = {
        path.relative_to(authority).as_posix(): path.read_bytes()
        for path in authority.rglob("*")
        if path.is_file()
    }
    repeated_authority = release_serving.publish_and_smoke_test_serving_authority(
        artifact_root=archive,
        authority_root=tmp_path / "authorities",
        repository_root=tmp_path / "checkout",
    )
    assert repeated_authority == authority
    assert {
        path.relative_to(authority).as_posix(): path.read_bytes()
        for path in authority.rglob("*")
        if path.is_file()
    } == authority_bytes
    assert [path for path in (tmp_path / "authorities").iterdir() if path.is_dir()] == [authority]
    backup = backup_root / f"{global_result.result_id}.zip"
    for member in members:
        shutil.rmtree(member.path) if member.path.is_dir() else member.path.unlink()
    assert not any(member.path.exists() for member in members)
    restore_and_validate_symile_export(
        backup, restoration_validator=symile_campaign.validate_restored_campaign
    )
    assert analysis_id == extension.manifest["core_analysis_id"]
