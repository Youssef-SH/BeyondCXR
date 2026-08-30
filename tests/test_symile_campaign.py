from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from neural_test_support import cpu_runtime
from symile_campaign_test_support import (
    _freeze,
    _synthetic_final_family_authorities,
    _synthetic_final_packages,
    _synthetic_held_out_capability,
    _synthetic_test_predictions,
)

import beyondcxr.release.serving as release_serving
import beyondcxr.serving.authority as serving_authority
import beyondcxr.training.symile_campaign as symile_campaign
import beyondcxr.training.symile_campaign_control as campaign_control
import beyondcxr.training.symile_development_decision as development_decision
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.release.binding import validate_result_binding
from beyondcxr.release.inputs import derive_result_binding
from beyondcxr.serving.predictor import SymileServingPredictor
from beyondcxr.training.config import load_symile_development_config, with_runtime
from beyondcxr.training.preservation import (
    PreservationMember,
    _validated_restoration_context,
    export_and_verify,
    restore_and_validate_export,
)
from beyondcxr.training.symile_campaign_control import (
    GlobalResultIntegrity,
)
from beyondcxr.training.symile_families import DEVELOPMENT_FAMILY_ORDER
from beyondcxr.utils.semantic_identity import canonical_json_bytes, canonical_scientific_id
from beyondcxr.utils.symile_publication import ValidatedDevelopmentResult


@pytest.fixture(autouse=True)
def _bypass_full_source_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Campaign orchestration tests isolate the separately tested read-only preflight."""
    monkeypatch.setattr(symile_campaign, "_preflight_symile_campaign", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        symile_campaign, "_validate_symile_storage_readiness", lambda *args, **kwargs: None
    )


@pytest.fixture(scope="module")
def bounded_restored_symile_consumers(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("bounded-restored-symile")
    patch = pytest.MonkeyPatch()
    canonical_validate_restored_campaign = symile_campaign.validate_restored_campaign
    live = root / "live"
    manifest_root = live / "data/manifests"
    model_root = live / "models/symile"
    report_root = live / "reports/symile"
    private_root = live / "private"
    manifest_root.mkdir(parents=True)
    (manifest_root / "authority.json").write_text("{}\n", encoding="utf-8")
    freeze = _freeze(private_root / "control" / "symile" / "freezes")
    capability = _synthetic_held_out_capability(live / "lifecycle", freeze)
    shutil.copytree(
        capability.frozen_campaign.lifecycle.control_root,
        private_root / "control" / "symile",
        dirs_exist_ok=True,
    )
    archived_execution_manifest = {
        **capability.execution.manifest,
        "git_commit": "f" * 40,
        "dependency_lock_sha256": "1" * 64,
    }
    (
        private_root
        / "control"
        / "symile"
        / "executions"
        / capability.execution.execution_id
        / "manifest.json"
    ).write_bytes(canonical_json_bytes(archived_execution_manifest))
    development_ids = {
        family: "development-" + f"{index:x}" * 64
        for index, family in enumerate(DEVELOPMENT_FAMILY_ORDER, start=1)
    }
    cv_id = "cv-assignment-" + "c" * 64
    representative_directory = (
        report_root / "development/families" / development_ids["labs_logistic"]
    )
    representative_directory.mkdir(parents=True)
    (representative_directory / "authority.json").write_text("{}\n", encoding="utf-8")
    representative_development = ValidatedDevelopmentResult(
        representative_directory,
        {
            "development_id": development_ids["labs_logistic"],
            "family_id": "labs_logistic",
            "scientific_context": {
                "bundle_id": freeze.manifest["bundle_id"],
                "cv_assignment_id": cv_id,
            },
        },
    )
    family_authorities = _synthetic_final_family_authorities(development_ids, freeze)
    decision_directory = (
        report_root / "development-decisions" / freeze.manifest["development_decision_id"]
    )
    decision_directory.mkdir(parents=True)
    decision_manifest = {
        "development_decision_schema_version": (
            development_decision.DEVELOPMENT_DECISION_SCHEMA_VERSION
        ),
        "development_decision_id": freeze.manifest["development_decision_id"],
        "dataset_id": "symile",
        "bundle_id": freeze.manifest["bundle_id"],
        "cv_assignment_id": cv_id,
        "task": dict(freeze.manifest["task"]),
        "family_development_ids": [
            {"family_id": family, "development_id": development_ids[family]}
            for family in DEVELOPMENT_FAMILY_ORDER
        ],
        "decision_policy": development_decision.DEVELOPMENT_DECISION_POLICY,
        "selected_family_authorities": family_authorities,
        "primary_thresholds": {
            "youden_j": 0.5,
            "target_sensitivity": 0.5,
        },
    }
    decision_claims: dict[str, object] = {}
    decision_manifest_bytes = (
        json.dumps(decision_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    decision_claims_bytes = (
        json.dumps(decision_claims, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    (decision_directory / "manifest.json").write_bytes(decision_manifest_bytes)
    (decision_directory / "claims.json").write_bytes(decision_claims_bytes)
    decision_authority = development_decision.ValidatedDevelopmentDecision(
        decision_directory,
        decision_manifest,
        hashlib.sha256(decision_manifest_bytes).hexdigest(),
        hashlib.sha256(decision_claims_bytes).hexdigest(),
        report_root,
        model_root / "development",
        private_root,
        manifest_root,
        development_decision._VALIDATION_GUARD,
    )
    package_values = _synthetic_final_packages(
        live / "package-templates",
        freeze,
        family_authorities=family_authorities,
    )
    packages = []
    for package in package_values:
        directory = model_root / "final/packages" / package.package_id
        directory.mkdir(parents=True)
        (directory / "authority.json").write_text("{}\n", encoding="utf-8")
        packages.append(replace(package, directory=directory))
    packages = tuple(packages)

    prediction_values = _synthetic_test_predictions(live / "prediction-templates", freeze, packages)
    predictions = []
    for prediction in prediction_values:
        directory = private_root / "predictions/symile/test" / prediction.prediction_id
        directory.mkdir(parents=True)
        (directory / "authority.json").write_text("{}\n", encoding="utf-8")
        predictions.append(replace(prediction, directory=directory))
    predictions = tuple(predictions)

    global_id = "global-result-" + "7" * 64
    global_directory = report_root / "global-results" / global_id
    global_directory.mkdir(parents=True)
    global_manifest = {
        "global_result_schema_version": 1,
        "global_result_id": global_id,
        "pretest_freeze_id": freeze.freeze_id,
        "prediction_ids": [prediction.prediction_id for prediction in predictions],
        "evaluation_policy": {},
    }
    (global_directory / "manifest.json").write_bytes(canonical_json_bytes(global_manifest))

    def bounded_global_artifact(path, **_kwargs):
        integrity = GlobalResultIntegrity(
            Path(path),
            global_manifest,
            "c" * 64,
            "d" * 64,
        )
        return integrity, (), object(), (), {}

    patch.setattr(campaign_control, "_validate_global_result_artifact", bounded_global_artifact)
    patch.setattr(campaign_control, "_derive_global_claims", lambda *_args: {})
    global_result = campaign_control.validate_global_result(
        global_directory,
        capability=freeze,
        predictions=(),
        final_packages=(),
        test_data=object(),
    )
    archive = export_and_verify(
        members=[
            PreservationMember(manifest_root.parent, Path("data")),
            PreservationMember(model_root.parent, Path("models")),
            PreservationMember(report_root.parent, Path("reports")),
            PreservationMember(private_root, Path("private")),
        ],
        export_root=root / "outbox",
        backup_root=root / "backup",
        export_name="bounded-symile-authority",
    )
    backup = root / "backup" / archive.name
    shutil.rmtree(live)
    authority_root = root / "serving-authorities"

    def consume(restored_root: Path):
        patch.setattr(campaign_control, "_validate_global_result_artifact", bounded_global_artifact)
        patch.setattr(campaign_control, "_derive_global_claims", lambda *_args: {})
        _validated_restoration_context(restored_root)
        restored_live = restored_root

        def rebase(path: Path) -> Path:
            return restored_live / path.relative_to(live)

        restored_freeze = replace(freeze, directory=rebase(freeze.directory))
        execution = replace(
            capability.execution,
            directory=(
                restored_live
                / "private/control/symile/executions"
                / capability.execution.execution_id
            ),
            manifest=archived_execution_manifest,
        )
        restored_decision = replace(
            decision_authority,
            directory=rebase(decision_authority.directory),
            report_root=rebase(decision_authority.report_root),
            model_root=rebase(decision_authority.model_root),
            prediction_root=rebase(decision_authority.prediction_root),
            manifest_root=rebase(decision_authority.manifest_root),
        )
        restored_development = replace(
            representative_development,
            directory=rebase(representative_development.directory),
        )
        restored_packages = tuple(
            replace(item, directory=rebase(item.directory)) for item in packages
        )
        restored_freeze = replace(
            restored_freeze,
            development_decision=restored_decision,
            final_packages=restored_packages,
        )
        restored_predictions = tuple(
            replace(item, directory=rebase(item.directory)) for item in predictions
        )
        restored_global = campaign_control.validate_global_result(
            rebase(global_result.directory),
            capability=restored_freeze,
            predictions=restored_predictions,
            final_packages=restored_packages,
            test_data=object(),
        )
        required_files = (
            restored_freeze.directory / "manifest.json",
            execution.directory / "manifest.json",
            restored_decision.directory / "manifest.json",
            restored_decision.directory / "claims.json",
            restored_development.directory / "authority.json",
            *(item.directory / "authority.json" for item in restored_packages),
            *(item.directory / "authority.json" for item in restored_predictions),
            restored_global.directory / "manifest.json",
        )
        if any(path.is_symlink() or not path.is_file() for path in required_files):
            raise ManifestBuildError("Bounded restored Symile authority is incomplete")
        by_package = {item.package_id: item for item in restored_packages}
        by_prediction = {item.prediction_id: item for item in restored_predictions}
        patch.setattr(
            symile_campaign,
            "validate_development_decision",
            lambda *_args, **_kwargs: restored_decision,
        )
        patch.setattr(
            symile_campaign,
            "analyze_symile_development",
            lambda *_args, **_kwargs: restored_live / "analysis.json",
        )
        patch.setattr(symile_campaign, "validate_analysis_result", lambda *_a, **_k: {})
        patch.setattr(
            symile_campaign,
            "validate_development_result",
            lambda *_args, **_kwargs: restored_development,
        )
        patch.setattr(
            symile_campaign,
            "_ensure_decision_subgroup_derivative",
            lambda *_args, **_kwargs: restored_live / "subgroup.json",
        )
        patch.setattr(
            symile_campaign,
            "validate_focused_subgroup_derivative",
            lambda *_args, **_kwargs: {"policy": {}, "strata": {}},
        )
        patch.setattr(
            symile_campaign,
            "validate_final_package",
            lambda path, **_kwargs: by_package[Path(path).name],
        )
        patch.setattr(
            symile_campaign,
            "validate_pretest_freeze",
            lambda *_args, **_kwargs: restored_freeze,
        )
        patch.setattr(
            symile_campaign,
            "validate_symile_execution",
            lambda *_args, **_kwargs: execution,
        )
        patch.setattr(
            symile_campaign,
            "validate_symile_execution_freeze_science",
            lambda *_args, **_kwargs: None,
        )
        opening = SimpleNamespace(path=restored_live / "private/control/symile/test-open.json")
        patch.setattr(
            symile_campaign,
            "_validate_preserved_test_open",
            lambda **_kwargs: opening,
        )
        patch.setattr(
            symile_campaign,
            "validate_prediction_evidence",
            lambda path: by_prediction[Path(path).name],
        )
        test_data = SimpleNamespace(evaluation_projection=lambda: object())
        patch.setattr(
            symile_campaign,
            "RestoredSymileTestData",
            lambda *_args, **_kwargs: test_data,
        )
        patch.setattr(
            symile_campaign,
            "load_final_package_config",
            lambda *_args, **_kwargs: object(),
        )
        patch.setattr(
            symile_campaign,
            "validate_restored_test_prediction_publication",
            lambda *_args, **_kwargs: None,
        )
        patch.setattr(
            symile_campaign,
            "validate_prediction_against_test_projection",
            lambda *_args, **_kwargs: None,
        )
        patch.setattr(
            symile_campaign,
            "publish_error_review",
            lambda **_kwargs: rebase(root / "live/private/error-review/symile/error-review.json"),
        )
        patch.setattr(
            symile_campaign,
            "validate_exact_restored_closure",
            lambda *_args, **_kwargs: None,
        )
        patch.setattr(
            symile_campaign,
            "_project_campaign_export_members",
            lambda **_kwargs: (),
        )
        campaign = canonical_validate_restored_campaign(restored_root)
        patch.setattr(
            development_decision,
            "_decision_document",
            lambda **_kwargs: (restored_decision.manifest, decision_claims),
        )
        patch.setattr(
            serving_authority,
            "validate_final_package",
            lambda path, *, expected_package_id=None, **_kwargs: replace(
                by_package[str(expected_package_id)], directory=Path(path)
            ),
        )
        patch.setattr(
            campaign_control,
            "validate_final_package",
            lambda _path, *, expected_package_id=None, **_kwargs: by_package[
                str(expected_package_id)
            ],
        )
        patch.setattr(
            serving_authority,
            "validate_global_result_integrity",
            lambda *_args, **_kwargs: GlobalResultIntegrity(
                restored_global.directory,
                restored_global.manifest,
                restored_global.claims_sha256,
                restored_global.reliability_sha256,
            ),
        )
        binding = derive_result_binding(campaign)
        binding_path = authority_root / "binding.json"
        binding_path.parent.mkdir(parents=True, exist_ok=True)
        binding_path.write_bytes(binding.canonical_bytes())
        publication = release_serving._prepare_validated_campaign_serving_authority(
            authority_root=authority_root,
            campaign=campaign,
            release_archive=backup,
            serving_release={
                "git_commit": "2" * 40,
                "dependency_lock_sha256": "3" * 64,
            },
        )
        assert copy.copy(campaign) is campaign
        assert copy.deepcopy(campaign) is campaign
        with pytest.raises(TypeError):
            replace(campaign, global_result=restored_global)
        with pytest.raises(TypeError):
            replace(campaign, restored_root=restored_live)
        with pytest.raises(AttributeError):
            campaign.global_result = restored_global
        with pytest.raises(TypeError):
            campaign.analysis["altered"] = True
        with pytest.raises(TypeError):
            campaign.subgroup["policy"]["altered"] = True

        forged = SimpleNamespace(
            freeze=campaign.freeze,
            execution=campaign.execution,
            development_decision=campaign.development_decision,
            analysis={"altered": True},
            subgroup=campaign.subgroup,
            developments=campaign.developments,
            packages=campaign.packages,
            predictions=campaign.predictions,
            global_result=campaign.global_result,
            test_data=campaign.test_data,
            restored_root=campaign.restored_root,
            restoration_context=campaign.restoration_context,
        )
        with pytest.raises(PermissionError, match="validated restored campaign"):
            derive_result_binding(forged)
        with pytest.raises(PermissionError, match="validated restored campaign"):
            release_serving._prepare_validated_campaign_serving_authority(
                authority_root=authority_root,
                campaign=forged,
                release_archive=backup,
                serving_release={"git_commit": "2" * 40, "dependency_lock_sha256": "3" * 64},
            )

        original_commit = execution.manifest["git_commit"]
        execution.manifest["git_commit"] = "0" * 40
        with pytest.raises(ManifestBuildError, match="changed after validation"):
            derive_result_binding(campaign)
        execution.manifest["git_commit"] = original_commit
        assert derive_result_binding(campaign) == binding
        return SimpleNamespace(
            binding=binding,
            binding_path=publication.result_binding,
            publication=publication,
            campaign=campaign,
            restore_campaign=consume,
            restoration_patches=patch,
            immutability_verified=True,
        )

    try:
        consumers = restore_and_validate_export(backup, restoration_validator=consume)
    finally:
        patch.undo()
    expected = SimpleNamespace(
        bundle_id=freeze.manifest["bundle_id"],
        cv_id=cv_id,
        decision_id=decision_authority.decision_id,
        freeze_id=freeze.freeze_id,
        package_ids=tuple(item.package_id for item in packages),
        prediction_ids=tuple(item.prediction_id for item in predictions),
        global_result_id=global_id,
        execution_id=capability.execution.execution_id,
    )
    return consumers, expected


def test_campaign_export_rejects_integrity_only_global_result(tmp_path: Path) -> None:
    integrity = GlobalResultIntegrity(
        tmp_path / ("global-result-" + "7" * 64),
        {"global_result_id": "global-result-" + "7" * 64},
        "c" * 64,
        "d" * 64,
    )

    with pytest.raises(PermissionError, match="full scientific certification"):
        symile_campaign._campaign_export_members(
            freeze=None,
            execution=None,
            opening_path=tmp_path / "opening.json",
            live_capability=None,
            development_decision=None,
            packages=(),
            predictions=(),
            global_result=integrity,
            error_review=tmp_path / "error-review.json",
            test_data=SimpleNamespace(),
            manifest_root=tmp_path / "manifests",
            model_root=tmp_path / "models",
            report_root=tmp_path / "reports",
            private_root=tmp_path / "private",
        )


def test_backup_archive_restores_bounded_authority_for_release_binding(
    bounded_restored_symile_consumers,
) -> None:
    consumers, expected = bounded_restored_symile_consumers
    binding = consumers.binding

    assert binding.data.bundle_id == expected.bundle_id
    assert binding.data.cv_assignment_id == expected.cv_id
    assert binding.development.development_decision_id == expected.decision_id
    assert binding.held_out.pretest_freeze_id == expected.freeze_id
    assert binding.held_out.final_package_ids == expected.package_ids
    assert binding.held_out.prediction_ids == expected.prediction_ids
    assert binding.held_out.global_result_id == expected.global_result_id


def test_restored_bounded_authority_reaches_serving_publication(
    bounded_restored_symile_consumers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumers, expected = bounded_restored_symile_consumers
    publication = consumers.publication

    assert publication.authority.manifest["bundle"] == {"bundle_id": expected.bundle_id}
    assert publication.authority.manifest["global_result"] == {
        "global_result_id": expected.global_result_id,
        "pretest_freeze_id": expected.freeze_id,
    }
    deployment = json.loads(publication.deployment_provenance.read_bytes())
    assert deployment["execution"]["execution_id"] == expected.execution_id

    def plain(value):
        if isinstance(value, dict) or hasattr(value, "items"):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    packages = tuple(
        replace(package, manifest=plain(package.manifest))
        for package in publication.authority.packages
    )
    by_id = {package.package_id: package for package in packages}
    monkeypatch.setattr(
        serving_authority,
        "validate_final_package",
        lambda path, **_kwargs: by_id[Path(path).name],
    )
    monkeypatch.setattr(
        SymileServingPredictor,
        "__init__",
        lambda self, authority, **_kwargs: setattr(self, "authority", authority),
    )
    monkeypatch.setattr(
        symile_campaign,
        "validate_restored_campaign",
        lambda root: consumers.restore_campaign(root).campaign,
    )
    try:
        loaded = SymileServingPredictor.load(
            publication.authority.directory,
            deployment_provenance=publication.deployment_provenance,
            result_binding=publication.result_binding,
            release_archive=publication.release_archive,
        )
    finally:
        consumers.restoration_patches.undo()
    assert loaded.authority.authority_id == publication.authority.authority_id


def test_restored_campaign_authority_is_immutable_and_nontransferable(
    bounded_restored_symile_consumers,
) -> None:
    consumers, _ = bounded_restored_symile_consumers
    assert consumers.immutability_verified is True
    authority = consumers.publication.authority
    assert copy.copy(authority) is authority
    assert copy.deepcopy(authority) is authority
    with pytest.raises(TypeError):
        replace(authority, directory=Path("/foreign"))
    with pytest.raises(TypeError):
        authority.manifest["serving_authority_id"] = "serving-authority-" + "0" * 64


@pytest.mark.parametrize(
    "mutation",
    (
        "foreign_packages",
        "wrong_freeze",
        "wrong_result",
        "wrong_decision",
        "altered_thresholds",
    ),
)
def test_serving_startup_rejects_self_consistent_foreign_authority(
    bounded_restored_symile_consumers,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    consumers, _ = bounded_restored_symile_consumers
    publication = consumers.publication
    document = json.loads((publication.authority.directory / "manifest.json").read_bytes())
    deployment = json.loads(publication.deployment_provenance.read_bytes())
    binding_document = consumers.binding.to_document()

    def mutable(value):
        if isinstance(value, dict) or hasattr(value, "items"):
            return {key: mutable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [mutable(item) for item in value]
        return value

    packages = tuple(
        replace(package, manifest=mutable(package.manifest))
        for package in publication.authority.packages
    )
    if mutation == "foreign_packages":
        foreign = []
        for index, package in enumerate(packages, start=1):
            package_manifest = mutable(package.manifest)
            package_manifest["final_package_id"] = "final-package-" + f"{index + 40:064x}"
            directory = tmp_path / "foreign-packages" / package_manifest["final_package_id"]
            directory.mkdir(parents=True)
            foreign.append(replace(package, directory=directory, manifest=package_manifest))
        packages = tuple(foreign)
        for member, package in zip(document["members"], packages, strict=True):
            member["package_id"] = package.package_id
        old_ids = deployment["serving_package_ids"]
        new_ids = [package.package_id for package in packages]
        deployment["serving_package_ids"] = new_ids
        binding_document["held_out"]["final_package_ids"] = [
            new_ids[old_ids.index(package_id)] if package_id in old_ids else package_id
            for package_id in binding_document["held_out"]["final_package_ids"]
        ]
    elif mutation == "wrong_freeze":
        document["global_result"]["pretest_freeze_id"] = "pretest-freeze-" + "8" * 64
        deployment["pretest_freeze_id"] = document["global_result"]["pretest_freeze_id"]
        binding_document["held_out"]["pretest_freeze_id"] = document["global_result"][
            "pretest_freeze_id"
        ]
    elif mutation == "wrong_result":
        document["global_result"]["global_result_id"] = "global-result-" + "8" * 64
        deployment["global_result_id"] = document["global_result"]["global_result_id"]
        binding_document["held_out"]["global_result_id"] = document["global_result"][
            "global_result_id"
        ]
    elif mutation == "wrong_decision":
        decision_id = "development-decision-" + "8" * 64
        deployment["development_decision_id"] = decision_id
        binding_document["development"]["development_decision_id"] = decision_id
    else:
        document["primary_thresholds"]["youden_j"] = 0.25
        deployment["primary_thresholds"] = dict(document["primary_thresholds"])
    semantic = {
        key: value
        for key, value in document.items()
        if key not in {"serving_authority_schema_version", "serving_authority_id"}
    }
    authority_id = canonical_scientific_id(
        serving_authority.SERVING_AUTHORITY_PREFIX,
        semantic,
        domain="serving-authority",
        schema_version=serving_authority.SERVING_AUTHORITY_SCHEMA_VERSION,
    )
    document["serving_authority_id"] = authority_id
    binding = validate_result_binding(binding_document)
    binding_path = tmp_path / "attacker-binding.json"
    binding_path.write_bytes(binding.canonical_bytes())
    deployment["serving_authority_id"] = authority_id
    deployment["result_binding_sha256"] = hashlib.sha256(binding.canonical_bytes()).hexdigest()
    deployment.pop("deployment_id")
    deployment["deployment_id"] = serving_authority.semantic_sha256(
        "serving-deployment-provenance",
        serving_authority.SERVING_DEPLOYMENT_SCHEMA_VERSION,
        deployment,
    )
    deployment_path = tmp_path / "attacker-deployment.json"
    deployment_path.write_bytes(canonical_json_bytes(deployment))
    directory = tmp_path / authority_id
    directory.mkdir()
    (directory / "manifest.json").write_bytes(canonical_json_bytes(document))
    copied_packages = directory / "packages"
    copied_packages.mkdir()
    for package in packages:
        shutil.copytree(package.directory, copied_packages / package.package_id)
    by_id = {package.package_id: package for package in packages}
    monkeypatch.setattr(
        serving_authority,
        "validate_final_package",
        lambda path, **_kwargs: replace(by_id[Path(path).name], directory=Path(path)),
    )

    observed = serving_authority.validate_serving_distribution(directory)
    assert observed.authority_id == authority_id
    with pytest.raises(ManifestBuildError, match="release"):
        serving_authority.validate_serving_authority(
            directory,
            deployment_provenance=deployment_path,
            result_binding=binding_path,
            release_archive=publication.release_archive,
        )


def test_storage_readiness_rejects_low_space_before_campaign_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    events: list[str] = []
    monkeypatch.setattr(symile_campaign, "discover_repository_root", lambda: repository)
    monkeypatch.setattr(
        symile_campaign,
        "_validate_symile_storage_readiness",
        lambda *args: (_ for _ in ()).throw(OSError("low storage")),
    )
    monkeypatch.setattr(
        symile_campaign,
        "build_symile_formal_preflight",
        lambda *args: events.append("formal_preflight"),
    )
    monkeypatch.setattr(
        symile_campaign,
        "run_symile_development",
        lambda *args, **kwargs: events.append("development"),
    )
    with pytest.raises(OSError, match="low storage"):
        symile_campaign.run_symile_campaign(
            source_root=source,
            backup_root=tmp_path / "backup",
            bundle_id="bundle-" + "a" * 64,
            cv_assignment_id="cv-assignment-" + "c" * 64,
        )
    assert events == []
    assert not (repository / "models").exists()


def test_storage_readiness_accounts_for_distinct_and_shared_filesystems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    repository = tmp_path / "repository"
    backup = tmp_path / "external/backup"
    gib = 1024**3
    monkeypatch.setattr(
        symile_campaign,
        "_storage_volume",
        lambda path: (1, 32 * gib) if path == repository else (2, 8 * gib),
    )
    symile_campaign._validate_symile_storage_readiness(repository, backup)

    monkeypatch.setattr(symile_campaign, "_storage_volume", lambda path: (1, 39 * gib))
    with pytest.raises(OSError, match="40 GiB"):
        symile_campaign._validate_symile_storage_readiness(repository, backup)


def test_final_fit_configures_the_explicit_tracking_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = with_runtime(
        load_symile_development_config(
            "configs/symile_labs_logistic.yaml",
            bundle_id="bundle-" + "a" * 64,
            cv_assignment_id="cv-assignment-" + "c" * 64,
        ),
        seed=42,
    )
    supplied = "sqlite:////tmp/explicit-final-fit.db"
    configured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        symile_campaign,
        "configure_mlflow",
        lambda *, experiment_name, tracking_uri: configured.append((experiment_name, tracking_uri)),
    )

    class Run:
        def __enter__(self):
            return "run-id"

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(symile_campaign, "tracked_run", lambda **kwargs: Run())
    monkeypatch.setattr(symile_campaign, "log_source_config", lambda config: None)
    monkeypatch.setattr(symile_campaign.mlflow, "set_tags", lambda values: None)
    monkeypatch.setattr(symile_campaign.mlflow, "set_tag", lambda key, value: None)
    package = SimpleNamespace(package_id="final-package-test")

    result = symile_campaign._tracked_final_fit(
        config,
        tracking_uri=supplied,
        member_seed=42,
        provenance={
            "git_commit": "f" * 40,
            "dependency_lock_sha256": "0" * 64,
        },
        fit=lambda operational: package,
    )

    assert result is package
    assert configured == [(config.runtime.experiment_name, supplied)]


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


def test_formal_campaign_accepts_existing_tracking_database_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    with sqlite3.connect(repository_root / "mlflow.db") as connection:
        connection.execute("CREATE TABLE tracking_state (value INTEGER NOT NULL)")
    source_root = tmp_path / "source"
    source_root.mkdir()
    backup_root = tmp_path / "backup"
    monkeypatch.setattr(symile_campaign, "discover_repository_root", lambda: repository_root)
    monkeypatch.setattr(
        symile_campaign,
        "load_symile_development_config",
        lambda path, **coordinates: load_symile_development_config(
            Path("configs") / path.name,
            bundle_id="bundle-" + "a" * 64,
            cv_assignment_id="cv-assignment-" + "c" * 64,
        ),
    )

    class DestinationValidationPassed(Exception):
        pass

    def stop_after_destination_validation(root: Path, **coordinates: str) -> None:
        assert root == repository_root
        raise DestinationValidationPassed

    monkeypatch.setattr(
        symile_campaign,
        "build_symile_formal_preflight",
        stop_after_destination_validation,
    )

    with pytest.raises(DestinationValidationPassed):
        symile_campaign.run_symile_campaign(
            source_root=source_root,
            backup_root=backup_root,
            bundle_id="bundle-" + "a" * 64,
            cv_assignment_id="cv-assignment-" + "c" * 64,
        )


def test_comprehensive_preflight_precedes_all_campaign_output_and_development(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    repository_root = Path(__file__).resolve().parents[1]
    events: list[str] = []
    monkeypatch.setattr(symile_campaign, "discover_repository_root", lambda: repository_root)

    def reject(configs, *, source_root, require_pretrained_weights):
        del configs, source_root
        assert require_pretrained_weights is True
        events.append("preflight")
        raise ManifestBuildError("preflight rejected")

    monkeypatch.setattr(symile_campaign, "_preflight_symile_campaign", reject)
    monkeypatch.setattr(
        symile_campaign,
        "run_symile_development",
        lambda *args, **kwargs: events.append("development"),
    )

    with pytest.raises(ManifestBuildError, match="preflight rejected"):
        symile_campaign.run_symile_campaign(
            source_root=source_root,
            backup_root=tmp_path / "backup",
            bundle_id="bundle-" + "a" * 64,
            cv_assignment_id="cv-assignment-" + "c" * 64,
        )

    assert events == ["preflight"]


def test_comprehensive_preflight_validates_development_assets_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    source_root = tmp_path / "source"
    source_root.mkdir()
    configs = {
        family: with_runtime(
            load_symile_development_config(
                path, bundle_id="bundle-" + "a" * 64, cv_assignment_id="cv-assignment-" + "c" * 64
            ),
            manifest_directory=tmp_path / "manifests",
            source_root=source_root,
            device="cpu",
            num_workers=0,
        )
        for family, path in symile_campaign._CONFIGS.items()
    }
    reference = next(iter(configs.values()))
    bundle = tmp_path / "manifests/symile/bundles" / reference.dataset.bundle_id
    events: list[str] = []
    monkeypatch.setattr(
        symile_campaign,
        "load_symile_development",
        lambda config: (
            events.append("bundle_cv")
            or SimpleNamespace(bundle=SimpleNamespace(bundle_directory=bundle))
        ),
    )
    manifest = {
        "source": {
            "checksum_manifest_sha256": "0" * 64,
            "source_assets": [
                {"relative_path": "cxr.npy"},
                {"relative_path": "ecg.npy"},
            ],
        }
    }
    monkeypatch.setattr(
        symile_campaign,
        "validate_symile_bundle_reference",
        lambda *args, **kwargs: (
            events.append("exact_manifest") or SimpleNamespace(manifest=manifest)
        ),
    )
    development_assets = (SimpleNamespace(),)
    monkeypatch.setattr(
        symile_campaign,
        "establish_development_release_assets",
        lambda *args, **kwargs: events.append("development_assets") or development_assets,
    )
    monkeypatch.setattr(
        symile_campaign,
        "SymileCxrStore",
        lambda *args, **kwargs: events.append("cxr_store"),
    )
    monkeypatch.setattr(
        symile_campaign,
        "SymileEcgStore",
        lambda *args, **kwargs: events.append("ecg_store"),
    )
    monkeypatch.setattr(
        symile_campaign,
        "fingerprint_pretrained_weights",
        lambda weights: events.append(f"weights:{weights}"),
    )
    monkeypatch.setattr(
        symile_campaign,
        "resolve_device",
        lambda *args, **kwargs: events.append("runtime") or cpu_runtime(),
    )

    symile_campaign._preflight_symile_campaign(configs, source_root=source_root)

    assert events[:2] == ["bundle_cv", "exact_manifest"]
    assert events[2:6] == [
        "development_assets",
        "cxr_store",
        "ecg_store",
        "weights:densenet121-res224-chex",
    ]
    assert events.count("runtime") == 5


def test_campaign_cli_requires_and_forwards_authorities_and_backup_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, object]] = []

    def campaign(*, source_root, backup_root, bundle_id, cv_assignment_id) -> dict[str, str]:
        calls.append(
            dict(
                source_root=source_root,
                backup_root=backup_root,
                bundle_id=bundle_id,
                cv_assignment_id=cv_assignment_id,
            )
        )
        return {}

    monkeypatch.setattr(symile_campaign, "run_symile_campaign", campaign)
    with pytest.raises(SystemExit) as missing:
        symile_campaign.main(["--source-root", "source"])
    assert missing.value.code == 2
    assert "--backup-root" in capsys.readouterr().err
    assert not calls
    backup = tmp_path / "persistent backup"
    arguments = ["--source-root", "source", "--backup-root", str(backup)]
    with pytest.raises(SystemExit) as unbound:
        symile_campaign.main(arguments)
    assert unbound.value.code == 2
    assert not calls
    arguments += [
        "--bundle-id",
        "bundle-" + "a" * 64,
        "--cv-assignment-id",
        "cv-assignment-" + "b" * 64,
    ]
    assert symile_campaign.main(arguments) == 0
    assert len(calls) == 1
    assert calls[0] == {
        "source_root": Path("source"),
        "backup_root": backup,
        "bundle_id": "bundle-" + "a" * 64,
        "cv_assignment_id": "cv-assignment-" + "b" * 64,
    }
    with pytest.raises(SystemExit) as unsupported:
        symile_campaign.main([*arguments, "--report-root", "elsewhere"])
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

    monkeypatch.setattr(symile_campaign, "build_symile_formal_preflight", unexpected)
    with pytest.raises(ManifestBuildError, match="outside the repository"):
        symile_campaign.run_symile_campaign(
            source_root=tmp_path / "source",
            backup_root=backup,
            bundle_id="bundle-" + "a" * 64,
            cv_assignment_id="cv-assignment-" + "c" * 64,
        )
    assert not tuple(tmp_path.iterdir())


def test_campaign_rejects_backup_symlink_into_repository(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(repository, target_is_directory=True)
    monkeypatch.setattr(symile_campaign, "discover_repository_root", lambda: repository)
    with pytest.raises(ManifestBuildError, match="outside the repository"):
        symile_campaign.run_symile_campaign(
            source_root=tmp_path / "source",
            backup_root=alias / "backup",
            bundle_id="bundle-" + "a" * 64,
            cv_assignment_id="cv-assignment-" + "c" * 64,
        )


def test_campaign_rejects_backup_parent_that_contains_repository(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(symile_campaign, "discover_repository_root", lambda: repository)
    with pytest.raises(ManifestBuildError, match="must be disjoint"):
        symile_campaign.run_symile_campaign(
            source_root=tmp_path / "source",
            backup_root=tmp_path,
            bundle_id="bundle-" + "a" * 64,
            cv_assignment_id="cv-assignment-" + "c" * 64,
        )


def test_preservation_topology_uses_broad_checkout_roots_without_nested_duplicates(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    sources = symile_campaign._preservation_topology_sources(
        repository_root=repository,
        configurable_roots=(
            repository / "data/manifests",
            repository / "models/symile",
            repository / "reports/symile",
            repository / "private",
            repository / "data/raw/symile",
        ),
    )

    assert sources == tuple(repository / name for name in ("data", "models", "reports", "private"))


@pytest.mark.parametrize("backup_contains_model", (False, True))
def test_external_formal_root_overlap_is_rejected_by_early_preservation_topology(
    tmp_path: Path,
    backup_contains_model: bool,
) -> None:
    repository = tmp_path / "repository"
    external = tmp_path / "external"
    if backup_contains_model:
        backup = external / "backup"
        model_root = backup / "models"
    else:
        model_root = external / "models"
        backup = model_root / "backup"
    sources = symile_campaign._preservation_topology_sources(
        repository_root=repository,
        configurable_roots=(
            repository / "data/manifests",
            model_root,
            repository / "reports/symile",
            repository / "private",
            repository / "data/raw/symile",
        ),
    )

    with pytest.raises(ManifestBuildError, match="disjoint"):
        symile_campaign.validate_preservation_paths(
            sources=sources,
            export_root=repository / "outbox",
            backup_root=backup,
        )

    assert not repository.exists()
    assert not external.exists()


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
    with pytest.raises(ManifestBuildError, match="symlink"):
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

    monkeypatch.setattr(symile_campaign, "build_symile_formal_preflight", unexpected)
    with pytest.raises(ManifestBuildError, match="symlink"):
        symile_campaign.run_symile_campaign(
            source_root=tmp_path / "source",
            backup_root=tmp_path / "backup",
            bundle_id="bundle-" + "a" * 64,
            cv_assignment_id="cv-assignment-" + "c" * 64,
        )


def test_pretest_campaign_refuses_to_resume_after_test_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control_root = tmp_path / "private" / "control" / "symile"
    control_root.mkdir(parents=True)
    (control_root / "test-open.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(symile_campaign, "discover_repository_root", lambda: tmp_path)
    monkeypatch.setattr(symile_campaign, "_validate_canonical_formal_layout", lambda root: None)
    monkeypatch.setattr(symile_campaign, "validate_preservation_paths", lambda **kwargs: None)
    monkeypatch.setattr(
        symile_campaign, "validate_writable_directory_destination", lambda *args: None
    )
    monkeypatch.setattr(symile_campaign, "validate_writable_file_destination", lambda *args: None)
    monkeypatch.setattr(symile_campaign, "validate_existing_sqlite_database", lambda *args: None)
    monkeypatch.setattr(symile_campaign, "_validate_symile_storage_readiness", lambda *args: None)
    config = load_symile_development_config(
        "configs/symile_cxr_densenet.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
    monkeypatch.setattr(
        symile_campaign, "load_symile_development_config", lambda path, **coordinates: config
    )
    monkeypatch.setattr(symile_campaign, "_preflight_symile_campaign", lambda *args, **kwargs: None)
    _forbid_campaign_training(monkeypatch)
    with pytest.raises(ManifestBuildError, match="already open"):
        symile_campaign.run_symile_campaign(
            source_root=tmp_path / "source",
            backup_root=tmp_path.parent / f"{tmp_path.name}-backup",
            bundle_id="bundle-" + "a" * 64,
            cv_assignment_id="cv-assignment-" + "c" * 64,
        )


def test_pretest_resolution_rejects_dangling_freeze_root(tmp_path: Path) -> None:
    control_root = tmp_path / "control"
    control_root.mkdir()
    arguments = {
        "control_root": control_root,
        "model_root": tmp_path / "models",
        "report_root": tmp_path / "reports",
        "private_root": tmp_path / "private",
        "manifest_root": tmp_path / "manifests",
    }
    assert symile_campaign._resolve_unique_pretest_state(**arguments) is None
    (control_root / "freezes").symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(ManifestBuildError, match="freeze root is invalid"):
        symile_campaign._resolve_unique_pretest_state(**arguments)


def test_export_projection_selects_only_explicit_authorities(tmp_path: Path):
    def artifact(relative):
        path = tmp_path / relative
        path.mkdir(parents=True, exist_ok=True)
        (path / "evidence.json").write_text("{}\n")
        return path

    manifests, models = tmp_path / "data/manifests", tmp_path / "models/symile"
    reports, private = tmp_path / "reports/symile", tmp_path / "private"
    artifact("data/manifests/symile/bundles/bundle-a")
    artifact("data/manifests/symile/cv/cv-a")
    development = SimpleNamespace(
        manifest={
            "development_id": "development-a",
            "scientific_context": {"cv_assignment_id": "cv-a"},
            "fold_packages": [{"fold_package_id": "fold-a", "prediction_id": "prediction-a"}],
        }
    )
    artifact("reports/symile/development/families/development-a")
    artifact("models/symile/development/packages/fold-a")
    artifact("private/predictions/symile/oof/prediction-a")
    freeze = SimpleNamespace(
        manifest={"bundle_id": "bundle-a"},
        freeze_id="freeze-a",
        directory=artifact("private/control/symile/freezes/freeze-a"),
    )
    execution = SimpleNamespace(
        execution_id="execution-a",
        directory=artifact("private/control/symile/executions/execution-a"),
    )
    decision = SimpleNamespace(
        decision_id="decision-a",
        directory=artifact("reports/symile/development-decisions/decision-a"),
    )
    package = SimpleNamespace(
        package_id="package-a", directory=artifact("models/symile/final/packages/package-a")
    )
    prediction = SimpleNamespace(
        prediction_id="prediction-b",
        directory=artifact("private/predictions/symile/test/prediction-b"),
    )
    result = SimpleNamespace(
        result_id="result-a", directory=artifact("reports/symile/global-results/result-a")
    )
    opening = private / "control/symile/test-open.json"
    opening.write_text("{}\n")
    (opening.parent / "active-execution.json").write_text("{}\n")
    error = artifact("private/error-review/symile") / "error-review.json"
    error.write_text("{}\n")
    analysis = artifact("reports/symile/development/analyses/core")
    subgroup = artifact("reports/symile/development-subgroups") / "focused.json"
    subgroup.write_text("{}\n")
    unrelated = artifact("reports/symile/unselected")
    members = symile_campaign._project_campaign_export_members(
        freeze=freeze,
        execution=execution,
        opening_path=opening,
        development_decision=decision,
        developments=(development,),
        packages=(package,),
        predictions=(prediction,),
        global_result=result,
        error_review=error,
        manifest_root=manifests,
        model_root=models,
        report_root=reports,
        private_root=private,
        analysis_path=analysis,
        subgroup_path=subgroup,
    )
    paths = {member.path for member in members}
    assert package.directory in paths and prediction.directory in paths
    assert unrelated not in paths
    assert len(paths) == len(members)


def test_symile_archive_name_is_specification_stable() -> None:
    arguments = {
        "global_result_id": "global-result-" + "a" * 64,
        "execution_id": "symile-execution-" + "b" * 64,
    }
    expected = "symile-global-result-" + "a" * 64 + "-symile-execution-" + "b" * 64

    assert symile_campaign._symile_export_name(**arguments) == expected
    assert f"{symile_campaign._symile_export_name(**arguments)}.zip" == f"{expected}.zip"


def test_restoration_rejects_freeze_path_before_reading_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "restored"
    opening = root / "private/control/symile/test-open.json"
    opening.parent.mkdir(parents=True)
    opening.write_text(json.dumps({"freeze_id": "../../outside"}), encoding="utf-8")
    monkeypatch.setattr(
        symile_campaign,
        "_validated_restoration_context",
        lambda _: SimpleNamespace(root=root),
    )
    original_read = Path.read_bytes

    def bounded_read(path: Path) -> bytes:
        assert path == opening, "Unvalidated freeze coordinate was used for filesystem access"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", bounded_read)
    with pytest.raises(ValueError, match="safe path component"):
        symile_campaign.validate_restored_campaign(root)
