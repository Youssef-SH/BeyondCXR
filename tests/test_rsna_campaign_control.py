from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

import beyondcxr.training.rsna_campaign_control as control
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.models.cxr_baseline import PretrainedWeightIdentity
from beyondcxr.training.config import load_experiment_config, with_runtime
from beyondcxr.training.device import ResolvedDevice
from beyondcxr.training.rsna_campaign_control import (
    FrozenScientificPackage,
    ValidatedRsnaExecution,
    authorize_rsna_held_out,
    publish_evaluation_record,
    publish_or_validate_execution,
    publish_or_validate_freeze,
    require_validated_freeze,
    require_validated_rsna_held_out,
    validate_current_rsna_execution,
    validate_execution,
    validate_execution_closure,
    validate_freeze,
)
from beyondcxr.training.rsna_datasets import RsnaDataset
from beyondcxr.training.rsna_evaluation_result import CompletedRsnaEvaluation
from beyondcxr.training.rsna_formal import (
    _FORMAL_PLAN_GUARD,
    RSNA_FORMAL_RUN_PLAN,
    RsnaAuthorityCoordinates,
    RsnaFormalRoots,
    ValidatedRsnaPlan,
    rsna_run_report_root,
)
from beyondcxr.training.rsna_training_report import (
    REQUIRED_REPORT_FILENAMES,
)
from beyondcxr.training.rsna_validation_evidence import (
    CanonicalValidationCohort,
)
from beyondcxr.utils.package_identity import package_scientific_config_payload
from beyondcxr.utils.rsna_model_publication import threshold_contract
from beyondcxr.utils.semantic_identity import canonical_json_bytes


def _configs():
    return tuple(
        with_runtime(
            load_experiment_config(spec.config_relative, bundle_id="bundle-" + "a" * 64),
            seed=spec.seed,
        )
        for spec in RSNA_FORMAL_RUN_PLAN
    )


def _plan(tmp_path: Path) -> ValidatedRsnaPlan:
    configs = tuple(
        with_runtime(config, manifest_directory=tmp_path / "manifests") for config in _configs()
    )
    first = configs[0]
    roots = RsnaFormalRoots(
        tmp_path,
        tmp_path / "source",
        tmp_path / "manifests",
        tmp_path / "cache",
        tmp_path / "models/rsna",
        tmp_path / "reports",
        tmp_path / "private",
        tmp_path / "control",
        tmp_path / "outbox",
        tmp_path / "backup",
        tmp_path / "mlflow.db",
    )
    runtime = ResolvedDevice(
        torch.device("cuda"),
        "cuda",
        "cuda",
        True,
        True,
        True,
        "auto",
        True,
        "torch",
        "torchvision",
        "xrv",
        "12.0",
        9000,
        "GPU",
        0,
        (8, 0),
    )
    return ValidatedRsnaPlan(
        RsnaAuthorityCoordinates(
            "rsna",
            first.dataset.bundle_id,
            first.task.task_id,
            first.task.label_policy_version,
            tmp_path / "manifests" / "rsna" / "bundles" / first.dataset.bundle_id,
        ),
        roots,
        RSNA_FORMAL_RUN_PLAN,
        configs,
        RsnaDataset(),
        "f" * 40,
        "0" * 64,
        PretrainedWeightIdentity("densenet121-res224-chex", "url", "weights.bin", 1, "1" * 64),
        runtime,
        SimpleNamespace(
            provenance=lambda: {
                "lifecycle": "reused",
                "num_workers": 2,
                "pin_memory": True,
            },
            prefetch_factor=2,
        ),
        SimpleNamespace(
            provenance=lambda: {
                "lifecycle": "one_shot",
                "num_workers": 0,
                "pin_memory": True,
            },
            prefetch_factor=None,
        ),
        _FORMAL_PLAN_GUARD,
    )


def test_completed_fit_resolution_rejects_dangling_package_root(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    execution = publish_or_validate_execution(plan)
    assert control.resolve_completed_rsna_packages(plan, execution) == (None,) * 8
    root = plan.roots.model_root / "packages"
    root.parent.mkdir(parents=True)
    root.symlink_to(tmp_path / "absent-packages", target_is_directory=True)
    with pytest.raises(ManifestBuildError, match="package root is invalid"):
        control.resolve_completed_rsna_packages(plan, execution)


def test_execution_control_freezes_the_exact_ordered_package_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(control, "validate_training_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        control,
        "load_canonical_validation_cohort",
        lambda *args, **kwargs: CanonicalValidationCohort(("sample",), (0,)),
    )
    monkeypatch.setattr(
        control,
        "validate_package_validation_cohort",
        lambda *args, **kwargs: None,
    )
    plan = _plan(tmp_path)
    configs = plan.configs
    execution = publish_or_validate_execution(plan)
    model_root = tmp_path / "models/rsna"
    report_root = tmp_path / "reports"
    package_manifests = {}
    results = []
    for index, config in enumerate(configs):
        package_id = "model-package-" + f"{index + 1:x}" * 64
        package = model_root / "packages" / package_id
        package.mkdir(parents=True)
        model_name = "model.skops" if index < 2 else "model.pt"
        model_path = package / model_name
        model_path.write_bytes(b"model")
        report = rsna_run_report_root(report_root) / f"run-{index}"
        report.mkdir(parents=True)
        for name in REQUIRED_REPORT_FILENAMES:
            (report / name).write_bytes(b"report")
        (report / "metrics.json").write_text(
            json.dumps(
                {
                    "training_package": {
                        "run_id": f"run-{index}",
                        "model_package_id": package_id,
                        "family_id": config.family.family_id,
                        "seed": config.runtime.seed,
                    }
                }
            ),
            encoding="utf-8",
        )
        package_manifests[package_id] = {
            "dataset_id": config.dataset.dataset_id,
            "bundle_id": config.dataset.bundle_id,
            "task_id": config.task.task_id,
            "label_policy_version": config.task.label_policy_version,
            "family_id": config.family.family_id,
            "modalities": list(config.family.modalities),
            "fit_config": package_scientific_config_payload(config),
            "seed": config.runtime.seed,
            "positive_class": 1,
            "threshold_contract": threshold_contract(
                sensitivity_target=config.evaluation.sensitivity_target
            ),
            **(
                {
                    "model_identity": {"pretrained_weight": plan.pretrained_weight.as_dict()},
                    "runtime_provenance": {
                        **plan.runtime.provenance(),
                        "loader_execution": {
                            **plan.training_execution.provenance(),
                            "prefetch_factor": plan.training_execution.prefetch_factor,
                        },
                    },
                    "source_provenance": {
                        "git_commit": plan.git_commit,
                        "git_dirty": False,
                        "dependency_lock_sha256": plan.dependency_lock_sha256,
                        "python_version": execution.manifest["framework_versions"]["python"],
                        "torch_version": execution.manifest["framework_versions"]["torch"],
                        "torchvision_version": plan.runtime.torchvision_version,
                        "torchxrayvision_version": plan.runtime.torchxrayvision_version,
                    },
                }
                if config.neural is not None
                else {
                    "git_commit": plan.git_commit,
                    "git_dirty": False,
                    "dependency_lock_sha256": plan.dependency_lock_sha256,
                    "python_version": execution.manifest["framework_versions"]["python"],
                }
            ),
        }
        results.append(
            SimpleNamespace(
                run_id=f"run-{index}",
                model_package_id=package_id,
                model_path=model_path,
                artifact_directory=report,
            )
        )
    monkeypatch.setattr(
        control,
        "validate_rsna_model_package",
        lambda root, package_id: package_manifests[package_id],
    )

    for index in range(5, 8):
        package_manifests[results[index].model_package_id]["source_package_id"] = results[
            index - 3
        ].model_package_id
    resolved = control.resolve_completed_rsna_packages(plan, execution)
    assert tuple(item.model_package_id for item in resolved) == tuple(
        item.model_package_id for item in results
    )
    missing = results[1].model_path.parent
    unpublished = missing.with_name(f".{missing.name}-staging-interrupted")
    missing.rename(unpublished)
    try:
        partial = control.resolve_completed_rsna_packages(plan, execution)
        assert partial[1] is None
        assert partial[:1] + partial[2:] == resolved[:1] + resolved[2:]
    finally:
        unpublished.rename(missing)
    package = package_manifests[results[0].model_package_id]
    for field, value in (
        ("fit_config", {}),
        ("dependency_lock_sha256", "9" * 64),
    ):
        with monkeypatch.context() as mutation:
            mutation.setitem(package, field, value)
            with pytest.raises(ManifestBuildError, match="differs from the formal plan"):
                control.resolve_completed_rsna_packages(plan, execution)
    with monkeypatch.context() as mutation:
        mutation.setitem(
            package_manifests[results[5].model_package_id],
            "source_package_id",
            results[3].model_package_id,
        )
        with pytest.raises(ManifestBuildError, match="conflicting CXR membership"):
            control.resolve_completed_rsna_packages(plan, execution)
    duplicate = model_root / "packages" / ("model-package-" + "a" * 64)
    duplicate.mkdir()
    (duplicate / "model.skops").write_bytes(b"another fit")
    package_manifests[duplicate.name] = dict(package)
    try:
        with pytest.raises(ManifestBuildError, match="ambiguous"):
            control.resolve_completed_rsna_packages(plan, execution)
    finally:
        shutil.rmtree(duplicate)
        del package_manifests[duplicate.name]

    frozen = publish_or_validate_freeze(
        plan=plan,
        execution=execution,
        results=results,
    )

    assert tuple(item.model_package_id for item in frozen.packages) == tuple(
        item.model_package_id for item in results
    )
    assert require_validated_freeze(frozen) is frozen
    unexpected = frozen.directory / "unexpected.txt"
    unexpected.write_text("not a freeze member\n", encoding="utf-8")
    try:
        with pytest.raises(ManifestBuildError, match="freeze directory membership"):
            require_validated_freeze(frozen)
    finally:
        unexpected.unlink()

    report = results[0].artifact_directory
    report_bytes = (report / "evaluation_report.md").read_bytes()
    (report / "evaluation_report.md").write_bytes(report_bytes + b"tampered")
    assert frozen.require(frozen.packages[0].model_package_id) == frozen.packages[0]
    materialized = False

    def load_pinned_bundle(*args, **kwargs):
        nonlocal materialized
        materialized = True
        raise AssertionError("held-out materialization must not begin")

    monkeypatch.setattr("beyondcxr.training.rsna_datasets._load_pinned_bundle", load_pinned_bundle)
    with pytest.raises(PermissionError, match="current-execution capability"):
        RsnaDataset().load_test(
            configs[0],
            authorization=frozen,
            package_id=frozen.packages[0].model_package_id,
        )
    assert materialized is False
    (report / "evaluation_report.md").write_bytes(report_bytes)
    held_out = authorize_rsna_held_out(plan=plan, execution=execution, freeze=frozen)
    calls = []

    def preflight(*, backup_root, bundle_id, source_root="data/raw/rsna/extracted"):
        assert bundle_id == plan.authority.bundle_id
        assert backup_root == plan.roots.backup_root
        assert source_root == plan.roots.source_root
        calls.append(bundle_id)
        return plan

    monkeypatch.setattr("beyondcxr.training.rsna_campaign._preflight_rsna_campaign", preflight)
    assert require_validated_rsna_held_out(held_out).freeze == frozen
    assert calls == [plan.authority.bundle_id]

    with monkeypatch.context() as drift:

        def drifted_preflight(*, backup_root, bundle_id, source_root="data/raw/rsna/extracted"):
            preflight(backup_root=backup_root, bundle_id=bundle_id, source_root=source_root)
            return replace(plan, git_commit="9" * 40)

        drift.setattr(
            "beyondcxr.training.rsna_campaign._preflight_rsna_campaign", drifted_preflight
        )
        with pytest.raises(ManifestBuildError, match="Current environment differs"):
            require_validated_rsna_held_out(held_out)

    extra_freeze = frozen.directory.parent / ("rsna-freeze-" + "9" * 64)
    shutil.copytree(frozen.directory, extra_freeze)
    with pytest.raises(ManifestBuildError, match="current formal execution plan"):
        require_validated_rsna_held_out(held_out)
    shutil.rmtree(extra_freeze)

    receipt = plan.roots.control_root / "active-execution.json"
    receipt_bytes = receipt.read_bytes()
    receipt.write_bytes(
        canonical_json_bytes({"schema_version": 1, "execution_id": "rsna-execution-" + "9" * 64})
    )
    with pytest.raises(ManifestBuildError, match="active execution receipt"):
        require_validated_rsna_held_out(held_out)
    receipt.write_bytes(receipt_bytes)

    freeze_manifest = frozen.directory / "manifest.json"
    freeze_bytes = freeze_manifest.read_bytes()
    freeze_manifest.write_bytes(freeze_bytes + b"tampered")
    with pytest.raises(ManifestBuildError, match="freeze"):
        require_validated_rsna_held_out(held_out)
    freeze_manifest.write_bytes(freeze_bytes)

    moved_directory = tmp_path / "unrelated-freezes" / frozen.freeze_id
    shutil.copytree(frozen.directory, moved_directory)
    moved_freeze = replace(frozen, directory=moved_directory)
    assert require_validated_freeze(moved_freeze) == moved_freeze
    with pytest.raises(ManifestBuildError, match="current formal execution plan"):
        authorize_rsna_held_out(plan=plan, execution=execution, freeze=moved_freeze)
    with pytest.raises(AssertionError, match="materialization must not begin"):
        RsnaDataset().load_test(
            configs[0],
            authorization=held_out,
            package_id=frozen.packages[0].model_package_id,
        )

    package_manifests[frozen.packages[0].model_package_id]["seed"] = 7
    with pytest.raises(ManifestBuildError, match="scientific freeze"):
        frozen.require(frozen.packages[0].model_package_id)
    package_manifests[frozen.packages[0].model_package_id]["seed"] = 42

    forged_member = FrozenScientificPackage(
        "model-package-" + "9" * 64,
        frozen.packages[0].model_path,
    )
    evaluations = execution.directory.parent.parent / "evaluations" / execution.execution_id
    evaluations.mkdir(parents=True)
    for member in (*frozen.packages, forged_member):
        (evaluations / f"{member.model_package_id}.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="canonical authority"):
        validate_execution_closure(
            execution,
            replace(frozen, packages=(*frozen.packages, forged_member)),
        )
    (evaluations / f"{forged_member.model_package_id}.json").unlink()
    validate_execution_closure(execution, frozen)
    for parent in (execution.directory, evaluations):
        unexpected = parent / "unexpected.json"
        unexpected.write_text("{}\n", encoding="utf-8")
        try:
            with pytest.raises(ManifestBuildError, match="membership"):
                validate_execution_closure(execution, frozen)
        finally:
            unexpected.unlink()
    moved_evaluations = tmp_path / "moved-evaluations"
    evaluations.rename(moved_evaluations)
    evaluations.symlink_to(moved_evaluations, target_is_directory=True)
    try:
        with pytest.raises(ManifestBuildError, match="records directory is invalid"):
            validate_execution_closure(execution, frozen)
    finally:
        evaluations.unlink()
        moved_evaluations.rename(evaluations)

    substituted = replace(
        frozen.packages[0],
        model_package_id="model-package-" + "f" * 64,
    )
    with pytest.raises(ManifestBuildError, match="differs from its canonical authority"):
        require_validated_freeze(replace(frozen, packages=(substituted, *frozen.packages[1:])))
    with pytest.raises(ManifestBuildError, match="differs from its canonical authority"):
        require_validated_freeze(
            replace(
                frozen,
                packages=(frozen.packages[1], frozen.packages[0], *frozen.packages[2:]),
            )
        )

    freeze_path = frozen.directory / "manifest.json"
    canonical_freeze = freeze_path.read_bytes()
    freeze_document = json.loads(canonical_freeze)
    freeze_document["packages"][0]["family_id"] = "tampered-family"
    freeze_path.write_text(
        json.dumps(freeze_document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestBuildError, match="scientific freeze"):
        require_validated_freeze(frozen)
    freeze_path.write_bytes(canonical_freeze)

    other_plan = replace(plan, git_commit="e" * 40)
    assert other_plan.git_commit != plan.git_commit
    assert (
        validate_freeze(
            frozen.directory,
            model_root=model_root,
            bundle_directory=plan.authority.bundle_directory,
        ).freeze_id
        == frozen.freeze_id
    )

    package_manifests[results[2].model_package_id]["seed"] = 42
    with pytest.raises(ManifestBuildError, match="scientific freeze"):
        validate_freeze(
            frozen.directory,
            model_root=model_root,
            bundle_directory=plan.authority.bundle_directory,
        )
    package_manifests[results[2].model_package_id]["seed"] = 17
    report_manifest = results[2].artifact_directory / "metrics.json"
    report_document = json.loads(report_manifest.read_text(encoding="utf-8"))
    report_document["training_package"]["model_package_id"] = "model-package-wrong"
    report_manifest.write_text(json.dumps(report_document), encoding="utf-8")
    assert (
        validate_freeze(
            frozen.directory,
            model_root=model_root,
            bundle_directory=plan.authority.bundle_directory,
        ).freeze_id
        == frozen.freeze_id
    )

    misplaced_validation = report_root / "some-other-place" / results[0].run_id
    shutil.copytree(results[0].artifact_directory, misplaced_validation)
    misplaced = report_root / "some-other-place" / results[0].run_id
    shutil.rmtree(misplaced_validation)
    results[0].artifact_directory.rename(misplaced)
    misplaced_result = SimpleNamespace(**{**results[0].__dict__, "artifact_directory": misplaced})
    assert (
        publish_or_validate_freeze(
            plan=plan,
            execution=execution,
            results=(misplaced_result, *results[1:]),
        ).freeze_id
        == frozen.freeze_id
    )


def test_scientific_freeze_rejects_unknown_fields(tmp_path: Path) -> None:
    directory = tmp_path / "rsna-freeze-invalid"
    directory.mkdir()
    (directory / "manifest.json").write_bytes(
        canonical_json_bytes({"schema_version": 1, "operational_path": "/forbidden"})
    )

    with pytest.raises(ManifestBuildError, match="scientific freeze contract"):
        validate_freeze(
            directory,
            model_root=tmp_path / "models",
            bundle_directory=tmp_path / "bundle",
        )


def test_rsna_execution_schema_is_v1_and_rejects_wrong_version(tmp_path: Path) -> None:
    execution = publish_or_validate_execution(_plan(tmp_path))

    assert control.RSNA_EXECUTION_SCHEMA_VERSION == 1
    assert execution.manifest["rsna_execution_schema_version"] == 1
    assert "freeze_id" not in execution.manifest

    path = execution.directory / "manifest.json"
    document = json.loads(path.read_bytes())
    document["rsna_execution_schema_version"] = 2
    path.write_bytes(canonical_json_bytes(document))

    with pytest.raises(ManifestBuildError, match="execution control contract"):
        validate_execution(execution.directory)


def test_rsna_execution_identity_changes_with_config_and_pretrained_science(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    baseline = control._execution_document(plan)["execution_id"]
    first = plan.configs[0]
    changed_config = replace(
        first,
        training=replace(
            first.training,
            parameters=MappingProxyType(
                {**first.training.parameters, "max_iter": first.training.parameters["max_iter"] + 1}
            ),
        ),
    )
    config_plan = replace(plan, configs=(changed_config, *plan.configs[1:]))
    assert control._execution_document(config_plan)["execution_id"] != baseline

    pretrained_plan = replace(
        plan,
        pretrained_weight=PretrainedWeightIdentity(
            plan.pretrained_weight.declared_name,
            plan.pretrained_weight.stable_identifier,
            "relocated.bin",
            plan.pretrained_weight.byte_size,
            "9" * 64,
        ),
    )
    assert control._execution_document(pretrained_plan)["execution_id"] != baseline


def test_scientific_freeze_rejects_a_valid_execution_from_a_different_plan(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    execution = publish_or_validate_execution(plan)
    different_plan = replace(plan, git_commit="e" * 40)

    with pytest.raises(ManifestBuildError, match="does not match the validated plan"):
        publish_or_validate_freeze(
            plan=different_plan,
            execution=execution,
            results=(),
        )


def test_incompatible_execution_cannot_adopt_partial_campaign_state(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    publish_or_validate_execution(plan)
    with pytest.raises(ManifestBuildError, match="active under another execution"):
        publish_or_validate_execution(replace(plan, git_commit="e" * 40))


def test_historical_execution_directory_does_not_select_active_campaign(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "campaign")
    active = publish_or_validate_execution(plan)
    historical_plan = replace(_plan(tmp_path / "history"), git_commit="e" * 40)
    historical = publish_or_validate_execution(historical_plan)
    shutil.copytree(
        historical.directory,
        active.directory.parent / historical.execution_id,
    )
    assert publish_or_validate_execution(plan) == active


def test_scientific_freeze_rejects_forged_and_tampered_executions(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    execution = publish_or_validate_execution(plan)
    forged = ValidatedRsnaExecution(
        execution.execution_id,
        execution.directory,
        execution.manifest,
        object(),
    )
    with pytest.raises(PermissionError, match="validated execution"):
        publish_or_validate_freeze(plan=plan, execution=forged, results=())

    tampered = replace(
        execution,
        manifest={**execution.manifest, "git_commit": "e" * 40},
    )
    with pytest.raises(ManifestBuildError, match="canonical authority"):
        validate_current_rsna_execution(plan, tampered)


def test_execution_records_are_append_only_and_completion_requires_evaluation_authority(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    execution = publish_or_validate_execution(plan)
    assert validate_execution(execution.directory) == execution
    path = execution.directory / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["configs"][0]["seed"] = 17
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ManifestBuildError):
        validate_execution(execution.directory)

    # Restore the exact control, then prove a dataclass alone cannot publish completion.
    path.unlink()
    execution = publish_or_validate_execution(plan)
    first = CompletedRsnaEvaluation(
        evaluation_id="evaluation-" + "1" * 64,
        prediction_id="prediction-" + "2" * 64,
        model_package_id="model-package-" + "3" * 64,
        mlflow_run_id="run-1",
        artifact_directory=tmp_path / "evaluation",
        private_prediction_directory=tmp_path / "prediction",
        average_precision=0.5,
    )
    with pytest.raises((OSError, ValueError, PermissionError, ManifestBuildError)):
        publish_evaluation_record(
            execution,
            first,
            freeze=object(),
            report_root=tmp_path / "reports",
            private_root=tmp_path / "private",
            model_root=tmp_path / "models/rsna",
        )
    assert {item.name for item in execution.directory.iterdir()} == {"manifest.json"}


def test_execution_control_rejects_a_forged_formal_plan_before_publication(
    tmp_path: Path,
) -> None:
    forged = replace(_plan(tmp_path), _guard=object())

    with pytest.raises(ValueError, match="genuine prevalidated"):
        publish_or_validate_execution(forged)

    assert not (tmp_path / "control").exists()
