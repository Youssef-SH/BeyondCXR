from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
import yaml
from symile_data_test_support import _published_synthetic_release

import beyondcxr.training.symile_campaign_control as control
import beyondcxr.training.symile_development as development
import beyondcxr.training.symile_development_decision as development_decision
import beyondcxr.training.symile_execution as execution_control
import beyondcxr.training.symile_final_packages as final_packages
import beyondcxr.training.symile_final_training as final_training
import beyondcxr.utils.private_predictions as private_predictions
import beyondcxr.utils.symile_publication as development_publication
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.hashing import logical_table_sha256
from beyondcxr.data.symile_cv import (
    _CV_TABLE_CONTRACT,
    CV_ASSIGNMENTS_FILENAME,
    _cv_manifest,
    cv_assignment_id,
    publish_symile_cv,
    validate_symile_cv_reference,
)
from beyondcxr.models.cxr_baseline import PretrainedWeightIdentity
from beyondcxr.training.config import (
    load_symile_development_config,
)
from beyondcxr.training.device import ResolvedDevice
from beyondcxr.training.execution import LoaderExecutionPolicy
from beyondcxr.training.symile_campaign_control import ValidatedPretestFreeze
from beyondcxr.training.symile_execution import (
    ValidatedSymileExecution,
    ValidatedSymileFormalPreflight,
    build_symile_formal_preflight,
    publish_or_validate_symile_execution,
    require_current_symile_execution,
    validate_active_symile_execution,
    validate_current_symile_execution,
    validate_symile_lifecycle,
)
from beyondcxr.training.symile_families import (
    FINAL_NEURAL_MEMBER_SEEDS,
    FINAL_PACKAGE_POLICY,
    FINAL_TABULAR_SEED,
)
from beyondcxr.training.symile_formal import SYMILE_FORMAL_RUN_PLAN
from beyondcxr.utils.semantic_identity import canonical_json_bytes, canonical_scientific_id


def _resolved_runtime(
    *,
    gpu_name: str = "GPU-A",
    cuda_runtime_version: str = "12.4",
    gpu_compute_capability: tuple[int, int] = (8, 6),
) -> ResolvedDevice:
    return ResolvedDevice(
        device=torch.device("cuda"),
        requested_device="cuda",
        resolved_device="cuda",
        cuda_available=True,
        mixed_precision_requested=True,
        mixed_precision_effective=True,
        pin_memory_requested="enabled",
        pin_memory_effective=True,
        torch_version="2.8.0",
        torchvision_version="0.23.0",
        torchxrayvision_version="1.3.5",
        cuda_runtime_version=cuda_runtime_version,
        cudnn_version=9100,
        gpu_device_name=gpu_name,
        gpu_device_index=0,
        gpu_compute_capability=gpu_compute_capability,
    )


def _preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    git: str = "a" * 40,
    lock: str = "d" * 64,
    cache_filename: str = "weights.pt",
    weight_size: int = 10,
    weight_sha: str = "e" * 64,
    gpu_name: str = "GPU-A",
    cuda_runtime_version: str = "12.4",
    gpu_compute_capability: tuple[int, int] = (8, 6),
    python_version: str = "3.13.7",
    logistic_c: float | None = None,
    training_workers: int = 2,
) -> ValidatedSymileFormalPreflight:
    repository = tmp_path / "project"
    config_root = repository / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    for spec in SYMILE_FORMAL_RUN_PLAN:
        shutil.copyfile(spec.config_relative, repository / spec.config_relative)
    if logistic_c is not None:
        path = repository / "configs/symile_labs_logistic.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["training"]["parameters"]["C"] = logistic_c
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(execution_control, "git_revision", lambda root: (git, False))
    monkeypatch.setattr(execution_control, "uv_lock_sha256", lambda path: lock)
    monkeypatch.setattr(
        execution_control,
        "resolve_symile_bundle",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        execution_control,
        "validate_symile_cv",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(execution_control.platform, "python_version", lambda: python_version)
    monkeypatch.setattr(
        execution_control,
        "resolve_device",
        lambda *args, **kwargs: _resolved_runtime(
            gpu_name=gpu_name,
            cuda_runtime_version=cuda_runtime_version,
            gpu_compute_capability=gpu_compute_capability,
        ),
    )
    monkeypatch.setattr(
        execution_control,
        "fingerprint_pretrained_weights",
        lambda name: PretrainedWeightIdentity(
            name, "weights", cache_filename, weight_size, weight_sha
        ),
    )
    monkeypatch.setattr(
        execution_control,
        "reused_loader_policy",
        lambda *, num_workers, pin_memory: LoaderExecutionPolicy(
            "reused", training_workers, pin_memory
        ),
    )
    return build_symile_formal_preflight(
        repository, bundle_id="bundle-" + "a" * 64, cv_assignment_id="cv-assignment-" + "c" * 64
    )


def _execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **kwargs: object,
) -> ValidatedSymileExecution:
    return publish_or_validate_symile_execution(_preflight(tmp_path, monkeypatch, **kwargs))


def _lifecycle(execution: ValidatedSymileExecution):
    control = execution.directory.parent.parent
    repository = control.parents[2]
    backup = repository.parent / f"{repository.name}-backup"
    backup.mkdir(parents=True, exist_ok=True)
    current = validate_current_symile_execution(
        execution,
        build_symile_formal_preflight(
            repository, bundle_id="bundle-" + "a" * 64, cv_assignment_id="cv-assignment-" + "c" * 64
        ),
    )
    return validate_symile_lifecycle(
        current,
        repository_root=repository,
        private_root=repository / "private",
        backup_root=backup,
    )


def _freeze(
    tmp_path: Path,
    *,
    suffix: str = "d",
    bundle_id: str | None = None,
    cv_assignment_id: str | None = None,
) -> ValidatedPretestFreeze:
    reference = load_symile_development_config(
        "configs/symile_labs_logistic.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
    packages = []
    index = 0
    for family, policy in FINAL_PACKAGE_POLICY.items():
        seeds = (FINAL_TABULAR_SEED,) if int(policy["members"]) == 1 else FINAL_NEURAL_MEMBER_SEEDS
        for seed in seeds:
            packages.append(
                {
                    "package_id": "final-package-" + f"{index:064x}",
                    "family_id": family,
                    "seed": seed,
                }
            )
            index += 1
    document = control.pretest_freeze_document(
        bundle_id=bundle_id or reference.dataset.bundle_id,
        cv_assignment_id=cv_assignment_id or str(reference.dataset.cv_assignment_id),
        task={"task_id": "pneumonia_strict", "label_policy_version": "symile-pneumonia-strict-v1"},
        development_decision_id="development-decision-" + suffix * 64,
        package_members=packages,
    )
    directory = tmp_path / document["pretest_freeze_id"]
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_bytes(canonical_json_bytes(document))
    return ValidatedPretestFreeze(  # type: ignore[arg-type]
        directory, document, None, (), control._FREEZE_GUARD
    )


def _canonical_freeze(
    execution: ValidatedSymileExecution, *, suffix: str = "d"
) -> ValidatedPretestFreeze:
    return _freeze(execution.directory.parent.parent / "freezes", suffix=suffix)


def _permit_synthetic_authorities(
    monkeypatch: pytest.MonkeyPatch,
    freeze: ValidatedPretestFreeze,
) -> None:
    monkeypatch.setattr(control, "_require_capability", lambda value: freeze)
    monkeypatch.setattr(control, "validate_symile_execution_freeze_science", lambda *a, **k: None)


def test_freeze_document_contains_science_only(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path)
    _, _, document = control._validate_pretest_freeze_document(freeze.directory)
    assert not {
        "execution_id",
        "git_commit",
        "dependency_lock_sha256",
        "neural_inference_runtime",
        "primary_thresholds",
    } & set(document)


@pytest.mark.parametrize("mutation", ["boolean_schema", "float_schema", "metric_policy"])
def test_freeze_rejects_invalid_schema_or_scientific_policy(tmp_path: Path, mutation: str) -> None:
    freeze = _freeze(tmp_path)
    document = copy.deepcopy(freeze.manifest)
    if mutation == "metric_policy":
        document["held_out_policy"]["metric_policy"]["primary_metric"] = "average_precision"
    else:
        document["pretest_freeze_schema_version"] = True if mutation == "boolean_schema" else 1.0
    (freeze.directory / "manifest.json").write_bytes(canonical_json_bytes(document))
    with pytest.raises(ManifestBuildError, match="contract|fields"):
        control._validate_pretest_freeze_document(freeze.directory)


def test_freeze_rejects_symlink_manifest(tmp_path: Path) -> None:
    freeze = _freeze(tmp_path)
    manifest = freeze.directory / "manifest.json"
    external = tmp_path / "external-manifest.json"
    manifest.rename(external)
    manifest.symlink_to(external)

    with pytest.raises(ManifestBuildError, match="directory is invalid"):
        control._validate_pretest_freeze_document(freeze.directory)


def test_symile_execution_schema_is_v1_and_rejects_wrong_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution(tmp_path, monkeypatch)

    assert execution_control.SYMILE_EXECUTION_SCHEMA_VERSION == 1
    assert execution.manifest["schema_version"] == 1
    assert "freeze_id" not in execution.manifest

    path = execution.directory / "manifest.json"
    document = json.loads(path.read_bytes())
    document["schema_version"] = 2
    path.write_bytes(canonical_json_bytes(document))

    with pytest.raises(ManifestBuildError, match="execution record contract"):
        execution_control.validate_symile_execution(execution.directory)


def test_execution_identity_changes_with_operational_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _execution(tmp_path / "first", monkeypatch, git="a" * 40)
    second = _execution(tmp_path / "second", monkeypatch, git="b" * 40)
    assert first.execution_id != second.execution_id
    assert "freeze_id" not in first.manifest


def test_execution_identity_excludes_pretrained_cache_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _execution(tmp_path / "first", monkeypatch, cache_filename="weights.pt")
    second = _execution(tmp_path / "second", monkeypatch, cache_filename="renamed-chexnet.pt")
    assert first.execution_id == second.execution_id


def test_current_execution_accepts_fresh_matching_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    current = validate_current_symile_execution(
        execution,
        _preflight(tmp_path / "execution", monkeypatch),
    )
    assert current.execution == execution


def test_require_current_execution_rebuilds_matching_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    preflight = _preflight(tmp_path / "execution", monkeypatch)
    current = validate_current_symile_execution(execution, preflight)
    monkeypatch.setattr(
        execution_control, "build_symile_formal_preflight", lambda root, **coordinates: preflight
    )

    assert require_current_symile_execution(current) == current


@pytest.mark.parametrize(
    "changed",
    [
        {"git": "9" * 40},
        {"lock": "9" * 64},
        {"logistic_c": 0.75},
        {"weight_sha": "9" * 64},
    ],
)
def test_require_current_execution_rejects_fresh_environment_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: dict[str, object],
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    original = _preflight(tmp_path / "execution", monkeypatch)
    current = validate_current_symile_execution(execution, original)
    changed_preflight = _preflight(tmp_path / "execution", monkeypatch, **changed)
    monkeypatch.setattr(
        execution_control,
        "build_symile_formal_preflight",
        lambda root, **coordinates: changed_preflight,
    )

    with pytest.raises(ManifestBuildError, match="Current environment differs"):
        require_current_symile_execution(current)


@pytest.mark.parametrize("mutation", ["deleted", "replaced"])
def test_require_current_execution_rejects_active_receipt_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    preflight = _preflight(tmp_path / "execution", monkeypatch)
    current = validate_current_symile_execution(execution, preflight)
    receipt = current.control_root / "active-execution.json"
    if mutation == "deleted":
        receipt.unlink()
    else:
        receipt.write_bytes(
            canonical_json_bytes(
                {"schema_version": 1, "execution_id": "symile-execution-" + "9" * 64}
            )
        )
    monkeypatch.setattr(
        execution_control, "build_symile_formal_preflight", lambda root, **coordinates: preflight
    )

    with pytest.raises(ManifestBuildError, match="active execution receipt"):
        require_current_symile_execution(current)


def test_incompatible_execution_cannot_adopt_partial_campaign_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _execution(tmp_path, monkeypatch, git="a" * 40)
    with pytest.raises(ManifestBuildError, match="active under another execution"):
        _execution(tmp_path, monkeypatch, git="b" * 40)


def test_historical_execution_coexists_without_selecting_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active = _execution(tmp_path / "campaign", monkeypatch, git="a" * 40)
    historical = _execution(tmp_path / "history", monkeypatch, git="b" * 40)
    shutil.copytree(
        historical.directory,
        active.directory.parent / historical.execution_id,
    )
    assert validate_active_symile_execution(active.directory.parent.parent) == active


@pytest.mark.parametrize(
    "changed",
    [
        {"lock": "9" * 64},
        {"python_version": "3.13.8"},
        {"cuda_runtime_version": "12.5"},
        {"gpu_name": "GPU-B"},
        {"gpu_compute_capability": (9, 0)},
        {"weight_size": 11},
        {"weight_sha": "f" * 64},
        {"logistic_c": 0.75},
    ],
)
def test_execution_identity_changes_with_each_operational_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: dict[str, object],
) -> None:
    baseline = _execution(tmp_path / "baseline", monkeypatch)
    changed_execution = _execution(tmp_path / "changed", monkeypatch, **changed)
    assert changed_execution.execution_id != baseline.execution_id


@pytest.mark.parametrize("mutation", ["decision", "membership", "policy"])
def test_freeze_identity_changes_only_with_frozen_science(tmp_path: Path, mutation: str) -> None:
    freeze = _freeze(tmp_path)
    semantic = {
        key: copy.deepcopy(value)
        for key, value in freeze.manifest.items()
        if key not in {"pretest_freeze_schema_version", "pretest_freeze_id"}
    }
    if mutation == "decision":
        semantic["development_decision_id"] = "development-decision-" + "9" * 64
    elif mutation == "membership":
        semantic["final_packages"] = list(reversed(semantic["final_packages"]))
    else:
        semantic["held_out_policy"]["predictor_views"][2]["reduction"] = "different-policy"
    assert (
        canonical_scientific_id(
            control.PRETEST_FREEZE_PREFIX,
            semantic,
            domain="pretest-freeze",
            schema_version=1,
        )
        != freeze.freeze_id
    )


def test_test_open_creates_identical_local_and_external_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    freeze = _canonical_freeze(execution)
    _permit_synthetic_authorities(monkeypatch, freeze)
    lifecycle = _lifecycle(execution)
    frozen = control.validate_symile_frozen_campaign(freeze=freeze, lifecycle=lifecycle)
    capability = control.open_official_test(frozen_campaign=frozen)
    assert capability.opening.path.read_bytes() == capability.opening.external_path.read_bytes()
    assert set(json.loads(capability.opening.document_bytes)) == {
        "schema_version",
        "freeze_id",
        "execution_id",
        "opened_at_utc",
    }
    assert control.require_symile_frozen_campaign(frozen).freeze.freeze_id == freeze.freeze_id


def test_prospective_mutation_and_test_open_share_campaign_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    freeze = _canonical_freeze(execution)
    _permit_synthetic_authorities(monkeypatch, freeze)
    lifecycle = _lifecycle(execution)
    frozen = control.validate_symile_frozen_campaign(freeze=freeze, lifecycle=lifecycle)
    with execution_control.symile_prospective_mutation(lifecycle):
        with pytest.raises(ManifestBuildError, match="campaign is active"):
            control.open_official_test(frozen_campaign=frozen)
    assert not (lifecycle.control_root / "test-open.json").exists()


def test_prospective_mutation_rejects_manual_and_copied_lookalikes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    lifecycle = _lifecycle(execution)
    fake = object.__new__(execution_control.ValidatedSymileProspectiveMutation)
    object.__setattr__(fake, "lifecycle", lifecycle)
    with pytest.raises(PermissionError, match="live unopened authority"):
        execution_control.require_symile_prospective_mutation(fake)

    with execution_control.symile_prospective_mutation(lifecycle) as authority:
        assert execution_control.require_symile_prospective_mutation(authority) is authority
        copied = copy.copy(authority)
        with pytest.raises(PermissionError, match="live unopened authority"):
            execution_control.require_symile_prospective_mutation(copied)
    with pytest.raises(PermissionError, match="live unopened authority"):
        execution_control.require_symile_prospective_mutation(authority)


def test_all_prospective_publication_boundaries_reject_expired_authority_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    freeze = _canonical_freeze(execution)
    _permit_synthetic_authorities(monkeypatch, freeze)
    lifecycle = _lifecycle(execution)
    frozen = control.validate_symile_frozen_campaign(freeze=freeze, lifecycle=lifecycle)
    with execution_control.symile_prospective_mutation(lifecycle) as authority:
        expired = authority
    control.open_official_test(frozen_campaign=frozen)
    with pytest.raises(ManifestBuildError, match="already open"):
        with execution_control.symile_prospective_mutation(lifecycle):
            pass

    operations = {
        "development": lambda: development.run_symile_development(
            None, prospective_mutation=expired
        ),
        "outer-fold": lambda: development.execute_symile_outer_fold(
            None,
            None,
            prospective_mutation=expired,
            repeat_seed=17,
            outer_fold=0,
        ),
        "fold-package": lambda: development_publication.publish_fold_package(
            prospective_mutation=expired,
            model_root=tmp_path,
            family="labs_logistic",
            repeat_seed=17,
            outer_fold=0,
            config_bytes=b"",
            lineage={},
            inner_split={},
            selection={},
            model=None,
        ),
        "outer-fold-prediction": lambda: private_predictions.publish_prediction_evidence(
            prospective_mutation=expired,
            private_root=lifecycle.private_root,
            dataset_id="symile",
            model_package_id="fold-package-" + "0" * 64,
            task_id="pneumonia_strict",
            bundle_id="bundle-" + "0" * 64,
            scope="outer_fold_oof",
            sample_ids=("symile:1",),
            targets=(0,),
            logits=(0.0,),
            cv_assignment_id="cv-assignment-" + "0" * 64,
            repeat_seed=17,
            outer_fold=0,
        ),
        "family-development": lambda: development_publication.publish_development_result(
            prospective_mutation=expired,
            report_root=tmp_path,
            model_root=tmp_path,
            prediction_root=tmp_path,
            family="labs_logistic",
            folds=(),
            predictions=(),
        ),
        "development-decision": lambda: development_decision.publish_development_decision(
            prospective_mutation=expired,
            report_root=tmp_path,
            family_developments={},
        ),
        "final-tabular-fit": lambda: final_training.fit_final_tabular_family(
            None,
            prospective_mutation=expired,
            development=None,
            operational={},
        ),
        "final-neural-fit": lambda: final_training.fit_final_neural_member(
            None,
            prospective_mutation=expired,
            development=None,
            member_seed=17,
            expected_pretrained_scientific_identity=None,
            operational={},
        ),
        "final-tabular-package": lambda: final_packages.publish_final_tabular_package(
            prospective_mutation=expired,
            model_root=tmp_path,
            config=None,
            family_development_id="",
            plan=None,
            model=None,
            operational={},
        ),
        "final-neural-package": lambda: final_packages.publish_final_neural_package(
            prospective_mutation=expired,
            model_root=tmp_path,
            config=None,
            family_development_id="",
            plan=None,
            seed=17,
            state_dict={},
            lab_preprocessor=None,
            source_cxr_package_id=None,
            pretrained_weight=None,
            operational={},
        ),
        "pretest-freeze": lambda: control.publish_pretest_freeze(
            prospective_mutation=expired,
            control_root=lifecycle.control_root,
            bundle_id="bundle-" + "0" * 64,
            cv_assignment_id="cv-assignment-" + "0" * 64,
            task={},
            development_decision=None,
            final_packages=(),
        ),
    }
    for name, operation in operations.items():
        with pytest.raises(PermissionError, match="live unopened authority") as error:
            operation()
        assert name
        assert error.value


def test_formal_preflight_uses_complete_bundle_relative_cv_validation(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "project"
    manifest_root = repository / "data/manifests"
    _, bundle = _published_synthetic_release(repository / "data")
    _, valid_directory = publish_symile_cv(bundle, manifest_directory=manifest_root)
    assignments = pq.read_table(valid_directory / CV_ASSIGNMENTS_FILENAME).slice(1)
    logical_hash = logical_table_sha256(assignments, _CV_TABLE_CONTRACT)
    invalid_id = cv_assignment_id(bundle.bundle_id, logical_hash)
    invalid_directory = manifest_root / "symile/cv" / invalid_id
    invalid_directory.mkdir(parents=True)
    pq.write_table(assignments, invalid_directory / CV_ASSIGNMENTS_FILENAME)
    (invalid_directory / "manifest.json").write_bytes(
        canonical_json_bytes(_cv_manifest(bundle.bundle_id, invalid_id, assignments, logical_hash))
    )
    validate_symile_cv_reference(
        invalid_directory,
        bundle_id=bundle.bundle_id,
        expected_assignment_id=invalid_id,
    )
    for spec in SYMILE_FORMAL_RUN_PLAN:
        destination = repository / spec.config_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        document = yaml.safe_load(spec.config_relative.read_text(encoding="utf-8"))
        destination.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ManifestBuildError, match="cover every eligible sample"):
        build_symile_formal_preflight(
            repository, bundle_id=bundle.bundle_id, cv_assignment_id=invalid_id
        )


@pytest.mark.parametrize("component", ["private", "control", "symile"])
def test_lifecycle_rejects_each_local_symlink_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    control_root = execution.directory.parent.parent
    repository = control_root.parents[2]
    backup = tmp_path / "backup"
    backup.mkdir()
    current = validate_current_symile_execution(
        execution,
        build_symile_formal_preflight(
            repository, bundle_id="bundle-" + "a" * 64, cv_assignment_id="cv-assignment-" + "c" * 64
        ),
    )
    victim = {
        "private": repository / "private",
        "control": repository / "private/control",
        "symile": control_root,
    }[component]
    relocated = tmp_path / f"relocated-{component}"
    victim.rename(relocated)
    victim.symlink_to(relocated, target_is_directory=True)
    with pytest.raises(ManifestBuildError, match="contains a symlink"):
        validate_symile_lifecycle(
            current,
            repository_root=repository,
            private_root=repository / "private",
            backup_root=backup,
        )


def test_lifecycle_rejects_symlink_backup_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    repository = execution.directory.parent.parent.parents[2]
    target = tmp_path / "backup-target"
    target.mkdir()
    backup = tmp_path / "backup-link"
    backup.symlink_to(target, target_is_directory=True)
    current = validate_current_symile_execution(
        execution,
        build_symile_formal_preflight(
            repository, bundle_id="bundle-" + "a" * 64, cv_assignment_id="cv-assignment-" + "c" * 64
        ),
    )
    with pytest.raises(ManifestBuildError, match="contains a symlink"):
        validate_symile_lifecycle(
            current,
            repository_root=repository,
            private_root=repository / "private",
            backup_root=backup,
        )


@pytest.mark.parametrize("component", ["control", "symile"])
def test_test_open_rejects_each_external_symlink_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    freeze = _canonical_freeze(execution)
    _permit_synthetic_authorities(monkeypatch, freeze)
    lifecycle = _lifecycle(execution)
    frozen = control.validate_symile_frozen_campaign(freeze=freeze, lifecycle=lifecycle)
    target = tmp_path / f"external-{component}"
    target.mkdir()
    if component == "control":
        link = lifecycle.backup_root / "control"
    else:
        (lifecycle.backup_root / "control").mkdir()
        link = lifecycle.backup_root / "control/symile"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ManifestBuildError, match="contains a symlink"):
        control.open_official_test(frozen_campaign=frozen)


def test_concurrent_test_open_fails_before_creating_divergent_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    freeze = _canonical_freeze(execution)
    _permit_synthetic_authorities(monkeypatch, freeze)
    lifecycle = _lifecycle(execution)
    frozen = control.validate_symile_frozen_campaign(freeze=freeze, lifecycle=lifecycle)
    with control.campaign_lock(
        lifecycle.control_root,
        "Symile",
        trusted_root=lifecycle.repository_root,
    ):
        with pytest.raises(ManifestBuildError, match="campaign is active"):
            control.open_official_test(frozen_campaign=frozen)
    assert not (lifecycle.control_root / "test-open.json").exists()
    assert not (lifecycle.backup_root / "control/symile/test-open.json").exists()


@pytest.mark.parametrize("missing", ["local", "external"])
def test_test_open_recovers_one_missing_copy_without_changing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    freeze = _canonical_freeze(execution)
    _permit_synthetic_authorities(monkeypatch, freeze)
    lifecycle = _lifecycle(execution)
    frozen = control.validate_symile_frozen_campaign(freeze=freeze, lifecycle=lifecycle)
    first = control.open_official_test(frozen_campaign=frozen)
    original = first.opening.document_bytes
    target = first.opening.path if missing == "local" else first.opening.external_path
    target.unlink()
    recovered = control.open_official_test(frozen_campaign=frozen)
    assert recovered.opening.document_bytes == original
    assert recovered.opening.path.read_bytes() == recovered.opening.external_path.read_bytes()


def test_test_open_does_not_repair_from_malformed_surviving_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    freeze = _canonical_freeze(execution)
    _permit_synthetic_authorities(monkeypatch, freeze)
    lifecycle = _lifecycle(execution)
    frozen = control.validate_symile_frozen_campaign(freeze=freeze, lifecycle=lifecycle)
    capability = control.open_official_test(frozen_campaign=frozen)
    malformed = b'{"schema_version":'
    capability.opening.path.write_bytes(malformed)
    capability.opening.external_path.unlink()

    with pytest.raises(ManifestBuildError, match="malformed"):
        control.open_official_test(frozen_campaign=frozen)

    assert capability.opening.path.read_bytes() == malformed
    assert not capability.opening.external_path.exists()


def test_test_open_rejects_conflicting_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    freeze = _canonical_freeze(execution)
    _permit_synthetic_authorities(monkeypatch, freeze)
    lifecycle = _lifecycle(execution)
    frozen = control.validate_symile_frozen_campaign(freeze=freeze, lifecycle=lifecycle)
    capability = control.open_official_test(frozen_campaign=frozen)
    document = json.loads(capability.opening.external_path.read_bytes())
    document["opened_at_utc"] = "2026-01-01T00:00:00Z"
    capability.opening.external_path.write_bytes(canonical_json_bytes(document))
    with pytest.raises(ManifestBuildError, match="differ"):
        control.open_official_test(frozen_campaign=frozen)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("opened_at_utc", "not-a-timestamp"),
        ("extra", "forbidden"),
    ],
)
def test_test_open_rejects_malformed_or_extended_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    freeze = _canonical_freeze(execution)
    _permit_synthetic_authorities(monkeypatch, freeze)
    lifecycle = _lifecycle(execution)
    frozen = control.validate_symile_frozen_campaign(freeze=freeze, lifecycle=lifecycle)
    capability = control.open_official_test(frozen_campaign=frozen)
    document = json.loads(capability.opening.document_bytes)
    document[field] = value
    raw = canonical_json_bytes(document)
    capability.opening.path.write_bytes(raw)
    capability.opening.external_path.write_bytes(raw)
    with pytest.raises(ManifestBuildError, match="another formal campaign"):
        control.open_official_test(frozen_campaign=frozen)


def test_test_open_rejects_another_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _execution(tmp_path / "campaign", monkeypatch, git="a" * 40)
    freeze = _canonical_freeze(original)
    _permit_synthetic_authorities(monkeypatch, freeze)
    lifecycle = _lifecycle(original)
    frozen = control.validate_symile_frozen_campaign(freeze=freeze, lifecycle=lifecycle)
    control.open_official_test(frozen_campaign=frozen)
    with pytest.raises(ManifestBuildError, match="another execution"):
        _execution(tmp_path / "campaign", monkeypatch, git="b" * 40)


def test_frozen_campaign_rejects_active_execution_receipt_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    freeze = _canonical_freeze(execution)
    _permit_synthetic_authorities(monkeypatch, freeze)
    lifecycle = _lifecycle(execution)
    lifecycle.control_root.joinpath("active-execution.json").write_bytes(
        canonical_json_bytes({"schema_version": 1, "execution_id": "symile-execution-" + "9" * 64})
    )
    with pytest.raises(ManifestBuildError, match="active execution receipt"):
        control.validate_symile_frozen_campaign(freeze=freeze, lifecycle=lifecycle)


@pytest.mark.parametrize(
    "changed",
    [
        {"git": "9" * 40},
        {"lock": "9" * 64},
        {"python_version": "3.13.8"},
        {"cuda_runtime_version": "12.5"},
        {"gpu_name": "GPU-B"},
        {"gpu_compute_capability": (9, 0)},
        {"logistic_c": 0.75},
        {"weight_sha": "9" * 64},
    ],
)
def test_test_open_rejects_current_environment_execution_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: dict[str, object],
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    changed_preflight = _preflight(tmp_path / "execution", monkeypatch, **changed)
    with pytest.raises(ManifestBuildError, match="Current environment differs"):
        validate_current_symile_execution(execution, changed_preflight)


def test_current_preflight_rejects_loader_policy_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _execution(tmp_path / "execution", monkeypatch)
    with pytest.raises(ManifestBuildError, match="execution fields"):
        _preflight(tmp_path / "execution", monkeypatch, training_workers=3)


def test_current_execution_accepts_relocated_pretrained_cache_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch, cache_filename="weights.pt")
    current = validate_current_symile_execution(
        execution,
        _preflight(tmp_path / "execution", monkeypatch, cache_filename="relocated.pt"),
    )
    assert current.execution == execution


def test_test_open_rejects_another_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    original = _canonical_freeze(execution)
    _permit_synthetic_authorities(monkeypatch, original)
    lifecycle = _lifecycle(execution)
    frozen = control.validate_symile_frozen_campaign(freeze=original, lifecycle=lifecycle)
    control.open_official_test(frozen_campaign=frozen)
    replacement = _freeze(tmp_path / "replacement-freeze", suffix="9")
    _permit_synthetic_authorities(monkeypatch, replacement)
    with pytest.raises(ManifestBuildError, match="active canonical frozen campaign"):
        control.validate_symile_frozen_campaign(freeze=replacement, lifecycle=lifecycle)


@pytest.mark.parametrize(
    "coordinate,value",
    [
        ("bundle_id", "bundle-" + "8" * 64),
        ("cv_assignment_id", "cv-assignment-" + "8" * 64),
    ],
)
def test_frozen_campaign_rejects_dataset_coordinate_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    coordinate: str,
    value: str,
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    kwargs = {coordinate: value}
    freeze = _freeze(
        execution.directory.parent.parent / "freezes",
        **kwargs,
    )
    monkeypatch.setattr(control, "_require_capability", lambda candidate: candidate)
    with pytest.raises(ManifestBuildError, match="active canonical frozen campaign"):
        control.validate_symile_frozen_campaign(
            freeze=freeze,
            lifecycle=_lifecycle(execution),
        )


def test_frozen_campaign_rejects_moved_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    canonical = _canonical_freeze(execution)
    moved_root = tmp_path / "unrelated-campaign/freezes"
    moved = _freeze(moved_root)
    monkeypatch.setattr(control, "_require_capability", lambda candidate: candidate)
    with pytest.raises(ManifestBuildError, match="active canonical frozen campaign"):
        control.validate_symile_frozen_campaign(
            freeze=moved,
            lifecycle=_lifecycle(execution),
        )
    assert canonical.freeze_id == moved.freeze_id


def test_direct_official_test_materialization_rejects_forged_capability() -> None:
    with pytest.raises(PermissionError, match="held-out capability"):
        control.materialize_official_test(object(), lambda: None)


def test_single_preserved_opening_is_evidence_not_live_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = _execution(tmp_path / "execution", monkeypatch)
    freeze = _canonical_freeze(execution)
    _permit_synthetic_authorities(monkeypatch, freeze)
    lifecycle = _lifecycle(execution)
    frozen = control.validate_symile_frozen_campaign(freeze=freeze, lifecycle=lifecycle)
    live = control.open_official_test(frozen_campaign=frozen)
    with pytest.raises(PermissionError, match="authenticated restoration context"):
        control._validate_preserved_test_open(
            path=live.opening.path,
            restoration=object(),
            freeze=freeze,
            execution=execution,
        )


def test_validated_development_decision_reruns_recursive_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision = development_decision.ValidatedDevelopmentDecision(
        tmp_path / "decision",
        {"development_decision_id": "development-decision-" + "d" * 64},
        "1" * 64,
        "2" * 64,
        tmp_path / "reports",
        tmp_path / "models",
        tmp_path / "private",
        tmp_path / "manifests",
        development_decision._VALIDATION_GUARD,
    )
    calls = 0

    def revalidate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return decision
        raise ManifestBuildError("ancestry changed")

    monkeypatch.setattr(development_decision, "validate_development_decision", revalidate)
    assert development_decision.require_validated_development_decision(decision) is decision
    with pytest.raises(ManifestBuildError, match="ancestry changed"):
        development_decision.require_validated_development_decision(decision)


def test_held_out_operation_authenticates_once_checks_receipts_and_expires(tmp_path, monkeypatch):
    from symile_campaign_test_support import _freeze as freeze_fixture
    from symile_campaign_test_support import _synthetic_held_out_capability

    capability = _synthetic_held_out_capability(tmp_path, freeze_fixture(tmp_path / "freeze"))
    traversals = []

    def authenticate(value):
        traversals.append(value)
        return value

    monkeypatch.setattr(control, "require_symile_frozen_campaign", authenticate)
    with control.held_out_operation(capability) as scoped:
        control.require_validated_held_out_capability(scoped)
        control.require_validated_held_out_capability(scoped)
        assert len(traversals) == 1
        assert control._require_capability(scoped.freeze) is scoped.freeze
        scoped.freeze.manifest["bundle_id"] = "bundle-" + "f" * 64
        with pytest.raises(PermissionError, match="changed"):
            control.require_validated_held_out_capability(scoped)
        scoped.freeze.manifest["bundle_id"] = "bundle-" + "a" * 64
        scoped.opening.external_path.write_bytes(b"{}\n")
        with pytest.raises(ManifestBuildError):
            control.require_validated_held_out_capability(scoped)
        scoped.opening.external_path.write_bytes(scoped.opening.document_bytes)
    with pytest.raises(PermissionError, match="expired"):
        control.require_validated_held_out_capability(scoped)
    control.require_validated_held_out_capability(capability)
    assert len(traversals) == 2
