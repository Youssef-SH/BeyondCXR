from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import create_autospec

import pytest

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.config import load_experiment_config, with_runtime
from beyondcxr.training.rsna_campaign import (
    _ensure_training_report_derivatives,
    _preflight_rsna_campaign,
    _rsna_export_name,
    _validate_config_agreement,
    _validate_neural_campaign_configs,
    _validate_outputs,
    _validated_rsna_freezes,
    execute_rsna_campaign,
    main,
)
from beyondcxr.training.rsna_compare import ComparisonResult
from beyondcxr.training.rsna_datasets import RsnaDataset
from beyondcxr.training.rsna_formal import (
    _FORMAL_PLAN_GUARD,
    RSNA_FORMAL_RUN_PLAN,
    RsnaAuthorityCoordinates,
    RsnaFormalRoots,
    ValidatedRsnaPlan,
)
from beyondcxr.training.rsna_localize import generate_localization_report

_PACKAGE_IDS = {
    name: "model-package-" + character * 64
    for name, character in zip(
        (
            "metadata-logistic",
            "metadata-lightgbm",
            "cxr-17",
            "cxr-42",
            "cxr-2026",
            "fusion-17",
            "fusion-42",
            "fusion-2026",
        ),
        "12345678",
        strict=True,
    )
}


def _evaluation_id(package_id: str) -> str:
    return "evaluation-" + hashlib.sha256(package_id.encode("ascii")).hexdigest()


def test_rsna_archive_name_is_specification_stable() -> None:
    arguments = {
        "freeze_id": "rsna-freeze-" + "a" * 64,
        "execution_id": "rsna-execution-" + "b" * 64,
    }
    expected = "rsna-rsna-freeze-" + "a" * 64 + "-rsna-execution-" + "b" * 64

    assert _rsna_export_name(**arguments) == expected
    assert f"{_rsna_export_name(**arguments)}.zip" == f"{expected}.zip"


def test_dangling_freeze_root_cannot_be_treated_as_unfrozen(tmp_path: Path) -> None:
    control = tmp_path / "control"
    control.mkdir()
    (control / "freezes").symlink_to(tmp_path / "absent", target_is_directory=True)
    plan = SimpleNamespace(roots=SimpleNamespace(control_root=control))
    with pytest.raises(ManifestBuildError, match="freeze root is invalid"):
        _validated_rsna_freezes(plan)


def _configs():
    return tuple(
        with_runtime(
            load_experiment_config(spec.config_relative, bundle_id="bundle-" + "a" * 64),
            seed=spec.seed,
        )
        for spec in RSNA_FORMAL_RUN_PLAN
    )


def _plan(tmp_path: Path) -> ValidatedRsnaPlan:
    configs = _configs()
    reference = configs[0]
    roots = RsnaFormalRoots(
        tmp_path,
        tmp_path / "data/raw/rsna/extracted",
        tmp_path / "data/manifests",
        tmp_path / "data/cache/rsna",
        tmp_path / "models/rsna",
        tmp_path / "reports",
        tmp_path / "private",
        tmp_path / "private/control/rsna",
        tmp_path / "outbox",
        tmp_path / "backup",
        tmp_path / "mlflow.db",
    )
    return ValidatedRsnaPlan(
        authority=RsnaAuthorityCoordinates(
            reference.dataset.dataset_id,
            reference.dataset.bundle_id,
            reference.task.task_id,
            reference.task.label_policy_version,
            tmp_path / f"data/manifests/rsna/bundles/{reference.dataset.bundle_id}",
        ),
        roots=roots,
        runs=RSNA_FORMAL_RUN_PLAN,
        configs=configs,
        dataset=RsnaDataset(),
        git_commit="f" * 40,
        dependency_lock_sha256="0" * 64,
        pretrained_weight=SimpleNamespace(as_dict=lambda: {}),
        runtime=SimpleNamespace(pin_memory_effective=True),
        training_execution=SimpleNamespace(),
        evaluation_execution=SimpleNamespace(),
        _guard=_FORMAL_PLAN_GUARD,
    )


def test_formal_preflight_accepts_existing_tracking_database_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    with sqlite3.connect(repository_root / "mlflow.db") as connection:
        connection.execute("CREATE TABLE tracking_state (value INTEGER NOT NULL)")
    backup_root = tmp_path / "backup"
    monkeypatch.chdir(repository_root)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.discover_repository_root",
        lambda: repository_root,
    )

    import shutil

    from beyondcxr.training import operational_validation

    shutil.copytree(Path(__file__).parents[1] / "configs", repository_root / "configs")
    source = repository_root / "data/raw/rsna/extracted"
    (source / "stage_2_train_images").mkdir(parents=True)
    for name in ("stage_2_train_labels.csv", "stage_2_detailed_class_info.csv"):
        (source / name).write_text("synthetic")
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.git_revision", lambda root: ("f" * 40, False)
    )
    monkeypatch.setattr("beyondcxr.training.rsna_campaign.uv_lock_sha256", lambda path: "0" * 64)
    for name in (
        "validate_bundle_reference",
        "validate_bundle_directory",
        "validate_rsna_cache_source_availability",
    ):
        monkeypatch.setattr("beyondcxr.training.rsna_campaign." + name, lambda *a, **kw: None)
    observed = []

    def volume(path):
        observed.append(path)
        return 1, 25 * 1024**3

    monkeypatch.setattr(operational_validation, "_storage_volume", volume)

    class DestinationValidationPassed(Exception):
        pass

    def stop_before_runtime(config):
        raise DestinationValidationPassed

    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign._required_cuda_runtime", stop_before_runtime
    )
    with pytest.raises(DestinationValidationPassed):
        _preflight_rsna_campaign(backup_root=backup_root, bundle_id="bundle-" + "a" * 64)
    assert observed == [repository_root, repository_root / "outbox", backup_root]
    monkeypatch.setattr(
        operational_validation, "_storage_volume", lambda path: (1, 25 * 1024**3 - 1)
    )
    with pytest.raises(OSError, match="insufficient"):
        _preflight_rsna_campaign(backup_root=backup_root, bundle_id="bundle-" + "a" * 64)


@pytest.mark.parametrize(
    "relative",
    (
        "models/rsna/packages",
        "reports/rsna/runs",
        "private/control/rsna",
        "private/predictions/rsna",
    ),
)
def test_formal_preflight_rejects_nested_canonical_layout_symlinks_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    repository_root = tmp_path / "repository"
    redirected = tmp_path / "redirected"
    repository_root.mkdir()
    redirected.mkdir()
    branch = repository_root / relative
    branch.parent.mkdir(parents=True)
    branch.symlink_to(redirected, target_is_directory=True)
    monkeypatch.chdir(repository_root)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.discover_repository_root",
        lambda: repository_root,
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.git_revision",
        lambda _root: (_ for _ in ()).throw(
            AssertionError("canonical layout validation must happen first")
        ),
    )

    with pytest.raises(ManifestBuildError, match="Canonical RSNA layout.*symlink redirect"):
        _preflight_rsna_campaign(backup_root=tmp_path / "backup", bundle_id="bundle-" + "a" * 64)


def test_formal_preflight_rejects_canonical_layout_type_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repository"
    runs = repository_root / "reports/rsna/runs"
    runs.parent.mkdir(parents=True)
    runs.write_bytes(b"not a directory")
    monkeypatch.chdir(repository_root)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.discover_repository_root",
        lambda: repository_root,
    )

    with pytest.raises(ManifestBuildError, match="Canonical RSNA layout.*root type mismatch"):
        _preflight_rsna_campaign(backup_root=tmp_path / "backup", bundle_id="bundle-" + "a" * 64)


@pytest.mark.parametrize("partial_resume", [False, True])
def test_campaign_consumes_preflight_authority_and_freezes_before_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, partial_resume: bool
) -> None:
    preflight = _plan(tmp_path)
    monkeypatch.chdir(tmp_path)
    events: list[tuple[str, object]] = []
    execution_directory = tmp_path / "private/control/rsna/executions/rsna-execution-test"
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign._preflight_rsna_campaign",
        lambda **kwargs: events.append(("preflight", kwargs["backup_root"])) or preflight,
    )
    execution = SimpleNamespace(
        execution_id="rsna-execution-" + "a" * 64,
        directory=execution_directory,
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.publish_or_validate_execution",
        lambda plan: events.append(("execution", plan)) or execution,
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.generate_rsna_audit",
        lambda *args, **kwargs: events.append(("audit", kwargs["bundle_id"])),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.prepare_rsna_cxr_cache",
        lambda *args, **kwargs: events.append(("cache", None)) or SimpleNamespace(),
    )

    def result(run_id: str):
        return SimpleNamespace(
            run_id=run_id,
            model_package_id=_PACKAGE_IDS[run_id],
            model_path=tmp_path / f"models/rsna/packages/{_PACKAGE_IDS[run_id]}/model.pt",
            artifact_directory=tmp_path / f"reports/rsna/runs/{run_id}",
        )

    reused_names = {"metadata-logistic", "cxr-17"} if partial_resume else set()
    reused = tuple(result(name) if name in reused_names else None for name in _PACKAGE_IDS)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.resolve_completed_rsna_packages",
        lambda plan, execution: reused,
    )
    repaired = []

    def restore_reports(plan, packages, runs):
        assert len(packages) == len(runs) == 1
        repaired.append(packages[0].model_package_id)
        return packages

    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign._ensure_training_report_derivatives", restore_reports
    )
    metadata_ids = iter(
        name for name in ("metadata-logistic", "metadata-lightgbm") if name not in reused_names
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.train_metadata_experiment",
        lambda *args, **kwargs: result(next(metadata_ids)),
    )
    cxr_ids = iter(name for name in ("cxr-17", "cxr-42", "cxr-2026") if name not in reused_names)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.train_cxr_experiment",
        lambda *args, **kwargs: result(next(cxr_ids)),
    )

    def train_fusion(config, *, source_cxr_package_id, **kwargs):
        run_id = f"fusion-{config.runtime.seed}"
        events.append(("fusion_source", (run_id, source_cxr_package_id)))
        return result(run_id)

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.train_fusion_experiment", train_fusion)

    frozen_authorities = []

    def freeze(**kwargs):
        values = tuple(kwargs["results"])
        events.append(("candidate_set_frozen", tuple(item.model_package_id for item in values)))
        authority = SimpleNamespace(
            freeze_id="rsna-freeze-" + "b" * 64,
            packages=values,
        )
        frozen_authorities.append(authority)
        return authority

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.publish_or_validate_freeze", freeze)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.authorize_rsna_held_out",
        lambda **kwargs: SimpleNamespace(freeze=kwargs["freeze"]),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.load_evaluation_record",
        lambda *args, **kwargs: None,
    )

    def evaluate(package_id, **kwargs):
        assert any(kind == "candidate_set_frozen" for kind, _ in events)
        events.append(("test_access", package_id))
        evaluation_id = _evaluation_id(package_id)
        return SimpleNamespace(
            mlflow_run_id=f"run-{package_id}",
            evaluation_id=evaluation_id,
            prediction_id="prediction-" + package_id[-64:],
            model_package_id=package_id,
            artifact_directory=tmp_path / f"reports/rsna/evaluations/{evaluation_id}",
            private_prediction_directory=tmp_path / f"private/predictions/rsna/{package_id}",
        )

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.evaluate_model_package", evaluate)

    def publish_evaluation(
        execution,
        result,
        *,
        freeze,
        report_root,
        private_root,
        model_root,
    ):
        assert execution is execution_authority
        assert freeze is frozen_authorities[0]
        assert report_root == preflight.roots.report_root
        assert private_root == preflight.roots.private_root
        assert model_root == preflight.roots.model_root
        events.append(("evaluation_record", result.evaluation_id))

    execution_authority = execution
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.publish_evaluation_record", publish_evaluation
    )

    summary_directories = iter(
        (
            tmp_path / "reports/rsna/seed-summaries/cxr_densenet",
            tmp_path / "reports/rsna/seed-summaries/cxr_metadata_concat",
        )
    )

    def summarize(ids, **kwargs):
        del ids, kwargs
        directory = next(summary_directories)
        return SimpleNamespace(
            directory=directory,
            report_directory=directory,
        )

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.publish_seed_summary", summarize)
    localization = tmp_path / "reports/rsna/localization/localization-test"
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.generate_localization_report",
        create_autospec(generate_localization_report, return_value=localization),
    )
    comparison = ComparisonResult(
        tmp_path / "reports/rsna/comparisons/comparison",
        tmp_path / "table.csv",
        tmp_path / "table.md",
        8,
        {},
    )

    def compare(
        evaluation_ids,
        *,
        output_directory="reports",
        private_directory="private",
        model_directory="models/rsna",
        freeze,
        execution: object,
    ):
        assert evaluation_ids == tuple(_evaluation_id(value) for value in _PACKAGE_IDS.values())
        assert output_directory == preflight.roots.report_root
        assert private_directory == preflight.roots.private_root
        assert model_directory == preflight.roots.model_root
        assert freeze is frozen_authorities[0]
        assert execution is execution_authority
        return comparison

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.regenerate_comparison", compare)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign._validate_outputs",
        lambda *args, **kwargs: events.append(("outputs_validated", None)),
    )
    from beyondcxr.data.rsna_audit import generate_rsna_audit

    monkeypatch.setattr(
        "beyondcxr.data.rsna_audit._generate_rsna_audit", lambda *args, **kwargs: {}
    )

    def prepare_export(**kwargs):
        # Keep the production audit wrapper at the preparation boundary; scientific
        # closure projection is exercised by the preservation composition tests.
        generate_rsna_audit()
        return ()

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.campaign_export_members", prepare_export)

    def export(**kwargs):
        path = tmp_path / "outbox/campaign.zip"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"archive")
        path.with_suffix(".zip.sha256").write_text("checksum\n", encoding="utf-8")
        events.append(("export", kwargs["export_name"]))
        return path

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.export_and_verify", export)

    campaign = execute_rsna_campaign(
        backup_root=tmp_path / "backup", bundle_id="bundle-" + "a" * 64
    )

    freeze_index = next(i for i, event in enumerate(events) if event[0] == "candidate_set_frozen")
    test_indices = [i for i, event in enumerate(events) if event[0] == "test_access"]
    assert freeze_index < min(test_indices)
    assert len(test_indices) == 8
    assert events[0][0] == "preflight"
    assert ("export", f"rsna-{frozen_authorities[0].freeze_id}-{execution.execution_id}") in events
    assert campaign.archive_path.is_file()
    assert campaign.checksum_path.is_file()
    assert set(repaired) == {_PACKAGE_IDS[name] for name in reused_names}
    assert list(metadata_ids) == list(cxr_ids) == []


def test_campaign_resume_reuses_frozen_packages_and_evaluations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = _plan(tmp_path)
    monkeypatch.chdir(tmp_path)
    execution_directory = tmp_path / "private/control/rsna/executions/execution"
    execution_directory.mkdir(parents=True)
    freeze_directory = tmp_path / "private/control/rsna/freezes/rsna-freeze-test"
    freeze_directory.mkdir(parents=True)
    (freeze_directory / "manifest.json").write_text("{}\n", encoding="utf-8")
    packages = tuple(
        SimpleNamespace(
            run_id=name,
            model_package_id=package_id,
            model_path=tmp_path / f"models/rsna/packages/{package_id}/model.pt",
            artifact_directory=tmp_path / f"reports/rsna/runs/{name}",
        )
        for name, package_id in _PACKAGE_IDS.items()
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign._preflight_rsna_campaign",
        lambda *, backup_root, bundle_id, source_root="data/raw/rsna/extracted": preflight,
    )
    execution = SimpleNamespace(execution_id="execution", directory=execution_directory)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.resolve_completed_rsna_packages",
        lambda plan, execution: (None,) * len(plan.configs),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.publish_or_validate_execution", lambda plan: execution
    )
    frozen = SimpleNamespace(freeze_id="rsna-freeze-test", packages=packages)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.validate_freeze", lambda *args, **kwargs: frozen
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign._ensure_training_report_derivatives",
        lambda plan, members, runs: packages,
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.authorize_rsna_held_out",
        lambda **kwargs: SimpleNamespace(freeze=kwargs["freeze"]),
    )
    for name in (
        "train_metadata_experiment",
        "train_cxr_experiment",
        "train_fusion_experiment",
        "evaluate_model_package",
    ):
        monkeypatch.setattr(
            f"beyondcxr.training.rsna_campaign.{name}",
            lambda *args, _name=name, **kwargs: pytest.fail(f"unexpected {_name}"),
        )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.generate_rsna_audit", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.prepare_rsna_cxr_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    def completed(*args, package_id, **kwargs):
        return SimpleNamespace(
            evaluation_id=_evaluation_id(package_id),
            prediction_id="prediction-" + package_id[-64:],
            model_package_id=package_id,
            mlflow_run_id="run",
            artifact_directory=tmp_path / "evaluation",
            private_prediction_directory=tmp_path / "prediction",
        )

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.load_evaluation_record", completed)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.publish_seed_summary",
        lambda ids, **kwargs: SimpleNamespace(directory=tmp_path, report_directory=tmp_path),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.generate_localization_report",
        create_autospec(generate_localization_report, return_value=tmp_path / "localization"),
    )

    def compare(
        evaluation_ids,
        *,
        output_directory="reports",
        private_directory="private",
        model_directory="models/rsna",
        freeze,
        execution: object,
    ):
        assert evaluation_ids == tuple(_evaluation_id(value) for value in _PACKAGE_IDS.values())
        assert output_directory == preflight.roots.report_root
        assert private_directory == preflight.roots.private_root
        assert model_directory == preflight.roots.model_root
        assert freeze is frozen
        assert execution is execution_authority
        return ComparisonResult(tmp_path, tmp_path, tmp_path, 8, {})

    execution_authority = execution
    monkeypatch.setattr("beyondcxr.training.rsna_campaign.regenerate_comparison", compare)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign._validate_outputs", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.campaign_export_members",
        lambda **kwargs: (SimpleNamespace(),),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.export_and_verify",
        lambda **kwargs: _archive(tmp_path),
    )

    execute_rsna_campaign(backup_root=tmp_path / "backup", bundle_id="bundle-" + "a" * 64)


def test_frozen_package_regenerates_missing_training_report_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    plan = replace(plan, runs=(plan.runs[0],), configs=(plan.configs[0],))
    package_id = _PACKAGE_IDS["metadata-logistic"]
    frozen = SimpleNamespace(
        packages=(
            SimpleNamespace(
                model_package_id=package_id,
                model_path=plan.roots.model_root / "packages" / package_id / "model.skops",
            ),
        )
    )
    regenerated: list[Path] = []
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.load_canonical_validation_cohort",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.validate_rsna_model_package",
        lambda *args, **kwargs: {},
    )

    def regenerate(directory: Path, **kwargs: object) -> None:
        regenerated.append(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metrics.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.regenerate_training_report",
        regenerate,
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.validate_training_report",
        lambda path, **kwargs: None,
    )

    result = _ensure_training_report_derivatives(plan, frozen.packages, plan.runs)

    assert len(regenerated) == 1
    assert result[0].model_package_id == package_id
    assert result[0].artifact_directory.name == package_id


def test_equivalent_duplicate_training_reports_do_not_ambiguate_frozen_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    plan = replace(plan, runs=(plan.runs[0],), configs=(plan.configs[0],))
    package_id = _PACKAGE_IDS["metadata-logistic"]
    frozen = SimpleNamespace(
        packages=(
            SimpleNamespace(
                model_package_id=package_id,
                model_path=plan.roots.model_root / "packages" / package_id / "model.skops",
            ),
        )
    )
    report_root = plan.roots.report_root / "rsna/runs"
    for name in ("run-b", "run-a"):
        directory = report_root / name
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(
            json.dumps({"training_package": {"model_package_id": package_id}}),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.load_canonical_validation_cohort",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.validate_rsna_model_package",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.validate_training_report",
        lambda path, **kwargs: None,
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.regenerate_training_report",
        lambda *args, **kwargs: pytest.fail("equivalent derivatives must be reused"),
    )

    result = _ensure_training_report_derivatives(plan, frozen.packages, plan.runs)

    assert result[0].artifact_directory.name == "run-a"


def test_invalid_matching_training_report_is_quarantined_and_regenerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    plan = replace(plan, runs=(plan.runs[0],), configs=(plan.configs[0],))
    package_id = _PACKAGE_IDS["metadata-logistic"]
    frozen = SimpleNamespace(
        packages=(
            SimpleNamespace(
                model_package_id=package_id,
                model_path=plan.roots.model_root / "packages" / package_id / "model.skops",
            ),
        )
    )
    directory = plan.roots.report_root / "rsna/runs" / package_id
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text(
        json.dumps({"training_package": {"model_package_id": package_id}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.load_canonical_validation_cohort",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.validate_rsna_model_package",
        lambda *args, **kwargs: {},
    )
    regenerated: list[Path] = []

    def regenerate(path: Path, **kwargs: object) -> None:
        regenerated.append(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "metrics.json").write_text("{}\n", encoding="utf-8")

    def validate(path: Path, **kwargs: object) -> None:
        if path == directory:
            raise ValueError("tampered")

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.regenerate_training_report", regenerate)
    monkeypatch.setattr("beyondcxr.training.rsna_campaign.validate_training_report", validate)

    result = _ensure_training_report_derivatives(plan, frozen.packages, plan.runs)

    assert len(regenerated) == 1
    assert result[0].artifact_directory == directory
    assert directory.is_dir()
    quarantine = directory.parent / ".invalid-training-reports"
    assert [path.name for path in quarantine.iterdir()] == [f"{package_id}-000000"]


def test_unrelated_malformed_training_report_does_not_block_frozen_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    plan = replace(plan, runs=(plan.runs[0],), configs=(plan.configs[0],))
    package_id = _PACKAGE_IDS["metadata-logistic"]
    frozen = SimpleNamespace(
        packages=(
            SimpleNamespace(
                model_package_id=package_id,
                model_path=plan.roots.model_root / "packages" / package_id / "model.skops",
            ),
        )
    )
    malformed = plan.roots.report_root / "rsna/runs/unrelated"
    malformed.mkdir(parents=True)
    (malformed / "metrics.json").write_bytes(b"not-json")
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.load_canonical_validation_cohort",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.validate_rsna_model_package",
        lambda *args, **kwargs: {},
    )

    def regenerate(path: Path, **kwargs: object) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "metrics.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.regenerate_training_report", regenerate)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.validate_training_report",
        lambda path, **kwargs: None,
    )

    result = _ensure_training_report_derivatives(plan, frozen.packages, plan.runs)

    assert result[0].artifact_directory.name == package_id
    assert (malformed / "metrics.json").read_bytes() == b"not-json"


@pytest.mark.parametrize(
    "failure", [RuntimeError("fit failed"), KeyboardInterrupt(), SystemExit(2)]
)
def test_training_failure_cannot_cross_scientific_freeze_or_heldout_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    plan = _plan(tmp_path)
    execution_directory = plan.roots.control_root / "executions/execution"
    execution_directory.mkdir(parents=True)
    execution = SimpleNamespace(execution_id="execution", directory=execution_directory)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.resolve_completed_rsna_packages",
        lambda plan, execution: (None,) * len(plan.configs),
    )
    heldout_calls = []
    freeze_calls = []
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign._preflight_rsna_campaign",
        lambda *, backup_root, bundle_id, source_root="data/raw/rsna/extracted": plan,
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.publish_or_validate_execution", lambda value: execution
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.generate_rsna_audit", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.prepare_rsna_cxr_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.train_metadata_experiment",
        lambda value, **kwargs: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.publish_or_validate_freeze",
        lambda **kwargs: freeze_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.evaluate_model_package",
        lambda *args, **kwargs: heldout_calls.append((args, kwargs)),
    )

    with pytest.raises(type(failure)):
        execute_rsna_campaign(backup_root=plan.roots.backup_root, bundle_id="bundle-" + "a" * 64)

    assert freeze_calls == []
    assert heldout_calls == []
    assert not (plan.roots.control_root / "freezes").exists()
    assert not (plan.roots.private_root / "predictions/rsna").exists()
    logs = tuple((plan.roots.report_root / "rsna/campaigns").glob("*/execution.log"))
    assert len(logs) == 1
    assert "event=campaign_failed" in logs[0].read_text(encoding="utf-8")


def test_source_authentication_failure_prevents_all_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    execution_directory = plan.roots.control_root / "executions/execution"
    execution_directory.mkdir(parents=True)
    execution = SimpleNamespace(execution_id="execution", directory=execution_directory)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.resolve_completed_rsna_packages",
        lambda plan, execution: (None,) * len(plan.configs),
    )
    training_calls = []
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign._preflight_rsna_campaign",
        lambda *, backup_root, bundle_id, source_root="data/raw/rsna/extracted": plan,
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.publish_or_validate_execution", lambda value: execution
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.generate_rsna_audit", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.prepare_rsna_cxr_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ManifestBuildError("source inventory SHA-256 mismatch")
        ),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.train_metadata_experiment",
        lambda value, **kwargs: training_calls.append(value),
    )

    with pytest.raises(ManifestBuildError, match="source inventory SHA-256 mismatch"):
        execute_rsna_campaign(backup_root=plan.roots.backup_root, bundle_id="bundle-" + "a" * 64)

    assert training_calls == []
    assert not (plan.roots.control_root / "freezes").exists()
    assert not (plan.roots.private_root / "predictions/rsna").exists()


def _archive(tmp_path: Path) -> Path:
    path = tmp_path / "campaign.zip"
    path.write_bytes(b"archive")
    path.with_suffix(".zip.sha256").write_text("checksum\n", encoding="utf-8")
    return path


def test_campaign_cli_requires_backup_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = []
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.execute_rsna_campaign",
        lambda **kwargs: (
            calls.append(kwargs)
            or SimpleNamespace(
                campaign_id="campaign",
                model_package_ids=(),
                evaluation_ids=(),
                training_run_ids=(),
                evaluation_run_ids=(),
                archive_path=tmp_path / "archive.zip",
                checksum_path=tmp_path / "archive.zip.sha256",
                campaign_log_path=tmp_path / "execution.log",
            )
        ),
    )
    with pytest.raises(SystemExit):
        main([])
    assert "--backup-root" in capsys.readouterr().err
    assert (
        main(["--backup-root", str(tmp_path / "backup"), "--bundle-id", "bundle-" + "a" * 64]) == 0
    )
    assert calls == [{"backup_root": tmp_path / "backup", "bundle_id": "bundle-" + "a" * 64}]


def test_preflight_rejection_occurs_before_campaign_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def reject(**kwargs):
        del kwargs
        raise ManifestBuildError("exact configured authority differs")

    monkeypatch.setattr("beyondcxr.training.rsna_campaign._preflight_rsna_campaign", reject)

    with pytest.raises(ManifestBuildError, match="exact configured authority differs"):
        execute_rsna_campaign(
            backup_root=tmp_path.parent / "backup", bundle_id="bundle-" + "a" * 64
        )

    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "private").exists()
    assert not (tmp_path / "outbox").exists()


@pytest.mark.parametrize("mismatch", ["preprocessing", "device", "batch_size", "workers", "seed"])
def test_campaign_rejects_neural_config_contract_mismatch(mismatch: str) -> None:
    configs = _configs()
    fusion_indices = tuple(
        index
        for index, spec in enumerate(RSNA_FORMAL_RUN_PLAN)
        if spec.family_id == "cxr_metadata_concat"
    )
    if mismatch == "preprocessing":
        fusions = tuple(
            replace(
                config,
                family=replace(
                    config.family,
                    parameters=MappingProxyType({**config.family.parameters, "image_size": 225}),
                ),
            )
            for config in (configs[index] for index in fusion_indices)
        )
    elif mismatch == "device":
        fusions = tuple(
            replace(config, runtime=replace(config.runtime, device="cpu"))
            for config in (configs[index] for index in fusion_indices)
        )
    elif mismatch == "seed":
        selected = tuple(configs[index] for index in fusion_indices)
        fusions = (
            replace(selected[0], runtime=replace(selected[0].runtime, seed=18)),
            *selected[1:],
        )
    elif mismatch == "batch_size":
        fusions = tuple(
            replace(config, neural=replace(config.neural, batch_size=16))
            for config in (configs[index] for index in fusion_indices)
        )
    else:
        fusions = tuple(
            replace(config, runtime=replace(config.runtime, num_workers=4))
            for config in (configs[index] for index in fusion_indices)
        )
    with pytest.raises(ValueError):
        changed = list(configs)
        for index, config in zip(fusion_indices, fusions, strict=True):
            changed[index] = config
        _validate_neural_campaign_configs(tuple(changed))


def test_campaign_rejects_cross_config_authority_disagreement() -> None:
    configs = _configs()
    changed = replace(
        configs[1],
        dataset=replace(configs[1].dataset, bundle_id="bundle-" + "0" * 64),
    )
    with pytest.raises(ManifestBuildError, match="disagree"):
        _validate_config_agreement((configs[0], changed, *configs[2:]))


def test_output_validation_rejects_evaluation_lineage_mismatch(tmp_path: Path) -> None:
    training = tuple(
        SimpleNamespace(
            run_id=f"train-{index}",
            model_package_id=f"model-package-{index}",
            model_path=tmp_path / "model",
            artifact_directory=tmp_path / "report",
        )
        for index in range(8)
    )
    evaluations = tuple(
        SimpleNamespace(
            evaluation_id=f"evaluation-{index}",
            model_package_id=f"wrong-package-{index}",
            artifact_directory=tmp_path / "test-report",
        )
        for index in range(8)
    )
    with pytest.raises(ValueError):
        _validate_outputs(
            training,
            evaluations,
            tmp_path / "audit",
            tmp_path / "cxr-summary",
            tmp_path / "fusion-summary",
            tmp_path / "localization",
            ComparisonResult(tmp_path, tmp_path, tmp_path, 8, {}),
            tmp_path / "execution.log",
            roots=_plan(tmp_path).roots,
        )
