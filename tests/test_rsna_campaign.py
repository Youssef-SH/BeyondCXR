from __future__ import annotations

import hashlib
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from rsna_preservation_test_support import build_real_rsna_campaign_closure

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.config import load_experiment_config, with_runtime
from beyondcxr.training.rsna_campaign import (
    _preflight_rsna_campaign,
    _validate_config_agreement,
    _validate_neural_campaign_configs,
    _validate_outputs,
    execute_rsna_campaign,
    main,
)
from beyondcxr.training.rsna_campaign_control import validate_execution
from beyondcxr.training.rsna_compare import ComparisonResult
from beyondcxr.training.rsna_datasets import RsnaDataset
from beyondcxr.training.rsna_formal import (
    _FORMAL_PLAN_GUARD,
    RSNA_FORMAL_RUN_PLAN,
    RsnaAuthorityCoordinates,
    RsnaFormalRoots,
    ValidatedRsnaPlan,
)

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


def _configs():
    return tuple(
        with_runtime(load_experiment_config(spec.config_relative), seed=spec.seed)
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
            reference.dataset.bundle_manifest_sha256,
            reference.dataset.split_assignment_id,
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

    class DestinationValidationPassed(Exception):
        pass

    def stop_after_destination_validation(root: Path) -> tuple[str, bool]:
        assert root == repository_root
        raise DestinationValidationPassed

    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.git_revision",
        stop_after_destination_validation,
    )

    with pytest.raises(DestinationValidationPassed):
        _preflight_rsna_campaign(backup_root=backup_root)


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
        _preflight_rsna_campaign(backup_root=tmp_path / "backup")


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
        _preflight_rsna_campaign(backup_root=tmp_path / "backup")


def test_campaign_consumes_preflight_authority_and_freezes_before_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    metadata_ids = iter(("metadata-logistic", "metadata-lightgbm"))
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.train_metadata_experiment",
        lambda *args, **kwargs: result(next(metadata_ids)),
    )
    cxr_ids = iter(("cxr-17", "cxr-42", "cxr-2026"))
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.train_cxr_experiment",
        lambda *args, **kwargs: result(next(cxr_ids)),
    )

    def train_fusion(config, *, source_cxr_package_id, **kwargs):
        run_id = f"fusion-{config.runtime.seed}"
        events.append(("fusion_source", (run_id, source_cxr_package_id)))
        return result(run_id)

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.train_fusion_experiment", train_fusion)

    def freeze(**kwargs):
        values = tuple(kwargs["results"])
        execution_directory.mkdir(parents=True)
        (execution_directory / "package-freeze.json").write_text("{}\n", encoding="utf-8")
        events.append(("candidate_set_frozen", tuple(item.model_package_id for item in values)))
        return SimpleNamespace(packages=values)

    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.publish_or_validate_package_freeze", freeze
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
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.publish_evaluation_record",
        lambda execution, result: events.append(("evaluation_record", result.evaluation_id)),
    )

    def summarize(ids, **kwargs):
        identity = "seed-summary-" + hashlib.sha256("".join(ids).encode()).hexdigest()
        return SimpleNamespace(
            seed_summary_id=identity,
            directory=tmp_path / f"reports/rsna/seed-summaries/{identity}",
            report_directory=tmp_path / f"reports/rsna/seed-summaries/{identity}",
        )

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.publish_seed_summary", summarize)
    localization = tmp_path / "reports/rsna/localization/localization-test"
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.generate_localization_report",
        lambda *args, **kwargs: localization,
    )
    comparison = ComparisonResult(
        "comparison-" + "b" * 64,
        tmp_path / ("reports/rsna/comparisons/comparison-" + "b" * 64),
        tmp_path / "table.csv",
        tmp_path / "table.md",
        8,
        {},
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.regenerate_comparison",
        lambda *args, **kwargs: comparison,
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign._validate_outputs",
        lambda *args, **kwargs: events.append(("outputs_validated", None)),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.campaign_export_members",
        lambda **kwargs: (SimpleNamespace(),),
    )

    def export(**kwargs):
        path = tmp_path / "outbox/campaign.zip"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"archive")
        path.with_suffix(".zip.sha256").write_text("checksum\n", encoding="utf-8")
        events.append(("export", kwargs["export_name"]))
        return path

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.export_and_verify", export)

    campaign = execute_rsna_campaign(backup_root=tmp_path / "backup")

    freeze_index = next(i for i, event in enumerate(events) if event[0] == "candidate_set_frozen")
    test_indices = [i for i, event in enumerate(events) if event[0] == "test_access"]
    assert freeze_index < min(test_indices)
    assert len(test_indices) == 8
    assert events[0][0] == "preflight"
    assert campaign.archive_path.is_file()
    assert campaign.checksum_path.is_file()


def test_campaign_resume_reuses_frozen_packages_and_evaluations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = _plan(tmp_path)
    monkeypatch.chdir(tmp_path)
    execution_directory = tmp_path / "private/control/rsna/executions/execution"
    execution_directory.mkdir(parents=True)
    (execution_directory / "package-freeze.json").write_text("{}\n", encoding="utf-8")
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
        "beyondcxr.training.rsna_campaign._preflight_rsna_campaign", lambda **kwargs: preflight
    )
    execution = SimpleNamespace(execution_id="execution", directory=execution_directory)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.publish_or_validate_execution", lambda plan: execution
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.validate_package_freeze",
        lambda *args, **kwargs: SimpleNamespace(packages=packages),
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
        lambda ids, **kwargs: SimpleNamespace(
            seed_summary_id="summary", directory=tmp_path, report_directory=tmp_path
        ),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.generate_localization_report",
        lambda *args, **kwargs: tmp_path / "localization",
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.regenerate_comparison",
        lambda *args, **kwargs: ComparisonResult("comparison", tmp_path, tmp_path, tmp_path, 8, {}),
    )
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

    execute_rsna_campaign(backup_root=tmp_path / "backup")


@pytest.mark.parametrize(
    "failure", [RuntimeError("fit failed"), KeyboardInterrupt(), SystemExit(2)]
)
def test_training_failure_cannot_cross_package_freeze_or_heldout_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    plan = _plan(tmp_path)
    execution_directory = plan.roots.control_root / "executions/execution"
    execution_directory.mkdir(parents=True)
    execution = SimpleNamespace(execution_id="execution", directory=execution_directory)
    heldout_calls = []
    freeze_calls = []
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign._preflight_rsna_campaign", lambda **kwargs: plan
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
        "beyondcxr.training.rsna_campaign._train_metadata",
        lambda value: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.publish_or_validate_package_freeze",
        lambda **kwargs: freeze_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.evaluate_model_package",
        lambda *args, **kwargs: heldout_calls.append((args, kwargs)),
    )

    with pytest.raises(type(failure)):
        execute_rsna_campaign(backup_root=plan.roots.backup_root)

    assert freeze_calls == []
    assert heldout_calls == []
    assert not (execution_directory / "package-freeze.json").exists()
    assert not (plan.roots.private_root / "predictions/rsna").exists()
    logs = tuple((plan.roots.report_root / "rsna/campaigns").glob("*/execution.log"))
    assert len(logs) == 1
    assert "event=campaign_failed" in logs[0].read_text(encoding="utf-8")


def test_partial_training_publication_is_not_completion_before_package_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closure = build_real_rsna_campaign_closure(tmp_path / "campaign")
    plan = closure.plan
    execution = closure.execution
    execution_id = execution.execution_id
    freeze_path = execution.directory / "package-freeze.json"
    freeze_path.unlink()
    shutil.rmtree(execution.directory / "evaluations")
    for path in (
        plan.roots.report_root / "rsna/evaluations",
        plan.roots.report_root / "rsna/seed-summaries",
        plan.roots.report_root / "rsna/localization",
        plan.roots.report_root / "rsna/comparisons",
        plan.roots.private_root / "predictions/rsna",
        plan.roots.private_root / "localization",
    ):
        shutil.rmtree(path)

    holding = tmp_path / "holding"
    holding_packages = holding / "packages"
    holding_reports = holding / "reports"
    holding_packages.mkdir(parents=True)
    holding_reports.mkdir(parents=True)
    results = {
        (spec.family_id, spec.seed): result
        for spec, result in zip(plan.runs, closure.training_results, strict=True)
    }
    for result in results.values():
        result.model_path.parent.rename(holding_packages / result.model_package_id)
        result.artifact_directory.rename(holding_reports / result.run_id)

    monkeypatch.chdir(plan.roots.repository_root)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign._preflight_rsna_campaign",
        lambda **kwargs: plan,
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.generate_rsna_audit", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.prepare_rsna_cxr_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    phase = "interrupt"
    calls: dict[str, list[tuple[str, int]]] = {
        "interrupt": [],
        "conflict": [],
        "complete": [],
    }
    restored: dict[str, list[tuple[str, int]]] = {
        "interrupt": [],
        "conflict": [],
        "complete": [],
    }

    class PlannedInterruption(RuntimeError):
        pass

    class HeldoutReached(RuntimeError):
        pass

    def train(config, **kwargs):
        del kwargs
        key = (config.family.family_id, config.runtime.seed)
        calls[phase].append(key)
        if phase == "interrupt" and len(calls[phase]) == 2:
            raise PlannedInterruption("interrupted after one durable training publication")
        result = results[key]
        package = result.model_path.parent
        report = result.artifact_directory
        if not package.exists():
            (holding_packages / result.model_package_id).rename(package)
            (holding_reports / result.run_id).rename(report)
            restored[phase].append(key)
        return result

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.train_metadata_experiment", train)
    monkeypatch.setattr("beyondcxr.training.rsna_campaign.train_cxr_experiment", train)

    def train_fusion(config, *, source_cxr_package_id, **kwargs):
        expected = results[("cxr_densenet", config.runtime.seed)].model_package_id
        assert source_cxr_package_id == expected
        return train(config, **kwargs)

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.train_fusion_experiment", train_fusion)
    heldout_calls: list[str] = []

    def heldout(package_id, *, authorization, **kwargs):
        del kwargs
        assert freeze_path.is_file()
        assert len(authorization.packages) == 8
        authorization.require(package_id)
        assert all(result.model_path.parent.is_dir() for result in results.values())
        assert all(result.artifact_directory.is_dir() for result in results.values())
        heldout_calls.append(package_id)
        raise HeldoutReached("held-out reached only after complete freeze")

    monkeypatch.setattr("beyondcxr.training.rsna_campaign.evaluate_model_package", heldout)

    with pytest.raises(PlannedInterruption, match="one durable training publication"):
        execute_rsna_campaign(backup_root=plan.roots.backup_root)

    first_key = ("metadata_logistic", 42)
    first = results[first_key]
    assert calls["interrupt"] == [first_key, ("metadata_lightgbm", 42)]
    assert restored["interrupt"] == [first_key]
    assert first.model_path.parent.is_dir()
    assert first.artifact_directory.is_dir()
    assert sum(result.model_path.parent.is_dir() for result in results.values()) == 1
    assert not freeze_path.exists()
    assert heldout_calls == []
    assert validate_execution(execution.directory).execution_id == execution_id

    report_file = first.artifact_directory / "evaluation_report.md"
    original_report = report_file.read_bytes()
    report_file.write_bytes(original_report + b"conflicting partial state\n")
    phase = "conflict"
    with pytest.raises(ValueError, match="differs from authoritative evidence rendering"):
        execute_rsna_campaign(backup_root=plan.roots.backup_root)

    assert calls["conflict"] == [(spec.family_id, spec.seed) for spec in plan.runs]
    assert len(restored["conflict"]) == 7
    assert all(result.model_path.parent.is_dir() for result in results.values())
    assert all(result.artifact_directory.is_dir() for result in results.values())
    assert not freeze_path.exists()
    assert heldout_calls == []
    assert validate_execution(execution.directory).execution_id == execution_id

    report_file.write_bytes(original_report)
    phase = "complete"
    with pytest.raises(HeldoutReached, match="only after complete freeze"):
        execute_rsna_campaign(backup_root=plan.roots.backup_root)

    assert calls["complete"] == [(spec.family_id, spec.seed) for spec in plan.runs]
    assert restored["complete"] == []
    assert freeze_path.is_file()
    assert len(heldout_calls) == 1
    assert validate_execution(execution.directory).execution_id == execution_id


def test_source_authentication_failure_prevents_all_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    execution_directory = plan.roots.control_root / "executions/execution"
    execution_directory.mkdir(parents=True)
    execution = SimpleNamespace(execution_id="execution", directory=execution_directory)
    training_calls = []
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign._preflight_rsna_campaign", lambda **kwargs: plan
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
        "beyondcxr.training.rsna_campaign._train_metadata",
        lambda value: training_calls.append(value),
    )

    with pytest.raises(ManifestBuildError, match="source inventory SHA-256 mismatch"):
        execute_rsna_campaign(backup_root=plan.roots.backup_root)

    assert training_calls == []
    assert not (execution_directory / "package-freeze.json").exists()
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
    assert main(["--backup-root", str(tmp_path / "backup")]) == 0
    assert calls == [{"backup_root": tmp_path / "backup"}]


def test_preflight_rejection_occurs_before_campaign_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def reject(**kwargs):
        del kwargs
        raise ManifestBuildError("exact configured authority differs")

    monkeypatch.setattr("beyondcxr.training.rsna_campaign._preflight_rsna_campaign", reject)

    with pytest.raises(ManifestBuildError, match="exact configured authority differs"):
        execute_rsna_campaign(backup_root=tmp_path.parent / "backup")

    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "private").exists()
    assert not (tmp_path / "outbox").exists()


def test_formal_campaign_module_has_no_authority_builder() -> None:
    import beyondcxr.training.rsna_campaign as campaign

    assert not hasattr(campaign, "build_and_write")
    assert not hasattr(campaign, "ensure_pretrained_weights")


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
        dataset=replace(configs[1].dataset, bundle_manifest_sha256="0" * 64),
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
            ComparisonResult("comparison", tmp_path, tmp_path, tmp_path, 8, {}),
            tmp_path / "execution.log",
            roots=_plan(tmp_path).roots,
        )
