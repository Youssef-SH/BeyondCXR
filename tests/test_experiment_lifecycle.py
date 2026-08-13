from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from radfusion.data.rsna_artifacts import build_and_write
from radfusion.data.rsna_metadata_preprocess import SOURCE_FEATURES
from radfusion.training.config import load_experiment_config, with_runtime
from radfusion.training.evaluate import (
    TestEvaluationResult as EvaluationResult,
)
from radfusion.training.evaluate import (
    _verify_training_lineage,
    evaluate_training_run,
)
from radfusion.training.evaluate import (
    main as evaluate_main,
)
from radfusion.training.rsna_datasets import RsnaDataset
from radfusion.training.rsna_interfaces import DatasetLineage, DatasetPartition, DatasetRunData
from radfusion.training.rsna_train_metadata import train_configured_experiment, validate_report_set
from radfusion.utils.mlflow_utils import configure_mlflow
from radfusion.utils.model_publication import (
    model_package_id,
    threshold_contract,
    validate_published_model,
)
from radfusion.utils.operational_logging import configure_logging

_SHA256 = "a" * 64


def _partition(name: str) -> DatasetPartition:
    size = 12
    targets = np.asarray([0, 1] * (size // 2), dtype=np.int8)
    features = pd.DataFrame(
        {
            "age_years": np.linspace(25.0, 75.0, size),
            "age_is_implausible": [False] * size,
            "sex": ["F", "M"] * (size // 2),
            "view_position": ["PA", "AP"] * (size // 2),
            "pixel_spacing_row_mm": np.linspace(0.14, 0.19, size),
            "pixel_spacing_col_mm": np.linspace(0.14, 0.19, size),
        },
        columns=SOURCE_FEATURES,
    )
    return DatasetPartition(
        features=features,
        targets=targets,
        sample_ids=tuple(f"rsna:{name}-{index}" for index in range(size)),
        patient_ids=tuple(f"{name}-{index}" for index in range(size)),
        partition=name,
    )


def _config(
    tmp_path: Path,
    *,
    bundle_id: str = "bundle-" + "a" * 64,
    manifest_directory: Path | None = None,
    filename: str = "rsna_metadata_logistic.yaml",
):
    document = yaml.safe_load((Path("configs") / filename).read_text(encoding="utf-8"))
    document["dataset"]["bundle_id"] = bundle_id
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return with_runtime(
        load_experiment_config(path),
        seed=42,
        report_directory=tmp_path / "reports",
        model_directory=tmp_path / "models" / "rsna",
        manifest_directory=manifest_directory,
    )


@pytest.fixture(autouse=True)
def _small_operational_latency_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("radfusion.training.rsna_train_metadata.LATENCY_WARMUP_CALLS", 1)
    monkeypatch.setattr("radfusion.training.rsna_train_metadata.LATENCY_MEASURED_CALLS", 3)
    monkeypatch.setattr("radfusion.training.evaluate.LATENCY_WARMUP_CALLS", 1)
    monkeypatch.setattr("radfusion.training.evaluate.LATENCY_MEASURED_CALLS", 3)


def _tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def _client(tmp_path: Path):
    return configure_mlflow(tracking_uri=_tracking_uri(tmp_path))


def _train(config, tmp_path: Path):
    return train_configured_experiment(config, tracking_uri=_tracking_uri(tmp_path))


def _install_dataset(monkeypatch: pytest.MonkeyPatch) -> tuple[DatasetRunData, DatasetPartition]:
    lineage = DatasetLineage(
        bundle_id="bundle-" + "a" * 64,
        split_assignment_id="assignment-synthetic",
        label_policy_version="label-synthetic",
        task_id="pneumonia",
    )
    data = DatasetRunData(
        train=_partition("train"),
        validation=_partition("validation"),
        lineage=lineage,
    )
    test = _partition("test")
    monkeypatch.setattr(
        RsnaDataset,
        "load_train_validation",
        lambda self, config: data,
    )
    monkeypatch.setattr(
        RsnaDataset,
        "load_test",
        lambda self, config: (test, lineage),
    )
    monkeypatch.setattr(
        RsnaDataset,
        "load_lineage",
        lambda self, config: lineage,
    )
    return data, test


def _fixed_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "radfusion.training.rsna_train_metadata.git_revision",
        lambda: ("commit-synthetic", False),
    )
    monkeypatch.setattr("radfusion.training.rsna_train_metadata.uv_lock_sha256", lambda: _SHA256)
    monkeypatch.setattr(
        "radfusion.training.evaluate.git_revision",
        lambda: ("commit-synthetic", False),
    )
    monkeypatch.setattr("radfusion.training.evaluate.uv_lock_sha256", lambda: _SHA256)


@pytest.mark.parametrize("filename", ["rsna_metadata_logistic.yaml", "rsna_metadata_lightgbm.yaml"])
def test_train_then_explicit_test_evaluation_uses_separate_partitions_and_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    log_stream = io.StringIO()
    configure_logging("INFO", stream=log_stream)
    data, _ = _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path, filename=filename)
    test_loads = 0
    original_test_loader = RsnaDataset.load_test

    def count_test_load(self, dataset_config):
        nonlocal test_loads
        test_loads += 1
        return original_test_loader(self, dataset_config)

    monkeypatch.setattr(RsnaDataset, "load_test", count_test_load)
    training = _train(config, tmp_path)
    assert test_loads == 0
    assert tuple(data.train.features.columns) == SOURCE_FEATURES
    assert not hasattr(data, "test")

    configure_mlflow(tracking_uri=f"sqlite:///{(tmp_path / 'other.db').as_posix()}")
    evaluation = evaluate_training_run(
        training.run_id,
        tracking_uri=_tracking_uri(tmp_path),
    )
    assert test_loads == 1
    training_run = mlflow.get_run(training.run_id)
    evaluation_run = mlflow.get_run(evaluation.run_id)
    assert training_run.info.status == "FINISHED"
    assert evaluation_run.info.status == "FINISHED"
    assert training_run.data.tags["run_complete"] == "true"
    assert evaluation_run.data.tags["run_complete"] == "true"
    assert training_run.data.tags["evaluation_scope"] == "validation"
    assert evaluation_run.data.tags["evaluation_scope"] == "test"
    assert evaluation_run.data.tags["source_training_run_id"] == training.run_id
    assert evaluation_run.data.tags["model_package_id"] == training.model_package_id
    assert evaluation_run.data.tags["git_commit"] == "commit-synthetic"
    assert evaluation_run.data.tags["git_dirty"] == "false"
    assert evaluation_run.data.tags["dependency_lock_sha256"] == _SHA256
    assert "test_average_precision" not in training_run.data.metrics
    assert "test_average_precision" in evaluation_run.data.metrics
    assert float(evaluation_run.data.tags["threshold_youden_j"]) == training.thresholds["youden_j"]
    assert evaluation_run.data.params["best_iteration"] == training_run.data.params.get(
        "best_iteration", "not_applicable"
    )

    manifest = validate_published_model(training.model_path.parent)
    assert {path.name for path in training.model_path.parent.iterdir()} == {
        "model.skops",
        "resolved_config.yaml",
        "model_manifest.json",
    }
    assert set(manifest["thresholds"]) == {"youden_j", "target_sensitivity"}
    assert manifest["model_package_id"] == training.model_package_id
    assert training_run.data.tags["model_package_id"] == training.model_package_id
    validate_report_set(training.artifact_directory)
    validate_report_set(evaluation.artifact_directory)
    metrics = json.loads(
        (evaluation.artifact_directory / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["evaluation_scope"] == "test"

    client = _client(tmp_path)
    assert {item.path for item in client.list_artifacts(training.run_id)} == {"config"}
    assert client.list_artifacts(evaluation.run_id) == []
    resolved_config = Path(
        client.download_artifacts(
            training.run_id,
            "config/resolved_config.yaml",
            tmp_path / "download",
        )
    )
    assert resolved_config.read_bytes() == config.source_bytes
    log_lines = log_stream.getvalue().splitlines()
    training_run_started = next(
        index
        for index, line in enumerate(log_lines)
        if "event=run_started" in line and f"run_id={training.run_id}" in line
    )
    dataset_phase_started = next(
        index
        for index, line in enumerate(log_lines)
        if "event=phase_started" in line
        and "phase=dataset_loading" in line
        and f"run_id={training.run_id}" in line
    )
    dataset_phase_completed = next(
        index
        for index, line in enumerate(log_lines)
        if "event=phase_completed" in line
        and "phase=dataset_loading" in line
        and f"run_id={training.run_id}" in line
    )
    package_publication = next(
        index
        for index, line in enumerate(log_lines)
        if "event=publication_completed" in line
        and "artifact=model_package" in line
        and f"run_id={training.run_id}" in line
    )
    validation_publication = next(
        index
        for index, line in enumerate(log_lines)
        if "event=publication_completed" in line
        and "artifact=validation_report" in line
        and f"run_id={training.run_id}" in line
    )
    test_publication = next(
        index
        for index, line in enumerate(log_lines)
        if "event=publication_completed" in line
        and "artifact=test_report" in line
        and f"run_id={evaluation.run_id}" in line
    )
    training_run_finished = next(
        index
        for index, line in enumerate(log_lines)
        if "event=run_finished" in line and f"run_id={training.run_id}" in line
    )
    assert (
        training_run_started
        < dataset_phase_started
        < dataset_phase_completed
        < package_publication
        < validation_publication
        < training_run_finished
    )
    evaluation_run_finished = next(
        index
        for index, line in enumerate(log_lines)
        if "event=run_finished" in line and f"run_id={evaluation.run_id}" in line
    )
    assert test_publication < evaluation_run_finished


def test_fit_failure_leaves_failed_mlflow_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_stream = io.StringIO()
    configure_logging("INFO", stream=log_stream)
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)

    class FailingModel:
        def fit(self, *args, **kwargs):
            assert mlflow.active_run() is not None
            raise RuntimeError("fit failed")

    monkeypatch.setattr(
        "radfusion.training.rsna_train_metadata.get_model", lambda _: FailingModel()
    )
    with pytest.raises(RuntimeError):
        _train(config, tmp_path)
    runs = _client(tmp_path).search_runs(
        [mlflow.get_experiment_by_name(config.runtime.experiment_name).experiment_id]
    )
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FAILED"
    assert run.data.tags["run_complete"] != "true"
    assert "level=ERROR event=run_failed" in log_stream.getvalue()
    assert "error_type=RuntimeError" in log_stream.getvalue()
    assert "event=run_finished" not in log_stream.getvalue()
    assert run.data.tags["split_assignment_id"] == "assignment-synthetic"
    assert run.data.tags["label_policy_version"] == "label-synthetic"
    downloaded = Path(
        _client(tmp_path).download_artifacts(
            run.info.run_id,
            "config/resolved_config.yaml",
            tmp_path / "failed-config-download",
        )
    )
    assert downloaded.read_bytes() == config.source_bytes


def test_run_start_precedes_post_creation_mlflow_metadata_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_stream = io.StringIO()
    configure_logging("INFO", stream=log_stream)
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    monkeypatch.setattr(
        "radfusion.utils.mlflow_utils.mlflow.set_tag",
        lambda key, value: (_ for _ in ()).throw(RuntimeError(f"metadata failed: {key}={value}")),
    )

    with pytest.raises(RuntimeError):
        _train(config, tmp_path)

    runs = _client(tmp_path).search_runs(
        [mlflow.get_experiment_by_name(config.runtime.experiment_name).experiment_id]
    )
    assert len(runs) == 1
    run_id = runs[0].info.run_id
    lines = log_stream.getvalue().splitlines()
    started = next(
        index
        for index, line in enumerate(lines)
        if "event=run_started" in line and f"run_id={run_id}" in line
    )
    failed = next(
        index
        for index, line in enumerate(lines)
        if "event=run_failed" in line and f"run_id={run_id}" in line
    )
    assert started < failed
    assert runs[0].info.status == "FAILED"


def test_required_model_publication_failure_leaves_failed_mlflow_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    monkeypatch.setattr(
        "radfusion.training.rsna_train_metadata.publish_model_run",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("publication failed")),
    )

    with pytest.raises(RuntimeError):
        _train(config, tmp_path)
    runs = _client(tmp_path).search_runs(
        [mlflow.get_experiment_by_name(config.runtime.experiment_name).experiment_id]
    )
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
    assert runs[0].data.tags["run_complete"] != "true"


def test_incomplete_report_set_fails_training_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    from radfusion.training import train_tabular

    write_reports = train_tabular.write_run_reports

    def write_incomplete_reports(*args, **kwargs):
        write_reports(*args, **kwargs)
        Path(args[0], "calibration_curve.png").unlink()

    monkeypatch.setattr(train_tabular, "write_run_reports", write_incomplete_reports)
    with pytest.raises(ValueError):
        _train(config, tmp_path)

    run = _client(tmp_path).search_runs(
        [mlflow.get_experiment_by_name(config.runtime.experiment_name).experiment_id]
    )[0]
    assert run.info.status == "FAILED"
    assert run.data.tags["run_complete"] != "true"


def test_training_report_publication_failure_does_not_complete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    monkeypatch.setattr(
        "radfusion.training.rsna_train_metadata.publish_directory",
        lambda *args: (_ for _ in ()).throw(RuntimeError("report publication failed")),
    )

    with pytest.raises(RuntimeError):
        _train(config, tmp_path)

    run = _client(tmp_path).search_runs(
        [mlflow.get_experiment_by_name(config.runtime.experiment_name).experiment_id]
    )[0]
    assert run.info.status == "FAILED"
    assert run.data.tags["run_complete"] != "true"


def test_training_archives_and_logs_loaded_config_bytes_after_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    original = config.source_bytes
    config.source_path.write_bytes(b"mutated-after-load: true\n")

    training = _train(config, tmp_path)

    assert (training.model_path.parent / "resolved_config.yaml").read_bytes() == original
    downloaded = Path(
        _client(tmp_path).download_artifacts(
            training.run_id,
            "config/resolved_config.yaml",
            tmp_path / "config-download",
        )
    )
    assert downloaded.read_bytes() == original


@pytest.mark.parametrize(
    "tag_name",
    [
        "dataset_bundle_id",
        "split_assignment_id",
        "task",
        "model",
        "seed",
        "git_commit",
        "config_source_sha256",
        "config_semantic_sha256",
        "dependency_lock_sha256",
        "local_model_sha256",
        "model_package_id",
        "git_dirty",
    ],
)
def test_evaluator_rejects_each_training_lineage_tag_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tag_name: str,
) -> None:
    config = _config(tmp_path)
    run_id = "training-run"
    model_path = config.runtime.model_directory / "runs" / run_id / "model.skops"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"serialized-model")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = {
        "training_mlflow_run_id": run_id,
        "config_source_sha256": config.config_source_sha256,
        "config_semantic_sha256": config.config_semantic_sha256,
        "bundle_id": config.dataset.bundle_id,
        "split_assignment_id": "assignment-synthetic",
        "task": config.task.task_id,
        "model": config.family.family_id,
        "seed": config.runtime.seed,
        "git_commit": "commit-synthetic",
        "git_dirty": False,
        "dependency_lock_sha256": _SHA256,
        "positive_class": 1,
        "model_sha256": model_hash,
        "model_package_id": "model-package-test",
        "threshold_contract": threshold_contract(
            sensitivity_target=config.evaluation.sensitivity_target
        ),
    }
    tags = {
        "dataset_bundle_id": manifest["bundle_id"],
        "split_assignment_id": manifest["split_assignment_id"],
        "task": manifest["task"],
        "model": config.family.family_id,
        "seed": str(manifest["seed"]),
        "git_commit": manifest["git_commit"],
        "git_dirty": str(manifest["git_dirty"]).lower(),
        "config_source_sha256": manifest["config_source_sha256"],
        "config_semantic_sha256": manifest["config_semantic_sha256"],
        "dependency_lock_sha256": manifest["dependency_lock_sha256"],
        "local_model_sha256": manifest["model_sha256"],
        "model_package_id": manifest["model_package_id"],
    }
    tags[tag_name] = "mismatch"
    run = SimpleNamespace(
        info=SimpleNamespace(run_id=run_id),
        data=SimpleNamespace(tags=tags),
    )
    with pytest.raises(ValueError):
        _verify_training_lineage(
            run,
            config,
            manifest,
            model_path,
            evaluator_commit="commit-synthetic",
            evaluator_dirty=False,
            evaluator_lock_hash=_SHA256,
        )


@pytest.mark.parametrize(
    "case",
    [
        "dirty_package",
        "dirty_evaluator",
        "commit_mismatch",
        "lock_mismatch",
    ],
)
def test_formal_evaluation_rejects_incompatible_source_before_test_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    if case == "dirty_package":
        monkeypatch.setattr(
            "radfusion.training.rsna_train_metadata.git_revision",
            lambda: ("commit-synthetic", True),
        )
    config = _config(tmp_path)
    training = _train(config, tmp_path)
    if case == "dirty_evaluator":
        monkeypatch.setattr(
            "radfusion.training.evaluate.git_revision",
            lambda: ("commit-synthetic", True),
        )
    elif case == "commit_mismatch":
        monkeypatch.setattr(
            "radfusion.training.evaluate.git_revision",
            lambda: ("different-commit", False),
        )
    elif case == "lock_mismatch":
        monkeypatch.setattr("radfusion.training.evaluate.uv_lock_sha256", lambda: "b" * 64)
    test_loads = 0

    def reject_test_load(self, dataset_config):
        nonlocal test_loads
        test_loads += 1
        raise AssertionError("test data must not be loaded")

    monkeypatch.setattr(RsnaDataset, "load_test", reject_test_load)
    with pytest.raises(ValueError):
        evaluate_training_run(training.run_id, tracking_uri=_tracking_uri(tmp_path))
    assert test_loads == 0


@pytest.mark.parametrize("policy", ["youden_j", "target_sensitivity"])
def test_evaluator_rejects_validation_threshold_mismatch_before_test_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    training = _train(config, tmp_path)
    manifest_path = training.model_path.parent / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["thresholds"][policy] = 0.0 if manifest["thresholds"][policy] > 0.5 else 1.0
    manifest["model_package_id"] = model_package_id(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _client(tmp_path).set_tag(training.run_id, "model_package_id", manifest["model_package_id"])
    test_loads = 0

    def reject_test_load(self, dataset_config):
        nonlocal test_loads
        test_loads += 1
        raise AssertionError("test data must not be loaded")

    monkeypatch.setattr(RsnaDataset, "load_test", reject_test_load)
    with pytest.raises(ValueError):
        evaluate_training_run(
            training.run_id,
            tracking_uri=_tracking_uri(tmp_path),
        )
    assert test_loads == 0


def test_evaluator_rejects_lightgbm_best_iteration_mismatch_before_test_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path, filename="rsna_metadata_lightgbm.yaml")
    training = _train(config, tmp_path)
    manifest_path = training.model_path.parent / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["best_iteration"] += 1
    manifest["model_package_id"] = model_package_id(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _client(tmp_path).set_tag(training.run_id, "model_package_id", manifest["model_package_id"])
    monkeypatch.setattr(
        RsnaDataset,
        "load_test",
        lambda self, dataset_config: (_ for _ in ()).throw(
            AssertionError("test data must not be loaded")
        ),
    )

    with pytest.raises(ValueError):
        evaluate_training_run(
            training.run_id,
            tracking_uri=_tracking_uri(tmp_path),
        )


def test_evaluation_report_publication_failure_does_not_complete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    training = _train(config, tmp_path)
    monkeypatch.setattr(
        "radfusion.training.evaluate.publish_directory",
        lambda *args: (_ for _ in ()).throw(RuntimeError("report publication failed")),
    )

    with pytest.raises(RuntimeError):
        evaluate_training_run(
            training.run_id,
            tracking_uri=_tracking_uri(tmp_path),
        )

    runs = _client(tmp_path).search_runs(
        [mlflow.get_experiment_by_name(config.runtime.experiment_name).experiment_id],
        filter_string="tags.run_kind = 'test_evaluation'",
    )
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
    assert runs[0].data.tags["run_complete"] != "true"


def test_evaluator_cli_serializes_completed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = EvaluationResult(
        run_id="evaluation-run",
        training_run_id="training-run",
        artifact_directory=tmp_path / "reports",
        average_precision=0.75,
    )

    def evaluate(
        run_id: str,
        *,
        tracking_uri: str,
        private_output_directory: Path | None,
    ) -> EvaluationResult:
        assert run_id == "training-run"
        assert tracking_uri == "sqlite:///test.db"
        assert private_output_directory is None
        return result

    monkeypatch.setattr("radfusion.training.evaluate.evaluate_training_run", evaluate)

    assert evaluate_main(["--run-id", "training-run", "--tracking-uri", "sqlite:///test.db"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "training_run_id": "training-run",
        "test_evaluation_run_id": "evaluation-run",
        "test_average_precision": 0.75,
        "artifact_directory": (tmp_path / "reports").as_posix(),
    }


def test_clean_and_purge_generated_have_distinct_scopes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    for path in (
        "reports/keep.txt",
        "models/keep.txt",
        "mlartifacts/keep.txt",
        "mlruns/keep.txt",
        "mlflow.db",
        "data/manifests/rsna/bundles/bundle-test/bundle.txt",
        "data/manifests/rsna/CURRENT",
        ".pytest_cache/cache.txt",
        "src/__pycache__/module.pyc",
        "reports/.run-staging-test/partial.txt",
        "reports/.comparison.tmp",
        ".git/keep.txt",
        ".venv/__pycache__/keep.pyc",
        "data/raw/rsna/__pycache__/keep.pyc",
    ):
        target = workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("keep\n", encoding="utf-8")
    makefile = Path("Makefile").resolve()

    clean = subprocess.run(
        ["make", "-f", str(makefile), "-C", str(workspace), "clean"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert clean.returncode == 0, clean.stderr
    for path in (
        "reports/keep.txt",
        "models/keep.txt",
        "mlartifacts/keep.txt",
        "mlruns/keep.txt",
        "mlflow.db",
        "data/manifests/rsna/bundles/bundle-test/bundle.txt",
        "data/manifests/rsna/CURRENT",
        ".git/keep.txt",
        ".venv/__pycache__/keep.pyc",
        "data/raw/rsna/__pycache__/keep.pyc",
    ):
        assert (workspace / path).is_file()
    for path in (
        ".pytest_cache",
        "src/__pycache__",
        "reports/.run-staging-test",
        "reports/.comparison.tmp",
    ):
        assert not (workspace / path).exists()

    purge = subprocess.run(
        ["make", "-f", str(makefile), "-C", str(workspace), "purge-generated"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert purge.returncode == 0, purge.stderr
    for path in ("reports", "models", "mlartifacts", "mlruns", "mlflow.db"):
        assert not (workspace / path).exists()
    assert not (workspace / "data/manifests/rsna/CURRENT").exists()
    assert not (workspace / "data/manifests/rsna/bundles/bundle-test").exists()
    for path in (
        ".git/keep.txt",
        ".venv/__pycache__/keep.pyc",
        "data/raw/rsna/__pycache__/keep.pyc",
    ):
        assert (workspace / path).is_file()


@pytest.mark.integration
def test_synthetic_raw_source_to_bundle_training_and_explicit_test_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = _write_raw_source(tmp_path / "raw")
    manifest_root = tmp_path / "manifests"
    bundle = build_and_write(raw_root, manifest_root)
    _fixed_provenance(monkeypatch)
    config = _config(
        tmp_path,
        bundle_id=bundle.paths.bundle_id,
        manifest_directory=manifest_root,
    )

    tracking_uri = _tracking_uri(tmp_path)
    training = train_configured_experiment(config, tracking_uri=tracking_uri)
    evaluation = evaluate_training_run(training.run_id, tracking_uri=tracking_uri)
    evaluation_run_id = evaluation.run_id
    evaluation_directory = evaluation.artifact_directory

    assert training.artifact_directory.is_dir()
    assert evaluation_directory.is_dir()
    assert mlflow.get_run(training.run_id).data.tags["evaluation_scope"] == "validation"
    assert _client(tmp_path).get_run(evaluation_run_id).data.tags["evaluation_scope"] == "test"


def _write_raw_source(root: Path) -> Path:
    images = root / "stage_2_train_images"
    images.mkdir(parents=True)
    labels = []
    classes = []
    for target in (0, 1):
        for index in range(6):
            patient_id = f"patient-{target}-{index}"
            _write_dicom(images / f"{patient_id}.dcm", patient_id, age=35 + target * 20 + index)
            labels.append(
                {
                    "patientId": patient_id,
                    "x": 10 if target else None,
                    "y": 20 if target else None,
                    "width": 30 if target else None,
                    "height": 40 if target else None,
                    "Target": target,
                }
            )
            classes.append(
                {
                    "patientId": patient_id,
                    "class": "Lung Opacity" if target else "Normal",
                }
            )
    pd.DataFrame(labels).to_csv(root / "stage_2_train_labels.csv", index=False)
    pd.DataFrame(classes).to_csv(root / "stage_2_detailed_class_info.csv", index=False)
    return root


def _write_dicom(path: Path, patient_id: str, *, age: int) -> None:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.PatientID = patient_id
    dataset.PatientAge = f"{age:03d}Y"
    dataset.PatientSex = "F" if age % 2 else "M"
    dataset.ViewPosition = "PA"
    dataset.PixelSpacing = [0.168, 0.168]
    dataset.Rows = 1024
    dataset.Columns = 1024
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.Modality = "CR"
    dataset.BodyPartExamined = "CHEST"
    dataset.save_as(path)
