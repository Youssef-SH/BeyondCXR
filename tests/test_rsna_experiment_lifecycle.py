from __future__ import annotations

import io
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from beyondcxr.data.hashing import sha256_file
from beyondcxr.data.rsna_artifacts import build_and_write
from beyondcxr.data.rsna_metadata_preprocess import SOURCE_FEATURES
from beyondcxr.training.config import load_experiment_config, with_runtime
from beyondcxr.training.rsna_datasets import RsnaDataset
from beyondcxr.training.rsna_evaluate import (
    evaluate_model_package,
)
from beyondcxr.training.rsna_evaluate import (
    main as evaluate_main,
)
from beyondcxr.training.rsna_evaluation_result import (
    CompletedRsnaEvaluation,
    validate_rsna_evaluation,
)
from beyondcxr.training.rsna_interfaces import DatasetLineage, DatasetPartition, DatasetRunData
from beyondcxr.training.rsna_train_metadata import train_metadata_experiment, validate_report_set
from beyondcxr.utils.mlflow_utils import configure_mlflow
from beyondcxr.utils.operational_logging import configure_logging
from beyondcxr.utils.private_predictions import validate_prediction_evidence
from beyondcxr.utils.rsna_model_publication import (
    model_package_id,
    validate_published_model,
)

_SHA256 = "a" * 64
_SPLIT_ASSIGNMENT_ID = "split-assignment-" + "b" * 64
_RESULT_PACKAGE_ID = "model-package-" + "c" * 64
_RESULT_PREDICTION_ID = "prediction-" + "d" * 64
_RESULT_EVALUATION_ID = "evaluation-" + "e" * 64


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
    split_assignment_id: str = _SPLIT_ASSIGNMENT_ID,
    bundle_manifest_sha256: str | None = None,
    calibration_bins: int | None = None,
    sensitivity_target: float | None = None,
    logistic_c: float | None = None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    document = yaml.safe_load((Path("configs") / filename).read_text(encoding="utf-8"))
    document["dataset"]["bundle_id"] = bundle_id
    document["dataset"]["split_assignment_id"] = split_assignment_id
    if bundle_manifest_sha256 is not None:
        document["dataset"]["bundle_manifest_sha256"] = bundle_manifest_sha256
    if calibration_bins is not None:
        document["evaluation"]["calibration_bins"] = calibration_bins
    if sensitivity_target is not None:
        document["evaluation"]["sensitivity_target"] = sensitivity_target
    if logistic_c is not None:
        document["training"]["parameters"]["C"] = logistic_c
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return with_runtime(
        load_experiment_config(path),
        seed=42,
        report_directory=tmp_path / "reports",
        model_directory=tmp_path / "models" / "rsna",
        private_output_directory=tmp_path / "private",
        manifest_directory=manifest_directory,
    )


@pytest.fixture(autouse=True)
def _small_operational_latency_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("beyondcxr.training.rsna_train_metadata.LATENCY_WARMUP_CALLS", 1)
    monkeypatch.setattr("beyondcxr.training.rsna_train_metadata.LATENCY_MEASURED_CALLS", 3)
    monkeypatch.setattr("beyondcxr.training.rsna_evaluate.LATENCY_WARMUP_CALLS", 1)
    monkeypatch.setattr("beyondcxr.training.rsna_evaluate.LATENCY_MEASURED_CALLS", 3)


def _tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def _client(tmp_path: Path):
    return configure_mlflow(tracking_uri=_tracking_uri(tmp_path))


def _train(config, tmp_path: Path):
    return train_metadata_experiment(config, tracking_uri=_tracking_uri(tmp_path))


def _install_dataset(monkeypatch: pytest.MonkeyPatch) -> tuple[DatasetRunData, DatasetPartition]:
    lineage = DatasetLineage(
        bundle_id="bundle-" + "a" * 64,
        split_assignment_id=_SPLIT_ASSIGNMENT_ID,
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
        "beyondcxr.training.rsna_train_metadata.git_revision",
        lambda: ("commit-synthetic", False),
    )
    monkeypatch.setattr("beyondcxr.training.rsna_train_metadata.uv_lock_sha256", lambda: _SHA256)


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
    evaluation = evaluate_model_package(
        training.model_package_id,
        evaluation_config=config,
        tracking_uri=_tracking_uri(tmp_path),
        model_directory=config.runtime.model_directory,
        report_directory=config.runtime.report_directory,
    )
    assert test_loads == 1
    training_run = mlflow.get_run(training.run_id)
    evaluation_run = mlflow.get_run(evaluation.mlflow_run_id)
    assert training_run.info.status == "FINISHED"
    assert evaluation_run.info.status == "FINISHED"
    assert training_run.data.tags["run_complete"] == "true"
    assert evaluation_run.data.tags["run_complete"] == "true"
    assert training_run.data.tags["evaluation_scope"] == "validation"
    assert evaluation_run.data.tags["evaluation_scope"] == "test"
    assert evaluation_run.data.tags["package_id"] == training.model_package_id
    assert evaluation_run.data.tags["prediction_id"] == evaluation.prediction_id
    assert evaluation_run.data.tags["evaluation_id"] == evaluation.evaluation_id
    assert {key for key in training_run.data.tags if not key.startswith("mlflow.")} == {
        "run_kind",
        "evaluation_scope",
        "dataset_id",
        "task_id",
        "family_id",
        "modalities",
        "bundle_id",
        "bundle_manifest_sha256",
        "split_assignment_id",
        "seed",
        "git_commit",
        "dependency_lock_sha256",
        "config_source_sha256",
        "config_semantic_sha256",
        "package_kind",
        "package_id",
        "run_complete",
    }
    assert {key for key in evaluation_run.data.tags if not key.startswith("mlflow.")} == {
        "run_kind",
        "evaluation_scope",
        "dataset_id",
        "task_id",
        "family_id",
        "modalities",
        "bundle_id",
        "bundle_manifest_sha256",
        "split_assignment_id",
        "seed",
        "git_commit",
        "dependency_lock_sha256",
        "config_source_sha256",
        "config_semantic_sha256",
        "package_kind",
        "package_id",
        "prediction_id",
        "evaluation_id",
        "run_complete",
    }
    assert "test_average_precision" not in training_run.data.metrics
    assert "test_average_precision" in evaluation_run.data.metrics

    manifest = validate_published_model(training.model_path.parent)
    assert {path.name for path in training.model_path.parent.iterdir()} == {
        "model.skops",
        "resolved_config.yaml",
        "manifest.json",
    }
    assert set(manifest["thresholds"]) == {"youden_j", "target_sensitivity"}
    assert manifest["model_package_id"] == training.model_package_id
    assert training_run.data.tags["package_id"] == training.model_package_id
    evidence = validate_prediction_evidence(
        evaluation.private_prediction_directory,
        expected_prediction_id=evaluation.prediction_id,
        expected_model_package_id=training.model_package_id,
    )
    scientific_result = validate_rsna_evaluation(
        evaluation.artifact_directory,
        private_root=evaluation.private_prediction_directory.parents[2],
        model_root=config.runtime.model_directory,
        expected_evaluation_id=evaluation.evaluation_id,
    )
    assert evidence.manifest["scope"] == "test"
    assert evidence.manifest["split_assignment_id"] == _SPLIT_ASSIGNMENT_ID
    assert scientific_result.manifest["prediction_id"] == evidence.prediction_id
    assert scientific_result.manifest["model_package_id"] == training.model_package_id
    assert scientific_result.manifest["claims"]["evaluation_scope"] == "test"
    assert training.model_path.parent.parent.name == "packages"
    assert evaluation.private_prediction_directory.parent.name == "rsna"
    assert evaluation.artifact_directory.parent.name == "evaluations"
    assert evaluation.artifact_directory.name == evaluation.evaluation_id
    validate_report_set(training.artifact_directory)
    validate_report_set(evaluation.artifact_directory / "derivatives")
    metrics = json.loads(
        (evaluation.artifact_directory / "derivatives" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["evaluation_scope"] == "test"

    client = _client(tmp_path)
    assert {item.path for item in client.list_artifacts(training.run_id)} == {"config"}
    assert client.list_artifacts(evaluation.mlflow_run_id) == []
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
        and "artifact=evaluation_result" in line
        and f"run_id={evaluation.mlflow_run_id}" in line
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
        if "event=run_finished" in line and f"run_id={evaluation.mlflow_run_id}" in line
    )
    assert test_publication < evaluation_run_finished


def test_explicit_evaluation_policy_controls_identity_and_package_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config_15 = _config(tmp_path / "config-15", calibration_bins=15)
    config_20 = _config(tmp_path / "config-20", calibration_bins=20)
    training_15 = _train(config_15, tmp_path / "config-15")
    training_20 = _train(config_20, tmp_path / "config-20")
    assert training_15.model_package_id == training_20.model_package_id

    results = {}
    for archive, training, tracking_uri in (
        ("archive-15", training_15, _tracking_uri(tmp_path / "config-15")),
        ("archive-20", training_20, _tracking_uri(tmp_path / "config-20")),
    ):
        model_root = training.model_path.parent.parent.parent
        for bins, evaluation_config in ((15, config_15), (20, config_20)):
            result = evaluate_model_package(
                training.model_package_id,
                evaluation_config=evaluation_config,
                tracking_uri=tracking_uri,
                model_directory=model_root,
                private_output_directory=tmp_path / archive / "private",
                report_directory=tmp_path / archive / "reports",
            )
            results[archive, bins] = validate_rsna_evaluation(
                result.artifact_directory,
                private_root=tmp_path / archive / "private",
                model_root=model_root,
            ).manifest

    assert results["archive-15", 15]["evaluation_policy"]["calibration_bins"] == 15
    assert results["archive-15", 20]["evaluation_policy"]["calibration_bins"] == 20
    assert results["archive-15", 15]["evaluation_id"] != results["archive-15", 20]["evaluation_id"]
    assert results["archive-15", 15]["evaluation_id"] == results["archive-20", 15]["evaluation_id"]
    assert results["archive-15", 20]["evaluation_id"] == results["archive-20", 20]["evaluation_id"]

    incompatible = _config(tmp_path / "incompatible", calibration_bins=15, logistic_c=2.0)
    monkeypatch.setattr(
        RsnaDataset,
        "load_test",
        lambda self, config: pytest.fail("held-out test must not be accessed"),
    )
    with pytest.raises(ValueError, match="incompatible"):
        evaluate_model_package(
            training_15.model_package_id,
            evaluation_config=incompatible,
            tracking_uri=_tracking_uri(tmp_path / "config-15"),
            model_directory=training_15.model_path.parent.parent.parent,
        )


def test_evaluation_rejects_sensitivity_target_mismatch_before_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    training_config = _config(tmp_path / "training", sensitivity_target=0.9)
    evaluation_config = _config(tmp_path / "evaluation", sensitivity_target=0.85)
    training = _train(training_config, tmp_path / "training")
    monkeypatch.setattr(
        RsnaDataset,
        "load_test",
        lambda self, config: pytest.fail("held-out test must not be accessed"),
    )

    with pytest.raises(ValueError, match="sensitivity target differs from the frozen"):
        evaluate_model_package(
            training.model_package_id,
            evaluation_config=evaluation_config,
            tracking_uri=_tracking_uri(tmp_path / "training"),
            model_directory=training.model_path.parent.parent.parent,
        )


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
        "beyondcxr.training.rsna_train_metadata.get_model", lambda _: FailingModel()
    )
    with pytest.raises(RuntimeError):
        _train(config, tmp_path)

    packages = config.runtime.model_directory / "packages"
    assert not packages.exists() or not any(packages.iterdir())
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
    assert run.data.tags["split_assignment_id"] == _SPLIT_ASSIGNMENT_ID
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
        "beyondcxr.utils.mlflow_utils.mlflow.set_tag",
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
        "beyondcxr.training.rsna_train_metadata.publish_model_package",
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
    from beyondcxr.training import rsna_train_metadata

    write_reports = rsna_train_metadata.write_run_reports

    def write_incomplete_reports(*args, **kwargs):
        write_reports(*args, **kwargs)
        Path(args[0], "calibration_curve.png").unlink()

    monkeypatch.setattr(rsna_train_metadata, "write_run_reports", write_incomplete_reports)
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
        "beyondcxr.training.rsna_train_metadata.publish_directory",
        lambda *args: (_ for _ in ()).throw(RuntimeError("report publication failed")),
    )

    with pytest.raises(RuntimeError):
        _train(config, tmp_path)

    run = _client(tmp_path).search_runs(
        [mlflow.get_experiment_by_name(config.runtime.experiment_name).experiment_id]
    )[0]
    assert run.info.status == "FAILED"
    assert run.data.tags["run_complete"] != "true"
    packages = list((config.runtime.model_directory / "packages").glob("model-package-*"))
    assert len(packages) == 1
    validate_published_model(packages[0])


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


def test_evaluation_uses_model_package_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    training = _train(_config(tmp_path), tmp_path)
    package = validate_published_model(training.model_path.parent)
    assert package["model_package_id"] == training.model_package_id
    assert mlflow.get_run(training.run_id).data.tags["package_id"] == training.model_package_id


@pytest.mark.parametrize("policy", ["youden_j", "target_sensitivity"])
def test_frozen_threshold_state_changes_model_package_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    training = _train(config, tmp_path)
    manifest_path = training.model_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["thresholds"][policy] = 0.0 if manifest["thresholds"][policy] > 0.5 else 1.0
    manifest["model_package_id"] = model_package_id(manifest)
    assert manifest["model_package_id"] != training.model_package_id


def test_evaluator_rejects_lightgbm_best_iteration_mismatch_before_test_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path, filename="rsna_metadata_lightgbm.yaml")
    training = _train(config, tmp_path)
    manifest_path = training.model_path.parent / "manifest.json"
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
        evaluate_model_package(
            training.model_package_id,
            evaluation_config=config,
            tracking_uri=_tracking_uri(tmp_path),
            model_directory=config.runtime.model_directory,
        )


def test_evaluation_report_publication_failure_does_not_complete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    training = _train(config, tmp_path)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_evaluate.publish_rsna_evaluation",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("report publication failed")),
    )

    with pytest.raises(RuntimeError):
        evaluate_model_package(
            training.model_package_id,
            evaluation_config=config,
            tracking_uri=_tracking_uri(tmp_path),
            model_directory=config.runtime.model_directory,
        )

    runs = _client(tmp_path).search_runs(
        [mlflow.get_experiment_by_name(config.runtime.experiment_name).experiment_id],
        filter_string="tags.run_kind = 'test_evaluation'",
    )
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
    assert runs[0].data.tags["run_complete"] != "true"


def test_operational_completion_failure_preserves_published_scientific_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_dataset(monkeypatch)
    _fixed_provenance(monkeypatch)
    config = _config(tmp_path)
    training = _train(config, tmp_path)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_evaluate.mlflow.log_metrics",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ledger write failed")),
    )

    with pytest.raises(RuntimeError, match="ledger write failed"):
        evaluate_model_package(
            training.model_package_id,
            evaluation_config=config,
            tracking_uri=_tracking_uri(tmp_path),
            model_directory=config.runtime.model_directory,
            private_output_directory=config.runtime.private_output_directory,
            report_directory=config.runtime.report_directory,
        )

    evaluation_root = config.runtime.report_directory / "rsna" / "evaluations"
    prediction_root = config.runtime.private_output_directory / "predictions" / "rsna"
    evaluations = list(evaluation_root.glob("evaluation-*"))
    predictions = list(prediction_root.glob("prediction-*"))
    assert len(evaluations) == len(predictions) == 1
    evidence = validate_prediction_evidence(predictions[0])
    result = validate_rsna_evaluation(
        evaluations[0],
        private_root=config.runtime.private_output_directory,
        model_root=config.runtime.model_directory,
    )
    assert result.manifest["prediction_id"] == evidence.prediction_id
    runs = _client(tmp_path).search_runs(
        [mlflow.get_experiment_by_name(config.runtime.experiment_name).experiment_id],
        filter_string="tags.run_kind = 'test_evaluation'",
    )
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
    assert runs[0].data.tags["run_complete"] != "true"


@pytest.mark.parametrize(
    "model_package_id",
    [
        "/tmp/model-package-" + "a" * 64,
        "../model-package-" + "a" * 64,
        "model-package-directory/" + "a" * 64,
        "model-package-directory\\" + "a" * 64,
        "package-" + "a" * 64,
        "model-package-",
        "model-package-" + "a" * 63,
        "model-package-" + "a" * 65,
        "model-package-" + "g" * 64,
        "model-package-" + "A" * 64,
    ],
)
def test_evaluator_rejects_noncanonical_package_identity_before_filesystem_access(
    model_package_id: str,
) -> None:
    class UnavailableModelRoot:
        def __fspath__(self) -> str:
            raise AssertionError("model root must not be inspected")

    with pytest.raises(ValueError, match="model package"):
        evaluate_model_package(
            model_package_id,
            evaluation_config=load_experiment_config("configs/rsna_metadata_logistic.yaml"),
            model_directory=UnavailableModelRoot(),
        )


def test_evaluator_cli_serializes_completed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = CompletedRsnaEvaluation(
        evaluation_id=_RESULT_EVALUATION_ID,
        prediction_id=_RESULT_PREDICTION_ID,
        model_package_id=_RESULT_PACKAGE_ID,
        mlflow_run_id="evaluation-run",
        artifact_directory=tmp_path / "reports",
        private_prediction_directory=tmp_path / "private",
        average_precision=0.75,
    )

    def evaluate(
        package_id: str,
        *,
        evaluation_config,
        tracking_uri: str,
        model_directory: Path,
        private_output_directory: Path | None,
        report_directory: Path | None,
    ) -> CompletedRsnaEvaluation:
        assert package_id == _RESULT_PACKAGE_ID
        assert evaluation_config.source_path == Path("configs/rsna_metadata_logistic.yaml")
        assert tracking_uri == "sqlite:///test.db"
        assert model_directory == Path("models/rsna")
        assert private_output_directory is None
        assert report_directory is None
        return result

    monkeypatch.setattr("beyondcxr.training.rsna_evaluate.evaluate_model_package", evaluate)

    with pytest.raises(SystemExit):
        evaluate_main(["--package-id", _RESULT_PACKAGE_ID])

    assert (
        evaluate_main(
            [
                "--package-id",
                _RESULT_PACKAGE_ID,
                "--config",
                "configs/rsna_metadata_logistic.yaml",
                "--tracking-uri",
                "sqlite:///test.db",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "model_package_id": _RESULT_PACKAGE_ID,
        "prediction_id": _RESULT_PREDICTION_ID,
        "evaluation_id": _RESULT_EVALUATION_ID,
        "mlflow_run_id": "evaluation-run",
        "test_average_precision": 0.75,
        "artifact_directory": (tmp_path / "reports").as_posix(),
    }


@pytest.mark.integration
def test_synthetic_raw_source_to_bundle_training_and_explicit_test_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = _write_raw_source(tmp_path / "raw")
    manifest_root = tmp_path / "data" / "manifests"
    bundle = build_and_write(raw_root, manifest_root)
    bundle_manifest = json.loads(bundle.paths.metadata_path.read_text(encoding="utf-8"))
    _fixed_provenance(monkeypatch)
    config = _config(
        tmp_path,
        bundle_id=bundle.paths.bundle_id,
        manifest_directory=manifest_root,
        split_assignment_id=bundle_manifest["membership"]["split"]["split_assignment_id"],
        bundle_manifest_sha256=sha256_file(bundle.paths.metadata_path),
    )

    tracking_uri = _tracking_uri(tmp_path)
    training = train_metadata_experiment(config, tracking_uri=tracking_uri)
    monkeypatch.chdir(tmp_path)
    evaluation = evaluate_model_package(
        training.model_package_id,
        evaluation_config=config,
        tracking_uri=tracking_uri,
        model_directory=config.runtime.model_directory,
        report_directory=config.runtime.report_directory,
    )
    evaluation_run_id = evaluation.mlflow_run_id
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
