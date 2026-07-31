from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import mlflow
import pytest
import yaml
from mlflow.exceptions import MlflowException

from radfusion.training.completed_runs import (
    SEED_SPECIFIC_METRIC_NAMES,
    require_completed_run,
    validated_image_test_metrics,
)
from radfusion.training.config import image_semantic_config_sha256, load_experiment_config
from radfusion.training.summarize_seeds import summarize_seed_runs
from radfusion.utils.mlflow_utils import configure_mlflow

_CONFIG_PATHS = {
    17: Path("configs/image_densenet_seed17.yaml"),
    42: Path("configs/image_densenet_seed42.yaml"),
    2026: Path("configs/image_densenet_seed2026.yaml"),
}
_BUNDLE_ID = "build-cfe6e3818fbc179af2cd6237641cdde716159a18c2d1430644bba71058adead0"
_BUNDLE_MANIFEST_SHA256 = "c" * 64
_SPLIT_ASSIGNMENT_ID = "split-assignment-test"
_LABEL_POLICY_VERSION = "label-policy-test"
_GIT_COMMIT = "d" * 40
_LOCK_SHA256 = "e" * 64
_CHECKPOINT_SHA256 = "f" * 64


@dataclass
class _Family:
    tracking_uri: str
    test_ids: dict[int, str]
    training_ids: dict[int, str]
    manifests: dict[Path, dict[str, object]]


def _tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def _metrics(average_precision: float) -> dict[str, float]:
    return {
        "average_precision": average_precision,
        "roc_auc": 0.70,
        "brier_score": 0.20,
        "expected_calibration_error": 0.10,
        "calibration_slope": 1.10,
        "calibration_intercept": -0.10,
        "youden_j_threshold": 0.50,
        "target_sensitivity_threshold": 0.30,
        "youden_j_precision": 0.60,
        "youden_j_recall": 0.70,
        "youden_j_specificity": 0.80,
        "youden_j_f1": 0.65,
        "youden_j_true_negative": 80.0,
        "youden_j_false_positive": 20.0,
        "youden_j_false_negative": 30.0,
        "youden_j_true_positive": 70.0,
        "target_sensitivity_precision": 0.40,
        "target_sensitivity_recall": 0.90,
        "target_sensitivity_specificity": 0.30,
        "target_sensitivity_f1": 0.55,
        "target_sensitivity_true_negative": 30.0,
        "target_sensitivity_false_positive": 70.0,
        "target_sensitivity_false_negative": 10.0,
        "target_sensitivity_true_positive": 90.0,
        "model_size_mib": 2.0,
    }


def _manifest(
    *,
    training_run_id: str,
    package_id: str,
    seed: int,
    source_config_sha256: str,
    semantic_config_sha256: str,
) -> dict[str, object]:
    return {
        "model_package_schema_version": 1,
        "model_package_id": package_id,
        "training_mlflow_run_id": training_run_id,
        "modality": "image",
        "model": "image_densenet",
        "task": "pneumonia",
        "positive_class": 1,
        "bundle_id": _BUNDLE_ID,
        "bundle_manifest_sha256": _BUNDLE_MANIFEST_SHA256,
        "split_assignment_id": _SPLIT_ASSIGNMENT_ID,
        "label_policy_version": _LABEL_POLICY_VERSION,
        "source_config_sha256": source_config_sha256,
        "semantic_config_sha256": semantic_config_sha256,
        "checkpoint_sha256": _CHECKPOINT_SHA256,
        "source_provenance": {
            "git_commit": _GIT_COMMIT,
            "git_dirty": False,
            "dependency_lock_sha256": _LOCK_SHA256,
            "python_version": "3.13",
            "torch_version": "test",
            "torchvision_version": "test",
            "torchxrayvision_version": "test",
        },
        "model_identity": {
            "registry_key": "image_densenet",
            "modality": "image",
            "encoder_architecture": "densenet121",
            "image_size": 224,
            "embedding_dimension": 1024,
            "classifier_output_dimension": 1,
            "pretrained_weight": {
                "declared_name": "densenet121-res224-chex",
                "stable_identifier": "test",
                "cache_filename": "weights.pt",
                "byte_size": 100,
                "sha256": "1" * 64,
            },
        },
        "input_contract": {"shape": [1, 224, 224]},
        "training_transform_contract": {"training": True, "image_size": 224},
        "evaluation_transform_contract": {"training": False, "image_size": 224},
        "training_policy": {
            "seed": seed,
            "permitted_partitions": ["train", "validation"],
            "optimizer": "AdamW",
        },
        "selection": {
            "selected_epoch": 2,
            "selected_stage": "fine_tune",
            "validation_average_precision": 0.5,
        },
        "thresholds": {"youden_j": 0.5, "target_sensitivity": 0.3},
        "threshold_contract": {
            "youden_j_policy_version": "test",
            "target_sensitivity_policy_version": "test",
            "sensitivity_target": 0.9,
            "positive_class": 1,
        },
        "metrics_policy": {
            "version": "test",
            "calibration_bins": 15,
            "threshold_policy_version": "test",
            "sensitivity_target": 0.9,
        },
        "source_authentication": {
            "policy_version": "test",
            "partitions": ["train", "validation"],
            "file_count": 100,
            "source_inventory_arrow_sha256": "2" * 64,
            "source_inventory_file_sha256": "3" * 64,
            "authenticated_rows_sha256": "4" * 64,
            "success": True,
        },
    }


def _create_member(
    tmp_path: Path,
    tracking_uri: str,
    seed: int,
    *,
    config_path: Path | None = None,
) -> tuple[str, str, Path, dict[str, object]]:
    source = config_path or _CONFIG_PATHS[seed]
    config = load_experiment_config(source)
    source_bytes = source.read_bytes()
    source_sha256 = config.source_sha256
    semantic_sha256 = image_semantic_config_sha256(config)
    package_id = f"model-package-seed-{seed}"
    common_tags = {
        "experiment_name": "image_densenet121",
        "dataset": "rsna",
        "dataset_bundle_id": _BUNDLE_ID,
        "split_assignment_id": _SPLIT_ASSIGNMENT_ID,
        "label_policy_version": _LABEL_POLICY_VERSION,
        "task": "pneumonia",
        "model": "image_densenet",
        "modality": "image",
        "seed": str(seed),
        "git_commit": _GIT_COMMIT,
        "git_dirty": "false",
        "dependency_lock_sha256": _LOCK_SHA256,
        "model_package_id": package_id,
    }
    configure_mlflow(experiment_name="seed-summary-test", tracking_uri=tracking_uri)
    with mlflow.start_run(
        tags={
            **common_tags,
            "run_kind": "training",
            "evaluation_scope": "validation",
            "run_complete": "true",
            "source_config_sha256": source_sha256,
            "semantic_config_sha256": semantic_sha256,
            "local_model_sha256": _CHECKPOINT_SHA256,
            "checkpoint_sha256": _CHECKPOINT_SHA256,
            "threshold_youden_j": "0.5",
            "threshold_target_sensitivity": "0.3",
        }
    ) as training_run:
        training_id = training_run.info.run_id
        package = tmp_path / "models" / "runs" / training_id
        package.mkdir(parents=True)
        model_path = package / "model.pt"
        model_path.write_bytes(b"checkpoint")
        (package / "resolved_config.yaml").write_bytes(source_bytes)
        mlflow.set_tag("local_model_path", model_path.as_posix())
        mlflow.log_param("bundle_manifest_sha256", _BUNDLE_MANIFEST_SHA256)
        mlflow.log_metrics(
            {
                "validation_youden_j_threshold": 0.5,
                "validation_target_sensitivity_threshold": 0.3,
            }
        )
    manifest = _manifest(
        training_run_id=training_id,
        package_id=package_id,
        seed=seed,
        source_config_sha256=source_sha256,
        semantic_config_sha256=semantic_sha256,
    )
    with mlflow.start_run(
        tags={
            **common_tags,
            "run_kind": "test_evaluation",
            "evaluation_scope": "test",
            "source_training_run_id": training_id,
            "run_complete": "true",
            "local_model_sha256": _CHECKPOINT_SHA256,
            "checkpoint_sha256": _CHECKPOINT_SHA256,
            "threshold_youden_j": "0.5",
            "threshold_target_sensitivity": "0.3",
        }
    ) as test_run:
        test_id = test_run.info.run_id
        mlflow.log_param("source_training_run_id", training_id)
        mlflow.log_param("bundle_manifest_sha256", _BUNDLE_MANIFEST_SHA256)
        mlflow.log_metrics(
            {
                f"test_{name}": value
                for name, value in _metrics({17: 0.1, 42: 0.2, 2026: 0.3}[seed]).items()
                if name != "model_size_mib"
            }
        )
        mlflow.log_metric("model_size_mib", 2.0)
    return test_id, training_id, package, manifest


@pytest.fixture
def family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Family:
    tracking_uri = _tracking_uri(tmp_path)
    tests: dict[int, str] = {}
    training: dict[int, str] = {}
    manifests: dict[Path, dict[str, object]] = {}
    for seed in (17, 42, 2026):
        test_id, training_id, package, manifest = _create_member(tmp_path, tracking_uri, seed)
        tests[seed] = test_id
        training[seed] = training_id
        manifests[package] = manifest
    monkeypatch.setattr(
        "radfusion.training.summarize_seeds.validate_neural_package_metadata",
        lambda package: manifests[Path(package)],
    )
    return _Family(tracking_uri, tests, training, manifests)


def test_three_explicit_compatible_members_publish_deterministic_summary(
    family: _Family, tmp_path: Path
) -> None:
    first = summarize_seed_runs(
        [family.test_ids[2026], family.test_ids[17], family.test_ids[42]],
        tracking_uri=family.tracking_uri,
        output_directory=tmp_path / "first",
    )
    second = summarize_seed_runs(
        [family.test_ids[42], family.test_ids[2026], family.test_ids[17]],
        tracking_uri=family.tracking_uri,
        output_directory=tmp_path / "second",
    )
    document = json.loads((first.report_directory / "summary.json").read_text(encoding="utf-8"))

    assert first.report_id == second.report_id
    assert first.test_run_ids == tuple(family.test_ids[seed] for seed in (17, 42, 2026))
    assert [member["seed"] for member in document["members"]] == [17, 42, 2026]
    assert document["aggregates"]["average_precision"]["mean"] == pytest.approx(0.2)
    assert document["aggregates"]["average_precision"][
        "sample_standard_deviation"
    ] == pytest.approx(0.1)
    compatibility_bytes = json.dumps(
        document["compatibility"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert document["compatibility_sha256"] == hashlib.sha256(compatibility_bytes).hexdigest()
    assert document["compatibility"]["dataset"] == "rsna"
    assert "runtime_provenance" not in document["compatibility"]
    assert "youden_j_threshold" not in document["aggregates"]
    assert set(path.name for path in first.report_directory.iterdir()) == {
        "summary.json",
        "metrics.csv",
        "summary.md",
    }
    for filename in ("summary.json", "metrics.csv", "summary.md"):
        assert (first.report_directory / filename).read_bytes() == (
            second.report_directory / filename
        ).read_bytes()


@pytest.mark.parametrize(
    "run_ids",
    [
        ("one", "two"),
        ("one", "two", "three", "four"),
        ("same", "same", "other"),
    ],
)
def test_membership_requires_exactly_three_distinct_ids(run_ids: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        summarize_seed_runs(run_ids, tracking_uri="sqlite:///unused.db")


@pytest.mark.parametrize("seed_value", ["", "42", "7"])
def test_summary_rejects_missing_duplicate_and_unexpected_seeds(
    seed_value: str, family: _Family, tmp_path: Path
) -> None:
    client = configure_mlflow(tracking_uri=family.tracking_uri)
    client.set_tag(family.test_ids[2026], "seed", seed_value)

    with pytest.raises(ValueError):
        summarize_seed_runs(
            list(family.test_ids.values()),
            tracking_uri=family.tracking_uri,
            output_directory=tmp_path / "reports",
        )


@pytest.mark.parametrize("failure", ["incomplete", "failed", "validation"])
def test_summary_rejects_noncompleted_or_non_test_members(
    failure: str, family: _Family, tmp_path: Path
) -> None:
    client = configure_mlflow(tracking_uri=family.tracking_uri)
    run_ids = list(family.test_ids.values())
    if failure == "incomplete":
        client.set_tag(family.test_ids[17], "run_complete", "false")
    elif failure == "failed":
        client.set_terminated(family.test_ids[17], status="FAILED")
    else:
        run_ids[0] = family.training_ids[17]

    with pytest.raises(ValueError):
        summarize_seed_runs(
            run_ids,
            tracking_uri=family.tracking_uri,
            output_directory=tmp_path / "reports",
        )


def test_summary_rejects_incomplete_training_parent(family: _Family, tmp_path: Path) -> None:
    client = configure_mlflow(tracking_uri=family.tracking_uri)
    client.set_tag(family.training_ids[17], "run_complete", "false")

    with pytest.raises(ValueError, match="not complete"):
        summarize_seed_runs(
            list(family.test_ids.values()),
            tracking_uri=family.tracking_uri,
            output_directory=tmp_path / "reports",
        )


@pytest.mark.parametrize(
    ("tag", "value"),
    [
        ("source_training_run_id", ""),
        ("model_package_id", "wrong-package"),
        ("model", "wrong-model"),
        ("modality", "metadata"),
        ("task", "wrong-task"),
        ("dataset_bundle_id", "wrong-bundle"),
        ("split_assignment_id", "wrong-split"),
    ],
)
def test_summary_rejects_malformed_test_parent_lineage(
    tag: str, value: str, family: _Family, tmp_path: Path
) -> None:
    client = configure_mlflow(tracking_uri=family.tracking_uri)
    client.set_tag(family.test_ids[17], tag, value)

    with pytest.raises((ValueError, MlflowException)):
        summarize_seed_runs(
            list(family.test_ids.values()),
            tracking_uri=family.tracking_uri,
            output_directory=tmp_path / "reports",
        )


@pytest.mark.parametrize(
    "field",
    [
        "bundle_manifest_sha256",
        "label_policy_version",
        "model_identity",
        "pretrained_weight",
        "training_transform_contract",
        "metrics_policy",
    ],
)
def test_summary_rejects_cross_seed_contract_drift(
    field: str, family: _Family, tmp_path: Path
) -> None:
    package = next(
        path
        for path, manifest in family.manifests.items()
        if manifest["training_policy"]["seed"] == 2026
    )
    manifest = family.manifests[package]
    if field == "bundle_manifest_sha256":
        replacement: object = "0" * 64
    elif field == "label_policy_version":
        replacement = "different-label-policy"
    elif field == "pretrained_weight":
        replacement = {
            **manifest["model_identity"],
            "pretrained_weight": {
                **manifest["model_identity"]["pretrained_weight"],
                "sha256": "0" * 64,
            },
        }
        field = "model_identity"
    else:
        replacement = {**manifest[field], "drift": True}
    manifest[field] = replacement
    client = configure_mlflow(tracking_uri=family.tracking_uri)
    if field == "label_policy_version":
        for run_id in (family.training_ids[2026], family.test_ids[2026]):
            client.set_tag(run_id, "label_policy_version", replacement)

    with pytest.raises(ValueError):
        summarize_seed_runs(
            list(family.test_ids.values()),
            tracking_uri=family.tracking_uri,
            output_directory=tmp_path / "reports",
        )


@pytest.mark.parametrize("field", ["git_commit", "dependency_lock_sha256"])
def test_summary_rejects_cross_seed_source_provenance_drift(
    field: str, family: _Family, tmp_path: Path
) -> None:
    replacement = "0" * (40 if field == "git_commit" else 64)
    client = configure_mlflow(tracking_uri=family.tracking_uri)
    for run_id in (family.training_ids[2026], family.test_ids[2026]):
        client.set_tag(run_id, field, replacement)
    package = next(
        path
        for path, manifest in family.manifests.items()
        if manifest["training_policy"]["seed"] == 2026
    )
    family.manifests[package]["source_provenance"][field] = replacement

    with pytest.raises(ValueError):
        summarize_seed_runs(
            list(family.test_ids.values()),
            tracking_uri=family.tracking_uri,
            output_directory=tmp_path / "reports",
        )


def test_summary_rejects_config_drift_beyond_seed(family: _Family, tmp_path: Path) -> None:
    package = next(
        path
        for path, manifest in family.manifests.items()
        if manifest["training_policy"]["seed"] == 2026
    )
    document = yaml.safe_load(_CONFIG_PATHS[2026].read_text(encoding="utf-8"))
    document["image"]["brightness_jitter"] = 0.04
    config_path = package / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(config_path)
    manifest = family.manifests[package]
    manifest["source_config_sha256"] = config.source_sha256
    manifest["semantic_config_sha256"] = image_semantic_config_sha256(config)
    client = configure_mlflow(tracking_uri=family.tracking_uri)
    client.set_tag(family.training_ids[2026], "source_config_sha256", config.source_sha256)
    client.set_tag(
        family.training_ids[2026],
        "semantic_config_sha256",
        image_semantic_config_sha256(config),
    )

    with pytest.raises(ValueError, match="scientifically compatible"):
        summarize_seed_runs(
            list(family.test_ids.values()),
            tracking_uri=family.tracking_uri,
            output_directory=tmp_path / "reports",
        )


def test_summary_rejects_invalid_local_package(
    family: _Family, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_package(path: Path) -> dict[str, object]:
        raise ValueError(f"invalid package: {path}")

    monkeypatch.setattr(
        "radfusion.training.summarize_seeds.validate_neural_package_metadata",
        reject_package,
    )

    with pytest.raises(ValueError, match="invalid package"):
        summarize_seed_runs(
            list(family.test_ids.values()),
            tracking_uri=family.tracking_uri,
            output_directory=tmp_path / "reports",
        )


@pytest.mark.parametrize("invalid", [None, True, math.nan, math.inf, -math.inf])
def test_image_test_metric_contract_rejects_missing_boolean_and_nonfinite_values(
    invalid: object, family: _Family
) -> None:
    client = configure_mlflow(tracking_uri=family.tracking_uri)
    record = require_completed_run(client.get_run(family.test_ids[17]))
    metrics = dict(record.metrics)
    metrics["average_precision"] = invalid

    with pytest.raises(ValueError, match="missing or non-finite"):
        validated_image_test_metrics(replace(record, metrics=metrics))


def test_seed_specific_metric_contract_is_complete() -> None:
    assert set(_metrics(0.5)) == set(SEED_SPECIFIC_METRIC_NAMES)
