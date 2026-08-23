from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import pandas as pd
import pytest
import torch
import yaml
from neural_test_support import TensorDataset as _TensorDataset
from neural_test_support import TinyImageModel as _TinyImageModel
from neural_test_support import build_synchronous_image_loaders as build_image_loaders
from neural_test_support import cpu_runtime as _runtime

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.hashing import sha256_file
from beyondcxr.data.rsna_cxr_cache import (
    SOURCE_AUTHENTICATION_POLICY_VERSION,
    CxrCacheIdentity,
    CxrCacheSourceAuthentication,
    preprocessing_identity,
)
from beyondcxr.models.cxr_baseline import PretrainedWeightIdentity
from beyondcxr.training.config import (
    ExperimentConfig,
    load_experiment_config,
    with_runtime,
)
from beyondcxr.training.neural import (
    CLASS_WEIGHT_POLICY_VERSION,
)
from beyondcxr.training.rsna_compare import regenerate_comparison
from beyondcxr.training.rsna_datasets import (
    CxrRunData,
    CxrTestData,
    SourceInventoryIdentity,
)
from beyondcxr.training.rsna_evaluate import evaluate_model_package
from beyondcxr.training.rsna_interfaces import DatasetLineage
from beyondcxr.training.rsna_train_cxr import train_cxr_experiment
from beyondcxr.utils.mlflow_utils import configure_mlflow
from beyondcxr.utils.operational_logging import configure_logging
from beyondcxr.utils.package_identity import package_scientific_config_payload
from beyondcxr.utils.private_predictions import validate_prediction_evidence
from beyondcxr.utils.rsna_model_publication import threshold_contract
from beyondcxr.utils.rsna_neural_publication import (
    CHECKPOINT_FIELDS,
    NEURAL_MODEL_FILENAME,
    checkpoint_document,
    load_neural_checkpoint,
    load_validated_neural_checkpoint,
    neural_model_package_id,
    publish_neural_model_package,
    save_neural_checkpoint,
    strict_load_checkpoint,
    validate_neural_package_metadata,
    validate_published_neural_model,
)

_SYNTHETIC_BUNDLE_ID = "bundle-" + "a" * 64


def _manifest(config_bytes: bytes, checkpoint: dict[str, object]) -> dict[str, object]:
    config_path = Path("configs/rsna_cxr_densenet.yaml")
    config = with_runtime(load_experiment_config(config_path), seed=42)
    neural = config.neural
    assert neural is not None
    digest = hashlib.sha256(config_bytes).hexdigest()
    transform_kwargs = {
        "image_size": 224,
        "rotation_degrees": neural.rotation_degrees,
        "translation_fraction": neural.translation_fraction,
        "brightness_jitter": neural.brightness_jitter,
        "contrast_jitter": neural.contrast_jitter,
    }
    training_transform = StandardCxrTransform(training=True, **transform_kwargs).contract()
    evaluation_transform = StandardCxrTransform(training=False, **transform_kwargs).contract()
    return {
        "family_id": "cxr_densenet",
        "modalities": ["cxr"],
        "task_id": "pneumonia",
        "positive_class": 1,
        "bundle_id": config.dataset.bundle_id,
        "bundle_manifest_sha256": "e" * 64,
        "split_assignment_id": config.dataset.split_assignment_id,
        "label_policy_version": config.task.label_policy_version,
        "config_source_sha256": digest,
        "config_semantic_sha256": config.config_semantic_sha256,
        "source_provenance": {
            "git_commit": "commit",
            "git_dirty": False,
            "dependency_lock_sha256": "b" * 64,
            "python_version": "3.13",
            "torch_version": "test",
            "torchvision_version": "test",
            "torchxrayvision_version": "test",
        },
        "model_identity": {
            "family_id": "cxr_densenet",
            "modalities": ["cxr"],
            "encoder_architecture": "densenet121",
            "image_size": 224,
            "embedding_dimension": 1024,
            "classifier_output_dimension": 1,
            "pretrained_weight": {
                "declared_name": "densenet121-res224-chex",
                "stable_identifier": "https://example.invalid/weights.pt",
                "cache_filename": "weights.pt",
                "byte_size": 100,
                "sha256": "c" * 64,
            },
        },
        "input_contract": evaluation_transform["input"],
        "training_transform_contract": training_transform,
        "evaluation_transform_contract": evaluation_transform,
        "training_policy": {
            "seed": 42,
            "permitted_partitions": ["train", "validation"],
            "class_weight": {
                "policy_version": CLASS_WEIGHT_POLICY_VERSION,
                "labels_used": "train",
                "positive_count": 1,
                "negative_count": 1,
                "pos_weight": 1.0,
            },
            "optimizer": "AdamW",
            "warmup": {"epochs": 2, "head_learning_rate": 0.001, "encoder_frozen": True},
            "fine_tuning": {
                "maximum_epochs": 28,
                "encoder_learning_rate": 0.00001,
                "head_learning_rate": 0.0001,
            },
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "mode": "max",
                "factor": 0.5,
                "patience": 2,
                "min_lr": 0.0000001,
            },
            "early_stopping": {
                "metric": "validation_average_precision",
                "patience": 5,
                "minimum_delta": 0.0001,
            },
        },
        "selection": {
            "selected_epoch": checkpoint["selected_epoch"],
            "selected_stage": checkpoint["selected_stage"],
            "validation_average_precision": checkpoint["validation_average_precision"],
        },
        "thresholds": {"youden_j": 0.5, "target_sensitivity": 0.3},
        "threshold_contract": threshold_contract(sensitivity_target=0.9),
        "source_authentication": {
            "policy_version": SOURCE_AUTHENTICATION_POLICY_VERSION,
            "partitions": ["train", "validation", "test"],
            "file_count": 3,
            "source_inventory_arrow_sha256": "4" * 64,
            "source_inventory_file_sha256": "5" * 64,
        },
        "runtime_provenance": {
            **_runtime().provenance(),
            "cxr_cache_id": "cache-" + "7" * 64,
            "loader_execution": {
                "lifecycle": "reused",
                "num_workers": 0,
                "pin_memory": False,
            },
        },
    }


def test_safe_neural_checkpoint_and_immutable_three_file_package(tmp_path: Path) -> None:
    model = _TinyImageModel()
    checkpoint = checkpoint_document(
        model.state_dict(),
        selected_epoch=3,
        selected_stage="fine_tune",
        validation_average_precision=0.75,
    )
    checkpoint_path = save_neural_checkpoint(checkpoint, tmp_path / "checkpoint.pt")
    restored = load_neural_checkpoint(checkpoint_path)
    strict_load_checkpoint(_TinyImageModel(), restored)
    config_bytes = Path("configs/rsna_cxr_densenet.yaml").read_bytes()
    published = publish_neural_model_package(
        model_root=tmp_path / "models" / "rsna",
        checkpoint_path=checkpoint_path,
        source_config_bytes=config_bytes,
        manifest=_manifest(config_bytes, checkpoint),
    )

    metadata = validate_neural_package_metadata(published.package_directory)
    loaded = load_validated_neural_checkpoint(published.package_directory, metadata)
    document = validate_published_neural_model(published.package_directory)
    assert set(path.name for path in published.package_directory.iterdir()) == {
        NEURAL_MODEL_FILENAME,
        "resolved_config.yaml",
        "manifest.json",
    }
    assert set(restored) == CHECKPOINT_FIELDS
    assert loaded["selected_epoch"] == checkpoint["selected_epoch"]
    assert document["model_package_id"] == neural_model_package_id(document)
    assert json.loads(published.manifest_path.read_text())["checkpoint_sha256"] == sha256_file(
        published.model_path
    )
    alternate_checkpoint = save_neural_checkpoint(checkpoint, tmp_path / "alternate.pt")
    assert alternate_checkpoint.read_bytes() != published.model_path.read_bytes()
    repeated = publish_neural_model_package(
        model_root=tmp_path / "models" / "rsna",
        checkpoint_path=alternate_checkpoint,
        source_config_bytes=config_bytes,
        manifest=_manifest(config_bytes, checkpoint),
    )
    assert repeated.model_package_id == published.model_package_id
    assert published.created is True
    assert repeated.created is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document["model_identity"].update({"unknown": True}),
        lambda document: document["training_policy"]["class_weight"].update({"pos_weight": 2.0}),
        lambda document: document["training_transform_contract"]["training_augmentation"].update(
            {"enabled": False}
        ),
        lambda document: document["runtime_provenance"].update({"hostname": "private"}),
        lambda document: document["runtime_provenance"]["loader_execution"].update(
            {"prefetch_factor": 2}
        ),
    ],
)
def test_neural_manifest_rejects_nested_contract_tampering(tmp_path: Path, mutation) -> None:
    checkpoint = checkpoint_document(
        _TinyImageModel().state_dict(),
        selected_epoch=1,
        selected_stage="warmup",
        validation_average_precision=0.5,
    )
    checkpoint_path = save_neural_checkpoint(checkpoint, tmp_path / "checkpoint.pt")
    config_bytes = Path("configs/rsna_cxr_densenet.yaml").read_bytes()
    published = publish_neural_model_package(
        model_root=tmp_path / "models" / "rsna",
        checkpoint_path=checkpoint_path,
        source_config_bytes=config_bytes,
        manifest=_manifest(config_bytes, checkpoint),
    )
    document = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    mutation(document)
    document["model_package_id"] = neural_model_package_id(document)
    published.manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_neural_package_metadata(published.package_directory)


_MISSING_SCHEMA_VERSION = object()


@pytest.mark.parametrize("value", [True, 1.0, "1", None, 0, _MISSING_SCHEMA_VERSION])
def test_neural_manifest_requires_integer_schema_version_one(tmp_path: Path, value: object) -> None:
    checkpoint = checkpoint_document(
        _TinyImageModel().state_dict(),
        selected_epoch=1,
        selected_stage="warmup",
        validation_average_precision=0.5,
    )
    checkpoint_path = save_neural_checkpoint(checkpoint, tmp_path / "checkpoint.pt")
    config_bytes = Path("configs/rsna_cxr_densenet.yaml").read_bytes()
    published = publish_neural_model_package(
        model_root=tmp_path / "models" / "rsna",
        checkpoint_path=checkpoint_path,
        source_config_bytes=config_bytes,
        manifest=_manifest(config_bytes, checkpoint),
    )
    document = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    if value is _MISSING_SCHEMA_VERSION:
        document.pop("model_package_schema_version")
    else:
        document["model_package_schema_version"] = value
        document["model_package_id"] = neural_model_package_id(document)
    published.manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_neural_package_metadata(published.package_directory)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document["model_state_dict"].update({1: torch.ones(1)}),
        lambda document: document["model_state_dict"].update({"bad": "not-a-tensor"}),
        lambda document: document["model_state_dict"].update({"bad": torch.tensor(float("nan"))}),
        lambda document: document.update({"optimizer_state": {}}),
        lambda document: document.update({"selected_epoch": True}),
        lambda document: document.update({"selected_stage": "latest"}),
    ],
)
def test_checkpoint_schema_rejects_unsafe_or_nonsemantic_state(tmp_path: Path, mutation) -> None:
    document = checkpoint_document(
        _TinyImageModel().state_dict(),
        selected_epoch=1,
        selected_stage="warmup",
        validation_average_precision=0.5,
    )
    mutation(document)

    with pytest.raises(ValueError):
        save_neural_checkpoint(document, tmp_path / "invalid.pt")


@pytest.mark.parametrize("value", [True, 1.0, "1", None, 0, _MISSING_SCHEMA_VERSION])
def test_neural_checkpoint_requires_integer_schema_version_one(
    tmp_path: Path, value: object
) -> None:
    document = checkpoint_document(
        _TinyImageModel().state_dict(),
        selected_epoch=1,
        selected_stage="warmup",
        validation_average_precision=0.5,
    )
    if value is _MISSING_SCHEMA_VERSION:
        document.pop("checkpoint_schema_version")
    else:
        document["checkpoint_schema_version"] = value

    with pytest.raises(ValueError):
        save_neural_checkpoint(document, tmp_path / "invalid.pt")


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_strict_checkpoint_loading_rejects_parameter_mismatch(mutation: str) -> None:
    document = checkpoint_document(
        _TinyImageModel().state_dict(),
        selected_epoch=1,
        selected_stage="warmup",
        validation_average_precision=0.5,
    )
    if mutation == "missing":
        document["model_state_dict"].pop(next(iter(document["model_state_dict"])))
    else:
        document["model_state_dict"]["unexpected"] = torch.ones(1)

    with pytest.raises(ValueError):
        strict_load_checkpoint(_TinyImageModel(), document)


def test_safe_loader_rejects_whole_module_and_package_identity_binds_provenance(
    tmp_path: Path,
) -> None:
    unsafe = tmp_path / "unsafe.pt"
    torch.save(_TinyImageModel(), unsafe)
    with pytest.raises(ValueError):
        load_neural_checkpoint(unsafe)

    checkpoint = checkpoint_document(
        _TinyImageModel().state_dict(),
        selected_epoch=1,
        selected_stage="warmup",
        validation_average_precision=0.6,
    )
    config_bytes = Path("configs/rsna_cxr_densenet.yaml").read_bytes()
    checkpoint_path = save_neural_checkpoint(checkpoint, tmp_path / "checkpoint.pt")
    published = publish_neural_model_package(
        model_root=tmp_path / "models" / "rsna",
        checkpoint_path=checkpoint_path,
        source_config_bytes=config_bytes,
        manifest=_manifest(config_bytes, checkpoint),
    )
    identity = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    baseline = neural_model_package_id(identity)
    config_document = yaml.safe_load(config_bytes)
    config_document["evaluation"]["calibration_bins"] = 20
    changed_config_path = tmp_path / "changed-evaluation.yaml"
    changed_config_path.write_text(
        yaml.safe_dump(config_document, sort_keys=False), encoding="utf-8"
    )
    original_config = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    changed_config = load_experiment_config(changed_config_path)
    assert changed_config.config_semantic_sha256 != original_config.config_semantic_sha256
    assert package_scientific_config_payload(changed_config) == package_scientific_config_payload(
        original_config
    )
    evaluation_changed = json.loads(json.dumps(identity))
    evaluation_changed["config_semantic_sha256"] = changed_config.config_semantic_sha256
    assert neural_model_package_id(evaluation_changed) == baseline
    mutations = (
        lambda value: value.update({"bundle_id": "bundle-changed"}),
        lambda value: value.update({"split_assignment_id": "split-changed"}),
        lambda value: value.update({"task_id": "changed-task"}),
        lambda value: value.update({"label_policy_version": "changed-label-policy"}),
        lambda value: value["training_policy"].update({"seed": 17}),
        lambda value: value["fit_config"]["training"]["parameters"].update(
            {"head_learning_rate": 0.5}
        ),
        lambda value: value["fit_config"]["preprocessing"].update(
            {"cxr_transform_policy": "changed-policy"}
        ),
        lambda value: value["fit_config"]["family"]["parameters"].update(
            {"embedding_dimension": 2048}
        ),
        lambda value: value.update({"model_state_sha256": "8" * 64}),
        lambda value: value["thresholds"].update({"youden_j": 0.51}),
        lambda value: value["model_identity"]["pretrained_weight"].update(
            {"declared_name": "changed-weight"}
        ),
        lambda value: value["model_identity"]["pretrained_weight"].update(
            {"stable_identifier": "https://example.invalid/changed.pt"}
        ),
        lambda value: value["model_identity"]["pretrained_weight"].update({"sha256": "d" * 64}),
    )
    for mutation in mutations:
        changed = json.loads(json.dumps(identity))
        mutation(changed)
        assert neural_model_package_id(changed) != baseline
    for mutation in (
        lambda value: value.update({"checkpoint_sha256": "e" * 64}),
        lambda value: value.update({"config_source_sha256": "e" * 64}),
        lambda value: value.update({"config_semantic_sha256": "e" * 64}),
        lambda value: value.update({"bundle_manifest_sha256": "8" * 64}),
        lambda value: value["model_identity"]["pretrained_weight"].update(
            {"cache_filename": "alternate-cache-name.pt"}
        ),
        lambda value: value["model_identity"]["pretrained_weight"].update({"byte_size": 200}),
        lambda value: value["source_provenance"].update({"git_commit": "different"}),
        lambda value: value["runtime_provenance"].update({"requested_device": "auto"}),
    ):
        changed = json.loads(json.dumps(identity))
        mutation(changed)
        assert neural_model_package_id(changed) == baseline


def test_dataset_loading_failure_precedes_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = with_runtime(
        load_experiment_config("configs/rsna_cxr_densenet.yaml"),
        seed=42,
        source_root=tmp_path / "raw",
        model_directory=tmp_path / "models" / "rsna",
        report_directory=tmp_path / "reports",
    )
    model_requested = []

    class FailingAdapter:
        def load_cxr_train_validation(self, dataset_config):
            del dataset_config
            raise ManifestBuildError("dataset loading failed")

    monkeypatch.setattr(
        "beyondcxr.training.rsna_train_cxr.get_dataset", lambda key: FailingAdapter()
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_train_cxr.get_model", lambda key: model_requested.append(key)
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_train_cxr.git_revision", lambda: ("commit-test", False)
    )
    monkeypatch.setattr("beyondcxr.training.rsna_train_cxr.uv_lock_sha256", lambda: "9" * 64)
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    with pytest.raises(ManifestBuildError):
        train_cxr_experiment(config, tracking_uri=tracking_uri)

    assert model_requested == []
    client = configure_mlflow(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(config.runtime.experiment_name)
    assert experiment is not None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
    assert runs[0].data.tags["run_complete"] == "false"


@dataclass
class _SyntheticCxrLifecycle:
    config: ExperimentConfig
    adapter: Any
    weight: PretrainedWeightIdentity
    tracking_uri: str
    build_calls: list[str]
    construction_events: list[str]
    seed_calls: list[int]


def _synthetic_cxr_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _SyntheticCxrLifecycle:
    document = yaml.safe_load(Path("configs/rsna_cxr_densenet.yaml").read_text(encoding="utf-8"))
    document["dataset"]["bundle_id"] = _SYNTHETIC_BUNDLE_ID
    document["dataset"]["bundle_manifest_sha256"] = "e" * 64
    document["dataset"]["split_assignment_id"] = "split-assignment-" + "6" * 64
    document["training"]["loader"].update({"batch_size": 2})
    document["training"]["parameters"].update(
        {
            "mixed_precision": False,
            "warmup_epochs": 1,
            "fine_tune_epochs": 1,
            "early_stopping_patience": 1,
        }
    )
    config_path = tmp_path / "cxr.yaml"
    config_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    config = with_runtime(
        load_experiment_config(config_path),
        seed=42,
        source_root=tmp_path / "raw",
        manifest_directory=tmp_path / "manifests",
        model_directory=tmp_path / "models" / "rsna",
        report_directory=tmp_path / "reports",
        private_output_directory=tmp_path / "explicit-private-root",
        device="cpu",
        num_workers=0,
        pin_memory_policy="disabled",
    )
    lineage = DatasetLineage(
        bundle_id=_SYNTHETIC_BUNDLE_ID,
        split_assignment_id=config.dataset.split_assignment_id,
        label_policy_version=config.task.label_policy_version,
        task_id=config.task.task_id,
    )

    def frame(partition: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (f"rsna:{partition}-0", f"patient-{partition}-0", "images/0.dcm", partition, 0),
                (f"rsna:{partition}-1", f"patient-{partition}-1", "images/1.dcm", partition, 1),
                (f"rsna:{partition}-2", f"patient-{partition}-2", "images/2.dcm", partition, 0),
                (f"rsna:{partition}-3", f"patient-{partition}-3", "images/3.dcm", partition, 1),
            ],
            columns=("sample_id", "patient_id", "image_path", "split_name", "target"),
        )

    source_inventory = SourceInventoryIdentity(
        source_inventory_arrow_sha256="a" * 64,
        source_inventory_file_sha256="b" * 64,
    )

    class Adapter:
        test_calls = 0

        def load_cxr_train_validation(self, dataset_config):
            assert dataset_config.dataset.bundle_id == _SYNTHETIC_BUNDLE_ID
            return CxrRunData(
                train=frame("train"),
                validation=frame("validation"),
                lineage=lineage,
                bundle_manifest_sha256=config.dataset.bundle_manifest_sha256,
                source_inventory=source_inventory,
            )

        def load_cxr_test(self, dataset_config, *, expected_manifest_sha256):
            assert expected_manifest_sha256 == config.dataset.bundle_manifest_sha256
            self.test_calls += 1
            return CxrTestData(
                test=frame("test"),
                lineage=lineage,
                bundle_manifest_sha256=config.dataset.bundle_manifest_sha256,
                source_inventory=source_inventory,
            )

    build_calls = []
    construction_events = []

    class Builder:
        def build(self, model_config):
            assert model_config.modalities == ("cxr",)
            build_calls.append(model_config.family_id)
            construction_events.append("build")
            return _TinyImageModel()

        def build_architecture(self, model_config):
            return self.build(model_config)

    adapter = Adapter()
    weight = PretrainedWeightIdentity(
        declared_name="densenet121-res224-chex",
        stable_identifier="https://example.invalid/weights.pt",
        cache_filename="weights.pt",
        byte_size=100,
        sha256="f" * 64,
    )

    def synthetic_dataset(frame_value, **kwargs):
        del kwargs
        return _TensorDataset(frame_value["target"].astype(int).tolist(), sample_prefix="rsna:")

    for module in ("beyondcxr.training.rsna_train_cxr", "beyondcxr.training.rsna_evaluate_cxr"):
        monkeypatch.setattr(f"{module}.get_dataset", lambda key: adapter)
        monkeypatch.setattr(f"{module}.get_model", lambda key: Builder())
        monkeypatch.setattr(f"{module}.RsnaCachedImageDataset", synthetic_dataset)

        def prepared_cache(*args, **kwargs):
            del kwargs
            transform = args[2]
            identity = CxrCacheIdentity(
                bundle_id=_SYNTHETIC_BUNDLE_ID,
                bundle_manifest_sha256="e" * 64,
                source_inventory_file_sha256="b" * 64,
                source_inventory_arrow_sha256="a" * 64,
                preprocessing_sha256=preprocessing_identity(transform),
            )
            return SimpleNamespace(
                identity=identity,
                source_authentication=CxrCacheSourceAuthentication(
                    policy_version=SOURCE_AUTHENTICATION_POLICY_VERSION,
                    partitions=("train", "validation", "test"),
                    file_count=12,
                    source_inventory_arrow_sha256="a" * 64,
                    source_inventory_file_sha256="b" * 64,
                ),
            )

        monkeypatch.setattr(f"{module}.prepare_rsna_cxr_cache", prepared_cache)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_train_cxr.git_revision", lambda: ("commit-test", False)
    )
    monkeypatch.setattr("beyondcxr.training.rsna_train_cxr.uv_lock_sha256", lambda: "9" * 64)

    def fingerprint(weights):
        assert weights == "densenet121-res224-chex"
        construction_events.append("fingerprint")
        return weight

    monkeypatch.setattr(
        "beyondcxr.training.rsna_train_cxr.fingerprint_pretrained_weights",
        fingerprint,
    )

    seed_calls = []
    monkeypatch.setattr("beyondcxr.training.rsna_train_cxr.seed_neural_runtime", seed_calls.append)

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    return _SyntheticCxrLifecycle(
        config=config,
        adapter=adapter,
        weight=weight,
        tracking_uri=tracking_uri,
        build_calls=build_calls,
        construction_events=construction_events,
        seed_calls=seed_calls,
    )


def _assert_single_failed_training_without_outputs(setup: _SyntheticCxrLifecycle) -> None:
    client = configure_mlflow(tracking_uri=setup.tracking_uri)
    experiment = client.get_experiment_by_name(setup.config.runtime.experiment_name)
    assert experiment is not None
    failed_runs = [
        run
        for run in client.search_runs(experiment_ids=[experiment.experiment_id])
        if run.info.status == "FAILED"
        and run.data.tags.get("run_kind") == "training"
        and run.data.tags.get("run_complete") == "false"
    ]
    assert len(failed_runs) == 1
    run_id = failed_runs[0].info.run_id
    packages = setup.config.runtime.model_directory / "packages"
    assert not packages.exists() or not any(packages.iterdir())
    assert not (
        setup.config.runtime.report_directory / setup.config.dataset.dataset_id / "runs" / run_id
    ).exists()


def test_cache_failure_precedes_neural_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_cxr_lifecycle(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_train_cxr.prepare_rsna_cxr_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(ManifestBuildError("cache invalid")),
    )

    with pytest.raises(ManifestBuildError):
        train_cxr_experiment(setup.config, tracking_uri=setup.tracking_uri)

    assert setup.build_calls == []


def test_cxr_training_progress_accepts_unsized_validation_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_cxr_lifecycle(tmp_path, monkeypatch)

    class UnsizedLoader:
        def __init__(self, loader) -> None:
            self.loader = loader

        def __iter__(self):
            return iter(self.loader)

    def unsized_validation_loader(*args, **kwargs):
        loaders = build_image_loaders(*args, **kwargs)
        return type(loaders)(loaders.train, UnsizedLoader(loaders.validation))

    monkeypatch.setattr(
        "beyondcxr.training.rsna_train_cxr.build_image_loaders", unsized_validation_loader
    )

    result = train_cxr_experiment(setup.config, tracking_uri=setup.tracking_uri)

    assert result.model_path.is_file()


def test_synthetic_cxr_training_package_and_separate_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_cxr_lifecycle(tmp_path, monkeypatch)
    training = train_cxr_experiment(setup.config, tracking_uri=setup.tracking_uri)
    assert setup.adapter.test_calls == 0
    assert setup.seed_calls == [setup.config.runtime.seed]
    assert setup.build_calls == ["cxr_densenet"]
    assert setup.construction_events == ["fingerprint", "build", "fingerprint"]
    report_directory = setup.config.runtime.report_directory / "rsna" / "runs" / training.run_id
    report = json.loads((report_directory / "metrics.json").read_bytes())
    assert report["cxr_training"]["source_authentication"]["file_count"] == 12
    assert report["cxr_training"]["limitations"][-1] == (
        "The shared CXR cache authenticates and decodes all partitions; "
        "test samples are not used for fitting, selection, or threshold derivation."
    )
    assert training.model_path.name == "model.pt"
    package_manifest = validate_published_neural_model(training.model_path.parent)
    assert package_manifest["model_package_schema_version"] == 1
    assert package_manifest["family_id"] == "cxr_densenet"
    assert package_manifest["modalities"] == ["cxr"]
    assert package_manifest["bundle_manifest_sha256"] == "e" * 64
    assert package_manifest["runtime_provenance"]["loader_execution"] == {
        "lifecycle": "reused",
        "num_workers": 0,
        "pin_memory": False,
    }
    recorded_training = configure_mlflow(tracking_uri=setup.tracking_uri).get_run(training.run_id)
    assert recorded_training.data.tags["run_complete"] == "true"
    assert recorded_training.data.tags["bundle_manifest_sha256"] == "e" * 64

    evaluation = evaluate_model_package(
        training.model_package_id,
        evaluation_config=setup.config,
        tracking_uri=setup.tracking_uri,
        model_directory=setup.config.runtime.model_directory,
        private_output_directory=setup.config.runtime.private_output_directory,
        report_directory=setup.config.runtime.report_directory,
    )
    assert setup.adapter.test_calls == 1
    assert setup.build_calls == ["cxr_densenet", "cxr_densenet"]
    assert evaluation.mlflow_run_id != training.run_id
    assert evaluation.artifact_directory.is_dir()
    private_manifest = validate_prediction_evidence(
        evaluation.private_prediction_directory
    ).manifest
    assert private_manifest["prediction_id"] == evaluation.prediction_id
    assert private_manifest["model_package_id"] == training.model_package_id
    assert evaluation.private_prediction_directory.parent.parent.parent == (
        setup.config.runtime.private_output_directory
    )
    assert not any(path.suffix == ".parquet" for path in evaluation.artifact_directory.rglob("*"))
    client = configure_mlflow(tracking_uri=setup.tracking_uri)
    evaluation_run = client.get_run(evaluation.mlflow_run_id)
    assert client.list_artifacts(evaluation.mlflow_run_id) == []
    assert evaluation_run.data.tags["run_complete"] == "true"
    assert evaluation_run.data.tags["package_id"] == training.model_package_id
    assert evaluation_run.data.tags["bundle_manifest_sha256"] == "e" * 64
    assert evaluation_run.data.params["bundle_manifest_sha256"] == "e" * 64
    assert evaluation_run.data.params["evaluation_runtime_resolved_device"] == "cpu"
    assert evaluation_run.data.params["evaluation_loader_num_workers"] == "0"
    assert evaluation_run.data.params["evaluation_cxr_cache_id"].startswith("cache-")

    csv_path, _, rows = regenerate_comparison(
        [evaluation.evaluation_id],
        output_directory=setup.config.runtime.report_directory,
        private_directory=setup.config.runtime.private_output_directory,
        model_directory=setup.config.runtime.model_directory,
    )
    comparison = pd.read_csv(csv_path)
    assert rows == 1
    assert comparison["evaluation_id"].tolist() == [evaluation.evaluation_id]
    assert comparison["model_package_id"].tolist() == [training.model_package_id]


@pytest.mark.parametrize("changed_field", ["byte_size", "sha256"])
def test_pretrained_weight_mutation_aborts_before_fitting(
    changed_field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_cxr_lifecycle(tmp_path, monkeypatch)
    changed = (
        replace(setup.weight, byte_size=setup.weight.byte_size + 1)
        if changed_field == "byte_size"
        else replace(setup.weight, sha256="0" * 64)
    )
    observed = iter((setup.weight, changed))
    monkeypatch.setattr(
        "beyondcxr.training.rsna_train_cxr.fingerprint_pretrained_weights",
        lambda weights: next(observed),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_train_cxr.fit_rsna_cxr_model",
        lambda *args, **kwargs: pytest.fail("fitting must not begin"),
    )

    with pytest.raises(RuntimeError):
        train_cxr_experiment(setup.config, tracking_uri=setup.tracking_uri)


def test_missing_pretrained_weight_prevents_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_cxr_lifecycle(tmp_path, monkeypatch)

    def missing_weight(weights):
        del weights
        raise FileNotFoundError("must be materialized before formal training")

    monkeypatch.setattr(
        "beyondcxr.training.rsna_train_cxr.fingerprint_pretrained_weights",
        missing_weight,
    )
    with pytest.raises(FileNotFoundError):
        train_cxr_experiment(setup.config, tracking_uri=setup.tracking_uri)
    assert setup.build_calls == []
    _assert_single_failed_training_without_outputs(setup)


def test_pretrained_weight_mutation_cleans_outputs_and_leaves_run_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_cxr_lifecycle(tmp_path, monkeypatch)
    observed = iter((setup.weight, replace(setup.weight, sha256="0" * 64)))
    monkeypatch.setattr(
        "beyondcxr.training.rsna_train_cxr.fingerprint_pretrained_weights",
        lambda weights: next(observed),
    )

    with pytest.raises(RuntimeError):
        train_cxr_experiment(setup.config, tracking_uri=setup.tracking_uri)
    _assert_single_failed_training_without_outputs(setup)


def test_cxr_evaluation_rejects_package_cache_identity_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_cxr_lifecycle(tmp_path, monkeypatch)
    training = train_cxr_experiment(setup.config, tracking_uri=setup.tracking_uri)
    manifest_path = training.model_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["runtime_provenance"]["cxr_cache_id"] = "cache-" + "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "beyondcxr.training.rsna_evaluate_cxr.deterministic_inference",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError((args, kwargs))),
    )

    with pytest.raises(ValueError):
        evaluate_model_package(
            training.model_package_id,
            evaluation_config=setup.config,
            tracking_uri=setup.tracking_uri,
            model_directory=setup.config.runtime.model_directory,
        )


def test_cxr_evaluation_rejects_package_and_source_lineage_before_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_cxr_lifecycle(tmp_path, monkeypatch)
    training = train_cxr_experiment(setup.config, tracking_uri=setup.tracking_uri)
    manifest_path = training.model_path.parent / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    tampered_manifest = json.loads(original_manifest)
    tampered_manifest["model_package_id"] = "model-package-" + "0" * 64
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_model_package(
            training.model_package_id,
            evaluation_config=setup.config,
            tracking_uri=setup.tracking_uri,
            model_directory=setup.config.runtime.model_directory,
        )
    assert setup.adapter.test_calls == 0
    manifest_path.write_bytes(original_manifest)

    config_archive = training.model_path.parent / "resolved_config.yaml"
    original_config = config_archive.read_bytes()
    config_archive.write_bytes(original_config + b"\n# tampered\n")
    with pytest.raises(ValueError):
        evaluate_model_package(
            training.model_package_id,
            evaluation_config=setup.config,
            tracking_uri=setup.tracking_uri,
            model_directory=setup.config.runtime.model_directory,
        )
    assert setup.adapter.test_calls == 0
    config_archive.write_bytes(original_config)

    original_checkpoint = training.model_path.read_bytes()
    training.model_path.write_bytes(original_checkpoint + b"tampered")
    with pytest.raises(ValueError):
        evaluate_model_package(
            training.model_package_id,
            evaluation_config=setup.config,
            tracking_uri=setup.tracking_uri,
            model_directory=setup.config.runtime.model_directory,
        )
    assert setup.adapter.test_calls == 0
    training.model_path.write_bytes(original_checkpoint)

    client = configure_mlflow(tracking_uri=setup.tracking_uri)
    source = client.get_run(training.run_id)
    original_ap = source.data.metrics["validation_average_precision"]
    client.log_metric(training.run_id, "validation_average_precision", original_ap + 0.01)
    original_size = source.data.metrics["model_size_mib"]
    client.log_metric(training.run_id, "model_size_mib", original_size + 0.01)
    evaluation = evaluate_model_package(
        training.model_package_id,
        evaluation_config=setup.config,
        tracking_uri=setup.tracking_uri,
        model_directory=setup.config.runtime.model_directory,
        private_output_directory=setup.config.runtime.private_output_directory,
        report_directory=setup.config.runtime.report_directory,
    )
    assert evaluation.model_package_id == training.model_package_id
    assert setup.adapter.test_calls == 1


def test_cxr_publication_failures_remain_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_cxr_lifecycle(tmp_path, monkeypatch)
    training = train_cxr_experiment(setup.config, tracking_uri=setup.tracking_uri)

    def fail_publication(*args, **kwargs):
        raise OSError((args, kwargs))

    monkeypatch.setattr(
        "beyondcxr.training.rsna_evaluate_cxr.publish_rsna_evaluation", fail_publication
    )
    with pytest.raises(OSError):
        evaluate_model_package(
            training.model_package_id,
            evaluation_config=setup.config,
            tracking_uri=setup.tracking_uri,
            model_directory=setup.config.runtime.model_directory,
            private_output_directory=setup.config.runtime.private_output_directory,
            report_directory=setup.config.runtime.report_directory,
        )
    evaluation_root = setup.config.runtime.report_directory / "rsna" / "evaluations"
    assert not evaluation_root.exists() or not any(evaluation_root.iterdir())
    client = configure_mlflow(tracking_uri=setup.tracking_uri)
    training_run = client.get_run(training.run_id)
    failed_runs = client.search_runs(experiment_ids=[training_run.info.experiment_id])
    assert any(
        run.info.status == "FAILED"
        and run.data.tags.get("run_kind") == "test_evaluation"
        and run.data.tags.get("run_complete") == "false"
        for run in failed_runs
    )
    prediction_root = setup.config.runtime.private_output_directory / "predictions" / "rsna"
    predictions = list(prediction_root.glob("prediction-*"))
    assert len(predictions) == 1
    evidence = validate_prediction_evidence(predictions[0])
    assert evidence.manifest["model_package_id"] == training.model_package_id

    monkeypatch.setattr("beyondcxr.training.rsna_train_cxr.write_run_reports", fail_publication)
    with pytest.raises(OSError):
        train_cxr_experiment(setup.config, tracking_uri=setup.tracking_uri)
    assert training.model_path.parent.is_dir()
    runs_after_training_failure = client.search_runs(
        experiment_ids=[training_run.info.experiment_id],
    )
    failed_training = next(
        run
        for run in runs_after_training_failure
        if run.info.status == "FAILED"
        and run.data.tags.get("run_kind") == "training"
        and run.info.run_id != training.run_id
    )
    assert failed_training.data.tags.get("run_complete") == "false"
    assert not (
        setup.config.runtime.report_directory
        / setup.config.dataset.dataset_id
        / "runs"
        / failed_training.info.run_id
    ).exists()


def test_cxr_operational_failure_preserves_published_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_cxr_lifecycle(tmp_path, monkeypatch)
    log_stream = io.StringIO()
    configure_logging("INFO", stream=log_stream)
    original_log_params = mlflow.log_params

    def fail_after_publication(parameters):
        if "model_path" in parameters:
            raise RuntimeError("post-publication metadata failed")
        original_log_params(parameters)

    monkeypatch.setattr(
        "beyondcxr.training.rsna_train_cxr.mlflow.log_params", fail_after_publication
    )

    with pytest.raises(RuntimeError):
        train_cxr_experiment(setup.config, tracking_uri=setup.tracking_uri)

    packages = list((setup.config.runtime.model_directory / "packages").glob("model-package-*"))
    assert len(packages) == 1
    validate_published_neural_model(packages[0])
    client = configure_mlflow(tracking_uri=setup.tracking_uri)
    experiment = client.get_experiment_by_name(setup.config.runtime.experiment_name)
    failed = [
        run
        for run in client.search_runs(experiment_ids=[experiment.experiment_id])
        if run.info.status == "FAILED"
    ]
    assert len(failed) == 1
    assert failed[0].data.tags["run_complete"] != "true"
    output = log_stream.getvalue()
    assert "event=publication_completed" not in output
    assert "event=run_failed" in output
