from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import Dataset

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.hashing import sha256_file
from beyondcxr.data.rsna_cxr_cache import (
    SOURCE_AUTHENTICATION_POLICY_VERSION,
    CxrCacheIdentity,
    CxrCacheSourceAuthentication,
    preprocessing_identity,
)
from beyondcxr.data.rsna_metadata_preprocess import (
    SOURCE_FEATURES,
    build_rsna_preprocessor,
    fitted_rsna_preprocessor_contract,
    save_preprocessor,
)
from beyondcxr.models.fusion_concat import (
    RsnaConcatFusionModel,
    RsnaCxrMetadataConcatModel,
    fusion_architecture_contract,
    initialize_fusion_encoder,
)
from beyondcxr.training.config import (
    ConfigError,
    load_experiment_config,
    with_runtime,
)
from beyondcxr.training.device import resolve_device
from beyondcxr.training.rsna_compare import regenerate_comparison
from beyondcxr.training.rsna_datasets import (
    CxrRunData,
    FusionRunData,
    FusionTestData,
    SourceInventoryIdentity,
)
from beyondcxr.training.rsna_evaluate import evaluate_model_package
from beyondcxr.training.rsna_fusion_source import VerifiedSourceCxr, _validate_source_contract
from beyondcxr.training.rsna_interfaces import DatasetLineage
from beyondcxr.training.rsna_train import main as train_main
from beyondcxr.training.rsna_train_cxr import _manifest as _cxr_manifest
from beyondcxr.training.rsna_train_fusion import (
    _manifest,
    load_validated_rsna_fusion_preprocessor,
    train_fusion_experiment,
)
from beyondcxr.utils.mlflow_utils import configure_mlflow
from beyondcxr.utils.private_predictions import validate_prediction_evidence
from beyondcxr.utils.rsna_neural_publication import (
    FUSION_MANIFEST_FIELDS,
    _validate_fusion_source_package,
    checkpoint_document,
    neural_model_package_id,
    publish_neural_model_package,
    save_neural_checkpoint,
    validate_published_neural_model,
)

_SYNTHETIC_BUNDLE_ID = "bundle-" + "a" * 64
_SYNTHETIC_SPLIT_ID = "split-assignment-" + "b" * 64


class _TinyEncoder(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 1024)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(image.flatten(1))


def _document() -> dict[str, object]:
    return yaml.safe_load(Path("configs/rsna_cxr_metadata_concat.yaml").read_text(encoding="utf-8"))


def _write(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "fusion.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _publish_source_cxr_package(
    tmp_path: Path,
    *,
    model_root: Path,
    fusion_config,
    lineage: DatasetLineage,
    bundle_manifest_sha256: str,
    source_inventory: SourceInventoryIdentity,
    source_authentication: dict[str, object],
    weight_identity: dict[str, object],
    seed: int = 42,
):
    fusion_document = yaml.safe_load(fusion_config.source_bytes)
    source_document = yaml.safe_load(
        Path("configs/rsna_cxr_densenet.yaml").read_text(encoding="utf-8")
    )
    for section in ("dataset", "training", "evaluation"):
        source_document[section] = fusion_document[section]
    source_document["preprocessing"] = {
        "cxr_transform_policy": fusion_document["preprocessing"]["cxr_transform_policy"]
    }
    source_path = tmp_path / "source-cxr.yaml"
    source_path.write_text(yaml.safe_dump(source_document, sort_keys=False), encoding="utf-8")
    source_config = with_runtime(load_experiment_config(source_path), seed=seed)
    checkpoint = checkpoint_document(
        nn.Linear(2, 1).state_dict(),
        selected_epoch=1,
        selected_stage="warmup",
        validation_average_precision=0.7,
    )
    checkpoint_path = save_neural_checkpoint(checkpoint, tmp_path / "source-cxr.pt")
    neural = source_config.neural
    assert neural is not None
    transform_kwargs = {
        "image_size": int(source_config.family.parameters["image_size"]),
        "rotation_degrees": neural.rotation_degrees,
        "translation_fraction": neural.translation_fraction,
        "brightness_jitter": neural.brightness_jitter,
        "contrast_jitter": neural.contrast_jitter,
    }
    runtime = {
        **resolve_device("cpu", mixed_precision=False, pin_memory_policy="disabled").provenance(),
        "cxr_cache_id": "cache-" + "0" * 64,
        "loader_execution": {"lifecycle": "reused", "num_workers": 0, "pin_memory": False},
    }
    manifest = _cxr_manifest(
        config=source_config,
        cxr_data=CxrRunData(
            pd.DataFrame(),
            pd.DataFrame(),
            lineage,
            bundle_manifest_sha256,
            source_inventory,
        ),
        source_authentication=source_authentication,
        commit="source-commit",
        dirty=False,
        lock_hash="5" * 64,
        environment={"environment_python_version": "3.13"},
        runtime=runtime,
        weight_identity=weight_identity,
        train_transform=StandardCxrTransform(training=True, **transform_kwargs).contract(),
        evaluation_transform=StandardCxrTransform(training=False, **transform_kwargs).contract(),
        positive_count=2,
        negative_count=2,
        pos_weight=1.0,
        fit=SimpleNamespace(selected_epoch=1, selected_stage="warmup"),
        final_average_precision=0.7,
        thresholds={"youden_j": 0.5, "target_sensitivity": 0.3},
    )
    return publish_neural_model_package(
        model_root=model_root,
        checkpoint_path=checkpoint_path,
        source_config_bytes=source_config.source_bytes,
        manifest=manifest,
    )


def test_locked_fusion_config_is_seed_free_and_runtime_compatible() -> None:
    paths = [Path("configs/rsna_cxr_metadata_concat.yaml")] * 3
    documents = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    configs = [load_experiment_config(path) for path in paths]

    assert {config.runtime.seed for config in configs} == {None}
    assert {config.family.modalities for config in configs} == {("cxr", "metadata")}
    assert {config.family.family_id for config in configs} == {"cxr_metadata_concat"}
    assert len({config.config_semantic_sha256 for config in configs}) == 1
    assert documents[1:] == documents[:-1]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document["family"]["parameters"].update({"unknown": 1}),
        lambda document: document["family"]["parameters"].pop("fusion_hidden_dimension"),
        lambda document: document["family"].update({"modalities": ["cxr"]}),
        lambda document: document["family"].update({"family_id": "cxr_densenet"}),
        lambda document: document["training"].pop("loader"),
    ],
)
def test_fusion_config_rejects_unknown_missing_and_nonfixed_fields(
    tmp_path: Path, mutation
) -> None:
    document = _document()
    mutation(document)
    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))


def test_fusion_training_requires_explicit_runtime_source_package(capsys) -> None:
    assert train_main(["--config", "configs/rsna_cxr_metadata_concat.yaml", "--seed", "42"]) == 1
    assert "Experiment failed: ValueError" in capsys.readouterr().err


def test_fixed_fusion_model_has_dynamic_structured_width_and_two_stage_ownership() -> None:
    config = load_experiment_config("configs/rsna_cxr_metadata_concat.yaml")
    model = RsnaCxrMetadataConcatModel(encoder_factory=_TinyEncoder).build(
        config.family,
        structured_dimension=7,
        weights=None,
    )
    logits = model(torch.ones((3, 1, 2, 2)), torch.ones((3, 7)))

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
    architecture = fusion_architecture_contract(config.family, structured_input_dimension=7)
    assert (
        model.classifier.image_projection[0].in_features
        == architecture["image_embedding_dimension"]
    )
    assert (
        model.classifier.image_projection[0].out_features
        == architecture["image_projection_dimension"]
    )
    assert (
        model.classifier.structured_projection[0].out_features
        == architecture["structured_hidden_dimension"]
    )
    assert model.classifier.output[0].in_features == architecture["fusion_input_dimension"]
    model.freeze_encoder()
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    model.unfreeze_encoder()
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


def test_fusion_encoder_initialization_loads_only_exact_source_encoder_state() -> None:
    config = load_experiment_config("configs/rsna_cxr_metadata_concat.yaml")
    source = RsnaConcatFusionModel(
        _TinyEncoder(),
        fusion_architecture_contract(config.family, structured_input_dimension=2),
    )
    destination = RsnaConcatFusionModel(
        _TinyEncoder(),
        fusion_architecture_contract(config.family, structured_input_dimension=5),
    )
    source_state = {
        **{f"encoder.{key}": value for key, value in source.encoder.state_dict().items()},
        "classifier.unused": torch.tensor([1.0]),
    }

    initialize_fusion_encoder(destination, source_state)

    for key, value in source.encoder.state_dict().items():
        assert torch.equal(destination.encoder.state_dict()[key], value)


def test_fusion_package_has_exact_artifacts_and_embedded_fitted_preprocessor(
    tmp_path: Path,
) -> None:
    config_path = Path("configs/rsna_cxr_metadata_concat.yaml")
    config = with_runtime(load_experiment_config(config_path), seed=42)
    features = pd.DataFrame(
        {
            "age_years": [20.0, 40.0, 60.0, 80.0],
            "age_is_implausible": [False] * 4,
            "sex": ["F", "M", None, "F"],
            "view_position": ["PA", "AP", "PA", None],
            "pixel_spacing_row_mm": [0.1, 0.2, 0.3, 0.4],
            "pixel_spacing_col_mm": [0.1, 0.2, 0.3, 0.4],
        }
    ).loc[:, SOURCE_FEATURES]
    preprocessor = build_rsna_preprocessor().fit(features)
    contract = fitted_rsna_preprocessor_contract(preprocessor)
    preprocessor_path = save_preprocessor(preprocessor, tmp_path / "preprocessor.skops")
    checkpoint = checkpoint_document(
        nn.Linear(2, 1).state_dict(),
        selected_epoch=3,
        selected_stage="fine_tune",
        validation_average_precision=0.7,
    )
    checkpoint_path = save_neural_checkpoint(checkpoint, tmp_path / "model.pt")
    source_inventory = SourceInventoryIdentity(
        source_inventory_arrow_sha256="1" * 64,
        source_inventory_file_sha256="2" * 64,
    )
    source_authentication = {
        "policy_version": SOURCE_AUTHENTICATION_POLICY_VERSION,
        "partitions": ["train", "validation", "test"],
        "file_count": 4,
        "source_inventory_arrow_sha256": "1" * 64,
        "source_inventory_file_sha256": "2" * 64,
    }
    data = FusionRunData(
        train=features,
        validation=features,
        lineage=DatasetLineage(
            bundle_id=config.dataset.bundle_id,
            split_assignment_id="split-test",
            label_policy_version="label-test",
            task_id="pneumonia",
        ),
        bundle_manifest_sha256="4" * 64,
        source_inventory=source_inventory,
    )
    neural = config.neural
    assert neural is not None
    transform_kwargs = {
        "image_size": 224,
        "rotation_degrees": neural.rotation_degrees,
        "translation_fraction": neural.translation_fraction,
        "brightness_jitter": neural.brightness_jitter,
        "contrast_jitter": neural.contrast_jitter,
    }
    source_weight = {
        "declared_name": "densenet121-res224-chex",
        "stable_identifier": "https://example.invalid/weights.pt",
        "cache_filename": "weights.pt",
        "byte_size": 100,
        "sha256": "8" * 64,
    }
    source_package = _publish_source_cxr_package(
        tmp_path,
        model_root=tmp_path / "models",
        fusion_config=config,
        lineage=data.lineage,
        bundle_manifest_sha256=data.bundle_manifest_sha256,
        source_inventory=source_inventory,
        source_authentication=source_authentication,
        weight_identity=source_weight,
    )
    manifest = _manifest(
        config=config,
        data=data,
        source_authentication=source_authentication,
        commit="commit-test",
        dirty=False,
        lock_hash="5" * 64,
        environment={"environment_python_version": "3.13"},
        runtime={
            **resolve_device(
                "cpu", mixed_precision=False, pin_memory_policy="disabled"
            ).provenance(),
            "cxr_cache_id": "cache-" + "0" * 64,
            "loader_execution": {
                "lifecycle": "reused",
                "num_workers": 0,
                "pin_memory": False,
            },
        },
        source_package_id=source_package.model_package_id,
        source_pretrained_weight=source_weight,
        structured_contract=contract,
        preprocessor_sha256=sha256_file(preprocessor_path),
        train_transform=StandardCxrTransform(training=True, **transform_kwargs).contract(),
        evaluation_transform=StandardCxrTransform(training=False, **transform_kwargs).contract(),
        positive_count=2,
        negative_count=2,
        pos_weight=1.0,
        fit=SimpleNamespace(
            selected_epoch=3,
            selected_stage="fine_tune",
            selected_validation_metric=0.7,
        ),
        thresholds={"youden_j": 0.5, "target_sensitivity": 0.3},
    )
    published = publish_neural_model_package(
        model_root=tmp_path / "models",
        checkpoint_path=checkpoint_path,
        source_config_bytes=config_path.read_bytes(),
        manifest=manifest,
        structured_preprocessor_path=preprocessor_path,
    )

    validated = validate_published_neural_model(published.package_directory)
    assert {path.name for path in published.package_directory.iterdir()} == {
        "model.pt",
        "resolved_config.yaml",
        "manifest.json",
        "structured_preprocessor.skops",
    }
    assert validated["structured_preprocessor_contract"] == contract
    assert (
        fitted_rsna_preprocessor_contract(
            load_validated_rsna_fusion_preprocessor(published.package_directory, validated)
        )
        == contract
    )
    assert validated["fusion_architecture"] == fusion_architecture_contract(
        config.family,
        structured_input_dimension=int(contract["transformed_dimension"]),
    )
    source_document = json.loads(source_package.manifest_path.read_text(encoding="utf-8"))
    source_document["model_identity"]["pretrained_weight"].update(
        {"cache_filename": "alternate-cache-name.pt", "byte_size": 200}
    )
    source_package.manifest_path.write_text(
        json.dumps(source_document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert (
        validate_published_neural_model(source_package.package_directory)["model_package_id"]
        == source_package.model_package_id
    )
    assert (
        validate_published_neural_model(published.package_directory)["model_package_id"]
        == published.model_package_id
    )


def test_fusion_package_identity_binds_source_package() -> None:
    document = {field: field for field in FUSION_MANIFEST_FIELDS}
    document["model_package_schema_version"] = 1
    document["dataset_id"] = "rsna"
    document["bundle_id"] = "bundle-a"
    document["split_assignment_id"] = "split-a"
    document["task_id"] = "pneumonia"
    document["label_policy_version"] = "label-a"
    document["positive_class"] = 1
    document["family_id"] = "cxr_metadata_concat"
    document["modalities"] = ["cxr", "metadata"]
    document["training_policy"] = {"seed": 42}
    document["fit_config"] = {"family": "fusion"}
    document["preprocessor_state_sha256"] = "4" * 64
    document["model_state_sha256"] = "5" * 64
    document["selection"] = {"selected_epoch": 2, "selected_stage": "fine_tune"}
    document["thresholds"] = {"youden_j": 0.5, "target_sensitivity": 0.4}
    document["model_identity"] = {
        "pretrained_weight": {
            "declared_name": "densenet121-res224-chex",
            "stable_identifier": "https://example.invalid/weights.pt",
            "cache_filename": "weights.pt",
            "byte_size": 100,
            "sha256": "8" * 64,
        }
    }
    document["source_package_id"] = "model-package-" + "a" * 64
    baseline = neural_model_package_id(document)
    document["runtime_provenance"] = {"mlflow_run_id": "run-b"}
    assert neural_model_package_id(document) == baseline
    document["source_package_id"] = "model-package-" + "b" * 64
    assert neural_model_package_id(document) != baseline


@pytest.mark.parametrize("source_state", ["missing", "invalid", "incompatible"])
def test_fusion_validation_resolves_source_cxr_package(tmp_path: Path, source_state: str) -> None:
    fusion_path = Path("configs/rsna_cxr_metadata_concat.yaml")
    fusion_config = with_runtime(load_experiment_config(fusion_path), seed=42)
    lineage = DatasetLineage(
        bundle_id=fusion_config.dataset.bundle_id,
        split_assignment_id=fusion_config.dataset.split_assignment_id,
        label_policy_version=fusion_config.task.label_policy_version,
        task_id=fusion_config.task.task_id,
    )
    inventory = SourceInventoryIdentity("1" * 64, "2" * 64)
    authentication = {
        "policy_version": SOURCE_AUTHENTICATION_POLICY_VERSION,
        "partitions": ["train", "validation", "test"],
        "file_count": 4,
        "source_inventory_arrow_sha256": "1" * 64,
        "source_inventory_file_sha256": "2" * 64,
    }
    weight = {
        "declared_name": "densenet121-res224-chex",
        "stable_identifier": "https://example.invalid/weights.pt",
        "cache_filename": "weights.pt",
        "byte_size": 100,
        "sha256": "8" * 64,
    }
    source = _publish_source_cxr_package(
        tmp_path,
        model_root=tmp_path / "models",
        fusion_config=fusion_config,
        lineage=lineage,
        bundle_manifest_sha256="4" * 64,
        source_inventory=inventory,
        source_authentication=authentication,
        weight_identity=weight,
        seed=17 if source_state == "incompatible" else 42,
    )
    fusion_directory = tmp_path / "models/packages/fusion-candidate"
    fusion_directory.mkdir()
    (fusion_directory / "resolved_config.yaml").write_bytes(fusion_path.read_bytes())
    source_id = source.model_package_id
    if source_state == "missing":
        source_id = "model-package-" + "9" * 64
    elif source_state == "invalid":
        source_id = "model-package-" + "7" * 64
        invalid = fusion_directory.parent / source_id
        invalid.mkdir()
        (invalid / "manifest.json").write_text("{}", encoding="utf-8")
    source_manifest = yaml.safe_load(source.manifest_path.read_text(encoding="utf-8"))
    document = {
        **source_manifest,
        "source_package_id": source_id,
        "training_policy": {**source_manifest["training_policy"], "seed": 42},
    }

    with pytest.raises(ValueError):
        _validate_fusion_source_package(fusion_directory, document)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("dataset_id", "other"),
        ("task_id", "other"),
        ("bundle_id", "bundle-other"),
        ("split_assignment_id", "split-other"),
        ("label_policy_version", "label-other"),
        ("model_package_id", "package-other"),
        ("modalities", ["cxr", "labs"]),
        ("family_id", "other-model"),
    ],
)
def test_source_cxr_contract_mismatch_is_rejected(field: str, replacement: object) -> None:
    source_config = with_runtime(load_experiment_config("configs/rsna_cxr_densenet.yaml"), seed=42)
    fusion_config = with_runtime(
        load_experiment_config("configs/rsna_cxr_metadata_concat.yaml"), seed=42
    )
    manifest: dict[str, object] = {
        "dataset_id": fusion_config.dataset.dataset_id,
        "family_id": "cxr_densenet",
        "modalities": ["cxr"],
        "task_id": fusion_config.task.task_id,
        "bundle_id": fusion_config.dataset.bundle_id,
        "bundle_manifest_sha256": "4" * 64,
        "split_assignment_id": fusion_config.dataset.split_assignment_id,
        "label_policy_version": fusion_config.task.label_policy_version,
        "config_source_sha256": source_config.config_source_sha256,
        "config_semantic_sha256": source_config.config_semantic_sha256,
        "checkpoint_sha256": "6" * 64,
        "model_package_id": "model-package-" + "a" * 64,
        "training_policy": {"seed": 42},
        "source_provenance": {
            "git_commit": "source-commit",
            "git_dirty": False,
            "dependency_lock_sha256": "5" * 64,
        },
        "model_identity": {"pretrained_weight": {"declared_name": "densenet121-res224-chex"}},
    }
    manifest[field] = replacement

    with pytest.raises(ValueError):
        _validate_source_contract(
            source_config,
            manifest,
            fusion_config,
            "model-package-" + "a" * 64,
        )


def test_source_cxr_scientific_contract_is_stable_across_reproducibility_witnesses() -> None:
    source_config = with_runtime(load_experiment_config("configs/rsna_cxr_densenet.yaml"), seed=42)
    fusion_config = with_runtime(
        load_experiment_config("configs/rsna_cxr_metadata_concat.yaml"), seed=42
    )
    manifest = {
        "dataset_id": fusion_config.dataset.dataset_id,
        "family_id": "cxr_densenet",
        "modalities": ["cxr"],
        "task_id": fusion_config.task.task_id,
        "bundle_id": fusion_config.dataset.bundle_id,
        "bundle_manifest_sha256": "4" * 64,
        "split_assignment_id": fusion_config.dataset.split_assignment_id,
        "label_policy_version": fusion_config.task.label_policy_version,
        "config_source_sha256": source_config.config_source_sha256,
        "config_semantic_sha256": source_config.config_semantic_sha256,
        "checkpoint_sha256": "6" * 64,
        "model_package_id": "model-package-" + "a" * 64,
        "training_policy": {"seed": 42},
        "source_provenance": {
            "git_commit": "source-commit",
            "git_dirty": False,
            "dependency_lock_sha256": "5" * 64,
        },
        "model_identity": {"pretrained_weight": {"declared_name": "densenet121-res224-chex"}},
    }
    manifest["source_provenance"]["git_commit"] = "different-commit"
    _validate_source_contract(
        source_config,
        manifest,
        fusion_config,
        "model-package-" + "a" * 64,
    )


def test_source_cxr_seed_mismatch_is_rejected_before_package_access() -> None:
    source_config = with_runtime(load_experiment_config("configs/rsna_cxr_densenet.yaml"), seed=17)
    fusion_config = with_runtime(
        load_experiment_config("configs/rsna_cxr_metadata_concat.yaml"), seed=42
    )
    manifest = {
        "dataset_id": fusion_config.dataset.dataset_id,
        "bundle_id": fusion_config.dataset.bundle_id,
        "split_assignment_id": fusion_config.dataset.split_assignment_id,
        "task_id": fusion_config.task.task_id,
        "label_policy_version": fusion_config.task.label_policy_version,
        "family_id": "cxr_densenet",
        "modalities": ["cxr"],
        "model_package_id": "model-package-" + "a" * 64,
        "training_policy": {"seed": 17},
        "model_identity": {"pretrained_weight": {"declared_name": "densenet121-res224-chex"}},
    }
    with pytest.raises(ValueError):
        _validate_source_contract(
            source_config,
            manifest,
            fusion_config,
            "model-package-" + "a" * 64,
        )


class _FusionTensorDataset(Dataset[dict[str, object]]):
    def __init__(self, frame: pd.DataFrame, structured: np.ndarray) -> None:
        self.frame = frame.reset_index(drop=True)
        self.structured = torch.from_numpy(np.asarray(structured, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        return {
            "image": torch.tensor(
                [float(index % 2), 1.0, float(index // 2), 0.5], dtype=torch.float32
            ),
            "structured": self.structured[index],
            "target": torch.tensor(float(row["target"]), dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
            "patient_id": str(row["patient_id"]),
        }


def test_synthetic_fusion_training_package_explicit_evaluation_and_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    document["dataset"]["bundle_id"] = _SYNTHETIC_BUNDLE_ID
    document["dataset"]["bundle_manifest_sha256"] = "5" * 64
    document["dataset"]["split_assignment_id"] = _SYNTHETIC_SPLIT_ID
    document["training"]["loader"].update({"batch_size": 2})
    document["training"]["parameters"].update(
        {
            "mixed_precision": False,
            "warmup_epochs": 1,
            "fine_tune_epochs": 1,
            "early_stopping_patience": 1,
        }
    )
    config_path = _write(tmp_path, document)
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
        split_assignment_id=_SYNTHETIC_SPLIT_ID,
        label_policy_version=config.task.label_policy_version,
        task_id="pneumonia",
    )

    def frame(partition: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "sample_id": f"rsna:{partition}-{index}",
                    "patient_id": f"patient-{partition}-{index}",
                    "image_path": f"images/{index}.dcm",
                    "age_years": 30.0 + index,
                    "age_is_implausible": False,
                    "sex": "F" if index % 2 else "M",
                    "view_position": "PA" if index % 2 else "AP",
                    "pixel_spacing_row_mm": 0.2,
                    "pixel_spacing_col_mm": 0.2,
                    "split_name": partition,
                    "target": index % 2,
                }
                for index in range(4)
            ]
        ).loc[
            :,
            (
                "sample_id",
                "patient_id",
                "image_path",
                *SOURCE_FEATURES,
                "split_name",
                "target",
            ),
        ]

    source_inventory = SourceInventoryIdentity(
        source_inventory_arrow_sha256="1" * 64,
        source_inventory_file_sha256="2" * 64,
    )

    class Adapter:
        test_calls = 0

        def load_fusion_train_validation(self, dataset_config):
            return FusionRunData(
                frame("train"),
                frame("validation"),
                lineage,
                "5" * 64,
                source_inventory,
            )

        def load_fusion_test(self, dataset_config, *, expected_manifest_sha256):
            assert expected_manifest_sha256 == "5" * 64
            self.test_calls += 1
            return FusionTestData(frame("test"), lineage, "5" * 64, source_inventory)

    source_model = RsnaConcatFusionModel(
        _TinyEncoder(),
        fusion_architecture_contract(config.family, structured_input_dimension=2),
    )
    source_weight = {
        "declared_name": "densenet121-res224-chex",
        "stable_identifier": "test",
        "cache_filename": "weights.pt",
        "byte_size": 100,
        "sha256": "9" * 64,
    }
    published_source = _publish_source_cxr_package(
        tmp_path,
        model_root=config.runtime.model_directory,
        fusion_config=config,
        lineage=lineage,
        bundle_manifest_sha256="5" * 64,
        source_inventory=source_inventory,
        source_authentication={
            "policy_version": SOURCE_AUTHENTICATION_POLICY_VERSION,
            "partitions": ["train", "validation", "test"],
            "file_count": 12,
            "source_inventory_arrow_sha256": "1" * 64,
            "source_inventory_file_sha256": "2" * 64,
        },
        weight_identity=source_weight,
    )
    source = VerifiedSourceCxr(
        published_source.model_package_id,
        yaml.safe_load(published_source.manifest_path.read_text(encoding="utf-8")),
        {
            "model_state_dict": {
                **{
                    f"encoder.{key}": value.detach().clone()
                    for key, value in source_model.encoder.state_dict().items()
                },
                "classifier.unused": torch.ones(1),
            }
        },
    )
    adapter = Adapter()
    builder = RsnaCxrMetadataConcatModel(encoder_factory=_TinyEncoder)

    def synthetic_dataset(frame_value, structured, **kwargs):
        del kwargs
        return _FusionTensorDataset(frame_value, structured)

    for module in (
        "beyondcxr.training.rsna_train_fusion",
        "beyondcxr.training.rsna_evaluate_fusion",
    ):
        monkeypatch.setattr(f"{module}.get_dataset", lambda key: adapter)
        monkeypatch.setattr(f"{module}.get_model", lambda key: builder)
        monkeypatch.setattr(f"{module}.RsnaCachedFusionDataset", synthetic_dataset)

        def prepared_cache(*args, **kwargs):
            del kwargs
            transform = args[2]
            identity = CxrCacheIdentity(
                bundle_id=_SYNTHETIC_BUNDLE_ID,
                bundle_manifest_sha256="5" * 64,
                source_inventory_file_sha256="2" * 64,
                source_inventory_arrow_sha256="1" * 64,
                preprocessing_sha256=preprocessing_identity(transform),
            )
            return SimpleNamespace(
                identity=identity,
                source_authentication=CxrCacheSourceAuthentication(
                    policy_version=SOURCE_AUTHENTICATION_POLICY_VERSION,
                    partitions=("train", "validation", "test"),
                    file_count=12,
                    source_inventory_arrow_sha256="1" * 64,
                    source_inventory_file_sha256="2" * 64,
                ),
            )

        monkeypatch.setattr(f"{module}.prepare_rsna_cxr_cache", prepared_cache)
        monkeypatch.setattr(f"{module}.resolve_source_cxr_package", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_train_fusion.git_revision", lambda: ("fusion-commit", False)
    )
    monkeypatch.setattr("beyondcxr.training.rsna_train_fusion.uv_lock_sha256", lambda: "8" * 64)

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    training = train_fusion_experiment(
        config,
        source_cxr_package_id=published_source.model_package_id,
        tracking_uri=tracking_uri,
    )
    assert adapter.test_calls == 0
    package = validate_published_neural_model(training.model_path.parent)
    assert package["source_package_id"] == published_source.model_package_id
    assert package["runtime_provenance"]["loader_execution"] == {
        "lifecycle": "reused",
        "num_workers": 0,
        "pin_memory": False,
    }
    assert (training.model_path.parent / "structured_preprocessor.skops").is_file()

    manifest_path = training.model_path.parent / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    tampered = yaml.safe_load(original_manifest)
    tampered["model_package_id"] = "model-package-" + "0" * 64
    manifest_path.write_text(yaml.safe_dump(tampered), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_model_package(
            training.model_package_id,
            evaluation_config=config,
            tracking_uri=tracking_uri,
            model_directory=config.runtime.model_directory,
        )
    assert adapter.test_calls == 0
    manifest_path.write_bytes(original_manifest)

    evaluation = evaluate_model_package(
        training.model_package_id,
        evaluation_config=config,
        tracking_uri=tracking_uri,
        model_directory=config.runtime.model_directory,
        private_output_directory=config.runtime.private_output_directory,
        report_directory=config.runtime.report_directory,
    )
    assert adapter.test_calls == 1
    private_manifest = validate_prediction_evidence(
        evaluation.private_prediction_directory
    ).manifest
    assert private_manifest["prediction_id"] == evaluation.prediction_id
    assert private_manifest["model_package_id"] == training.model_package_id
    assert evaluation.private_prediction_directory.parent.parent.parent == (
        config.runtime.private_output_directory
    )
    assert not any(path.suffix == ".parquet" for path in evaluation.artifact_directory.rglob("*"))
    recorded = configure_mlflow(tracking_uri=tracking_uri).get_run(evaluation.mlflow_run_id)
    assert (
        configure_mlflow(tracking_uri=tracking_uri).list_artifacts(evaluation.mlflow_run_id) == []
    )
    assert recorded.data.tags["run_complete"] == "true"
    assert recorded.data.params["evaluation_loader_num_workers"] == "0"
    assert recorded.data.params["evaluation_cxr_cache_id"].startswith("cache-")
    comparison_path, _, rows = regenerate_comparison(
        [evaluation.evaluation_id],
        output_directory=config.runtime.report_directory,
        private_directory=config.runtime.private_output_directory,
        model_directory=config.runtime.model_directory,
    )
    assert rows == 1
    assert pd.read_csv(comparison_path)["model_package_id"].tolist() == [training.model_package_id]

    def fail_publication(*args, **kwargs):
        raise OSError((args, kwargs))

    monkeypatch.setattr(
        "beyondcxr.training.rsna_evaluate_fusion.publish_rsna_evaluation", fail_publication
    )
    with pytest.raises(OSError):
        evaluate_model_package(
            training.model_package_id,
            evaluation_config=config,
            tracking_uri=tracking_uri,
            model_directory=config.runtime.model_directory,
            private_output_directory=config.runtime.private_output_directory,
            report_directory=config.runtime.report_directory,
        )
    failed = configure_mlflow(tracking_uri=tracking_uri).search_runs(
        experiment_ids=[recorded.info.experiment_id],
        filter_string="attributes.status = 'FAILED'",
    )
    assert any(run.data.tags.get("run_complete") == "false" for run in failed)
    assert any(run.data.tags.get("run_kind") == "test_evaluation" for run in failed)
