from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import Dataset

from radfusion.data.cxr_cache import (
    SOURCE_AUTHENTICATION_POLICY_VERSION,
    CxrCacheIdentity,
    CxrCacheSourceAuthentication,
    preprocessing_identity,
)
from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.hashing import sha256_file
from radfusion.data.tabular_preprocess import (
    SOURCE_FEATURES,
    build_rsna_preprocessor,
    fitted_rsna_preprocessor_contract,
    save_preprocessor,
)
from radfusion.models.fusion_concat import (
    FusionConcatModel,
    RsnaConcatFusionModel,
    initialize_fusion_encoder,
)
from radfusion.training.compare import regenerate_comparison
from radfusion.training.config import (
    ConfigError,
    fusion_architecture_contract,
    fusion_seed_compatibility_sha256,
    fusion_semantic_config_sha256,
    image_semantic_config_sha256,
    load_experiment_config,
)
from radfusion.training.datasets import (
    FusionRunData,
    FusionTestData,
    SourceInventoryIdentity,
)
from radfusion.training.device import resolve_device
from radfusion.training.evaluate import evaluate_training_run
from radfusion.training.fusion_source import (
    SourceCxrLineage,
    VerifiedSourceCxr,
    _validate_source_contract,
    resolve_source_cxr_training_run,
)
from radfusion.training.interfaces import DatasetLineage
from radfusion.training.train import main as train_main
from radfusion.training.train_fusion import (
    _manifest,
    load_validated_rsna_fusion_preprocessor,
    train_fusion_experiment,
)
from radfusion.utils.mlflow_utils import configure_mlflow
from radfusion.utils.neural_publication import (
    FUSION_MANIFEST_FIELDS,
    checkpoint_document,
    neural_model_package_id,
    publish_neural_model_run,
    save_neural_checkpoint,
    validate_published_neural_model,
)
from radfusion.utils.private_predictions import validate_private_neural_predictions

_SYNTHETIC_BUNDLE_ID = "build-" + "a" * 64


class _TinyEncoder(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 1024)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(image.flatten(1))


def _document() -> dict[str, object]:
    return yaml.safe_load(Path("configs/fusion_concat_seed42.yaml").read_text(encoding="utf-8"))


def _write(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "fusion.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_locked_fusion_configs_are_strict_and_differ_only_by_seed() -> None:
    paths = [
        Path("configs/fusion_concat_seed17.yaml"),
        Path("configs/fusion_concat_seed42.yaml"),
        Path("configs/fusion_concat_seed2026.yaml"),
    ]
    documents = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    configs = [load_experiment_config(path) for path in paths]

    assert {config.training.seed for config in configs} == {17, 42, 2026}
    assert {config.model.modality for config in configs} == {"fusion"}
    assert {config.model.registry_key for config in configs} == {"fusion_concat"}
    assert all("source_training_run_id" not in str(document) for document in documents)
    assert len({fusion_semantic_config_sha256(config) for config in configs}) == 3
    assert len({fusion_seed_compatibility_sha256(config) for config in configs}) == 1
    for document in documents:
        del document["training"]["seed"]
    assert documents[1:] == documents[:-1]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document["model"]["parameters"].update({"dropout": 0.3}),
        lambda document: document["model"]["parameters"].update({"unknown": 1}),
        lambda document: document["model"]["parameters"].pop("fusion_hidden_dimension"),
        lambda document: document["model"].update({"modality": "image"}),
        lambda document: document.pop("image"),
    ],
)
def test_fusion_config_rejects_unknown_missing_and_nonfixed_fields(
    tmp_path: Path, mutation
) -> None:
    document = _document()
    mutation(document)
    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))


def test_fusion_training_requires_explicit_runtime_source_run(capsys) -> None:
    assert train_main(["--config", "configs/fusion_concat_seed42.yaml"]) == 1
    assert "--source-training-run-id" in capsys.readouterr().err


def test_fixed_fusion_model_has_dynamic_structured_width_and_two_stage_ownership() -> None:
    config = load_experiment_config("configs/fusion_concat_seed42.yaml")
    model = FusionConcatModel(encoder_factory=_TinyEncoder).build(
        config.model,
        structured_dimension=7,
        weights=None,
    )
    logits = model(torch.ones((3, 1, 2, 2)), torch.ones((3, 7)))

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
    architecture = fusion_architecture_contract(config.model, structured_input_dimension=7)
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
    config = load_experiment_config("configs/fusion_concat_seed42.yaml")
    source = RsnaConcatFusionModel(
        _TinyEncoder(),
        fusion_architecture_contract(config.model, structured_input_dimension=2),
    )
    destination = RsnaConcatFusionModel(
        _TinyEncoder(),
        fusion_architecture_contract(config.model, structured_input_dimension=5),
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
    config_path = Path("configs/fusion_concat_seed42.yaml")
    config = load_experiment_config(config_path)
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
    image = config.image
    assert image is not None
    transform_kwargs = {
        "image_size": 224,
        "rotation_degrees": image.rotation_degrees,
        "translation_fraction": image.translation_fraction,
        "brightness_jitter": image.brightness_jitter,
        "contrast_jitter": image.contrast_jitter,
    }
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
        source_lineage=SourceCxrLineage(
            training_run_id="source-run",
            model_package_id="source-package",
            checkpoint_sha256="6" * 64,
            semantic_config_sha256="7" * 64,
            git_commit="commit-test",
            dependency_lock_sha256="5" * 64,
        ),
        source_pretrained_weight={
            "declared_name": "densenet121-res224-chex",
            "stable_identifier": "https://example.invalid/weights.pt",
            "cache_filename": "weights.pt",
            "byte_size": 100,
            "sha256": "8" * 64,
        },
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
            selected_validation_average_precision=0.7,
        ),
        thresholds={"youden_j": 0.5, "target_sensitivity": 0.3},
    )
    published = publish_neural_model_run(
        model_root=tmp_path / "models",
        mlflow_run_id="fusion-run",
        checkpoint_path=checkpoint_path,
        source_config_bytes=config_path.read_bytes(),
        manifest=manifest,
        structured_preprocessor_path=preprocessor_path,
    )

    validated = validate_published_neural_model(published.run_directory)
    assert {path.name for path in published.run_directory.iterdir()} == {
        "model.pt",
        "resolved_config.yaml",
        "model_manifest.json",
        "structured_preprocessor.skops",
    }
    assert validated["structured_preprocessor_contract"] == contract
    assert (
        fitted_rsna_preprocessor_contract(
            load_validated_rsna_fusion_preprocessor(published.run_directory, validated)
        )
        == contract
    )
    assert validated["fusion_architecture"] == fusion_architecture_contract(
        config.model,
        structured_input_dimension=int(contract["transformed_dimension"]),
    )


def test_fusion_package_identity_binds_source_package_but_not_source_run() -> None:
    document = {field: field for field in FUSION_MANIFEST_FIELDS}
    document["modality"] = "fusion"
    document["source_cxr_lineage"] = {
        "training_run_id": "run-a",
        "model_package_id": "package-a",
        "checkpoint_sha256": "1" * 64,
        "semantic_config_sha256": "2" * 64,
        "git_commit": "commit-a",
        "dependency_lock_sha256": "3" * 64,
    }
    baseline = neural_model_package_id(document)
    document["source_cxr_lineage"]["training_run_id"] = "run-b"
    assert neural_model_package_id(document) == baseline
    document["source_cxr_lineage"]["model_package_id"] = "package-b"
    assert neural_model_package_id(document) != baseline


class _SourceRecord(SimpleNamespace):
    def integer_seed(self) -> int:
        return int(self.seed)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("dataset", "other"),
        ("task", "other"),
        ("bundle_id", "build-other"),
        ("split_assignment_id", "split-other"),
        ("label_policy_version", "label-other"),
        ("model_package_id", "package-other"),
        ("checkpoint_sha256", "9" * 64),
        ("local_model_sha256", "9" * 64),
        ("git_commit", "other-commit"),
        ("dependency_lock_sha256", "9" * 64),
        ("bundle_manifest_sha256", "9" * 64),
    ],
)
def test_source_cxr_lineage_mismatch_is_rejected(field: str, replacement: str) -> None:
    source_config = load_experiment_config("configs/image_densenet_seed42.yaml")
    fusion_config = load_experiment_config("configs/fusion_concat_seed42.yaml")
    manifest = {
        "training_mlflow_run_id": "source-run",
        "modality": "image",
        "model": "image_densenet",
        "task": "pneumonia",
        "bundle_id": fusion_config.dataset.bundle_id,
        "bundle_manifest_sha256": "4" * 64,
        "split_assignment_id": "split-test",
        "label_policy_version": "label-test",
        "source_config_sha256": source_config.source_sha256,
        "semantic_config_sha256": image_semantic_config_sha256(source_config),
        "checkpoint_sha256": "6" * 64,
        "model_package_id": "source-package",
        "training_policy": {"seed": 42},
        "source_provenance": {
            "git_commit": "source-commit",
            "git_dirty": False,
            "dependency_lock_sha256": "5" * 64,
        },
        "model_identity": {"pretrained_weight": {"declared_name": "densenet121-res224-chex"}},
    }
    record = _SourceRecord(
        run_id="source-run",
        seed="42",
        dataset="rsna",
        task="pneumonia",
        bundle_id=fusion_config.dataset.bundle_id,
        model_package_id="source-package",
        split_assignment_id="split-test",
        label_policy_version="label-test",
        semantic_config_sha256=image_semantic_config_sha256(source_config),
        checkpoint_sha256="6" * 64,
        local_model_sha256="6" * 64,
        git_commit="source-commit",
        git_dirty="false",
        dependency_lock_sha256="5" * 64,
        bundle_manifest_sha256="4" * 64,
    )
    setattr(record, field, replacement)

    with pytest.raises(ValueError):
        _validate_source_contract(
            record,
            source_config,
            manifest,
            fusion_config,
            training_run_id="source-run",
            current_git_commit="source-commit",
            current_git_dirty=False,
            current_dependency_lock_sha256="5" * 64,
        )


def test_source_cxr_git_commit_must_match_current_fusion_revision() -> None:
    source_config = load_experiment_config("configs/image_densenet_seed42.yaml")
    fusion_config = load_experiment_config("configs/fusion_concat_seed42.yaml")
    manifest = {
        "training_mlflow_run_id": "source-run",
        "modality": "image",
        "model": "image_densenet",
        "task": "pneumonia",
        "bundle_id": fusion_config.dataset.bundle_id,
        "bundle_manifest_sha256": "4" * 64,
        "split_assignment_id": "split-test",
        "label_policy_version": "label-test",
        "source_config_sha256": source_config.source_sha256,
        "semantic_config_sha256": image_semantic_config_sha256(source_config),
        "checkpoint_sha256": "6" * 64,
        "model_package_id": "source-package",
        "training_policy": {"seed": 42},
        "source_provenance": {
            "git_commit": "source-commit",
            "git_dirty": False,
            "dependency_lock_sha256": "5" * 64,
        },
        "model_identity": {"pretrained_weight": {"declared_name": "densenet121-res224-chex"}},
    }
    record = _SourceRecord(
        run_id="source-run",
        seed="42",
        dataset="rsna",
        task="pneumonia",
        bundle_id=fusion_config.dataset.bundle_id,
        model_package_id="source-package",
        split_assignment_id="split-test",
        label_policy_version="label-test",
        semantic_config_sha256=image_semantic_config_sha256(source_config),
        checkpoint_sha256="6" * 64,
        local_model_sha256="6" * 64,
        git_commit="source-commit",
        git_dirty="false",
        dependency_lock_sha256="5" * 64,
        bundle_manifest_sha256="4" * 64,
    )

    with pytest.raises(ValueError):
        _validate_source_contract(
            record,
            source_config,
            manifest,
            fusion_config,
            training_run_id="source-run",
            current_git_commit="different-commit",
            current_git_dirty=False,
            current_dependency_lock_sha256="5" * 64,
        )


def test_source_cxr_seed_mismatch_is_rejected_before_package_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _SourceRecord(
        run_kind="training",
        evaluation_scope="validation",
        modality="image",
        model="image_densenet",
        seed="17",
    )
    monkeypatch.setattr(
        "radfusion.training.fusion_source.require_completed_run", lambda run: record
    )
    monkeypatch.setattr(
        "radfusion.training.fusion_source.validate_neural_package_metadata",
        lambda path: pytest.fail("package access must follow the seed gate"),
    )

    with pytest.raises(ValueError):
        resolve_source_cxr_training_run(
            SimpleNamespace(get_run=lambda run_id: object()),
            "source-run",
            load_experiment_config("configs/fusion_concat_seed42.yaml"),
            current_git_commit="fusion-commit",
            current_git_dirty=False,
            current_dependency_lock_sha256="5" * 64,
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
    document["dataset"].update(
        {
            "bundle_id": _SYNTHETIC_BUNDLE_ID,
            "dataset_root": str(tmp_path / "raw"),
            "manifest_directory": str(tmp_path / "manifests"),
        }
    )
    document["training"].update(
        {
            "model_directory": str(tmp_path / "models" / "rsna"),
            "report_directory": str(tmp_path / "reports"),
        }
    )
    document["image"].update(
        {
            "batch_size": 2,
            "num_workers": 0,
            "device": "cpu",
            "mixed_precision": False,
            "warmup_epochs": 1,
            "fine_tune_epochs": 1,
            "early_stopping_patience": 1,
        }
    )
    config_path = _write(tmp_path, document)
    config = load_experiment_config(config_path)
    lineage = DatasetLineage(
        bundle_id=_SYNTHETIC_BUNDLE_ID,
        split_assignment_id="split-synthetic",
        label_policy_version="label-v1",
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
        fusion_architecture_contract(config.model, structured_input_dimension=2),
    )
    source_lineage = SourceCxrLineage(
        training_run_id="source-image-run",
        model_package_id="source-image-package",
        checkpoint_sha256="6" * 64,
        semantic_config_sha256="7" * 64,
        git_commit="fusion-commit",
        dependency_lock_sha256="8" * 64,
    )
    source = VerifiedSourceCxr(
        source_lineage,
        {
            "bundle_id": _SYNTHETIC_BUNDLE_ID,
            "bundle_manifest_sha256": "5" * 64,
            "split_assignment_id": "split-synthetic",
            "task": "pneumonia",
            "label_policy_version": "label-v1",
            "model_identity": {
                "pretrained_weight": {
                    "declared_name": "densenet121-res224-chex",
                    "stable_identifier": "test",
                    "cache_filename": "weights.pt",
                    "byte_size": 100,
                    "sha256": "9" * 64,
                }
            },
        },
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
    builder = FusionConcatModel(encoder_factory=_TinyEncoder)

    def synthetic_dataset(frame_value, structured, **kwargs):
        del kwargs
        return _FusionTensorDataset(frame_value, structured)

    for module in ("radfusion.training.train_fusion", "radfusion.training.evaluate_fusion"):
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
        monkeypatch.setattr(f"{module}.git_revision", lambda: ("fusion-commit", False))
        monkeypatch.setattr(f"{module}.uv_lock_sha256", lambda: "8" * 64)
        monkeypatch.setattr(
            f"{module}.resolve_source_cxr_training_run", lambda *args, **kwargs: source
        )

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    training = train_fusion_experiment(
        config,
        source_training_run_id="source-image-run",
        tracking_uri=tracking_uri,
    )
    assert adapter.test_calls == 0
    package = validate_published_neural_model(training.model_path.parent)
    assert package["source_cxr_lineage"] == source_lineage.as_dict()
    assert package["runtime_provenance"]["loader_execution"] == {
        "lifecycle": "reused",
        "num_workers": 0,
        "pin_memory": False,
    }
    assert (training.model_path.parent / "structured_preprocessor.skops").is_file()

    manifest_path = training.model_path.parent / "model_manifest.json"
    original_manifest = manifest_path.read_bytes()
    tampered = yaml.safe_load(original_manifest)
    tampered["model_package_id"] = "model-package-" + "0" * 64
    manifest_path.write_text(yaml.safe_dump(tampered), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_training_run(training.run_id, tracking_uri=tracking_uri)
    assert adapter.test_calls == 0
    manifest_path.write_bytes(original_manifest)

    evaluation = evaluate_training_run(training.run_id, tracking_uri=tracking_uri)
    assert adapter.test_calls == 1
    assert evaluation.training_run_id == training.run_id
    private_manifest = validate_private_neural_predictions(evaluation.private_prediction_directory)
    assert private_manifest["training_run_id"] == training.run_id
    assert private_manifest["test_evaluation_run_id"] == evaluation.run_id
    assert private_manifest["model_package_id"] == training.model_package_id
    assert not any(path.suffix == ".parquet" for path in evaluation.artifact_directory.rglob("*"))
    recorded = configure_mlflow(tracking_uri=tracking_uri).get_run(evaluation.run_id)
    assert configure_mlflow(tracking_uri=tracking_uri).list_artifacts(evaluation.run_id) == []
    assert recorded.data.tags["run_complete"] == "true"
    assert recorded.data.tags["source_cxr_checkpoint_sha256"] == "6" * 64
    assert recorded.data.params["evaluation_loader_num_workers"] == "0"
    assert recorded.data.params["evaluation_cxr_cache_id"].startswith("cache-")
    assert float(recorded.data.tags["threshold_youden_j"]) == package["thresholds"]["youden_j"]
    comparison_path, _, rows = regenerate_comparison(
        tracking_uri=tracking_uri,
        output_directory=tmp_path / "comparison",
    )
    assert rows == 1
    assert pd.read_csv(comparison_path)["modality"].tolist() == ["fusion"]

    def fail_publication(*args, **kwargs):
        raise OSError((args, kwargs))

    monkeypatch.setattr("radfusion.training.evaluate_fusion.publish_directory", fail_publication)
    with pytest.raises(OSError):
        evaluate_training_run(training.run_id, tracking_uri=tracking_uri)
    failed = configure_mlflow(tracking_uri=tracking_uri).search_runs(
        experiment_ids=[recorded.info.experiment_id],
        filter_string="attributes.status = 'FAILED'",
    )
    assert any(run.data.tags.get("run_complete") == "false" for run in failed)
    assert not any(
        (tmp_path / "private/predictions/rsna" / run.info.run_id).exists() for run in failed
    )
