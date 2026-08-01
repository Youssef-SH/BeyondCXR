from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from radfusion.training.config import (
    ConfigError,
    image_seed_compatibility_sha256,
    image_semantic_config_sha256,
    load_experiment_config,
)
from radfusion.training.train import main as train_main


def _image_document() -> dict[str, object]:
    return yaml.safe_load(Path("configs/image_densenet_seed42.yaml").read_text(encoding="utf-8"))


def _write(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "image.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_image_config_is_strict_and_single_seed() -> None:
    config = load_experiment_config("configs/image_densenet_seed42.yaml")

    assert config.config_version == 1
    assert config.training.seed == 42
    assert not hasattr(config.training, "seeds")
    assert config.dataset.dataset_root == Path("data/raw/rsna/extracted")
    assert config.model.modality == "image"
    assert dict(config.model.parameters) == {
        "encoder_name": "densenet121",
        "weights": "densenet121-res224-chex",
        "image_size": 224,
        "embedding_dimension": 1024,
        "class_weighting": "train_pos_weight",
    }
    assert config.image is not None
    assert config.image.rotation_degrees == 7.0
    assert config.image.translation_fraction == 0.05
    assert config.image.pin_memory_policy == "auto"


def test_locked_image_seed_configs_differ_only_by_training_seed() -> None:
    paths = (
        Path("configs/image_densenet_seed17.yaml"),
        Path("configs/image_densenet_seed42.yaml"),
        Path("configs/image_densenet_seed2026.yaml"),
    )
    documents = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    configs = [load_experiment_config(path) for path in paths]
    seeds = {config.training.seed for config in configs}
    semantic_hashes = {image_semantic_config_sha256(config) for config in configs}
    compatibility_hashes = {image_seed_compatibility_sha256(config) for config in configs}

    for document in documents:
        del document["training"]["seed"]

    assert seeds == {17, 42, 2026}
    assert documents[1:] == documents[:-1]
    assert len(semantic_hashes) == 3
    assert len(compatibility_hashes) == 1


def test_image_semantic_config_identity_excludes_paths_but_binds_training_meaning(
    tmp_path: Path,
) -> None:
    baseline = load_experiment_config("configs/image_densenet_seed42.yaml")
    path_changed = _image_document()
    path_changed["dataset"]["dataset_root"] = "/different/raw/root"
    path_changed["dataset"]["manifest_directory"] = "/different/manifests"
    path_changed["training"]["model_directory"] = "/different/models"
    path_changed["training"]["report_directory"] = "/different/reports"
    changed_path = load_experiment_config(_write(tmp_path, path_changed))
    meaning_changed = _image_document()
    meaning_changed["training"]["seed"] = 17
    meaning_path = tmp_path / "meaning.yaml"
    meaning_path.write_text(yaml.safe_dump(meaning_changed, sort_keys=False), encoding="utf-8")
    changed_meaning = load_experiment_config(meaning_path)

    assert image_semantic_config_sha256(changed_path) == image_semantic_config_sha256(baseline)
    assert image_semantic_config_sha256(changed_meaning) != image_semantic_config_sha256(baseline)


def test_operational_image_fields_do_not_change_semantic_identity(tmp_path: Path) -> None:
    baseline = load_experiment_config("configs/image_densenet_seed42.yaml")
    document = _image_document()
    document["evaluation"]["latency_warmup_calls"] = 1
    document["evaluation"]["latency_measured_calls"] = 2
    changed = load_experiment_config(_write(tmp_path, document))

    assert image_semantic_config_sha256(changed) == image_semantic_config_sha256(baseline)
    assert changed.source_sha256 != baseline.source_sha256


def test_metadata_configs_use_the_explicit_metadata_modality() -> None:
    for path in ("configs/metadata_logistic.yaml", "configs/metadata_lightgbm.yaml"):
        config = load_experiment_config(path)
        assert config.model.modality == "metadata"
        assert config.image is None
        assert config.dataset.dataset_root is None


def test_image_config_dispatches_to_image_runner(monkeypatch, capsys) -> None:
    captured = {}

    def fake_training(config, *, tracking_uri):
        captured.update(config=config, tracking_uri=tracking_uri)
        return type(
            "Result",
            (),
            {
                "model_name": "image_densenet",
                "run_id": "image-run",
                "validation_probability": type("Metrics", (), {"average_precision": 0.5})(),
                "model_path": Path("models/rsna/runs/image-run/model.pt"),
                "artifact_directory": Path("reports/rsna/runs/image-run"),
            },
        )()

    monkeypatch.setattr("radfusion.training.train.train_image_experiment", fake_training)

    assert train_main(["--config", "configs/image_densenet_seed42.yaml"]) == 0
    assert captured["config"].model.modality == "image"
    assert captured["tracking_uri"] == "sqlite:///mlflow.db"
    assert '"mlflow_run_id": "image-run"' in capsys.readouterr().out


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document["training"].update({"seeds": [42]}),
        lambda document: document["image"].update({"unknown": 1}),
        lambda document: document["image"].pop("batch_size"),
        lambda document: document["model"].pop("modality"),
    ],
)
def test_image_config_rejects_unknown_and_missing_fields(
    tmp_path: Path,
    mutation,
) -> None:
    document = _image_document()
    mutation(document)

    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document["model"].update({"registry_key": "metadata_logistic"}),
        lambda document: document["model"].update({"registry_key": "metadata_lightgbm"}),
        lambda document: document["dataset"].pop("dataset_root"),
        lambda document: document.pop("image"),
        lambda document: document["model"].update({"fit_parameters": {"epochs": 1}}),
        lambda document: (
            document["model"].update({"modality": "metadata", "registry_key": "image_densenet"}),
            document.pop("image"),
            document["dataset"].pop("dataset_root"),
        ),
        lambda document: (
            document["model"].update({"modality": "metadata", "registry_key": "metadata_logistic"}),
            document.pop("image"),
        ),
        lambda document: (
            document["model"].update({"modality": "metadata", "registry_key": "metadata_logistic"}),
            document["dataset"].pop("dataset_root"),
        ),
    ],
)
def test_modality_cross_field_contract_is_closed(
    tmp_path: Path,
    mutation,
) -> None:
    document = _image_document()
    mutation(document)

    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("encoder_name", "resnet50"),
        ("weights", "other"),
        ("image_size", True),
        ("embedding_dimension", 512),
        ("class_weighting", "none"),
    ],
)
def test_image_model_contract_is_fixed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    document = _image_document()
    document["model"]["parameters"][field] = value

    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", True),
        ("head_learning_rate", float("nan")),
        ("head_learning_rate", float("inf")),
        ("encoder_learning_rate", float("-inf")),
        ("translation_fraction", 1.1),
        ("device", "mps"),
        ("optimizer", "sgd"),
        ("pin_memory_policy", "always"),
        ("scheduler_factor", 1.0),
        ("early_stopping_patience", -1),
    ],
)
def test_image_runtime_and_optimization_values_are_strict(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    document = _image_document()
    document["image"][field] = value

    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))
