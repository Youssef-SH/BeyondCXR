from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from radfusion.training.config import ConfigError, load_experiment_config, with_runtime
from radfusion.training.train import main as train_main


def _image_document() -> dict[str, object]:
    return yaml.safe_load(Path("configs/rsna_cxr_densenet.yaml").read_text(encoding="utf-8"))


def _write(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "image.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_image_config_is_strict_and_seed_free() -> None:
    config = load_experiment_config("configs/rsna_cxr_densenet.yaml")

    assert config.config_version == 1
    assert config.runtime.seed is None
    assert not hasattr(config.training, "seed")
    assert config.runtime.source_root == Path("data/raw/rsna/extracted")
    assert config.family.modalities == ("cxr",)
    assert dict(config.family.parameters) == {
        "encoder_name": "densenet121",
        "weights": "densenet121-res224-chex",
        "image_size": 224,
        "embedding_dimension": 1024,
    }
    assert config.training.parameters["class_weighting"] == "train_pos_weight"
    assert config.training.parameters["fine_tune_scope"] == "all"
    assert config.preprocessing["cxr_transform_policy"] == ("torchxrayvision-densenet121-res224-v1")
    assert config.neural is not None
    assert config.neural.rotation_degrees == 7.0
    assert config.neural.translation_fraction == 0.05
    assert config.runtime.num_workers == 2
    assert config.runtime.pin_memory_policy == "auto"


def test_runtime_seeds_share_one_locked_image_config() -> None:
    path = Path("configs/rsna_cxr_densenet.yaml")
    documents = [yaml.safe_load(path.read_text(encoding="utf-8")) for _ in range(3)]
    baseline = load_experiment_config(path)
    configs = [with_runtime(baseline, seed=seed) for seed in (17, 42, 2026)]
    seeds = {config.runtime.seed for config in configs}
    semantic_hashes = {config.config_semantic_sha256 for config in configs}

    assert seeds == {17, 42, 2026}
    assert documents[1:] == documents[:-1]
    assert len(semantic_hashes) == 1


def test_image_semantic_config_identity_excludes_paths_but_binds_training_meaning(
    tmp_path: Path,
) -> None:
    baseline = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    changed_path = with_runtime(
        baseline,
        source_root="/different/raw/root",
        manifest_directory="/different/manifests",
        model_directory="/different/models",
        report_directory="/different/reports",
    )
    meaning_changed = _image_document()
    meaning_changed["training"]["augmentation"]["rotation_degrees"] = 8.0
    meaning_path = tmp_path / "meaning.yaml"
    meaning_path.write_text(yaml.safe_dump(meaning_changed, sort_keys=False), encoding="utf-8")
    changed_meaning = load_experiment_config(meaning_path)

    assert changed_path.config_semantic_sha256 == baseline.config_semantic_sha256
    assert changed_meaning.config_semantic_sha256 != baseline.config_semantic_sha256


@pytest.mark.parametrize("field", ["latency_warmup_calls", "latency_measured_calls"])
def test_operational_latency_fields_are_rejected_by_scientific_config(
    tmp_path: Path, field: str
) -> None:
    document = _image_document()
    document["evaluation"][field] = 1

    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))


def test_loader_execution_changes_are_semantically_neutral_but_batch_size_is_not(
    tmp_path: Path,
) -> None:
    baseline = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    execution_changed = with_runtime(
        baseline,
        num_workers=24,
        pin_memory_policy="disabled",
    )
    document = _image_document()
    document["training"]["loader"]["batch_size"] = 16
    batch_changed = load_experiment_config(_write(tmp_path, document))

    assert execution_changed.config_semantic_sha256 == baseline.config_semantic_sha256
    assert batch_changed.config_semantic_sha256 != baseline.config_semantic_sha256


def test_scientific_evaluation_policy_changes_semantic_identity(tmp_path: Path) -> None:
    baseline = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    document = _image_document()
    document["evaluation"]["calibration_bins"] = 20
    changed = load_experiment_config(_write(tmp_path, document))

    assert changed.config_semantic_sha256 != baseline.config_semantic_sha256


def test_metadata_configs_use_the_explicit_metadata_modality() -> None:
    for path in ("configs/rsna_metadata_logistic.yaml", "configs/rsna_metadata_lightgbm.yaml"):
        config = load_experiment_config(path)
        assert config.family.modalities == ("metadata",)
        assert config.neural is None
        assert config.runtime.source_root == Path("data/raw/rsna/extracted")


def test_image_config_dispatches_to_image_runner(monkeypatch, capsys) -> None:
    captured = {}

    def fake_training(config, *, tracking_uri):
        captured.update(config=config, tracking_uri=tracking_uri)
        return type(
            "Result",
            (),
            {
                "model_name": "cxr_densenet",
                "run_id": "image-run",
                "validation_probability": type("Metrics", (), {"average_precision": 0.5})(),
                "model_path": Path("models/rsna/runs/image-run/model.pt"),
                "artifact_directory": Path("reports/rsna/runs/image-run"),
            },
        )()

    monkeypatch.setattr("radfusion.training.train.train_image_experiment", fake_training)

    assert train_main(["--config", "configs/rsna_cxr_densenet.yaml", "--seed", "42"]) == 0
    assert captured["config"].family.modalities == ("cxr",)
    assert captured["tracking_uri"] == "sqlite:///mlflow.db"
    assert '"mlflow_run_id": "image-run"' in capsys.readouterr().out


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document["training"].update({"seeds": [42]}),
        lambda document: document["training"]["loader"].update({"unknown": 1}),
        lambda document: document["training"]["loader"].pop("batch_size"),
        lambda document: document["family"].pop("modalities"),
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
        lambda document: document["family"].update({"family_id": "metadata_logistic"}),
        lambda document: document["family"].update({"family_id": "metadata_lightgbm"}),
        lambda document: document["dataset"].update({"dataset_id": "symile"}),
        lambda document: document["training"].pop("augmentation"),
        lambda document: document["family"].update({"fit_parameters": {"epochs": 1}}),
        lambda document: (
            document["family"].update({"modalities": ["metadata"]}),
            document["training"].pop("augmentation"),
        ),
        lambda document: (
            document["family"].update({"family_id": "metadata_logistic"}),
            document["family"].update({"modalities": ["metadata"]}),
        ),
        lambda document: (
            document["family"].update({"family_id": "metadata_logistic"}),
            document["family"].update({"modalities": ["cxr"]}),
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
    ],
)
def test_image_model_contract_is_fixed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    document = _image_document()
    document["family"]["parameters"][field] = value

    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))


def test_executable_alternate_scientific_value_is_representable(tmp_path: Path) -> None:
    document = _image_document()
    document["training"]["parameters"]["early_stopping_patience"] = 6

    config = load_experiment_config(_write(tmp_path, document))

    assert config.training.parameters["early_stopping_patience"] == 6


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("family", "embedding_dimension", 512),
        ("training", "class_weighting", "none"),
    ],
)
def test_unexecutable_scientific_values_are_rejected(
    tmp_path: Path, section: str, field: str, value: object
) -> None:
    document = _image_document()
    document[section]["parameters"][field] = value

    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", True),
        ("warmup_epochs", 0),
        ("fine_tune_epochs", 0),
        ("warmup_head_learning_rate", 0.0),
        ("encoder_learning_rate", 0.0),
        ("head_learning_rate", 0.0),
        ("gradient_clip_norm", 0.0),
        ("weight_decay", -0.1),
        ("scheduler_min_learning_rate", -0.1),
        ("early_stopping_min_delta", -0.1),
        ("head_learning_rate", float("nan")),
        ("head_learning_rate", float("inf")),
        ("encoder_learning_rate", float("-inf")),
        ("rotation_degrees", 180.1),
        ("translation_fraction", 1.1),
        ("brightness_jitter", 1.1),
        ("contrast_jitter", 1.1),
        ("optimizer", "sgd"),
        ("num_workers", 4),
        ("pin_memory_policy", "always"),
        ("scheduler_factor", 0.0),
        ("scheduler_factor", 1.0),
        ("scheduler_min_learning_rate", 0.00001),
        ("early_stopping_patience", -1),
    ],
)
def test_image_runtime_and_optimization_values_are_strict(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    document = _image_document()
    section = (
        "loader" if field in {"batch_size", "num_workers", "pin_memory_policy"} else "parameters"
    )
    if field in {
        "rotation_degrees",
        "translation_fraction",
        "brightness_jitter",
        "contrast_jitter",
    }:
        section = "augmentation"
    document["training"][section][field] = value

    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weight_decay", 0.0),
        ("scheduler_min_learning_rate", 0.0),
        ("early_stopping_min_delta", 0.0),
        ("rotation_degrees", 180.0),
        ("translation_fraction", 1.0),
        ("brightness_jitter", 1.0),
        ("contrast_jitter", 1.0),
    ],
)
def test_neural_closed_interval_boundaries_are_executable(
    tmp_path: Path, field: str, value: float
) -> None:
    document = _image_document()
    section = "augmentation" if field in document["training"]["augmentation"] else "parameters"
    document["training"][section][field] = value

    assert load_experiment_config(_write(tmp_path, document)).neural is not None
