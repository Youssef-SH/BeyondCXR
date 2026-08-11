from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from radfusion.training.config import (
    SYMILE_M5_FAMILIES,
    ConfigError,
    load_experiment_config,
    load_symile_development_config,
    symile_development_semantic_sha256,
)


def test_supported_configs_load_as_immutable_typed_values() -> None:
    logistic = load_experiment_config("configs/metadata_logistic.yaml")
    lightgbm = load_experiment_config("configs/metadata_lightgbm.yaml")

    assert logistic.config_version == lightgbm.config_version == 1
    assert logistic.dataset.registry_key == "rsna"
    assert logistic.dataset.bundle_id.startswith("build-")
    assert logistic.model.registry_key == "metadata_logistic"
    assert "class_weight" not in logistic.model.parameters
    assert lightgbm.model.registry_key == "metadata_lightgbm"
    assert lightgbm.model.fit_parameters["early_stopping_rounds"] == 50
    assert lightgbm.evaluation.latency_measured_calls == 1_000
    assert (
        logistic.source_sha256
        == hashlib.sha256(Path("configs/metadata_logistic.yaml").read_bytes()).hexdigest()
    )
    with pytest.raises(FrozenInstanceError):
        logistic.name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        logistic.model.parameters["C"] = 2.0  # type: ignore[index]


def test_invalid_config_fails_before_execution(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("config_version: 1\nname: incomplete\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_experiment_config(path)


def test_unsupported_config_version_is_rejected(tmp_path: Path) -> None:
    document = yaml.safe_load(Path("configs/metadata_logistic.yaml").read_text(encoding="utf-8"))
    document["config_version"] = 0
    path = tmp_path / "unsupported-version.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_experiment_config(path)


def test_unknown_config_fields_are_rejected(tmp_path: Path) -> None:
    content = Path("configs/metadata_logistic.yaml").read_text(encoding="utf-8")
    path = tmp_path / "unknown.yaml"
    path.write_text(content + "unexpected: true\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_experiment_config(path)


def test_unknown_nested_config_fields_are_rejected(tmp_path: Path) -> None:
    content = Path("configs/metadata_logistic.yaml").read_text(encoding="utf-8")
    path = tmp_path / "unknown-nested.yaml"
    path.write_text(
        content.replace(
            "  task_id: pneumonia",
            "  task_id: pneumonia\n  unexpected_field: true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_experiment_config(path)


@pytest.mark.parametrize("bundle_id", ["../escape", "a/b", "a\\b", ".", "..", "/absolute"])
def test_bundle_id_must_be_one_safe_path_component(tmp_path: Path, bundle_id: str) -> None:
    document = yaml.safe_load(Path("configs/metadata_logistic.yaml").read_text(encoding="utf-8"))
    document["dataset"]["bundle_id"] = bundle_id
    path = tmp_path / "unsafe-bundle.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_experiment_config(path)


def test_dataset_registry_key_must_be_one_safe_path_component(tmp_path: Path) -> None:
    content = Path("configs/metadata_logistic.yaml").read_text(encoding="utf-8")
    path = tmp_path / "unsafe-dataset-key.yaml"
    path.write_text(
        content.replace("  registry_key: rsna", "  registry_key: '../escape'", 1),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_experiment_config(path)


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("config_version: 1\nconfig_version: 1\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_experiment_config(path)


@pytest.mark.parametrize("seed", [-1, 2**31])
def test_training_seed_must_be_supported_by_registered_estimators(
    tmp_path: Path, seed: int
) -> None:
    content = Path("configs/metadata_logistic.yaml").read_text(encoding="utf-8")
    path = tmp_path / "invalid-seed.yaml"
    path.write_text(content.replace("  seed: 42", f"  seed: {seed}"), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_experiment_config(path)


@pytest.mark.parametrize(
    "seed_key",
    [
        "random_state",
        "seed",
        "bagging_seed",
        "feature_fraction_seed",
        "data_random_seed",
        "drop_seed",
        "extra_seed",
    ],
)
def test_model_randomness_controls_are_rejected_in_yaml(tmp_path: Path, seed_key: str) -> None:
    content = Path("configs/metadata_logistic.yaml").read_text(encoding="utf-8")
    path = tmp_path / "duplicate-seed-authority.yaml"
    path.write_text(
        content.replace("    max_iter: 2000", f"    max_iter: 2000\n    {seed_key}: 7"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_experiment_config(path)


def test_exact_six_symile_development_configs_are_strict() -> None:
    paths = tuple(Path("configs") / f"symile_{family}.yaml" for family in SYMILE_M5_FAMILIES)
    configs = tuple(load_symile_development_config(path) for path in paths)

    assert tuple(config.family for config in configs) == SYMILE_M5_FAMILIES
    assert all(config.dataset.task_id == "pneumonia_strict" for config in configs)
    assert len({symile_development_semantic_sha256(config) for config in configs}) == 6


def test_symile_gated_configs_differ_scientifically_only_by_observedness() -> None:
    gated = yaml.safe_load(Path("configs/symile_gated.yaml").read_text(encoding="utf-8"))
    ablation = yaml.safe_load(
        Path("configs/symile_gated_no_observedness.yaml").read_text(encoding="utf-8")
    )
    del gated["name"], gated["family"]
    del ablation["name"], ablation["family"]
    observedness = gated["model"].pop("use_observedness")
    ablated = ablation["model"].pop("use_observedness")

    assert observedness is True
    assert ablated is False
    assert gated == ablation


def test_symile_config_rejects_scientific_mutation(tmp_path: Path) -> None:
    document = yaml.safe_load(Path("configs/symile_cxr.yaml").read_text(encoding="utf-8"))
    document["image"]["warmup_epochs"] = 3
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_symile_development_config(path)
