from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
import skops.io as sio

from radfusion.models.tabular_baseline import MetadataLightgbmModel, MetadataLogisticModel
from radfusion.training.config import load_experiment_config
from radfusion.training.registry import get_model
from radfusion.utils.mlflow_utils import cpu_model, environment_provenance
from radfusion.utils.skops_io import load_skops, save_skops


def _features() -> tuple[pd.DataFrame, np.ndarray]:
    size = 40
    target = np.asarray([0, 1] * (size // 2), dtype=np.int8)
    return (
        pd.DataFrame(
            {
                "age_years": np.linspace(20.0, 80.0, size),
                "age_is_implausible": [False] * size,
                "sex": ["F", "M"] * (size // 2),
                "view_position": ["PA", "AP"] * (size // 2),
                "pixel_spacing_row_mm": np.linspace(0.14, 0.20, size),
                "pixel_spacing_col_mm": np.linspace(0.14, 0.20, size),
            }
        ),
        target,
    )


def _fit_registered_model(config_path: str, *, seed: int | None = None):
    config = load_experiment_config(config_path)
    features, target = _features()
    fitted = get_model(config.model.registry_key).fit(
        config.model,
        config.training.seed if seed is None else seed,
        features,
        target,
        features,
        target,
    )
    return config, features, target, fitted


def test_fixed_baselines_fit_with_shared_preprocessing() -> None:
    logistic_config, features, _, logistic_fit = _fit_registered_model(
        "configs/metadata_logistic.yaml"
    )
    lightgbm_config, _, _, lightgbm_fit = _fit_registered_model("configs/metadata_lightgbm.yaml")
    logistic = logistic_fit.pipeline
    lightgbm = lightgbm_fit.pipeline
    logistic_parameters = logistic.named_steps["classifier"].get_params()
    lightgbm_parameters = lightgbm.named_steps["classifier"].get_params()

    assert logistic.predict_proba(features).shape == (len(features), 2)
    assert lightgbm.predict_proba(features).shape == (len(features), 2)
    for key, value in logistic_config.model.parameters.items():
        assert logistic_parameters[key] == value
    for key, value in lightgbm_config.model.parameters.items():
        assert lightgbm_parameters[key] == value
    assert logistic_parameters["random_state"] == logistic_config.training.seed
    assert lightgbm_parameters["random_state"] == lightgbm_config.training.seed
    assert logistic_parameters["class_weight"] == "balanced"
    assert lightgbm_parameters["scale_pos_weight"] == 1.0
    assert lightgbm_parameters["metric"] == "None"
    assert lightgbm_parameters["deterministic"] is True
    assert lightgbm_parameters["force_col_wise"] is True
    assert lightgbm_parameters["n_jobs"] == 1
    assert set(lightgbm.named_steps["classifier"].evals_result_) == {"validation"}
    assert set(lightgbm.named_steps["classifier"].evals_result_["validation"]) == {
        "average_precision"
    }
    assert lightgbm_fit.derived_parameters["early_stopping_greater_is_better"] is True
    assert lightgbm.named_steps["classifier"].best_iteration_ > 0
    transformed = lightgbm.named_steps["preprocess"].transform(features)
    classifier = lightgbm.named_steps["classifier"]
    np.testing.assert_array_equal(
        classifier.predict_proba(transformed),
        classifier.predict_proba(transformed, num_iteration=classifier.best_iteration_),
    )


def test_skops_round_trip_preserves_baseline_predictions(tmp_path) -> None:
    for name, config_path in (
        ("logistic", "configs/metadata_logistic.yaml"),
        ("lightgbm", "configs/metadata_lightgbm.yaml"),
    ):
        _, features, _, fitted = _fit_registered_model(config_path)
        model = fitted.pipeline
        path = save_skops(model, tmp_path / f"{name}.skops")
        restored = load_skops(path)
        np.testing.assert_array_equal(
            restored.predict_proba(features), model.predict_proba(features)
        )


class _UnexpectedType:
    pass


def test_skops_loader_rejects_unapproved_types(tmp_path) -> None:
    rejected = tmp_path / "rejected.skops"
    with pytest.raises(ValueError):
        save_skops(_UnexpectedType(), rejected)
    assert not rejected.exists()

    path = tmp_path / "unexpected.skops"
    sio.dump(_UnexpectedType(), path)
    with pytest.raises(ValueError):
        load_skops(path)


def test_environment_provenance_contains_required_runtime_versions() -> None:
    assert set(environment_provenance()) == {
        "environment_python_version",
        "environment_operating_system",
        "environment_cpu_architecture",
        "environment_cpu_model",
        "environment_numpy_version",
        "environment_pyarrow_version",
        "environment_scikit_learn_version",
        "environment_lightgbm_version",
        "environment_mlflow_version",
        "environment_skops_version",
    }


def test_cpu_model_uses_linux_cpuinfo(monkeypatch) -> None:
    monkeypatch.setattr("radfusion.utils.mlflow_utils.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "radfusion.utils.mlflow_utils.Path.read_text",
        lambda *args, **kwargs: "processor: 0\nmodel name: Test CPU 9000\n",
    )

    assert cpu_model() == "Test CPU 9000"


def test_cpu_model_has_safe_unknown_fallback(monkeypatch) -> None:
    monkeypatch.setattr("radfusion.utils.mlflow_utils.platform.system", lambda: "Other")
    monkeypatch.setattr("radfusion.utils.mlflow_utils.platform.processor", lambda: "")

    assert cpu_model() == "unknown"


def test_environment_provenance_distinguishes_cpu_architecture_and_model(monkeypatch) -> None:
    monkeypatch.setattr("radfusion.utils.mlflow_utils.platform.machine", lambda: "test-arch")
    monkeypatch.setattr("radfusion.utils.mlflow_utils.cpu_model", lambda: "test-model")

    provenance = environment_provenance()

    assert provenance["environment_cpu_architecture"] == "test-arch"
    assert provenance["environment_cpu_model"] == "test-model"


def test_logistic_model_rejects_silent_fit_parameters() -> None:
    features, target = _features()
    config = load_experiment_config("configs/metadata_logistic.yaml").model
    invalid = replace(config, fit_parameters=MappingProxyType({"unexpected": True}))

    with pytest.raises(ValueError):
        MetadataLogisticModel().fit(invalid, 42, features, target, features, target)


@pytest.mark.parametrize("key", ["class_weight", "scale_pos_weight", "is_unbalance"])
def test_logistic_model_rejects_weighting_parameter_conflicts(key: str) -> None:
    features, target = _features()
    config = load_experiment_config("configs/metadata_logistic.yaml").model
    invalid = replace(
        config,
        parameters=MappingProxyType({**config.parameters, key: "balanced"}),
    )
    with pytest.raises(ValueError):
        MetadataLogisticModel().fit(invalid, 42, features, target, features, target)


@pytest.mark.parametrize("key", ["class_weight", "scale_pos_weight", "is_unbalance"])
def test_lightgbm_model_rejects_weighting_parameter_conflicts(key: str) -> None:
    features, target = _features()
    config = load_experiment_config("configs/metadata_lightgbm.yaml").model
    invalid = replace(
        config,
        parameters=MappingProxyType({**config.parameters, key: 1}),
    )
    with pytest.raises(ValueError):
        MetadataLightgbmModel().fit(invalid, 42, features, target, features, target)


def test_lightgbm_rejects_degenerate_training_targets() -> None:
    features, _ = _features()
    target = np.zeros(len(features), dtype=np.int8)
    config = load_experiment_config("configs/metadata_lightgbm.yaml").model
    with pytest.raises(ValueError):
        MetadataLightgbmModel().fit(config, 42, features, target, features, target)


@pytest.mark.parametrize(
    "config_path",
    ["configs/metadata_logistic.yaml", "configs/metadata_lightgbm.yaml"],
)
def test_training_seed_sets_estimator_random_state(config_path: str) -> None:
    _, _, _, first = _fit_registered_model(config_path, seed=42)
    _, _, _, second = _fit_registered_model(config_path, seed=7)

    assert first.pipeline.named_steps["classifier"].random_state == 42
    assert second.pipeline.named_steps["classifier"].random_state == 7
