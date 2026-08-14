from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn

from radfusion.data.symile_preprocess import LAB_FEATURE_COLUMNS
from radfusion.models.fusion_concat import initialize_fusion_encoder
from radfusion.models.symile_fusion import (
    SymileConcatFusionModel,
    SymileGatedFusionHead,
    SymileGatedFusionModel,
    gated_fusion_core,
)
from radfusion.models.symile_tabular import (
    fit_symile_labs_lightgbm,
    fit_symile_labs_logistic,
    symile_tabular_logits,
)
from radfusion.training.config import load_symile_development_config
from radfusion.training.neural import seed_neural_runtime


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 1024)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(2, 3)))


def _labs(rows: int) -> pd.DataFrame:
    records = []
    for row_index in range(rows):
        record: dict[str, object] = {}
        for column_index, column in enumerate(LAB_FEATURE_COLUMNS[:50]):
            record[column] = float(row_index + column_index)
        for column in LAB_FEATURE_COLUMNS[50:]:
            record[column] = True
        records.append(record)
    return pd.DataFrame(records, columns=LAB_FEATURE_COLUMNS)


def test_symile_logistic_is_exact_unweighted_and_produces_logits() -> None:
    config = load_symile_development_config("configs/symile_labs_logistic.yaml")
    targets = np.tile(np.array([0, 1], dtype=np.int8), 10)
    fit = fit_symile_labs_logistic(
        _labs(20),
        targets,
        parameters=config.training.parameters,
        selection_metric=config.training.selection_metric,
        lab_policy=str(config.preprocessing["lab_policy"]),
        repeat_seed=17,
    )
    classifier = fit.pipeline.named_steps["classifier"]

    assert isinstance(classifier, LogisticRegression)
    assert classifier.get_params()["class_weight"] is None
    assert classifier.get_params()["random_state"] == 17
    assert classifier.get_params()["solver"] == "liblinear"
    assert fit.best_iteration is None
    assert symile_tabular_logits(fit.pipeline, _labs(2)).shape == (2,)


def test_symile_lightgbm_is_exact_unweighted_and_seed_owned() -> None:
    config = load_symile_development_config("configs/symile_labs_lightgbm.yaml")
    targets = np.tile(np.array([0, 1], dtype=np.int8), 20)
    training = np.arange(30, dtype=np.int64)
    validation = np.arange(30, 40, dtype=np.int64)
    fit = fit_symile_labs_lightgbm(
        _labs(40),
        targets,
        parameters={**config.family.parameters, **config.training.parameters},
        selection_metric=config.training.selection_metric,
        lab_policy=str(config.preprocessing["lab_policy"]),
        inner_training_indices=training,
        inner_validation_indices=validation,
        repeat_seed=42,
    )
    classifier = fit.pipeline.named_steps["classifier"]
    parameters = classifier.get_params()

    assert isinstance(classifier, LGBMClassifier)
    assert parameters["class_weight"] is None
    assert parameters["random_state"] == 42
    assert parameters["bagging_seed"] == 42
    assert parameters["feature_fraction_seed"] == 42
    assert parameters["deterministic"] is True
    assert parameters["verbosity"] == -1
    assert fit.best_iteration == classifier.best_iteration_
    assert fit.best_iteration > 0


def test_symile_tabular_builders_consume_canonical_selection_metric() -> None:
    targets = np.tile(np.array([0, 1], dtype=np.int8), 10)
    logistic = load_symile_development_config("configs/symile_labs_logistic.yaml")
    with pytest.raises(ValueError, match="does not perform model selection"):
        fit_symile_labs_logistic(
            _labs(20),
            targets,
            parameters=logistic.training.parameters,
            selection_metric="roc_auc",
            lab_policy=str(logistic.preprocessing["lab_policy"]),
            repeat_seed=17,
        )

    lightgbm = load_symile_development_config("configs/symile_labs_lightgbm.yaml")
    with pytest.raises(ValueError, match="selection requires roc_auc"):
        fit_symile_labs_lightgbm(
            _labs(20),
            targets,
            parameters={**lightgbm.family.parameters, **lightgbm.training.parameters},
            selection_metric="average_precision",
            lab_policy=str(lightgbm.preprocessing["lab_policy"]),
            inner_training_indices=np.arange(16, dtype=np.int64),
            inner_validation_indices=np.arange(16, 20, dtype=np.int64),
            repeat_seed=17,
        )


def test_symile_tabular_builder_consumes_lab_preprocessing_policy() -> None:
    config = load_symile_development_config("configs/symile_labs_logistic.yaml")
    with pytest.raises(ValueError, match="preprocessing policy"):
        fit_symile_labs_logistic(
            _labs(20),
            np.tile(np.array([0, 1], dtype=np.int8), 10),
            parameters=config.training.parameters,
            selection_metric=config.training.selection_metric,
            lab_policy="unsupported-policy",
            repeat_seed=17,
        )


def test_concat_and_gated_shapes_and_per_feature_softmax() -> None:
    concat_config = load_symile_development_config("configs/symile_cxr_labs_concat.yaml")
    gated_config = load_symile_development_config("configs/symile_cxr_labs_gated.yaml")
    images = torch.ones((3, 1, 224, 224), dtype=torch.float32)
    labs = torch.ones((3, 100), dtype=torch.float32)
    concat = SymileConcatFusionModel(_TinyEncoder(), concat_config.family.parameters).eval()
    gated = SymileGatedFusionModel(_TinyEncoder(), gated_config.family.parameters).eval()

    assert concat(images, labs).shape == (3,)
    embedding = gated.encoder.encode(images)
    logits, weights = gated.classifier.forward_with_gates(embedding, labs)
    assert logits.shape == (3,)
    assert weights.shape == (3, 2, 256)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones((3, 256)))


def test_symile_fusion_initializes_only_matching_cxr_encoder_state() -> None:
    gated_config = load_symile_development_config("configs/symile_cxr_labs_gated.yaml")
    concat_config = load_symile_development_config("configs/symile_cxr_labs_concat.yaml")
    source = SymileGatedFusionModel(_TinyEncoder(), gated_config.family.parameters)
    target = SymileConcatFusionModel(_TinyEncoder(), concat_config.family.parameters)
    source_state = {
        f"encoder.{name}": value.detach().clone()
        for name, value in source.encoder.state_dict().items()
    }

    initialize_fusion_encoder(target, source_state)

    for name, value in target.encoder.state_dict().items():
        torch.testing.assert_close(value, source_state[f"encoder.{name}"])
    with pytest.raises(ValueError):
        initialize_fusion_encoder(target, {"encoder.unexpected": torch.ones(1)})


def test_gated_ablation_uses_same_class_parameters_and_initial_non_cxr_state() -> None:
    observed_config = load_symile_development_config("configs/symile_cxr_labs_gated.yaml")
    ablated_config = load_symile_development_config(
        "configs/symile_cxr_labs_gated_no_observedness.yaml"
    )
    seed_neural_runtime(2026)
    observed = SymileGatedFusionModel(_TinyEncoder(), observed_config.family.parameters)
    seed_neural_runtime(2026)
    ablated = SymileGatedFusionModel(_TinyEncoder(), ablated_config.family.parameters)

    assert type(observed) is type(ablated)
    assert sum(parameter.numel() for parameter in observed.parameters()) == sum(
        parameter.numel() for parameter in ablated.parameters()
    )
    observed_state = {
        name: value
        for name, value in observed.state_dict().items()
        if not name.startswith("encoder.")
    }
    ablated_state = {
        name: value
        for name, value in ablated.state_dict().items()
        if not name.startswith("encoder.")
    }
    assert set(observed_state) == set(ablated_state)
    for name, value in observed_state.items():
        torch.testing.assert_close(value, ablated_state[name])


def test_gated_observedness_zeroing_occurs_at_both_consumers() -> None:
    observed_config = load_symile_development_config("configs/symile_cxr_labs_gated.yaml")
    ablated_config = load_symile_development_config(
        "configs/symile_cxr_labs_gated_no_observedness.yaml"
    )
    seed_neural_runtime(17)
    observed = SymileGatedFusionHead(observed_config.family.parameters).eval()
    seed_neural_runtime(17)
    ablated = SymileGatedFusionHead(ablated_config.family.parameters).eval()
    ablated.load_state_dict(observed.state_dict(), strict=True)
    embeddings = torch.ones((2, 1024), dtype=torch.float32)
    labs = torch.cat((torch.full((2, 50), 0.5), torch.tensor([[1.0] * 50, [0.0] * 50])), dim=1)
    zeroed = torch.cat((labs[:, :50], torch.zeros_like(labs[:, 50:])), dim=1)

    ablated_logits, ablated_gates = ablated.forward_with_gates(embeddings, labs)
    expected_logits, expected_gates = observed.forward_with_gates(embeddings, zeroed)

    torch.testing.assert_close(ablated_logits, expected_logits)
    torch.testing.assert_close(ablated_gates, expected_gates)


@pytest.mark.parametrize("modalities", [1, 3, 4])
def test_symile_gated_wrapper_requires_two_modalities(modalities: int) -> None:
    config = load_symile_development_config("configs/symile_cxr_labs_gated.yaml")
    parameters = {**dict(config.family.parameters), "modality_count": modalities}
    with pytest.raises(ValueError):
        SymileGatedFusionHead(parameters)


def test_gated_fusion_core_accepts_two_modalities() -> None:
    modality_count = 2
    representations = tuple(
        torch.arange(6, dtype=torch.float32).reshape(2, 3) + index
        for index in range(modality_count)
    )
    gate_logits = torch.arange(2 * modality_count * 3, dtype=torch.float32).reshape(
        2, modality_count * 3
    )

    fused, weights = gated_fusion_core(representations, gate_logits)

    expected_weights = torch.softmax(gate_logits.reshape(2, modality_count, 3), dim=1)
    expected_fused = torch.sum(expected_weights * torch.stack(representations, dim=1), dim=1)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(fused, expected_fused, rtol=0, atol=0)


@pytest.mark.parametrize("modality_count", [0, 1, 3, 4, 5])
def test_gated_fusion_core_rejects_other_modality_counts(modality_count: int) -> None:
    representations = tuple(torch.ones((2, 3)) for _ in range(modality_count))
    with pytest.raises(ValueError, match="exactly two modalities"):
        gated_fusion_core(representations, torch.ones((2, max(modality_count, 1) * 3)))


def test_gated_fusion_core_rejects_mixed_representation_dtypes() -> None:
    representations = (
        torch.ones((2, 3), dtype=torch.float32),
        torch.ones((2, 3), dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="shared dtype"):
        gated_fusion_core(representations, torch.ones((2, 6), dtype=torch.float32))


def test_gated_fusion_core_rejects_gate_dtype_mismatch() -> None:
    representations = (torch.ones((2, 3), dtype=torch.float32),) * 2
    with pytest.raises(ValueError, match="representation dtype"):
        gated_fusion_core(representations, torch.ones((2, 6), dtype=torch.float64))


def test_gated_fusion_core_rejects_cross_device_inputs_when_cuda_is_available() -> None:
    if not torch.cuda.is_available():
        return
    cpu = torch.ones((2, 3), dtype=torch.float32)
    cuda = cpu.to("cuda")
    with pytest.raises(ValueError, match="one device"):
        gated_fusion_core((cpu, cuda), torch.ones((2, 6), dtype=torch.float32))
    with pytest.raises(ValueError, match="representation device"):
        gated_fusion_core((cpu, cpu), torch.ones((2, 6), dtype=torch.float32, device="cuda"))
