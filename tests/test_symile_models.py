from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_preprocess import LAB_FEATURE_COLUMNS
from beyondcxr.models.fusion_concat import initialize_fusion_encoder
from beyondcxr.models.symile_ecg_fusion import (
    ECG_ENCODER_PARAMETER_COUNT,
    TRIMODAL_EXCLUDING_CXR_ENCODER_PARAMETER_COUNT,
    SymileEcgEncoder,
    SymileTriModalGatedHead,
    SymileTriModalGatedModel,
    parameter_count,
)
from beyondcxr.models.symile_fusion import (
    SymileConcatFusionModel,
    SymileGatedFusionHead,
    SymileGatedFusionModel,
    gated_fusion_core,
)
from beyondcxr.models.symile_tabular import (
    fit_symile_labs_lightgbm,
    fit_symile_labs_logistic,
    symile_tabular_logits,
)
from beyondcxr.training.config import load_symile_development_config
from beyondcxr.training.neural import seed_neural_runtime
from beyondcxr.training.symile_data import validated_symile_lab_matrix
from beyondcxr.training.symile_ecg_data import SymileEcgStore


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
        training_seed=17,
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
        training_seed=42,
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
            training_seed=17,
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
            training_seed=17,
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
            training_seed=17,
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


def test_ecg_encoder_exact_shape_count_and_seeded_initialization() -> None:
    seed_neural_runtime(17)
    first = SymileEcgEncoder().eval()
    seed_neural_runtime(17)
    second = SymileEcgEncoder().eval()

    assert parameter_count(first) == ECG_ENCODER_PARAMETER_COUNT
    output = first(torch.ones((2, 12, 5000), dtype=torch.float32))
    assert output.shape == (2, 256)
    assert torch.isfinite(output).all()
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name])
    with pytest.raises(ValueError, match="B x 12 x 5000"):
        first(torch.zeros((2, 1, 5000, 12), dtype=torch.float32))
    with pytest.raises(ValueError, match="B x 12 x 5000"):
        first(torch.zeros((2, 12, 5000), dtype=torch.float64))


@pytest.mark.parametrize("value", [0.0, 1.1, -1.1, float("nan")])
def test_ecg_store_enforces_source_value_contract_before_model_access(value: float) -> None:
    store = object.__new__(SymileEcgStore)
    store._arrays = {  # type: ignore[attr-defined]
        "train": np.full((1, 1, 5000, 12), 0.25, dtype=np.float32)
    }

    signal = store.signal("train", 0)

    assert signal.shape == (12, 5000)
    assert signal.dtype == np.float32
    assert np.allclose(signal, 0.25)

    store._arrays = {  # type: ignore[attr-defined]
        "train": np.full((1, 1, 5000, 12), value, dtype=np.float32)
    }
    with pytest.raises(ManifestBuildError, match="frozen contract"):
        store.signal("train", 0)


def test_tri_modal_gate_is_separate_from_core_and_normalizes_three_modalities() -> None:
    config = load_symile_development_config("configs/symile_cxr_labs_ecg_gated.yaml")
    head = SymileTriModalGatedHead(config.family.parameters).eval()
    embedding = torch.ones((2, 1024), dtype=torch.float32)
    labs = torch.cat((torch.full((2, 50), 0.5), torch.ones((2, 50))), dim=1)
    ecg = torch.ones((2, 12, 5000), dtype=torch.float32)

    logits, weights = head.forward_with_gates(embedding, labs, ecg)

    assert logits.shape == (2,)
    assert weights.shape == (2, 3, 256)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones((2, 256)), rtol=0, atol=1e-6)
    assert parameter_count(head) == TRIMODAL_EXCLUDING_CXR_ENCODER_PARAMETER_COUNT
    assert SymileGatedFusionHead is not type(head)
    assert head.gate[0].in_features == 818
    assert head.gate[-1].out_features == 768


def test_tri_modal_gate_appends_exact_lab_observedness_context() -> None:
    config = load_symile_development_config("configs/symile_cxr_labs_ecg_gated.yaml")
    head = SymileTriModalGatedHead(config.family.parameters).eval()
    observedness = (torch.arange(50) % 2).reshape(1, 50).to(torch.float32)
    labs = torch.cat((torch.full((1, 50), 0.25), observedness), dim=1)
    captured: list[torch.Tensor] = []
    hook = head.gate[0].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        head.forward_with_gates(
            torch.ones((1, 1024), dtype=torch.float32),
            labs,
            torch.ones((1, 12, 5000), dtype=torch.float32),
        )
    finally:
        hook.remove()

    assert len(captured) == 1
    torch.testing.assert_close(captured[0][:, -50:], observedness)


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_tri_modal_autocast_forward_backward(device_type: str) -> None:
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    dtype = torch.bfloat16 if device_type == "cpu" else torch.float16
    config = load_symile_development_config("configs/symile_cxr_labs_ecg_gated.yaml")
    model = SymileTriModalGatedModel(_TinyEncoder(), config.family.parameters).to(device_type)
    images = torch.ones(2, 1, 16, 16, device=device_type)
    labs = torch.ones(2, 100, device=device_type)
    ecg = torch.ones(2, 12, 5000, device=device_type)
    # Some CPU oneDNN builds lack reduced-precision convolution backward.
    # Change only backend availability; mkldnn.flags() also changes TF32 policy.
    with pytest.MonkeyPatch.context() as backend:
        if device_type == "cpu":
            backend.setattr(torch.backends.mkldnn, "enabled", False)
        with torch.autocast(device_type, dtype=dtype):
            embedding = model.encoder.encode(images)
            assert embedding.dtype == dtype
            logits, weights = model.classifier.forward_with_gates(embedding, labs, ecg)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, torch.tensor([0.0, 1.0], device=device_type)
            )
        assert logits.shape == (2,)
        assert weights.shape == (2, 3, 256)
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
        torch.testing.assert_close(
            weights.float().sum(1),
            torch.ones(2, 256, device=device_type),
            rtol=0,
            atol=torch.finfo(dtype).eps / 2,
        )
        loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        with torch.no_grad(), torch.autocast(device_type, dtype=dtype):
            assert torch.isfinite(model(images, labs, ecg)).all()
    if device_type == "cuda":
        with pytest.raises(ValueError, match="laboratory input"):
            model.classifier(embedding, labs.cpu(), ecg)
        with pytest.raises(ValueError, match="share the fusion input device"):
            model.classifier(embedding, labs, ecg.cpu())


@pytest.mark.parametrize("invalid", ["dtype", "shape", "labs"])
def test_tri_modal_autocast_preserves_structural_input_rejections(invalid: str) -> None:
    config = load_symile_development_config("configs/symile_cxr_labs_ecg_gated.yaml")
    head = SymileTriModalGatedHead(config.family.parameters)
    embedding = torch.ones(1, 1024, dtype=torch.bfloat16)
    labs = torch.ones(1, 100)
    ecg = torch.ones(1, 12, 5000)
    if invalid == "dtype":
        ecg = ecg.bfloat16()
    elif invalid == "shape":
        ecg = ecg.transpose(1, 2)
    else:
        labs = labs[:, :99]
    with (
        torch.autocast("cpu", dtype=torch.bfloat16),
        pytest.raises(ValueError, match="ECG input|laboratory input"),
    ):
        head(embedding, labs, ecg)


@pytest.mark.parametrize("value", [0.5, 1.0001, -0.1, float("nan"), float("inf")])
def test_transformed_lab_boundary_rejects_invalid_observedness(value: float) -> None:
    labs = np.zeros((1, 100), dtype=np.float32)
    labs[0, -1] = value

    with pytest.raises(ManifestBuildError, match="laboratory matrix"):
        validated_symile_lab_matrix(labs, rows=1)


def test_tri_modal_freeze_changes_only_the_cxr_encoder() -> None:
    config = load_symile_development_config("configs/symile_cxr_labs_ecg_gated.yaml")
    model = SymileTriModalGatedModel(_TinyEncoder(), config.family.parameters)

    model.freeze_encoder()

    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    model.unfreeze_encoder()
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


def test_tri_modal_non_cxr_initialization_is_training_seed_deterministic() -> None:
    config = load_symile_development_config("configs/symile_cxr_labs_ecg_gated.yaml")
    seed_neural_runtime(42)
    first = SymileTriModalGatedModel(_TinyEncoder(), config.family.parameters)
    seed_neural_runtime(42)
    second = SymileTriModalGatedModel(_TinyEncoder(), config.family.parameters)

    for name, value in first.classifier.state_dict().items():
        torch.testing.assert_close(value, second.classifier.state_dict()[name])
