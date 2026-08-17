from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest
import torch
from neural_test_support import TensorDataset as _TensorDataset
from neural_test_support import TinyImageModel as _TinyImageModel
from neural_test_support import build_synchronous_image_loaders as build_image_loaders
from neural_test_support import cpu_runtime as _runtime
from torch import nn
from torch.utils.data import Dataset

from radfusion.training.config import (
    load_experiment_config,
)
from radfusion.training.device import resolve_device
from radfusion.training.execution import LoaderExecutionPolicy
from radfusion.training.neural import (
    NeuralTrainingError,
    build_evaluation_loader,
    candidate_is_improvement,
    configure_neural_determinism,
    deterministic_inference,
    fit_rsna_cxr_model,
    fit_rsna_two_stage_binary_model,
    fit_two_stage_binary_model,
    seed_neural_runtime,
    train_one_epoch,
    training_class_weight,
)


class _MultiInputDataset(Dataset[dict[str, object]]):
    def __init__(self, targets: list[int]) -> None:
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "image": torch.tensor([float(index % 2), 1.0], dtype=torch.float32),
            "structured": torch.tensor([float(index), -float(index)], dtype=torch.float32),
            "target": torch.tensor(float(self.targets[index]), dtype=torch.float32),
            "sample_id": f"rsna:sample-{index}",
            "patient_id": f"patient-{index}",
        }


class _TinyCompositeClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.structured_projection = nn.Linear(2, 2)
        self.output = nn.Linear(4, 1)

    def forward(
        self,
        image_embedding: torch.Tensor,
        structured: torch.Tensor,
    ) -> torch.Tensor:
        structured_embedding = self.structured_projection(structured)
        return self.output(torch.cat((image_embedding, structured_embedding), dim=1)).squeeze(1)


class _TinyMultiInputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.classifier = _TinyCompositeClassifier()
        self.transitions: list[str] = []

    def forward(self, image: torch.Tensor, structured: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(image), structured)

    def freeze_encoder(self) -> None:
        self.transitions.append("freeze")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        self.transitions.append("unfreeze")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True


def _rsna_cxr_neural_config():
    config = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    assert config.neural is not None
    return replace(
        config.neural,
        batch_size=2,
        warmup_epochs=1,
        fine_tune_epochs=2,
        early_stopping_patience=2,
    )


def test_deterministic_loaders_class_weight_and_two_stage_training() -> None:
    train = _TensorDataset([0, 1, 0, 1])
    validation = _TensorDataset([0, 1, 0, 1])
    first = build_image_loaders(
        train, validation, config=_rsna_cxr_neural_config(), runtime=_runtime(), seed=42
    )
    second = build_image_loaders(
        train, validation, config=_rsna_cxr_neural_config(), runtime=_runtime(), seed=42
    )
    first_order = [item for batch in first.train for item in batch["sample_id"]]
    second_order = [item for batch in second.train for item in batch["sample_id"]]

    assert first_order == second_order
    assert len(first_order) == 4
    assert [item for batch in first.validation for item in batch["sample_id"]] == [
        "sample-0",
        "sample-1",
        "sample-2",
        "sample-3",
    ]
    assert training_class_weight(np.array([0, 0, 1])) == (1, 2, 2.0)

    model = _TinyImageModel()
    epochs = []
    epoch_starts = []
    stages = []
    fit = fit_rsna_cxr_model(
        model,
        build_image_loaders(
            train,
            validation,
            config=_rsna_cxr_neural_config(),
            runtime=_runtime(),
            seed=17,
        ),
        config=_rsna_cxr_neural_config(),
        runtime=_runtime(),
        pos_weight=1.0,
        epoch_callback=epochs.append,
        epoch_started_callback=lambda stage, global_epoch, stage_epoch: epoch_starts.append(
            (stage, global_epoch, stage_epoch)
        ),
        stage_callback=lambda stage, count: stages.append((stage, count)),
    )
    assert tuple(epochs) == fit.history
    assert epoch_starts == [
        ("warmup", 1, 1),
        ("fine_tune", 2, 1),
        ("fine_tune", 3, 2),
    ]
    assert stages == [("warmup", 1), ("fine_tune", 2)]
    assert fit.history[0].stage == "warmup"
    assert any(record.stage == "fine_tune" for record in fit.history)
    assert fit.selected_stage in {"warmup", "fine_tune"}
    assert fit.selected_epoch >= 1
    assert all(tensor.device.type == "cpu" for tensor in fit.selected_state_dict.values())


def test_loaders_reject_pin_memory_policy_that_differs_from_runtime() -> None:
    dataset = _TensorDataset([0, 1])
    runtime = _runtime()
    assert runtime.pin_memory_effective is False
    training_execution = LoaderExecutionPolicy("reused", 0, True)
    evaluation_execution = LoaderExecutionPolicy("one_shot", 0, True)

    with pytest.raises(ValueError):
        build_image_loaders(
            dataset,
            dataset,
            config=_rsna_cxr_neural_config(),
            runtime=runtime,
            seed=42,
            execution=training_execution,
        )
    with pytest.raises(ValueError):
        build_evaluation_loader(
            dataset,
            batch_size=_rsna_cxr_neural_config().batch_size,
            runtime=runtime,
            execution=evaluation_execution,
        )


def test_neural_determinism_configures_algorithms_and_cudnn(monkeypatch) -> None:
    calls: list[tuple[bool, bool]] = []
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled, *, warn_only: calls.append((enabled, warn_only)),
    )
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)

    configure_neural_determinism()

    assert calls == [(True, True)]
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_seed_neural_runtime_uses_shared_determinism_configurator(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "radfusion.training.neural.configure_neural_determinism",
        lambda: calls.append("configured"),
    )

    seed_neural_runtime(42)

    assert calls == ["configured"]


def test_repeated_tiny_training_is_deterministic() -> None:
    dataset = _TensorDataset([0, 1, 0, 1, 0])
    config = replace(_rsna_cxr_neural_config(), batch_size=3, warmup_epochs=1, fine_tune_epochs=1)
    results = []
    observations: list[tuple[object, ...]] = []

    def observe(*args: object) -> None:
        observations.append(args)

    for index in range(2):
        seed_neural_runtime(42)
        model = _TinyImageModel()
        loaders = build_image_loaders(dataset, dataset, config=config, runtime=_runtime(), seed=42)
        callbacks = {}
        if index == 1:
            callbacks = {
                "stage_callback": observe,
                "epoch_started_callback": observe,
                "epoch_callback": observe,
                "progress_callback": observe,
            }
        results.append(
            fit_rsna_cxr_model(
                model,
                loaders,
                config=config,
                runtime=_runtime(),
                pos_weight=1.5,
                **callbacks,
            )
        )
    assert observations

    first, second = results
    assert first.history == second.history
    assert first.selected_stage == second.selected_stage
    assert first.selected_epoch == second.selected_epoch
    assert first.selected_validation_metric == second.selected_validation_metric
    assert set(first.selected_state_dict) == set(second.selected_state_dict)
    for key in first.selected_state_dict:
        torch.testing.assert_close(first.selected_state_dict[key], second.selected_state_dict[key])


def test_two_stage_core_dispatches_ordered_multiple_inputs() -> None:
    dataset = _MultiInputDataset([0, 1, 0, 1])
    config = replace(_rsna_cxr_neural_config(), warmup_epochs=1, fine_tune_epochs=1)
    loaders = build_image_loaders(dataset, dataset, config=config, runtime=_runtime(), seed=42)
    seed_neural_runtime(42)
    model = _TinyMultiInputModel()
    structured_before = {
        name: value.detach().clone()
        for name, value in model.classifier.structured_projection.state_dict().items()
    }
    fit = fit_rsna_two_stage_binary_model(
        model,
        loaders.train,
        loaders.validation,
        input_keys=("image", "structured"),
        config=config,
        runtime=_runtime(),
        pos_weight=1.0,
    )
    model.load_state_dict(fit.selected_state_dict, strict=True)
    inference = deterministic_inference(
        model,
        loaders.validation,
        runtime=_runtime(),
        input_keys=("image", "structured"),
    )
    expected_logits = []
    with torch.inference_mode():
        for batch in loaders.validation:
            expected_logits.append(model(batch["image"], batch["structured"]).numpy())

    assert model.transitions == ["freeze", "unfreeze"]
    assert len(fit.history) == 2
    assert any(
        not torch.equal(
            fit.selected_state_dict[f"classifier.structured_projection.{name}"],
            value,
        )
        for name, value in structured_before.items()
    )
    assert inference.sample_ids == tuple(f"rsna:sample-{index}" for index in range(4))
    np.testing.assert_allclose(inference.logits, np.concatenate(expected_logits))
    assert np.isfinite(inference.probabilities).all()


@pytest.mark.parametrize(
    "input_keys",
    [(), ("image", "image"), ("target",), ("",)],
)
def test_multi_input_seam_rejects_invalid_input_keys(input_keys: tuple[str, ...]) -> None:
    batch = {
        "image": torch.ones((2, 2), dtype=torch.float32),
        "target": torch.tensor([0.0, 1.0]),
        "sample_id": ["a", "b"],
        "patient_id": ["p-a", "p-b"],
    }
    with pytest.raises(NeuralTrainingError):
        deterministic_inference(
            _TinyImageModel(),
            [batch],
            runtime=_runtime(),
            input_keys=input_keys,
        )


@pytest.mark.parametrize(
    "structured",
    [
        None,
        [1.0, 2.0],
        torch.ones((1, 2), dtype=torch.float32),
    ],
)
def test_multi_input_seam_rejects_malformed_batches(structured: object) -> None:
    batch = {
        "image": torch.ones((2, 2), dtype=torch.float32),
        "target": torch.tensor([0.0, 1.0]),
        "sample_id": ["a", "b"],
        "patient_id": ["p-a", "p-b"],
    }
    if structured is not None:
        batch["structured"] = structured

    with pytest.raises(NeuralTrainingError):
        deterministic_inference(
            _TinyMultiInputModel(),
            [batch],
            runtime=_runtime(),
            input_keys=("image", "structured"),
        )


def test_different_loader_seeds_change_training_order() -> None:
    dataset = _TensorDataset([0, 1, 0, 1, 0, 1, 0])
    first = build_image_loaders(
        dataset, dataset, config=_rsna_cxr_neural_config(), runtime=_runtime(), seed=17
    )
    second = build_image_loaders(
        dataset, dataset, config=_rsna_cxr_neural_config(), runtime=_runtime(), seed=42
    )

    assert [value for batch in first.train for value in batch["sample_id"]] != [
        value for batch in second.train for value in batch["sample_id"]
    ]


def test_fine_tune_history_records_learning_rate_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import radfusion.training.neural as neural_module

    dataset = _TensorDataset([0, 1, 0, 1])
    config = replace(
        _rsna_cxr_neural_config(),
        warmup_epochs=0,
        fine_tune_epochs=3,
        scheduler_patience=0,
        early_stopping_patience=3,
    )
    monkeypatch.setattr(neural_module, "train_one_epoch", lambda *args, **kwargs: 1.0)
    scores = iter((0.8, 0.7, 0.6))
    monkeypatch.setattr(
        neural_module,
        "deterministic_inference",
        lambda *args, **kwargs: type("Result", (), {"average_precision": next(scores)})(),
    )
    result = fit_rsna_cxr_model(
        _TinyImageModel(),
        build_image_loaders(dataset, dataset, config=config, runtime=_runtime(), seed=42),
        config=config,
        runtime=_runtime(),
        pos_weight=1.0,
    )

    assert result.history[0].encoder_learning_rate == config.encoder_learning_rate
    assert result.history[1].encoder_learning_rate == config.encoder_learning_rate
    assert result.history[2].encoder_learning_rate == pytest.approx(
        config.encoder_learning_rate * config.scheduler_factor
    )


def test_shared_two_stage_lifecycle_supports_auroc_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import radfusion.training.neural as neural_module

    dataset = _TensorDataset([0, 1, 0, 1])
    config = replace(
        _rsna_cxr_neural_config(),
        warmup_epochs=1,
        fine_tune_epochs=2,
        early_stopping_patience=1,
    )
    monkeypatch.setattr(neural_module, "train_one_epoch", lambda *args, **kwargs: 1.0)
    scheduler_metrics: list[float] = []

    class _Scheduler:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self.last_epoch = 0

        def step(self, metric: float) -> None:
            scheduler_metrics.append(metric)
            self.last_epoch += 1

    monkeypatch.setattr(neural_module, "ReduceLROnPlateau", _Scheduler)
    scores = iter(
        (
            ([0.4, 0.1, 0.9, 0.6], 0.9),
            ([0.1, 0.2, 0.8, 0.9], 0.8),
            ([0.4, 0.1, 0.9, 0.6], 0.7),
        )
    )

    def inference(*args: object, **kwargs: object):
        del args, kwargs
        probabilities, average_precision = next(scores)
        return type(
            "Result",
            (),
            {
                "targets": np.array([0, 1, 0, 1], dtype=np.int8),
                "probabilities": np.asarray(probabilities, dtype=np.float64),
                "average_precision": average_precision,
            },
        )()

    monkeypatch.setattr(neural_module, "deterministic_inference", inference)
    loaders = build_image_loaders(dataset, dataset, config=config, runtime=_runtime(), seed=42)
    result = fit_two_stage_binary_model(
        _TinyImageModel(),
        loaders.train,
        loaders.validation,
        input_keys=("image",),
        config=config,
        runtime=_runtime(),
        pos_weight=1.0,
        selection_metric="roc_auc",
        fine_tune_scope="all",
    )

    assert result.selection_metric == "roc_auc"
    assert result.selected_epoch == 2
    assert len(result.history) == 3
    assert result.history[-1].no_improvement_count == 1
    assert scheduler_metrics == [0.75, 0.25]


def test_cpu_and_cuda_runtime_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu = resolve_device("cpu", mixed_precision=True, pin_memory_policy="enabled").provenance()
    gpu_fields = (
        "cuda_runtime_version",
        "cudnn_version",
        "gpu_device_name",
        "gpu_device_index",
        "gpu_compute_capability",
    )
    assert all(cpu[field] is None for field in gpu_fields)

    monkeypatch.setattr("radfusion.training.device.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("radfusion.training.device.torch.cuda.current_device", lambda: 2)
    monkeypatch.setattr(
        "radfusion.training.device.torch.cuda.get_device_name", lambda index: f"GPU-{index}"
    )
    monkeypatch.setattr(
        "radfusion.training.device.torch.cuda.get_device_capability", lambda index: (8, 6)
    )
    monkeypatch.setattr("radfusion.training.device.torch.backends.cudnn.version", lambda: 9100)
    monkeypatch.setattr("radfusion.training.device.torch.version.cuda", "12.4")
    cuda = resolve_device("cuda", mixed_precision=True, pin_memory_policy="auto").provenance()
    assert cuda["cuda_runtime_version"] == "12.4"
    assert cuda["cudnn_version"] == 9100
    assert cuda["gpu_device_name"] == "GPU-2"
    assert cuda["gpu_device_index"] == 2
    assert cuda["gpu_compute_capability"] == [8, 6]


@pytest.mark.parametrize("targets", [[0, 0], [1, 1], [0, 2]])
def test_training_class_weight_requires_both_exact_classes(targets: list[int]) -> None:
    with pytest.raises(NeuralTrainingError):
        training_class_weight(np.asarray(targets))


def test_checkpoint_comparison_requires_strict_minimum_delta() -> None:
    assert candidate_is_improvement(0.5, float("-inf"), 0.01)
    assert not candidate_is_improvement(0.5, 0.5, 0.0)
    assert not candidate_is_improvement(0.505, 0.5, 0.01)
    assert candidate_is_improvement(0.511, 0.5, 0.01)
    with pytest.raises(NeuralTrainingError):
        candidate_is_improvement(0.5, float("-inf"), -0.01)


def test_warmup_preserves_encoder_state_and_fine_tuning_uses_configured_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _TensorDataset([0, 1, 0, 1])
    warmup_config = replace(_rsna_cxr_neural_config(), fine_tune_epochs=0)
    warmup_model = _TinyImageModel()
    encoder_before = {
        key: value.detach().clone() for key, value in warmup_model.encoder.state_dict().items()
    }
    classifier_before = {
        key: value.detach().clone() for key, value in warmup_model.classifier.state_dict().items()
    }
    warmup_fit = fit_rsna_cxr_model(
        warmup_model,
        build_image_loaders(dataset, dataset, config=warmup_config, runtime=_runtime(), seed=17),
        config=warmup_config,
        runtime=_runtime(),
        pos_weight=1.0,
    )

    assert warmup_fit.selected_stage == "warmup"
    assert len(warmup_fit.history) == warmup_config.warmup_epochs
    assert all(
        torch.equal(value, encoder_before[key])
        for key, value in warmup_model.encoder.state_dict().items()
    )
    assert any(
        not torch.equal(value, classifier_before[key])
        for key, value in warmup_model.classifier.state_dict().items()
    )

    import radfusion.training.neural as neural_module

    optimizers = []
    schedulers = []
    clipping_calls = []
    real_adamw = neural_module.AdamW
    real_scheduler = neural_module.ReduceLROnPlateau
    real_clip = torch.nn.utils.clip_grad_norm_

    def recording_adamw(*args, **kwargs):
        optimizer = real_adamw(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    def recording_scheduler(*args, **kwargs):
        scheduler = real_scheduler(*args, **kwargs)
        schedulers.append(scheduler)
        return scheduler

    def recording_clip(*args, **kwargs):
        clipping_calls.append((args, kwargs))
        return real_clip(*args, **kwargs)

    monkeypatch.setattr(neural_module, "AdamW", recording_adamw)
    monkeypatch.setattr(neural_module, "ReduceLROnPlateau", recording_scheduler)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    config = replace(_rsna_cxr_neural_config(), warmup_epochs=1, fine_tune_epochs=1)
    model = _TinyImageModel()
    fit = fit_rsna_cxr_model(
        model,
        build_image_loaders(dataset, dataset, config=config, runtime=_runtime(), seed=42),
        config=config,
        runtime=_runtime(),
        pos_weight=1.0,
    )

    assert len(optimizers) == 2
    assert len(optimizers[0].param_groups) == 1
    assert optimizers[0].param_groups[0]["lr"] == config.warmup_head_learning_rate
    assert optimizers[0].param_groups[0]["weight_decay"] == config.weight_decay
    assert len(optimizers[1].param_groups) == 2
    assert [group["lr"] for group in optimizers[1].param_groups] == [
        config.encoder_learning_rate,
        config.head_learning_rate,
    ]
    assert [group["weight_decay"] for group in optimizers[1].param_groups] == [
        config.weight_decay,
        config.weight_decay,
    ]
    assert len(schedulers) == 1
    assert schedulers[0].factor == config.scheduler_factor
    assert schedulers[0].patience == config.scheduler_patience
    assert schedulers[0].min_lrs == [
        config.scheduler_min_learning_rate,
        config.scheduler_min_learning_rate,
    ]
    assert len(clipping_calls) == len(fit.history) * 2
    assert len(fit.history) == 2


def test_cpu_training_avoids_amp_and_rejects_nonfinite_loss(monkeypatch) -> None:
    import radfusion.training.neural as neural_module

    monkeypatch.setattr(
        neural_module.torch,
        "autocast",
        lambda *args, **kwargs: pytest.fail((args, kwargs, "CPU entered CUDA autocast")),
    )
    model = _TinyImageModel()
    loader = build_image_loaders(
        _TensorDataset([0, 1]),
        _TensorDataset([0, 1]),
        config=_rsna_cxr_neural_config(),
        runtime=_runtime(),
        seed=42,
    ).train
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    class NonfiniteLoss(nn.Module):
        def forward(self, logits, targets):
            del targets
            return logits.sum() * torch.tensor(float("nan"))

    with pytest.raises(NeuralTrainingError):
        train_one_epoch(
            model,
            loader,
            optimizer=optimizer,
            loss_function=NonfiniteLoss(),
            runtime=_runtime(),
            gradient_clip_norm=1.0,
            warmup=False,
        )


def test_epoch_loss_is_sample_weighted_for_partial_final_batch() -> None:
    model = _TinyImageModel()
    config = replace(_rsna_cxr_neural_config(), batch_size=2)
    loader = build_image_loaders(
        _TensorDataset([0, 1, 0, 1, 0]),
        _TensorDataset([0, 1]),
        config=config,
        runtime=_runtime(),
        seed=42,
    ).train

    class BatchSizeLoss(nn.Module):
        def forward(self, logits, targets):
            return logits.sum() * 0.0 + len(targets)

    loss = train_one_epoch(
        model,
        loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_function=BatchSizeLoss(),
        runtime=_runtime(),
        gradient_clip_norm=1.0,
        warmup=True,
    )

    assert loss == pytest.approx((2 * 2 + 2 * 2 + 1 * 1) / 5)


def test_injected_amp_path_unscales_before_clipping(monkeypatch) -> None:
    import radfusion.training.neural as neural_module

    runtime = replace(_runtime(), mixed_precision_effective=True)
    events = []

    class ScaledLoss:
        def __init__(self, loss):
            self.loss = loss

        def backward(self):
            events.append("backward")
            self.loss.backward()

    class RecordingScaler:
        def scale(self, loss):
            events.append("scale")
            return ScaledLoss(loss)

        def unscale_(self, optimizer):
            del optimizer
            events.append("unscale")

        def step(self, optimizer):
            events.append("step")
            optimizer.step()

        def update(self):
            events.append("update")

    monkeypatch.setattr(neural_module.torch, "autocast", lambda **kwargs: nullcontext())
    monkeypatch.setattr(
        neural_module.torch.nn.utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: events.append("clip"),
    )
    model = _TinyImageModel()
    loader = build_image_loaders(
        _TensorDataset([0, 1]),
        _TensorDataset([0, 1]),
        config=_rsna_cxr_neural_config(),
        runtime=runtime,
        seed=42,
    ).train

    train_one_epoch(
        model,
        loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_function=nn.BCEWithLogitsLoss(),
        runtime=runtime,
        gradient_clip_norm=1.0,
        warmup=False,
        scaler=RecordingScaler(),
    )

    assert events == ["scale", "backward", "unscale", "clip", "step", "update"]


def test_fine_tuning_patience_is_exact_and_best_stage_can_vary(monkeypatch) -> None:
    import radfusion.training.neural as neural_module

    dataset = _TensorDataset([0, 1, 0, 1])
    config = replace(
        _rsna_cxr_neural_config(),
        warmup_epochs=1,
        fine_tune_epochs=5,
        early_stopping_patience=2,
        early_stopping_min_delta=0.01,
    )
    monkeypatch.setattr(neural_module, "train_one_epoch", lambda *args, **kwargs: 1.0)
    scores = iter((0.8, 0.805, 0.79))
    monkeypatch.setattr(
        neural_module,
        "deterministic_inference",
        lambda *args, **kwargs: type("Result", (), {"average_precision": next(scores)})(),
    )
    warmup_best = fit_rsna_cxr_model(
        _TinyImageModel(),
        build_image_loaders(dataset, dataset, config=config, runtime=_runtime(), seed=42),
        config=config,
        runtime=_runtime(),
        pos_weight=1.0,
    )
    assert warmup_best.selected_stage == "warmup"
    assert len(warmup_best.history) == 3
    assert [record.no_improvement_count for record in warmup_best.history[1:]] == [1, 2]

    scores = iter((0.5, 0.7, 0.69, 0.68))
    fine_best = fit_rsna_cxr_model(
        _TinyImageModel(),
        build_image_loaders(dataset, dataset, config=config, runtime=_runtime(), seed=42),
        config=config,
        runtime=_runtime(),
        pos_weight=1.0,
    )
    assert fine_best.selected_stage == "fine_tune"
    assert fine_best.selected_epoch == 2


def test_inference_rejects_nonfinite_average_precision(monkeypatch) -> None:
    monkeypatch.setattr(
        "radfusion.training.neural.average_precision_score",
        lambda targets, probabilities: float("nan"),
    )
    loader = build_image_loaders(
        _TensorDataset([0, 1]),
        _TensorDataset([0, 1]),
        config=_rsna_cxr_neural_config(),
        runtime=_runtime(),
        seed=42,
    ).validation

    with pytest.raises(NeuralTrainingError):
        deterministic_inference(_TinyImageModel(), loader, runtime=_runtime())


def test_inference_rejects_non_finite_logits() -> None:
    model = _TinyImageModel()
    with torch.no_grad():
        model.classifier.weight.fill_(float("nan"))
    loader = build_image_loaders(
        _TensorDataset([0, 1]),
        _TensorDataset([0, 1]),
        config=_rsna_cxr_neural_config(),
        runtime=_runtime(),
        seed=42,
    ).validation

    with pytest.raises(NeuralTrainingError):
        deterministic_inference(model, loader, runtime=_runtime())


def test_inference_progress_does_not_require_a_sized_loader() -> None:
    batch = {
        "image": torch.ones((2, 2), dtype=torch.float32),
        "target": torch.tensor([0.0, 1.0]),
        "sample_id": ["a", "b"],
        "patient_id": ["p-a", "p-b"],
    }
    progress: list[tuple[int, int]] = []

    result = deterministic_inference(
        _TinyImageModel(),
        iter([batch]),
        runtime=_runtime(),
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert result.targets.tolist() == [0, 1]
    assert progress == []


@pytest.mark.parametrize(
    "target",
    [
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0.0, float("nan")]),
        torch.tensor([0.0, 2.0]),
        torch.tensor([[0.0], [1.0]]),
    ],
)
def test_lifecycle_rejects_malformed_batch_targets(target: torch.Tensor) -> None:
    batch = {
        "image": torch.ones((2, 2), dtype=torch.float32),
        "target": target,
        "sample_id": ["a", "b"],
        "patient_id": ["p-a", "p-b"],
    }
    with pytest.raises(NeuralTrainingError):
        deterministic_inference(_TinyImageModel(), [batch], runtime=_runtime())


def test_inference_rejects_identifier_length_mismatch() -> None:
    batch = {
        "image": torch.ones((2, 2), dtype=torch.float32),
        "target": torch.tensor([0.0, 1.0]),
        "sample_id": ["only-one"],
        "patient_id": ["p-a", "p-b"],
    }
    with pytest.raises(NeuralTrainingError):
        deterministic_inference(_TinyImageModel(), [batch], runtime=_runtime())
