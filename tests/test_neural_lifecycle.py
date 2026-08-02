from __future__ import annotations

import hashlib
import io
import json
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import Dataset

from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.hashing import sha256_file
from radfusion.data.rsna_artifacts import SOURCE_INVENTORY_FILENAME
from radfusion.data.rsna_source import ManifestBuildError
from radfusion.models.cxr_baseline import PretrainedWeightIdentity
from radfusion.training.compare import regenerate_comparison
from radfusion.training.config import (
    ExperimentConfig,
    image_semantic_config_sha256,
    load_experiment_config,
)
from radfusion.training.datasets import (
    SOURCE_AUTHENTICATION_POLICY_VERSION,
    ImageRunData,
    ImageTestData,
    SourceAuthentication,
    _authenticate_source_rows,
    _PinnedBundlePaths,
)
from radfusion.training.device import resolve_device
from radfusion.training.evaluate import evaluate_training_run
from radfusion.training.interfaces import DatasetLineage
from radfusion.training.neural import (
    CLASS_WEIGHT_POLICY_VERSION,
    NeuralTrainingError,
    build_image_loaders,
    candidate_is_improvement,
    deterministic_inference,
    fit_image_model,
    fit_two_stage_binary_model,
    seed_neural_runtime,
    train_one_epoch,
    training_class_weight,
)
from radfusion.training.train_image import train_image_experiment
from radfusion.utils.mlflow_utils import configure_mlflow
from radfusion.utils.model_publication import threshold_contract
from radfusion.utils.neural_publication import (
    CHECKPOINT_FIELDS,
    NEURAL_MODEL_FILENAME,
    checkpoint_document,
    load_neural_checkpoint,
    load_validated_neural_checkpoint,
    neural_model_package_id,
    publish_neural_model_run,
    save_neural_checkpoint,
    strict_load_checkpoint,
    validate_neural_package_metadata,
    validate_published_neural_model,
)
from radfusion.utils.operational_logging import configure_logging


class _TensorDataset(Dataset[dict[str, object]]):
    def __init__(self, targets: list[int]) -> None:
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, object]:
        target = self.targets[index]
        return {
            "image": torch.tensor([float(index % 2), 1.0], dtype=torch.float32),
            "target": torch.tensor(float(target), dtype=torch.float32),
            "sample_id": f"sample-{index}",
            "patient_id": f"patient-{index}",
        }


class _TinyImageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.ReLU())
        self.classifier = nn.Linear(3, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(images)).squeeze(1)

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True


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
            "sample_id": f"sample-{index}",
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


def _runtime():
    return resolve_device("cpu", mixed_precision=True, pin_memory_policy="enabled")


def _image_config():
    config = load_experiment_config("configs/image_densenet_seed42.yaml")
    assert config.image is not None
    return replace(
        config.image,
        batch_size=2,
        num_workers=0,
        warmup_epochs=1,
        fine_tune_epochs=2,
        early_stopping_patience=2,
    )


def test_deterministic_loaders_class_weight_and_two_stage_training() -> None:
    train = _TensorDataset([0, 1, 0, 1])
    validation = _TensorDataset([0, 1, 0, 1])
    first = build_image_loaders(
        train, validation, config=_image_config(), runtime=_runtime(), seed=42
    )
    second = build_image_loaders(
        train, validation, config=_image_config(), runtime=_runtime(), seed=42
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
    fit = fit_image_model(
        model,
        build_image_loaders(train, validation, config=_image_config(), runtime=_runtime(), seed=17),
        config=_image_config(),
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


def test_repeated_tiny_training_is_deterministic() -> None:
    dataset = _TensorDataset([0, 1, 0, 1, 0])
    config = replace(_image_config(), batch_size=3, warmup_epochs=1, fine_tune_epochs=1)
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
            fit_image_model(
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
    assert (
        first.selected_validation_average_precision == second.selected_validation_average_precision
    )
    assert set(first.selected_state_dict) == set(second.selected_state_dict)
    for key in first.selected_state_dict:
        torch.testing.assert_close(first.selected_state_dict[key], second.selected_state_dict[key])


def test_two_stage_core_dispatches_ordered_multiple_inputs() -> None:
    dataset = _MultiInputDataset([0, 1, 0, 1])
    config = replace(_image_config(), warmup_epochs=1, fine_tune_epochs=1)
    loaders = build_image_loaders(dataset, dataset, config=config, runtime=_runtime(), seed=42)
    seed_neural_runtime(42)
    model = _TinyMultiInputModel()
    structured_before = {
        name: value.detach().clone()
        for name, value in model.classifier.structured_projection.state_dict().items()
    }
    fit = fit_two_stage_binary_model(
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
    assert inference.sample_ids == tuple(f"sample-{index}" for index in range(4))
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
        dataset, dataset, config=_image_config(), runtime=_runtime(), seed=17
    )
    second = build_image_loaders(
        dataset, dataset, config=_image_config(), runtime=_runtime(), seed=42
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
        _image_config(),
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
    result = fit_image_model(
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
    warmup_config = replace(_image_config(), fine_tune_epochs=0)
    warmup_model = _TinyImageModel()
    encoder_before = {
        key: value.detach().clone() for key, value in warmup_model.encoder.state_dict().items()
    }
    classifier_before = {
        key: value.detach().clone() for key, value in warmup_model.classifier.state_dict().items()
    }
    warmup_fit = fit_image_model(
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
    config = replace(_image_config(), warmup_epochs=1, fine_tune_epochs=1)
    model = _TinyImageModel()
    fit = fit_image_model(
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
        config=_image_config(),
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
        config=_image_config(),
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
        _image_config(),
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
    warmup_best = fit_image_model(
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
    fine_best = fit_image_model(
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
        config=_image_config(),
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
        config=_image_config(),
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


def test_partition_source_authentication_is_exact_and_deterministic(tmp_path: Path) -> None:
    log_stream = io.StringIO()
    configure_logging("INFO", stream=log_stream)
    config = load_experiment_config("configs/image_densenet_seed42.yaml").dataset
    root = tmp_path / "raw"
    image_directory = root / "images"
    image_directory.mkdir(parents=True)
    rows = []
    frame_rows = []
    for index, partition in enumerate(("train", "validation")):
        path = image_directory / f"{partition}.dcm"
        path.write_bytes(f"dicom-{partition}".encode())
        sample_id = f"rsna:{partition}"
        relative = f"images/{partition}.dcm"
        rows.append(
            {
                "sample_id": sample_id,
                "relative_path": relative,
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        frame_rows.append(
            (sample_id, f"patient-{index}", relative, partition, index),
        )
    inventory_path = tmp_path / SOURCE_INVENTORY_FILENAME
    pq.write_table(pa.Table.from_pylist(rows), inventory_path)
    metadata_path = tmp_path / "rsna_manifest_metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")
    bundle = _PinnedBundlePaths(
        tmp_path / "samples",
        tmp_path / "labels",
        tmp_path / "splits",
        inventory_path,
        metadata_path,
    )
    inventory_hash = "a" * 64
    metadata = {
        "generated_artifact_hashes": {
            SOURCE_INVENTORY_FILENAME: {
                "arrow_ipc_sha256": inventory_hash,
                "file_sha256": sha256_file(inventory_path),
            }
        }
    }
    frame = pd.DataFrame(
        frame_rows,
        columns=("sample_id", "patient_id", "image_path", "split_name", "target"),
    )
    configured = replace(config, dataset_root=root)

    first = _authenticate_source_rows(
        configured,
        bundle,
        metadata,
        frame,
        partitions=("train", "validation"),
    )
    second = _authenticate_source_rows(
        configured,
        bundle,
        metadata,
        frame,
        partitions=("train", "validation"),
    )

    assert first == second
    assert first.policy_version == SOURCE_AUTHENTICATION_POLICY_VERSION
    assert first.file_count == 2
    assert first.success is True
    assert first.source_inventory_arrow_sha256 == inventory_hash
    log_output = log_stream.getvalue()
    assert "event=source_authentication_progress" in log_output
    assert not any(value in log_output for value in frame["sample_id"])
    assert not any(value in log_output for value in frame["patient_id"])
    assert not any(value in log_output for value in frame["image_path"])

    train_path = image_directory / "train.dcm"
    original_bytes = train_path.read_bytes()
    train_path.write_bytes(b"x" * len(original_bytes))
    with pytest.raises(ManifestBuildError):
        _authenticate_source_rows(
            configured,
            bundle,
            metadata,
            frame,
            partitions=("train", "validation"),
        )
    train_path.write_bytes(original_bytes)
    train_path.write_bytes(original_bytes + b"changed-size")
    with pytest.raises(ManifestBuildError):
        _authenticate_source_rows(
            configured,
            bundle,
            metadata,
            frame,
            partitions=("train", "validation"),
        )
    train_path.write_bytes(original_bytes)

    pq.write_table(pa.Table.from_pylist(rows[:1]), inventory_path)
    with pytest.raises(ManifestBuildError):
        _authenticate_source_rows(
            configured,
            bundle,
            metadata,
            frame,
            partitions=("train", "validation"),
        )

    unsafe_rows = [dict(rows[0]), dict(rows[1])]
    unsafe_rows[0]["relative_path"] = "../train.dcm"
    pq.write_table(pa.Table.from_pylist(unsafe_rows), inventory_path)
    with pytest.raises(ManifestBuildError):
        _authenticate_source_rows(
            configured,
            bundle,
            metadata,
            frame,
            partitions=("train", "validation"),
        )

    pq.write_table(pa.Table.from_pylist([rows[0], rows[0], rows[1]]), inventory_path)
    with pytest.raises(ManifestBuildError):
        _authenticate_source_rows(
            configured,
            bundle,
            metadata,
            frame,
            partitions=("train", "validation"),
        )


def _manifest(config_bytes: bytes, checkpoint: dict[str, object]) -> dict[str, object]:
    config_path = Path("configs/image_densenet_seed42.yaml")
    config = load_experiment_config(config_path)
    image = config.image
    assert image is not None
    digest = hashlib.sha256(config_bytes).hexdigest()
    transform_kwargs = {
        "image_size": 224,
        "rotation_degrees": image.rotation_degrees,
        "translation_fraction": image.translation_fraction,
        "brightness_jitter": image.brightness_jitter,
        "contrast_jitter": image.contrast_jitter,
    }
    training_transform = StandardCxrTransform(training=True, **transform_kwargs).contract()
    evaluation_transform = StandardCxrTransform(training=False, **transform_kwargs).contract()
    return {
        "modality": "image",
        "model": "image_densenet",
        "task": "pneumonia",
        "positive_class": 1,
        "bundle_id": config.dataset.bundle_id,
        "bundle_manifest_sha256": "e" * 64,
        "split_assignment_id": "split-test",
        "label_policy_version": "label-v1",
        "source_config_sha256": digest,
        "semantic_config_sha256": image_semantic_config_sha256(config),
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
            "registry_key": "image_densenet",
            "modality": "image",
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
        "metrics_policy": {
            "version": "binary-probability-and-frozen-operating-points-v1",
            "calibration_bins": 15,
            "threshold_policy_version": "validation-frozen-thresholds-v1",
            "sensitivity_target": 0.9,
        },
        "source_authentication": {
            "policy_version": "partition-inventory-sha256-v1",
            "partitions": ["train", "validation"],
            "file_count": 2,
            "source_inventory_arrow_sha256": "4" * 64,
            "source_inventory_file_sha256": "5" * 64,
            "authenticated_rows_sha256": "6" * 64,
            "success": True,
        },
        "runtime_provenance": _runtime().provenance(),
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
    config_bytes = Path("configs/image_densenet_seed42.yaml").read_bytes()
    published = publish_neural_model_run(
        model_root=tmp_path / "models" / "rsna",
        mlflow_run_id="training-run",
        checkpoint_path=checkpoint_path,
        source_config_bytes=config_bytes,
        manifest=_manifest(config_bytes, checkpoint),
    )

    metadata = validate_neural_package_metadata(published.run_directory)
    loaded = load_validated_neural_checkpoint(published.run_directory, metadata)
    document = validate_published_neural_model(published.run_directory)
    assert set(path.name for path in published.run_directory.iterdir()) == {
        NEURAL_MODEL_FILENAME,
        "resolved_config.yaml",
        "model_manifest.json",
    }
    assert set(restored) == CHECKPOINT_FIELDS
    assert loaded["selected_epoch"] == checkpoint["selected_epoch"]
    assert document["model_package_id"] == neural_model_package_id(document)
    assert json.loads(published.manifest_path.read_text())["checkpoint_sha256"] == sha256_file(
        published.model_path
    )
    with pytest.raises(FileExistsError):
        publish_neural_model_run(
            model_root=tmp_path / "models" / "rsna",
            mlflow_run_id="training-run",
            checkpoint_path=checkpoint_path,
            source_config_bytes=config_bytes,
            manifest=_manifest(config_bytes, checkpoint),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document["model_identity"].update({"unknown": True}),
        lambda document: document["training_policy"]["class_weight"].update({"pos_weight": 2.0}),
        lambda document: document["training_transform_contract"]["training_augmentation"].update(
            {"enabled": False}
        ),
        lambda document: document["runtime_provenance"].update({"hostname": "private"}),
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
    config_bytes = Path("configs/image_densenet_seed42.yaml").read_bytes()
    published = publish_neural_model_run(
        model_root=tmp_path / "models" / "rsna",
        mlflow_run_id="training-run",
        checkpoint_path=checkpoint_path,
        source_config_bytes=config_bytes,
        manifest=_manifest(config_bytes, checkpoint),
    )
    document = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    mutation(document)
    document["model_package_id"] = neural_model_package_id(document)
    published.manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_neural_package_metadata(published.run_directory)


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
    identity = {
        **_manifest(Path("configs/image_densenet_seed42.yaml").read_bytes(), checkpoint),
        "model_package_schema_version": 1,
        "training_mlflow_run_id": "run",
        "checkpoint_sha256": "d" * 64,
    }
    baseline = neural_model_package_id(identity)
    mutations = (
        lambda value: value.update({"checkpoint_sha256": "e" * 64}),
        lambda value: value.update({"semantic_config_sha256": "e" * 64}),
        lambda value: value.update({"bundle_id": "build-changed"}),
        lambda value: value.update({"bundle_manifest_sha256": "8" * 64}),
        lambda value: value.update({"split_assignment_id": "split-changed"}),
        lambda value: value.update({"task": "changed-task"}),
        lambda value: value.update({"label_policy_version": "changed-label-policy"}),
        lambda value: value["training_policy"].update({"seed": 17}),
        lambda value: value["training_policy"].update({"class_weight": {"pos_weight": 2.0}}),
        lambda value: value["model_identity"].update({"encoder": "changed"}),
        lambda value: value["model_identity"].update({"image_size": 448}),
        lambda value: value["model_identity"].update({"embedding_dimension": 512}),
        lambda value: value["model_identity"]["pretrained_weight"].update({"sha256": "8" * 64}),
        lambda value: value["training_transform_contract"].update({"rotation": 1.0}),
        lambda value: value["evaluation_transform_contract"].update({"image_size": 448}),
        lambda value: value["selection"].update({"selected_epoch": 2}),
        lambda value: value["selection"].update({"validation_average_precision": 0.7}),
        lambda value: value["thresholds"].update({"youden_j": 0.51}),
        lambda value: value["training_policy"]["warmup"].update({"epochs": 3}),
        lambda value: value["training_policy"]["fine_tuning"].update({"maximum_epochs": 29}),
        lambda value: value["training_policy"]["scheduler"].update({"factor": 0.4}),
        lambda value: value["training_policy"]["early_stopping"].update({"patience": 6}),
        lambda value: value["metrics_policy"].update({"calibration_bins": 10}),
        lambda value: value["source_authentication"].update(
            {"authenticated_rows_sha256": "7" * 64}
        ),
    )
    for mutation in mutations:
        changed = json.loads(json.dumps(identity))
        mutation(changed)
        assert neural_model_package_id(changed) != baseline
    with_runtime_change = json.loads(json.dumps(identity))
    with_runtime_change["runtime_provenance"].update(
        {
            "resolved_device": "cuda",
            "cuda_runtime_version": "12.4",
            "cudnn_version": 9100,
            "gpu_device_name": "different GPU",
            "gpu_device_index": 3,
            "gpu_compute_capability": [9, 0],
        }
    )
    assert neural_model_package_id(with_runtime_change) == baseline
    with_source_path_change = json.loads(json.dumps(identity))
    with_source_path_change["source_config_sha256"] = "e" * 64
    assert neural_model_package_id(with_source_path_change) == baseline


def test_source_authentication_failure_precedes_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = yaml.safe_load(
        Path("configs/image_densenet_seed42.yaml").read_text(encoding="utf-8")
    )
    document["dataset"]["dataset_root"] = str(tmp_path / "raw")
    document["training"]["model_directory"] = str(tmp_path / "models" / "rsna")
    document["training"]["report_directory"] = str(tmp_path / "reports")
    config_path = tmp_path / "image.yaml"
    config_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(config_path)
    model_requested = []

    class FailingAdapter:
        def load_image_train_validation(self, dataset_config):
            del dataset_config
            raise ManifestBuildError("source authentication failed")

    monkeypatch.setattr("radfusion.training.train_image.get_dataset", lambda key: FailingAdapter())
    monkeypatch.setattr(
        "radfusion.training.train_image.get_model", lambda key: model_requested.append(key)
    )
    monkeypatch.setattr(
        "radfusion.training.train_image.git_revision", lambda: ("commit-test", False)
    )
    monkeypatch.setattr("radfusion.training.train_image.uv_lock_sha256", lambda: "9" * 64)
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    with pytest.raises(ManifestBuildError):
        train_image_experiment(config, tracking_uri=tracking_uri)

    assert model_requested == []
    client = configure_mlflow(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
    assert experiment is not None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
    assert runs[0].data.tags["run_complete"] == "false"


@dataclass
class _SyntheticImageLifecycle:
    config: ExperimentConfig
    adapter: Any
    weight: PretrainedWeightIdentity
    tracking_uri: str
    build_calls: list[str]
    construction_events: list[str]
    seed_calls: list[int]


def _synthetic_image_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _SyntheticImageLifecycle:
    document = yaml.safe_load(
        Path("configs/image_densenet_seed42.yaml").read_text(encoding="utf-8")
    )
    document["dataset"].update(
        {
            "bundle_id": "build-synthetic",
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
    config_path = tmp_path / "image.yaml"
    config_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(config_path)
    lineage = DatasetLineage(
        bundle_id="build-synthetic",
        split_assignment_id="split-synthetic",
        label_policy_version="label-v1",
        task_id="pneumonia",
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

    def authentication(partitions: tuple[str, ...]) -> SourceAuthentication:
        return SourceAuthentication(
            policy_version=SOURCE_AUTHENTICATION_POLICY_VERSION,
            partitions=partitions,
            file_count=4 if partitions == ("test",) else 8,
            source_inventory_arrow_sha256="a" * 64,
            source_inventory_file_sha256="b" * 64,
            authenticated_rows_sha256=("c" if partitions == ("test",) else "d") * 64,
        )

    class Adapter:
        test_calls = 0

        def load_image_train_validation(self, dataset_config):
            assert dataset_config.bundle_id == "build-synthetic"
            return ImageRunData(
                train=frame("train"),
                validation=frame("validation"),
                lineage=lineage,
                bundle_manifest_sha256="e" * 64,
                authentication=authentication(("train", "validation")),
            )

        def load_image_test(self, dataset_config, *, expected_manifest_sha256):
            assert expected_manifest_sha256 == "e" * 64
            self.test_calls += 1
            return ImageTestData(
                test=frame("test"),
                lineage=lineage,
                bundle_manifest_sha256="e" * 64,
                authentication=authentication(("test",)),
            )

    build_calls = []
    construction_events = []

    class Builder:
        def build(self, model_config):
            assert model_config.modality == "image"
            build_calls.append(model_config.registry_key)
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
        return _TensorDataset(frame_value["target"].astype(int).tolist())

    for module in ("radfusion.training.train_image", "radfusion.training.evaluate_image"):
        monkeypatch.setattr(f"{module}.get_dataset", lambda key: adapter)
        monkeypatch.setattr(f"{module}.get_model", lambda key: Builder())
        monkeypatch.setattr(f"{module}.RsnaImageDataset", synthetic_dataset)
        monkeypatch.setattr(f"{module}.git_revision", lambda: ("commit-test", False))
        monkeypatch.setattr(f"{module}.uv_lock_sha256", lambda: "9" * 64)

    def fingerprint(weights):
        assert weights == "densenet121-res224-chex"
        construction_events.append("fingerprint")
        return weight

    monkeypatch.setattr(
        "radfusion.training.train_image.fingerprint_pretrained_weights",
        fingerprint,
    )

    seed_calls = []
    monkeypatch.setattr("radfusion.training.train_image.seed_neural_runtime", seed_calls.append)

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    return _SyntheticImageLifecycle(
        config=config,
        adapter=adapter,
        weight=weight,
        tracking_uri=tracking_uri,
        build_calls=build_calls,
        construction_events=construction_events,
        seed_calls=seed_calls,
    )


def _assert_single_failed_training_without_outputs(setup: _SyntheticImageLifecycle) -> None:
    client = configure_mlflow(tracking_uri=setup.tracking_uri)
    experiment = client.get_experiment_by_name(setup.config.mlflow.experiment_name)
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
    assert not (setup.config.training.model_directory / "runs" / run_id).exists()
    assert not (
        setup.config.training.report_directory / setup.config.dataset.registry_key / "runs" / run_id
    ).exists()


def test_image_training_progress_accepts_unsized_validation_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_image_lifecycle(tmp_path, monkeypatch)

    class UnsizedLoader:
        def __init__(self, loader) -> None:
            self.loader = loader

        def __iter__(self):
            return iter(self.loader)

    def unsized_validation_loader(*args, **kwargs):
        loaders = build_image_loaders(*args, **kwargs)
        return type(loaders)(loaders.train, UnsizedLoader(loaders.validation))

    monkeypatch.setattr(
        "radfusion.training.train_image.build_image_loaders", unsized_validation_loader
    )

    result = train_image_experiment(setup.config, tracking_uri=setup.tracking_uri)

    assert result.model_path.is_file()


def test_synthetic_image_training_package_and_separate_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_image_lifecycle(tmp_path, monkeypatch)
    training = train_image_experiment(setup.config, tracking_uri=setup.tracking_uri)
    assert setup.adapter.test_calls == 0
    assert setup.seed_calls == [setup.config.training.seed]
    assert setup.build_calls == ["image_densenet"]
    assert setup.construction_events == ["fingerprint", "build", "fingerprint"]
    assert training.model_path.name == "model.pt"
    package_manifest = validate_published_neural_model(training.model_path.parent)
    assert package_manifest["model_package_schema_version"] == 1
    assert package_manifest["modality"] == "image"
    assert package_manifest["bundle_manifest_sha256"] == "e" * 64
    recorded_training = configure_mlflow(tracking_uri=setup.tracking_uri).get_run(training.run_id)
    assert recorded_training.data.tags["run_complete"] == "true"
    assert "bundle_manifest_sha256" not in recorded_training.data.tags
    assert recorded_training.data.params["bundle_manifest_sha256"] == "e" * 64

    evaluation = evaluate_training_run(training.run_id, tracking_uri=setup.tracking_uri)
    assert setup.adapter.test_calls == 1
    assert setup.build_calls == ["image_densenet", "image_densenet"]
    assert evaluation.training_run_id == training.run_id
    assert evaluation.run_id != training.run_id
    assert evaluation.artifact_directory.is_dir()
    client = configure_mlflow(tracking_uri=setup.tracking_uri)
    evaluation_run = client.get_run(evaluation.run_id)
    assert evaluation_run.data.tags["run_complete"] == "true"
    assert evaluation_run.data.tags["source_training_run_id"] == training.run_id
    assert evaluation_run.data.tags["model_package_id"] == training.model_package_id
    assert "bundle_manifest_sha256" not in evaluation_run.data.tags
    assert evaluation_run.data.params["bundle_manifest_sha256"] == "e" * 64
    assert evaluation_run.data.params["evaluation_runtime_resolved_device"] == "cpu"

    csv_path, _, rows = regenerate_comparison(
        tracking_uri=setup.tracking_uri,
        output_directory=tmp_path / "comparison",
    )
    comparison = pd.read_csv(csv_path)
    assert rows == 1
    assert comparison["run_id"].tolist() == [evaluation.run_id]
    assert comparison["modality"].tolist() == ["image"]


@pytest.mark.parametrize("changed_field", ["byte_size", "sha256"])
def test_pretrained_weight_mutation_aborts_before_fitting(
    changed_field: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_image_lifecycle(tmp_path, monkeypatch)
    changed = (
        replace(setup.weight, byte_size=setup.weight.byte_size + 1)
        if changed_field == "byte_size"
        else replace(setup.weight, sha256="0" * 64)
    )
    observed = iter((setup.weight, changed))
    monkeypatch.setattr(
        "radfusion.training.train_image.fingerprint_pretrained_weights",
        lambda weights: next(observed),
    )
    monkeypatch.setattr(
        "radfusion.training.train_image.fit_image_model",
        lambda *args, **kwargs: pytest.fail("fitting must not begin"),
    )

    with pytest.raises(RuntimeError):
        train_image_experiment(setup.config, tracking_uri=setup.tracking_uri)


def test_missing_pretrained_weight_prevents_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_image_lifecycle(tmp_path, monkeypatch)

    def missing_weight(weights):
        del weights
        raise FileNotFoundError("must be materialized before formal training")

    monkeypatch.setattr(
        "radfusion.training.train_image.fingerprint_pretrained_weights",
        missing_weight,
    )
    with pytest.raises(FileNotFoundError):
        train_image_experiment(setup.config, tracking_uri=setup.tracking_uri)
    assert setup.build_calls == []
    _assert_single_failed_training_without_outputs(setup)


def test_pretrained_weight_mutation_cleans_outputs_and_leaves_run_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_image_lifecycle(tmp_path, monkeypatch)
    observed = iter((setup.weight, replace(setup.weight, sha256="0" * 64)))
    monkeypatch.setattr(
        "radfusion.training.train_image.fingerprint_pretrained_weights",
        lambda weights: next(observed),
    )

    with pytest.raises(RuntimeError):
        train_image_experiment(setup.config, tracking_uri=setup.tracking_uri)
    _assert_single_failed_training_without_outputs(setup)


def test_image_evaluation_rejects_package_and_source_lineage_before_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_image_lifecycle(tmp_path, monkeypatch)
    training = train_image_experiment(setup.config, tracking_uri=setup.tracking_uri)
    manifest_path = training.model_path.parent / "model_manifest.json"
    original_manifest = manifest_path.read_bytes()
    tampered_manifest = json.loads(original_manifest)
    tampered_manifest["model_package_id"] = "model-package-" + "0" * 64
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_training_run(training.run_id, tracking_uri=setup.tracking_uri)
    assert setup.adapter.test_calls == 0
    manifest_path.write_bytes(original_manifest)

    config_archive = training.model_path.parent / "resolved_config.yaml"
    original_config = config_archive.read_bytes()
    config_archive.write_bytes(original_config + b"\n# tampered\n")
    with pytest.raises(ValueError):
        evaluate_training_run(training.run_id, tracking_uri=setup.tracking_uri)
    assert setup.adapter.test_calls == 0
    config_archive.write_bytes(original_config)

    original_checkpoint = training.model_path.read_bytes()
    training.model_path.write_bytes(original_checkpoint + b"tampered")
    with pytest.raises(ValueError):
        evaluate_training_run(training.run_id, tracking_uri=setup.tracking_uri)
    assert setup.adapter.test_calls == 0
    training.model_path.write_bytes(original_checkpoint)

    monkeypatch.setattr("radfusion.training.evaluate_image.git_revision", lambda: ("other", False))
    with pytest.raises(ValueError):
        evaluate_training_run(training.run_id, tracking_uri=setup.tracking_uri)
    assert setup.adapter.test_calls == 0
    monkeypatch.setattr(
        "radfusion.training.evaluate_image.git_revision", lambda: ("commit-test", False)
    )

    monkeypatch.setattr("radfusion.training.evaluate_image.uv_lock_sha256", lambda: "8" * 64)
    with pytest.raises(ValueError):
        evaluate_training_run(training.run_id, tracking_uri=setup.tracking_uri)
    assert setup.adapter.test_calls == 0
    monkeypatch.setattr("radfusion.training.evaluate_image.uv_lock_sha256", lambda: "9" * 64)

    client = configure_mlflow(tracking_uri=setup.tracking_uri)
    source = client.get_run(training.run_id)
    original_ap = source.data.metrics["validation_average_precision"]
    client.log_metric(training.run_id, "validation_average_precision", original_ap + 0.01)
    with pytest.raises(ValueError):
        evaluate_training_run(training.run_id, tracking_uri=setup.tracking_uri)
    assert setup.adapter.test_calls == 0
    client.log_metric(training.run_id, "validation_average_precision", original_ap)

    original_size = source.data.metrics["model_size_mib"]
    client.log_metric(training.run_id, "model_size_mib", original_size + 0.01)
    with pytest.raises(ValueError):
        evaluate_training_run(training.run_id, tracking_uri=setup.tracking_uri)
    assert setup.adapter.test_calls == 0


def test_image_publication_failures_remain_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_image_lifecycle(tmp_path, monkeypatch)
    training = train_image_experiment(setup.config, tracking_uri=setup.tracking_uri)

    def fail_publication(*args, **kwargs):
        raise OSError((args, kwargs))

    monkeypatch.setattr("radfusion.training.evaluate_image.publish_directory", fail_publication)
    with pytest.raises(OSError):
        evaluate_training_run(training.run_id, tracking_uri=setup.tracking_uri)
    _, _, rows_after_failure = regenerate_comparison(
        tracking_uri=setup.tracking_uri,
        output_directory=tmp_path / "comparison-after-failure",
    )
    assert rows_after_failure == 0
    client = configure_mlflow(tracking_uri=setup.tracking_uri)
    training_run = client.get_run(training.run_id)
    failed_runs = client.search_runs(experiment_ids=[training_run.info.experiment_id])
    assert any(
        run.info.status == "FAILED"
        and run.data.tags.get("run_kind") == "test_evaluation"
        and run.data.tags.get("run_complete") == "false"
        for run in failed_runs
    )

    monkeypatch.setattr("radfusion.training.train_image.write_run_reports", fail_publication)
    with pytest.raises(OSError):
        train_image_experiment(setup.config, tracking_uri=setup.tracking_uri)
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
        setup.config.training.model_directory / "runs" / failed_training.info.run_id
    ).exists()
    assert not (
        setup.config.training.report_directory
        / setup.config.dataset.registry_key
        / "runs"
        / failed_training.info.run_id
    ).exists()


def test_image_rollback_failure_path_has_no_final_publication_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _synthetic_image_lifecycle(tmp_path, monkeypatch)
    log_stream = io.StringIO()
    configure_logging("INFO", stream=log_stream)
    original_set_tags = mlflow.set_tags

    def fail_after_publication(tags):
        if "local_model_path" in tags:
            raise RuntimeError("post-publication metadata failed")
        original_set_tags(tags)

    monkeypatch.setattr("radfusion.training.train_image.mlflow.set_tags", fail_after_publication)

    with pytest.raises(RuntimeError):
        train_image_experiment(setup.config, tracking_uri=setup.tracking_uri)

    _assert_single_failed_training_without_outputs(setup)
    output = log_stream.getvalue()
    assert "event=publication_completed" not in output
    assert "event=run_failed" in output
