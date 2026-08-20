"""Provide deterministic neural training and inference primitives."""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Sampler

from radfusion.models.cxr_baseline import (
    set_cxr_encoder_trainability,
    set_cxr_encoder_training_mode,
)
from radfusion.training.config import NeuralConfig
from radfusion.training.device import ResolvedDevice
from radfusion.training.execution import (
    LoaderExecutionPolicy,
    one_shot_loader_policy,
)

CLASS_WEIGHT_POLICY_VERSION = "training-label-prevalence-pos-weight-v1"


class NeuralTrainingError(ValueError):
    """Raised when neural training or inference violates its numeric contract."""


class TwoStageBinaryModel(Protocol):
    """Minimal model surface required by the neural lifecycle."""

    encoder: nn.Module
    classifier: nn.Module

    def train(self, mode: bool = True) -> Any: ...
    def eval(self) -> Any: ...
    def parameters(self) -> Any: ...
    def state_dict(self) -> Any: ...
    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor: ...
    def freeze_encoder(self) -> None: ...
    def unfreeze_encoder(self) -> None: ...


@dataclass(frozen=True)
class EpochThroughput:
    """Operational train/validation timing for one completed epoch."""

    training_elapsed_s: float
    validation_elapsed_s: float
    training_batches_per_second: float
    validation_batches_per_second: float
    training_samples_per_second: float
    validation_samples_per_second: float


@dataclass(frozen=True)
class InferenceResult:
    """Deterministically ordered binary targets, logits, probabilities, and identifiers."""

    targets: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    sample_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    average_precision: float


SelectionMetricName = Literal["average_precision", "roc_auc"]
FineTuneScope = Literal["all", "terminal"]


@dataclass(frozen=True)
class TrainingEpochRecord:
    """Metric-neutral state recorded for one selected-metric training epoch."""

    global_epoch: int
    stage_epoch: int
    stage: str
    training_loss: float
    validation_metric: float
    selected_best: bool
    encoder_learning_rate: float | None
    head_learning_rate: float
    scheduler_last_epoch: int | None
    no_improvement_count: int


@dataclass(frozen=True)
class SelectedTrainingResult:
    """Selected neural state under one explicit validation metric."""

    selected_state_dict: dict[str, torch.Tensor]
    selected_epoch: int
    selected_stage: str
    selection_metric: SelectionMetricName
    selected_validation_metric: float
    history: tuple[TrainingEpochRecord, ...]


@dataclass(frozen=True)
class TerminalTrainingResult:
    """Terminal full-development state with no validation-derived selection."""

    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class ImageLoaders:
    """Deterministic train and evaluation DataLoaders."""

    train: DataLoader[Any]
    validation: DataLoader[Any]


EpochStartedCallback = Callable[[str, int, int], None]
StageCallback = Callable[[str, int], None]
BatchProgressCallback = Callable[[int, int], None]
NeuralProgressCallback = Callable[[str, str, int, int, int], None]
TrainingEpochCallback = Callable[[TrainingEpochRecord], None]
TrainingEpochThroughputCallback = Callable[[TrainingEpochRecord, EpochThroughput], None]


class EpochPermutationSampler(Sampler[tuple[int, int]]):
    """Emit deterministic epoch-tagged requests independent of worker topology."""

    def __init__(self, dataset: Dataset[Any], *, seed: int, epoch: int = 0) -> None:
        _validate_seed(seed)
        if epoch < 0:
            raise ValueError("Sampler epoch must be nonnegative")
        self._dataset = dataset
        self._seed = seed
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self):
        epoch = self._epoch
        payload = f"radfusion-rsna-order\0{self._seed}\0{epoch}".encode()
        permutation_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        generator = torch.Generator().manual_seed(permutation_seed)
        order = torch.randperm(len(self._dataset), generator=generator).tolist()
        self._epoch += 1
        return iter((epoch, index) for index in order)


def configure_neural_determinism() -> None:
    """Establish the deterministic backend state shared by training and inference."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_neural_runtime(seed: int) -> None:
    """Seed Python, NumPy, PyTorch CPU/CUDA, and deterministic kernels."""
    _validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_neural_determinism()


def dataloader_generator(seed: int) -> torch.Generator:
    """Return a deterministically seeded DataLoader shuffle generator."""
    _validate_seed(seed)
    return torch.Generator().manual_seed(seed)


def build_image_loaders(
    train_dataset: Dataset[Any],
    validation_dataset: Dataset[Any],
    *,
    config: NeuralConfig,
    runtime: ResolvedDevice,
    seed: int,
    execution: LoaderExecutionPolicy,
) -> ImageLoaders:
    """Construct deterministic training and validation loaders."""
    policy = execution
    if policy.lifecycle != "reused":
        raise ValueError("Training requires a reused DataLoader execution policy")
    if policy.pin_memory != runtime.pin_memory_effective:
        raise ValueError("DataLoader pin-memory policy differs from the resolved runtime")
    common = {
        "batch_size": config.batch_size,
        "num_workers": policy.num_workers,
        "pin_memory": policy.pin_memory,
        "drop_last": False,
        "persistent_workers": policy.persistent_workers,
    }
    if policy.num_workers > 0:
        common["prefetch_factor"] = policy.prefetch_factor
        common["multiprocessing_context"] = "spawn"
    train_arguments: dict[str, Any] = dict(common)
    train_arguments["generator"] = dataloader_generator(seed)
    if getattr(train_dataset, "epoch_tagged_requests", False):
        train_arguments["sampler"] = EpochPermutationSampler(train_dataset, seed=seed)
    else:
        train_arguments["shuffle"] = True
    return ImageLoaders(
        train=DataLoader(train_dataset, **train_arguments),
        validation=DataLoader(
            validation_dataset,
            shuffle=False,
            generator=dataloader_generator((seed + 1) % (2**31)),
            **common,
        ),
    )


def build_evaluation_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    runtime: ResolvedDevice,
    execution: LoaderExecutionPolicy | None = None,
) -> DataLoader[Any]:
    """Construct one deterministic, ordered image-evaluation loader."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Evaluation batch size must be a positive integer")
    policy = execution or one_shot_loader_policy(pin_memory=runtime.pin_memory_effective)
    if policy.lifecycle != "one_shot":
        raise ValueError("Evaluation requires a one-shot DataLoader execution policy")
    if policy.pin_memory != runtime.pin_memory_effective:
        raise ValueError("DataLoader pin-memory policy differs from the resolved runtime")
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": policy.num_workers,
        "pin_memory": policy.pin_memory,
        "persistent_workers": policy.persistent_workers,
        "generator": dataloader_generator(0),
    }
    if policy.num_workers > 0:
        arguments["prefetch_factor"] = policy.prefetch_factor
        arguments["multiprocessing_context"] = "spawn"
    return DataLoader(**arguments)


def training_class_weight(targets: np.ndarray) -> tuple[int, int, float]:
    """Derive BCE positive-class weight from training labels only."""
    array = np.asarray(targets)
    if array.ndim != 1 or array.size == 0:
        raise NeuralTrainingError("Training targets must be a non-empty one-dimensional array")
    positive = int((array == 1).sum())
    negative = int((array == 0).sum())
    if positive + negative != len(array) or positive == 0 or negative == 0:
        raise NeuralTrainingError("Training targets must contain both binary classes")
    return positive, negative, negative / positive


def candidate_is_improvement(candidate: float, best: float, minimum_delta: float) -> bool:
    """Return whether validation AP is a strict qualifying improvement."""
    if not np.isfinite(minimum_delta) or minimum_delta < 0.0:
        raise NeuralTrainingError("Checkpoint minimum delta must be finite and nonnegative")
    if not all(np.isfinite(value) for value in (candidate, best, minimum_delta)):
        if best == float("-inf") and np.isfinite(candidate):
            return True
        raise NeuralTrainingError("Checkpoint comparison values must be finite")
    return bool(candidate > best + minimum_delta)


def copy_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy one model state into detached finite CPU tensors."""
    copied: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise NeuralTrainingError("Model state dictionary must map strings to tensors")
        tensor = value.detach().cpu().clone()
        if not torch.isfinite(tensor).all():
            raise NeuralTrainingError(f"Model state contains non-finite tensor: {name}")
        copied[name] = tensor
    return copied


def train_one_epoch(
    model: TwoStageBinaryModel,
    loader: DataLoader[Any],
    *,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    runtime: ResolvedDevice,
    gradient_clip_norm: float,
    warmup: bool,
    input_keys: tuple[str, ...] = ("image",),
    scaler: torch.amp.GradScaler | None = None,
    progress_callback: BatchProgressCallback | None = None,
    encoder_trainability: str = "all",
) -> float:
    """Train one epoch and return finite sample-weighted mean loss."""
    model.train()
    set_cxr_encoder_training_mode(
        model.encoder,
        "frozen" if warmup else encoder_trainability,
    )
    batch_losses: list[torch.Tensor] = []
    batch_sizes: list[int] = []
    total_samples = 0
    effective_scaler = scaler if runtime.mixed_precision_effective else None
    total_batches = _progress_total(loader) if progress_callback is not None else None
    for completed_batches, batch in enumerate(loader, start=1):
        inputs, targets = _device_batch(batch, runtime, input_keys=input_keys)
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if runtime.mixed_precision_effective
            else nullcontext()
        )
        with context:
            logits = model(*inputs)
            _require_valid_logits_structure(logits, len(targets))
            loss = loss_function(logits, targets)
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise NeuralTrainingError("Training produced a non-finite batch loss")
        if effective_scaler is not None:
            effective_scaler.scale(loss).backward()
            effective_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            effective_scaler.step(optimizer)
            effective_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        batch_size = len(targets)
        batch_losses.append(loss.detach())
        batch_sizes.append(batch_size)
        total_samples += batch_size
        if total_batches is not None:
            _best_effort_callback(progress_callback, completed_batches, total_batches)
    if total_samples == 0:
        raise NeuralTrainingError("Training DataLoader produced no samples")
    loss_values = torch.stack(batch_losses).cpu().tolist()
    total_loss = sum(value * size for value, size in zip(loss_values, batch_sizes, strict=True))
    mean_loss = total_loss / total_samples
    if not np.isfinite(mean_loss):
        raise NeuralTrainingError("Training produced a non-finite epoch loss")
    return float(mean_loss)


def deterministic_inference(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    runtime: ResolvedDevice,
    input_keys: tuple[str, ...] = ("image",),
    progress_callback: BatchProgressCallback | None = None,
) -> InferenceResult:
    """Run one ordered inference pass and calculate finite validation AP."""
    model.eval()
    targets: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    sample_ids: list[str] = []
    patient_ids: list[str] = []
    with torch.inference_mode():
        total_batches = _progress_total(loader) if progress_callback is not None else None
        for completed_batches, batch in enumerate(loader, start=1):
            inputs, batch_targets = _device_batch(batch, runtime, input_keys=input_keys)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if runtime.mixed_precision_effective
                else nullcontext()
            )
            with context:
                batch_logits = model(*inputs)
            _require_valid_logits_structure(batch_logits, len(batch_targets))
            targets.append(batch_targets.detach())
            logits.append(batch_logits.detach().float())
            sample_ids.extend(str(value) for value in batch["sample_id"])
            patient_ids.extend(str(value) for value in batch["patient_id"])
            if total_batches is not None:
                _best_effort_callback(progress_callback, completed_batches, total_batches)
    if not targets:
        raise NeuralTrainingError("Evaluation DataLoader produced no samples")
    target_array = torch.cat(targets).cpu().numpy().astype(np.int8)
    logit_array = torch.cat(logits).cpu().numpy().astype(np.float64)
    if set(np.unique(target_array).tolist()) != {0, 1}:
        raise NeuralTrainingError("Inference targets must contain both binary classes")
    if not np.isfinite(logit_array).all():
        raise NeuralTrainingError("Inference produced non-finite logits")
    probability_array = torch.sigmoid(torch.from_numpy(logit_array)).numpy()
    if (
        not np.isfinite(probability_array).all()
        or (probability_array < 0.0).any()
        or (probability_array > 1.0).any()
    ):
        raise NeuralTrainingError("Inference produced invalid probabilities")
    lengths = {
        len(sample_ids),
        len(patient_ids),
        len(target_array),
        len(logit_array),
        len(probability_array),
    }
    if len(lengths) != 1:
        raise NeuralTrainingError("Inference identifiers and numeric outputs have unequal lengths")
    average_precision = float(average_precision_score(target_array, probability_array))
    if not np.isfinite(average_precision) or not 0.0 <= average_precision <= 1.0:
        raise NeuralTrainingError("Inference produced invalid Average Precision")
    return InferenceResult(
        targets=target_array,
        logits=logit_array,
        probabilities=probability_array,
        sample_ids=tuple(sample_ids),
        patient_ids=tuple(patient_ids),
        average_precision=average_precision,
    )


def fit_rsna_cxr_model(
    model: nn.Module,
    loaders: ImageLoaders,
    *,
    config: NeuralConfig,
    runtime: ResolvedDevice,
    pos_weight: float,
    epoch_callback: TrainingEpochCallback | None = None,
    epoch_started_callback: EpochStartedCallback | None = None,
    stage_callback: StageCallback | None = None,
    progress_callback: NeuralProgressCallback | None = None,
    throughput_callback: TrainingEpochThroughputCallback | None = None,
) -> SelectedTrainingResult:
    """Run head warm-up and full fine-tuning with validation checkpoint selection."""
    return fit_rsna_two_stage_binary_model(
        model,
        loaders.train,
        loaders.validation,
        input_keys=("image",),
        config=config,
        runtime=runtime,
        pos_weight=pos_weight,
        epoch_callback=epoch_callback,
        epoch_started_callback=epoch_started_callback,
        stage_callback=stage_callback,
        progress_callback=progress_callback,
        throughput_callback=throughput_callback,
    )


def fit_rsna_two_stage_binary_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    *,
    input_keys: tuple[str, ...],
    config: NeuralConfig,
    runtime: ResolvedDevice,
    pos_weight: float,
    epoch_callback: TrainingEpochCallback | None = None,
    epoch_started_callback: EpochStartedCallback | None = None,
    stage_callback: StageCallback | None = None,
    progress_callback: NeuralProgressCallback | None = None,
    throughput_callback: TrainingEpochThroughputCallback | None = None,
) -> SelectedTrainingResult:
    """Run the frozen RSNA AP-selected two-stage binary lifecycle."""
    return fit_two_stage_binary_model(
        model,
        train_loader,
        validation_loader,
        input_keys=input_keys,
        config=config,
        runtime=runtime,
        pos_weight=pos_weight,
        selection_metric="average_precision",
        fine_tune_scope="all",
        epoch_callback=epoch_callback,
        epoch_started_callback=epoch_started_callback,
        stage_callback=stage_callback,
        progress_callback=progress_callback,
        throughput_callback=throughput_callback,
    )


def fit_two_stage_binary_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    *,
    input_keys: tuple[str, ...],
    config: NeuralConfig,
    runtime: ResolvedDevice,
    pos_weight: float,
    selection_metric: SelectionMetricName,
    fine_tune_scope: FineTuneScope,
    epoch_callback: TrainingEpochCallback | None = None,
    epoch_started_callback: EpochStartedCallback | None = None,
    stage_callback: StageCallback | None = None,
    progress_callback: NeuralProgressCallback | None = None,
    throughput_callback: TrainingEpochThroughputCallback | None = None,
) -> SelectedTrainingResult:
    """Run one shared two-stage lifecycle under an explicit selection metric."""
    _validate_input_keys(input_keys)
    if selection_metric not in {"average_precision", "roc_auc"}:
        raise NeuralTrainingError("Neural selection metric is invalid")
    if fine_tune_scope not in {"all", "terminal"}:
        raise NeuralTrainingError("Neural fine-tune scope is invalid")
    neural_model = _validated_neural_model(model)
    encoder = neural_model.encoder
    classifier = neural_model.classifier
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=runtime.device)
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True) if runtime.mixed_precision_effective else None
    )
    best_metric = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    selected_epoch = 0
    selected_stage = ""
    history: list[TrainingEpochRecord] = []
    global_epoch = 0

    neural_model.freeze_encoder()
    _best_effort_callback(stage_callback, "warmup", config.warmup_epochs)
    warmup_optimizer = AdamW(
        classifier.parameters(),
        lr=config.warmup_head_learning_rate,
        weight_decay=config.weight_decay,
    )
    for stage_epoch in range(1, config.warmup_epochs + 1):
        global_epoch += 1
        _best_effort_callback(epoch_started_callback, "warmup", global_epoch, stage_epoch)
        training_started = time.perf_counter()
        loss = train_one_epoch(
            neural_model,
            train_loader,
            optimizer=warmup_optimizer,
            loss_function=loss_function,
            runtime=runtime,
            gradient_clip_norm=config.gradient_clip_norm,
            warmup=True,
            input_keys=input_keys,
            scaler=scaler,
            progress_callback=_operation_callback(
                progress_callback, "training", "warmup", global_epoch
            ),
        )
        training_elapsed = time.perf_counter() - training_started
        validation_started = time.perf_counter()
        validation = deterministic_inference(
            model,
            validation_loader,
            runtime=runtime,
            input_keys=input_keys,
            progress_callback=_operation_callback(
                progress_callback, "validation", "warmup", global_epoch
            ),
        )
        metric = _selected_metric_value(validation, selection_metric)
        validation_elapsed = time.perf_counter() - validation_started
        selected = candidate_is_improvement(
            metric,
            best_metric,
            config.early_stopping_min_delta,
        )
        if selected:
            best_metric = metric
            best_state = copy_state_dict_to_cpu(model)
            selected_epoch = global_epoch
            selected_stage = "warmup"
        record = TrainingEpochRecord(
            global_epoch,
            stage_epoch,
            "warmup",
            loss,
            metric,
            selected,
            None,
            float(warmup_optimizer.param_groups[0]["lr"]),
            None,
            0,
        )
        history.append(record)
        _best_effort_callback(epoch_callback, record)
        _best_effort_callback(
            throughput_callback,
            record,
            EpochThroughput(
                training_elapsed,
                validation_elapsed,
                _batches_per_second(train_loader, training_elapsed),
                _batches_per_second(validation_loader, validation_elapsed),
                _samples_per_second(train_loader, training_elapsed),
                _samples_per_second(validation_loader, validation_elapsed),
            ),
        )

    if fine_tune_scope == "all":
        neural_model.unfreeze_encoder()
    else:
        set_cxr_encoder_trainability(encoder, fine_tune_scope)
    _best_effort_callback(stage_callback, "fine_tune", config.fine_tune_epochs)
    fine_optimizer = AdamW(
        [
            {
                "params": tuple(
                    parameter for parameter in encoder.parameters() if parameter.requires_grad
                ),
                "lr": config.encoder_learning_rate,
                "weight_decay": config.weight_decay,
            },
            {
                "params": classifier.parameters(),
                "lr": config.head_learning_rate,
                "weight_decay": config.weight_decay,
            },
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        fine_optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_learning_rate,
    )
    no_improvement = 0
    for stage_epoch in range(1, config.fine_tune_epochs + 1):
        global_epoch += 1
        _best_effort_callback(epoch_started_callback, "fine_tune", global_epoch, stage_epoch)
        encoder_learning_rate_used = float(fine_optimizer.param_groups[0]["lr"])
        head_learning_rate_used = float(fine_optimizer.param_groups[1]["lr"])
        training_started = time.perf_counter()
        loss = train_one_epoch(
            neural_model,
            train_loader,
            optimizer=fine_optimizer,
            loss_function=loss_function,
            runtime=runtime,
            gradient_clip_norm=config.gradient_clip_norm,
            warmup=False,
            input_keys=input_keys,
            scaler=scaler,
            progress_callback=_operation_callback(
                progress_callback, "training", "fine_tune", global_epoch
            ),
            encoder_trainability=fine_tune_scope,
        )
        training_elapsed = time.perf_counter() - training_started
        validation_started = time.perf_counter()
        validation = deterministic_inference(
            model,
            validation_loader,
            runtime=runtime,
            input_keys=input_keys,
            progress_callback=_operation_callback(
                progress_callback, "validation", "fine_tune", global_epoch
            ),
        )
        metric = _selected_metric_value(validation, selection_metric)
        validation_elapsed = time.perf_counter() - validation_started
        selected = candidate_is_improvement(
            metric,
            best_metric,
            config.early_stopping_min_delta,
        )
        if selected:
            best_metric = metric
            best_state = copy_state_dict_to_cpu(model)
            selected_epoch = global_epoch
            selected_stage = "fine_tune"
            no_improvement = 0
        else:
            no_improvement += 1
        scheduler.step(metric)
        record = TrainingEpochRecord(
            global_epoch,
            stage_epoch,
            "fine_tune",
            loss,
            metric,
            selected,
            encoder_learning_rate_used,
            head_learning_rate_used,
            scheduler.last_epoch,
            no_improvement,
        )
        history.append(record)
        _best_effort_callback(epoch_callback, record)
        _best_effort_callback(
            throughput_callback,
            record,
            EpochThroughput(
                training_elapsed,
                validation_elapsed,
                _batches_per_second(train_loader, training_elapsed),
                _batches_per_second(validation_loader, validation_elapsed),
                _samples_per_second(train_loader, training_elapsed),
                _samples_per_second(validation_loader, validation_elapsed),
            ),
        )
        if not selected and no_improvement >= config.early_stopping_patience:
            break
    if best_state is None or selected_stage not in {"warmup", "fine_tune"}:
        raise NeuralTrainingError("Training did not produce a selected validation checkpoint")
    return SelectedTrainingResult(
        selected_state_dict=best_state,
        selected_epoch=selected_epoch,
        selected_stage=selected_stage,
        selection_metric=selection_metric,
        selected_validation_metric=best_metric,
        history=tuple(history),
    )


def build_terminal_training_loader(
    dataset: Dataset[Any],
    *,
    config: NeuralConfig,
    runtime: ResolvedDevice,
    seed: int,
    execution: LoaderExecutionPolicy,
) -> DataLoader[Any]:
    """Build the one full-development training loader without a validation view."""
    if execution.lifecycle != "reused":
        raise NeuralTrainingError("Terminal training requires reused loader execution")
    if execution.pin_memory != runtime.pin_memory_effective:
        raise NeuralTrainingError("Terminal loader pinning differs from runtime")
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "num_workers": execution.num_workers,
        "pin_memory": execution.pin_memory,
        "drop_last": False,
        "persistent_workers": execution.persistent_workers,
        "generator": dataloader_generator(seed),
    }
    if execution.num_workers > 0:
        arguments["prefetch_factor"] = execution.prefetch_factor
        arguments["multiprocessing_context"] = "spawn"
    if getattr(dataset, "epoch_tagged_requests", False):
        arguments["sampler"] = EpochPermutationSampler(dataset, seed=seed)
    else:
        arguments["shuffle"] = True
    return DataLoader(**arguments)


def fit_terminal_two_stage_binary_model(
    model: TwoStageBinaryModel,
    train_loader: DataLoader[Any],
    *,
    input_keys: tuple[str, ...],
    config: NeuralConfig,
    runtime: ResolvedDevice,
    pos_weight: float,
    stage1_epochs: int,
    stage2_epochs: int,
    fine_tune_scope: FineTuneScope,
) -> TerminalTrainingResult:
    """Run a fixed terminal fit without validation, scheduling, or early stopping."""
    _validate_input_keys(input_keys)
    if (
        isinstance(stage1_epochs, bool)
        or isinstance(stage2_epochs, bool)
        or not isinstance(stage1_epochs, int)
        or not isinstance(stage2_epochs, int)
        or stage1_epochs < 0
        or stage2_epochs < 0
        or stage1_epochs + stage2_epochs <= 0
        or stage1_epochs > config.warmup_epochs
        or stage2_epochs > config.fine_tune_epochs
        or fine_tune_scope not in {"all", "terminal"}
    ):
        raise NeuralTrainingError("Terminal neural fit schedule is invalid")
    neural_model = _validated_neural_model(model)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=runtime.device)
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True) if runtime.mixed_precision_effective else None
    )
    neural_model.freeze_encoder()
    stage1_optimizer = AdamW(
        neural_model.classifier.parameters(),
        lr=config.warmup_head_learning_rate,
        weight_decay=config.weight_decay,
    )
    for _ in range(stage1_epochs):
        train_one_epoch(
            neural_model,
            train_loader,
            optimizer=stage1_optimizer,
            loss_function=loss_function,
            runtime=runtime,
            gradient_clip_norm=config.gradient_clip_norm,
            warmup=True,
            input_keys=input_keys,
            scaler=scaler,
        )
    if stage2_epochs:
        if fine_tune_scope == "all":
            neural_model.unfreeze_encoder()
        else:
            set_cxr_encoder_trainability(neural_model.encoder, fine_tune_scope)
        encoder_parameters = tuple(
            parameter for parameter in neural_model.encoder.parameters() if parameter.requires_grad
        )
        if not encoder_parameters:
            raise NeuralTrainingError("Terminal neural fit has no trainable CXR parameters")
        stage2_optimizer = AdamW(
            [
                {
                    "params": encoder_parameters,
                    "lr": config.encoder_learning_rate,
                    "weight_decay": config.weight_decay,
                },
                {
                    "params": neural_model.classifier.parameters(),
                    "lr": config.head_learning_rate,
                    "weight_decay": config.weight_decay,
                },
            ],
            weight_decay=config.weight_decay,
        )
        for _ in range(stage2_epochs):
            train_one_epoch(
                neural_model,
                train_loader,
                optimizer=stage2_optimizer,
                loss_function=loss_function,
                runtime=runtime,
                gradient_clip_norm=config.gradient_clip_norm,
                warmup=False,
                input_keys=input_keys,
                scaler=scaler,
                encoder_trainability=fine_tune_scope,
            )
    return TerminalTrainingResult(copy_state_dict_to_cpu(neural_model))


def _selected_metric_value(inference: InferenceResult, metric: SelectionMetricName) -> float:
    if metric == "average_precision":
        value = inference.average_precision
    else:
        value = float(roc_auc_score(inference.targets, inference.probabilities))
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise NeuralTrainingError("Neural validation selection metric is invalid")
    return value


def _samples_per_second(loader: DataLoader[Any], elapsed_s: float) -> float:
    if elapsed_s <= 0.0:
        return 0.0
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return 0.0
    try:
        return float(len(dataset) / elapsed_s)
    except TypeError:
        return 0.0


def _batches_per_second(loader: DataLoader[Any], elapsed_s: float) -> float:
    if elapsed_s <= 0.0:
        return 0.0
    try:
        return float(len(loader) / elapsed_s)
    except TypeError:
        return 0.0


def _device_batch(
    batch: dict[str, Any],
    runtime: ResolvedDevice,
    *,
    input_keys: tuple[str, ...],
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    _validate_input_keys(input_keys)
    inputs = tuple(batch.get(key) for key in input_keys)
    targets = batch["target"]
    if not all(isinstance(value, torch.Tensor) for value in inputs) or not isinstance(
        targets, torch.Tensor
    ):
        raise NeuralTrainingError(
            "Neural DataLoader batches must contain tensor inputs and targets"
        )
    tensors = cast(tuple[torch.Tensor, ...], inputs)
    if any(value.ndim < 1 for value in tensors):
        raise NeuralTrainingError("Neural inputs must expose a batch dimension")
    batch_size = len(tensors[0])
    if any(len(value) != batch_size for value in tensors):
        raise NeuralTrainingError("Neural inputs must have equal batch dimensions")
    if (
        not targets.is_floating_point()
        or targets.shape != (batch_size,)
        or not torch.isfinite(targets).all()
        or not torch.all((targets == 0) | (targets == 1))
    ):
        raise NeuralTrainingError("Targets must be finite floating binary values shaped [batch]")
    non_blocking = runtime.device.type == "cuda" and runtime.pin_memory_effective
    return (
        tuple(value.to(runtime.device, non_blocking=non_blocking) for value in tensors),
        targets.to(runtime.device, non_blocking=non_blocking),
    )


def _validate_input_keys(input_keys: object) -> None:
    if (
        not isinstance(input_keys, tuple)
        or not input_keys
        or any(not isinstance(key, str) or not key for key in input_keys)
        or len(input_keys) != len(set(input_keys))
        or any(key in {"target", "sample_id", "patient_id"} for key in input_keys)
    ):
        raise NeuralTrainingError("Neural input keys must be distinct non-empty field names")


def _operation_callback(
    callback: NeuralProgressCallback | None,
    operation: str,
    stage: str,
    global_epoch: int,
) -> BatchProgressCallback | None:
    if callback is None:
        return None

    def report(completed: int, total: int) -> None:
        _best_effort_callback(callback, operation, stage, global_epoch, completed, total)

    return report


def _best_effort_callback(callback: Callable[..., None] | None, *args: object) -> None:
    if callback is None:
        return
    try:
        callback(*args)
    except Exception:
        return


def _progress_total(loader: object) -> int | None:
    try:
        total = len(cast(Any, loader))
    except Exception:
        return None
    return total if isinstance(total, int) and not isinstance(total, bool) and total > 0 else None


def _require_valid_logits_structure(logits: object, batch_size: int) -> None:
    if (
        not isinstance(logits, torch.Tensor)
        or logits.shape != (batch_size,)
        or not logits.is_floating_point()
    ):
        raise NeuralTrainingError("Model produced invalid binary logits")


def _validate_seed(seed: object) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise ValueError("Neural seed must be an integer between 0 and 2147483647")


def _validated_neural_model(model: nn.Module) -> TwoStageBinaryModel:
    """Validate the neural lifecycle surface once and return its typed view."""
    if not isinstance(model, nn.Module):
        raise NeuralTrainingError("Neural model does not implement the neural lifecycle contract")
    encoder = getattr(model, "encoder", None)
    classifier = getattr(model, "classifier", None)
    freeze_encoder = getattr(model, "freeze_encoder", None)
    unfreeze_encoder = getattr(model, "unfreeze_encoder", None)
    if not isinstance(encoder, nn.Module) or not isinstance(classifier, nn.Module):
        raise NeuralTrainingError("Neural model must expose encoder and classifier modules")
    if not callable(freeze_encoder) or not callable(unfreeze_encoder):
        raise NeuralTrainingError("Neural model must expose encoder freeze controls")
    return cast(TwoStageBinaryModel, model)
