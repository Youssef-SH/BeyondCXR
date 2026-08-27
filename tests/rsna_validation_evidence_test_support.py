"""Small builders for package-bound synthetic RSNA validation evidence."""

from __future__ import annotations

from pathlib import Path

from beyondcxr.training.config import load_experiment_config, with_runtime
from beyondcxr.training.rsna_validation_evidence import write_validation_evidence


def write_synthetic_validation_evidence(
    destination: Path,
    config_source: Path | bytes,
    *,
    seed: int,
    selected_epoch: int = 3,
    selected_stage: str = "fine_tune",
    selected_validation_average_precision: float = 0.75,
    epoch_history=None,
) -> Path:
    if isinstance(config_source, bytes):
        config_path = destination.with_suffix(".yaml")
        config_path.write_bytes(config_source)
    else:
        config_path = config_source
    config = with_runtime(load_experiment_config(config_path), seed=seed)
    history = epoch_history
    if history is None:
        history = (
            synthetic_cxr_epoch_history(
                config,
                selected_epoch=selected_epoch,
                selected_stage=selected_stage,
                selected_validation_average_precision=selected_validation_average_precision,
            )
            if config.family.family_id == "cxr_densenet"
            else ()
        )
    return write_validation_evidence(
        destination,
        config=config,
        sample_ids=tuple(f"rsna:synthetic-validation-{index}" for index in range(6)),
        targets=(0, 1, 0, 1, 0, 1),
        probabilities=(0.1, 0.9, 0.2, 0.8, 0.3, 0.7),
        epoch_history=history,
    )


def synthetic_cxr_epoch_history(
    config,
    *,
    selected_epoch: int,
    selected_stage: str,
    selected_validation_average_precision: float,
) -> tuple[dict[str, object], ...]:
    """Build the smallest complete synthetic CXR history for one selected checkpoint."""
    neural = config.neural
    if neural is None:
        raise ValueError("Synthetic CXR history requires a neural configuration")
    expected_stage = "warmup" if selected_epoch <= neural.warmup_epochs else "fine_tune"
    if selected_stage != expected_stage or selected_validation_average_precision <= 0.0:
        raise ValueError("Synthetic CXR selection is inconsistent with its training lifecycle")
    total_epochs = max(
        neural.warmup_epochs + neural.early_stopping_patience,
        selected_epoch + neural.early_stopping_patience,
    )
    prior_average_precision = max(0.0, selected_validation_average_precision - 0.1)
    best = float("-inf")
    no_improvement = 0
    history = []
    for global_epoch in range(1, total_epochs + 1):
        stage = "warmup" if global_epoch <= neural.warmup_epochs else "fine_tune"
        stage_epoch = global_epoch if stage == "warmup" else global_epoch - neural.warmup_epochs
        average_precision = (
            prior_average_precision
            if global_epoch < selected_epoch
            else selected_validation_average_precision
        )
        selected_best = average_precision > best + neural.early_stopping_min_delta
        if selected_best:
            best = average_precision
            no_improvement = 0
        elif stage == "fine_tune":
            no_improvement += 1
        history.append(
            {
                "global_epoch": global_epoch,
                "stage_epoch": stage_epoch,
                "stage": stage,
                "training_loss": 0.1,
                "validation_average_precision": average_precision,
                "selected_best": selected_best,
                "encoder_learning_rate": None if stage == "warmup" else 0.00001,
                "head_learning_rate": 0.001 if stage == "warmup" else 0.0001,
                "scheduler_last_epoch": None if stage == "warmup" else stage_epoch,
                "no_improvement_count": no_improvement if stage == "fine_tune" else 0,
            }
        )
    return tuple(history)
