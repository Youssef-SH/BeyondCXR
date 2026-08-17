from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from radfusion.training.config import load_experiment_config, with_runtime
from radfusion.training.device import resolve_device
from radfusion.training.execution import (
    LoaderExecutionPolicy,
    one_shot_loader_policy,
    reused_loader_policy,
)
from radfusion.training.neural import build_evaluation_loader, build_image_loaders


@pytest.mark.parametrize("workers", [0, 2])
def test_reused_loader_policy_has_exact_runtime_contract(workers: int) -> None:
    policy = reused_loader_policy(num_workers=workers, pin_memory=True)

    assert policy.lifecycle == "reused"
    assert policy.persistent_workers is (workers > 0)
    assert policy.prefetch_factor == (2 if workers > 0 else None)
    assert LoaderExecutionPolicy.from_provenance(policy.provenance()) == policy


def test_one_shot_loader_policy_is_synchronous() -> None:
    policy = one_shot_loader_policy(pin_memory=True)

    assert policy.lifecycle == "one_shot"
    assert policy.num_workers == 0
    assert policy.persistent_workers is False
    assert policy.prefetch_factor is None
    assert LoaderExecutionPolicy.from_provenance(policy.provenance()) == policy


def test_loader_builders_apply_distinct_lifecycle_topologies() -> None:
    config = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    assert config.neural is not None
    dataset = TensorDataset(torch.arange(4))
    runtime = resolve_device("cpu", mixed_precision=False, pin_memory_policy="disabled")

    reused = build_image_loaders(
        dataset,
        dataset,
        config=config.neural,
        runtime=runtime,
        seed=42,
        execution=reused_loader_policy(
            num_workers=config.runtime.num_workers,
            pin_memory=runtime.pin_memory_effective,
        ),
    )
    one_shot = build_evaluation_loader(
        dataset, batch_size=config.neural.batch_size, runtime=runtime
    )

    assert reused.train.num_workers == config.runtime.num_workers
    assert reused.train.persistent_workers is True
    assert reused.train.prefetch_factor == 2
    assert reused.train.multiprocessing_context.get_start_method() == "spawn"
    assert one_shot.num_workers == 0
    assert one_shot.persistent_workers is False
    assert one_shot.prefetch_factor is None
    assert one_shot.multiprocessing_context is None


@pytest.mark.parametrize(
    "arguments",
    [
        ("invalid", 0, False),
        ("reused", True, False),
        ("reused", -1, False),
        ("reused", 0, 1),
        ("one_shot", 1, False),
    ],
)
def test_loader_execution_policy_rejects_invalid_contract(arguments) -> None:
    with pytest.raises(ValueError):
        LoaderExecutionPolicy(*arguments)


def test_loader_execution_provenance_rejects_redundant_derived_fields() -> None:
    provenance = reused_loader_policy(num_workers=2, pin_memory=False).provenance()
    provenance["persistent_workers"] = False

    with pytest.raises(ValueError):
        LoaderExecutionPolicy.from_provenance(provenance)


def test_execution_knobs_do_not_change_neural_semantic_identity() -> None:
    config = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    assert config.neural is not None
    baseline = config.config_semantic_sha256
    changed_execution = with_runtime(
        config,
        source_root="/different/source",
        model_directory="/different/models",
        report_directory="/different/reports",
        device="cpu",
        seed=2026,
        num_workers=24,
        pin_memory_policy="disabled",
    )
    assert changed_execution.config_semantic_sha256 == baseline
    assert changed_execution.runtime.seed == 2026
    assert changed_execution.runtime.device == "cpu"
    assert changed_execution.runtime.num_workers == 24
    assert changed_execution.runtime.pin_memory_policy == "disabled"
