from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.utils.data import TensorDataset

from radfusion.training.config import image_semantic_config_sha256, load_experiment_config
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
    config = load_experiment_config("configs/image_densenet_seed42.yaml")
    assert config.image is not None
    dataset = TensorDataset(torch.arange(4))
    runtime = resolve_device("cpu", mixed_precision=False, pin_memory_policy="disabled")

    reused = build_image_loaders(
        dataset,
        dataset,
        config=config.image,
        runtime=runtime,
        seed=42,
    )
    one_shot = build_evaluation_loader(dataset, config=config.image, runtime=runtime)

    assert reused.train.num_workers == config.image.num_workers
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
    config = load_experiment_config("configs/image_densenet_seed42.yaml")
    assert config.image is not None
    baseline = image_semantic_config_sha256(config)
    changed_execution = replace(
        config,
        image=replace(config.image, num_workers=24, pin_memory_policy="disabled"),
    )
    assert image_semantic_config_sha256(changed_execution) == baseline
    assert (
        image_semantic_config_sha256(replace(config, image=replace(config.image, batch_size=16)))
        != baseline
    )
    assert (
        image_semantic_config_sha256(
            replace(config, image=replace(config.image, rotation_degrees=8.0))
        )
        != baseline
    )
