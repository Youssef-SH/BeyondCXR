from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from beyondcxr.training.device import resolve_device
from beyondcxr.training.neural import (
    dataloader_generator,
    seed_neural_runtime,
)


def test_auto_device_resolution_and_cpu_effective_policies(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime = resolve_device("auto", mixed_precision=True, pin_memory_policy="enabled")

    assert runtime.device.type == "cpu"
    assert runtime.requested_device == "auto"
    assert runtime.resolved_device == "cpu"
    assert runtime.mixed_precision_requested is True
    assert runtime.mixed_precision_effective is False
    assert runtime.pin_memory_requested == "enabled"
    assert runtime.pin_memory_effective is False
    assert runtime.provenance()["torch_version"] == torch.__version__


def test_auto_cuda_resolution_enables_requested_policies(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"GPU-{index}")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 9000)
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    runtime = resolve_device("auto", mixed_precision=True, pin_memory_policy="auto")

    assert runtime.device.type == "cuda"
    assert runtime.mixed_precision_effective is True
    assert runtime.pin_memory_effective is True
    assert runtime.gpu_device_index == 1
    assert runtime.gpu_device_name == "GPU-1"
    assert runtime.gpu_compute_capability == (8, 0)


def test_explicit_unavailable_cuda_fails(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError):
        resolve_device("cuda", mixed_precision=False, pin_memory_policy="disabled")


@pytest.mark.parametrize(
    ("requested_device", "mixed_precision", "pin_memory_policy", "exception"),
    [
        ("mps", False, "disabled", ValueError),
        ("cpu", 1, "disabled", TypeError),
        ("cpu", False, "always", ValueError),
    ],
)
def test_device_resolution_rejects_invalid_policies(
    requested_device: str,
    mixed_precision: object,
    pin_memory_policy: str,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        resolve_device(
            requested_device,
            mixed_precision=mixed_precision,  # type: ignore[arg-type]
            pin_memory_policy=pin_memory_policy,
        )


def test_neural_seeding_reproduces_python_numpy_and_torch_sequences() -> None:
    seed_neural_runtime(42)
    first = (random.random(), np.random.random(), torch.rand(3))
    seed_neural_runtime(42)
    second = (random.random(), np.random.random(), torch.rand(3))

    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])
    assert torch.backends.cudnn.benchmark is False


def test_dataloader_generators_are_deterministic() -> None:
    first = torch.randperm(20, generator=dataloader_generator(17))
    second = torch.randperm(20, generator=dataloader_generator(17))
    different = torch.randperm(20, generator=dataloader_generator(42))

    assert torch.equal(first, second)
    assert not torch.equal(first, different)


@pytest.mark.parametrize("seed", [True, 1.5, -1, 2**31])
def test_neural_seed_contract_rejects_invalid_values(seed: object) -> None:
    with pytest.raises(ValueError):
        seed_neural_runtime(seed)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        dataloader_generator(seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [0, 2**31 - 1])
def test_neural_seed_contract_accepts_boundaries(seed: int, monkeypatch) -> None:
    deterministic_calls: list[tuple[bool, bool]] = []
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled, *, warn_only=False: deterministic_calls.append((enabled, warn_only)),
    )

    seed_neural_runtime(seed)

    assert dataloader_generator(seed).initial_seed() == seed
    assert deterministic_calls == [(True, True)]
