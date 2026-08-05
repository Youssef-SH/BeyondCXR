from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from radfusion.training.config import (
    image_semantic_config_sha256,
    load_experiment_config,
)
from radfusion.training.execution import (
    LoaderExecutionPolicy,
    calibrate_loader_policy,
    configured_loader_policy,
    effective_cpu_capacity,
    smallest_near_best,
    worker_candidates,
)


def test_effective_cpu_capacity_respects_affinity_and_cgroup(tmp_path: Path) -> None:
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("3071999 100000\n", encoding="utf-8")
    assert effective_cpu_capacity(affinity_count=256, cpu_max_path=cpu_max) == pytest.approx(
        30.71999
    )
    cpu_max.write_text("max 100000\n", encoding="utf-8")
    assert effective_cpu_capacity(affinity_count=12, cpu_max_path=cpu_max) == 12.0


def test_effective_cpu_capacity_falls_back_to_os_cpu_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("max 100000\n", encoding="utf-8")
    monkeypatch.delattr(
        "radfusion.training.execution.os.sched_getaffinity",
        raising=False,
    )
    monkeypatch.setattr("radfusion.training.execution.os.cpu_count", lambda: 7)

    assert effective_cpu_capacity(cpu_max_path=cpu_max) == 7.0


def test_bounded_candidates_and_smallest_near_best_selection() -> None:
    candidates = worker_candidates(30.72)
    assert candidates[0] == 0
    assert candidates == tuple(sorted(set(candidates)))
    assert max(candidates) <= 28
    assert max(candidates) > 16
    assert worker_candidates(0.5) == (0,)
    assert worker_candidates(1.0) == (0,)
    assert worker_candidates(1.999) == (0,)
    assert worker_candidates(2.0) == (0,)
    assert worker_candidates(3.0) == (0, 1)
    assert max(worker_candidates(6.0)) <= 4
    assert smallest_near_best(((0, 9.6), (2, 10.0), (4, 10.05))) == 0


def test_calibration_is_bounded_with_fixed_prefetch() -> None:
    measurements = {0: 8.0, 1: 8.5, 2: 9.6, 4: 10.0}
    policy = calibrate_loader_policy(
        lambda workers: measurements[workers],
        pin_memory=True,
        capacity=6.0,
    )
    assert policy.num_workers == 2
    assert policy.prefetch_factor == 2
    assert policy.persistent_workers is True
    assert policy.pin_memory is True
    assert policy.candidate_throughput == ((0, 8.0), (1, 8.5), (2, 9.6), (4, 10.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_workers": True},
        {"num_workers": 1, "prefetch_factor": 2.0},
        {"effective_cpu_capacity": float("nan")},
        {"pin_memory": 1},
        {"candidate_throughput": ((0, 1.0), (0, 2.0))},
        {"candidate_throughput": ((1, float("inf")),)},
        {"num_workers": 1, "candidate_throughput": ((1, 1.0), (2, 2.0))},
        {"num_workers": 2, "candidate_throughput": ((0, 1.0),)},
    ],
)
def test_loader_execution_policy_rejects_invalid_runtime_contract(
    kwargs: dict[str, object],
) -> None:
    arguments = {
        "effective_cpu_capacity": 4.0,
        "num_workers": 0,
        "persistent_workers": False,
        "prefetch_factor": None,
        "pin_memory": False,
        "candidate_throughput": (),
        **kwargs,
    }
    if (
        isinstance(arguments["num_workers"], int)
        and not isinstance(arguments["num_workers"], bool)
        and arguments["num_workers"] > 0
    ):
        arguments.setdefault("persistent_workers", True)
        arguments.setdefault("prefetch_factor", 2)
    with pytest.raises(ValueError):
        LoaderExecutionPolicy(**arguments)  # type: ignore[arg-type]


def test_effective_cpu_capacity_rejects_boolean_affinity() -> None:
    with pytest.raises(ValueError):
        effective_cpu_capacity(affinity_count=True)


@pytest.mark.parametrize(("workers", "persistent", "prefetch"), [(2, True, 2), (0, False, None)])
def test_configured_policy_separates_cpu_capacity_from_worker_count(
    workers: int,
    persistent: bool,
    prefetch: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("radfusion.training.execution.effective_cpu_capacity", lambda: 12.5)

    policy = configured_loader_policy(num_workers=workers, pin_memory=True)

    assert policy.effective_cpu_capacity == 12.5
    assert policy.num_workers == workers
    assert policy.persistent_workers is persistent
    assert policy.prefetch_factor == prefetch


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
