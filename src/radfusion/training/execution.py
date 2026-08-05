"""Resolve bounded execution policy for RSNA neural DataLoaders."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class LoaderExecutionPolicy:
    """Operational DataLoader settings excluded from scientific identity."""

    effective_cpu_capacity: float
    num_workers: int
    persistent_workers: bool
    prefetch_factor: int | None
    pin_memory: bool
    candidate_throughput: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.effective_cpu_capacity, bool)
            or not isinstance(self.effective_cpu_capacity, int | float)
            or not math.isfinite(self.effective_cpu_capacity)
            or self.effective_cpu_capacity <= 0
            or isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or not isinstance(self.persistent_workers, bool)
            or not isinstance(self.pin_memory, bool)
            or not isinstance(self.candidate_throughput, tuple)
        ):
            raise ValueError("DataLoader execution policy fields are invalid")
        multiprocessing = self.num_workers > 0
        if self.num_workers < 0 or self.persistent_workers is not multiprocessing:
            raise ValueError("DataLoader worker policy is inconsistent")
        if (
            multiprocessing
            and (
                isinstance(self.prefetch_factor, bool)
                or not isinstance(self.prefetch_factor, int)
                or self.prefetch_factor != 2
            )
        ) or (not multiprocessing and self.prefetch_factor is not None):
            raise ValueError("DataLoader prefetch policy is inconsistent")
        workers: set[int] = set()
        for measurement in self.candidate_throughput:
            if (
                not isinstance(measurement, tuple)
                or len(measurement) != 2
                or isinstance(measurement[0], bool)
                or not isinstance(measurement[0], int)
                or measurement[0] < 0
                or measurement[0] in workers
                or isinstance(measurement[1], bool)
                or not isinstance(measurement[1], int | float)
                or not math.isfinite(measurement[1])
                or measurement[1] <= 0
            ):
                raise ValueError("DataLoader calibration measurements are invalid")
            workers.add(measurement[0])
        if workers and (
            workers != set(worker_candidates(float(self.effective_cpu_capacity)))
            or self.num_workers not in workers
        ):
            raise ValueError("DataLoader calibration candidates are inconsistent")

    def provenance(self) -> dict[str, object]:
        """Return compact execution provenance."""
        return {
            "effective_cpu_capacity": self.effective_cpu_capacity,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "pin_memory": self.pin_memory,
            "candidate_throughput_batches_per_second": {
                str(workers): throughput for workers, throughput in self.candidate_throughput
            },
        }

    @classmethod
    def from_provenance(cls, value: object) -> LoaderExecutionPolicy:
        """Validate and reconstruct serialized loader execution provenance."""
        fields = {
            "effective_cpu_capacity",
            "num_workers",
            "persistent_workers",
            "prefetch_factor",
            "pin_memory",
            "candidate_throughput_batches_per_second",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("DataLoader execution provenance has an unexpected field set")
        rates = value["candidate_throughput_batches_per_second"]
        if not isinstance(rates, Mapping):
            raise ValueError("DataLoader calibration provenance is invalid")
        measurements: list[tuple[int, float]] = []
        for workers, rate in rates.items():
            if not isinstance(workers, str) or not workers.isascii() or not workers.isdecimal():
                raise ValueError("DataLoader calibration worker ID is invalid")
            measurements.append((int(workers), cast(float, rate)))
        try:
            return cls(
                effective_cpu_capacity=cast(float, value["effective_cpu_capacity"]),
                num_workers=cast(int, value["num_workers"]),
                persistent_workers=cast(bool, value["persistent_workers"]),
                prefetch_factor=cast(int | None, value["prefetch_factor"]),
                pin_memory=cast(bool, value["pin_memory"]),
                candidate_throughput=tuple(measurements),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("DataLoader execution provenance is invalid") from exc


def effective_cpu_capacity(
    *,
    affinity_count: int | None = None,
    cpu_max_path: str | Path = "/sys/fs/cgroup/cpu.max",
) -> float:
    """Return the smaller of available CPU count and cgroup-v2 quota."""
    if affinity_count is not None:
        if (
            isinstance(affinity_count, bool)
            or not isinstance(affinity_count, int)
            or affinity_count <= 0
        ):
            raise ValueError("Injected CPU affinity count must be a positive integer")
        affinity = affinity_count
    elif hasattr(os, "sched_getaffinity"):
        affinity = len(os.sched_getaffinity(0))
    else:
        affinity = os.cpu_count()
    if affinity is None:
        raise RuntimeError("Effective CPU capacity must be positive")
    capacity = float(affinity)
    try:
        quota_text, period_text = Path(cpu_max_path).read_text(encoding="utf-8").split()
        if quota_text != "max":
            quota = int(quota_text)
            period = int(period_text)
            if quota <= 0 or period <= 0:
                raise ValueError("invalid cgroup CPU quota")
            capacity = min(capacity, quota / period)
    except FileNotFoundError:
        pass
    if not math.isfinite(capacity) or capacity <= 0:
        raise RuntimeError("Effective CPU capacity must be positive")
    return capacity


def worker_candidates(capacity: float) -> tuple[int, ...]:
    """Return zero plus positive candidates that preserve two CPUs when possible."""
    if isinstance(capacity, bool) or not isinstance(capacity, int | float):
        raise ValueError("CPU capacity must be positive")
    if not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("CPU capacity must be positive")
    safe_maximum = min(32, max(0, math.floor(capacity) - 2))
    candidates = [0]
    workers = 1
    while workers <= safe_maximum:
        candidates.append(workers)
        workers *= 2
    if safe_maximum > 0 and candidates[-1] != safe_maximum:
        candidates.append(safe_maximum)
    return tuple(candidates)


def configured_loader_policy(*, num_workers: int, pin_memory: bool) -> LoaderExecutionPolicy:
    """Return truthful execution provenance for an explicitly configured loader."""
    return LoaderExecutionPolicy(
        effective_cpu_capacity=effective_cpu_capacity(),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=pin_memory,
    )


def smallest_near_best(
    measurements: tuple[tuple[int, float], ...], *, tolerance: float = 0.95
) -> int:
    """Select the smallest candidate within tolerance of best throughput."""
    if not measurements or not 0.0 < tolerance <= 1.0:
        raise ValueError("Worker measurements and tolerance are invalid")
    if any(not math.isfinite(rate) or rate <= 0.0 for _, rate in measurements):
        raise ValueError("Worker throughput measurements must be finite and positive")
    threshold = max(rate for _, rate in measurements) * tolerance
    return min(candidate for candidate, rate in measurements if rate >= threshold)


def calibrate_loader_policy(
    benchmark: Callable[[int], float],
    *,
    pin_memory: bool,
    capacity: float | None = None,
) -> LoaderExecutionPolicy:
    """Benchmark bounded final-loader candidates and return the knee policy."""
    resolved_capacity = effective_cpu_capacity() if capacity is None else capacity
    candidates = worker_candidates(resolved_capacity)
    worker_rates = tuple((workers, benchmark(workers)) for workers in candidates)
    selected = smallest_near_best(worker_rates)
    return LoaderExecutionPolicy(
        effective_cpu_capacity=resolved_capacity,
        num_workers=selected,
        persistent_workers=selected > 0,
        prefetch_factor=2 if selected > 0 else None,
        pin_memory=pin_memory,
        candidate_throughput=worker_rates,
    )
