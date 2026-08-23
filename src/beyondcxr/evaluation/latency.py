"""Benchmark single-sample model inference latency."""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np
import pandas as pd

from beyondcxr.evaluation.probabilities import positive_class_probabilities

LATENCY_WARMUP_CALLS = 100
LATENCY_MEASURED_CALLS = 1_000
LATENCY_SAMPLE_POLICY = "first-sample-in-deterministic-test-order"


class ProbabilityModel(Protocol):
    """Model that returns class probabilities for tabular samples."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return class probabilities."""


def benchmark_single_sample_latency_ms(
    model: ProbabilityModel,
    features: pd.DataFrame,
    *,
    warmup_calls: int = LATENCY_WARMUP_CALLS,
    measured_calls: int = LATENCY_MEASURED_CALLS,
    best_iteration: int | None = None,
) -> float:
    """Return median latency for repeated single-sample probability inference."""
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("Latency benchmarking requires a non-empty pandas DataFrame")
    if warmup_calls < 0:
        raise ValueError("warmup_calls must be nonnegative")
    if measured_calls <= 0:
        raise ValueError("measured_calls must be positive")

    sample = features.iloc[[0]]
    for _ in range(warmup_calls):
        positive_class_probabilities(model, sample, best_iteration=best_iteration)

    durations_ns = np.empty(measured_calls, dtype=np.int64)
    for index in range(measured_calls):
        start_ns = time.perf_counter_ns()
        positive_class_probabilities(model, sample, best_iteration=best_iteration)
        durations_ns[index] = time.perf_counter_ns() - start_ns
    return float(np.median(durations_ns) / 1_000_000.0)
