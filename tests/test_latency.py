from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from beyondcxr.evaluation.latency import (
    LATENCY_MEASURED_CALLS,
    LATENCY_WARMUP_CALLS,
    benchmark_single_sample_latency_ms,
)


class _RecordingModel:
    def __init__(self) -> None:
        self.samples: list[str] = []
        self.classes_ = np.asarray([0, 1])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.samples.append(str(features.index[0]))
        return np.asarray([[0.75, 0.25]], dtype=np.float64)


def test_operational_latency_defaults_are_frozen() -> None:
    assert LATENCY_WARMUP_CALLS == 100
    assert LATENCY_MEASURED_CALLS == 1_000


def test_latency_benchmark_uses_warmup_and_median_single_sample_calls(monkeypatch) -> None:
    model = _RecordingModel()
    features = pd.DataFrame({"feature": [1.0, 2.0]}, index=["first", "second"])
    timestamps = iter([0, 1_000_000, 10_000_000, 13_000_000, 20_000_000, 25_000_000])
    monkeypatch.setattr(
        "beyondcxr.evaluation.latency.time.perf_counter_ns", lambda: next(timestamps)
    )

    latency_ms = benchmark_single_sample_latency_ms(
        model, features, warmup_calls=2, measured_calls=3
    )

    assert latency_ms == pytest.approx(3.0)
    assert model.samples == ["first"] * 5


@pytest.mark.parametrize(
    ("warmup_calls", "measured_calls"),
    [(-1, 1), (0, 0)],
)
def test_latency_benchmark_validates_call_counts(warmup_calls: int, measured_calls: int) -> None:
    with pytest.raises(ValueError):
        benchmark_single_sample_latency_ms(
            _RecordingModel(),
            pd.DataFrame({"feature": [1.0]}),
            warmup_calls=warmup_calls,
            measured_calls=measured_calls,
        )
