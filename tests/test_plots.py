from __future__ import annotations

import numpy as np

from beyondcxr.evaluation.metrics import CALIBRATION_BINNING_STRATEGY
from beyondcxr.evaluation.plots import write_evaluation_plots


def test_calibration_plot_uses_configured_bins_and_shared_strategy(tmp_path, monkeypatch) -> None:
    observed_call: dict[str, object] = {}

    def fake_calibration_curve(targets, probabilities, *, n_bins, strategy):
        observed_call.update(n_bins=n_bins, strategy=strategy)
        return np.asarray([0.25, 0.75]), np.asarray([0.2, 0.8])

    monkeypatch.setattr("beyondcxr.evaluation.plots.calibration_curve", fake_calibration_curve)
    paths = write_evaluation_plots(
        [0, 0, 1, 1],
        [0.1, 0.3, 0.7, 0.9],
        youden_j_threshold=0.5,
        target_sensitivity_threshold=0.3,
        calibration_bins=7,
        output_directory=tmp_path,
    )

    assert observed_call == {"n_bins": 7, "strategy": CALIBRATION_BINNING_STRATEGY}
    assert {path.name for path in paths} == {
        "confusion_matrix_youden_j.png",
        "confusion_matrix_target_sensitivity.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "calibration_curve.png",
    }
