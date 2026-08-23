"""Render aggregate binary-classification evaluation plots."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "beyondcxr-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from beyondcxr.evaluation.metrics import (
    CALIBRATION_BINNING_STRATEGY,
    validated_binary_arrays,
)


def write_evaluation_plots(
    targets: ArrayLike,
    probabilities: ArrayLike,
    *,
    youden_j_threshold: float,
    target_sensitivity_threshold: float | None,
    calibration_bins: int,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Write threshold-qualified confusion matrices and probability curves."""
    truth, scores = validated_binary_arrays(targets, probabilities)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.append(
        _confusion_plot(
            truth,
            scores,
            threshold=youden_j_threshold,
            path=destination / "confusion_matrix_youden_j.png",
            title="Confusion matrix — Youden-J operating point",
        )
    )
    if target_sensitivity_threshold is not None:
        paths.append(
            _confusion_plot(
                truth,
                scores,
                threshold=target_sensitivity_threshold,
                path=destination / "confusion_matrix_target_sensitivity.png",
                title="Confusion matrix — target-sensitivity operating point",
            )
        )

    roc_path = destination / "roc_curve.png"
    false_positive_rate, true_positive_rate, _ = roc_curve(truth, scores)
    _line_plot(
        false_positive_rate,
        true_positive_rate,
        roc_path,
        title="ROC curve",
        x_label="False-positive rate",
        y_label="True-positive rate",
        diagonal=True,
    )
    paths.append(roc_path)

    precision_recall_path = destination / "precision_recall_curve.png"
    precision, recall, _ = precision_recall_curve(truth, scores)
    _line_plot(
        recall,
        precision,
        precision_recall_path,
        title="Precision-recall curve",
        x_label="Recall",
        y_label="Precision",
    )
    paths.append(precision_recall_path)

    calibration_path = destination / "calibration_curve.png"
    observed, predicted = calibration_curve(
        truth,
        scores,
        n_bins=calibration_bins,
        strategy=CALIBRATION_BINNING_STRATEGY,
    )
    _line_plot(
        predicted,
        observed,
        calibration_path,
        title=f"Calibration curve — {calibration_bins} equal-width bins",
        x_label="Mean predicted probability",
        y_label="Observed positive fraction",
        diagonal=True,
    )
    paths.append(calibration_path)
    return tuple(paths)


def _confusion_plot(
    truth: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    path: Path,
    title: str,
) -> Path:
    predictions = (scores >= threshold).astype(np.int8)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=["negative", "positive"])
    display.plot(cmap="Blues", colorbar=False)
    display.ax_.set_title(f"{title}\nthreshold={threshold:.6f}")
    display.figure_.tight_layout()
    display.figure_.savefig(path, dpi=160)
    plt.close(display.figure_)
    return path


def _line_plot(
    x: np.ndarray,
    y: np.ndarray,
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    diagonal: bool = False,
) -> None:
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(x, y, linewidth=2)
    if diagonal:
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set(xlim=(0, 1), ylim=(0, 1), title=title, xlabel=x_label, ylabel=y_label)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
