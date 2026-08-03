from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from radfusion.utils.private_predictions import (
    PREDICTION_MANIFEST_FILENAME,
    PREDICTIONS_FILENAME,
    publish_private_neural_predictions,
    validate_private_neural_predictions,
)


def _publish(tmp_path: Path) -> Path:
    logits = np.asarray([-1.0, 2.0], dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return publish_private_neural_predictions(
        private_root=tmp_path / "private",
        dataset="synthetic",
        training_run_id="training-run",
        test_evaluation_run_id="test-run",
        model_package_id="model-package-test",
        seed=42,
        sample_ids=("sample-a", "sample-b"),
        patient_keys=("patient-a", "patient-b"),
        targets=np.asarray([0, 1], dtype=np.int8),
        logits=logits,
        probabilities=probabilities,
    )


def test_private_predictions_publish_exact_aligned_contract(tmp_path: Path) -> None:
    destination = _publish(tmp_path)
    manifest = validate_private_neural_predictions(destination)
    table = pq.read_table(destination / PREDICTIONS_FILENAME)

    assert destination == tmp_path / "private/predictions/synthetic/test-run"
    assert {path.name for path in destination.iterdir()} == {
        PREDICTIONS_FILENAME,
        PREDICTION_MANIFEST_FILENAME,
    }
    assert manifest["row_count"] == 2
    assert table.column("sample_id").to_pylist() == ["sample-a", "sample-b"]
    assert table.column("private_patient_key").to_pylist() == ["patient-a", "patient-b"]
    assert set(table.column("training_run_id").to_pylist()) == {"training-run"}
    assert set(table.column("test_evaluation_run_id").to_pylist()) == {"test-run"}
    assert set(table.column("model_package_id").to_pylist()) == {"model-package-test"}
    assert set(table.column("seed").to_pylist()) == {42}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda values: values.update({"sample_ids": ("sample-b", "sample-a")}),
        lambda values: values.update({"patient_keys": ("patient-a", "")}),
        lambda values: values.update({"targets": np.asarray([0, 2])}),
        lambda values: values.update({"logits": np.asarray([-1.0, np.inf])}),
        lambda values: values.update({"probabilities": np.asarray([0.5, 0.5])}),
    ],
)
def test_private_predictions_reject_misaligned_or_invalid_rows(tmp_path: Path, mutation) -> None:
    values = {
        "sample_ids": ("sample-a", "sample-b"),
        "patient_keys": ("patient-a", "patient-b"),
        "targets": np.asarray([0, 1]),
        "logits": np.asarray([-1.0, 2.0]),
        "probabilities": 1.0 / (1.0 + np.exp(-np.asarray([-1.0, 2.0]))),
    }
    mutation(values)

    with pytest.raises(ValueError):
        publish_private_neural_predictions(
            private_root=tmp_path / "private",
            dataset="synthetic",
            training_run_id="training-run",
            test_evaluation_run_id="test-run",
            model_package_id="model-package-test",
            seed=42,
            **values,
        )


def test_private_prediction_validation_rejects_tampering_and_symlinks(tmp_path: Path) -> None:
    destination = _publish(tmp_path)
    manifest_path = destination / PREDICTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_count"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_private_neural_predictions(destination)

    destination = _publish(tmp_path / "second")
    prediction_path = destination / PREDICTIONS_FILENAME
    target = tmp_path / "predictions.parquet"
    prediction_path.rename(target)
    prediction_path.symlink_to(target)
    with pytest.raises(ValueError):
        validate_private_neural_predictions(destination)


def test_private_prediction_publication_failure_leaves_no_valid_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "radfusion.utils.private_predictions.publish_directory",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError((args, kwargs))),
    )

    with pytest.raises(OSError):
        _publish(tmp_path)

    assert not (tmp_path / "private/predictions/synthetic/test-run").exists()
