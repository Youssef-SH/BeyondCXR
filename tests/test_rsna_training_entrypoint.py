from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from radfusion.training.rsna_train import main


def test_training_entrypoint_invokes_runner_from_config(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}
    package_id = "model-package-" + "a" * 64

    def fake_train(config, *, tracking_uri):
        captured["config"] = config
        captured["tracking_uri"] = tracking_uri
        return SimpleNamespace(
            model_package_id=package_id,
            run_id="run-test",
            validation_probability=SimpleNamespace(average_precision=0.4),
            model_path=Path(f"models/rsna/packages/{package_id}/model.skops"),
            artifact_directory=Path("reports/rsna/runs/run-test"),
        )

    monkeypatch.setattr("radfusion.training.rsna_train.train_metadata_experiment", fake_train)

    assert main(["--config", "configs/rsna_metadata_logistic.yaml", "--seed", "42"]) == 0
    assert captured["config"].family.family_id == "metadata_logistic"
    assert captured["config"].runtime.seed == 42
    assert captured["tracking_uri"] == "sqlite:///mlflow.db"
    output = capsys.readouterr().out
    assert '"config": "configs/rsna_metadata_logistic.yaml"' in output
    assert f'"model_package_id": "{package_id}"' in output
    assert '"family_id": "metadata_logistic"' in output
