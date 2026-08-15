from __future__ import annotations

import hashlib
from pathlib import Path

import mlflow
import pytest

from radfusion.utils.mlflow_utils import configure_mlflow, discover_repository_root, uv_lock_sha256


def test_uv_lock_hash_uses_exact_file_bytes(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    content = b"version = 1\n"
    lock.write_bytes(content)
    assert uv_lock_sha256(lock) == hashlib.sha256(content).hexdigest()


def test_repository_discovery_validates_the_project_identity(tmp_path: Path) -> None:
    assert (discover_repository_root() / "pyproject.toml").is_file()
    with pytest.raises(ValueError, match="cannot be discovered"):
        discover_repository_root(tmp_path)


def test_mlflow_initialization_uses_isolated_sqlite_and_local_artifacts(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{database.as_posix()}"

    client = configure_mlflow(experiment_name="test-experiment", tracking_uri=tracking_uri)
    with mlflow.start_run() as run:
        artifact = tmp_path / "artifact.txt"
        artifact.write_text("artifact\n", encoding="utf-8")
        mlflow.log_artifact(artifact)

    experiment = client.get_experiment(run.info.experiment_id)
    assert database.is_file()
    assert experiment.artifact_location == (tmp_path / "mlartifacts").as_uri()
    assert Path(client.download_artifacts(run.info.run_id, "artifact.txt")).read_bytes() == (
        artifact.read_bytes()
    )


@pytest.mark.parametrize(
    "tracking_uri",
    ["file:///tmp/mlruns", "sqlite:///:memory:", "sqlite:///"],
)
def test_mlflow_initialization_rejects_nonpersistent_local_backends(tracking_uri: str) -> None:
    with pytest.raises(ValueError):
        configure_mlflow(tracking_uri=tracking_uri)
