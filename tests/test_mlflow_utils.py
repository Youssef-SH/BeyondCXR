from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import mlflow
import pytest

from radfusion.utils.mlflow_utils import configure_mlflow, uv_lock_sha256


def test_uv_lock_hash_uses_exact_file_bytes(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    content = b"version = 1\n"
    lock.write_bytes(content)
    assert uv_lock_sha256(lock) == hashlib.sha256(content).hexdigest()


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


def test_make_purge_generated_removes_sqlite_state_and_preserves_raw_data(tmp_path: Path) -> None:
    for name in (
        "mlflow.db",
        "mlflow.db-wal",
        "mlflow.db-shm",
        "mlartifacts/run/artifact.txt",
        "data/manifests/rsna/builds/build-test/artifact",
        "data/manifests/rsna/CURRENT",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated\n", encoding="utf-8")
    raw = tmp_path / "data/raw/rsna/source.dcm"
    raw.parent.mkdir(parents=True)
    raw.write_text("source\n", encoding="utf-8")

    completed = subprocess.run(
        ["make", "-f", str(Path("Makefile").resolve()), "purge-generated"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert raw.is_file()
    assert not (tmp_path / "mlflow.db").exists()
    assert not (tmp_path / "mlflow.db-wal").exists()
    assert not (tmp_path / "mlflow.db-shm").exists()
    assert not (tmp_path / "mlartifacts").exists()
    assert not (tmp_path / "data/manifests/rsna/CURRENT").exists()
    assert not (tmp_path / "data/manifests/rsna/builds/build-test").exists()
