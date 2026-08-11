"""Configure local SQLite-backed MLflow experiment tracking."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import lightgbm
import mlflow
import numpy
import pyarrow
import sklearn
import skops
from mlflow.tracking import MlflowClient

from radfusion.data.hashing import sha256_file
from radfusion.utils.operational_logging import get_operational_logger, log_event


class SourceConfig(Protocol):
    """Minimal exact-config artifact accepted by MLflow logging."""

    source_path: Path
    source_bytes: bytes


DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_ARTIFACT_DIRECTORY = "mlartifacts"
_LOGGER = get_operational_logger(__name__)


def configure_mlflow(
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str | None = None,
) -> MlflowClient:
    """Initialize one SQLite backend and optionally select an experiment."""
    resolved_uri, database_path = _resolve_sqlite_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(resolved_uri)
    client = MlflowClient(tracking_uri=resolved_uri)
    if experiment_name is None:
        return client

    artifact_location = (database_path.parent / MLFLOW_ARTIFACT_DIRECTORY).resolve().as_uri()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )
    else:
        if experiment.artifact_location != artifact_location:
            raise ValueError(
                f"MLflow experiment {experiment_name!r} uses artifact location "
                f"{experiment.artifact_location!r}, expected {artifact_location!r}"
            )
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return client


@contextmanager
def tracked_run(
    *,
    run_name: str,
    tags: dict[str, str],
    parameters: dict[str, Any],
) -> Iterator[str]:
    """Start an MLflow run and log normalized tags and parameters."""
    context: dict[str, object] | None = None
    try:
        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            seed = tags.get("seed")
            context = {
                "run_id": run.info.run_id,
                "run_kind": tags.get("run_kind", "unknown"),
                "experiment": tags.get("experiment_name", run_name),
                "dataset": tags.get("dataset", "unknown"),
                "model": tags.get("model", "unknown"),
                "seed": _operational_seed(seed),
            }
            log_event(_LOGGER, "run_started", **context)
            mlflow.set_tag("run_date", datetime.now(UTC).date().isoformat())
            mlflow.log_params({key: _parameter_value(value) for key, value in parameters.items()})
            yield run.info.run_id
    except BaseException as exc:
        if context is not None:
            log_event(
                _LOGGER,
                "run_failed",
                level=logging.ERROR,
                error_type=type(exc).__name__,
                **context,
            )
        raise
    else:
        if context is not None:
            log_event(_LOGGER, "run_finished", **context)


def git_revision() -> tuple[str, bool]:
    """Return the current Git commit and dirty-worktree flag."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _operational_seed(value: object) -> int | None:
    try:
        if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
            return None
        return int(value)
    except Exception:
        return None


def uv_lock_sha256(path: str | Path = "uv.lock") -> str:
    """Return the SHA-256 of the exact dependency lock."""
    lock = Path(path)
    if not lock.is_file():
        raise FileNotFoundError(f"Dependency lock is missing: {lock}")
    return sha256_file(lock)


def log_source_config(config: SourceConfig) -> None:
    """Log the exact loaded experiment configuration for the active MLflow run."""
    with tempfile.TemporaryDirectory(prefix="radfusion-config-") as temporary_directory:
        path = Path(temporary_directory) / "resolved_config.yaml"
        path.write_bytes(config.source_bytes)
        if sha256_file(path) != config.source_sha256:
            raise ValueError("Loaded source configuration SHA-256 mismatch")
        mlflow.log_artifact(str(path), artifact_path="config")


def environment_provenance() -> dict[str, str]:
    """Return runtime provenance for experiment tracking."""
    return {
        "environment_python_version": platform.python_version(),
        "environment_operating_system": platform.platform(),
        "environment_cpu_architecture": platform.machine(),
        "environment_cpu_model": cpu_model(),
        "environment_numpy_version": numpy.__version__,
        "environment_pyarrow_version": pyarrow.__version__,
        "environment_scikit_learn_version": sklearn.__version__,
        "environment_lightgbm_version": lightgbm.__version__,
        "environment_mlflow_version": mlflow.__version__,
        "environment_skops_version": skops.__version__,
    }


def cpu_model() -> str:
    """Return the detected CPU model, falling back to ``unknown``."""
    try:
        system = platform.system()
        if system == "Linux":
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
            for line in cpuinfo.splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip().lower() in {"model name", "hardware"}:
                    if model := value.strip():
                        return model
        elif system == "Darwin":
            completed = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if model := completed.stdout.strip():
                return model
        elif system == "Windows":
            if model := os.environ.get("PROCESSOR_IDENTIFIER", "").strip():
                return model
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        return platform.processor().strip() or "unknown"
    except (OSError, ValueError):
        return "unknown"


def _parameter_value(value: Any) -> str | float | int | bool:
    if value is None:
        return "not_applicable"
    if isinstance(value, str | float | int | bool):
        return value
    return str(value)


def _resolve_sqlite_tracking_uri(tracking_uri: str) -> tuple[str, Path]:
    prefix = "sqlite:///"
    if not isinstance(tracking_uri, str) or not tracking_uri.startswith(prefix):
        raise ValueError("MLflow tracking URI must use a local sqlite:/// database")
    database_text = tracking_uri[len(prefix) :]
    if not database_text or database_text == ":memory:" or "?" in database_text:
        raise ValueError("MLflow tracking URI must name a persistent local SQLite database")
    database_path = Path(database_text)
    if not database_path.is_absolute():
        database_path = database_path.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return f"{prefix}{database_path.as_posix()}", database_path
