"""Run one experiment from a validated YAML configuration."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mlflow.exceptions import MlflowException
from sqlalchemy.exc import SQLAlchemyError

from radfusion.training.config import ConfigError, load_experiment_config
from radfusion.training.registry import RegistryError
from radfusion.training.train_image import train_image_experiment
from radfusion.training.train_tabular import train_configured_experiment
from radfusion.utils.mlflow_utils import DEFAULT_TRACKING_URI
from radfusion.utils.operational_logging import add_logging_argument, configure_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML file")
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow SQLite tracking URI",
    )
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load one config, execute it, and print aggregate run lineage."""
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_experiment_config(args.config)
        result = (
            train_image_experiment(config, tracking_uri=args.tracking_uri)
            if config.model.modality == "image"
            else train_configured_experiment(config, tracking_uri=args.tracking_uri)
        )
    except (
        ConfigError,
        RegistryError,
        MlflowException,
        SQLAlchemyError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "experiment": config.name,
                "config": config.source_path.as_posix(),
                "model_name": result.model_name,
                "mlflow_run_id": result.run_id,
                "validation_average_precision": result.validation_probability.average_precision,
                "model_path": result.model_path.as_posix(),
                "artifact_directory": result.artifact_directory.as_posix(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
