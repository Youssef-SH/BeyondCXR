"""Publish a validated RSNA artifact bundle."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.rsna_artifacts import build_and_write
from beyondcxr.data.rsna_splitting import SplitConfig
from beyondcxr.utils.operational_logging import (
    add_logging_argument,
    configure_logging,
    get_operational_logger,
    timed_phase,
)

_LOGGER = get_operational_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/raw/rsna/extracted"),
        help="Directory containing the extracted RSNA Stage 2 files",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/manifests"),
        help="Root directory for versioned RSNA artifact bundles",
    )
    parser.add_argument("--split-seed", type=int, default=42, help="Patient split seed")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training patient ratio",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="Validation patient ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test patient ratio",
    )
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        split_config = SplitConfig(
            seed=args.split_seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
        )
        with timed_phase(_LOGGER, "manifest_construction"):
            written = build_and_write(args.source_root, args.output_directory, split_config)
    except (ManifestBuildError, OSError) as exc:
        print(f"RSNA manifest build failed: {type(exc).__name__}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "bundle_id": written.paths.bundle_id,
                "bundle_directory": str(written.paths.bundle_directory),
                "current_marker": str(written.paths.current_path),
                "sample_count": written.sample_count,
                "label_count": written.label_count,
                "annotation_count": written.annotation_count,
                "split_count": written.split_count,
                "split_recipe_id": written.split_recipe_id,
                "logical_table_sha256": dict(written.logical_table_sha256),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
