"""Publish an authenticated immutable Symile-MIMIC bundle."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_source import publish_symile_source_bundle
from beyondcxr.utils.operational_logging import (
    add_logging_argument,
    cli_operation,
    get_operational_logger,
    timed_phase,
)

_LOGGER = get_operational_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/raw/symile/extracted"),
        help="Extracted official Symile-MIMIC 1.0.0 release",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/manifests"),
        help="Root directory for immutable dataset bundles",
    )
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with cli_operation("symile_manifest_cli", level=args.log_level):
            with timed_phase(_LOGGER, "source_qualification_and_bundle_publication"):
                paths, sample_count = publish_symile_source_bundle(
                    args.source_root,
                    output_directory=args.output_directory,
                )
    except (ManifestBuildError, OSError, ValueError, KeyError) as exc:
        print(f"Symile manifest build failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "bundle_id": paths.bundle_id,
                "bundle_directory": paths.bundle_directory.as_posix(),
                "current_marker": paths.current_path.as_posix(),
                "sample_count": sample_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
