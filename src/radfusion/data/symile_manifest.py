"""Publish an authenticated immutable Symile-MIMIC bundle."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from radfusion.data.errors import ManifestBuildError
from radfusion.data.hashing import sha256_file
from radfusion.data.symile_artifacts import build_symile_artifacts, write_symile_bundle
from radfusion.data.symile_source import qualify_symile_source
from radfusion.utils.operational_logging import (
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
    configure_logging(args.log_level)
    try:
        with timed_phase(_LOGGER, "source_qualification"):
            source = qualify_symile_source(args.source_root)
        with timed_phase(_LOGGER, "manifest_construction"):
            result = build_symile_artifacts(source)
        with timed_phase(_LOGGER, "bundle_publication"):
            paths = write_symile_bundle(result, args.output_directory)
    except (ManifestBuildError, OSError, ValueError, KeyError) as exc:
        print(f"Symile manifest build failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "bundle_id": paths.bundle_id,
                "bundle_manifest_sha256": sha256_file(paths.metadata_path),
                "bundle_directory": paths.bundle_directory.as_posix(),
                "current_marker": paths.current_path.as_posix(),
                "sample_count": result.samples.num_rows,
                "split_assignment_id": result.metadata["membership"]["split_assignment_id"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
