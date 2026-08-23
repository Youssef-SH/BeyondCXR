"""Publish explicit aggregate cross-family Symile core-development analysis."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.symile_families import SYMILE_CORE_DEVELOPMENT_FAMILIES
from beyondcxr.utils.operational_logging import add_logging_argument, configure_logging
from beyondcxr.utils.publication import validate_path_component
from beyondcxr.utils.symile_publication import (
    ValidatedDevelopmentResult,
    publish_analysis_result,
    validate_development_result,
)


def analyze_symile_development(
    development_ids: Sequence[str],
    *,
    report_root: str | Path = "reports/symile/development",
    model_root: str | Path = "models/symile/development",
    prediction_root: str | Path = "private",
    manifest_root: str | Path = "data/manifests",
) -> tuple[str, Path]:
    """Validate six explicit family authorities and publish aggregate evidence."""
    if len(development_ids) != 6 or len(set(development_ids)) != 6:
        raise ManifestBuildError("Symile analysis requires six unique development IDs")
    developments = _resolve_developments(
        development_ids,
        report_root,
        model_root,
        prediction_root,
        manifest_root,
    )
    if set(developments) != set(SYMILE_CORE_DEVELOPMENT_FAMILIES):
        raise ManifestBuildError(
            "Symile analysis development IDs do not cover exact core-development families"
        )
    return publish_analysis_result(
        report_root=report_root,
        model_root=model_root,
        prediction_root=prediction_root,
        manifest_root=manifest_root,
        family_development_ids={
            family: str(result.manifest["development_id"])
            for family, result in developments.items()
        },
    )


def _resolve_developments(
    development_ids: Sequence[str],
    report_root: str | Path,
    model_root: str | Path,
    prediction_root: str | Path,
    manifest_root: str | Path,
) -> dict[str, ValidatedDevelopmentResult]:
    root = Path(report_root) / "families"
    result: dict[str, ValidatedDevelopmentResult] = {}
    for development_id in development_ids:
        validate_path_component(development_id, "development ID")
        reference = validate_development_result(
            root / development_id,
            model_root=model_root,
            prediction_root=prediction_root,
            manifest_root=manifest_root,
            expected_development_id=development_id,
        )
        family = str(reference.manifest["family_id"])
        if family in result:
            raise ManifestBuildError("Symile analysis received duplicate family authorities")
        result[family] = reference
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-ids", nargs=6, required=True)
    parser.add_argument("--report-root", type=Path, default=Path("reports/symile/development"))
    parser.add_argument("--model-root", type=Path, default=Path("models/symile/development"))
    parser.add_argument("--prediction-root", type=Path, default=Path("private"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        analysis_id, directory = analyze_symile_development(
            args.development_ids,
            report_root=args.report_root,
            model_root=args.model_root,
            prediction_root=args.prediction_root,
            manifest_root=args.manifest_root,
        )
    except (ManifestBuildError, OSError, ValueError) as exc:
        print(f"Symile development analysis failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(
        json.dumps({"analysis_id": analysis_id, "report_directory": directory.as_posix()}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
