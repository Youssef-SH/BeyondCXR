"""Command-line entry points for result publication, reproduction, serving authority,
and release checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.release.checks import check_repository, run_final_acceptance


def main(argv: list[str] | None = None) -> int:
    """Run result publication, exact reproduction, serving-authority publication,
    or release checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    results = subparsers.add_parser("results")
    results.add_argument("--artifact-root", type=Path, required=True)
    results.add_argument("--output-root", type=Path, default=Path("results"))
    results.add_argument("--readme", type=Path, default=Path("README.md"))
    results.add_argument("--model-card", type=Path, default=Path("docs/model_card.md"))
    reproduce = subparsers.add_parser("reproduce")
    reproduce.add_argument("--binding", type=Path, default=Path("results/symile/binding.json"))
    reproduce.add_argument("--artifact-root", type=Path, required=True)
    reproduce.add_argument("--output-root", type=Path, default=Path("results"))
    reproduce.add_argument("--readme", type=Path, default=Path("README.md"))
    reproduce.add_argument("--model-card", type=Path, default=Path("docs/model_card.md"))
    check = subparsers.add_parser("check")
    check.add_argument("--root", type=Path, default=Path.cwd())
    check.add_argument("--final", action="store_true")
    check.add_argument("--artifact-root", type=Path)
    check.add_argument("--authority-root", type=Path)
    serving = subparsers.add_parser("serving-authority")
    serving.add_argument("--artifact-root", type=Path, required=True)
    serving.add_argument("--authority-root", type=Path, required=True)
    serving.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        if args.command == "results":
            from beyondcxr.release.reproduction import publish_results

            publish_results(
                artifact_root=args.artifact_root,
                output_root=args.output_root,
                readme_path=args.readme,
                model_card_path=args.model_card,
            )
        elif args.command == "reproduce":
            from beyondcxr.release.reproduction import reproduce_results

            reproduce_results(
                binding_path=args.binding,
                artifact_root=args.artifact_root,
                output_root=args.output_root,
                readme_path=args.readme,
                model_card_path=args.model_card,
            )
        elif args.command == "serving-authority":
            from beyondcxr.release.serving import publish_and_smoke_test_serving_authority

            authority = publish_and_smoke_test_serving_authority(
                artifact_root=args.artifact_root,
                authority_root=args.authority_root,
                repository_root=args.repository_root,
            )
            print(authority.name)
        else:
            if args.final:
                if args.artifact_root is None or args.authority_root is None:
                    raise ManifestBuildError(
                        "Final acceptance requires artifact and authority roots"
                    )
                run_final_acceptance(
                    args.root,
                    artifact_root=args.artifact_root,
                    authority_root=args.authority_root,
                )
            else:
                check_repository(args.root)
    except Exception as exc:
        print(f"Release validation failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
