"""Compose machine setup and study, release, and serving workflows."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.utils.mlflow_utils import discover_repository_root
from beyondcxr.utils.operational_logging import observed_operation
from beyondcxr.workflow_config import configure_local_machine, load_local_config


class WorkflowError(RuntimeError):
    """An actionable, privacy-safe public workflow failure."""


@observed_operation("run_setup")
def run_setup(
    *,
    backup_root: str | Path | None = None,
    rsna_source_root: str | Path | None = None,
    symile_source_root: str | Path | None = None,
    interactive: bool | None = None,
) -> None:
    """Configure machine-local paths; dependency installation belongs to make setup."""
    repository = discover_repository_root()
    try:
        configure_local_machine(
            repository,
            backup_root=backup_root,
            rsna_source_root=rsna_source_root,
            symile_source_root=symile_source_root,
            interactive=sys.stdin.isatty() if interactive is None else interactive,
        )
    except ManifestBuildError as exc:
        raise WorkflowError(
            "Machine setup paths are invalid; use an external writable backup directory and "
            "regular source directories without symbolic-link components. Run "
            "`make setup BACKUP_ROOT=/approved/persistent/backup`"
        ) from exc
    print("BeyondCXR machine setup complete.")


@observed_operation("run_rsna")
def run_rsna() -> None:
    """Run or resume the complete RSNA lifecycle using explicit authority coordinates."""
    repository = discover_repository_root()
    config = _required_config(repository)
    from beyondcxr.data.rsna_artifacts import build_and_write
    from beyondcxr.data.rsna_audit import generate_rsna_audit
    from beyondcxr.data.rsna_splitting import SplitConfig
    from beyondcxr.models.cxr_baseline import ensure_pretrained_weights
    from beyondcxr.rsna_acquisition import ensure_rsna_source
    from beyondcxr.training.rsna_campaign import execute_rsna_campaign

    try:
        source = ensure_rsna_source(config.rsna_source_root)
    except ManifestBuildError as exc:
        raise WorkflowError(
            "RSNA source readiness failed; verify the existing Stage 2 release layout or Kaggle "
            "credentials and competition access, then run `make rsna` again"
        ) from exc
    bundle = build_and_write(source, repository / "data/manifests", SplitConfig())
    generate_rsna_audit(
        repository / "data/manifests",
        repository / "reports/rsna/audit",
        bundle_id=bundle.paths.bundle_id,
    )
    ensure_pretrained_weights()
    result = execute_rsna_campaign(
        backup_root=config.backup_root, source_root=source, bundle_id=bundle.paths.bundle_id
    )
    print("RSNA study complete.")
    print(f"Archive: {result.archive_path.name}")


@observed_operation("run_symile")
def run_symile() -> None:
    """Run or resume Symile through its immutable pre-test freeze, then stop."""
    repository = discover_repository_root()
    config = _required_config(repository)
    if not config.symile_source_root.is_dir():
        raise WorkflowError(
            "Place the authorized private Symile release at the configured source root, "
            "then run `make symile`"
        )
    from beyondcxr.data.symile_audit import generate_symile_audit
    from beyondcxr.data.symile_cv import publish_symile_cv
    from beyondcxr.data.symile_source import publish_symile_source_bundle
    from beyondcxr.models.cxr_baseline import ensure_pretrained_weights
    from beyondcxr.training.symile_campaign import run_symile_campaign

    try:
        bundle, _ = publish_symile_source_bundle(
            config.symile_source_root,
            output_directory=repository / "data/manifests",
        )
    except ManifestBuildError as exc:
        raise WorkflowError(
            "The configured Symile source is invalid; verify the authorized release layout and "
            "checksums, then run `make symile` again"
        ) from exc
    generate_symile_audit(
        repository / "data/manifests",
        repository / "reports/symile/audit",
        bundle_id=bundle.bundle_id,
    )
    cv_assignment_id, _ = publish_symile_cv(
        bundle, manifest_directory=repository / "data/manifests"
    )
    ensure_pretrained_weights()
    result = run_symile_campaign(
        bundle_id=bundle.bundle_id,
        cv_assignment_id=cv_assignment_id,
        source_root=config.symile_source_root,
        backup_root=config.backup_root,
    )
    print("Symile pre-test phase complete.")
    print()
    print("Official test data have NOT been opened.")
    print()
    print("Freeze:")
    print(f"    {result['freeze_id']}")
    print()
    print("Review the freeze summary, then run:")
    print()
    print("    make symile-test")
    print()
    print("to cross the irreversible held-out boundary.")


@observed_operation("run_symile_test")
def run_symile_test() -> None:
    """Open or resume the sole exact frozen Symile execution."""
    repository = discover_repository_root()
    config = _required_config(repository)
    from beyondcxr.training.symile_campaign import resume_symile_test_phase

    try:
        result = resume_symile_test_phase(
            source_root=config.symile_source_root,
            backup_root=config.backup_root,
        )
    except ManifestBuildError as exc:
        message = str(exc).lower()
        if "ambiguous" in message:
            raise WorkflowError(
                "Multiple completed Symile pre-test freezes are ambiguous; resolve the exact "
                "frozen execution with expert tooling"
            ) from exc
        if "no completed symile pre-test freeze" in message:
            raise WorkflowError(
                "No completed Symile pre-test freeze exists; run `make symile` first"
            ) from exc
        raise
    print("Symile held-out phase complete.")
    print(f"Global result: {result['global_result_id']}")


@observed_operation("run_release")
def run_release() -> None:
    """Resolve the sole completed campaign and run canonical final acceptance."""
    repository = discover_repository_root()
    config = _required_config(repository)
    from beyondcxr.release.checks import run_final_acceptance

    try:
        run_final_acceptance(
            repository,
            artifact_root=repository / "outbox",
            authority_root=_serving_authority_root(config.backup_root),
        )
    except ManifestBuildError as exc:
        message = str(exc).lower()
        if message in {
            "final release metadata must declare version 1.0.0",
            "final release checks require a clean git tree",
        }:
            raise WorkflowError(
                "Public results have been derived and reproduced. Review and commit the "
                "generated public results and final 1.0.0 release metadata as described in "
                "docs/reproducibility.md, then rerun `make release` for final acceptance"
            ) from exc
        if "multiple valid completed symile release candidates" in message:
            raise WorkflowError(
                "Multiple completed Symile releases are ambiguous; select one with expert tooling"
            ) from exc
        if "release candidate" in message:
            raise WorkflowError(
                "No completed valid Symile release is available; run `make symile-test` first"
            ) from exc
        raise
    print("Final release accepted.")


@observed_operation("run_serve")
def run_serve() -> None:
    """Validate and run the sole installed accepted authority with local CPU defaults."""
    repository = discover_repository_root()
    config = _required_config(repository)
    from beyondcxr.release.serving import resolve_accepted_serving_context
    from beyondcxr.serving.cli import main as serving_main

    try:
        with resolve_accepted_serving_context(
            artifact_root=repository / "outbox",
            authority_root=_serving_authority_root(config.backup_root),
        ) as serving:
            state = serving.publication
            serving_main(
                [
                    "--authority",
                    str(state.authority.directory),
                    "--deployment-provenance",
                    str(state.deployment_provenance),
                    "--result-binding",
                    str(state.result_binding),
                    "--release-archive",
                    str(state.release_archive),
                    "--device",
                    "cpu",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                ],
                predictor=serving.predictor,
            )
    except ManifestBuildError as exc:
        if "multiple" in str(exc).lower():
            raise WorkflowError(
                "Multiple accepted serving states exist; select one with expert tooling"
            ) from exc
        raise WorkflowError("No accepted serving state exists; run `make release` first") from exc


def _required_config(repository: Path):
    try:
        config = load_local_config(repository)
    except ManifestBuildError as exc:
        raise WorkflowError(
            "Machine setup is missing or invalid; run `make setup` with a valid external backup "
            "root"
        ) from exc
    if config is None:
        raise WorkflowError("Machine setup is incomplete; run `make setup` first")
    return config


def _serving_authority_root(backup_root: Path) -> Path:
    return backup_root / "serving-authorities"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="Set up this machine")
    setup.add_argument("--backup-root", type=Path)
    setup.add_argument("--rsna-source-root", type=Path)
    setup.add_argument("--symile-source-root", type=Path)
    for command in ("rsna", "symile", "symile-test", "release", "serve"):
        commands.add_parser(command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "setup":
            run_setup(
                backup_root=args.backup_root,
                rsna_source_root=args.rsna_source_root,
                symile_source_root=args.symile_source_root,
            )
        elif args.command == "rsna":
            run_rsna()
        elif args.command == "symile":
            run_symile()
        elif args.command == "symile-test":
            run_symile_test()
        elif args.command == "release":
            run_release()
        else:
            run_serve()
    except WorkflowError as exc:
        print(f"BeyondCXR: {exc}", file=sys.stderr)
        return 2
    except (ManifestBuildError, FileNotFoundError, OSError, ValueError) as exc:
        action = {
            "rsna": "Resolve the reported source or authority problem, then run `make rsna` again",
            "symile": "Resolve the reported pre-test problem, then run `make symile` again",
            "symile-test": "Run `make symile` first or validate the frozen execution",
            "release": (
                "Check the release prerequisites and acceptance steps in "
                "docs/reproducibility.md, then run `make release` again"
            ),
            "serve": "Run `make release` first",
            "setup": "Provide a valid persistent backup root",
        }[args.command]
        print(f"BeyondCXR {args.command} failed ({type(exc).__name__}). {action}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
