from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from beyondcxr import workflow
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.rsna_campaign import execute_rsna_campaign
from beyondcxr.training.symile_campaign import run_symile_campaign
from beyondcxr.workflow import WorkflowError
from beyondcxr.workflow_config import configure_local_machine


def test_noninteractive_setup_explains_exact_backup_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(workflow, "discover_repository_root", lambda: tmp_path)
    monkeypatch.setattr(workflow.sys.stdin, "isatty", lambda: False)
    assert workflow.main(["setup"]) == 2
    assert "make setup BACKUP_ROOT=/approved/persistent/backup" in capsys.readouterr().err
    assert not (tmp_path / ".beyondcxr.local.toml").exists()


def _configured_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    backup = tmp_path / "backup"
    configure_local_machine(repository, backup_root=backup)
    monkeypatch.setattr(workflow, "discover_repository_root", lambda: repository)
    return repository, backup


def test_rsna_workflow_composes_canonical_owners_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repository, backup = _configured_repository(tmp_path, monkeypatch)
    source = repository / "data/raw/rsna/extracted"
    events: list[object] = []
    bundle = SimpleNamespace(paths=SimpleNamespace(bundle_id="bundle-rsna"))
    campaign = SimpleNamespace(archive_path=repository / "outbox/rsna-test.zip")

    monkeypatch.setattr(
        "beyondcxr.rsna_acquisition.ensure_rsna_source",
        lambda path: events.append(("source", path)) or source,
    )
    monkeypatch.setattr(
        "beyondcxr.data.rsna_artifacts.build_and_write",
        lambda *args: events.append(("bundle", args)) or bundle,
    )
    monkeypatch.setattr(
        "beyondcxr.data.rsna_audit.generate_rsna_audit",
        lambda *args, **kwargs: events.append(("audit", args, kwargs)),
    )
    monkeypatch.setattr(
        "beyondcxr.models.cxr_baseline.ensure_pretrained_weights",
        lambda: events.append("weights"),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_campaign.execute_rsna_campaign",
        create_autospec(
            execute_rsna_campaign,
            side_effect=lambda **kwargs: events.append(("campaign", kwargs)) or campaign,
        ),
    )

    workflow.run_rsna()

    assert [item if isinstance(item, str) else item[0] for item in events] == [
        "source",
        "bundle",
        "audit",
        "weights",
        "campaign",
    ]
    assert events[-1][1] == {
        "backup_root": backup.resolve(),
        "source_root": source,
        "bundle_id": bundle.paths.bundle_id,
    }
    assert events[2][2] == {"bundle_id": bundle.paths.bundle_id}
    assert "RSNA study complete." in capsys.readouterr().out


def test_symile_workflow_never_acquires_source_and_stops_at_pretest_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repository, backup = _configured_repository(tmp_path, monkeypatch)
    source = repository / "data/raw/symile/extracted"
    source.mkdir(parents=True)
    events: list[object] = []
    bundle = SimpleNamespace(bundle_id="bundle-symile")
    monkeypatch.setattr(
        "beyondcxr.data.symile_source.publish_symile_source_bundle",
        lambda *args, **kwargs: events.append(("bundle", args, kwargs)) or (bundle, 10),
    )
    monkeypatch.setattr(
        "beyondcxr.data.symile_audit.generate_symile_audit",
        lambda *args, **kwargs: events.append(("audit", args, kwargs)),
    )
    monkeypatch.setattr(
        "beyondcxr.data.symile_cv.publish_symile_cv",
        lambda *args, **kwargs: (
            events.append(("cv", args, kwargs)) or ("cv-returned", repository / "cv")
        ),
    )
    monkeypatch.setattr(
        "beyondcxr.models.cxr_baseline.ensure_pretrained_weights",
        lambda: events.append("weights"),
    )
    monkeypatch.setattr(
        "beyondcxr.training.symile_campaign.run_symile_campaign",
        create_autospec(
            run_symile_campaign,
            side_effect=lambda **kwargs: (
                events.append(("campaign", kwargs)) or {"freeze_id": "pretest-freeze-test"}
            ),
        ),
    )

    workflow.run_symile()

    assert [item if isinstance(item, str) else item[0] for item in events] == [
        "bundle",
        "audit",
        "cv",
        "weights",
        "campaign",
    ]
    assert events[-1][1] == {
        "source_root": source.resolve(),
        "backup_root": backup.resolve(),
        "bundle_id": bundle.bundle_id,
        "cv_assignment_id": "cv-returned",
    }
    assert events[1][2] == {"bundle_id": bundle.bundle_id}
    assert events[2][1] == (bundle,)
    output = capsys.readouterr().out
    assert "Official test data have NOT been opened." in output
    assert "pretest-freeze-test" in output
    assert "make symile-test" in output


def test_symile_missing_source_is_actionable_and_has_no_acquisition_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configured_repository(tmp_path, monkeypatch)
    with pytest.raises(WorkflowError, match="Place the authorized private Symile release"):
        workflow.run_symile()


def test_rsna_source_failure_has_safe_actionable_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configured_repository(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "beyondcxr.rsna_acquisition.ensure_rsna_source",
        lambda _path: (_ for _ in ()).throw(ManifestBuildError("private-source-detail")),
    )

    with pytest.raises(WorkflowError, match="Kaggle credentials and competition access") as error:
        workflow.run_rsna()

    assert "private-source-detail" not in str(error.value)


def test_invalid_symile_source_has_safe_actionable_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _backup = _configured_repository(tmp_path, monkeypatch)
    (repository / "data/raw/symile/extracted").mkdir(parents=True)
    monkeypatch.setattr(
        "beyondcxr.data.symile_source.publish_symile_source_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ManifestBuildError("private-row-detail")),
    )

    with pytest.raises(WorkflowError, match="release layout and checksums") as error:
        workflow.run_symile()

    assert "private-row-detail" not in str(error.value)


def test_symile_test_delegates_to_the_campaign_composition_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, backup = _configured_repository(tmp_path, monkeypatch)
    observed: list[dict[str, Path]] = []
    monkeypatch.setattr(
        "beyondcxr.training.symile_campaign.resume_symile_test_phase",
        lambda **kwargs: observed.append(kwargs) or {"global_result_id": "global-result-test"},
    )
    workflow.run_symile_test()
    assert observed == [
        {
            "source_root": (repository / "data/raw/symile/extracted").resolve(),
            "backup_root": backup.resolve(),
        }
    ]


@pytest.mark.parametrize(
    ("domain_message", "public_message"),
    [
        (
            "No completed Symile pre-test freeze exists; run `make symile` first",
            "No completed Symile pre-test freeze exists",
        ),
        (
            "Symile scientific freeze selection is ambiguous",
            "Multiple completed Symile pre-test freezes are ambiguous",
        ),
    ],
)
def test_symile_test_reports_zero_and_ambiguous_freeze_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    domain_message: str,
    public_message: str,
) -> None:
    _configured_repository(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "beyondcxr.training.symile_campaign.resume_symile_test_phase",
        lambda **_kwargs: (_ for _ in ()).throw(ManifestBuildError(domain_message)),
    )

    with pytest.raises(WorkflowError, match=public_message):
        workflow.run_symile_test()


def test_release_delegates_to_final_acceptance_with_stable_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, backup = _configured_repository(tmp_path, monkeypatch)
    observed: list[tuple[Path, dict[str, Path]]] = []
    monkeypatch.setattr(
        "beyondcxr.release.checks.run_final_acceptance",
        lambda root, **kwargs: observed.append((root, kwargs)),
    )
    workflow.run_release()
    assert observed == [
        (
            repository,
            {
                "artifact_root": repository / "outbox",
                "authority_root": backup.resolve() / "serving-authorities",
            },
        )
    ]


@pytest.mark.parametrize(
    ("domain_message", "public_message"),
    [
        (
            "No valid completed Symile release candidate exists",
            "No completed valid Symile release is available",
        ),
        (
            "Multiple valid completed Symile release candidates are ambiguous",
            "Multiple completed Symile releases are ambiguous",
        ),
    ],
)
def test_release_reports_zero_and_ambiguous_candidate_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    domain_message: str,
    public_message: str,
) -> None:
    _configured_repository(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "beyondcxr.release.checks.run_final_acceptance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ManifestBuildError(domain_message)),
    )

    with pytest.raises(WorkflowError, match=public_message):
        workflow.run_release()


def test_serve_uses_validated_state_and_fixed_local_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, backup = _configured_repository(tmp_path, monkeypatch)
    state = SimpleNamespace(
        authority=SimpleNamespace(directory=tmp_path / "authority"),
        deployment_provenance=tmp_path / "deployment.json",
        result_binding=tmp_path / "binding.json",
        release_archive=tmp_path / "release.zip",
    )
    observed_resolution: list[dict[str, Path]] = []
    monkeypatch.setattr(
        "beyondcxr.release.serving.resolve_accepted_serving_state",
        lambda **kwargs: observed_resolution.append(kwargs) or state,
    )
    observed: list[list[str]] = []
    monkeypatch.setattr(
        "beyondcxr.serving.cli.main", lambda arguments: observed.append(arguments) or 0
    )
    workflow.run_serve()
    assert observed == [
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
        ]
    ]
    assert observed_resolution == [
        {
            "artifact_root": repository / "outbox",
            "authority_root": backup / "serving-authorities",
        }
    ]


@pytest.mark.parametrize("command", ["rsna", "release"])
def test_cli_does_not_echo_private_domain_exception_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    monkeypatch.setattr(
        workflow,
        f"run_{command}",
        lambda: (_ for _ in ()).throw(ManifestBuildError("patient-secret-value")),
    )
    assert workflow.main([command]) == 1
    error = capsys.readouterr().err
    assert "patient-secret-value" not in error
    assert f"make {command}" in error
    if command == "release":
        assert "docs/reproducibility.md" in error
        assert "make symile-test" not in error


@pytest.mark.parametrize(
    ("domain_message", "public_message"),
    [
        ("No accepted state: private-detail", "run `make release` first"),
        ("Multiple accepted states: private-detail", "Multiple accepted serving states"),
    ],
)
def test_serve_rejects_missing_or_ambiguous_state_without_starting_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    domain_message: str,
    public_message: str,
) -> None:
    _configured_repository(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "beyondcxr.release.serving.resolve_accepted_serving_state",
        lambda **_kwargs: (_ for _ in ()).throw(ManifestBuildError(domain_message)),
    )
    monkeypatch.setattr(
        "beyondcxr.serving.cli.main", lambda *_args: pytest.fail("service must not start")
    )
    assert workflow.main(["serve"]) == 2
    error = capsys.readouterr().err
    assert public_message in error
    assert "private-detail" not in error


@pytest.mark.parametrize("action", ["rsna", "symile", "symile-test", "release", "serve"])
def test_direct_public_workflow_failure_is_structured_without_private_text(
    tmp_path, monkeypatch, capsys, action
):
    from beyondcxr.utils.operational_logging import configured_logging

    def reject(repository):
        raise RuntimeError("PRIVATE_SOURCE_DETAIL")

    monkeypatch.setattr(workflow, "discover_repository_root", lambda: tmp_path)
    monkeypatch.setattr(workflow, "_required_config", reject)
    with configured_logging(stream=workflow.sys.stderr, context={}):
        with pytest.raises(RuntimeError, match="PRIVATE_SOURCE_DETAIL"):
            getattr(workflow, "run_" + action.replace("-", "_"))()
    output = capsys.readouterr()
    assert "event=phase_started" in output.err
    assert "event=phase_failed" in output.err
    assert "PRIVATE_SOURCE_DETAIL" not in output.err
    assert output.out == ""
