"""Synthetic contracts for the shared operational attempt owner."""

import io
import json
import logging
import shutil

import pytest

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.campaign_observability import campaign_log_members, campaign_session
from beyondcxr.utils.operational_logging import (
    get_operational_logger,
    log_event,
    observed_operation,
    operational_output,
    parse_operational_record,
    timed_phase,
)


@pytest.mark.parametrize(
    "study,operation", [("rsna", "campaign"), ("symile", "development"), ("symile", "test")]
)
def test_session_restores_handlers_and_keeps_distinct_safe_attempts(
    tmp_path, capsys, study, operation
):
    root = logging.getLogger("beyondcxr.operational")
    previous = tuple(root.handlers)
    previous_level = root.level
    logger = get_operational_logger("synthetic")
    paths = []
    for _ in range(2):
        with campaign_session(
            repository_root=tmp_path,
            report_root=tmp_path / "reports" / study,
            study=study,
            operation=operation,
        ) as session:
            paths.append(session.log_path)
            session.bind_execution("execution-" + "a" * 64)
            with timed_phase(logger, "synthetic_phase"):
                log_event(logger, "progress", subject_id="PRIVATE", source_path="/private/data")
        assert tuple(root.handlers) == previous
        assert root.level == previous_level
    assert paths[0] != paths[1]
    for path in paths:
        content = path.read_text()
        for event in ("campaign_started", "phase_started", "phase_completed", "campaign_succeeded"):
            assert content.count(f"event={event}") == 1
        assert "PRIVATE" not in content and "/private/data" not in content
        assert "execution_id=execution-" in content
        assert "elapsed_s=" in content
    output = capsys.readouterr()
    assert output.out == ""
    assert "event=campaign_started" in output.err


def test_failure_preserves_exception_and_prior_handler(tmp_path):
    root = logging.getLogger("beyondcxr.operational")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root.addHandler(handler)
    failure = RuntimeError("private exception text")
    try:
        with pytest.raises(RuntimeError) as caught:
            with campaign_session(
                repository_root=tmp_path,
                report_root=tmp_path / "reports",
                study="symile",
                operation="test",
            ) as session:
                raise failure
        assert caught.value is failure
        assert handler in root.handlers
        text = session.log_path.read_text()
        assert text.count("event=campaign_failed") == 1
        assert "campaign_succeeded" not in text
        assert "private exception text" not in text
        assert not (tmp_path / "private").exists()
    finally:
        root.removeHandler(handler)


def test_export_log_members_are_explicit_and_execution_bound(tmp_path):
    with campaign_session(
        repository_root=tmp_path,
        report_root=tmp_path / "reports/symile",
        study="symile",
        operation="test",
    ) as session:
        session.bind_execution("symile-execution-" + "a" * 64)
        with pytest.raises(ManifestBuildError, match="cannot change"):
            session.bind_execution("symile-execution-" + "b" * 64)
        log_event(get_operational_logger("test"), "campaign_ready_for_export")
        members = campaign_log_members(
            session.log_path, study="symile", execution_id="symile-execution-" + "a" * 64
        )
        assert len(members) == 2
        assert all(member.role == "operational_evidence" for member in members)
        assert {member.path.name for member in members} == {"session.json", "execution.log"}
        with pytest.raises(ManifestBuildError, match="differs"):
            campaign_log_members(session.log_path, study="symile", execution_id="foreign")
        assert (
            json.loads(session.log_path.with_name("session.json").read_bytes())["campaign_id"]
            == session.attempt_id
        )


def test_session_rejects_symlinked_log_destination(tmp_path):
    (tmp_path / "reports").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ManifestBuildError, match="symlink"):
        with campaign_session(
            repository_root=tmp_path,
            report_root=tmp_path / "reports",
            study="rsna",
            operation="campaign",
        ):
            pytest.fail("invalid path entered session")


@pytest.mark.parametrize("study,operation", [("rsna", "campaign"), ("symile", "test")])
def test_completed_export_retains_explicit_original_attempt(tmp_path, study, operation):
    from beyondcxr.training.preservation import (
        export_and_verify,
        restore_and_validate_export,
        validate_exact_restored_closure,
    )

    execution_id = f"{study}-execution-" + "a" * 64

    def validate(root):
        archived_log = root / f"reports/{study}/campaigns/export/execution.log"
        assert parse_operational_record(archived_log.read_text().splitlines()[-1])["event"] == (
            "campaign_ready_for_export"
        )
        members = campaign_log_members(
            archived_log,
            study=study,
            execution_id=execution_id,
        )
        validate_exact_restored_closure(root, members)
        return members[0].path.read_bytes()

    archives = []
    identities = []
    live_logs = []
    for _ in range(2):
        with campaign_session(
            repository_root=tmp_path,
            report_root=tmp_path / "reports" / study,
            study=study,
            operation=operation,
        ) as session:
            session.bind_execution(execution_id)
            identities.append(session.attempt_id)
            live_logs.append(session.log_path)
            log_event(get_operational_logger("test"), "campaign_ready_for_export")
            members = campaign_log_members(session.log_path, study=study, execution_id=execution_id)
            archive = export_and_verify(
                members=members,
                export_root=tmp_path / "outbox",
                backup_root=tmp_path / "backup",
                export_name="synthetic-operational-closure",
                restoration_validator=validate,
            )
            archives.append(archive.read_bytes())
    assert all("event=campaign_succeeded" in path.read_text() for path in live_logs)
    assert archives[0] == archives[1]
    preserved = json.loads(restore_and_validate_export(archive, restoration_validator=validate))
    assert preserved["campaign_id"] == identities[0] != identities[1]


def _ready_attempt(root, execution_id="symile-execution-" + "a" * 64):
    with campaign_session(
        repository_root=root, report_root=root / "reports/symile", study="symile", operation="test"
    ) as session:
        session.bind_execution(execution_id)
        session.bind_execution(execution_id)  # Repeated association is idempotent.
        log_event(get_operational_logger("test"), "campaign_ready_for_export", phase="export")
        snapshot = root / f"snapshot-{session.attempt_id}"
        snapshot.mkdir()
        shutil.copyfile(session.log_path, snapshot / "execution.log")
        shutil.copyfile(session.log_path.with_name("session.json"), snapshot / "session.json")
    return snapshot / "execution.log"


def test_same_execution_attempts_cannot_exchange_metadata(tmp_path):
    first = _ready_attempt(tmp_path)
    second = _ready_attempt(tmp_path)
    first.with_name("session.json").write_bytes(second.with_name("session.json").read_bytes())
    with pytest.raises(ManifestBuildError, match="attempt association"):
        campaign_log_members(first, study="symile", execution_id="symile-execution-" + "a" * 64)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_execution",
        "wrong_attempt",
        "malformed",
        "missing",
        "duplicate",
        "unrelated",
        "post_ready",
        "archive_created",
        "campaign_succeeded",
        "unknown_field",
        "premature_execution",
    ],
)
def test_operational_event_binding_rejects_ambiguous_logs(tmp_path, mutation):
    path = _ready_attempt(tmp_path)
    content = path.read_text()
    lines = content.splitlines(keepends=True)
    bound = next(line for line in lines if "event=campaign_execution_bound " in line)
    ready = next(line for line in lines if "event=campaign_ready_for_export " in line)
    if mutation == "wrong_execution":
        content = content.replace(
            "execution_id=symile-execution-" + "a" * 64, "execution_id=symile-execution-" + "b" * 64
        )
    elif mutation == "wrong_attempt":
        identity = json.loads(path.with_name("session.json").read_bytes())["campaign_id"]
        content = content.replace(identity, "campaign-" + "f" * 32)
    elif mutation == "malformed":
        content = content.replace(ready, ready.rstrip() + ' event="unterminated\n')
    elif mutation == "missing":
        content = content.replace(bound, "")
    elif mutation == "duplicate":
        content += ready
    elif mutation == "unrelated":
        content = content.replace(
            ready, ready.replace("event=campaign_ready_for_export", "event=unrelated")
        )
    elif mutation in {"archive_created", "campaign_succeeded"}:
        content = content.replace(
            ready,
            ready.replace("event=campaign_ready_for_export", f"event={mutation}") + ready,
        )
    elif mutation == "unknown_field":
        content = content.replace(ready, ready.rstrip() + " patient_id=private\n")
    elif mutation == "premature_execution":
        started = lines[0]
        content = content.replace(
            started,
            started.rstrip() + " execution_id=symile-execution-" + "a" * 64 + "\n",
        )
    else:
        content += ready.replace("event=campaign_ready_for_export", "event=progress")
    path.write_text(content)
    with pytest.raises(ManifestBuildError, match="operational"):
        campaign_log_members(path, study="symile", execution_id="symile-execution-" + "a" * 64)


def test_campaign_context_domain_is_exact(tmp_path):
    for study, operation in (
        ("rsna", "development"),
        ("rsna", "test"),
        ("symile", "campaign"),
        ("unknown", "test"),
    ):
        with pytest.raises(ValueError, match="Unknown campaign operational context"):
            with campaign_session(
                repository_root=tmp_path,
                report_root=tmp_path / "reports",
                study=study,
                operation=operation,
            ):
                pytest.fail("unsupported campaign context entered")
    path = _ready_attempt(tmp_path)
    with pytest.raises(ManifestBuildError, match="unsupported"):
        campaign_log_members(path, study="unknown", execution_id="symile-execution-" + "a" * 64)


def test_campaign_context_protects_identifiers_from_caller_fields(tmp_path):
    logger = get_operational_logger("test")
    with campaign_session(
        repository_root=tmp_path,
        report_root=tmp_path / "reports",
        study="rsna",
        operation="campaign",
    ) as session:
        forged = dict.fromkeys(("campaign_id", "dataset", "operation", "execution_id"), "forged")
        log_event(logger, "before_binding", phase="preparation", **forged)
        session.bind_execution("execution-real")
        log_event(logger, "after_binding", phase="work", **forged)
    records = [parse_operational_record(line) for line in session.log_path.read_text().splitlines()]
    before = next(record for record in records if record["event"] == "before_binding")
    after = next(record for record in records if record["event"] == "after_binding")
    for record in (before, after):
        assert record["campaign_id"] == session.attempt_id
        assert record["dataset"] == "rsna"
        assert record["operation"] == "campaign"
        assert "forged" not in record.values()
    assert "execution_id" not in before
    assert after["execution_id"] == "execution-real"
    assert (before["phase"], after["phase"]) == ("preparation", "work")


def test_context_cannot_replace_event_or_phase():
    from beyondcxr.utils.operational_logging import configured_logging

    for key in ("event", "phase", "subject_id"):
        with pytest.raises(ValueError, match="context"):
            with configured_logging(stream=io.StringIO(), context={key: "forged"}):
                pytest.fail("invalid context entered")


def test_operational_members_participate_in_complete_topology_validation(tmp_path):
    from pathlib import Path

    from beyondcxr.training.preservation import PreservationMember, validate_preservation_members

    log = _ready_attempt(tmp_path)
    members = campaign_log_members(log, study="symile", execution_id="symile-execution-" + "a" * 64)
    with pytest.raises(ManifestBuildError, match="source paths collide"):
        validate_preservation_members((*members, PreservationMember(log, Path("elsewhere/log"))))
    other = tmp_path / "other"
    other.write_text("synthetic")
    with pytest.raises(ManifestBuildError, match="restore targets collide"):
        validate_preservation_members(
            (*members, PreservationMember(other, members[-1].restore_relative))
        )


def test_overlapping_thread_cannot_take_logging_ownership(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    def enter_competing_session():
        with campaign_session(
            repository_root=tmp_path,
            report_root=tmp_path / "reports",
            study="rsna",
            operation="campaign",
        ):
            pytest.fail("competing thread acquired logging ownership")

    with campaign_session(
        repository_root=tmp_path,
        report_root=tmp_path / "reports",
        study="rsna",
        operation="campaign",
    ) as session:
        logger = get_operational_logger(__name__)
        session.bind_execution("execution-real")
        log_event(logger, "owner_before_worker")
        attempts = {path.name for path in (tmp_path / "reports/campaigns").iterdir()}
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="cannot overlap"):
                pool.submit(enter_competing_session).result()
            pool.submit(log_event, logger, "foreign_worker").result()
        assert {path.name for path in (tmp_path / "reports/campaigns").iterdir()} == attempts
        log_event(get_operational_logger(__name__), "owner_retained")
    records = [parse_operational_record(line) for line in session.log_path.read_text().splitlines()]
    assert "foreign_worker" not in {record["event"] for record in records}
    for event in ("owner_before_worker", "owner_retained", "campaign_succeeded"):
        record = next(record for record in records if record["event"] == event)
        assert record["campaign_id"] == session.attempt_id
        assert record["execution_id"] == "execution-real"

    # Ordinary non-campaign destinations continue to accept worker diagnostics.
    from beyondcxr.utils.operational_logging import configured_logging

    stream = io.StringIO()
    with configured_logging(stream=stream, context={}):
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(log_event, logger, "ordinary_worker").result()
    assert parse_operational_record(stream.getvalue().strip())["event"] == "ordinary_worker"


def test_competing_operational_output_cannot_reuse_campaign_destination(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    def compete():
        with operational_output():
            log_event(get_operational_logger(__name__), "competing_output")

    with campaign_session(
        repository_root=tmp_path,
        report_root=tmp_path / "reports",
        study="rsna",
        operation="campaign",
    ) as session:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="cannot overlap"):
                pool.submit(compete).result()
        log_event(get_operational_logger(__name__), "owner_after_output_rejection")
    content = session.log_path.read_text()
    assert "competing_output" not in content
    assert "event=owner_after_output_rejection" in content


def test_competing_observed_operation_cannot_reuse_campaign_destination(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    @observed_operation("competing_observed")
    def compete():
        log_event(get_operational_logger(__name__), "competing_observed_body")

    with campaign_session(
        repository_root=tmp_path,
        report_root=tmp_path / "reports",
        study="symile",
        operation="test",
    ) as session:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="cannot overlap"):
                pool.submit(compete).result()
        log_event(get_operational_logger(__name__), "owner_after_observed_rejection")
    content = session.log_path.read_text()
    assert "competing_observed" not in content
    assert "event=owner_after_observed_rejection" in content


def test_same_thread_observed_operation_reuses_campaign_destination(tmp_path):
    @observed_operation("nested_observed")
    def nested():
        log_event(get_operational_logger(__name__), "nested_body")

    with campaign_session(
        repository_root=tmp_path,
        report_root=tmp_path / "reports",
        study="rsna",
        operation="campaign",
    ) as session:
        nested()
    content = session.log_path.read_text()
    assert "phase=nested_observed" in content
    assert "event=nested_body" in content
    assert f"campaign_id={session.attempt_id}" in content
