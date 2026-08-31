from __future__ import annotations

import argparse
import importlib
import io
import json
import logging
import math
import shlex
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from beyondcxr.training.rsna_train import main as train_main
from beyondcxr.utils.operational_logging import (
    CountProgress,
    configure_logging,
    get_operational_logger,
    log_event,
    timed_phase,
)


@pytest.mark.parametrize(
    ("module_name", "operation", "evaluation_ids", "result"),
    [
        (
            "training.rsna_compare",
            "regenerate_comparison",
            ["evaluation-a"],
            SimpleNamespace(
                row_count=1,
                csv_path=Path("comparison.csv"),
                markdown_path=Path("comparison.md"),
            ),
        ),
        (
            "training.rsna_seed_summary",
            "publish_seed_summary",
            ["evaluation-a", "evaluation-b", "evaluation-c"],
            SimpleNamespace(directory=Path("seed-summary")),
        ),
    ],
)
def test_rsna_report_clis_forward_explicit_authority_directories(
    module_name: str,
    operation: str,
    evaluation_ids: list[str],
    result: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(f"beyondcxr.{module_name}")
    execution = object()
    freeze = object()
    observed: dict[str, object] = {}

    def validate_execution(path: Path) -> object:
        observed["execution_directory"] = path
        return execution

    def validate_freeze(path: Path, **kwargs: object) -> object:
        observed["freeze_directory"] = path
        observed.update(kwargs)
        return freeze

    def run(ids: list[str], **kwargs: object) -> SimpleNamespace:
        observed["evaluation_ids"] = ids
        observed.update(kwargs)
        return result

    monkeypatch.setattr(module, "validate_execution", validate_execution)
    monkeypatch.setattr(module, "validate_freeze", validate_freeze)
    monkeypatch.setattr(module, operation, run)

    arguments = [
        "--evaluation-ids",
        *evaluation_ids,
        "--execution-directory",
        "explicit/execution",
        "--freeze-directory",
        "explicit/freeze",
        "--bundle-directory",
        "explicit/bundle",
    ]
    assert module.main(arguments) == 0
    assert observed["execution_directory"] == Path("explicit/execution")
    assert observed["freeze_directory"] == Path("explicit/freeze")
    assert observed["bundle_directory"] == Path("explicit/bundle")
    assert observed["execution"] is execution
    assert observed["freeze"] is freeze
    assert observed["evaluation_ids"] == evaluation_ids


@pytest.mark.parametrize(
    ("module_name", "operation", "arguments"),
    [
        ("data.rsna_manifest", "build_and_write", []),
        ("data.symile_manifest", "publish_symile_source_bundle", []),
        ("data.rsna_audit", "generate_rsna_audit", []),
        ("data.symile_audit", "generate_symile_audit", []),
        ("data.symile_cv", "resolve_symile_bundle", []),
        (
            "training.rsna_train",
            "load_experiment_config",
            ["--config", "unused", "--seed", "42", "--bundle-id", "bundle-" + "a" * 64],
        ),
        (
            "training.rsna_compare",
            "regenerate_comparison",
            [
                "--evaluation-ids",
                "unused",
                "--execution-directory",
                "execution",
                "--freeze-directory",
                "freeze",
                "--bundle-directory",
                "bundle",
            ],
        ),
        (
            "training.rsna_seed_summary",
            "publish_seed_summary",
            [
                "--evaluation-ids",
                "a",
                "b",
                "c",
                "--execution-directory",
                "execution",
                "--freeze-directory",
                "freeze",
                "--bundle-directory",
                "bundle",
            ],
        ),
        (
            "training.rsna_campaign",
            "execute_rsna_campaign",
            ["--backup-root", "unused-backup", "--bundle-id", "bundle-" + "a" * 64],
        ),
        (
            "training.symile_development",
            "load_symile_development_config",
            ["--config", "unused", "--backup-root", "unused-backup"],
        ),
        (
            "training.symile_development_reports",
            "analyze_symile_development",
            ["--development-ids", "a", "b", "c", "d", "e", "f"],
        ),
        (
            "training.symile_campaign",
            "run_symile_campaign",
            [
                "--source-root",
                "unused",
                "--backup-root",
                "unused-backup",
                "--bundle-id",
                "bundle-" + "a" * 64,
                "--cv-assignment-id",
                "cv-assignment-" + "b" * 64,
            ],
        ),
        ("release.cli", "check_repository", ["check", "--root", "unused"]),
    ],
)
def test_cli_failure_does_not_echo_private_exception_text(
    module_name, operation, arguments, monkeypatch, capsys
) -> None:
    module = importlib.import_module(f"beyondcxr.{module_name}")

    def fail(*args, **kwargs):
        raise OSError("synthetic-patient-secret /private/source/secret.dcm token=demo")

    if module_name in {"training.rsna_compare", "training.rsna_seed_summary"}:
        monkeypatch.setattr(module, "validate_execution", lambda *args, **kwargs: object())
        monkeypatch.setattr(module, "validate_freeze", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, operation, fail)
    assert (module.main() if arguments is None else module.main(arguments)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "OSError" in captured.err
    assert "secret" not in captured.err
    assert "/private/" not in captured.err


def test_expected_cli_failure_logs_failed_phase(monkeypatch, capsys) -> None:
    from beyondcxr.data import rsna_manifest

    monkeypatch.setattr(
        rsna_manifest,
        "build_and_write",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("private detail")),
    )
    assert rsna_manifest.main([]) == 1
    captured = capsys.readouterr()
    lines = captured.err.splitlines()
    assert any("event=phase_failed" in line and "phase=rsna_manifest_cli" in line for line in lines)
    assert not any(
        "event=phase_completed" in line and "phase=rsna_manifest_cli" in line for line in lines
    )


@pytest.mark.parametrize(
    ("module_name", "operation", "arguments"),
    [
        (
            "training.symile_campaign",
            "run_symile_campaign",
            [
                "--source-root",
                "unused",
                "--backup-root",
                "unused-backup",
                "--bundle-id",
                "bundle-" + "a" * 64,
                "--cv-assignment-id",
                "cv-assignment-" + "b" * 64,
            ],
        ),
        ("release.cli", "check_repository", ["check", "--root", "unused"]),
    ],
)
def test_broad_exception_clis_sanitize_runtime_errors(
    module_name, operation, arguments, monkeypatch, capsys
) -> None:
    module = importlib.import_module(f"beyondcxr.{module_name}")

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic-patient-secret /private/source/secret.dcm token=demo")

    monkeypatch.setattr(module, operation, fail)
    assert module.main(arguments) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RuntimeError" in captured.err
    assert "secret" not in captured.err
    assert "/private/" not in captured.err


def test_cli_logs_to_stderr_without_contaminating_json_stdout(monkeypatch, capsys) -> None:
    package_id = "model-package-" + "a" * 64
    result = type(
        "Result",
        (),
        {
            "model_package_id": package_id,
            "run_id": "run-test",
            "validation_probability": type("Metrics", (), {"average_precision": 0.4})(),
            "model_path": type("PathValue", (), {"as_posix": lambda self: "model.skops"})(),
            "artifact_directory": type(
                "PathValue", (), {"as_posix": lambda self: "validation-report"}
            )(),
        },
    )()

    def fake_train(config, *, tracking_uri):
        del config, tracking_uri
        log_event(get_operational_logger("test"), "test_progress")
        return result

    monkeypatch.setattr("beyondcxr.training.rsna_train.train_metadata_experiment", fake_train)

    for _ in range(2):
        assert (
            train_main(
                [
                    "--config",
                    "configs/rsna_metadata_logistic.yaml",
                    "--seed",
                    "42",
                    "--bundle-id",
                    "bundle-" + "a" * 64,
                ]
            )
            == 0
        )
        captured = capsys.readouterr()
        assert json.loads(captured.out)["mlflow_run_id"] == "run-test"
        assert "event=test_progress" not in captured.out
        assert captured.err.count("event=test_progress") == 1
        lines = captured.err.splitlines()
        assert any(
            "event=phase_completed" in line and "phase=rsna_training_cli" in line for line in lines
        )
        assert not any(
            "event=phase_failed" in line and "phase=rsna_training_cli" in line for line in lines
        )


def test_configuration_owns_one_dedicated_handler_and_leaves_library_loggers_alone() -> None:
    operational_stream = io.StringIO()
    ordinary_stream = io.StringIO()
    configure_logging("INFO", stream=operational_stream)
    operational_handler = next(
        handler
        for handler in logging.getLogger("beyondcxr.operational").handlers
        if getattr(handler, "_beyondcxr_console_handler", False)
    )
    operational_handler.setFormatter(logging.Formatter("%(message)s"))
    configure_logging("INFO", stream=operational_stream)
    log_event(get_operational_logger("test"), "handler_reconfigured")
    ordinary = logging.getLogger("beyondcxr.some_library")
    ordinary_handler = logging.StreamHandler(ordinary_stream)
    ordinary_level = ordinary.level
    ordinary_propagate = ordinary.propagate
    ordinary_handlers = ordinary.handlers[:]
    try:
        ordinary.addHandler(ordinary_handler)
        ordinary.setLevel(logging.INFO)
        ordinary.propagate = False
        ordinary.info("library message")
    finally:
        ordinary.removeHandler(ordinary_handler)
        ordinary.setLevel(ordinary_level)
        ordinary.propagate = ordinary_propagate

    operational_handlers = [
        handler
        for handler in logging.getLogger("beyondcxr.operational").handlers
        if getattr(handler, "_beyondcxr_console_handler", False)
    ]
    assert len(operational_handlers) == 1
    assert not any(
        getattr(handler, "_beyondcxr_console_handler", False)
        for handler in logging.getLogger("beyondcxr").handlers
    )
    assert ordinary_stream.getvalue() == "library message\n"
    assert ordinary.handlers == ordinary_handlers
    assert ordinary.level == ordinary_level
    assert ordinary.propagate is ordinary_propagate
    assert operational_stream.getvalue().count("event=handler_reconfigured") == 1
    assert operational_stream.getvalue().startswith("timestamp=")


def test_log_event_rebinds_a_closed_configured_stream_to_current_stderr(
    monkeypatch,
) -> None:
    expired_stream = io.StringIO()
    configure_logging("INFO", stream=expired_stream)
    expired_stream.close()
    current_stderr = io.StringIO()
    monkeypatch.setattr("beyondcxr.utils.operational_logging.sys.stderr", current_stderr)

    log_event(get_operational_logger("test"), "stream_lifetime_changed")

    handlers = [
        handler
        for handler in logging.getLogger("beyondcxr.operational").handlers
        if getattr(handler, "_beyondcxr_console_handler", False)
    ]
    assert len(handlers) == 1
    assert handlers[0].stream is current_stderr
    assert current_stderr.getvalue().count("event=stream_lifetime_changed") == 1


def test_operational_events_do_not_propagate_to_root_before_configuration() -> None:
    namespace = logging.getLogger("beyondcxr.operational")
    assert namespace.propagate is False
    configured_handlers = [
        handler
        for handler in namespace.handlers
        if getattr(handler, "_beyondcxr_console_handler", False)
    ]
    root = logging.getLogger()
    root_stream = io.StringIO()
    root_handler = logging.StreamHandler(root_stream)
    root_level = root.level
    root_handlers = root.handlers[:]
    for handler in configured_handlers:
        namespace.removeHandler(handler)
    root.addHandler(root_handler)
    root.setLevel(logging.INFO)
    try:
        log_event(get_operational_logger("preconfiguration"), "silent_event")
    finally:
        root.setLevel(root_level)
        root.removeHandler(root_handler)
        for handler in configured_handlers:
            namespace.addHandler(handler)

    assert root_stream.getvalue() == ""
    assert root.level == root_level
    assert root.handlers == root_handlers


def test_logfmt_is_utc_single_line_stable_and_safely_quoted() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    log_event(
        get_operational_logger("test"),
        "unsafe event\n",
        record_count=3,
        device="cpu",
    )

    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    fields = shlex.split(lines[0])
    assert fields[0].startswith("timestamp=")
    timestamp = datetime.fromisoformat(fields[0].removeprefix("timestamp=").replace("Z", "+00:00"))
    assert timestamp.tzinfo == UTC
    assert fields[1:3] == ["level=INFO", "event=<redacted>"]
    assert 'event="<redacted>"' in lines[0]
    assert fields[3:] == ["device=cpu", "record_count=3"]


def test_formatter_ignores_nonmapping_record_fields() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    logger = get_operational_logger("test")

    logger.info("malformed", extra={"beyondcxr_fields": object()})

    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert shlex.split(lines[0])[1:] == ["level=INFO"]


def test_formatter_drops_unallowed_fields_and_redacts_unsafe_values() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    logger = get_operational_logger("test")

    logger.info(
        "unsafe",
        extra={
            "beyondcxr_fields": {
                "event": "unsafe event\n",
                "artifact": "/private/patient.dcm",
                "unknown": object(),
            }
        },
    )

    output = stream.getvalue()
    lines = output.splitlines()
    assert len(lines) == 1
    assert "/private/patient.dcm" not in output
    assert "unknown" not in output
    assert shlex.split(lines[0])[1:] == ["level=INFO", "event=<redacted>"]


def test_allowlist_validates_value_shapes_and_preserves_aggregate_counts() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    log_event(
        get_operational_logger("test"),
        "privacy_check",
        record_count=10,
        phase="private/path",
        partition_count=["train", "validation"],
        elapsed_s=math.nan,
        average_precision=math.inf,
        device="cpu",
        custom_detail=object(),
    )

    fields = shlex.split(stream.getvalue())
    assert "record_count=10" in fields
    assert "device=cpu" in fields
    assert not any(field.startswith("phase=") for field in fields)
    assert not any(field.startswith("partition_count=") for field in fields)
    assert not any(field.startswith("elapsed_s=") for field in fields)
    assert not any(field.startswith("average_precision=") for field in fields)
    assert not any(field.startswith("custom_detail=") for field in fields)


def test_count_progress_is_rate_limited_and_reports_completion(monkeypatch) -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    timestamps = iter((0.0, 1.0, 31.0, 32.0))
    monkeypatch.setattr(
        "beyondcxr.utils.operational_logging.time.perf_counter", lambda: next(timestamps)
    )
    progress = CountProgress(
        get_operational_logger("test"),
        "source_authentication_progress",
        total=3,
        unit="files",
        count_interval=10,
        time_interval_s=30.0,
    )

    progress.update(1)
    progress.update(2)
    progress.update(3)

    output = stream.getvalue()
    assert output.count("event=source_authentication_progress") == 2
    records = [shlex.split(line) for line in output.splitlines()]
    assert "completed=2" in records[0]
    assert "completed=3" in records[1]
    assert all("total=3" in record and "unit=files" in record for record in records)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total": 0},
        {"total": True},
        {"count_interval": 0},
        {"time_interval_s": math.inf},
    ],
)
def test_count_progress_rejects_invalid_configuration(kwargs) -> None:
    parameters = {"total": 2, "unit": "files", **kwargs}
    with pytest.raises(ValueError):
        CountProgress(get_operational_logger("test"), "progress", **parameters)


def test_count_progress_runtime_updates_are_nonfatal_and_complete_once() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    progress = CountProgress(
        get_operational_logger("test"),
        "source_authentication_progress",
        total=2,
        unit="files",
        count_interval=10,
    )

    progress.update(1)
    progress.update(1)
    progress.update(0)
    progress.update("invalid")
    progress.update(3)
    progress.update(4)

    output = stream.getvalue()
    assert output.count("event=source_authentication_progress") == 1
    fields = shlex.split(output)
    assert "completed=2" in fields
    assert "total=2" in fields


def test_log_level_filters_info_and_preserves_error() -> None:
    stream = io.StringIO()
    configure_logging("WARNING", stream=stream)
    logger = get_operational_logger("test")

    log_event(logger, "info_event")
    log_event(logger, "error_event", level=logging.ERROR)

    output = stream.getvalue()
    assert "event=info_event" not in output
    assert "level=ERROR event=error_event" in output


def test_timed_phase_logs_failure_and_preserves_exception(monkeypatch) -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    timestamps = iter((10.0, 12.5))
    monkeypatch.setattr(
        "beyondcxr.utils.operational_logging.time.perf_counter", lambda: next(timestamps)
    )
    failure = RuntimeError("original failure")

    with pytest.raises(RuntimeError) as raised:
        with timed_phase(get_operational_logger("test"), "model_fitting"):
            raise failure

    assert raised.value is failure
    records = [shlex.split(line) for line in stream.getvalue().splitlines()]
    assert records[0][2:] == ["event=phase_started", "phase=model_fitting"]
    assert records[1][1:] == [
        "level=ERROR",
        "event=phase_failed",
        "elapsed_s=2.5",
        "error_type=RuntimeError",
        "phase=model_fitting",
    ]


def test_direct_operations_configure_logging_and_preserve_enclosing_file(tmp_path, capsys):
    from beyondcxr.utils.operational_logging import observed_operation, operational_output

    @observed_operation("synthetic_direct")
    def operation(*, fail=False):
        log_event(get_operational_logger(__name__), "synthetic_progress", source_path="PRIVATE")
        if fail:
            raise RuntimeError("PRIVATE")
        return "result"

    path = tmp_path / "operation.log"
    with operational_output(path):
        assert operation() == "result"
        with pytest.raises(RuntimeError, match="PRIVATE"):
            operation(fail=True)
    text = path.read_text()
    assert text.count("event=phase_started") == 2
    assert text.count("event=phase_completed") == 1
    assert text.count("event=phase_failed") == 1
    assert "PRIVATE" not in text
    output = capsys.readouterr()
    assert output.out == ""
    assert "phase=synthetic_direct" in output.err


def test_cli_operation_uses_parsed_level_and_leaves_help_to_argparse(capsys):
    from beyondcxr.utils.operational_logging import add_logging_argument, cli_operation

    parser = argparse.ArgumentParser()
    add_logging_argument(parser)
    args = parser.parse_args(["--log-level", "DEBUG"])
    with cli_operation("synthetic_cli", level=args.log_level):
        log_event(get_operational_logger(__name__), "debug_progress", level=logging.DEBUG)
    assert "level=DEBUG event=debug_progress" in capsys.readouterr().err

    with pytest.raises(SystemExit) as help_exit:
        parser.parse_args(["--help"])
    assert help_exit.value.code == 0
    assert "synthetic_cli" not in capsys.readouterr().err
