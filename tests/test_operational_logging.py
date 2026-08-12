from __future__ import annotations

import io
import json
import logging
import math
import shlex
from datetime import UTC, datetime

import pytest

from radfusion.training.train import main as train_main
from radfusion.utils.operational_logging import (
    CountProgress,
    configure_logging,
    get_operational_logger,
    log_event,
    timed_phase,
)


def test_cli_logs_to_stderr_without_contaminating_json_stdout(monkeypatch, capsys) -> None:
    result = type(
        "Result",
        (),
        {
            "model_name": "metadata_logistic_regression",
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

    monkeypatch.setattr("radfusion.training.train.train_configured_experiment", fake_train)

    for _ in range(2):
        assert train_main(["--config", "configs/rsna_metadata_logistic.yaml", "--seed", "42"]) == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out)["mlflow_run_id"] == "run-test"
        assert "event=test_progress" not in captured.out
        assert captured.err.count("event=test_progress") == 1


def test_configuration_owns_one_dedicated_handler_and_leaves_library_loggers_alone() -> None:
    operational_stream = io.StringIO()
    ordinary_stream = io.StringIO()
    configure_logging("INFO", stream=operational_stream)
    operational_handler = next(
        handler
        for handler in logging.getLogger("radfusion.operational").handlers
        if getattr(handler, "_radfusion_console_handler", False)
    )
    operational_handler.setFormatter(logging.Formatter("%(message)s"))
    configure_logging("INFO", stream=operational_stream)
    log_event(get_operational_logger("test"), "handler_reconfigured")
    ordinary = logging.getLogger("radfusion.some_library")
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
        for handler in logging.getLogger("radfusion.operational").handlers
        if getattr(handler, "_radfusion_console_handler", False)
    ]
    assert len(operational_handlers) == 1
    assert not any(
        getattr(handler, "_radfusion_console_handler", False)
        for handler in logging.getLogger("radfusion").handlers
    )
    assert ordinary_stream.getvalue() == "library message\n"
    assert ordinary.handlers == ordinary_handlers
    assert ordinary.level == ordinary_level
    assert ordinary.propagate is ordinary_propagate
    assert operational_stream.getvalue().count("event=handler_reconfigured") == 1
    assert operational_stream.getvalue().startswith("timestamp=")


def test_operational_events_do_not_propagate_to_root_before_configuration() -> None:
    namespace = logging.getLogger("radfusion.operational")
    assert namespace.propagate is False
    configured_handlers = [
        handler
        for handler in namespace.handlers
        if getattr(handler, "_radfusion_console_handler", False)
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

    logger.info("malformed", extra={"radfusion_fields": object()})

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
            "radfusion_fields": {
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
        "radfusion.utils.operational_logging.time.perf_counter", lambda: next(timestamps)
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
        "radfusion.utils.operational_logging.time.perf_counter", lambda: next(timestamps)
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
