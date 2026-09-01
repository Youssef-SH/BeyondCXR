"""Configure privacy-safe operational logging for BeyondCXR commands."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TextIO, TypeVar

_LOGGING_OWNERSHIP_LOCK = threading.Lock()
_LOGGING_OWNER_THREAD: int | None = None
_LOGGING_OWNER_DEPTH = 0
_CONTEXT_FIELDS = frozenset({"campaign_id", "dataset", "operation", "execution_id"})

_HANDLER_MARKER = "_beyondcxr_console_handler"
_DEFAULT_STREAM_MARKER = "_beyondcxr_uses_default_stream"
_LOGGER_NAMESPACE = "beyondcxr.operational"
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:+-]{1,128}$")
_COUNT_FIELDS = frozenset(
    {
        "completed",
        "epoch",
        "file_count",
        "global_epoch",
        "no_improvement_count",
        "outer_fold",
        "partition_count",
        "patience",
        "planned_epochs",
        "record_count",
        "report_count",
        "repeat_seed",
        "rows",
        "seed",
        "selected_epoch",
        "stage_epoch",
        "total",
        "train_count",
        "validation_count",
    }
)
_BOUNDED_NUMERIC_FIELDS = frozenset(
    {
        "average_precision",
        "brier_score",
        "roc_auc",
        "validation_average_precision",
        "validation_metric",
    }
)
_NONNEGATIVE_NUMERIC_FIELDS = frozenset(
    {
        "elapsed_s",
        "encoder_learning_rate",
        "head_learning_rate",
        "training_elapsed_s",
        "training_batches_per_second",
        "training_loss",
        "training_samples_per_second",
        "validation_elapsed_s",
        "validation_batches_per_second",
        "validation_samples_per_second",
    }
)
_BOOLEAN_FIELDS = frozenset({"selected_best"})
_TOKEN_FIELDS = frozenset(
    {
        "artifact",
        "campaign_id",
        "dataset",
        "device",
        "error_type",
        "experiment",
        "execution_id",
        "family_id",
        "metric_name",
        "operation",
        "package_id",
        "partition",
        "phase",
        "run_id",
        "run_kind",
        "selected_stage",
        "selection_metric",
        "stage",
        "training_run_id",
        "unit",
    }
)
_OPERATIONAL_ROOT = logging.getLogger(_LOGGER_NAMESPACE)
_OPERATIONAL_ROOT.addHandler(logging.NullHandler())
_OPERATIONAL_ROOT.propagate = False


class LogfmtFormatter(logging.Formatter):
    """Render one stable UTC logfmt record without terminal control sequences."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds")
        fields = _sanitize_fields(getattr(record, "beyondcxr_fields", {}))
        rendered = [
            f"timestamp={timestamp.replace('+00:00', 'Z')}",
            f"level={record.levelname}",
        ]
        rendered.extend(f"{key}={_format_value(value)}" for key, value in fields.items())
        return " ".join(rendered)


def add_logging_argument(parser: argparse.ArgumentParser) -> None:
    """Add the common operational log-level option to one CLI parser."""
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console progress level",
    )


def configure_logging(level: str = "INFO", *, stream: TextIO | None = None) -> None:
    """Configure one immediate handler for the operational logger namespace."""
    numeric_level = logging.getLevelNamesMapping().get(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {level!r}")
    root = _OPERATIONAL_ROOT
    destination = stream if stream is not None else sys.stderr
    handlers = [handler for handler in root.handlers if getattr(handler, _HANDLER_MARKER, False)]
    handler = next(
        (candidate for candidate in handlers if isinstance(candidate, logging.StreamHandler)),
        None,
    )
    for candidate in handlers:
        if candidate is not handler:
            root.removeHandler(candidate)
    if handler is None:
        handler = logging.StreamHandler(destination)
        setattr(handler, _HANDLER_MARKER, True)
        root.addHandler(handler)
    elif getattr(handler.stream, "closed", False):
        handler.acquire()
        try:
            handler.stream = destination
        finally:
            handler.release()
    else:
        handler.setStream(destination)
    setattr(handler, _DEFAULT_STREAM_MARKER, stream is None)
    handler.setFormatter(LogfmtFormatter())
    handler.setLevel(numeric_level)
    root.setLevel(numeric_level)
    root.propagate = False


def get_operational_logger(component: str) -> logging.Logger:
    """Return a logger beneath the dedicated operational namespace."""
    suffix = component.removeprefix("beyondcxr.")
    return logging.getLogger(f"{_LOGGER_NAMESPACE}.{suffix}")


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: object,
) -> None:
    """Best-effort emission of one allowlisted structured operational event."""
    try:
        _refresh_operational_streams()
        safe_fields = _sanitize_fields({"event": event, **fields})
        safe_event = str(safe_fields["event"])
        logger.log(level, safe_event, extra={"beyondcxr_fields": safe_fields})
    except Exception:
        return


def _refresh_operational_streams() -> None:
    """Rebind operational handlers whose bound stream is stale."""
    for handler in _OPERATIONAL_ROOT.handlers:
        if not isinstance(handler, logging.StreamHandler) or not getattr(
            handler, _HANDLER_MARKER, False
        ):
            continue
        handler.acquire()
        try:
            if getattr(handler.stream, "closed", False):
                handler.stream = sys.stderr
                setattr(handler, _DEFAULT_STREAM_MARKER, True)
            elif (
                getattr(handler, _DEFAULT_STREAM_MARKER, False) and handler.stream is not sys.stderr
            ):
                handler.setStream(sys.stderr)
        finally:
            handler.release()


@contextmanager
def timed_phase(
    logger: logging.Logger,
    phase: str,
    **fields: object,
) -> Iterator[None]:
    """Log one phase boundary and elapsed time without changing exceptions."""
    started = time.perf_counter()
    log_event(logger, "phase_started", phase=phase, **fields)
    try:
        yield
    except BaseException as exc:
        log_event(
            logger,
            "phase_failed",
            level=logging.ERROR,
            phase=phase,
            error_type=type(exc).__name__,
            elapsed_s=time.perf_counter() - started,
            **fields,
        )
        raise
    log_event(
        logger,
        "phase_completed",
        phase=phase,
        elapsed_s=time.perf_counter() - started,
        **fields,
    )


class CountProgress:
    """Emit low-frequency count and time based progress for one long operation."""

    def __init__(
        self,
        logger: logging.Logger,
        event: str,
        *,
        total: int,
        unit: str,
        count_interval: int = 1_000,
        time_interval_s: float = 30.0,
        fields: Mapping[str, object] | None = None,
    ) -> None:
        self._logger = logger
        self._event = event
        self._total = _positive_int(total, "total")
        self._unit = unit
        self._count_interval = _positive_int(count_interval, "count_interval")
        self._time_interval_s = _positive_number(time_interval_s, "time_interval_s")
        self._fields = dict(fields or {})
        self._started = time.perf_counter()
        self._last_emitted = self._started
        self._last_count = 0
        self._last_seen = 0
        self._completed = False

    def update(self, completed: int) -> None:
        """Emit progress when either configured interval has elapsed."""
        try:
            if self._completed or isinstance(completed, bool) or not isinstance(completed, int):
                return
            if completed <= 0 or completed <= self._last_seen:
                return
            self._last_seen = completed
            now = time.perf_counter()
            reached_completion = completed >= self._total
            reported_completed = min(completed, self._total)
            if (
                reached_completion
                or completed - self._last_count >= self._count_interval
                or now - self._last_emitted >= self._time_interval_s
            ):
                log_event(
                    self._logger,
                    self._event,
                    completed=reported_completed,
                    total=self._total,
                    unit=self._unit,
                    elapsed_s=now - self._started,
                    **self._fields,
                )
                self._last_count = reported_completed
                self._last_emitted = now
                self._completed = reached_completion
        except Exception:
            return


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else '"<redacted>"'
    if isinstance(value, str):
        return value if _SAFE_TOKEN.fullmatch(value) else json.dumps(value, ensure_ascii=True)
    return '"<redacted>"'


def _sanitize_fields(fields: object) -> dict[str, object]:
    """Return safe deterministic operational fields from an arbitrary record extra."""
    if not isinstance(fields, Mapping):
        return {}
    try:
        items = [(key, value) for key, value in fields.items() if isinstance(key, str)]
        safe: dict[str, object] = {}
        event_values = [value for key, value in items if key == "event"]
        if event_values:
            event = event_values[0]
            safe["event"] = (
                event if isinstance(event, str) and _SAFE_TOKEN.fullmatch(event) else "<redacted>"
            )
        for key, value in sorted(items, key=lambda item: item[0]):
            if key == "event":
                continue
            if (safe_value := _safe_field_value(key, value)) is not None:
                safe[key] = safe_value
        return safe
    except Exception:
        return {}


def _safe_field_value(key: str, value: object) -> bool | float | int | str | None:
    if key in _COUNT_FIELDS:
        return (
            value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
        )
    if key in _BOUNDED_NUMERIC_FIELDS:
        if isinstance(value, int | float) and not isinstance(value, bool):
            number = float(value)
            return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None
        return None
    if key in _NONNEGATIVE_NUMERIC_FIELDS:
        if isinstance(value, int | float) and not isinstance(value, bool):
            number = float(value)
            return number if math.isfinite(number) and number >= 0.0 else None
        return None
    if key in _BOOLEAN_FIELDS:
        return value if isinstance(value, bool) else None
    if key in _TOKEN_FIELDS:
        return value if isinstance(value, str) and _SAFE_TOKEN.fullmatch(value) else None
    return None


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


@contextmanager
def logging_ownership() -> Iterator[None]:
    """Own process-global operational handlers in one explicitly tracked thread."""
    global _LOGGING_OWNER_DEPTH, _LOGGING_OWNER_THREAD
    thread_id = threading.get_ident()
    with _LOGGING_OWNERSHIP_LOCK:
        if _LOGGING_OWNER_THREAD not in {None, thread_id}:
            raise RuntimeError("Operational logging sessions cannot overlap across threads")
        _LOGGING_OWNER_THREAD = thread_id
        _LOGGING_OWNER_DEPTH += 1
    try:
        yield
    finally:
        with _LOGGING_OWNERSHIP_LOCK:
            if _LOGGING_OWNER_THREAD != thread_id or _LOGGING_OWNER_DEPTH <= 0:
                raise RuntimeError("Operational logging ownership state is invalid")
            _LOGGING_OWNER_DEPTH -= 1
            if _LOGGING_OWNER_DEPTH == 0:
                _LOGGING_OWNER_THREAD = None


def _logging_owner_thread() -> int | None:
    with _LOGGING_OWNERSHIP_LOCK:
        return _LOGGING_OWNER_THREAD


@contextmanager
def configured_logging(
    *, stream: TextIO, context: Mapping[str, object], level: str = "INFO"
) -> Iterator[None]:
    """Temporarily own operational handlers without changing caller logging state."""
    if any(
        key not in _CONTEXT_FIELDS or _safe_field_value(key, value) is None
        for key, value in context.items()
    ):
        raise ValueError("Invalid operational session context")
    with logging_ownership():
        root = _OPERATIONAL_ROOT
        previous = tuple(root.handlers)
        state = (root.level, root.propagate, root.disabled)
        campaign_owner = threading.get_ident() if "campaign_id" in context else None

        class ContextFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                if campaign_owner is not None and record.thread != campaign_owner:
                    return False
                fields = _sanitize_fields(getattr(record, "beyondcxr_fields", {}))
                if campaign_owner is not None:
                    fields = {
                        key: value for key, value in fields.items() if key not in _CONTEXT_FIELDS
                    }
                protected = {key: value for key, value in context.items() if key in _CONTEXT_FIELDS}
                record.beyondcxr_fields = {**fields, **protected}
                return True

        for handler in previous:
            root.removeHandler(handler)
        try:
            root.disabled = False
            configure_logging(level, stream=stream)
            for handler in root.handlers:
                handler.addFilter(ContextFilter())
            yield
        finally:
            for handler in tuple(root.handlers):
                root.removeHandler(handler)
                handler.close()
            for handler in previous:
                root.addHandler(handler)
            root.setLevel(state[0])
            root.propagate, root.disabled = state[1:]


def parse_operational_record(line: str) -> dict[str, str]:
    """Parse one record that the canonical operational writer could emit."""
    result: dict[str, str] = {}
    position = 0
    decoder = json.JSONDecoder()
    if not line or any(character in line for character in "\r\n"):
        raise ValueError("Operational log record must be one nonempty line")
    while position < len(line):
        key = re.match(r"([a-z_]+)=", line[position:])
        if key is None or key[1] in result:
            raise ValueError("Operational log key is malformed or duplicated")
        position += key.end()
        if position < len(line) and line[position] == '"':
            value, consumed = decoder.raw_decode(line[position:])
            if not isinstance(value, str):
                raise ValueError("Operational quoted value must be a string")
            position += consumed
        else:
            token = re.match(r"[^\s\"=]+", line[position:])
            if token is None:
                raise ValueError("Operational log value is malformed")
            value = token[0]
            position += token.end()
        result[key[1]] = value
        if position < len(line):
            if line[position] != " " or position + 1 == len(line):
                raise ValueError("Operational log separator is malformed")
            position += 1
    if (
        not {"timestamp", "level", "event"} <= result.keys()
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", result["timestamp"])
        or result["level"] not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        or not _SAFE_TOKEN.fullmatch(result["event"])
        or not _writer_fields_are_valid(result)
    ):
        raise ValueError("Operational log envelope is invalid")
    datetime.fromisoformat(result["timestamp"])
    return result


def _writer_fields_are_valid(record: Mapping[str, str]) -> bool:
    envelope = {"timestamp", "level", "event"}
    allowed = (
        envelope
        | _COUNT_FIELDS
        | _BOUNDED_NUMERIC_FIELDS
        | _NONNEGATIVE_NUMERIC_FIELDS
        | _BOOLEAN_FIELDS
        | _TOKEN_FIELDS
    )
    if not set(record) <= allowed:
        return False
    for key, value in record.items():
        if key in envelope:
            continue
        if key in _COUNT_FIELDS:
            if not re.fullmatch(r"0|[1-9]\d*", value):
                return False
            typed: object = int(value)
        elif key in _BOUNDED_NUMERIC_FIELDS | _NONNEGATIVE_NUMERIC_FIELDS:
            try:
                typed = float(value)
            except ValueError:
                return False
        elif key in _BOOLEAN_FIELDS:
            if value not in {"true", "false"}:
                return False
            typed = value == "true"
        else:
            typed = value
        safe = _safe_field_value(key, typed)
        if safe is None or _format_value(safe) != value:
            return False
    return True


class TeeStream:
    """Immediately flush safe operational records to file and console."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, value: str) -> int:
        for stream in self._streams:
            if not stream.closed:
                stream.write(value)
                stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self._streams:
            if not stream.closed:
                stream.flush()


@contextmanager
def operational_output(log_path: Path | None = None, *, level: str = "INFO") -> Iterator[None]:
    """Configure terminal diagnostics, optionally teeing a new operational log."""
    if log_path is not None:
        with logging_ownership():
            with log_path.open("x", encoding="utf-8", buffering=1) as stream:
                with configured_logging(
                    stream=TeeStream(sys.stderr, stream), context={}, level=level
                ):
                    yield
    else:
        owner = _logging_owner_thread()
        if owner == threading.get_ident():
            yield
        elif owner is not None:
            raise RuntimeError("Operational logging sessions cannot overlap across threads")
        else:
            with configured_logging(stream=sys.stderr, context={}, level=level):
                yield


_P = ParamSpec("_P")
_R = TypeVar("_R")


def observed_operation(phase: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Observe a direct entry path without replacing an enclosing campaign's handlers."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def observed(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with (
                operational_output(),
                timed_phase(get_operational_logger(function.__module__), phase),
            ):
                return function(*args, **kwargs)

        return observed

    return decorate


@contextmanager
def cli_operation(phase: str, *, level: str) -> Iterator[None]:
    """Observe a successfully parsed CLI invocation at its requested level."""
    with (
        operational_output(level=level),
        timed_phase(get_operational_logger("cli"), phase),
    ):
        yield
