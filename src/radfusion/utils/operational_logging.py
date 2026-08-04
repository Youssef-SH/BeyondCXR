"""Configure privacy-safe operational logging for RadFusion commands."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, TextIO

_HANDLER_MARKER = "_radfusion_console_handler"
_LOGGER_NAMESPACE = "radfusion.operational"
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:+-]{1,128}$")
_COUNT_FIELDS = frozenset(
    {
        "completed",
        "epoch",
        "file_count",
        "global_epoch",
        "num_workers",
        "no_improvement_count",
        "partition_count",
        "patience",
        "planned_epochs",
        "record_count",
        "report_count",
        "rows",
        "seed",
        "selected_epoch",
        "stage_epoch",
        "prefetch_factor",
        "total",
        "train_count",
        "validation_count",
    }
)
_BOUNDED_NUMERIC_FIELDS = frozenset(
    {"average_precision", "brier_score", "roc_auc", "validation_average_precision"}
)
_NONNEGATIVE_NUMERIC_FIELDS = frozenset(
    {"elapsed_s", "encoder_learning_rate", "head_learning_rate", "training_loss"}
)
_BOOLEAN_FIELDS = frozenset({"selected_best"})
_TOKEN_FIELDS = frozenset(
    {
        "artifact",
        "dataset",
        "device",
        "error_type",
        "experiment",
        "model",
        "operation",
        "partition",
        "phase",
        "run_id",
        "run_kind",
        "selected_stage",
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
        fields = _sanitize_fields(getattr(record, "radfusion_fields", {}))
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
    handler.setFormatter(LogfmtFormatter())
    handler.setLevel(numeric_level)
    root.setLevel(numeric_level)
    root.propagate = False


def get_operational_logger(component: str) -> logging.Logger:
    """Return a logger beneath the dedicated operational namespace."""
    suffix = component.removeprefix("radfusion.")
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
        safe_fields = _sanitize_fields({"event": event, **fields})
        safe_event = str(safe_fields["event"])
        logger.log(level, safe_event, extra={"radfusion_fields": safe_fields})
    except Exception:
        return


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
