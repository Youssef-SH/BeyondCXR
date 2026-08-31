"""Operational attempt logs, independent of scientific lifecycle authority."""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.operational_validation import validate_directory_chain
from beyondcxr.training.preservation import PreservationMember
from beyondcxr.utils.operational_logging import (
    TeeStream,
    configured_logging,
    get_operational_logger,
    log_event,
    logging_ownership,
    parse_operational_record,
)
from beyondcxr.utils.publication import publish_bytes_no_replace, validate_path_component
from beyondcxr.utils.semantic_identity import canonical_json_bytes

_LOGGER = get_operational_logger(__name__)
_EXPORT_OPERATIONS = {"rsna": "campaign", "symile": "test"}
_CAMPAIGN_CONTEXTS = frozenset((*_EXPORT_OPERATIONS.items(), ("symile", "development")))


@dataclass(frozen=True)
class CampaignSession:
    """One invocation's diagnostics; possession confers no scientific permission."""

    attempt_id: str
    log_path: Path
    _context: dict[str, object] = field(repr=False)

    def bind_execution(self, execution_id: str) -> None:
        """Record an explicit operational association supplied by the orchestrator."""
        validate_path_component(execution_id, "Campaign execution identity")
        if not re.fullmatch(r"[A-Za-z0-9_.:+-]{1,128}", execution_id):
            raise ManifestBuildError("Campaign execution identity is invalid")
        document = {**self._context, "execution_id": execution_id}
        path = self.log_path.with_name("session.json")
        encoded = canonical_json_bytes(document)
        created = publish_bytes_no_replace(path, encoded)
        if not created and path.read_bytes() != encoded:
            raise ManifestBuildError("Operational attempt cannot change execution association")
        self._context["execution_id"] = execution_id
        if not created:
            return
        log_event(_LOGGER, "campaign_execution_bound")


@contextmanager
def campaign_session(
    *, repository_root: Path, report_root: Path, study: str, operation: str
) -> Iterator[CampaignSession]:
    """Own one unique durable log and restore the caller's handlers on every exit."""
    if (study, operation) not in _CAMPAIGN_CONTEXTS:
        raise ValueError("Unknown campaign operational context")
    with logging_ownership():
        root = validate_directory_chain(
            repository_root, report_root / "campaigns", "Campaign log destination", create=True
        )
        attempt_id = "campaign-" + uuid4().hex
        directory = root / attempt_id
        directory.mkdir(exist_ok=False)
        path = directory / "execution.log"
        context: dict[str, object] = {
            "campaign_id": attempt_id,
            "dataset": study,
            "operation": operation,
        }
        session = CampaignSession(attempt_id, path, context)
        with path.open("x", encoding="utf-8", buffering=1) as stream:
            with configured_logging(stream=TeeStream(sys.stderr, stream), context=context):
                started = time.perf_counter()
                log_event(_LOGGER, "campaign_started", phase="preparation")
                try:
                    yield session
                except BaseException as exc:
                    log_event(
                        _LOGGER,
                        "campaign_failed",
                        phase="failed",
                        level=logging.ERROR,
                        error_type=type(exc).__name__,
                        elapsed_s=time.perf_counter() - started,
                    )
                    raise
                else:
                    log_event(
                        _LOGGER,
                        "campaign_succeeded",
                        phase="complete",
                        elapsed_s=time.perf_counter() - started,
                    )


def campaign_log_members(
    log_path: Path, *, study: str, execution_id: str
) -> tuple[PreservationMember, ...]:
    """Project exactly the exporting attempt, never discover historical logs by recency."""
    operation = _EXPORT_OPERATIONS.get(study)
    if operation is None:
        raise ManifestBuildError("Campaign export study is unsupported")
    metadata = log_path.with_name("session.json")
    if any(path.is_symlink() or not path.is_file() for path in (log_path, metadata)):
        raise ManifestBuildError("Campaign operational evidence is missing or invalid")
    try:
        document = json.loads(metadata.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise ManifestBuildError("Campaign operational association is malformed") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"campaign_id", "dataset", "operation", "execution_id"}
        or document["dataset"] != study
        or document["execution_id"] != execution_id
        or document["operation"] != operation
        or not isinstance(document["campaign_id"], str)
        or not re.fullmatch(r"campaign-[0-9a-f]{32}", document["campaign_id"])
    ):
        raise ManifestBuildError("Campaign operational association differs from execution")
    try:
        records = [
            parse_operational_record(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
    except (ValueError, UnicodeError) as exc:
        raise ManifestBuildError("Campaign operational log is malformed") from exc
    required = ("campaign_started", "campaign_execution_bound", "campaign_ready_for_export")
    positions: list[int] = []
    for event in required:
        matches = [index for index, record in enumerate(records) if record["event"] == event]
        if len(matches) != 1:
            raise ManifestBuildError(
                "Campaign operational lifecycle event is missing or duplicated"
            )
        positions.append(matches[0])
    if positions[0] != 0 or positions != sorted(positions):
        raise ManifestBuildError("Campaign operational lifecycle ordering is invalid")
    if positions[2] != len(records) - 1:
        raise ManifestBuildError("Campaign operational snapshot extends beyond ready-for-export")
    if records[0].get("phase") != "preparation":
        raise ManifestBuildError("Campaign operational preparation event is invalid")
    for index, record in enumerate(records):
        if (
            any(record.get(key) != document[key] for key in ("campaign_id", "dataset", "operation"))
            or (index < positions[1] and "execution_id" in record)
            or (index >= positions[1] and record.get("execution_id") != execution_id)
            or record["event"] in {"campaign_failed", "archive_created", "campaign_succeeded"}
        ):
            raise ManifestBuildError("Campaign operational log differs from attempt association")
    root = Path("reports") / study / "campaigns" / "export"
    return tuple(
        PreservationMember(path, root / path.name, "operational_evidence")
        for path in (metadata, log_path)
    )
