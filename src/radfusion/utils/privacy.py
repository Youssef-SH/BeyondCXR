"""Validate that public aggregate reports contain no source identifiers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

_UUID_PATTERN = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
_DICOM_UID_PATTERN = re.compile(r"(?<![\d.])(?:\d+\.){4,}\d+(?![\d.])")
_RSNA_SAMPLE_ID_PATTERN = re.compile(r"(?<![\w:])rsna:[^\s,}\]]+")
_SYMILE_SAMPLE_ID_PATTERN = re.compile(r"(?<![\w:])symile:\d+")
_DICOM_PATH_PATTERN = re.compile(r"(?<!\S)[^\s,}\]]+\.dcm\b", re.IGNORECASE)


def validate_public_reports(
    paths: Iterable[str | Path], *, forbidden_source_values: Iterable[str]
) -> None:
    """Reject source identifiers and identifier-shaped values in public text reports."""
    forbidden = sorted({value for value in forbidden_source_values if value}, key=len, reverse=True)
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() not in {".md", ".csv", ".json", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        matches = [value for value in forbidden if value in text]
        if matches:
            raise ValueError(f"Public report contains source identifiers: {path}")
        if _UUID_PATTERN.search(text):
            raise ValueError(f"Public report contains a UUID-shaped identifier: {path}")
        if _DICOM_UID_PATTERN.search(text):
            raise ValueError(f"Public report contains a DICOM UID-shaped value: {path}")
        if _RSNA_SAMPLE_ID_PATTERN.search(text):
            raise ValueError(f"Public report contains an RSNA sample identifier: {path}")
        if _SYMILE_SAMPLE_ID_PATTERN.search(text):
            raise ValueError(f"Public report contains a Symile sample identifier: {path}")
        if _DICOM_PATH_PATTERN.search(text):
            raise ValueError(f"Public report contains a DICOM path: {path}")
