"""Parse DICOM patient ages and report data-quality outcomes."""

from __future__ import annotations

import re
from dataclasses import dataclass

_DICOM_AGE = re.compile(r"^(\d{3})([DWMY])$")
_RSNA_BARE_AGE = re.compile(r"^\d+$")
_DAYS_PER_YEAR = 365.2425


@dataclass(frozen=True)
class AgeParseResult:
    """Normalized age and its parsing outcome."""

    value_years: float | None
    status: str
    source_format: str
    warning: str | None = None

    @property
    def implausible(self) -> bool:
        return self.value_years is not None and self.value_years > 120.0


def parse_dicom_age(raw_value: str | None) -> AgeParseResult:
    """Parse standard DICOM age strings and RSNA's bare-year source format."""
    if raw_value is None or not raw_value.strip():
        return AgeParseResult(None, "missing", "missing", "age value is missing")

    value = raw_value.strip().upper()
    match = _DICOM_AGE.fullmatch(value)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        divisors = {
            "D": _DAYS_PER_YEAR,
            "W": _DAYS_PER_YEAR / 7.0,
            "M": 12.0,
            "Y": 1.0,
        }
        years = amount / divisors[unit]
        warning = "age exceeds 120 years" if years > 120.0 else None
        return AgeParseResult(years, "parsed", f"dicom_{unit.lower()}", warning)

    if _RSNA_BARE_AGE.fullmatch(value):
        years = float(int(value))
        warning = "age exceeds 120 years" if years > 120.0 else None
        return AgeParseResult(years, "parsed", "rsna_bare_years", warning)

    return AgeParseResult(
        None,
        "malformed",
        "unknown",
        "age does not match DICOM nnn[DWMY] or RSNA bare-year format",
    )
