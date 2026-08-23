from __future__ import annotations

from pathlib import Path

import pytest

from beyondcxr.utils.privacy import validate_public_reports


@pytest.mark.parametrize(
    "identifier",
    [
        "patient-secret",
        "rsna:sample-secret",
        "image-secret.dcm",
        "stage_2_train_images/image-secret.dcm",
        "123e4567-e89b-42d3-a456-426614174000",
        "1.2.840.10008.5.1.4.1.1.7.12345",
    ],
)
def test_public_report_guard_rejects_source_and_identifier_shapes(
    tmp_path: Path, identifier: str
) -> None:
    report = tmp_path / "report.md"
    report.write_text(f"Aggregate output: {identifier}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_public_reports([report], forbidden_source_values={"patient-secret"})


def test_public_report_guard_accepts_aggregate_metrics(tmp_path: Path) -> None:
    report = tmp_path / "metrics.json"
    report.write_text('{"average_precision": 0.4123}\n', encoding="utf-8")
    validate_public_reports([report], forbidden_source_values={"patient-secret"})
