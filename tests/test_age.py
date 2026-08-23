import pytest

from beyondcxr.data.age import parse_dicom_age


@pytest.mark.parametrize(
    ("raw", "expected", "source_format"),
    [
        ("057Y", 57.0, "dicom_y"),
        ("120M", 10.0, "dicom_m"),
        ("011M", 11.0 / 12.0, "dicom_m"),
        ("010W", 70.0 / 365.2425, "dicom_w"),
        ("030D", 30.0 / 365.2425, "dicom_d"),
        ("57", 57.0, "rsna_bare_years"),
    ],
)
def test_parse_supported_age_formats(raw: str, expected: float, source_format: str) -> None:
    result = parse_dicom_age(raw)

    assert result.value_years == pytest.approx(expected)
    assert result.status == "parsed"
    assert result.source_format == source_format


def test_malformed_age_is_null_with_warning() -> None:
    result = parse_dicom_age("unknown")

    assert result.value_years is None
    assert result.status == "malformed"
    assert result.warning is not None


def test_implausible_age_is_preserved_and_flagged() -> None:
    result = parse_dicom_age("155")

    assert result.value_years == 155.0
    assert result.implausible
    assert result.warning == "age exceeds 120 years"
