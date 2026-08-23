import hashlib
from pathlib import Path

import numpy as np
import pytest
from pydicom.data import get_testdata_file

from beyondcxr.data.dicom_loader import read_dicom


def test_read_dicom_returns_normalized_2d_pixels() -> None:
    sample_path = get_testdata_file("CT_small.dcm")

    pixels, record = read_dicom(sample_path)

    assert pixels.ndim == 2
    assert pixels.dtype == np.float32
    assert float(pixels.min()) >= 0.0
    assert float(pixels.max()) <= 1.0
    assert record.rows == pixels.shape[0]
    assert record.columns == pixels.shape[1]


def test_read_dicom_missing_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        read_dicom("does-not-exist.dcm")


def test_read_dicom_authenticates_the_bytes_it_decodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoded = Path(get_testdata_file("CT_small.dcm")).read_bytes()
    sample_path = tmp_path / "sample.dcm"
    sample_path.write_bytes(encoded)

    original_read_bytes = Path.read_bytes

    def authenticated_then_corrupted(path: Path) -> bytes:
        if path != sample_path:
            return original_read_bytes(path)
        sample_path.write_bytes(b"corrupted after authenticated read")
        return encoded

    monkeypatch.setattr(Path, "read_bytes", authenticated_then_corrupted)

    pixels, _ = read_dicom(
        sample_path,
        expected_byte_size=len(encoded),
        expected_sha256=hashlib.sha256(encoded).hexdigest(),
    )

    assert pixels.ndim == 2
    assert original_read_bytes(sample_path) == b"corrupted after authenticated read"
    with pytest.raises(ValueError):
        read_dicom(
            sample_path,
            expected_byte_size=len(encoded),
            expected_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("expected_byte_size", "expected_sha256"),
    [(1, None), (None, "0" * 64)],
)
def test_dicom_authentication_requires_size_and_hash_together(
    expected_byte_size: int | None,
    expected_sha256: str | None,
) -> None:
    with pytest.raises(ValueError):
        read_dicom(
            get_testdata_file("CT_small.dcm"),
            expected_byte_size=expected_byte_size,
            expected_sha256=expected_sha256,
        )
