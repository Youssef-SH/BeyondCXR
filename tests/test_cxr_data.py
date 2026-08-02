from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.dicom_loader import DicomRecord
from radfusion.data.rsna_source import ManifestBuildError
from radfusion.training.datasets import RsnaImageDataset
from radfusion.training.neural import seed_neural_runtime

_COLUMNS = ("sample_id", "patient_id", "image_path", "split_name", "target")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("rsna:a", "patient-a", "stage_2_train_images/a.dcm", "train", 0),
            ("rsna:b", "patient-b", "stage_2_train_images/b.dcm", "train", 1),
        ],
        columns=_COLUMNS,
    )


def test_validation_transform_is_deterministic_finite_and_serializable() -> None:
    image = np.linspace(0.0, 1.0, 40 * 60, dtype=np.float32).reshape(40, 60)
    transform = StandardCxrTransform(training=False)

    first = transform(image)
    second = transform(image)

    assert torch.equal(first, second)
    assert first.shape == (1, 224, 224)
    assert first.dtype == torch.float32
    assert first.is_contiguous()
    assert torch.isfinite(first).all()
    contract = transform.contract()
    assert contract["input"]["canonical_range"] == [0.0, 1.0]
    assert contract["output"]["shape"] == [1, 224, 224]
    assert contract["output"]["dtype"] == "torch.float32"
    assert contract["output"]["channels"] == {
        "count": 1,
        "policy": "single grayscale channel",
    }
    assert contract["resize"]["target"] == [224, 224]
    assert contract["normalization"] == {
        "implementation": "torchxrayvision.utils.normalize",
        "maxval": 1.0,
    }
    assert contract["operation_order"] == [
        "center_crop",
        "resize",
        "training_augmentation_if_enabled",
        "torchxrayvision_normalization",
    ]
    assert contract["training_augmentation"]["enabled"] is False
    assert contract["training_augmentation"]["affine"] == {
        "interpolation": "bilinear",
        "fill": 0.0,
    }


@pytest.mark.parametrize(
    ("image", "exception"),
    [
        ([0.0, 1.0], TypeError),
        (np.empty((0, 2), dtype=np.float32), ValueError),
        (np.zeros(2, dtype=np.float32), ValueError),
        (np.zeros((1, 2, 3), dtype=np.float32), ValueError),
        (np.array([["invalid"]], dtype=object), ValueError),
        (np.array([[0.0, np.nan]], dtype=np.float32), ValueError),
        (np.array([[0.0, np.inf]], dtype=np.float32), ValueError),
        (np.array([[0.0, -np.inf]], dtype=np.float32), ValueError),
        (np.array([[-0.01, 1.0]], dtype=np.float32), ValueError),
        (np.array([[0.0, 1.01]], dtype=np.float32), ValueError),
    ],
)
def test_transform_rejects_invalid_canonical_arrays(
    image: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        StandardCxrTransform(training=False)(image)  # type: ignore[arg-type]


def test_training_transform_is_seeded_and_never_flips() -> None:
    image = np.zeros((48, 64), dtype=np.float32)
    image[4:20, 7:18] = 1.0
    transform = StandardCxrTransform(training=True)

    seed_neural_runtime(42)
    first = transform(image)
    seed_neural_runtime(42)
    second = transform(image)

    assert torch.equal(first, second)
    assert transform.horizontal_flip is False
    assert transform.vertical_flip is False
    contract = transform.contract()
    assert contract["training_augmentation"]["enabled"] is True
    assert contract["training_augmentation"]["horizontal_flip"] is False
    assert contract["training_augmentation"]["vertical_flip"] is False
    assert contract["training_augmentation"]["affine"] == {
        "interpolation": "bilinear",
        "fill": 0.0,
    }
    assert (
        StandardCxrTransform(training=False).contract()["training_augmentation"]["enabled"] is False
    )


def test_image_dataset_validates_without_decoding_and_decodes_lazily(tmp_path: Path) -> None:
    calls: list[Path] = []
    transformed: list[np.ndarray] = []

    def decoder(path: str | Path) -> tuple[np.ndarray, DicomRecord]:
        calls.append(Path(path))
        pixels = np.ones((8, 8), dtype=np.float32)
        return pixels, DicomRecord(
            path=str(path),
            patient_id=f"patient-{Path(path).stem}",
            patient_age=None,
            patient_sex=None,
            view_position=None,
            rows=8,
            columns=8,
            photometric_interpretation="MONOCHROME2",
        )

    def transform(image: np.ndarray) -> torch.Tensor:
        transformed.append(image)
        return torch.ones((1, 224, 224), dtype=torch.float32)

    dataset = RsnaImageDataset(
        _frame(),
        dataset_root=tmp_path,
        partition="train",
        transform=transform,
        decoder=decoder,
    )

    assert len(dataset) == 2
    assert calls == []
    sample = dataset[0]
    assert calls == [tmp_path / "stage_2_train_images/a.dcm"]
    assert len(transformed) == 1
    assert sample["sample_id"] == "rsna:a"
    assert sample["patient_id"] == "patient-a"
    assert sample["target"].dtype == torch.float32
    assert sample["target"].shape == ()
    assert sample["image"].shape == (1, 224, 224)
    second = dataset[1]
    assert calls == [
        tmp_path / "stage_2_train_images/a.dcm",
        tmp_path / "stage_2_train_images/b.dcm",
    ]
    assert len(transformed) == 2
    assert second["sample_id"] == "rsna:b"
    assert second["patient_id"] == "patient-b"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda frame: frame.drop(columns="image_path"),
        lambda frame: frame.assign(unexpected=1),
        lambda frame: frame.loc[:, list(reversed(_COLUMNS))],
        lambda frame: frame.iloc[0:0],
        lambda frame: frame.assign(sample_id=[1, "rsna:b"]),
        lambda frame: frame.assign(sample_id=["", "rsna:b"]),
        lambda frame: frame.iloc[::-1].reset_index(drop=True),
        lambda frame: frame.assign(sample_id=["rsna:a", "rsna:a"]),
        lambda frame: frame.assign(target=[0, 2]),
        lambda frame: frame.assign(target=[False, 1]),
        lambda frame: frame.assign(target=[1.0, 1]),
        lambda frame: frame.assign(target=[0.0, 1]),
        lambda frame: frame.assign(target=[0.5, 1]),
        lambda frame: frame.assign(patient_id=["patient-a", ""]),
        lambda frame: frame.assign(patient_id=["patient-a", 2]),
        lambda frame: frame.assign(image_path=["../a.dcm", "b.dcm"]),
        lambda frame: frame.assign(image_path=["/a.dcm", "b.dcm"]),
        lambda frame: frame.assign(image_path=[r"images\a.dcm", "b.dcm"]),
        lambda frame: frame.assign(image_path=["images/./a.dcm", "b.dcm"]),
        lambda frame: frame.assign(image_path=["images//a.dcm", "b.dcm"]),
        lambda frame: frame.assign(split_name=["train", "test"]),
    ],
)
def test_image_dataset_rejects_invalid_rows_before_pixel_access(
    tmp_path: Path,
    mutation,
) -> None:
    calls = 0

    def decoder(path):
        nonlocal calls
        calls += 1
        raise AssertionError(path)

    with pytest.raises(ManifestBuildError):
        RsnaImageDataset(
            mutation(_frame()),
            dataset_root=tmp_path,
            partition="train",
            transform=lambda image: torch.from_numpy(image),
            decoder=decoder,
        )
    assert calls == 0


def test_image_dataset_rejects_unknown_partition_before_rows(tmp_path: Path) -> None:
    with pytest.raises(ManifestBuildError):
        RsnaImageDataset(
            _frame(),
            dataset_root=tmp_path,
            partition="holdout",
            transform=lambda image: torch.from_numpy(image),
        )


def test_image_dataset_rejects_decoded_patient_mismatch(tmp_path: Path) -> None:
    def decoder(path: str | Path) -> tuple[np.ndarray, DicomRecord]:
        return np.ones((8, 8), dtype=np.float32), DicomRecord(
            path=str(path),
            patient_id="different-patient",
            patient_age=None,
            patient_sex=None,
            view_position=None,
            rows=8,
            columns=8,
            photometric_interpretation="MONOCHROME2",
        )

    dataset = RsnaImageDataset(
        _frame(),
        dataset_root=tmp_path,
        partition="train",
        transform=lambda _: torch.zeros((1, 224, 224), dtype=torch.float32),
        decoder=decoder,
    )

    with pytest.raises(ManifestBuildError):
        dataset[0]


@pytest.mark.parametrize(
    "transformed",
    [
        np.zeros((1, 224, 224), dtype=np.float32),
        torch.zeros((1, 224, 224), dtype=torch.float64),
        torch.zeros((224, 224), dtype=torch.float32),
        torch.full((1, 224, 224), float("inf"), dtype=torch.float32),
    ],
)
def test_image_dataset_rejects_invalid_transform_output(
    tmp_path: Path,
    transformed: object,
) -> None:
    def decoder(path: str | Path) -> tuple[np.ndarray, DicomRecord]:
        return np.ones((8, 8), dtype=np.float32), DicomRecord(
            path=str(path),
            patient_id="patient-a",
            patient_age=None,
            patient_sex=None,
            view_position=None,
            rows=8,
            columns=8,
            photometric_interpretation="MONOCHROME2",
        )

    dataset = RsnaImageDataset(
        _frame().iloc[[0]].copy(),
        dataset_root=tmp_path,
        partition="train",
        transform=lambda _: transformed,  # type: ignore[return-value]
        decoder=decoder,
    )

    with pytest.raises(ManifestBuildError):
        dataset[0]
