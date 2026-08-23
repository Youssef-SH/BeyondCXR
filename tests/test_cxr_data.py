from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torchxrayvision as xrv
from torchvision.transforms import ColorJitter, InterpolationMode, RandomAffine

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.training.neural import seed_neural_runtime


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


def test_validation_pixels_match_authoritative_torchxrayvision_pipeline() -> None:
    image = np.arange(41 * 60, dtype=np.float32).reshape(41, 60)
    image /= float(image.max())
    expected = xrv.utils.normalize(
        xrv.datasets.XRayResizer(224)(xrv.datasets.XRayCenterCrop()(image[None, :, :])),
        maxval=1.0,
    )

    actual = StandardCxrTransform(training=False)(image)

    assert np.array_equal(actual.numpy(), expected)


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


@pytest.mark.parametrize("seed", [0, 17, 42, 2026])
def test_deterministic_augmentation_matches_torchvision_reference_within_float_quantum(
    seed: int,
) -> None:
    transform = StandardCxrTransform(training=True)
    bases = (
        torch.linspace(0.0, 1.0, 224 * 224, dtype=torch.float32).reshape(1, 224, 224),
        torch.arange(224 * 224, dtype=torch.float32).remainder(97).div(96).reshape(1, 224, 224),
    )
    for base in bases:
        torch.manual_seed(seed)
        augmented = ColorJitter(brightness=0.05, contrast=0.05)(
            RandomAffine(
                degrees=7.0,
                translate=(0.05, 0.05),
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )(base)
        )
        expected = torch.from_numpy(
            np.ascontiguousarray(xrv.utils.normalize(augmented.numpy(), maxval=1.0))
        ).to(dtype=torch.float32)

        actual = transform.from_deterministic_base(base, augmentation_seed=seed)

        # Deterministic contrast stays within one XRV-scale float quantum of torchvision.
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=2**-14)


def test_zero_augmentation_boundary_preserves_exact_normalization() -> None:
    base = torch.linspace(0.0, 1.0, 224 * 224, dtype=torch.float32).reshape(1, 224, 224)
    transform = StandardCxrTransform(
        training=True,
        rotation_degrees=0.0,
        translation_fraction=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )
    expected = torch.from_numpy(
        np.ascontiguousarray(xrv.utils.normalize(base.numpy(), maxval=1.0))
    ).to(dtype=torch.float32)

    assert torch.equal(transform.from_deterministic_base(base, augmentation_seed=0), expected)


def test_explicit_augmentation_is_isolated_from_all_caller_rng(monkeypatch) -> None:
    transform = StandardCxrTransform(training=True)
    base = torch.linspace(0.0, 1.0, 224 * 224, dtype=torch.float32).reshape(1, 224, 224)
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    monkeypatch.setattr(
        torch,
        "manual_seed",
        lambda seed: (_ for _ in ()).throw(AssertionError("global seed mutation")),
    )
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: (_ for _ in ()).throw(AssertionError("CUDA seed mutation")),
    )

    first = transform.from_validated_cache_base(base, augmentation_seed=17)
    second = transform.from_validated_cache_base(base, augmentation_seed=17)

    assert torch.equal(first, second)
    assert random.getstate() == python_state
    assert np.random.get_state()[0] == numpy_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    if cuda_state is not None:
        assert all(
            torch.equal(before, after)
            for before, after in zip(cuda_state, torch.cuda.get_rng_state_all(), strict=True)
        )
