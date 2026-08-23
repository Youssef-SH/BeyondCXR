"""Transform canonical decoded chest radiographs for TorchXRayVision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torchxrayvision as xrv
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

CXR_TRANSFORM_POLICY_VERSION = "torchxrayvision-densenet121-res224-v1"
STANDARD_CXR_IMAGE_SIZE = 224


@dataclass(frozen=True)
class CenterCropGeometry:
    """Integer geometry of TorchXRayVision's center-square crop."""

    source_rows: int
    source_columns: int
    crop_size: int
    offset_x: int
    offset_y: int


def center_crop_geometry(rows: int, columns: int) -> CenterCropGeometry:
    """Return the exact integer crop used by XRayCenterCrop."""
    if (
        isinstance(rows, bool)
        or isinstance(columns, bool)
        or not isinstance(rows, int)
        or not isinstance(columns, int)
        or rows <= 0
        or columns <= 0
    ):
        raise ValueError("CXR source dimensions must be positive integers")
    crop = min(rows, columns)
    return CenterCropGeometry(
        source_rows=rows,
        source_columns=columns,
        crop_size=crop,
        offset_x=columns // 2 - crop // 2,
        offset_y=rows // 2 - crop // 2,
    )


class StandardCxrTransform:
    """Apply the fixed DenseNet121 input contract to one canonical CXR."""

    def __init__(
        self,
        *,
        training: bool,
        policy_version: str = CXR_TRANSFORM_POLICY_VERSION,
        image_size: int = STANDARD_CXR_IMAGE_SIZE,
        rotation_degrees: float = 7.0,
        translation_fraction: float = 0.05,
        brightness_jitter: float = 0.05,
        contrast_jitter: float = 0.05,
    ) -> None:
        if not isinstance(training, bool):
            raise TypeError("training must be Boolean")
        if policy_version != CXR_TRANSFORM_POLICY_VERSION:
            raise ValueError("Standard CXR preprocessing policy is unsupported")
        if (
            isinstance(image_size, bool)
            or not isinstance(image_size, int)
            or image_size != STANDARD_CXR_IMAGE_SIZE
        ):
            raise ValueError(f"Standard CXR image_size must be integer {STANDARD_CXR_IMAGE_SIZE}")
        self.training = training
        self.image_size = image_size
        self.rotation_degrees = _bounded(rotation_degrees, "rotation_degrees", 0.0, 180.0)
        self.translation_fraction = _bounded(translation_fraction, "translation_fraction", 0.0, 1.0)
        self.brightness_jitter = _bounded(brightness_jitter, "brightness_jitter", 0.0, 1.0)
        self.contrast_jitter = _bounded(contrast_jitter, "contrast_jitter", 0.0, 1.0)
        self.horizontal_flip = False
        self.vertical_flip = False
        self._crop = xrv.datasets.XRayCenterCrop()
        self._resize = xrv.datasets.XRayResizer(image_size)

    def deterministic_base(self, image: np.ndarray) -> torch.Tensor:
        """Return the deterministic cropped and resized float32 cache value."""
        if not isinstance(image, np.ndarray):
            raise TypeError("Canonical CXR input must be a NumPy array")
        try:
            array = np.asarray(image, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Canonical CXR input must contain float-compatible values") from exc
        if array.ndim != 2 or array.size == 0:
            raise ValueError(f"Canonical CXR must be a non-empty 2D array, received {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Canonical CXR contains non-finite values")
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError("Canonical CXR values must be within [0, 1]")

        cropped = self._crop(array[None, :, :])
        resized = self._resize(cropped)
        output = torch.from_numpy(np.ascontiguousarray(resized)).to(dtype=torch.float32)
        if output.shape != (1, self.image_size, self.image_size):
            raise ValueError(f"Unexpected standard CXR tensor shape: {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise ValueError("Standard CXR deterministic transform produced non-finite values")
        return output.contiguous()

    def from_deterministic_base(
        self,
        image: torch.Tensor,
        *,
        augmentation_seed: int | None = None,
    ) -> torch.Tensor:
        """Apply live augmentation and XRV intensity normalization to a cache value."""
        if (
            not self._valid_base_structure(image)
            or not torch.isfinite(image).all()
            or (float(image.min()) < 0.0 or float(image.max()) > 1.0)
        ):
            raise ValueError(
                "Deterministic CXR base must be finite float32 [1, 224, 224] in [0, 1]"
            )
        return self._transform_base(
            image,
            augmentation_seed=augmentation_seed,
            validate_output=True,
        )

    def from_validated_cache_base(
        self,
        image: torch.Tensor,
        *,
        augmentation_seed: int | None = None,
    ) -> torch.Tensor:
        """Transform one structurally valid base from a fully validated cache."""
        if not self._valid_base_structure(image):
            raise ValueError("Validated CXR cache base must be float32 [1, 224, 224]")
        return self._transform_base(
            image,
            augmentation_seed=augmentation_seed,
            validate_output=False,
        )

    def _transform_base(
        self,
        image: torch.Tensor,
        *,
        augmentation_seed: int | None,
        validate_output: bool,
    ) -> torch.Tensor:
        if augmentation_seed is not None and (
            isinstance(augmentation_seed, bool)
            or not isinstance(augmentation_seed, int)
            or augmentation_seed < 0
        ):
            raise ValueError("augmentation_seed must be a nonnegative integer")
        output = image
        if self.training:
            generator = None
            if augmentation_seed is not None:
                generator = torch.Generator(device="cpu").manual_seed(augmentation_seed)
            output = self._apply_training_augmentation(output, generator)
        normalized = xrv.utils.normalize(output.numpy(), maxval=1.0)
        output = torch.from_numpy(np.ascontiguousarray(normalized)).to(dtype=torch.float32)
        if output.shape != (1, self.image_size, self.image_size):
            raise ValueError(f"Unexpected standard CXR tensor shape: {tuple(output.shape)}")
        if validate_output and not torch.isfinite(output).all():
            raise ValueError("Standard CXR transform produced non-finite values")
        return output.contiguous()

    def _apply_training_augmentation(
        self, image: torch.Tensor, generator: torch.Generator | None
    ) -> torch.Tensor:
        """Sample the fixed augmentation policy from one isolated CPU generator."""
        angle = _uniform(-self.rotation_degrees, self.rotation_degrees, generator)
        maximum_translation = self.translation_fraction * self.image_size
        translation = (
            int(round(_uniform(-maximum_translation, maximum_translation, generator))),
            int(round(_uniform(-maximum_translation, maximum_translation, generator))),
        )
        output = image
        if angle != 0.0 or translation != (0, 0):
            output = vision_functional.affine(
                image,
                angle=angle,
                translate=translation,
                scale=1.0,
                shear=(0.0, 0.0),
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
        order = torch.randperm(4, generator=generator)
        brightness = _uniform(
            max(0.0, 1.0 - self.brightness_jitter),
            1.0 + self.brightness_jitter,
            generator,
        )
        contrast = _uniform(
            max(0.0, 1.0 - self.contrast_jitter),
            1.0 + self.contrast_jitter,
            generator,
        )
        for operation in order:
            if operation == 0 and brightness != 1.0:
                output = vision_functional.adjust_brightness(output, brightness)
            elif operation == 1 and contrast != 1.0:
                output = _adjust_contrast_deterministically(output, contrast)
        return output

    def _valid_base_structure(self, image: object) -> bool:
        return (
            isinstance(image, torch.Tensor)
            and image.dtype == torch.float32
            and image.shape == (1, self.image_size, self.image_size)
        )

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Return a contiguous one-channel tensor in XRV intensity space."""
        return self.from_deterministic_base(self.deterministic_base(image))

    def contract(self) -> dict[str, Any]:
        """Return the serializable transform and input contract."""
        return {
            "policy_version": CXR_TRANSFORM_POLICY_VERSION,
            "input": {
                "type": "numpy.ndarray",
                "shape": "H,W",
                "canonical_range": [0.0, 1.0],
                "finite": True,
            },
            "output": {
                "type": "torch.Tensor",
                "dtype": "torch.float32",
                "shape": [1, self.image_size, self.image_size],
                "channels": {
                    "count": 1,
                    "policy": "single grayscale channel",
                },
            },
            "center_crop": {
                "implementation": "torchxrayvision.XRayCenterCrop",
            },
            "resize": {
                "implementation": "torchxrayvision.XRayResizer",
                "engine": "skimage",
                "interpolation_order": 1,
                "anti_aliasing": "skimage-default",
                "anti_aliasing_sigma": None,
                "mode": "constant",
                "constant_value": 0.0,
                "clip": True,
                "preserve_range": True,
                "target": [self.image_size, self.image_size],
            },
            "normalization": {
                "implementation": "torchxrayvision.utils.normalize",
                "maxval": 1.0,
            },
            "operation_order": [
                "center_crop",
                "resize",
                "training_augmentation_if_enabled",
                "torchxrayvision_normalization",
            ],
            "training_augmentation": {
                "enabled": self.training,
                "rotation_degrees": self.rotation_degrees,
                "translation_fraction": self.translation_fraction,
                "brightness_jitter": self.brightness_jitter,
                "contrast_jitter": self.contrast_jitter,
                "affine": {
                    "interpolation": "bilinear",
                    "fill": 0.0,
                },
                "horizontal_flip": False,
                "vertical_flip": False,
            },
        }


def _bounded(value: object, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{name} must be finite and within [{lower}, {upper}]")
    return number


def _uniform(lower: float, upper: float, generator: torch.Generator | None) -> float:
    return float(torch.empty(1).uniform_(lower, upper, generator=generator).item())


def _adjust_contrast_deterministically(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Apply torchvision's grayscale contrast formula with a stable reduction."""
    mean = float(np.mean(image.numpy(), dtype=np.float64))
    return image.mul(factor).add(mean * (1.0 - factor)).clamp(0.0, 1.0)
