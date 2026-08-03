"""Transform canonical decoded chest radiographs for TorchXRayVision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torchxrayvision as xrv
from torchvision.transforms import ColorJitter, InterpolationMode, RandomAffine

CXR_TRANSFORM_POLICY_VERSION = "torchxrayvision-densenet121-res224-v1"


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
        image_size: int = 224,
        rotation_degrees: float = 7.0,
        translation_fraction: float = 0.05,
        brightness_jitter: float = 0.05,
        contrast_jitter: float = 0.05,
    ) -> None:
        if not isinstance(training, bool):
            raise TypeError("training must be Boolean")
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size != 224:
            raise ValueError("Standard CXR image_size must be integer 224")
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
        self._affine = RandomAffine(
            degrees=self.rotation_degrees,
            translate=(self.translation_fraction, self.translation_fraction),
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        self._jitter = ColorJitter(
            brightness=self.brightness_jitter,
            contrast=self.contrast_jitter,
        )

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Return a contiguous one-channel tensor in XRV intensity space."""
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
        tensor = torch.from_numpy(np.ascontiguousarray(resized)).to(dtype=torch.float32)
        if self.training:
            tensor = self._jitter(self._affine(tensor))
        normalized = xrv.utils.normalize(tensor.numpy(), maxval=1.0)
        output = torch.from_numpy(np.ascontiguousarray(normalized)).to(dtype=torch.float32)
        if output.shape != (1, self.image_size, self.image_size):
            raise ValueError(f"Unexpected standard CXR tensor shape: {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise ValueError("Standard CXR transform produced non-finite values")
        return output.contiguous()

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
