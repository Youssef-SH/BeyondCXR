"""Strict, stateless serving inputs aligned with frozen Symile evaluation."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.symile_preprocess import LAB_FEATURE_COLUMNS
from beyondcxr.serving.authority import LAB_KEYS

MAX_JPEG_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_LABS_JSON_CHARACTERS = 100_000
SERVING_CANONICAL_SIZE = 320


class ServingInputError(ValueError):
    """One privacy-safe invalid-request category."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


@dataclass(frozen=True)
class ValidatedServingInput:
    image: torch.Tensor
    laboratory_frame: pd.DataFrame
    missing_labs: tuple[str, ...]


def decode_jpeg_once(content: bytes) -> np.ndarray:
    """Decode exactly one bounded JPEG into a finite grayscale source array."""
    if not isinstance(content, bytes) or not content or len(content) > MAX_JPEG_BYTES:
        raise ServingInputError("invalid_image_content")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "JPEG" or getattr(image, "n_frames", 1) != 1:
                raise ServingInputError("invalid_image_content")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ServingInputError("invalid_image_content")
            image.load()
            grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        if isinstance(exc, ServingInputError):
            raise
        raise ServingInputError("invalid_image_content") from exc
    if grayscale.ndim != 2 or not grayscale.size:
        raise ServingInputError("invalid_image_content")
    return grayscale


def serving_canonical_image(grayscale: np.ndarray) -> np.ndarray:
    """Map decoded grayscale into the frozen M7 canonical spatial representation."""
    if (
        not isinstance(grayscale, np.ndarray)
        or grayscale.dtype != np.uint8
        or grayscale.ndim != 2
        or not grayscale.size
    ):
        raise ServingInputError("invalid_image_content")
    height, width = grayscale.shape
    if width <= height:
        resized_width = SERVING_CANONICAL_SIZE
        resized_height = int(SERVING_CANONICAL_SIZE * height / width)
    else:
        resized_height = SERVING_CANONICAL_SIZE
        resized_width = int(SERVING_CANONICAL_SIZE * width / height)
    if resized_width * resized_height > MAX_IMAGE_PIXELS:
        raise ServingInputError("invalid_image_content")
    image = Image.fromarray(grayscale, mode="L")
    resized = image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    left = int(round((resized_width - SERVING_CANONICAL_SIZE) / 2.0))
    top = int(round((resized_height - SERVING_CANONICAL_SIZE) / 2.0))
    cropped = resized.crop(
        (
            left,
            top,
            left + SERVING_CANONICAL_SIZE,
            top + SERVING_CANONICAL_SIZE,
        )
    )
    array = np.asarray(cropped, dtype=np.float32) / np.float32(255.0)
    if (
        array.shape != (SERVING_CANONICAL_SIZE, SERVING_CANONICAL_SIZE)
        or not np.isfinite(array).all()
        or array.min() < 0.0
        or array.max() > 1.0
    ):
        raise ServingInputError("invalid_image_content")
    return np.ascontiguousarray(array, dtype=np.float32)


def serving_tensor_from_canonical_image(
    canonical: np.ndarray, transform: StandardCxrTransform
) -> torch.Tensor:
    """Map one canonical 320-square image through the frozen evaluation transform."""
    if (
        not isinstance(canonical, np.ndarray)
        or canonical.dtype != np.float32
        or canonical.shape != (SERVING_CANONICAL_SIZE, SERVING_CANONICAL_SIZE)
        or not canonical.flags.c_contiguous
        or not np.isfinite(canonical).all()
        or canonical.min() < 0.0
        or canonical.max() > 1.0
    ):
        raise ServingInputError("invalid_image_content")
    tensor = transform(canonical)
    if tensor.dtype != torch.float32 or tensor.shape != (1, 224, 224) or not tensor.is_contiguous():
        raise RuntimeError("Serving image preprocessing violated the frozen tensor contract")
    return tensor


def serving_image_tensor(grayscale: np.ndarray, transform: StandardCxrTransform) -> torch.Tensor:
    """Compose M7 spatial adaptation with the frozen evaluation transform."""
    return serving_tensor_from_canonical_image(serving_canonical_image(grayscale), transform)


def parse_laboratories(document: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Parse exactly 50 canonical raw laboratory values and observedness bits."""
    if not isinstance(document, str) or len(document) > MAX_LABS_JSON_CHARACTERS:
        raise ServingInputError("invalid_labs_json")
    try:
        value = json.loads(document, object_pairs_hook=_unique_json_object)
    except ServingInputError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise ServingInputError("invalid_labs_json") from exc
    if not isinstance(value, dict):
        raise ServingInputError("invalid_labs_json")
    observed_keys = set(value)
    expected_keys = set(LAB_KEYS)
    if observed_keys != expected_keys:
        category = "missing_lab_keys" if expected_keys - observed_keys else "extra_lab_keys"
        raise ServingInputError(category)
    values: dict[str, object] = {}
    missing: list[str] = []
    for key in LAB_KEYS:
        raw = value[key]
        item_id = key.removeprefix("lab_")
        value_column = f"lab_{item_id}_value"
        observed_column = f"lab_{item_id}_observed"
        if raw is None:
            values[value_column] = np.nan
            values[observed_column] = False
            missing.append(key)
        else:
            if isinstance(raw, bool) or not isinstance(raw, int | float):
                raise ServingInputError("invalid_lab_value")
            try:
                numeric = float(raw)
            except (OverflowError, ValueError) as exc:
                raise ServingInputError("invalid_lab_value") from exc
            if not math.isfinite(numeric):
                raise ServingInputError("invalid_lab_value")
            values[value_column] = numeric
            values[observed_column] = True
    frame = pd.DataFrame([values], columns=LAB_FEATURE_COLUMNS)
    return frame, tuple(missing)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ServingInputError("duplicate_lab_keys")
        value[key] = item
    return value


def validate_view_position(value: object) -> str:
    if value not in {"AP", "PA"}:
        raise ServingInputError("invalid_view_position")
    return str(value)


def validated_serving_input(
    *,
    image_content: bytes,
    image_media_type: str | None,
    view_position: object,
    labs_json: str,
    transform: StandardCxrTransform,
) -> ValidatedServingInput:
    if image_media_type != "image/jpeg":
        raise ServingInputError("invalid_media_type")
    validate_view_position(view_position)
    grayscale = decode_jpeg_once(image_content)
    laboratory_frame, missing = parse_laboratories(labs_json)
    return ValidatedServingInput(
        image=serving_image_tensor(grayscale, transform),
        laboratory_frame=laboratory_frame,
        missing_labs=missing,
    )
