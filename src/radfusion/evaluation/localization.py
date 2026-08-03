"""Map RSNA boxes into model space and calculate localization metrics."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from radfusion.data.cxr_transforms import center_crop_geometry


@dataclass(frozen=True)
class ModelSpaceBox:
    """One positive-area box in the 224 x 224 model-input frame."""

    x0: float
    y0: float
    x1: float
    y1: float


def transform_rsna_box(
    box: Mapping[str, object],
    *,
    source_rows: int,
    source_columns: int,
    output_size: int = 224,
) -> ModelSpaceBox | None:
    """Intersect one source box with the center crop and map it to model space."""
    geometry = center_crop_geometry(source_rows, source_columns)
    values = []
    for field in ("x", "y", "width", "height"):
        value = box.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("RSNA box coordinates must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("RSNA box coordinates must be finite")
        values.append(number)
    x, y, width, height = values
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > source_columns
        or y + height > source_rows
    ):
        raise ValueError("RSNA source box must be positive and within source-image bounds")
    crop_x1 = geometry.offset_x + geometry.crop_size
    crop_y1 = geometry.offset_y + geometry.crop_size
    clipped_x0 = max(x, float(geometry.offset_x))
    clipped_y0 = max(y, float(geometry.offset_y))
    clipped_x1 = min(x + width, float(crop_x1))
    clipped_y1 = min(y + height, float(crop_y1))
    if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
        return None
    scale = output_size / geometry.crop_size
    result = ModelSpaceBox(
        max(0.0, min(float(output_size), (clipped_x0 - geometry.offset_x) * scale)),
        max(0.0, min(float(output_size), (clipped_y0 - geometry.offset_y) * scale)),
        max(0.0, min(float(output_size), (clipped_x1 - geometry.offset_x) * scale)),
        max(0.0, min(float(output_size), (clipped_y1 - geometry.offset_y) * scale)),
    )
    if not (
        0.0 <= result.x0 < result.x1 <= output_size and 0.0 <= result.y0 < result.y1 <= output_size
    ):
        raise ValueError("Transformed RSNA box is outside model-input bounds")
    return result


def union_box_mask(boxes: Sequence[ModelSpaceBox], *, output_size: int = 224) -> np.ndarray:
    """Rasterize the union of model-space boxes."""
    if not boxes:
        raise ValueError("Localization requires at least one transformed box")
    mask = np.zeros((output_size, output_size), dtype=bool)
    for box in boxes:
        x0 = max(0, min(output_size, math.floor(box.x0)))
        y0 = max(0, min(output_size, math.floor(box.y0)))
        x1 = max(0, min(output_size, math.ceil(box.x1)))
        y1 = max(0, min(output_size, math.ceil(box.y1)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Rasterized localization box has no area")
        mask[y0:y1, x0:x1] = True
    return mask


def localization_metrics(heatmap: np.ndarray, union_mask: np.ndarray) -> tuple[int, float, bool]:
    """Return pointing-game hit, activation-energy fraction, and zero-map flag."""
    heat = np.asarray(heatmap, dtype=np.float64)
    mask = np.asarray(union_mask, dtype=bool)
    if heat.ndim != 2 or heat.shape != mask.shape or not np.isfinite(heat).all():
        raise ValueError("Localization heatmap and union mask must be aligned finite 2D arrays")
    if (heat < 0).any() or not mask.any():
        raise ValueError("Localization requires a nonnegative map and nonempty union mask")
    total = float(heat.sum())
    if total == 0.0:
        return 0, 0.0, True
    row, column = np.unravel_index(int(np.argmax(heat)), heat.shape)
    pointing = int(mask[row, column])
    energy = float(heat[mask].sum() / total)
    return pointing, energy, False


def deterministic_qualitative_selection(
    rows: Sequence[Mapping[str, object]],
    *,
    policy_version: str = "sha256-stratum-order-v1",
) -> dict[str, Mapping[str, object] | None]:
    """Select one deterministic internal example per prediction stratum."""
    selected: dict[str, Mapping[str, object] | None] = {}
    for stratum in ("TP", "FN", "FP", "TN"):
        candidates = [row for row in rows if row.get("stratum") == stratum]
        if not candidates:
            selected[stratum] = None
            continue
        selected[stratum] = min(
            candidates,
            key=lambda row: qualitative_selection_key(
                str(row["sample_id"]), policy_version=policy_version
            ),
        )
    return selected


def qualitative_selection_key(
    sample_id: str,
    *,
    policy_version: str = "sha256-stratum-order-v1",
) -> str:
    """Return the deterministic internal ordering key for one qualitative case."""
    if not sample_id:
        raise ValueError("Qualitative selection requires a non-empty sample ID")
    return hashlib.sha256(f"{policy_version}\0{sample_id}".encode()).hexdigest()
