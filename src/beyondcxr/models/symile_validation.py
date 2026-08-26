"""Shared input validation for Symile fusion models."""

from __future__ import annotations

import torch


def _validate_fusion_input_devices(
    image_embedding: torch.Tensor, *other_inputs: torch.Tensor
) -> None:
    if any(value.device != image_embedding.device for value in other_inputs):
        raise ValueError("Fusion inputs must share the fusion input device")
