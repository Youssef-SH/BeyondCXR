"""Compute deterministic Grad-CAM maps for the standard CXR image branch."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def standard_cxr_gradcam_target(model: nn.Module) -> nn.Module:
    """Return DenseNet's complete final spatial feature sequence."""
    encoder = getattr(model, "encoder", None)
    backbone = getattr(encoder, "backbone", None)
    features = getattr(backbone, "features", None)
    if not isinstance(features, nn.Module):
        raise ValueError("Model does not expose the standard CXR spatial feature sequence")
    return features


def gradcam_heatmaps(
    model: nn.Module,
    image: torch.Tensor,
    *,
    structured: torch.Tensor | None = None,
    target_module: nn.Module | None = None,
    output_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Return normalized Grad-CAM maps for raw positive-class logits."""
    if image.ndim != 4 or image.shape[0] == 0:
        raise ValueError("Grad-CAM image input must be a non-empty NCHW tensor")
    if structured is not None and (structured.ndim != 2 or structured.shape[0] != image.shape[0]):
        raise ValueError("Grad-CAM structured input is not batch-aligned")
    target = target_module or standard_cxr_gradcam_target(model)
    activation: torch.Tensor | None = None
    gradient: torch.Tensor | None = None

    def capture(_: nn.Module, __: tuple[object, ...], output: object) -> None:
        nonlocal activation, gradient
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Grad-CAM target must emit an N x C x H x W tensor")
        activation = output.detach().clone()

        def capture_gradient(value: torch.Tensor) -> None:
            nonlocal gradient
            gradient = value.detach().clone()

        output.register_hook(capture_gradient)

    handle = target.register_forward_hook(capture)
    was_training = model.training
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(image, structured) if structured is not None else model(image)
        if logits.shape != (len(image),) or not torch.isfinite(logits).all():
            raise ValueError("Grad-CAM requires one finite raw logit per input")
        logits.sum().backward()
        if activation is None or gradient is None or activation.shape != gradient.shape:
            raise ValueError("Grad-CAM did not capture aligned activations and gradients")
        if activation.shape[0] != len(image) or not all(
            torch.isfinite(value).all() for value in (activation, gradient)
        ):
            raise ValueError("Grad-CAM captured invalid tensors")
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        maps = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        flat = maps.flatten(1)
        maxima = flat.max(dim=1).values
        normalized = torch.zeros_like(maps)
        nonzero = maxima > 0
        if nonzero.any():
            normalized[nonzero] = maps[nonzero] / maxima[nonzero, None, None, None]
        resized = F.interpolate(
            normalized,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        if not torch.isfinite(resized).all() or (resized < 0).any() or (resized > 1).any():
            raise ValueError("Grad-CAM produced an invalid normalized heatmap")
        return resized.detach()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
