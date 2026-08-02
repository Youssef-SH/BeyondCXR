"""Define the fixed RSNA image-metadata concat model."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import nn

from radfusion.models.cxr_baseline import StandardCxrEncoder
from radfusion.training.config import ModelConfig, fusion_architecture_contract


class ConcatFusionHead(nn.Module):
    """Project image and structured representations into one binary head."""

    def __init__(self, architecture: Mapping[str, int | float]) -> None:
        super().__init__()
        self.structured_dimension = int(architecture["structured_input_dimension"])
        self.image_embedding_dimension = int(architecture["image_embedding_dimension"])
        image_projection = int(architecture["image_projection_dimension"])
        structured_hidden = int(architecture["structured_hidden_dimension"])
        structured_projection = int(architecture["structured_projection_dimension"])
        fusion_input = int(architecture["fusion_input_dimension"])
        fusion_hidden = int(architecture["fusion_hidden_dimension"])
        dropout = float(architecture["dropout"])
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_embedding_dimension, image_projection),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.structured_projection = nn.Sequential(
            nn.Linear(self.structured_dimension, structured_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(structured_hidden, structured_projection),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Sequential(
            nn.Linear(fusion_input, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, int(architecture["output_dimension"])),
        )

    def forward(self, image_embedding: torch.Tensor, structured: torch.Tensor) -> torch.Tensor:
        """Return one raw binary logit per aligned image-structured row."""
        if image_embedding.ndim != 2 or image_embedding.shape[1] != self.image_embedding_dimension:
            raise ValueError(
                f"Fusion image embeddings must have shape N x {self.image_embedding_dimension}"
            )
        if (
            structured.ndim != 2
            or structured.shape[0] != image_embedding.shape[0]
            or structured.shape[1] != self.structured_dimension
        ):
            raise ValueError(
                f"Fusion structured input must have shape N x {self.structured_dimension}"
            )
        if not image_embedding.is_floating_point() or not structured.is_floating_point():
            raise ValueError("Fusion inputs must contain floating-point values")
        if not torch.isfinite(image_embedding).all() or not torch.isfinite(structured).all():
            raise ValueError("Fusion inputs must contain finite values")
        combined = torch.cat(
            (self.image_projection(image_embedding), self.structured_projection(structured)),
            dim=1,
        )
        logits = self.output(combined).squeeze(1)
        if logits.shape != (len(image_embedding),) or not torch.isfinite(logits).all():
            raise ValueError("Fusion model produced invalid logits")
        return logits


class RsnaConcatFusionModel(nn.Module):
    """Combine the standard CXR encoder with the fixed concat fusion head."""

    def __init__(
        self,
        encoder: nn.Module,
        architecture: Mapping[str, int | float],
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("Fusion encoder must be a torch.nn.Module")
        self.encoder = encoder
        self.classifier = ConcatFusionHead(architecture)

    @property
    def structured_dimension(self) -> int:
        """Return the required transformed metadata width."""
        return self.classifier.structured_dimension

    def forward(self, image: torch.Tensor, structured: torch.Tensor) -> torch.Tensor:
        """Return one raw binary logit per multimodal sample."""
        encode = getattr(self.encoder, "encode", None)
        image_embedding = encode(image) if callable(encode) else self.encoder(image)
        return self.classifier(image_embedding, structured)

    def freeze_encoder(self) -> None:
        """Freeze only the DenseNet image encoder."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Restore DenseNet image-encoder trainability."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True


class FusionConcatModel:
    """Build the fixed RSNA image-metadata concat model."""

    def __init__(
        self,
        encoder_factory: Callable[..., nn.Module] = StandardCxrEncoder,
    ) -> None:
        self._encoder_factory = encoder_factory

    def build(
        self,
        config: ModelConfig,
        *,
        structured_dimension: int,
        weights: str | None = None,
    ) -> RsnaConcatFusionModel:
        """Build the fusion architecture with an explicit structured width."""
        architecture = fusion_architecture_contract(
            config,
            structured_input_dimension=structured_dimension,
        )
        encoder = self._encoder_factory(
            weights=weights,
            expected_embedding_dimension=config.parameters["embedding_dimension"],
            image_size=config.parameters["image_size"],
        )
        if not isinstance(encoder, nn.Module):
            raise TypeError("Fusion encoder factory must return a torch.nn.Module")
        return RsnaConcatFusionModel(encoder, architecture)


def initialize_fusion_encoder(
    model: RsnaConcatFusionModel,
    source_state: Mapping[str, torch.Tensor],
) -> None:
    """Strictly initialize only the fusion image encoder from a CXR checkpoint."""
    prefix = "encoder."
    extracted = {
        key[len(prefix) :]: value
        for key, value in source_state.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    if not extracted:
        raise ValueError("Source CXR checkpoint contains no encoder state")
    expected = model.encoder.state_dict()
    if set(extracted) != set(expected):
        raise ValueError("Source CXR encoder state does not match the fusion encoder")
    if any(
        not isinstance(value, torch.Tensor)
        or value.shape != expected[key].shape
        or value.dtype != expected[key].dtype
        or not torch.isfinite(value).all()
        for key, value in extracted.items()
    ):
        raise ValueError("Source CXR encoder state contains incompatible tensors")
    incompatible = model.encoder.load_state_dict(extracted, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("Source CXR encoder state did not load strictly")
