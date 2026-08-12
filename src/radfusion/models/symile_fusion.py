"""Compose the frozen Symile concat and missingness-aware gated models."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from radfusion.models.cxr_baseline import (
    StandardCxrEncoder,
    set_cxr_encoder_trainability,
)
from radfusion.models.fusion_concat import ConcatFusionHead


def symile_concat_architecture(parameters: Mapping[str, object]) -> dict[str, int | float]:
    """Return the exact lower-level concat-head contract."""
    return {
        "image_embedding_dimension": int(parameters["embedding_dimension"]),
        "image_projection_dimension": int(parameters["image_projection_dimension"]),
        "structured_input_dimension": int(parameters["lab_input_dimension"]),
        "structured_hidden_dimension": int(parameters["lab_hidden_dimension"]),
        "structured_projection_dimension": int(parameters["lab_projection_dimension"]),
        "fusion_input_dimension": int(parameters["image_projection_dimension"])
        + int(parameters["lab_projection_dimension"]),
        "fusion_hidden_dimension": int(parameters["fusion_hidden_dimension"]),
        "dropout": float(parameters["dropout"]),
        "output_dimension": 1,
    }


class SymileConcatFusionModel(nn.Module):
    """Compose the standard CXR encoder and reusable concat head for Symile labs."""

    def __init__(self, encoder: nn.Module, parameters: Mapping[str, object]) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("Symile concat encoder must be a torch module")
        self.encoder = encoder
        self.classifier = ConcatFusionHead(symile_concat_architecture(parameters))

    def forward(self, image: torch.Tensor, labs: torch.Tensor) -> torch.Tensor:
        encode = getattr(self.encoder, "encode", None)
        embedding = encode(image) if callable(encode) else self.encoder(image)
        return self.classifier(embedding, labs)

    def freeze_encoder(self) -> None:
        set_cxr_encoder_trainability(self.encoder, "frozen")

    def unfreeze_encoder(self) -> None:
        set_cxr_encoder_trainability(self.encoder, "all")


class SymileGatedFusionHead(nn.Module):
    """Fuse CXR and labs through per-feature two-modality softmax gates."""

    def __init__(self, parameters: Mapping[str, object]) -> None:
        super().__init__()
        use_observedness = parameters["use_observedness"]
        modality_count = parameters["modality_count"]
        if not isinstance(use_observedness, bool):
            raise TypeError("use_observedness must be Boolean")
        if modality_count != 2:
            raise ValueError("M5 gated fusion requires exactly CXR and laboratories")
        self.use_observedness = use_observedness
        self.modality_count = modality_count
        self.lab_input_dimension = int(parameters["lab_input_dimension"])
        self.observedness_dimension = int(parameters["observedness_dimension"])
        self.latent_dimension = int(parameters["latent_dimension"])
        dropout = float(parameters["dropout"])
        self.image_projection = nn.Sequential(
            nn.Linear(int(parameters["embedding_dimension"]), self.latent_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lab_core = nn.Sequential(
            nn.Linear(self.lab_input_dimension, int(parameters["lab_hidden_dimension"])),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                int(parameters["lab_hidden_dimension"]),
                int(parameters["lab_core_dimension"]),
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lab_projection = nn.Sequential(
            nn.Linear(int(parameters["lab_core_dimension"]), self.latent_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        gate_input = modality_count * self.latent_dimension + self.observedness_dimension
        self.gate = nn.Sequential(
            nn.Linear(gate_input, int(parameters["gate_hidden_dimension"])),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                int(parameters["gate_hidden_dimension"]),
                modality_count * self.latent_dimension,
            ),
        )
        self.output = nn.Sequential(
            nn.Linear(self.latent_dimension, int(parameters["classifier_hidden_dimension"])),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(parameters["classifier_hidden_dimension"]), 1),
        )

    def forward(
        self,
        image_embedding: torch.Tensor,
        labs: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.forward_with_gates(image_embedding, labs)
        return logits

    def forward_with_gates(
        self,
        image_embedding: torch.Tensor,
        labs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw logits and normalized per-feature modality weights."""
        _validate_matrix(
            image_embedding,
            len(image_embedding),
            self.image_projection[0].in_features,
            "CXR embedding",
        )
        _validate_matrix(labs, len(image_embedding), self.lab_input_dimension, "laboratory input")
        values = labs[:, : self.lab_input_dimension - self.observedness_dimension]
        observedness = labs[:, -self.observedness_dimension :]
        if not self.use_observedness:
            observedness = torch.zeros_like(observedness)
        lab_input = torch.cat((values, observedness), dim=1)
        representations = [
            self.image_projection(image_embedding),
            self.lab_projection(self.lab_core(lab_input)),
        ]
        gate_input = torch.cat((*representations, observedness), dim=1)
        weights = torch.softmax(
            self.gate(gate_input).reshape(-1, self.modality_count, self.latent_dimension),
            dim=1,
        )
        stacked = torch.stack(representations, dim=1)
        fused = torch.sum(weights * stacked, dim=1)
        logits = self.output(fused).squeeze(1)
        if logits.shape != (len(image_embedding),):
            raise ValueError("Gated fusion produced invalid logits")
        return logits, weights


class SymileGatedFusionModel(nn.Module):
    """Combine the standard CXR encoder with one observedness-configurable gate."""

    def __init__(self, encoder: nn.Module, parameters: Mapping[str, object]) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("Symile gated encoder must be a torch module")
        self.encoder = encoder
        self.classifier = SymileGatedFusionHead(parameters)

    def forward(self, image: torch.Tensor, labs: torch.Tensor) -> torch.Tensor:
        encode = getattr(self.encoder, "encode", None)
        embedding = encode(image) if callable(encode) else self.encoder(image)
        return self.classifier(embedding, labs)

    def freeze_encoder(self) -> None:
        set_cxr_encoder_trainability(self.encoder, "frozen")

    def unfreeze_encoder(self) -> None:
        set_cxr_encoder_trainability(self.encoder, "all")


def build_symile_concat_model(
    parameters: Mapping[str, object], *, weights: str | None
) -> SymileConcatFusionModel:
    """Build one frozen-topology Symile concat model."""
    return SymileConcatFusionModel(
        StandardCxrEncoder(
            weights=weights,
            expected_embedding_dimension=int(parameters["embedding_dimension"]),
            image_size=int(parameters["image_size"]),
        ),
        parameters,
    )


def build_symile_gated_model(
    parameters: Mapping[str, object], *, weights: str | None
) -> SymileGatedFusionModel:
    """Build one frozen-topology Symile gated model."""
    return SymileGatedFusionModel(
        StandardCxrEncoder(
            weights=weights,
            expected_embedding_dimension=int(parameters["embedding_dimension"]),
            image_size=int(parameters["image_size"]),
        ),
        parameters,
    )


def _validate_matrix(value: object, rows: int, columns: int, name: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (rows, columns)
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a finite floating matrix shaped {rows} x {columns}")
