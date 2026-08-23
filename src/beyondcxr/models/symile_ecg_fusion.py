"""Implement the exact Symile ECG encoder and three-modality gated fusion."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch
from torch import nn

from beyondcxr.models.cxr_baseline import StandardCxrEncoder, set_cxr_encoder_trainability

ECG_ENCODER_PARAMETER_COUNT = 2_189_632
TRIMODAL_EXCLUDING_CXR_ENCODER_PARAMETER_COUNT = 2_726_785
TRIMODAL_ARCHITECTURE = MappingProxyType(
    {
        "embedding_dimension": 1024,
        "lab_input_dimension": 100,
        "lab_hidden_dimension": 128,
        "lab_core_dimension": 64,
        "latent_dimension": 256,
        "observedness_dimension": 50,
        "gate_hidden_dimension": 128,
        "classifier_hidden_dimension": 128,
        "modality_count": 3,
        "dropout": 0.2,
    }
)


class BasicBlock1d(nn.Module):
    """The fixed kernel-seven residual block used by the ECG branch."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(value)
        output = self.relu(self.bn1(self.conv1(value)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + residual)


class SymileEcgEncoder(nn.Module):
    """Train-from-scratch 12-lead ECG encoder with an exact 256-wide output."""

    output_dimension = 256

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(BasicBlock1d(32, 32, stride=1), BasicBlock1d(32, 32, stride=1))
        self.stage2 = nn.Sequential(BasicBlock1d(32, 64, stride=2), BasicBlock1d(64, 64, stride=1))
        self.stage3 = nn.Sequential(
            BasicBlock1d(64, 128, stride=2), BasicBlock1d(128, 128, stride=1)
        )
        self.stage4 = nn.Sequential(
            BasicBlock1d(128, 256, stride=2), BasicBlock1d(256, 256, stride=1)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self._initialize()
        if parameter_count(self) != ECG_ENCODER_PARAMETER_COUNT:
            raise RuntimeError("ECG encoder parameter contract is invalid")

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        _validate_ecg_structure(ecg)
        return self._forward_validated(ecg)

    def _forward_validated(self, ecg: torch.Tensor) -> torch.Tensor:
        """Encode an ECG whose structural contract was checked by the caller."""
        output = self.stem(ecg)
        output = self.stage1(output)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.stage4(output)
        output = self.pool(output).flatten(1)
        if output.shape != (len(ecg), self.output_dimension):
            raise ValueError("ECG encoder output is invalid")
        return output


class SymileTriModalGatedHead(nn.Module):
    """The separately frozen CXR/labs/ECG per-feature softmax gate."""

    def __init__(self, parameters: Mapping[str, object]) -> None:
        super().__init__()
        _validate_parameters(parameters)
        latent = int(parameters["latent_dimension"])
        dropout = float(parameters["dropout"])
        self.latent_dimension = latent
        self.lab_input_dimension = int(parameters["lab_input_dimension"])
        self.observedness_dimension = int(parameters["observedness_dimension"])
        self.image_projection = nn.Sequential(
            nn.Linear(int(parameters["embedding_dimension"]), latent),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lab_encoder = nn.Sequential(
            nn.Linear(
                self.lab_input_dimension,
                int(parameters["lab_hidden_dimension"]),
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                int(parameters["lab_hidden_dimension"]),
                int(parameters["lab_core_dimension"]),
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(parameters["lab_core_dimension"]), latent),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.ecg_encoder = SymileEcgEncoder()
        self.gate = nn.Sequential(
            nn.Linear(
                3 * latent + self.observedness_dimension,
                int(parameters["gate_hidden_dimension"]),
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(parameters["gate_hidden_dimension"]), 3 * latent),
        )
        self.output = nn.Sequential(
            nn.Linear(latent, int(parameters["classifier_hidden_dimension"])),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(parameters["classifier_hidden_dimension"]), 1),
        )

    def forward(
        self, image_embedding: torch.Tensor, labs: torch.Tensor, ecg: torch.Tensor
    ) -> torch.Tensor:
        logits, _ = self.forward_with_gates(image_embedding, labs, ecg)
        return logits

    def forward_with_gates(
        self, image_embedding: torch.Tensor, labs: torch.Tensor, ecg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not isinstance(image_embedding, torch.Tensor)
            or image_embedding.ndim != 2
            or image_embedding.shape[1] != self.image_projection[0].in_features
            or not image_embedding.is_floating_point()
        ):
            raise ValueError("CXR embedding is invalid")
        if (
            not isinstance(labs, torch.Tensor)
            or labs.shape != (len(image_embedding), self.lab_input_dimension)
            or labs.dtype != torch.float32
            or labs.device != image_embedding.device
        ):
            raise ValueError("laboratory input is invalid")
        _validate_ecg_structure(
            ecg,
            batch_size=len(image_embedding),
            device=image_embedding.device,
        )
        observedness = labs[:, -self.observedness_dimension :]
        representations = (
            self.image_projection(image_embedding),
            self.lab_encoder(labs),
            self.ecg_encoder._forward_validated(ecg),
        )
        gate_logits = self.gate(torch.cat((*representations, observedness), dim=1))
        weights = torch.softmax(gate_logits.reshape(-1, 3, self.latent_dimension), dim=1)
        fused = torch.sum(weights * torch.stack(representations, dim=1), dim=1)
        logits = self.output(fused).squeeze(1)
        if logits.shape != (len(image_embedding),):
            raise ValueError("tri-modal gate produced invalid logits")
        return logits, weights


class SymileTriModalGatedModel(nn.Module):
    """Join the standard CXR encoder to the exact three-modality head."""

    def __init__(self, encoder: nn.Module, parameters: Mapping[str, object]) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("CXR encoder must be a torch module")
        self.encoder = encoder
        self.classifier = SymileTriModalGatedHead(parameters)
        non_cxr = parameter_count(self.classifier)
        if non_cxr != TRIMODAL_EXCLUDING_CXR_ENCODER_PARAMETER_COUNT:
            raise RuntimeError("tri-modal excluding-CXR-encoder parameter contract is invalid")

    def _embedding(self, image: torch.Tensor) -> torch.Tensor:
        encode = getattr(self.encoder, "encode", None)
        return encode(image) if callable(encode) else self.encoder(image)

    def forward(self, image: torch.Tensor, labs: torch.Tensor, ecg: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._embedding(image), labs, ecg)

    def freeze_encoder(self) -> None:
        set_cxr_encoder_trainability(self.encoder, "frozen")

    def unfreeze_encoder(self) -> None:
        set_cxr_encoder_trainability(self.encoder, "all")


def build_symile_trimodal_gated_model(
    parameters: Mapping[str, object], *, weights: str | None
) -> SymileTriModalGatedModel:
    """Build the exact three-modality model without a generic modality mechanism."""
    return SymileTriModalGatedModel(
        StandardCxrEncoder(
            weights=weights,
            expected_embedding_dimension=int(parameters["embedding_dimension"]),
            image_size=int(parameters["image_size"]),
        ),
        parameters,
    )


def parameter_count(module: nn.Module) -> int:
    """Return trainable and non-trainable parameter count for a topology assertion."""
    return sum(parameter.numel() for parameter in module.parameters())


def _validate_parameters(parameters: Mapping[str, object]) -> None:
    if any(parameters.get(key) != value for key, value in TRIMODAL_ARCHITECTURE.items()):
        raise ValueError("tri-modal architecture parameters are invalid")


def _validate_ecg_structure(
    ecg: object,
    *,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> None:
    if (
        not isinstance(ecg, torch.Tensor)
        or ecg.ndim != 3
        or ecg.shape[1:] != (12, 5000)
        or len(ecg) == 0
        or (batch_size is not None and len(ecg) != batch_size)
        or ecg.dtype != torch.float32
        or (device is not None and ecg.device != device)
    ):
        raise ValueError("ECG input must be a B x 12 x 5000 float32 tensor")
