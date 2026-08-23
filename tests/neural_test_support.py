"""Small deterministic neural objects shared by lifecycle tests."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset

from beyondcxr.training.device import resolve_device
from beyondcxr.training.execution import reused_loader_policy
from beyondcxr.training.neural import build_image_loaders as _build_image_loaders


class TensorDataset(Dataset[dict[str, object]]):
    def __init__(self, targets: list[int], *, sample_prefix: str = "") -> None:
        self.targets = targets
        self.sample_prefix = sample_prefix

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, object]:
        target = self.targets[index]
        return {
            "image": torch.tensor([float(index % 2), 1.0], dtype=torch.float32),
            "target": torch.tensor(float(target), dtype=torch.float32),
            "sample_id": f"{self.sample_prefix}sample-{index}",
            "patient_id": f"patient-{index}",
        }


class TinyImageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.ReLU())
        self.classifier = nn.Linear(3, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(images)).squeeze(1)

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True


def cpu_runtime():
    return resolve_device("cpu", mixed_precision=True, pin_memory_policy="enabled")


def build_synchronous_image_loaders(*args, execution=None, **kwargs):
    """Build deterministic loaders without worker-process overhead."""
    runtime = kwargs["runtime"]
    policy = execution or reused_loader_policy(
        num_workers=0,
        pin_memory=runtime.pin_memory_effective,
    )
    return _build_image_loaders(*args, execution=policy, **kwargs)
