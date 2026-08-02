"""Define typed boundaries for tabular experiment components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from torch import nn

from radfusion.training.config import DatasetConfig, ModelConfig


@dataclass(frozen=True)
class DatasetLineage:
    """Pinned dataset and task lineage shared by all partitions."""

    bundle_id: str
    split_assignment_id: str
    label_policy_version: str
    task_id: str


@dataclass(frozen=True)
class DatasetPartition:
    """Approved model inputs separated from identifiers and partition lineage."""

    features: pd.DataFrame
    targets: np.ndarray
    sample_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    partition: str


@dataclass(frozen=True)
class DatasetRunData:
    """The only partitions available to the training runner."""

    train: DatasetPartition
    validation: DatasetPartition
    lineage: DatasetLineage


@dataclass(frozen=True)
class ModelFitResult:
    """Fitted pipeline and model-derived logging parameters."""

    pipeline: Pipeline
    derived_parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "derived_parameters",
            MappingProxyType(dict(self.derived_parameters)),
        )


class DatasetImplementation(Protocol):
    """Dataset adapter used by the tabular runner and evaluator."""

    def load_train_validation(self, config: DatasetConfig) -> DatasetRunData:
        """Load only train and validation partitions from a pinned bundle."""

    def load_lineage(self, config: DatasetConfig) -> DatasetLineage:
        """Validate a pinned bundle and return task lineage."""

    def load_test(self, config: DatasetConfig) -> tuple[DatasetPartition, DatasetLineage]:
        """Load only the test partition and its pinned lineage."""

    def load_image_train_validation(self, config: DatasetConfig) -> Any:
        """Load and authenticate image train and validation rows."""

    def load_image_test(
        self,
        config: DatasetConfig,
        *,
        expected_manifest_sha256: str,
    ) -> Any:
        """Load and authenticate image test rows."""

    def load_fusion_train_validation(self, config: DatasetConfig) -> Any:
        """Load and authenticate aligned fusion train and validation rows."""

    def load_fusion_test(
        self,
        config: DatasetConfig,
        *,
        expected_manifest_sha256: str,
    ) -> Any:
        """Load and authenticate aligned fusion test rows."""


class ModelImplementation(Protocol):
    """Registered model implementation used by the experiment runner."""

    def fit(
        self,
        config: ModelConfig,
        training_seed: int,
        train_features: pd.DataFrame,
        train_targets: np.ndarray,
        validation_features: pd.DataFrame,
        validation_targets: np.ndarray,
    ) -> ModelFitResult:
        """Fit one model from training data with validation monitoring."""


class ImageModelImplementation(Protocol):
    """Registered image model builder used by the neural runner."""

    def build(self, config: ModelConfig) -> nn.Module:
        """Build an unfitted image model."""


class FusionModelImplementation(Protocol):
    """Registered fusion model builder used by the neural runner."""

    def build(
        self,
        config: ModelConfig,
        *,
        structured_dimension: int,
        weights: str | None = None,
    ) -> nn.Module:
        """Build a fusion model for an exact transformed metadata width."""
