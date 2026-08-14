"""Define typed boundaries for RSNA experiment components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from torch import nn

from radfusion.training.config import ExperimentConfig, FamilyConfig

if TYPE_CHECKING:
    from radfusion.training.rsna_datasets import (
        CxrRunData,
        CxrTestData,
        FusionRunData,
        FusionTestData,
    )


@dataclass(frozen=True)
class DatasetLineage:
    """Pinned RSNA dataset and task lineage shared by all partitions."""

    bundle_id: str
    split_assignment_id: str
    label_policy_version: str
    task_id: str


@dataclass(frozen=True)
class DatasetPartition:
    """Own mutable scientific arrays for one approved, isolated RSNA partition."""

    features: pd.DataFrame
    targets: np.ndarray
    sample_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    partition: str


@dataclass(frozen=True)
class DatasetRunData:
    """The only RSNA partitions available to the training runner."""

    train: DatasetPartition
    validation: DatasetPartition
    lineage: DatasetLineage


@dataclass(frozen=True)
class ModelFitResult:
    """Fitted RSNA metadata pipeline and model-derived logging parameters."""

    pipeline: Pipeline
    derived_parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "derived_parameters",
            MappingProxyType(dict(self.derived_parameters)),
        )


class RsnaDatasetImplementation(Protocol):
    """RSNA dataset adapter used by training and evaluation runners."""

    def load_train_validation(self, config: ExperimentConfig) -> DatasetRunData:
        """Load only train and validation partitions from a pinned bundle."""

    def load_lineage(self, config: ExperimentConfig) -> DatasetLineage:
        """Validate a pinned bundle and return task lineage."""

    def load_test(self, config: ExperimentConfig) -> tuple[DatasetPartition, DatasetLineage]:
        """Load only the test partition and its pinned lineage."""

    def load_cxr_train_validation(self, config: ExperimentConfig) -> CxrRunData:
        """Load source-inventory-bound CXR train and validation rows."""

    def load_cxr_test(
        self,
        config: ExperimentConfig,
        *,
        expected_manifest_sha256: str,
    ) -> CxrTestData:
        """Load source-inventory-bound CXR test rows."""

    def load_fusion_train_validation(self, config: ExperimentConfig) -> FusionRunData:
        """Load source-inventory-bound aligned fusion train and validation rows."""

    def load_fusion_test(
        self,
        config: ExperimentConfig,
        *,
        expected_manifest_sha256: str,
    ) -> FusionTestData:
        """Load source-inventory-bound aligned fusion test rows."""


class RsnaMetadataModelImplementation(Protocol):
    """Registered RSNA metadata model implementation."""

    def fit(
        self,
        config: ExperimentConfig,
        training_seed: int,
        train_features: pd.DataFrame,
        train_targets: np.ndarray,
        validation_features: pd.DataFrame,
        validation_targets: np.ndarray,
    ) -> ModelFitResult:
        """Fit one model from training data with validation monitoring."""


class RsnaCxrModelImplementation(Protocol):
    """Registered RSNA CXR model builder used by the neural runner."""

    def build(self, config: FamilyConfig) -> nn.Module:
        """Build an unfitted CXR model."""

    def build_architecture(self, config: FamilyConfig) -> nn.Module:
        """Build the package architecture without loading pretrained weights."""


class RsnaFusionModelImplementation(Protocol):
    """Registered RSNA fusion model builder used by the neural runner."""

    def build(
        self,
        config: FamilyConfig,
        *,
        structured_dimension: int,
        weights: str | None = None,
    ) -> nn.Module:
        """Build a fusion model for an exact transformed metadata width."""
