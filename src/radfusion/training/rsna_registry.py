"""Provide immutable built-in dataset and model mappings."""

from __future__ import annotations

from types import MappingProxyType

from radfusion.models.cxr_baseline import ImageDenseNetModel
from radfusion.models.fusion_concat import FusionConcatModel
from radfusion.models.rsna_metadata import MetadataLightgbmModel, MetadataLogisticModel
from radfusion.training.rsna_datasets import RsnaDataset
from radfusion.training.rsna_interfaces import (
    DatasetImplementation,
    FusionModelImplementation,
    ImageModelImplementation,
    ModelImplementation,
)


class RegistryError(LookupError):
    """Raised when a built-in component key is unknown."""


DATASETS: MappingProxyType[str, DatasetImplementation] = MappingProxyType({"rsna": RsnaDataset()})
MODELS: MappingProxyType[
    str, ModelImplementation | ImageModelImplementation | FusionModelImplementation
] = MappingProxyType(
    {
        "metadata_logistic": MetadataLogisticModel(),
        "metadata_lightgbm": MetadataLightgbmModel(),
        "cxr_densenet": ImageDenseNetModel(),
        "cxr_metadata_concat": FusionConcatModel(),
    }
)


def get_dataset(key: str) -> DatasetImplementation:
    """Return one built-in dataset adapter."""
    return _get(DATASETS, key, "dataset")


def get_model(
    key: str,
) -> ModelImplementation | ImageModelImplementation | FusionModelImplementation:
    """Return one built-in model adapter."""
    return _get(MODELS, key, "model")


def _get[T](mapping: MappingProxyType[str, T], key: str, kind: str) -> T:
    try:
        return mapping[key]
    except KeyError as exc:
        raise RegistryError(f"Unknown {kind} key: {key!r}") from exc
