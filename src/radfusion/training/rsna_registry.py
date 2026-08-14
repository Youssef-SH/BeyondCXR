"""Provide immutable built-in RSNA dataset and model mappings."""

from __future__ import annotations

from types import MappingProxyType

from radfusion.models.cxr_baseline import CxrDenseNetModel
from radfusion.models.fusion_concat import RsnaCxrMetadataConcatModel
from radfusion.models.rsna_metadata import MetadataLightgbmModel, MetadataLogisticModel
from radfusion.training.rsna_datasets import RsnaDataset
from radfusion.training.rsna_interfaces import (
    RsnaCxrModelImplementation,
    RsnaDatasetImplementation,
    RsnaFusionModelImplementation,
    RsnaMetadataModelImplementation,
)


class RegistryError(LookupError):
    """Raised when a built-in component key is unknown."""


DATASETS: MappingProxyType[str, RsnaDatasetImplementation] = MappingProxyType(
    {"rsna": RsnaDataset()}
)
MODELS: MappingProxyType[
    str,
    RsnaMetadataModelImplementation | RsnaCxrModelImplementation | RsnaFusionModelImplementation,
] = MappingProxyType(
    {
        "metadata_logistic": MetadataLogisticModel(),
        "metadata_lightgbm": MetadataLightgbmModel(),
        "cxr_densenet": CxrDenseNetModel(),
        "cxr_metadata_concat": RsnaCxrMetadataConcatModel(),
    }
)


def get_dataset(key: str) -> RsnaDatasetImplementation:
    """Return one built-in dataset adapter."""
    return _get(DATASETS, key, "dataset")


def get_model(
    key: str,
) -> RsnaMetadataModelImplementation | RsnaCxrModelImplementation | RsnaFusionModelImplementation:
    """Return one built-in model adapter."""
    return _get(MODELS, key, "model")


def _get[T](mapping: MappingProxyType[str, T], key: str, kind: str) -> T:
    try:
        return mapping[key]
    except KeyError as exc:
        raise RegistryError(f"Unknown {kind} key: {key!r}") from exc
