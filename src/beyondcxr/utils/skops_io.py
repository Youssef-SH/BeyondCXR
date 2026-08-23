"""Persist approved BeyondCXR estimators with skops."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import skops.io as sio

TRUSTED_MODEL_TYPES = frozenset(
    {
        "collections.OrderedDict",
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMClassifier",
        "numpy.dtype",
        "beyondcxr.data.symile_preprocess.SymileLabEcdfTransformer",
        "beyondcxr.data.rsna_metadata_preprocess.RsnaMetadataFeatures",
    }
)


def save_skops(value: Any, path: str | Path) -> Path:
    """Serialize a model or preprocessing pipeline with skops."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        sio.dump(value, temporary)
        trusted_types_for_file(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_skops(path: str | Path) -> Any:
    """Load a skops artifact after rejecting unexpected executable types."""
    source = Path(path)
    trusted = trusted_types_for_file(source)
    return sio.load(source, trusted=trusted)


def trusted_types_for_file(path: str | Path) -> list[str]:
    """Return the approved non-default types required by a skops artifact."""
    required = set(sio.get_untrusted_types(file=Path(path)))
    unexpected = sorted(required - TRUSTED_MODEL_TYPES)
    if unexpected:
        raise ValueError(f"Skops artifact contains unexpected types: {unexpected}")
    return sorted(required)
