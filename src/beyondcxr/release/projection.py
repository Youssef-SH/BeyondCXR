"""Allowlisted aggregate projection for public result rendering."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from beyondcxr.data.errors import ManifestBuildError

RELIABILITY_FAMILIES = ("cxr_densenet", "cxr_labs_gated")


@dataclass(frozen=True)
class PublicResultProjection:
    """Aggregate-only public view derived from validated scientific claims."""

    cohort: Mapping[str, Any]
    development: Mapping[str, Any]
    held_out: Mapping[str, Any]
    thresholds: Mapping[str, float]
    subgroups: Mapping[str, Any]
    provenance: Mapping[str, Any]


def project_public_results(
    *,
    cohort: Mapping[str, Any],
    core_analysis: Mapping[str, Any],
    extension_claims: Mapping[str, Any],
    global_claims: Mapping[str, Any],
    development_reliability: Mapping[str, Any],
    thresholds: Mapping[str, float],
    subgroups: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> PublicResultProjection:
    """Build a deep-checked aggregate projection after callers validate its authorities."""
    development = {
        "repeat_metrics": core_analysis["repeat_metrics"],
        "ensemble_metrics": core_analysis["ensemble_metrics"],
        "paired_effects": core_analysis["paired_effects"],
        "observedness_ablation": core_analysis["observedness_ablation"],
        "ecg_repeat_metrics": extension_claims["ecg_repeat_metrics"],
        "ecg_ensemble_metrics": extension_claims["ecg_ensemble_metrics"],
        "ecg_vs_gated_repeat_effects": extension_claims["ecg_vs_gated_repeat_effects"],
        "ecg_vs_gated_ensemble_effect": extension_claims["ecg_vs_gated_ensemble_effect"],
        "reliability_curves": development_reliability["curves"],
        "calibration": development_reliability["calibration"],
    }
    held_out = {
        "predictor_views": global_claims["predictor_views"],
        "discrimination_curves": global_claims["discrimination_curves"],
        "reliability_curves": {
            family: global_claims["reliability_curves"][family] for family in RELIABILITY_FAMILIES
        },
        "paired_effects": global_claims["paired_effects"],
        "primary_operating_points": global_claims["primary_operating_points"],
    }
    result = PublicResultProjection(
        _freeze(cohort),
        _freeze(development),
        _freeze(held_out),
        _freeze(thresholds),
        _freeze(subgroups),
        _freeze(provenance),
    )
    _validate_public_value(result)
    return result


def _validate_public_value(value: object) -> None:
    if isinstance(value, PublicResultProjection):
        for field in value.__dataclass_fields__:
            _validate_public_value(getattr(value, field))
    elif isinstance(value, Mapping):
        forbidden = {
            "sample_id",
            "subject_id",
            "hadm_id",
            "patient_id",
            "study_id",
            "dicom_id",
            "source_row",
            "prediction_rows",
            "path",
            "directory",
        }
        if any(not isinstance(key, str) for key in value):
            raise ManifestBuildError("Public projection mapping keys must be strings")
        if forbidden & set(value):
            raise ManifestBuildError("Public projection contains private or row-level fields")
        for item in value.values():
            _validate_public_value(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_public_value(item)
    elif value is None or type(value) in (bool, int, str):
        return
    elif type(value) is float:
        if not math.isfinite(value):
            raise ManifestBuildError("Public projection contains non-finite values")
    else:
        raise ManifestBuildError("Public projection contains an unsupported value type")


def _freeze(value: Any) -> Any:
    """Recursively freeze the small JSON-like public projection."""
    if isinstance(value, np.generic):
        return _freeze(value.item())
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ManifestBuildError("Public projection mapping keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value
