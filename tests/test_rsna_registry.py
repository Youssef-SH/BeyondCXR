from __future__ import annotations

import pytest

from beyondcxr.training.rsna_registry import DATASETS, MODELS, RegistryError, get_dataset, get_model


def test_builtin_component_mappings_are_immutable_and_complete() -> None:
    assert tuple(DATASETS) == ("rsna",)
    assert tuple(MODELS) == (
        "metadata_logistic",
        "metadata_lightgbm",
        "cxr_densenet",
        "cxr_metadata_concat",
    )
    assert get_dataset("rsna") is DATASETS["rsna"]
    assert get_model("metadata_logistic") is MODELS["metadata_logistic"]
    assert get_model("cxr_densenet") is MODELS["cxr_densenet"]
    assert get_model("cxr_metadata_concat") is MODELS["cxr_metadata_concat"]
    with pytest.raises(TypeError):
        DATASETS["other"] = object()  # type: ignore[index]
    with pytest.raises(TypeError):
        MODELS["other"] = object()  # type: ignore[index]


@pytest.mark.parametrize(
    "lookup",
    [get_dataset, get_model],
)
def test_unknown_builtin_component_keys_fail(lookup) -> None:
    with pytest.raises(RegistryError):
        lookup("missing")
