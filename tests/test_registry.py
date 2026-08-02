from __future__ import annotations

import pytest

from radfusion.training.registry import DATASETS, MODELS, RegistryError, get_dataset, get_model


def test_builtin_component_mappings_are_immutable_and_complete() -> None:
    assert tuple(DATASETS) == ("rsna",)
    assert tuple(MODELS) == ("metadata_logistic", "metadata_lightgbm", "image_densenet")
    assert get_dataset("rsna") is DATASETS["rsna"]
    assert get_model("metadata_logistic") is MODELS["metadata_logistic"]
    assert get_model("image_densenet") is MODELS["image_densenet"]
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
