from pathlib import Path

import pytest

from beyondcxr.utils.immutable_json import immutable_json, json_projection


def test_immutable_json_detaches_and_projects_ordered_values():
    original = {"values": [1, True, None, {"score": 0.25}]}
    frozen = immutable_json(original)
    original["values"].append(2)
    with pytest.raises(TypeError):
        frozen["values"][3]["score"] = 0.5
    projected = json_projection(frozen)
    assert projected == {"values": [1, True, None, {"score": 0.25}]}
    projected["values"].append(3)
    assert immutable_json(projected) != frozen
    assert immutable_json(json_projection(frozen)) == frozen


@pytest.mark.parametrize("value", [Path("x"), {1: "x"}, float("nan"), {"x"}, object()])
def test_immutable_json_rejects_non_json_values(value):
    for convert in (immutable_json, json_projection):
        with pytest.raises((TypeError, ValueError)):
            convert(value)
