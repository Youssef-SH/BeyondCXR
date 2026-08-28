from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pytest

from beyondcxr.data.hashing import logical_arrow_sha256
from beyondcxr.utils.semantic_identity import (
    canonical_json_bytes,
    canonical_scientific_id,
    semantic_sha256,
)


def test_logical_arrow_hash_is_stable_across_chunk_layout() -> None:
    schema = pa.schema(
        [
            pa.field("sample", pa.string(), nullable=False),
            pa.field("value", pa.float64(), nullable=True),
            pa.field("observed", pa.bool_(), nullable=False),
        ]
    )
    contiguous = pa.Table.from_pylist(
        [
            {"sample": "a", "value": 1.5, "observed": True},
            {"sample": "b", "value": None, "observed": False},
        ],
        schema=schema,
    )
    chunked = pa.Table.from_arrays(
        [
            pa.chunked_array([["a"], ["b"]], type=pa.string()),
            pa.chunked_array([[1.5], [None]], type=pa.float64()),
            pa.chunked_array([[True], [False]], type=pa.bool_()),
        ],
        schema=schema,
    )

    assert logical_arrow_sha256(contiguous) == logical_arrow_sha256(chunked)


def test_logical_arrow_hash_identity_contract_is_stable() -> None:
    schema = pa.schema(
        [
            pa.field("sample", pa.string(), nullable=False),
            pa.field("value", pa.float64(), nullable=True),
            pa.field("observed", pa.bool_(), nullable=False),
        ]
    )
    table = pa.Table.from_pylist(
        [
            {"sample": "a", "value": 1.5, "observed": True},
            {"sample": "b", "value": None, "observed": False},
        ],
        schema=schema,
    )

    assert logical_arrow_sha256(table) == (
        "a1cb281ee3e11a3e7877b1130933b1c90d474837258f495e49876f0cf430be3d"
    )


def test_logical_arrow_hash_binds_schema_order_nulls_and_values() -> None:
    baseline = pa.table({"value": pa.array([0.0, None], type=pa.float64())})
    negative_zero = pa.table({"value": pa.array([-0.0, None], type=pa.float64())})
    filled = pa.table({"value": pa.array([0.0, 0.0], type=pa.float64())})
    renamed = pa.table({"changed": pa.array([0.0, None], type=pa.float64())})

    identity = logical_arrow_sha256(baseline)
    assert logical_arrow_sha256(negative_zero) != identity
    assert logical_arrow_sha256(filled) != identity
    assert logical_arrow_sha256(renamed) != identity


def test_logical_arrow_hash_binds_schema_metadata() -> None:
    plain = pa.table({"value": pa.array([1], type=pa.int64())})
    annotated = plain.replace_schema_metadata({b"contract": b"one"})

    assert logical_arrow_sha256(plain) != logical_arrow_sha256(annotated)


def test_logical_arrow_hash_rejects_nonfinite_floating_point_content() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite"):
            logical_arrow_sha256(pa.table({"value": pa.array([value], type=pa.float64())}))


def test_canonical_json_contract() -> None:
    assert canonical_json_bytes({"b": 2, "a": [1, True]}) == canonical_json_bytes(
        {"a": [1, True], "b": 2}
    )
    assert canonical_json_bytes([1, 2]) != canonical_json_bytes([2, 1])
    assert canonical_json_bytes({"value": "e\u0301"}) == canonical_json_bytes({"value": "\u00e9"})
    assert canonical_json_bytes({"value": -0.0}) == canonical_json_bytes({"value": 0.0})
    assert canonical_json_bytes({"value": True}) != canonical_json_bytes({"value": 1})


def test_canonical_json_v1_golden_vector() -> None:
    assert canonical_json_bytes({"z": ["e\u0301", True], "a": 1}) == (
        b'{"a":1,"z":["\xc3\xa9",true]}\n'
    )


def test_canonical_json_rejects_ambiguous_object_keys() -> None:
    with pytest.raises(TypeError, match="string dictionary keys"):
        canonical_json_bytes({1: "value"})
    with pytest.raises(ValueError, match="keys collide after NFC normalization"):
        canonical_json_bytes({"e\u0301": 1, "\u00e9": 2})


def test_semantic_hash_has_domain_and_schema_separation() -> None:
    payload = {"value": 1}
    assert semantic_sha256("one", 1, payload) != semantic_sha256("two", 1, payload)
    assert semantic_sha256("one", 1, payload) != semantic_sha256("one", 2, payload)


def test_scientific_id_requires_explicit_domain_and_schema_version() -> None:
    payload = {"value": 1}
    with pytest.raises(TypeError):
        canonical_scientific_id("example-", payload)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        canonical_scientific_id("example-", payload, domain="example")  # type: ignore[call-arg]


def test_semantic_sha256_v1_golden_vector() -> None:
    assert semantic_sha256("golden-example", 1, {"value": [1, "two"]}) == (
        "1c3503dbe4829b8474e071a60d674a500ddc03988b69e7413ec2d237ebcf0cab"
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json_bytes({"value": value})


@pytest.mark.parametrize(
    "value",
    [Path("a"), datetime(2026, 1, 1), Decimal("1"), {1, 2}, (1, 2)],
)
def test_canonical_json_rejects_unsupported_types(value: object) -> None:
    with pytest.raises(TypeError, match="does not support"):
        canonical_json_bytes({"value": value})
