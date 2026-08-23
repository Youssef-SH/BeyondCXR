from __future__ import annotations

import pyarrow as pa
import pytest

from beyondcxr.data.hashing import logical_arrow_sha256


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
