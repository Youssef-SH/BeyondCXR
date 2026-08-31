from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from beyondcxr.data.hashing import LogicalTableContract, logical_table_sha256
from beyondcxr.utils.package_identity import tensor_state_sha256
from beyondcxr.utils.semantic_identity import (
    canonical_json_bytes,
    canonical_scientific_id,
    semantic_sha256,
)

_SCHEMA = pa.schema(
    [
        pa.field("sample", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=True),
        pa.field("observed", pa.bool_(), nullable=False),
    ]
)


_CONTRACT = LogicalTableContract("test.measurements", 1, _SCHEMA, ("sample",), ("sample",))


def _table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {"sample": "a", "value": 1.5, "observed": True},
            {"sample": "b", "value": None, "observed": False},
        ],
        schema=_SCHEMA,
    )


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


def test_logical_table_hash_is_representation_and_order_invariant(tmp_path: Path) -> None:
    contiguous = _table()
    reversed_rows = pa.Table.from_pylist(contiguous.to_pylist()[::-1], schema=_SCHEMA)
    chunked = pa.Table.from_arrays(
        [
            pa.chunked_array([["a"], ["b"]], type=pa.string()),
            pa.chunked_array([[1.5], [None]], type=pa.float64()),
            pa.chunked_array([[True], [False]], type=pa.bool_()),
        ],
        schema=_SCHEMA,
    )
    annotated = contiguous.replace_schema_metadata({b"writer": b"irrelevant"})
    zstd = tmp_path / "zstd.parquet"
    snappy = tmp_path / "snappy.parquet"
    pq.write_table(contiguous, zstd, compression="zstd", row_group_size=1)
    pq.write_table(contiguous, snappy, compression="snappy")

    expected = logical_table_sha256(contiguous, _CONTRACT)
    assert logical_table_sha256(reversed_rows, _CONTRACT) == expected
    assert logical_table_sha256(chunked, _CONTRACT) == expected
    assert logical_table_sha256(annotated, _CONTRACT) == expected
    assert logical_table_sha256(pq.read_table(zstd), _CONTRACT) == expected
    assert logical_table_sha256(pq.read_table(snappy), _CONTRACT) == expected


def test_logical_table_hash_uses_total_canonical_key_order() -> None:
    schema = pa.schema(
        [
            pa.field("subject_id", pa.int64(), nullable=False),
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("value", pa.int64(), nullable=False),
        ]
    )
    contract = LogicalTableContract(
        "test.repeated-subjects",
        1,
        schema,
        ("subject_id",),
        ("subject_id", "sample_id"),
    )
    rows = [
        {"subject_id": 1, "sample_id": "b", "value": 20},
        {"subject_id": 1, "sample_id": "a", "value": 10},
    ]
    first = pa.Table.from_pylist(rows, schema=schema)
    reversed_rows = pa.Table.from_pylist(rows[::-1], schema=schema)
    changed = pa.Table.from_pylist([rows[0], {**rows[1], "value": 11}], schema=schema)
    expected = logical_table_sha256(first, contract)
    assert logical_table_sha256(reversed_rows, contract) == expected
    assert logical_table_sha256(changed, contract) != expected


def test_logical_table_normalizes_unicode_keys_before_sorting() -> None:
    def table(key: str) -> pa.Table:
        return pa.Table.from_pylist(
            [
                {"sample": key, "value": 1.0, "observed": True},
                {"sample": "z", "value": 2.0, "observed": True},
            ],
            schema=_SCHEMA,
        )

    assert logical_table_sha256(table("e\u0301"), _CONTRACT) == logical_table_sha256(
        table("\u00e9"), _CONTRACT
    )
    duplicate = pa.Table.from_pylist(
        [{"sample": key, "value": 1.0, "observed": True} for key in ("e\u0301", "\u00e9")],
        schema=_SCHEMA,
    )
    with pytest.raises(ValueError, match="duplicate canonical keys"):
        logical_table_sha256(duplicate, _CONTRACT)


def test_logical_table_hash_enforces_actual_nullability() -> None:
    nonnullable = pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("value", pa.int64(), nullable=False),
        ]
    )
    nonnullable_contract = LogicalTableContract(
        "test.nullability", 1, nonnullable, ("sample_id",), ("sample_id",)
    )
    invalid = pa.Table.from_pylist([{"sample_id": "a", "value": None}], schema=nonnullable)
    with pytest.raises(ValueError, match="non-nullable column contains null values: value"):
        logical_table_sha256(invalid, nonnullable_contract)

    nullable = pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("value", pa.int64(), nullable=True),
        ]
    )
    nullable_contract = LogicalTableContract(
        "test.nullability", 1, nullable, ("sample_id",), ("sample_id",)
    )
    first = pa.Table.from_pylist(
        [{"sample_id": "b", "value": 2}, {"sample_id": "a", "value": None}],
        schema=nullable,
    )
    reversed_rows = pa.Table.from_pylist(first.to_pylist()[::-1], schema=nullable)
    assert logical_table_sha256(first, nullable_contract) == logical_table_sha256(
        reversed_rows, nullable_contract
    )


def test_logical_table_sha256_v1_golden_vector() -> None:
    assert logical_table_sha256(_table(), _CONTRACT) == (
        "6903a4194987f06986c3813d26a5b9cb3363b01621ba9ed7082bf0f7a5cd4d65"
    )


def test_tensor_state_sha256_v1_golden_vector() -> None:
    state = {"weight": torch.tensor([[1.0, -0.0]], dtype=torch.float32)}
    assert tensor_state_sha256(state) == (
        "5f14b41b3cecc8e684bc911401ec50e4c5301980e5d7dc0be1854d0200b79b9f"
    )


def test_tensor_state_hash_uses_exact_logical_bytes_not_layout_or_device() -> None:
    reconstructed = {
        "bias": torch.tensor([0.5], dtype=torch.float32),
        "weight": torch.tensor([[1.0, -0.0]], dtype=torch.float32),
    }
    independently_reconstructed = {
        "weight": torch.tensor([[1.0, -0.0]], dtype=torch.float32),
        "bias": torch.tensor([0.5], dtype=torch.float32),
    }
    assert tensor_state_sha256(reconstructed) == tensor_state_sha256(independently_reconstructed)

    noncontiguous = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).transpose(0, 1)
    assert not noncontiguous.is_contiguous()
    assert tensor_state_sha256({"weight": noncontiguous}) == tensor_state_sha256(
        {"weight": noncontiguous.contiguous()}
    )

    if torch.cuda.is_available():
        assert tensor_state_sha256(reconstructed) == tensor_state_sha256(
            {name: tensor.cuda() for name, tensor in reconstructed.items()}
        )


def test_tensor_state_normalizes_names_before_ordering_and_rejects_collisions() -> None:
    weight = torch.tensor([1.0])
    other = torch.tensor([2.0])
    assert tensor_state_sha256({"e\u0301": weight, "z": other}) == tensor_state_sha256(
        {"\u00e9": weight, "z": other}
    )
    with pytest.raises(ValueError, match="collide"):
        tensor_state_sha256({"e\u0301": weight, "\u00e9": other})


def test_tensor_state_hash_binds_name_dtype_shape_value_and_zero_sign() -> None:
    baseline = tensor_state_sha256({"weight": torch.tensor([[1.0, 0.0]])})
    assert tensor_state_sha256({"renamed": torch.tensor([[1.0, 0.0]])}) != baseline
    assert tensor_state_sha256({"weight": torch.tensor([[1.0, 0.0]]).double()}) != baseline
    assert tensor_state_sha256({"weight": torch.tensor([1.0, 0.0])}) != baseline
    assert tensor_state_sha256({"weight": torch.tensor([[2.0, 0.0]])}) != baseline
    assert tensor_state_sha256({"weight": torch.tensor([[1.0, -0.0]])}) != baseline


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_tensor_state_hash_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        tensor_state_sha256({"weight": torch.tensor([value])})


def test_logical_table_hash_binds_values_and_scientific_schema() -> None:
    changed = pa.Table.from_pylist(
        [
            {"sample": "a", "value": 2.5, "observed": True},
            {"sample": "b", "value": None, "observed": False},
        ],
        schema=_SCHEMA,
    )
    changed_contract = LogicalTableContract("test.changed", 1, _SCHEMA, ("sample",), ("sample",))
    baseline = logical_table_sha256(_table(), _CONTRACT)
    assert logical_table_sha256(changed, _CONTRACT) != baseline
    assert logical_table_sha256(_table(), changed_contract) != baseline
    filled = _table().set_column(1, _SCHEMA.field("value"), pa.array([1.5, 0.0], type=pa.float64()))
    assert logical_table_sha256(filled, _CONTRACT) != baseline
    renamed_schema = _SCHEMA.set(1, pa.field("measurement", pa.float64(), nullable=True))
    variants = (
        (_table().rename_columns(renamed_schema.names), renamed_schema),
        (
            _table().cast(_SCHEMA.set(1, pa.field("value", pa.float32()))),
            _SCHEMA.set(1, pa.field("value", pa.float32())),
        ),
        (
            _table().cast(_SCHEMA.set(2, pa.field("observed", pa.bool_(), nullable=True))),
            _SCHEMA.set(2, pa.field("observed", pa.bool_(), nullable=True)),
        ),
    )
    for table, schema in variants:
        contract = LogicalTableContract("test.measurements", 1, schema, ("sample",), ("sample",))
        assert logical_table_sha256(table, contract) != baseline
        with pytest.raises(ValueError, match="schema"):
            logical_table_sha256(table, _CONTRACT)


def test_logical_table_validation_rejects_duplicate_keys_and_nonfinite_values() -> None:
    duplicate = pa.Table.from_pylist(
        [
            {"sample": "a", "value": 1.0, "observed": True},
            {"sample": "a", "value": 2.0, "observed": True},
        ],
        schema=_SCHEMA,
    )
    with pytest.raises(ValueError, match="duplicate canonical keys"):
        logical_table_sha256(duplicate, _CONTRACT)
    for value in (float("nan"), float("inf"), float("-inf")):
        invalid = pa.Table.from_pylist(
            [{"sample": "a", "value": value, "observed": True}], schema=_SCHEMA
        )
        with pytest.raises(ValueError, match="non-finite"):
            logical_table_sha256(invalid, _CONTRACT)
