"""Hash source files and logical scientific tables deterministically."""

from __future__ import annotations

import hashlib
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from beyondcxr.utils.semantic_identity import canonical_json_bytes

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Hash file bytes with SHA-256."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LogicalTableContract:
    """Meaning-bearing schema, ordering, and key constraints for one table."""

    schema_name: str
    schema_version: int
    schema: pa.Schema
    sort_key: tuple[str, ...]
    unique_key: tuple[str, ...]


def logical_table_sha256(table: pa.Table, contract: LogicalTableContract) -> str:
    """Hash validated logical values independently of Arrow/Parquet representation."""
    if not isinstance(table, pa.Table):
        raise TypeError("Logical table hashing requires a pyarrow.Table")
    _validate_contract(contract)
    expected = tuple((field.name, field.type, field.nullable) for field in contract.schema)
    observed = tuple((field.name, field.type, field.nullable) for field in table.schema)
    if observed != expected:
        raise ValueError("Logical table schema differs from its scientific contract")
    for field, column in zip(contract.schema, table.columns, strict=True):
        if not field.nullable and column.null_count:
            raise ValueError(
                f"Logical table non-nullable column contains null values: {field.name}"
            )
    canonical_sort_key = (
        *contract.sort_key,
        *(column for column in contract.unique_key if column not in contract.sort_key),
    )
    canonical = table.combine_chunks()
    # Sort the same normalized text that the semantic encoding hashes.
    for name in canonical_sort_key:
        index = canonical.schema.get_field_index(name)
        field = canonical.schema.field(index)
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            canonical = canonical.set_column(
                index,
                field,
                pa.array(
                    [
                        None if value is None else unicodedata.normalize("NFC", value)
                        for value in canonical.column(index).to_pylist()
                    ],
                    type=field.type,
                ),
            )
    canonical = canonical.sort_by([(column, "ascending") for column in canonical_sort_key])
    key_rows = canonical.select(list(contract.unique_key)).to_pylist()
    keys = [canonical_json_bytes(row) for row in key_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Logical table contains duplicate canonical keys")
    schema = {
        "columns": [
            {
                "name": field.name,
                "nullable": field.nullable,
                "type": _arrow_type(field.type),
            }
            for field in contract.schema
        ],
        "row_count": canonical.num_rows,
        "sort_key": list(contract.sort_key),
        "unique_key": list(contract.unique_key),
    }
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "domain": f"logical-table:{contract.schema_name}",
                "schema": schema,
                "schema_version": contract.schema_version,
            }
        )
    )
    columns = [column.to_pylist() for column in canonical.columns]
    for row_index in range(canonical.num_rows):
        row = [
            _logical_scalar(field.type, column[row_index])
            for field, column in zip(contract.schema, columns, strict=True)
        ]
        digest.update(canonical_json_bytes(row))
    return digest.hexdigest()


def _arrow_type(value: pa.DataType) -> dict[str, object]:
    if pa.types.is_boolean(value):
        return {"kind": "bool"}
    if pa.types.is_integer(value):
        return {"kind": "int", "bits": value.bit_width, "signed": pa.types.is_signed_integer(value)}
    if pa.types.is_floating(value):
        return {"kind": "float", "bits": value.bit_width}
    if pa.types.is_string(value):
        return {"kind": "string"}
    if pa.types.is_large_string(value):
        return {"kind": "large_string"}
    if pa.types.is_binary(value):
        return {"kind": "binary"}
    if pa.types.is_large_binary(value):
        return {"kind": "large_binary"}
    raise TypeError(f"Unsupported logical Arrow type: {value}")


def _logical_scalar(data_type: pa.DataType, value: object) -> object:
    if value is None:
        return {"null": True}
    if pa.types.is_boolean(data_type):
        return {"bool": bool(value)}
    if pa.types.is_integer(data_type):
        return {"int": int(value)}
    if pa.types.is_floating(data_type):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Logical table hashing rejects non-finite floating-point values")
        return {"float": 0.0 if number == 0.0 else number}
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        if type(value) is not str:
            raise TypeError("Logical table string value has an invalid type")
        return {"string": unicodedata.normalize("NFC", value)}
    if pa.types.is_binary(data_type) or pa.types.is_large_binary(data_type):
        return {"binary": bytes(value).hex()}
    raise TypeError(f"Unsupported logical Arrow type: {data_type}")


def _validate_contract(contract: LogicalTableContract) -> None:
    if (
        type(contract.schema_name) is not str
        or not contract.schema_name
        or type(contract.schema_version) is not int
        or contract.schema_version <= 0
        or not isinstance(contract.schema, pa.Schema)
        or not contract.sort_key
        or not contract.unique_key
    ):
        raise ValueError("Logical table contract is invalid")
    names = tuple(contract.schema.names)
    if (
        len(names) != len(set(names))
        or any(column not in names for column in (*contract.sort_key, *contract.unique_key))
        or len(contract.sort_key) != len(set(contract.sort_key))
        or len(contract.unique_key) != len(set(contract.unique_key))
    ):
        raise ValueError("Logical table key contract is invalid")
