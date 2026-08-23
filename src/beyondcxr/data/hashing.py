"""Hash source files and Arrow artifacts deterministically."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import pyarrow as pa

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Hash file bytes with SHA-256."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def logical_arrow_sha256(table: pa.Table) -> str:
    """Hash ordered logical Arrow schema, nulls, and scalar values without IPC serialization."""
    if not isinstance(table, pa.Table):
        raise TypeError("Logical Arrow hashing requires a pyarrow.Table")
    digest = hashlib.sha256(b"beyondcxr-logical-arrow-v1\0")
    schema = {
        "fields": [
            {
                "name": field.name,
                "nullable": field.nullable,
                "type": _arrow_type(field.type),
                "metadata": _arrow_metadata(field.metadata),
            }
            for field in table.schema
        ],
        "metadata": _arrow_metadata(table.schema.metadata),
    }
    digest.update(
        json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(struct.pack("<Q", table.num_rows))
    for field, column in zip(table.schema, table.columns, strict=True):
        digest.update(field.name.encode("utf-8"))
        digest.update(b"\0")
        for value in column.combine_chunks().to_pylist():
            _update_arrow_scalar(digest, field.type, value)
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


def _arrow_metadata(value: dict[bytes, bytes] | None) -> list[list[str]]:
    if value is None:
        return []
    return [[key.hex(), item.hex()] for key, item in sorted(value.items())]


def _update_arrow_scalar(digest: Any, data_type: pa.DataType, value: object) -> None:
    if value is None:
        digest.update(b"N")
        return
    digest.update(b"V")
    if pa.types.is_boolean(data_type):
        digest.update(b"\x01" if value else b"\x00")
        return
    if pa.types.is_integer(data_type):
        width = data_type.bit_width // 8
        digest.update(
            int(value).to_bytes(
                width,
                "little",
                signed=pa.types.is_signed_integer(data_type),
            )
        )
        return
    if pa.types.is_float32(data_type):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Logical Arrow hashing rejects non-finite floating-point values")
        digest.update(struct.pack("<f", number))
        return
    if pa.types.is_float64(data_type):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Logical Arrow hashing rejects non-finite floating-point values")
        digest.update(struct.pack("<d", number))
        return
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        encoded = str(value).encode("utf-8")
    elif pa.types.is_binary(data_type) or pa.types.is_large_binary(data_type):
        encoded = bytes(value)
    else:
        raise TypeError(f"Unsupported logical Arrow type: {data_type}")
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)
