"""Canonical semantic identity primitives shared by scientific authorities."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping


def canonical_json_bytes(value: object) -> bytes:
    """Encode the deliberately small canonical JSON value domain."""
    normalized = _normalize(value)
    return (
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def semantic_sha256(domain: str, schema_version: int, payload: object) -> str:
    """Hash a canonical payload with explicit domain and schema-version framing."""
    if type(domain) is not str or not domain or unicodedata.normalize("NFC", domain) != domain:
        raise ValueError("Semantic identity domain must be non-empty NFC text")
    if type(schema_version) is not int or schema_version <= 0:
        raise ValueError("Semantic identity schema version must be a positive integer")
    framed = {
        "domain": domain,
        "payload": payload,
        "schema_version": schema_version,
    }
    return hashlib.sha256(canonical_json_bytes(framed)).hexdigest()


def canonical_scientific_id(
    prefix: str,
    payload: Mapping[str, object],
    *,
    domain: str,
    schema_version: int,
) -> str:
    """Return a prefixed semantic ID under explicit domain and schema framing."""
    if type(prefix) is not str or not prefix or not prefix.endswith("-"):
        raise ValueError("Scientific identity prefix must be non-empty and end with '-'")
    return prefix + semantic_sha256(domain, schema_version, payload)


def _normalize(value: object) -> object:
    if value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Canonical JSON rejects non-finite floating-point values")
        return 0.0 if value == 0.0 else value
    if type(value) is str:
        return unicodedata.normalize("NFC", value)
    if type(value) is list:
        return [_normalize(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("Canonical JSON requires string dictionary keys")
        normalized: dict[str, object] = {}
        for key, item in value.items():
            canonical_key = unicodedata.normalize("NFC", key)
            if canonical_key in normalized:
                raise ValueError("Canonical JSON keys collide after NFC normalization")
            normalized[canonical_key] = _normalize(item)
        return normalized
    raise TypeError(f"Canonical JSON does not support {type(value).__name__}")
