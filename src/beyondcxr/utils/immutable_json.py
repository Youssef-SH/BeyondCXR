"""Detached immutable JSON values and explicit mutable JSON projections."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from beyondcxr.utils.semantic_identity import canonical_json_bytes


def immutable_json(value: object) -> Any:
    """Freeze a JSON tree; lists and tuples share ordered sequence semantics."""
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("JSON mappings require string keys")
        return MappingProxyType({key: immutable_json(item) for key, item in value.items()})
    if type(value) in (list, tuple):
        return tuple(immutable_json(item) for item in value)
    canonical_json_bytes(value)
    return value


def json_projection(value: object) -> Any:
    """Return detached JSON containers from a mutable or immutable JSON tree."""
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("JSON mappings require string keys")
        return {key: json_projection(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [json_projection(item) for item in value]
    canonical_json_bytes(value)
    return value
