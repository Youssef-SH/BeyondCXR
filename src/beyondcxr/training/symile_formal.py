"""Canonical reviewed run plan for formal Symile execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from beyondcxr.training.symile_families import (
    SYMILE_CORE_DEVELOPMENT_FAMILIES,
    SYMILE_ECG_GATED_FAMILY,
)


@dataclass(frozen=True)
class SymileFormalRunSpec:
    """One required reviewed family/config coordinate."""

    family_id: str
    config_relative: Path


SYMILE_FORMAL_RUN_PLAN = tuple(
    SymileFormalRunSpec(family, Path("configs") / f"symile_{family}.yaml")
    for family in (*SYMILE_CORE_DEVELOPMENT_FAMILIES, SYMILE_ECG_GATED_FAMILY)
)
