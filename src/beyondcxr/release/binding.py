"""Immutable, path-neutral coordinates derived from validated restored scientific campaign state."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.symile_families import (
    FINAL_PACKAGE_COUNT,
    SYMILE_CORE_DEVELOPMENT_FAMILIES,
)

RESULT_BINDING_SCHEMA_VERSION = 1
_HEX40 = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ScienceExecutionCoordinates:
    git_commit: str


@dataclass(frozen=True)
class DataCoordinates:
    bundle_id: str
    split_assignment_id: str
    cv_assignment_id: str


@dataclass(frozen=True)
class DevelopmentCoordinates:
    core_analysis_id: str
    family_development_ids: tuple[tuple[str, str], ...]
    ecg_development_id: str
    ecg_extension_result_id: str


@dataclass(frozen=True)
class HeldOutCoordinates:
    pretest_freeze_id: str
    final_package_ids: tuple[str, ...]
    prediction_ids: tuple[str, ...]
    global_result_id: str


@dataclass(frozen=True)
class ResultBinding:
    """Deeply immutable coordinates for one validated scientific evidence chain."""

    science_execution: ScienceExecutionCoordinates
    data: DataCoordinates
    development: DevelopmentCoordinates
    held_out: HeldOutCoordinates

    def to_document(self) -> dict[str, Any]:
        document = asdict(self)
        document["development"]["family_development_ids"] = dict(
            self.development.family_development_ids
        )
        return {
            "result_binding_schema_version": RESULT_BINDING_SCHEMA_VERSION,
            "science_execution": document["science_execution"],
            "data": document["data"],
            "development": document["development"],
            "held_out": document["held_out"],
        }

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()


def load_result_binding(path: str | Path) -> ResultBinding:
    """Load a canonical schema-1 binding without resolving operational paths."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ManifestBuildError("Result binding must be one regular file")
    try:
        raw = source.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Result binding is unreadable") from exc
    binding = validate_result_binding(document)
    if raw != binding.canonical_bytes():
        raise ManifestBuildError("Result binding is not canonically serialized")
    return binding


def validate_result_binding(document: object) -> ResultBinding:
    """Validate an in-memory binding and freeze every nested coordinate."""
    top = _mapping(
        document,
        {
            "result_binding_schema_version",
            "science_execution",
            "data",
            "development",
            "held_out",
        },
    )
    version = top["result_binding_schema_version"]
    if type(version) is not int or version != RESULT_BINDING_SCHEMA_VERSION:
        raise ManifestBuildError("Result binding contract is invalid")
    science = _mapping(top["science_execution"], {"git_commit"})
    if not _fullmatch(_HEX40, science["git_commit"]):
        raise ManifestBuildError("Result binding Git provenance is invalid")
    data = _mapping(
        top["data"],
        {"bundle_id", "split_assignment_id", "cv_assignment_id"},
    )
    _prefixed(data["bundle_id"], "bundle-")
    _prefixed(data["split_assignment_id"], "split-assignment-")
    _prefixed(data["cv_assignment_id"], "cv-assignment-")
    development = _mapping(
        top["development"],
        {
            "core_analysis_id",
            "family_development_ids",
            "ecg_development_id",
            "ecg_extension_result_id",
        },
    )
    _prefixed(development["core_analysis_id"], "analysis-")
    _prefixed(development["ecg_development_id"], "development-")
    _prefixed(development["ecg_extension_result_id"], "ecg-extension-result-")
    families = _mapping(
        development["family_development_ids"], set(SYMILE_CORE_DEVELOPMENT_FAMILIES)
    )
    ordered_families = tuple(
        (family, _identity_value(families[family], "development-"))
        for family in SYMILE_CORE_DEVELOPMENT_FAMILIES
    )
    if len({identity for _, identity in ordered_families}) != len(ordered_families):
        raise ManifestBuildError("Result binding development identities are duplicated")
    held = _mapping(
        top["held_out"],
        {"pretest_freeze_id", "final_package_ids", "prediction_ids", "global_result_id"},
    )
    _prefixed(held["pretest_freeze_id"], "pretest-freeze-")
    _prefixed(held["global_result_id"], "global-result-")
    packages = _identities(held["final_package_ids"], "final-package-", FINAL_PACKAGE_COUNT)
    predictions = _identities(held["prediction_ids"], "prediction-", FINAL_PACKAGE_COUNT)
    if len(set(packages)) != len(packages) or len(set(predictions)) != len(predictions):
        raise ManifestBuildError("Result binding held-out identities are duplicated")
    _reject_path_values(top)
    return ResultBinding(
        science_execution=ScienceExecutionCoordinates(**science),
        data=DataCoordinates(**data),
        development=DevelopmentCoordinates(
            core_analysis_id=development["core_analysis_id"],
            family_development_ids=ordered_families,
            ecg_development_id=development["ecg_development_id"],
            ecg_extension_result_id=development["ecg_extension_result_id"],
        ),
        held_out=HeldOutCoordinates(
            pretest_freeze_id=held["pretest_freeze_id"],
            final_package_ids=packages,
            prediction_ids=predictions,
            global_result_id=held["global_result_id"],
        ),
    )


def _mapping(value: object, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ManifestBuildError("Result binding fields are invalid")
    return dict(value)


def _prefixed(value: object, prefix: str) -> None:
    _identity_value(value, prefix)


def _identity_value(value: object, prefix: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(rf"{re.escape(prefix)}[0-9a-f]{{64}}", value) is None
    ):
        raise ManifestBuildError(f"Result binding {prefix} identity is invalid")
    return value


def _fullmatch(pattern: re.Pattern[str], value: object) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _identities(value: object, prefix: str, count: int) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != count:
        raise ManifestBuildError("Result binding ordered membership is invalid")
    return tuple(_identity_value(identity, prefix) for identity in value)


def _reject_path_values(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if "path" in str(key).lower() or "root" in str(key).lower():
                raise ManifestBuildError("Operational paths do not belong in result binding")
            _reject_path_values(item)
    elif isinstance(value, Sequence) and not isinstance(value, str):
        for item in value:
            _reject_path_values(item)
    elif isinstance(value, str) and (value.startswith(("/", "~", "file:")) or "\\" in value):
        raise ManifestBuildError("Result binding contains a filesystem path")
