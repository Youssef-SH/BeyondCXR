"""Shared physical and manifest contract for immutable dataset bundles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from radfusion.data.errors import ManifestBuildError

BUNDLE_MANIFEST_SCHEMA_VERSION = "1"
BUNDLES_DIRECTORY = "bundles"
BUNDLE_PREFIX = "bundle-"
MANIFEST_FILENAME = "manifest.json"
SAMPLES_FILENAME = "samples.parquet"
CURRENT_FILENAME = "CURRENT"

BUNDLE_MANIFEST_FIELDS = frozenset(
    {
        "bundle_manifest_schema_version",
        "dataset",
        "tasks",
        "membership",
        "source",
        "modalities",
        "privacy",
        "bundle",
        "artifacts",
        "provenance",
        "generation",
        "qualification",
    }
)
ARTIFACT_FIELDS = frozenset({"logical_arrow_sha256", "physical_file_sha256", "row_count"})


def validate_common_bundle_envelope(
    manifest: object,
    *,
    expected_artifacts: set[str],
) -> Mapping[str, Any]:
    """Validate the vocabulary shared by every dataset bundle manifest."""
    if not isinstance(manifest, dict) or set(manifest) != BUNDLE_MANIFEST_FIELDS:
        raise ManifestBuildError("Bundle manifest field set is invalid")
    if manifest.get("bundle_manifest_schema_version") != BUNDLE_MANIFEST_SCHEMA_VERSION:
        raise ManifestBuildError("Bundle manifest schema version is unsupported")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ManifestBuildError("Bundle manifest artifact set is invalid")
    for filename, declaration in artifacts.items():
        if not isinstance(declaration, dict) or set(declaration) != ARTIFACT_FIELDS:
            raise ManifestBuildError(f"Bundle artifact declaration is invalid: {filename}")
        row_count = declaration.get("row_count")
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
            raise ManifestBuildError(f"Bundle artifact row count is invalid: {filename}")
        if not all(_sha256(value) for key, value in declaration.items() if key != "row_count"):
            raise ManifestBuildError(f"Bundle artifact hash is invalid: {filename}")
    return manifest


def valid_bundle_id(value: object) -> bool:
    """Return whether *value* is a canonical immutable bundle identity."""
    return (
        isinstance(value, str)
        and value.startswith(BUNDLE_PREFIX)
        and _sha256(value.removeprefix(BUNDLE_PREFIX))
    )


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
