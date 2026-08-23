"""Immutable non-scientific control for the primary Symile serving ensemble."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from beyondcxr.data.cxr_transforms import CXR_TRANSFORM_POLICY_VERSION
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_preprocess import LAB_ECDF_POLICY_VERSION
from beyondcxr.data.symile_schemas import LAB_ITEM_IDS, LABEL_POLICY_VERSION, TASK_ID
from beyondcxr.training.symile_campaign_control import (
    ValidatedGlobalResult,
    ValidatedPretestFreeze,
    validate_global_result,
    validated_pretest_freeze_manifest,
)
from beyondcxr.training.symile_families import FINAL_NEURAL_MEMBER_SEEDS
from beyondcxr.training.symile_final_packages import (
    ValidatedFinalPackage,
    validate_final_package,
)
from beyondcxr.training.symile_test_data import FrozenSymileTestData
from beyondcxr.utils.package_identity import canonical_scientific_id
from beyondcxr.utils.private_predictions import ValidatedPredictionEvidence
from beyondcxr.utils.publication import install_immutable_directory, staging_directory

SERVING_AUTHORITY_PREFIX = "serving-authority-"
SERVING_AUTHORITY_SCHEMA_VERSION = 1
PRIMARY_FAMILY = "cxr_labs_gated"
ENSEMBLE_POLICY = "ordered-seed-17-42-2026-mean-logit-then-sigmoid-v1"
TRANSPORT_POLICY = "symile-jpeg-ap-pa-exact-50-labs-v1"
SERVING_SPATIAL_POLICY = "symile-jpeg-spatial-short-side-320-center-crop-bilinear-v1"
RESEARCH_WARNING = (
    "Research prototype only. Not for clinical decision-making. Not a medical device or "
    "physician replacement. Predicts a radiology-derived Pneumonia finding, not confirmed "
    "infectious pneumonia."
)
LAB_KEYS = tuple(f"lab_{item_id}" for item_id in LAB_ITEM_IDS)
_GUARD = object()


@dataclass(frozen=True)
class ValidatedServingAuthority:
    """Validated control plus its strictly reconstructed package members."""

    directory: Path
    manifest: Mapping[str, Any]
    packages: tuple[ValidatedFinalPackage, ...]
    _guard: object

    @property
    def authority_id(self) -> str:
        return str(self.manifest["serving_authority_id"])


def publish_serving_authority(
    *,
    authority_root: str | Path,
    capability: ValidatedPretestFreeze,
    global_result: ValidatedGlobalResult,
    global_predictions: Sequence[ValidatedPredictionEvidence],
    global_test_data: FrozenSymileTestData,
    final_packages: Sequence[ValidatedFinalPackage],
    serving_release: Mapping[str, str],
) -> ValidatedServingAuthority:
    """Publish the exact primary three-member serving control after formal M6."""
    frozen = validated_pretest_freeze_manifest(capability)
    release = _validated_provenance(serving_release, "Serving release")
    global_reference = _validated_global_reference(
        global_result,
        capability=capability,
        predictions=global_predictions,
        final_packages=final_packages,
        test_data=global_test_data,
    )
    packages = _select_primary_packages(final_packages, frozen)
    semantic = _semantic_document(frozen, packages, global_reference, release)
    authority_id = canonical_scientific_id(SERVING_AUTHORITY_PREFIX, semantic)
    document = {
        "serving_authority_schema_version": SERVING_AUTHORITY_SCHEMA_VERSION,
        "serving_authority_id": authority_id,
        **semantic,
    }
    destination = Path(authority_root) / authority_id
    stage = staging_directory(destination)
    try:
        (stage / "manifest.json").write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )

        def validator(path: Path, **kwargs: object) -> ValidatedServingAuthority:
            return validate_serving_authority(
                path,
                package_root=packages[0].directory.parent,
                **kwargs,
            )

        install_immutable_directory(stage, destination, validator)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return validate_serving_authority(destination, package_root=packages[0].directory.parent)


def validate_serving_authority(
    directory: str | Path,
    *,
    package_root: str | Path,
    enforce_directory_name: bool = True,
) -> ValidatedServingAuthority:
    """Validate one authority and recursively validate its exact package membership."""
    root = Path(directory)
    if (
        root.is_symlink()
        or not root.is_dir()
        or {item.name for item in root.iterdir()} != {"manifest.json"}
    ):
        raise ManifestBuildError("Serving authority directory is invalid")
    try:
        document = json.loads((root / "manifest.json").read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Serving authority manifest is unreadable") from exc
    required = {
        "serving_authority_schema_version",
        "serving_authority_id",
        "task",
        "positive_class",
        "family",
        "ordered_seeds",
        "members",
        "ensemble_policy",
        "bundle",
        "preprocessing",
        "input_contract",
        "primary_thresholds",
        "global_result",
        "science_execution",
        "serving_release",
        "distribution",
        "warning",
    }
    version = (
        document.get("serving_authority_schema_version") if isinstance(document, dict) else None
    )
    if (
        not isinstance(document, dict)
        or set(document) != required
        or isinstance(version, bool)
        or version != SERVING_AUTHORITY_SCHEMA_VERSION
    ):
        raise ManifestBuildError("Serving authority contract is invalid")
    semantic = {
        key: document[key]
        for key in required - {"serving_authority_schema_version", "serving_authority_id"}
    }
    expected_id = canonical_scientific_id(SERVING_AUTHORITY_PREFIX, semantic)
    if document["serving_authority_id"] != expected_id or (
        enforce_directory_name and root.name != expected_id
    ):
        raise ManifestBuildError("Serving authority semantic identity is invalid")
    _validate_document_fields(document)
    packages = tuple(
        validate_final_package(
            Path(package_root) / member["package_id"],
            expected_package_id=member["package_id"],
        )
        for member in document["members"]
    )
    _validate_package_members(packages, document)
    return ValidatedServingAuthority(root, document, packages, _GUARD)


def require_serving_authority(value: ValidatedServingAuthority) -> ValidatedServingAuthority:
    if not isinstance(value, ValidatedServingAuthority) or value._guard is not _GUARD:
        raise ManifestBuildError("A genuine validated serving authority is required")
    return validate_serving_authority(
        value.directory,
        package_root=value.packages[0].directory.parent,
    )


def _select_primary_packages(
    packages: Sequence[ValidatedFinalPackage], frozen: Mapping[str, Any]
) -> tuple[ValidatedFinalPackage, ...]:
    candidates = tuple(
        package
        for package in packages
        if package.manifest["input"]["family"]["family_id"] == PRIMARY_FAMILY
    )
    validated = tuple(
        validate_final_package(package.directory, expected_package_id=package.package_id)
        for package in candidates
    )
    by_seed = {package.manifest["seed_policy"]: package for package in validated}
    if len(validated) != 3 or set(by_seed) != set(FINAL_NEURAL_MEMBER_SEEDS):
        raise ManifestBuildError("Serving requires the exact three primary gated packages")
    ordered = tuple(by_seed[seed] for seed in FINAL_NEURAL_MEMBER_SEEDS)
    frozen_refs = {item["package_id"]: item["manifest_sha256"] for item in frozen["final_packages"]}
    if any(frozen_refs.get(package.package_id) != package.manifest_sha256 for package in ordered):
        raise ManifestBuildError("Serving packages differ from the pre-test freeze")
    return ordered


def _validated_global_reference(
    result: ValidatedGlobalResult,
    *,
    capability: ValidatedPretestFreeze,
    predictions: Sequence[ValidatedPredictionEvidence],
    final_packages: Sequence[ValidatedFinalPackage],
    test_data: FrozenSymileTestData,
) -> dict[str, str]:
    if not isinstance(result, ValidatedGlobalResult):
        raise ManifestBuildError("Serving authority requires a validated global result")
    validated = validate_global_result(
        result.directory,
        capability=capability,
        predictions=predictions,
        final_packages=final_packages,
        test_data=test_data,
    )
    path = validated.directory / "manifest.json"
    try:
        raw = path.read_bytes()
        observed = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Serving global-result provenance is unreadable") from exc
    if (
        observed != validated.manifest
        or observed.get("global_result_id") != validated.result_id
        or observed.get("pretest_freeze_id") != capability.freeze_id
    ):
        raise ManifestBuildError("Serving global-result provenance is invalid")
    return {
        "global_result_id": validated.result_id,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "pretest_freeze_id": capability.freeze_id,
    }


def _semantic_document(
    frozen: Mapping[str, Any],
    packages: Sequence[ValidatedFinalPackage],
    global_reference: Mapping[str, str],
    serving_release: Mapping[str, str],
) -> dict[str, object]:
    return {
        "task": {
            "task_id": TASK_ID,
            "label_policy_version": LABEL_POLICY_VERSION,
        },
        "positive_class": {
            "value": 1,
            "meaning": "report-derived Pneumonia finding present",
        },
        "family": PRIMARY_FAMILY,
        "ordered_seeds": list(FINAL_NEURAL_MEMBER_SEEDS),
        "members": [
            {
                "seed": package.manifest["seed_policy"],
                "package_id": package.package_id,
                "manifest_sha256": package.manifest_sha256,
                "model_state_sha256": package.manifest["model_state_sha256"],
                "preprocessor_state_sha256": package.manifest["preprocessor_state_sha256"],
            }
            for package in packages
        ],
        "ensemble_policy": ENSEMBLE_POLICY,
        "bundle": dict(frozen["bundle"]),
        "preprocessing": {
            "spatial_policy": SERVING_SPATIAL_POLICY,
            "cxr_transform_policy": CXR_TRANSFORM_POLICY_VERSION,
            "lab_policy": LAB_ECDF_POLICY_VERSION,
            "lab_keys": list(LAB_KEYS),
            "missing_lab_value": None,
        },
        "input_contract": {
            "transport_policy": TRANSPORT_POLICY,
            "content_type": "multipart/form-data",
            "fields": ["image", "view_position", "labs"],
            "image_media_type": "image/jpeg",
            "view_positions": ["AP", "PA"],
        },
        "primary_thresholds": dict(frozen["primary_thresholds"]),
        "global_result": dict(global_reference),
        "science_execution": {
            "git_commit": frozen["science_git_commit"],
            "dependency_lock_sha256": frozen["dependency_lock_sha256"],
        },
        "serving_release": dict(serving_release),
        "distribution": {
            "classification": "restricted trained scientific artifacts",
            "public_image_embedding": False,
            "runtime_mount_required_unless_distribution_is_authorized": True,
        },
        "warning": RESEARCH_WARNING,
    }


def _validate_document_fields(document: Mapping[str, Any]) -> None:
    thresholds = document["primary_thresholds"]
    members = document["members"]
    global_result = document["global_result"]
    science_execution = document["science_execution"]
    serving_release = document["serving_release"]
    if (
        not isinstance(members, list)
        or len(members) != 3
        or any(not isinstance(member, Mapping) for member in members)
    ):
        raise ManifestBuildError("Serving authority fields are invalid")
    if (
        document["task"] != {"task_id": TASK_ID, "label_policy_version": LABEL_POLICY_VERSION}
        or document["positive_class"]
        != {"value": 1, "meaning": "report-derived Pneumonia finding present"}
        or document["family"] != PRIMARY_FAMILY
        or document["ordered_seeds"] != list(FINAL_NEURAL_MEMBER_SEEDS)
        or document["ensemble_policy"] != ENSEMBLE_POLICY
        or document["warning"] != RESEARCH_WARNING
        or [member.get("seed") for member in members] != list(FINAL_NEURAL_MEMBER_SEEDS)
        or any(
            set(member)
            != {
                "seed",
                "package_id",
                "manifest_sha256",
                "model_state_sha256",
                "preprocessor_state_sha256",
            }
            or not _identity(member["package_id"], "final-package-")
            or any(
                not _sha256(member[key])
                for key in (
                    "manifest_sha256",
                    "model_state_sha256",
                    "preprocessor_state_sha256",
                )
            )
            for member in members
        )
        or not isinstance(thresholds, Mapping)
        or set(thresholds) != {"youden_j", "target_sensitivity"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in thresholds.values()
        )
        or document["preprocessing"]
        != {
            "spatial_policy": SERVING_SPATIAL_POLICY,
            "cxr_transform_policy": CXR_TRANSFORM_POLICY_VERSION,
            "lab_policy": LAB_ECDF_POLICY_VERSION,
            "lab_keys": list(LAB_KEYS),
            "missing_lab_value": None,
        }
        or document["input_contract"]
        != {
            "transport_policy": TRANSPORT_POLICY,
            "content_type": "multipart/form-data",
            "fields": ["image", "view_position", "labs"],
            "image_media_type": "image/jpeg",
            "view_positions": ["AP", "PA"],
        }
        or not isinstance(document["bundle"], Mapping)
        or set(document["bundle"]) != {"bundle_id", "bundle_manifest_sha256", "split_assignment_id"}
        or not _identity(document["bundle"].get("bundle_id"), "bundle-")
        or not _sha256(document["bundle"].get("bundle_manifest_sha256"))
        or not _identity(document["bundle"].get("split_assignment_id"), "split-assignment-")
        or not isinstance(global_result, Mapping)
        or set(global_result) != {"global_result_id", "manifest_sha256", "pretest_freeze_id"}
        or not _identity(global_result.get("global_result_id"), "global-result-")
        or not _sha256(global_result.get("manifest_sha256"))
        or not _identity(global_result.get("pretest_freeze_id"), "pretest-freeze-")
        or not isinstance(science_execution, Mapping)
        or set(science_execution) != {"git_commit", "dependency_lock_sha256"}
        or not _git_commit(science_execution.get("git_commit"))
        or not _sha256(science_execution.get("dependency_lock_sha256"))
        or _invalid_provenance(serving_release)
        or document["distribution"]
        != {
            "classification": "restricted trained scientific artifacts",
            "public_image_embedding": False,
            "runtime_mount_required_unless_distribution_is_authorized": True,
        }
    ):
        raise ManifestBuildError("Serving authority fields are invalid")


def _validated_provenance(value: object, label: str) -> dict[str, str]:
    if _invalid_provenance(value):
        raise ManifestBuildError(f"{label} provenance is invalid")
    return {
        "git_commit": str(value["git_commit"]),
        "dependency_lock_sha256": str(value["dependency_lock_sha256"]),
    }


def _invalid_provenance(value: object) -> bool:
    return (
        not isinstance(value, Mapping)
        or set(value) != {"git_commit", "dependency_lock_sha256"}
        or not _git_commit(value.get("git_commit"))
        or not _sha256(value.get("dependency_lock_sha256"))
    )


def _validate_package_members(
    packages: Sequence[ValidatedFinalPackage], document: Mapping[str, Any]
) -> None:
    reference_input = packages[0].manifest["input"]
    reference_preprocessor = packages[0].manifest["preprocessor_state_sha256"]
    for package, member, seed in zip(
        packages, document["members"], FINAL_NEURAL_MEMBER_SEEDS, strict=True
    ):
        manifest = package.manifest
        family = manifest["input"]["family"]
        if (
            package.package_id != member["package_id"]
            or package.manifest_sha256 != member["manifest_sha256"]
            or manifest["model_state_sha256"] != member["model_state_sha256"]
            or manifest["preprocessor_state_sha256"] != member["preprocessor_state_sha256"]
            or manifest["preprocessor_state_sha256"] != reference_preprocessor
            or manifest["package_kind"] != "neural"
            or manifest["execution_scope"] != "full_development"
            or manifest["seed_policy"] != seed
            or family["family_id"] != PRIMARY_FAMILY
            or family["modalities"] != ["cxr", "labs"]
            or family["parameters"].get("use_observedness") is not True
            or family["parameters"].get("modality_count") != 2
            or manifest["input"]["preprocessing"]
            != {
                "cxr_transform_policy": CXR_TRANSFORM_POLICY_VERSION,
                "lab_policy": LAB_ECDF_POLICY_VERSION,
            }
            or manifest["input"] != reference_input
            or manifest["input"]["task"] != document["task"]
            or manifest["input"]["dataset"]
            != {
                "dataset_id": "symile",
                "bundle_id": document["bundle"]["bundle_id"],
                "split_assignment_id": document["bundle"]["split_assignment_id"],
                "cohort": "official_train_plus_validation_strict_pneumonia",
            }
        ):
            raise ManifestBuildError("Serving package membership is incompatible")


def _identity(value: object, prefix: str) -> bool:
    return isinstance(value, str) and value.startswith(prefix) and _sha256(value[len(prefix) :])


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )
