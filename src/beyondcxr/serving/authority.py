"""Semantic distribution authority for the primary Symile serving ensemble."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from beyondcxr.data.cxr_transforms import CXR_TRANSFORM_POLICY_VERSION
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_preprocess import LAB_ECDF_POLICY_VERSION
from beyondcxr.data.symile_schemas import LAB_ITEM_IDS, LABEL_POLICY_VERSION, TASK_ID
from beyondcxr.release.binding import ResultBinding, load_result_binding
from beyondcxr.training.preservation import restore_and_validate_export
from beyondcxr.training.symile_campaign_control import (
    CertifiedGlobalResult,
    ValidatedPretestFreeze,
    require_certified_global_result,
    validate_global_result_integrity,
    validated_pretest_freeze_manifest,
)
from beyondcxr.training.symile_families import FINAL_NEURAL_MEMBER_SEEDS
from beyondcxr.training.symile_final_packages import (
    ValidatedFinalPackage,
    validate_final_package,
)
from beyondcxr.training.symile_test_data import FrozenSymileTestData
from beyondcxr.utils.immutable_json import immutable_json
from beyondcxr.utils.private_predictions import ValidatedPredictionEvidence
from beyondcxr.utils.publication import install_immutable_directory, staging_directory
from beyondcxr.utils.semantic_identity import (
    canonical_json_bytes,
    canonical_scientific_id,
    semantic_sha256,
)

if TYPE_CHECKING:
    from beyondcxr.training.symile_campaign import RestoredCampaignClosure

SERVING_AUTHORITY_PREFIX = "serving-authority-"
SERVING_AUTHORITY_SCHEMA_VERSION = 1
SERVING_DEPLOYMENT_SCHEMA_VERSION = 1
PRIMARY_FAMILY = "cxr_labs_gated"
ENSEMBLE_POLICY = "ordered-seed-17-42-2026-mean-logit-then-sigmoid-v1"
TRANSPORT_POLICY = "symile-jpeg-ap-pa-exact-50-labs-v1"
SERVING_SPATIAL_POLICY = "symile-jpeg-spatial-short-side-320-center-crop-bilinear-v1"
RESEARCH_WARNING = (
    "Research use only. BeyondCXR is not intended for clinical decision-making, is not a medical "
    "device, does not replace physician judgment, and predicts a radiology-derived Pneumonia "
    "finding rather than confirmed infectious pneumonia."
)
LAB_KEYS = tuple(f"lab_{item_id}" for item_id in LAB_ITEM_IDS)
_GUARD = object()


@dataclass(frozen=True)
class ServingDistributionIntegrity:
    """Integrity observation for a serving manifest and its package distribution."""

    directory: Path
    manifest: Mapping[str, Any]
    packages: tuple[ValidatedFinalPackage, ...]

    @property
    def authority_id(self) -> str:
        return str(self.manifest["serving_authority_id"])


class ValidatedServingAuthority:
    """Opaque immutable authority minted only by the authenticated release join."""

    __slots__ = ("__directory", "__manifest", "__packages", "__guard")

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("ValidatedServingAuthority is created only by release validation")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("ValidatedServingAuthority is immutable")

    def __copy__(self) -> ValidatedServingAuthority:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> ValidatedServingAuthority:
        memo[id(self)] = self
        return self

    @property
    def directory(self) -> Path:
        return self.__directory

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self.__manifest

    @property
    def packages(self) -> tuple[ValidatedFinalPackage, ...]:
        return self.__packages

    @property
    def authority_id(self) -> str:
        return str(self.__manifest["serving_authority_id"])


def publish_serving_authority(
    *,
    authority_root: str | Path,
    capability: ValidatedPretestFreeze,
    global_result: CertifiedGlobalResult,
    global_predictions: Sequence[ValidatedPredictionEvidence],
    global_test_data: FrozenSymileTestData,
    final_packages: Sequence[ValidatedFinalPackage],
) -> ServingDistributionIntegrity:
    """Publish and validate the primary three-member serving distribution."""
    frozen = validated_pretest_freeze_manifest(capability)
    global_reference = _validated_global_reference(
        global_result,
        capability=capability,
        predictions=global_predictions,
        final_packages=final_packages,
        test_data=global_test_data,
    )
    packages = _select_primary_packages(final_packages, frozen)
    semantic = _semantic_document(capability, packages, global_reference)
    authority_id = canonical_scientific_id(
        SERVING_AUTHORITY_PREFIX,
        semantic,
        domain="serving-authority",
        schema_version=SERVING_AUTHORITY_SCHEMA_VERSION,
    )
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
        package_directory = stage / "packages"
        package_directory.mkdir()
        for package in packages:
            shutil.copytree(package.directory, package_directory / package.package_id)

        install_immutable_directory(stage, destination, validate_serving_distribution)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return validate_serving_distribution(destination)


def validate_serving_distribution(
    directory: str | Path,
    *,
    enforce_directory_name: bool = True,
) -> ServingDistributionIntegrity:
    """Observe one self-contained manifest and recursively validate its package copies."""
    root = Path(directory)
    if (
        root.is_symlink()
        or not root.is_dir()
        or {item.name for item in root.iterdir()} != {"manifest.json", "packages"}
    ):
        raise ManifestBuildError("Serving authority directory is invalid")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ManifestBuildError("Serving authority manifest is invalid")
    try:
        document = json.loads(manifest_path.read_bytes())
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
        "distribution",
        "warning",
    }
    version = (
        document.get("serving_authority_schema_version") if isinstance(document, dict) else None
    )
    if (
        not isinstance(document, dict)
        or set(document) != required
        or type(version) is not int
        or version != SERVING_AUTHORITY_SCHEMA_VERSION
    ):
        raise ManifestBuildError("Serving authority contract is invalid")
    semantic = {
        key: document[key]
        for key in required - {"serving_authority_schema_version", "serving_authority_id"}
    }
    expected_id = canonical_scientific_id(
        SERVING_AUTHORITY_PREFIX,
        semantic,
        domain="serving-authority",
        schema_version=SERVING_AUTHORITY_SCHEMA_VERSION,
    )
    if document["serving_authority_id"] != expected_id or (
        enforce_directory_name and root.name != expected_id
    ):
        raise ManifestBuildError("Serving authority semantic identity is invalid")
    _validate_document_fields(document)
    package_root = root / "packages"
    expected_package_ids = {member["package_id"] for member in document["members"]}
    if (
        package_root.is_symlink()
        or not package_root.is_dir()
        or {item.name for item in package_root.iterdir()} != expected_package_ids
        or any(item.is_symlink() or not item.is_dir() for item in package_root.iterdir())
    ):
        raise ManifestBuildError("Serving package distribution membership is invalid")
    packages = tuple(
        validate_final_package(
            package_root / member["package_id"],
            expected_package_id=member["package_id"],
        )
        for member in document["members"]
    )
    _validate_package_members(packages, document)
    return ServingDistributionIntegrity(root, document, packages)


def validate_serving_authority(
    directory: str | Path,
    *,
    deployment_provenance: str | Path,
    result_binding: str | Path,
    release_archive: str | Path,
    enforce_directory_name: bool = True,
    restored_campaign: RestoredCampaignClosure | None = None,
) -> ValidatedServingAuthority:
    """Authenticate a serving distribution against release-owned campaign evidence."""
    distribution = validate_serving_distribution(
        directory,
        enforce_directory_name=enforce_directory_name,
    )
    binding = load_result_binding(result_binding)
    deployment = _validate_deployment_provenance(
        Path(deployment_provenance), distribution.authority_id
    )
    archive = Path(release_archive)
    archive_sha256 = _regular_file_sha256(archive)
    if deployment["preservation_archive_sha256"] != archive_sha256:
        raise ManifestBuildError("Serving authority does not match release-owned campaign evidence")
    if restored_campaign is None:
        from beyondcxr.training.symile_campaign import validate_restored_campaign

        def validate_restored_release(restored_root: Path) -> None:
            _validate_release_relationship(
                distribution,
                deployment,
                binding,
                campaign=validate_restored_campaign(restored_root),
                release_archive_sha256=archive_sha256,
            )

        try:
            restore_and_validate_export(
                archive,
                restoration_validator=validate_restored_release,
            )
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise ManifestBuildError("Canonical release archive evidence is invalid") from exc
    else:
        _validate_release_relationship(
            distribution,
            deployment,
            binding,
            campaign=restored_campaign,
            release_archive_sha256=archive_sha256,
        )
    # This unchecked allocation is deliberately lexical to the terminal successful
    # validation point. There is no reusable distribution-to-authority factory.
    authority = object.__new__(ValidatedServingAuthority)
    object.__setattr__(
        authority,
        "_ValidatedServingAuthority__directory",
        distribution.directory,
    )
    object.__setattr__(
        authority,
        "_ValidatedServingAuthority__manifest",
        immutable_json(distribution.manifest),
    )
    packages = tuple(
        ValidatedFinalPackage(
            package.directory,
            immutable_json(package.manifest),
        )
        for package in distribution.packages
    )
    object.__setattr__(authority, "_ValidatedServingAuthority__packages", packages)
    object.__setattr__(authority, "_ValidatedServingAuthority__guard", _GUARD)
    return authority


def require_serving_authority(value: ValidatedServingAuthority) -> ValidatedServingAuthority:
    if (
        not isinstance(value, ValidatedServingAuthority)
        or value._ValidatedServingAuthority__guard is not _GUARD
    ):
        raise ManifestBuildError("A genuine validated serving authority is required")
    return value


def _validate_deployment_provenance(path: Path, authority_id: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Serving deployment provenance is unreadable") from exc
    required = {
        "schema_version",
        "deployment_id",
        "serving_authority_id",
        "result_binding_sha256",
        "preservation_archive_sha256",
        "pretest_freeze_id",
        "global_result_id",
        "development_decision_id",
        "serving_package_ids",
        "primary_thresholds",
        "execution",
        "release",
    }
    if (
        path.is_symlink()
        or not path.is_file()
        or not isinstance(document, dict)
        or set(document) != required
        or type(document["schema_version"]) is not int
        or document["schema_version"] != SERVING_DEPLOYMENT_SCHEMA_VERSION
        or document["serving_authority_id"] != authority_id
        or raw != canonical_json_bytes(document)
    ):
        raise ManifestBuildError("Serving deployment provenance contract is invalid")
    semantic = {key: value for key, value in document.items() if key != "deployment_id"}
    expected = semantic_sha256(
        "serving-deployment-provenance", SERVING_DEPLOYMENT_SCHEMA_VERSION, semantic
    )
    if document["deployment_id"] != expected:
        raise ManifestBuildError("Serving deployment provenance identity is invalid")
    execution = document["execution"]
    release = document["release"]
    if (
        not _sha256(document["result_binding_sha256"])
        or not _sha256(document["preservation_archive_sha256"])
        or not _identity(document["pretest_freeze_id"], "pretest-freeze-")
        or not _identity(document["global_result_id"], "global-result-")
        or not _identity(document["development_decision_id"], "development-decision-")
        or not isinstance(document["serving_package_ids"], list)
        or len(document["serving_package_ids"]) != 3
        or any(
            not _identity(package_id, "final-package-")
            for package_id in document["serving_package_ids"]
        )
        or not isinstance(document["primary_thresholds"], dict)
        or set(document["primary_thresholds"]) != {"youden_j", "target_sensitivity"}
        or any(
            isinstance(threshold, bool)
            or not isinstance(threshold, int | float)
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
            for threshold in document["primary_thresholds"].values()
        )
        or not isinstance(execution, dict)
        or set(execution) != {"execution_id", "git_commit", "dependency_lock_sha256"}
        or not isinstance(execution["execution_id"], str)
        or not execution["execution_id"].startswith("symile-execution-")
        or not _git_commit(execution["git_commit"])
        or not _sha256(execution["dependency_lock_sha256"])
        or not isinstance(release, dict)
        or set(release) != {"git_commit", "dependency_lock_sha256"}
        or not _git_commit(release["git_commit"])
        or not _sha256(release["dependency_lock_sha256"])
    ):
        raise ManifestBuildError("Serving deployment provenance is invalid")
    return document


def _validate_release_relationship(
    distribution: ServingDistributionIntegrity,
    deployment: Mapping[str, Any],
    binding: ResultBinding,
    *,
    campaign: RestoredCampaignClosure,
    release_archive_sha256: str,
) -> None:
    from beyondcxr.release.inputs import derive_result_binding
    from beyondcxr.training.symile_campaign import require_restored_campaign

    restored = require_restored_campaign(campaign)
    manifest = distribution.manifest
    package_ids = tuple(member["package_id"] for member in manifest["members"])
    thresholds = dict(manifest["primary_thresholds"])
    expected_binding = derive_result_binding(restored)
    expected_packages = _select_primary_packages(
        restored.packages,
        restored.freeze.manifest,
    )
    expected_package_ids = tuple(package.package_id for package in expected_packages)
    expected_thresholds = dict(restored.development_decision.manifest["primary_thresholds"])
    execution = restored.execution.manifest
    expected_execution = {
        "execution_id": restored.execution.execution_id,
        "git_commit": execution["git_commit"],
        "dependency_lock_sha256": execution["dependency_lock_sha256"],
    }
    if (
        deployment["result_binding_sha256"] != hashlib.sha256(binding.canonical_bytes()).hexdigest()
        or deployment["preservation_archive_sha256"] != release_archive_sha256
        or restored.restoration_context.archive_sha256 != release_archive_sha256
        or manifest["bundle"]["bundle_id"] != binding.data.bundle_id
        or manifest["global_result"]["pretest_freeze_id"] != binding.held_out.pretest_freeze_id
        or manifest["global_result"]["global_result_id"] != binding.held_out.global_result_id
        or deployment["pretest_freeze_id"] != binding.held_out.pretest_freeze_id
        or deployment["global_result_id"] != binding.held_out.global_result_id
        or deployment["development_decision_id"] != binding.development.development_decision_id
        or tuple(deployment["serving_package_ids"]) != package_ids
        or package_ids != expected_package_ids
        or deployment["primary_thresholds"] != thresholds
        or binding != expected_binding
        or thresholds != expected_thresholds
        or deployment["execution"] != expected_execution
    ):
        raise ManifestBuildError("Serving authority does not match release-owned campaign evidence")


def _regular_file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ManifestBuildError("Canonical release archive is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    frozen_ids = {item["package_id"] for item in frozen["final_packages"]}
    if any(package.package_id not in frozen_ids for package in ordered):
        raise ManifestBuildError("Serving packages differ from the pre-test freeze")
    return ordered


def _validated_global_reference(
    result: CertifiedGlobalResult,
    *,
    capability: ValidatedPretestFreeze,
    predictions: Sequence[ValidatedPredictionEvidence],
    final_packages: Sequence[ValidatedFinalPackage],
    test_data: FrozenSymileTestData,
) -> dict[str, str]:
    result = require_certified_global_result(result)
    validated = validate_global_result_integrity(
        result.directory,
        capability=capability,
        predictions=predictions,
        final_packages=final_packages,
        test_data=test_data,
    )
    if not result.matches_integrity(validated):
        raise ManifestBuildError("Serving global-result artifact changed after certification")
    path = validated.directory / "manifest.json"
    try:
        raw = path.read_bytes()
        observed = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Serving global-result provenance is unreadable") from exc
    if (
        immutable_json(observed) != immutable_json(validated.manifest)
        or observed.get("global_result_id") != validated.result_id
        or observed.get("pretest_freeze_id") != capability.freeze_id
    ):
        raise ManifestBuildError("Serving global-result provenance is invalid")
    return {
        "global_result_id": validated.result_id,
        "pretest_freeze_id": capability.freeze_id,
    }


def _semantic_document(
    capability: ValidatedPretestFreeze,
    packages: Sequence[ValidatedFinalPackage],
    global_reference: Mapping[str, str],
) -> dict[str, object]:
    frozen = capability.manifest
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
                "model_state_sha256": package.manifest["model_state_sha256"],
                "preprocessor_state_sha256": package.manifest["preprocessor_state_sha256"],
            }
            for package in packages
        ],
        "ensemble_policy": ENSEMBLE_POLICY,
        "bundle": {"bundle_id": frozen["bundle_id"]},
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
        "primary_thresholds": dict(capability.development_decision.manifest["primary_thresholds"]),
        "global_result": dict(global_reference),
        "distribution": {
            "classification": "restricted trained scientific artifacts",
            "public_image_embedding": False,
            "package_copies_embedded": True,
        },
        "warning": RESEARCH_WARNING,
    }


def _validate_document_fields(document: Mapping[str, Any]) -> None:
    thresholds = document["primary_thresholds"]
    members = document["members"]
    global_result = document["global_result"]
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
                "model_state_sha256",
                "preprocessor_state_sha256",
            }
            or not _identity(member["package_id"], "final-package-")
            or any(
                not _sha256(member[key])
                for key in (
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
        or set(document["bundle"]) != {"bundle_id"}
        or not _identity(document["bundle"].get("bundle_id"), "bundle-")
        or not isinstance(global_result, Mapping)
        or set(global_result) != {"global_result_id", "pretest_freeze_id"}
        or not _identity(global_result.get("global_result_id"), "global-result-")
        or not _identity(global_result.get("pretest_freeze_id"), "pretest-freeze-")
        or document["distribution"]
        != {
            "classification": "restricted trained scientific artifacts",
            "public_image_embedding": False,
            "package_copies_embedded": True,
        }
    ):
        raise ManifestBuildError("Serving authority fields are invalid")


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
