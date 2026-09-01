"""Derive, exercise, and publish serving authority for preserved scientific evidence."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.release.binding import ResultBinding, load_result_binding
from beyondcxr.release.filesystem import directory_bytes
from beyondcxr.release.inputs import derive_result_binding, export_manifest_sha256
from beyondcxr.release.reproduction import reproduce_results_from_campaign
from beyondcxr.release.resolution import resolved_release
from beyondcxr.serving.api import create_app
from beyondcxr.serving.authority import (
    LAB_KEYS,
    RESEARCH_WARNING,
    SERVING_AUTHORITY_PREFIX,
    SERVING_DEPLOYMENT_SCHEMA_VERSION,
    ValidatedServingAuthority,
    load_serving_distribution,
    publish_loaded_serving_distribution,
    reauthenticate_serving_authority_integrity,
    require_serving_authority,
    validate_loaded_serving_authority,
    validate_serving_authority,
)
from beyondcxr.serving.predictor import SymileServingPredictor
from beyondcxr.training.symile_campaign import (
    RestoredCampaignClosure,
)
from beyondcxr.training.symile_final_packages import ValidatedFinalNeuralRuntime
from beyondcxr.utils.operational_logging import observed_operation
from beyondcxr.utils.provenance import git_revision
from beyondcxr.utils.publication import publish_bytes_no_replace, rename_directory_no_replace
from beyondcxr.utils.semantic_identity import canonical_json_bytes, semantic_sha256


@dataclass(frozen=True)
class ValidatedServingPublication:
    authority: ValidatedServingAuthority
    deployment_provenance: Path
    result_binding: Path
    release_archive: Path


@dataclass(frozen=True)
class _LoadedServingPublication:
    publication: ValidatedServingPublication
    runtime_members: tuple[ValidatedFinalNeuralRuntime, ...]


_SERVING_CONTEXT_GUARD = object()


class ResolvedServingContext:
    """Opaque process-local serving runtime owned by one active restoration."""

    __slots__ = ("__active", "__guard", "__predictor", "__process_id", "__publication")

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("ResolvedServingContext is created only by serving resolution")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("ResolvedServingContext is immutable")

    def _require_active(self) -> None:
        if (
            self.__guard is not _SERVING_CONTEXT_GUARD
            or not self.__active
            or self.__process_id != os.getpid()
        ):
            raise ManifestBuildError("Resolved serving context lifetime has ended")

    @property
    def predictor(self) -> SymileServingPredictor:
        self._require_active()
        return self.__predictor

    @property
    def publication(self) -> ValidatedServingPublication:
        self._require_active()
        return self.__publication


@observed_operation("resolve_accepted_serving_state")
def resolve_accepted_serving_state(
    *,
    artifact_root: str | Path,
    authority_root: str | Path,
) -> ValidatedServingPublication:
    """Resolve the sole recursively validated installed serving publication."""
    with resolved_release(artifact_root) as (archive, campaign):
        publications = _accepted_serving_publications(
            authority_root=Path(authority_root),
            archive=archive,
            campaign=campaign,
        )
        return _sole_serving_publication([publication.publication for publication in publications])


@contextmanager
def resolve_accepted_serving_context(
    *, artifact_root: str | Path, authority_root: str | Path
) -> Iterator[ResolvedServingContext]:
    """Own one restoration, authority relationship, and three-model serving runtime."""
    with resolved_release(artifact_root) as (archive, campaign):
        loaded_publication = _sole_serving_publication(
            _accepted_serving_publications(
                authority_root=Path(authority_root),
                archive=archive,
                campaign=campaign,
            )
        )
        publication = loaded_publication.publication
        predictor = SymileServingPredictor.from_validated_runtime(
            publication.authority,
            loaded_publication.runtime_members,
            deployment_provenance=publication.deployment_provenance,
            result_binding=publication.result_binding,
            release_archive=publication.release_archive,
            device="cpu",
        )
        context = object.__new__(ResolvedServingContext)
        object.__setattr__(context, "_ResolvedServingContext__guard", _SERVING_CONTEXT_GUARD)
        object.__setattr__(context, "_ResolvedServingContext__process_id", os.getpid())
        object.__setattr__(context, "_ResolvedServingContext__active", True)
        object.__setattr__(context, "_ResolvedServingContext__publication", publication)
        object.__setattr__(context, "_ResolvedServingContext__predictor", predictor)
        try:
            yield context
        finally:
            predictor.close()
            object.__setattr__(context, "_ResolvedServingContext__active", False)


def _accepted_serving_publications(
    *,
    authority_root: Path,
    archive: Path,
    campaign: RestoredCampaignClosure,
) -> list[_LoadedServingPublication]:
    if authority_root.is_symlink() or not authority_root.is_dir():
        raise ManifestBuildError("No accepted serving authority exists")
    publications: list[_LoadedServingPublication] = []
    for authority_path in authority_root.iterdir():
        if (
            not authority_path.name.startswith(SERVING_AUTHORITY_PREFIX)
            or authority_path.is_symlink()
            or not authority_path.is_dir()
        ):
            continue
        binding = authority_root / "release-evidence" / authority_path.name / "binding.json"
        provenance_root = authority_root / "deployment-provenance" / authority_path.name
        if provenance_root.is_symlink() or not provenance_root.is_dir():
            continue
        provenances = tuple(
            provenance
            for provenance in provenance_root.iterdir()
            if not provenance.is_symlink() and provenance.is_file() and provenance.suffix == ".json"
        )
        if not provenances:
            continue
        try:
            distribution = load_serving_distribution(authority_path)
        except (ManifestBuildError, OSError, ValueError):
            continue
        for provenance in provenances:
            try:
                authority, members = validate_loaded_serving_authority(
                    distribution,
                    deployment_provenance=provenance,
                    result_binding=binding,
                    release_archive=archive,
                    restored_campaign=campaign,
                )
            except (ManifestBuildError, OSError, ValueError):
                continue
            publications.append(
                _LoadedServingPublication(
                    ValidatedServingPublication(
                        authority,
                        provenance,
                        binding,
                        archive,
                    ),
                    members,
                )
            )
    return publications


def _sole_serving_publication[T](publications: list[T]) -> T:
    if not publications:
        raise ManifestBuildError("No valid accepted serving authority exists")
    if len(publications) != 1:
        raise ManifestBuildError("Multiple valid accepted serving authorities are ambiguous")
    return publications[0]


def _prepare_validated_campaign_serving_authority(
    *,
    authority_root: str | Path,
    campaign: RestoredCampaignClosure,
    release_archive: Path,
    serving_release: dict[str, str],
    result_binding: ResultBinding | None = None,
) -> _LoadedServingPublication:
    """Prepare a complete serving publication in a caller-owned staging root."""
    restored = campaign
    derived_binding = derive_result_binding(restored)
    if result_binding is not None and result_binding != derived_binding:
        raise ManifestBuildError("Release result binding differs from restored campaign")
    binding = derived_binding if result_binding is None else result_binding
    archive_sha256 = _file_sha256(release_archive)
    if archive_sha256 != restored.restoration_context.archive_sha256:
        raise ManifestBuildError("Serving release archive differs from restored campaign")
    distribution = publish_loaded_serving_distribution(
        authority_root=authority_root,
        capability=restored.freeze,
        global_result=restored.global_result,
        global_predictions=restored.predictions,
        global_test_data=restored.test_data,
        final_packages=restored.packages,
    )
    release = _validated_release_provenance(serving_release)
    execution = restored.execution.manifest
    document: dict[str, object] = {
        "schema_version": SERVING_DEPLOYMENT_SCHEMA_VERSION,
        "serving_authority_id": distribution.authority_id,
        "result_binding_sha256": hashlib.sha256(binding.canonical_bytes()).hexdigest(),
        "preservation_archive_sha256": archive_sha256,
        "pretest_freeze_id": restored.freeze.freeze_id,
        "global_result_id": restored.global_result.result_id,
        "development_decision_id": restored.development_decision.decision_id,
        "serving_package_ids": [
            member["package_id"] for member in distribution.manifest["members"]
        ],
        "primary_thresholds": dict(distribution.manifest["primary_thresholds"]),
        "execution": {
            "execution_id": restored.execution.execution_id,
            "git_commit": execution["git_commit"],
            "dependency_lock_sha256": execution["dependency_lock_sha256"],
        },
        "release": release,
    }
    deployment_id = semantic_sha256(
        "serving-deployment-provenance",
        SERVING_DEPLOYMENT_SCHEMA_VERSION,
        document,
    )
    document["deployment_id"] = deployment_id
    path = (
        Path(authority_root)
        / "deployment-provenance"
        / distribution.authority_id
        / f"{deployment_id}.json"
    )
    encoded = canonical_json_bytes(document)
    if publish_bytes_no_replace(path, encoded) is False and path.read_bytes() != encoded:
        raise ManifestBuildError("Serving deployment provenance conflicts with existing state")
    binding_path = (
        Path(authority_root) / "release-evidence" / distribution.authority_id / "binding.json"
    )
    binding_bytes = binding.canonical_bytes()
    if (
        publish_bytes_no_replace(binding_path, binding_bytes) is False
        and binding_path.read_bytes() != binding_bytes
    ):
        raise ManifestBuildError("Serving result binding conflicts with existing state")
    authority, members = validate_loaded_serving_authority(
        distribution,
        deployment_provenance=path,
        result_binding=binding_path,
        release_archive=release_archive,
        restored_campaign=restored,
    )
    return _LoadedServingPublication(
        ValidatedServingPublication(authority, path, binding_path, release_archive),
        members,
    )


def publish_accepted_serving_state(
    *,
    candidate_authority: str | Path,
    candidate_deployment_provenance: str | Path,
    candidate_result_binding: str | Path,
    release_archive: str | Path,
    authority_root: str | Path,
    restored_campaign: RestoredCampaignClosure | None = None,
    validated_candidate: ValidatedServingAuthority | None = None,
) -> ValidatedServingPublication:
    """Validate and failure-safely publish one complete accepted serving state."""
    candidate_directory = Path(candidate_authority)
    candidate_provenance = Path(candidate_deployment_provenance)
    candidate_binding = Path(candidate_result_binding)
    archive = Path(release_archive)
    destination_root = Path(authority_root)
    if validated_candidate is None:
        candidate = validate_serving_authority(
            candidate_directory,
            deployment_provenance=candidate_provenance,
            result_binding=candidate_binding,
            release_archive=archive,
            restored_campaign=restored_campaign,
        )
    else:
        candidate = require_serving_authority(validated_candidate)
        if candidate.directory.resolve() != candidate_directory.resolve():
            raise ManifestBuildError("Validated serving candidate belongs to a different path")
        reauthenticate_serving_authority_integrity(candidate)
    final_directory = destination_root / candidate_directory.name
    final_provenance = (
        destination_root
        / "deployment-provenance"
        / candidate_directory.name
        / candidate_provenance.name
    )
    final_binding = (
        destination_root / "release-evidence" / candidate_directory.name / "binding.json"
    )
    authority_exists = final_directory.exists() or final_directory.is_symlink()
    binding_exists = final_binding.exists() or final_binding.is_symlink()
    provenance_exists = final_provenance.exists() or final_provenance.is_symlink()
    interrupted = not authority_exists and provenance_exists
    if interrupted:
        if (
            final_provenance.is_symlink()
            or not final_provenance.is_file()
            or final_provenance.read_bytes() != candidate_provenance.read_bytes()
            or (
                binding_exists
                and (
                    final_binding.is_symlink()
                    or not final_binding.is_file()
                    or final_binding.read_bytes() != candidate_binding.read_bytes()
                )
            )
        ):
            raise ManifestBuildError("Existing serving publication differs from candidate")
    reuse_core = authority_exists or (binding_exists and not interrupted)
    if reuse_core:
        if (
            not authority_exists
            or not binding_exists
            or final_directory.is_symlink()
            or not final_directory.is_dir()
            or final_binding.is_symlink()
            or not final_binding.is_file()
        ):
            raise ManifestBuildError("Existing serving publication is incomplete")
        if (
            directory_bytes(final_directory) != directory_bytes(candidate_directory)
            or final_binding.read_bytes() != candidate_binding.read_bytes()
        ):
            raise ManifestBuildError("Existing serving publication differs from candidate")
        if provenance_exists:
            if (
                final_provenance.is_symlink()
                or not final_provenance.is_file()
                or final_provenance.read_bytes() != candidate_provenance.read_bytes()
            ):
                raise ManifestBuildError("Existing serving publication differs from candidate")
            existing = validate_serving_authority(
                final_directory,
                deployment_provenance=final_provenance,
                result_binding=final_binding,
                release_archive=archive,
                restored_campaign=restored_campaign,
            )
            if existing.authority_id != candidate.authority_id:
                raise ManifestBuildError("Existing serving publication differs from candidate")
            return ValidatedServingPublication(existing, final_provenance, final_binding, archive)
        if (
            _validated_existing_deployment(
                authority=final_directory,
                provenance_root=final_provenance.parent,
                binding=final_binding,
                archive=archive,
                restored_campaign=restored_campaign,
            )
            is None
        ):
            raise ManifestBuildError("Existing serving publication is incomplete")
    elif not interrupted and (
        provenance_exists
        or (final_provenance.parent.is_dir() and any(final_provenance.parent.iterdir()))
    ):
        raise ManifestBuildError("Existing serving publication is incomplete")

    candidate_provenance_bytes = candidate_provenance.read_bytes()
    candidate_binding_bytes = candidate_binding.read_bytes()
    possible_directories = (
        destination_root,
        destination_root / "deployment-provenance",
        final_provenance.parent,
        destination_root / "release-evidence",
        final_binding.parent,
    )
    absent_directories = tuple(
        directory
        for directory in possible_directories
        if not directory.exists() and not directory.is_symlink()
    )
    created_provenance = False
    created_binding = False
    created_authority = False
    try:
        if not interrupted:
            created_provenance = publish_bytes_no_replace(
                final_provenance, candidate_provenance_bytes
            )
            if not created_provenance:
                raise ManifestBuildError(
                    "Serving deployment provenance appeared during publication"
                )
        if not reuse_core:
            if not binding_exists:
                created_binding = publish_bytes_no_replace(final_binding, candidate_binding_bytes)
                if not created_binding:
                    raise ManifestBuildError("Serving result binding appeared during publication")
            destination_root.mkdir(parents=True, exist_ok=True)
            rename_directory_no_replace(candidate_directory, final_directory)
            created_authority = True
        published = validate_serving_authority(
            final_directory,
            deployment_provenance=final_provenance,
            result_binding=final_binding,
            release_archive=archive,
            restored_campaign=restored_campaign,
        )
        if published.authority_id != candidate.authority_id:
            raise ManifestBuildError("Published serving authority differs from candidate")
        return ValidatedServingPublication(published, final_provenance, final_binding, archive)
    except BaseException:
        if created_authority:
            shutil.rmtree(final_directory)
        if created_binding:
            final_binding.unlink(missing_ok=True)
        if created_provenance:
            final_provenance.unlink(missing_ok=True)
        for directory in reversed(absent_directories):
            try:
                directory.rmdir()
            except FileNotFoundError:
                pass
            except OSError:
                continue
        raise


def _validated_existing_deployment(
    *,
    authority: Path,
    provenance_root: Path,
    binding: Path,
    archive: Path,
    restored_campaign: RestoredCampaignClosure | None,
) -> Path | None:
    """Return one valid existing deployment for a reusable authority/binding pair."""
    if provenance_root.is_symlink() or not provenance_root.is_dir():
        return None
    candidates = sorted(provenance_root.iterdir())
    if not candidates:
        return None
    try:
        distribution = load_serving_distribution(authority)
    except (ManifestBuildError, OSError, ValueError):
        return None
    for provenance in candidates:
        if provenance.is_symlink() or not provenance.is_file():
            raise ManifestBuildError("Existing serving publication differs from candidate")
        try:
            validate_loaded_serving_authority(
                distribution,
                deployment_provenance=provenance,
                result_binding=binding,
                release_archive=archive,
                restored_campaign=restored_campaign,
            )
        except (ManifestBuildError, OSError, ValueError):
            continue
        return provenance
    return None


def _validated_release_provenance(value: object) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"git_commit", "dependency_lock_sha256"}
        or not _git_commit(value["git_commit"])
        or not _sha256(value["dependency_lock_sha256"])
    ):
        raise ManifestBuildError("Serving release provenance is invalid")
    return {key: str(value[key]) for key in ("git_commit", "dependency_lock_sha256")}


def _git_commit(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)
    )


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ManifestBuildError("Serving release archive is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@observed_operation("publish_and_smoke_test_serving_authority")
def publish_and_smoke_test_serving_authority(
    *, artifact_root: str | Path, authority_root: str | Path, repository_root: str | Path
) -> Path:
    """Validate restored packages, exercise predictor and API, then publish the
    serving authority."""
    with resolved_release(artifact_root) as (archive, campaign):
        destination_root = Path(authority_root)
        repository = Path(repository_root).resolve()
        if destination_root.is_symlink():
            raise ManifestBuildError("Serving authority root must not be a symlink")
        if destination_root.resolve().is_relative_to(repository):
            raise ManifestBuildError("Serving authority root must be outside the repository")
        release = clean_release_provenance(repository)
        binding_path = repository / "results" / "symile" / "binding.json"
        release_binding = load_result_binding(binding_path)
        destination_root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=destination_root.parent, prefix=".beyondcxr-authority-"
        ) as temporary:
            candidate_root = Path(temporary) / "authorities"

            def publish() -> Path:
                restored_root = campaign.restored_root
                reproduce_results_from_campaign(
                    binding=release_binding,
                    campaign=campaign,
                    restored_root=restored_root,
                    export_manifest_sha256=export_manifest_sha256(archive),
                    output_root=repository / "results",
                    readme_path=repository / "README.md",
                    model_card_path=repository / "docs/model_card.md",
                )
                loaded_publication = _prepare_validated_campaign_serving_authority(
                    authority_root=candidate_root,
                    campaign=campaign,
                    release_archive=archive,
                    serving_release=release,
                    result_binding=release_binding,
                )
                publication = loaded_publication.publication
                authority = publication.authority
                predictor = SymileServingPredictor.from_validated_runtime(
                    authority,
                    loaded_publication.runtime_members,
                    deployment_provenance=publication.deployment_provenance,
                    result_binding=publication.result_binding,
                    release_archive=archive,
                    device="cpu",
                )
                try:
                    if predictor.authority.authority_id != authority.authority_id:
                        raise ManifestBuildError("Serving predictor loaded a different authority")
                    asyncio.run(
                        _exercise_api(
                            authority.directory,
                            deployment_provenance=publication.deployment_provenance,
                            result_binding=publication.result_binding,
                            release_archive=archive,
                            expected_model_info=predictor.model_info(),
                            predictor=predictor,
                        )
                    )
                finally:
                    predictor.close()
                accepted = publish_accepted_serving_state(
                    candidate_authority=authority.directory,
                    candidate_deployment_provenance=publication.deployment_provenance,
                    candidate_result_binding=publication.result_binding,
                    release_archive=archive,
                    authority_root=destination_root,
                    restored_campaign=campaign,
                    validated_candidate=authority,
                )
                return accepted.authority.directory

            return publish()


def clean_release_provenance(repository_root: str | Path) -> dict[str, str]:
    root = Path(repository_root).resolve()
    try:
        commit, dirty = git_revision(root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestBuildError(
            "Serving release provenance requires a valid Git checkout"
        ) from exc
    if dirty:
        raise ManifestBuildError("Serving authority publication requires a clean release candidate")
    return {
        "git_commit": commit,
        "dependency_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
    }


async def _exercise_api(
    authority_path: Path,
    *,
    deployment_provenance: Path,
    result_binding: Path,
    release_archive: Path,
    expected_model_info: dict[str, object],
    predictor: SymileServingPredictor,
) -> None:
    import httpx

    image, labs = synthetic_serving_request()
    application = create_app(
        authority_path=authority_path,
        deployment_provenance=deployment_provenance,
        result_binding=result_binding,
        release_archive=release_archive,
        device="cpu",
        predictor=predictor,
    )
    transport = httpx.ASGITransport(app=application)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=transport, base_url="http://release.test") as client:
            health = await client.get("/health")
            model_info = await client.get("/model-info")
            prediction = await client.post(
                "/predict",
                files=[
                    ("image", ("synthetic.jpg", image, "image/jpeg")),
                    ("view_position", (None, "AP")),
                    ("labs", (None, json.dumps(labs))),
                ],
            )
    validate_serving_api_responses(
        health.status_code,
        health.json(),
        model_info.status_code,
        model_info.json(),
        prediction.status_code,
        prediction.json(),
        authority_path.name,
        expected_model_info=expected_model_info,
        expected_missing_labs=[key for key, value in labs.items() if value is None],
    )


def synthetic_serving_request() -> tuple[bytes, dict[str, float | None]]:
    """Build the deterministic synthetic checkerboard serving request."""
    rows, columns = np.indices((360, 480))
    pixels = (((rows // 24) + (columns // 24)) % 2 * 255).astype(np.uint8)
    encoded = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(encoded, format="JPEG", quality=90)
    labs = {key: (None if index % 11 == 0 else float(index)) for index, key in enumerate(LAB_KEYS)}
    return encoded.getvalue(), labs


def validate_serving_api_responses(
    health_status: int,
    health: object,
    model_status: int,
    model: object,
    prediction_status: int,
    prediction: object,
    authority_id: str,
    *,
    expected_model_info: dict[str, object],
    expected_missing_labs: list[str],
) -> None:
    """Validate the three endpoint responses used by host and container acceptance."""
    if (
        health_status != 200
        or not isinstance(health, dict)
        or set(health)
        != {
            "status",
            "serving_authority_id",
        }
    ):
        raise ManifestBuildError("Synthetic serving health request failed")
    if health != {"status": "ready", "serving_authority_id": authority_id}:
        raise ManifestBuildError("Synthetic health response differs from the serving authority")
    if model_status != 200 or model != expected_model_info:
        raise ManifestBuildError("Synthetic model-info response is invalid")
    if prediction_status != 200 or not isinstance(prediction, dict):
        raise ManifestBuildError("Synthetic serving prediction request failed")
    if set(prediction) != {
        "task",
        "probability",
        "missing_labs",
        "serving_authority_id",
        "warning",
    }:
        raise ManifestBuildError("Synthetic response schema is invalid")
    probability = prediction.get("probability")
    if (
        type(probability) not in (int, float)
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise ManifestBuildError("Synthetic response probability is invalid")
    if (
        prediction.get("task") != "pneumonia_strict"
        or prediction.get("serving_authority_id") != authority_id
        or prediction.get("warning") != RESEARCH_WARNING
        or prediction.get("missing_labs") != expected_missing_labs
    ):
        raise ManifestBuildError("Synthetic response differs from the serving authority")
