"""Derive, exercise, and publish serving authority for preserved scientific evidence."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.release.filesystem import directory_bytes
from beyondcxr.serving.api import create_app
from beyondcxr.serving.authority import (
    LAB_KEYS,
    RESEARCH_WARNING,
    publish_serving_authority,
    validate_serving_authority,
)
from beyondcxr.serving.predictor import SymileServingPredictor
from beyondcxr.training.symile_campaign import validate_restored_campaign
from beyondcxr.training.symile_export import restore_and_validate_symile_export


def publish_and_smoke_test_serving_authority(
    *, artifact_root: str | Path, authority_root: str | Path, repository_root: str | Path
) -> Path:
    """Validate restored packages, exercise predictor and API, then publish the
    serving authority."""
    destination_root = Path(authority_root)
    repository = Path(repository_root).resolve()
    if destination_root.is_symlink():
        raise ManifestBuildError("Serving authority root must not be a symlink")
    if destination_root.resolve().is_relative_to(repository):
        raise ManifestBuildError("Serving authority root must be outside the repository")
    release = clean_release_provenance(repository)
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination_root.parent, prefix=".beyondcxr-authority-"
    ) as temporary:
        candidate_root = Path(temporary) / "authorities"

        def publish(restored_root: Path) -> Path:
            campaign = validate_restored_campaign(restored_root)
            authority = publish_serving_authority(
                authority_root=candidate_root,
                capability=campaign.freeze,
                global_result=campaign.global_result,
                global_predictions=campaign.predictions,
                global_test_data=campaign.test_data,
                final_packages=campaign.packages,
                serving_release=release,
            )
            package_root = campaign.packages[0].directory.parent
            predictor = SymileServingPredictor.load(
                authority.directory, package_root=package_root, device="cpu"
            )
            if predictor.authority.authority_id != authority.authority_id:
                raise ManifestBuildError("Serving predictor loaded a different authority")
            asyncio.run(
                _exercise_api(
                    authority.directory,
                    package_root,
                    expected_model_info=predictor.model_info(),
                )
            )
            destination_root.mkdir(parents=True, exist_ok=True)
            final = destination_root / authority.directory.name
            if final.exists() or final.is_symlink():
                existing = validate_serving_authority(final, package_root=package_root)
                if existing.authority_id != authority.authority_id or directory_bytes(
                    final
                ) != directory_bytes(authority.directory):
                    raise ManifestBuildError("Existing serving authority differs from candidate")
            else:
                os.replace(authority.directory, final)
            return final

        return restore_and_validate_symile_export(artifact_root, restoration_validator=publish)


def clean_release_provenance(repository_root: str | Path) -> dict[str, str]:
    root = Path(repository_root).resolve()
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestBuildError(
            "Serving release provenance requires a valid Git checkout"
        ) from exc
    if status:
        raise ManifestBuildError("Serving authority publication requires a clean release candidate")
    return {
        "git_commit": commit,
        "dependency_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
    }


async def _exercise_api(
    authority_path: Path,
    package_root: Path,
    *,
    expected_model_info: dict[str, object],
) -> None:
    import httpx

    image, labs = synthetic_serving_request()
    application = create_app(authority_path=authority_path, package_root=package_root, device="cpu")
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
