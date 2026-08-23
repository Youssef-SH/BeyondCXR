from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from symile_campaign_test_support import (
    _final_provenance,
    _freeze,
    _projection_for_freeze,
    _synthetic_test_predictions,
)
from symile_campaign_test_support import (
    _synthetic_final_packages as _all_synthetic_final_packages,
)

import beyondcxr.training.symile_campaign_control as campaign_control
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_preprocess import (
    LAB_FEATURE_COLUMNS,
    SymileLabEcdfTransformer,
)
from beyondcxr.models.symile_fusion import build_symile_gated_model
from beyondcxr.serving.api import create_app
from beyondcxr.serving.authority import LAB_KEYS, publish_serving_authority
from beyondcxr.serving.predictor import SymileServingPredictor
from beyondcxr.training.config import load_symile_development_config
from beyondcxr.training.symile_campaign_control import (
    PRETEST_FREEZE_PREFIX,
    ValidatedPretestFreeze,
)
from beyondcxr.training.symile_final_packages import (
    ValidatedFinalPackage,
    final_training_plan,
    publish_final_neural_package,
)
from beyondcxr.training.symile_test_data import FrozenSymileTestData
from beyondcxr.utils.package_identity import canonical_scientific_id


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _lab_frame() -> pd.DataFrame:
    values: dict[str, object] = {}
    for index, column in enumerate(LAB_FEATURE_COLUMNS[:50]):
        values[column] = [float(index), float(index + 1)]
    for column in LAB_FEATURE_COLUMNS[50:]:
        values[column] = [True, True]
    return pd.DataFrame(values, columns=LAB_FEATURE_COLUMNS)


def _synthetic_packages(tmp_path: Path) -> tuple[ValidatedFinalPackage, ...]:
    config = load_symile_development_config("configs/symile_cxr_labs_gated.yaml")
    config = replace(
        config,
        dataset=replace(
            config.dataset,
            bundle_id="bundle-" + "1" * 64,
            bundle_manifest_sha256="2" * 64,
            split_assignment_id="split-assignment-" + "3" * 64,
            cv_assignment_id="cv-assignment-" + "4" * 64,
        ),
    )
    preprocessor = SymileLabEcdfTransformer().fit(_lab_frame())
    packages = []
    for seed in (17, 42, 2026):
        torch.manual_seed(seed)
        model = build_symile_gated_model(config.family.parameters, weights=None)
        packages.append(
            publish_final_neural_package(
                model_root=tmp_path / "synthetic-packages",
                config=config,
                family_development_id="development-" + "1" * 64,
                plan=final_training_plan("cxr_labs_gated", 1),
                seed=seed,
                state_dict=model.state_dict(),
                lab_preprocessor=preprocessor,
                source_cxr_package_id="final-package-" + f"{seed:064x}",
                pretrained_weight=None,
                operational=_final_provenance(neural=True),
            )
        )
    return tuple(packages)


def _freeze_for_packages(
    tmp_path: Path, packages: tuple[ValidatedFinalPackage, ...]
) -> ValidatedPretestFreeze:
    base = _freeze(tmp_path / "base-freeze")
    semantic = {
        key: value
        for key, value in base.manifest.items()
        if key not in {"pretest_freeze_schema_version", "pretest_freeze_id"}
    }
    dataset = packages[0].manifest["input"]["dataset"]
    semantic["bundle"] = {
        "bundle_id": dataset["bundle_id"],
        "bundle_manifest_sha256": packages[0].manifest["bundle_manifest_sha256"],
        "split_assignment_id": dataset["split_assignment_id"],
    }
    semantic["task"] = dict(packages[0].manifest["input"]["task"])
    references = list(semantic["final_packages"])
    references[2:5] = [
        {
            "package_id": package.manifest["source_cxr_package_id"],
            "manifest_sha256": "5" * 64,
        }
        for package in packages
    ]
    references[8:11] = [
        {"package_id": package.package_id, "manifest_sha256": package.manifest_sha256}
        for package in packages
    ]
    semantic["final_packages"] = references
    freeze_id = canonical_scientific_id(PRETEST_FREEZE_PREFIX, semantic)
    document = {
        "pretest_freeze_schema_version": 1,
        "pretest_freeze_id": freeze_id,
        **semantic,
    }
    directory = tmp_path / "synthetic-freeze" / freeze_id
    directory.mkdir(parents=True)
    path = directory / "manifest.json"
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    raw = path.read_bytes()
    return ValidatedPretestFreeze(
        directory,
        json.loads(raw),
        hashlib.sha256(raw).hexdigest(),
        campaign_control._CAPABILITY_GUARD,
    )


def _canonical_global_result(
    tmp_path: Path,
    freeze: ValidatedPretestFreeze,
    primary_packages: tuple[ValidatedFinalPackage, ...],
    monkeypatch: pytest.MonkeyPatch,
):
    packages = list(_all_synthetic_final_packages(tmp_path / "all-packages", freeze))
    packages[8:11] = primary_packages
    all_packages = tuple(packages)
    predictions = _synthetic_test_predictions(tmp_path / "predictions", freeze, all_packages)
    package_by_path = {package.directory: package for package in all_packages}
    evidence_by_path = {evidence.directory: evidence for evidence in predictions}
    monkeypatch.setattr(
        campaign_control,
        "validate_final_package",
        lambda directory, **kwargs: package_by_path[Path(directory)],
    )
    monkeypatch.setattr(
        campaign_control,
        "validate_prediction_evidence",
        lambda directory, **kwargs: evidence_by_path[Path(directory)],
    )
    monkeypatch.setattr(
        campaign_control,
        "cluster_bootstrap_effect",
        lambda frame, **kwargs: {"subject_count": int(frame["subject_id"].nunique())},
    )
    projection = _projection_for_freeze(freeze, grouped=False)
    test_data = object.__new__(FrozenSymileTestData)
    test_data.bundle = type("Bundle", (), {"bundle_id": projection.bundle_id})()
    test_data.frame = projection.frame()
    test_data.split_assignment_id = projection.split_assignment_id
    test_data._capability = freeze
    result = campaign_control.publish_global_result(
        report_root=tmp_path / "synthetic-reports",
        capability=freeze,
        predictions=predictions,
        final_packages=all_packages,
        test_data=test_data,
    )
    return result, all_packages, predictions, test_data


def _jpeg() -> bytes:
    pixels = ((np.indices((360, 480)).sum(axis=0) % 2) * 255).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(output, format="JPEG", quality=90)
    return output.getvalue()


@pytest.mark.anyio
async def test_synthetic_package_authority_reconstruction_and_api_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packages = _synthetic_packages(tmp_path)
    freeze = _freeze_for_packages(tmp_path, packages)
    global_result, all_packages, predictions, test_data = _canonical_global_result(
        tmp_path,
        freeze,
        packages,
        monkeypatch,
    )
    authority = publish_serving_authority(
        authority_root=tmp_path / "synthetic-authorities",
        capability=freeze,
        global_result=global_result,
        global_predictions=predictions,
        global_test_data=test_data,
        final_packages=all_packages,
        serving_release={
            "git_commit": "6" * 40,
            "dependency_lock_sha256": "7" * 64,
        },
    )
    predictor = SymileServingPredictor.load(
        authority.directory,
        package_root=packages[0].directory.parent,
    )
    application = create_app(
        authority_path=authority.directory,
        package_root=packages[0].directory.parent,
    )
    transport = httpx.ASGITransport(app=application)
    labs = {key: float(index) for index, key in enumerate(LAB_KEYS)}
    files = [
        ("image", ("synthetic.jpg", _jpeg(), "image/jpeg")),
        ("view_position", (None, "AP")),
        ("labs", (None, json.dumps(labs))),
    ]
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post("/predict", files=files)
    assert response.status_code == 200
    probability = response.json()["probability"]
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0
    assert response.json()["serving_authority_id"] == authority.authority_id
    assert [package.manifest["seed_policy"] for package in predictor.authority.packages] == [
        17,
        42,
        2026,
    ]

    (global_result.directory / "claims.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="claims do not rederive"):
        publish_serving_authority(
            authority_root=tmp_path / "rejected-authorities",
            capability=freeze,
            global_result=global_result,
            global_predictions=predictions,
            global_test_data=test_data,
            final_packages=all_packages,
            serving_release={
                "git_commit": "6" * 40,
                "dependency_lock_sha256": "7" * 64,
            },
        )
