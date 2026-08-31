from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
from symile_lifecycle_test_support import prospective_symile_mutation

import beyondcxr.serving.predictor as serving_predictor
import beyondcxr.training.symile_campaign_control as campaign_control
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_preprocess import (
    LAB_FEATURE_COLUMNS,
    SymileLabEcdfTransformer,
)
from beyondcxr.models.symile_fusion import build_symile_gated_model
from beyondcxr.release.binding import validate_result_binding
from beyondcxr.serving.api import create_app
from beyondcxr.serving.authority import (
    LAB_KEYS,
    SERVING_DEPLOYMENT_SCHEMA_VERSION,
    ValidatedServingAuthority,
    publish_serving_authority,
    validate_serving_authority,
)
from beyondcxr.serving.predictor import SymileServingPredictor
from beyondcxr.training.config import load_symile_development_config
from beyondcxr.training.preservation import PreservationMember, export_and_verify
from beyondcxr.training.symile_campaign_control import (
    PRETEST_FREEZE_PREFIX,
    ValidatedPretestFreeze,
)
from beyondcxr.training.symile_families import DEVELOPMENT_FAMILY_ORDER
from beyondcxr.training.symile_final_packages import (
    ValidatedFinalPackage,
    final_training_plan,
    publish_final_neural_package,
)
from beyondcxr.training.symile_test_data import FrozenSymileTestData
from beyondcxr.utils.semantic_identity import (
    canonical_json_bytes,
    canonical_scientific_id,
    semantic_sha256,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def bounded_cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


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


def _synthetic_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ValidatedFinalPackage, ...]:
    config = load_symile_development_config(
        "configs/symile_cxr_labs_gated.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
    config = replace(
        config,
        dataset=replace(
            config.dataset,
            bundle_id="bundle-" + "1" * 64,
            cv_assignment_id="cv-assignment-" + "4" * 64,
        ),
    )
    preprocessor = SymileLabEcdfTransformer().fit(_lab_frame())
    packages = []
    with prospective_symile_mutation(tmp_path, monkeypatch) as mutation:
        for seed in (17, 42, 2026):
            torch.manual_seed(seed)
            model = build_symile_gated_model(config.family.parameters, weights=None)
            packages.append(
                publish_final_neural_package(
                    prospective_mutation=mutation,
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
    semantic["bundle_id"] = dataset["bundle_id"]
    semantic["task"] = dict(packages[0].manifest["input"]["task"])
    references = list(semantic["final_packages"])
    references[2:5] = [
        {**references[index], "package_id": package.manifest["source_cxr_package_id"]}
        for index, package in zip(range(2, 5), packages, strict=True)
    ]
    references[8:11] = [
        {**references[index], "package_id": package.package_id}
        for index, package in zip(range(8, 11), packages, strict=True)
    ]
    semantic["final_packages"] = references
    freeze_id = canonical_scientific_id(
        PRETEST_FREEZE_PREFIX, semantic, domain="pretest-freeze", schema_version=1
    )
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
    return ValidatedPretestFreeze(
        directory,
        json.loads(path.read_bytes()),
        base.development_decision,
        (),
        campaign_control._FREEZE_GUARD,
    )


def _canonical_global_result(
    tmp_path: Path,
    freeze: ValidatedPretestFreeze,
    primary_packages: tuple[ValidatedFinalPackage, ...],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(campaign_control, "_require_capability", lambda value: freeze)
    packages = list(_all_synthetic_final_packages(tmp_path / "all-packages", freeze))
    packages[8:11] = primary_packages
    all_packages = tuple(packages)
    predictions = _synthetic_test_predictions(tmp_path / "predictions", freeze, all_packages)
    package_by_path = {package.directory: package for package in all_packages}
    package_by_id = {package.package_id: package for package in all_packages}
    evidence_by_path = {evidence.directory: evidence for evidence in predictions}
    monkeypatch.setattr(
        campaign_control,
        "validate_final_package",
        lambda directory, **kwargs: package_by_path.get(
            Path(directory), package_by_id.get(Path(directory).name)
        ),
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
    test_data._bundle = SimpleNamespace(bundle_id=projection.bundle_id)
    test_data._canonical_frame = projection.frame()
    test_data._capability = freeze
    result = campaign_control.publish_global_result(
        report_root=tmp_path / "synthetic-reports",
        capability=freeze,
        predictions=predictions,
        final_packages=all_packages,
        test_data=test_data,
    )
    return result, all_packages, predictions, test_data


def _release_evidence(
    tmp_path: Path,
    authority,
    freeze,
    packages,
    predictions,
    global_result,
) -> tuple[Path, Path, Path]:
    commit = "f" * 40
    binding = validate_result_binding(
        {
            "result_binding_schema_version": 1,
            "data": {
                "bundle_id": freeze.manifest["bundle_id"],
                "cv_assignment_id": "cv-assignment-" + "4" * 64,
            },
            "development": {
                "family_development_ids": {
                    family: "development-" + f"{index:064x}"
                    for index, family in enumerate(DEVELOPMENT_FAMILY_ORDER, start=1)
                },
                "development_decision_id": freeze.manifest["development_decision_id"],
            },
            "held_out": {
                "pretest_freeze_id": freeze.freeze_id,
                "final_package_ids": [package.package_id for package in packages],
                "prediction_ids": [prediction.prediction_id for prediction in predictions],
                "global_result_id": global_result.result_id,
            },
        }
    )
    binding_path = tmp_path / "binding.json"
    binding_path.write_bytes(binding.canonical_bytes())
    attacker_archive = _selected_document_only_archive(
        tmp_path,
        freeze=freeze,
        packages=packages,
        predictions=predictions,
        global_result=global_result,
        commit=commit,
    )
    document = {
        "schema_version": SERVING_DEPLOYMENT_SCHEMA_VERSION,
        "serving_authority_id": authority.authority_id,
        "result_binding_sha256": hashlib.sha256(binding.canonical_bytes()).hexdigest(),
        "preservation_archive_sha256": hashlib.sha256(attacker_archive.read_bytes()).hexdigest(),
        "pretest_freeze_id": freeze.freeze_id,
        "global_result_id": global_result.result_id,
        "development_decision_id": freeze.manifest["development_decision_id"],
        "serving_package_ids": [member["package_id"] for member in authority.manifest["members"]],
        "primary_thresholds": dict(authority.manifest["primary_thresholds"]),
        "execution": {
            "execution_id": "symile-execution-" + "e" * 64,
            "git_commit": commit,
            "dependency_lock_sha256": "d" * 64,
        },
        "release": {"git_commit": "c" * 40, "dependency_lock_sha256": "b" * 64},
    }
    document["deployment_id"] = semantic_sha256(
        "serving-deployment-provenance", SERVING_DEPLOYMENT_SCHEMA_VERSION, document
    )
    deployment = tmp_path / "deployment.json"
    deployment.write_bytes(canonical_json_bytes(document))
    return deployment, binding_path, attacker_archive


def _selected_document_only_archive(
    tmp_path: Path,
    *,
    freeze: ValidatedPretestFreeze,
    packages: tuple[ValidatedFinalPackage, ...],
    predictions: tuple[object, ...],
    global_result: object,
    commit: str,
) -> Path:
    """Build transport-valid evidence that omits the canonical scientific closure."""
    source = tmp_path / "selected-document-source"
    private = source / "private/control/symile"
    reports = source / "reports/symile"
    execution_id = "symile-execution-" + "e" * 64
    decision_id = freeze.manifest["development_decision_id"]
    documents = {
        private / "test-open.json": {"freeze_id": freeze.freeze_id},
        private / "active-execution.json": {
            "schema_version": 1,
            "execution_id": execution_id,
        },
        private / f"freezes/{freeze.freeze_id}/manifest.json": {
            "pretest_freeze_id": freeze.freeze_id,
            "bundle_id": freeze.manifest["bundle_id"],
            "development_decision_id": decision_id,
            "final_packages": [{"package_id": package.package_id} for package in packages],
        },
        private / f"executions/{execution_id}/manifest.json": {
            "execution_id": execution_id,
            "git_commit": commit,
            "dependency_lock_sha256": "d" * 64,
        },
        reports / f"development-decisions/{decision_id}/manifest.json": {
            "development_decision_id": decision_id,
            "cv_assignment_id": "cv-assignment-" + "4" * 64,
            "family_development_ids": [
                {
                    "family_id": family,
                    "development_id": "development-" + f"{index:064x}",
                }
                for index, family in enumerate(DEVELOPMENT_FAMILY_ORDER, start=1)
            ],
            "primary_thresholds": dict(freeze.development_decision.manifest["primary_thresholds"]),
        },
        reports / f"global-results/{global_result.result_id}/manifest.json": {
            "global_result_id": global_result.result_id,
            "pretest_freeze_id": freeze.freeze_id,
            "prediction_ids": [prediction.prediction_id for prediction in predictions],
        },
    }
    for path, document in documents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(document))
    return export_and_verify(
        members=(
            PreservationMember(source / "private", Path("private")),
            PreservationMember(source / "reports", Path("reports")),
        ),
        export_root=tmp_path / "selected-document-export",
        backup_root=tmp_path / "selected-document-backup",
        export_name="selected-document-only-release",
    )


@pytest.mark.anyio
async def test_synthetic_distribution_rejects_incomplete_release_and_runs_cpu_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packages = _synthetic_packages(tmp_path, monkeypatch)
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
    )
    deployment, binding, attacker_archive = _release_evidence(
        tmp_path,
        authority,
        freeze,
        all_packages,
        predictions,
        global_result,
    )
    assert not isinstance(authority, ValidatedServingAuthority)
    # The copied packages and selected release documents are valid and mutually
    # consistent. Terminal admission must still reject the archive because the
    # canonical completed-campaign closure cannot be reconstructed from it.
    with pytest.raises(ManifestBuildError):
        validate_serving_authority(
            authority.directory,
            deployment_provenance=deployment,
            result_binding=binding,
            release_archive=attacker_archive,
        )

    canonical_archive = tmp_path / "canonical-release.zip"
    canonical_archive.write_bytes(b"operator-selected canonical release")
    assert attacker_archive.read_bytes() != canonical_archive.read_bytes()
    with pytest.raises(ManifestBuildError, match="release"):
        validate_serving_authority(
            authority.directory,
            deployment_provenance=deployment,
            result_binding=binding,
            release_archive=canonical_archive,
        )

    # Release admission is tested above. Isolate that boundary to exercise real
    # copied-package reconstruction, preprocessing, ensemble inference, and HTTP.
    admitted = SimpleNamespace(
        authority_id=authority.authority_id,
        manifest=authority.manifest,
        packages=authority.packages,
    )
    with monkeypatch.context() as inference_patch:
        inference_patch.setattr(serving_predictor, "require_serving_authority", lambda value: value)
        predictor = SymileServingPredictor(admitted)
        assert predictor.device == torch.device("cpu")
        assert [package.manifest["seed_policy"] for package in predictor.authority.packages] == [
            17,
            42,
            2026,
        ]
        assert all(not model.training for model in predictor._models)
        inference_patch.setattr(
            "beyondcxr.serving.api.SymileServingPredictor.load", lambda *args, **kwargs: predictor
        )
        application = create_app(
            authority_path=authority.directory,
            deployment_provenance=deployment,
            result_binding=binding,
            release_archive=attacker_archive,
        )
        pixels = ((np.indices((360, 480)).sum(axis=0) % 2) * 255).astype(np.uint8)
        jpeg = io.BytesIO()
        Image.fromarray(pixels).save(jpeg, format="JPEG", quality=90)
        files = [
            ("image", ("synthetic.jpg", jpeg.getvalue(), "image/jpeg")),
            ("view_position", (None, "AP")),
            ("labs", (None, json.dumps({key: float(index) for index, key in enumerate(LAB_KEYS)}))),
        ]
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application), base_url="http://testserver"
            ) as client:
                response = await client.post("/predict", files=files)
        assert response.status_code == 200
        probability = response.json()["probability"]
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0
        assert response.json()["serving_authority_id"] == authority.authority_id

    (global_result.directory / "claims.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="claims contract is invalid"):
        publish_serving_authority(
            authority_root=tmp_path / "rejected-authorities",
            capability=freeze,
            global_result=global_result,
            global_predictions=predictions,
            global_test_data=test_data,
            final_packages=all_packages,
        )
