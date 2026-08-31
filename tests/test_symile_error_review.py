from __future__ import annotations

import copy
import json
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from symile_campaign_test_support import (
    _freeze,
    _projection_for_freeze,
    _synthetic_final_packages,
    _synthetic_test_predictions,
)

import beyondcxr.training.symile_campaign_control as campaign_control
import beyondcxr.training.symile_statistics as symile_statistics
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.serving.authority import publish_serving_authority
from beyondcxr.training.preservation import PreservationMember, export_and_verify
from beyondcxr.training.symile_final_packages import (
    ValidatedFinalPackage,
)
from beyondcxr.training.symile_test_data import (
    FrozenSymileTestData,
    require_held_out_evaluation_projection,
)
from beyondcxr.utils.private_predictions import (
    ValidatedPredictionEvidence,
)


@pytest.fixture
def result_evidence(tmp_path, monkeypatch):
    freeze = _freeze(tmp_path / "freeze")
    packages = _synthetic_final_packages(tmp_path, freeze)
    predictions = _synthetic_test_predictions(tmp_path, freeze, packages)
    package_by_path = {package.directory: package for package in packages}
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
    monkeypatch.setattr(campaign_control, "_require_capability", lambda value: value)
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
        report_root=tmp_path / "reports",
        capability=freeze,
        predictions=predictions,
        final_packages=packages,
        test_data=test_data,
    )
    review_arguments = {
        "capability": freeze,
        "predictions": predictions,
        "final_packages": packages,
        "test_data": test_data,
    }
    return review_arguments, result, package_by_path, evidence_by_path


def test_global_result_schema_and_idempotent_publication(tmp_path, result_evidence):
    review_arguments, result, _, _ = result_evidence
    freeze, predictions, packages, test_data = (
        review_arguments[key]
        for key in ("capability", "predictions", "final_packages", "test_data")
    )
    for value in (True, 1.0):
        altered = tmp_path / f"altered-global-{value!r}"
        shutil.copytree(result.directory, altered)
        manifest_path = altered / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["global_result_schema_version"] = value
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ManifestBuildError, match="contract"):
            campaign_control.validate_global_result(
                altered,
                capability=freeze,
                predictions=predictions,
                final_packages=packages,
                test_data=test_data,
                enforce_directory_name=False,
            )
    resumed_result = campaign_control.publish_global_result(
        report_root=tmp_path / "reports",
        capability=freeze,
        predictions=predictions,
        final_packages=packages,
        test_data=test_data,
    )
    assert resumed_result.result_id == result.result_id


@pytest.mark.parametrize("member", ["manifest.json", "claims.json", "reliability.svg"])
def test_global_result_rejects_symlink_members(tmp_path, result_evidence, member):
    review_arguments, result, _, _ = result_evidence
    source = result.directory / member
    external = tmp_path / member
    source.rename(external)
    source.symlink_to(external)

    with pytest.raises(ManifestBuildError, match="directory is invalid"):
        campaign_control.validate_global_result_integrity(result.directory, **review_arguments)


def test_global_result_integrity_does_not_repeat_scientific_bootstrap(
    result_evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_arguments, result, _, _ = result_evidence
    monkeypatch.setattr(
        campaign_control,
        "_derive_global_claims",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("scientific recomputation")),
    )

    validated = campaign_control.validate_global_result_integrity(
        result.directory, **review_arguments
    )

    assert validated.result_id == result.result_id
    assert validated.claims_sha256 == result.claims_sha256
    assert validated.reliability_sha256 == result.reliability_sha256
    assert isinstance(validated, campaign_control.GlobalResultIntegrity)
    assert not isinstance(validated, campaign_control.CertifiedGlobalResult)
    with pytest.raises(AssertionError, match="scientific recomputation"):
        campaign_control.validate_global_result(result.directory, **review_arguments)


def test_integrity_only_global_result_cannot_publish_serving_authority(result_evidence) -> None:
    review_arguments, result, _, _ = result_evidence
    claims_path = result.directory / "claims.json"
    claims = json.loads(claims_path.read_bytes())
    family_metrics = next(iter(claims["predictor_views"].values()))
    metric = next(iter(family_metrics))
    family_metrics[metric] = 0.0 if family_metrics[metric] != 0.0 else 1.0
    claims_path.write_text(
        json.dumps(claims, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )

    integrity = campaign_control.validate_global_result_integrity(
        result.directory, **review_arguments
    )

    with pytest.raises(PermissionError, match="full scientific certification"):
        publish_serving_authority(
            authority_root=result.directory.parent / "serving",
            capability=review_arguments["capability"],
            global_result=integrity,
            global_predictions=review_arguments["predictions"],
            global_test_data=review_arguments["test_data"],
            final_packages=review_arguments["final_packages"],
        )


def test_certification_cannot_transfer_to_altered_integrity_result(
    tmp_path: Path, result_evidence
) -> None:
    review_arguments, certified, _, _ = result_evidence
    altered_directory = tmp_path / "foreign" / certified.directory.name
    shutil.copytree(certified.directory, altered_directory)
    claims_path = altered_directory / "claims.json"
    claims = json.loads(claims_path.read_bytes())
    family_metrics = next(iter(claims["predictor_views"].values()))
    metric = next(iter(family_metrics))
    family_metrics[metric] = 0.0 if family_metrics[metric] != 0.0 else 1.0
    claims_path.write_text(
        json.dumps(claims, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    integrity = campaign_control.validate_global_result_integrity(
        altered_directory, **review_arguments
    )

    with pytest.raises(TypeError):
        replace(
            certified,
            directory=integrity.directory,
            manifest=integrity.manifest,
            claims_sha256=integrity.claims_sha256,
            reliability_sha256=integrity.reliability_sha256,
        )
    with pytest.raises(TypeError):
        certified.manifest["global_result_id"] = integrity.result_id
    assert copy.copy(certified) is certified
    assert copy.deepcopy(certified) is certified

    with pytest.raises(PermissionError, match="full scientific certification"):
        publish_serving_authority(
            authority_root=tmp_path / "serving",
            capability=review_arguments["capability"],
            global_result=integrity,
            global_predictions=review_arguments["predictions"],
            global_test_data=review_arguments["test_data"],
            final_packages=review_arguments["final_packages"],
        )


def test_serving_rejects_global_result_changed_after_certification(result_evidence) -> None:
    review_arguments, certified, _, _ = result_evidence
    claims_path = certified.directory / "claims.json"
    claims = json.loads(claims_path.read_bytes())
    family_metrics = next(iter(claims["predictor_views"].values()))
    metric = next(iter(family_metrics))
    family_metrics[metric] = 0.0 if family_metrics[metric] != 0.0 else 1.0
    claims_path.write_text(
        json.dumps(claims, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    campaign_control.validate_global_result_integrity(certified.directory, **review_arguments)

    with pytest.raises(ManifestBuildError, match="changed after certification"):
        publish_serving_authority(
            authority_root=certified.directory.parent / "serving",
            capability=review_arguments["capability"],
            global_result=certified,
            global_predictions=review_arguments["predictions"],
            global_test_data=review_arguments["test_data"],
            final_packages=review_arguments["final_packages"],
        )


def test_escaped_frame_and_replaced_projection_cannot_change_global_result(result_evidence):
    review_arguments, result, _, _ = result_evidence
    freeze = review_arguments["capability"]
    test_data = review_arguments["test_data"]
    canonical = test_data.evaluation_projection().frame()
    escaped = test_data.frame
    escaped["subject_id"] = np.arange(100_000, 100_000 + len(escaped))
    escaped["sample_id"] = [f"foreign:{index}" for index in range(len(escaped))]
    escaped["target"] = 1 - escaped["target"]

    campaign_control.validate_global_result(result.directory, **review_arguments)
    assert test_data.evaluation_projection().frame().equals(canonical)

    altered = canonical.copy()
    altered["subject_id"] = np.arange(200_000, 200_000 + len(altered))
    transferred = replace(test_data.evaluation_projection(), _frame=altered)
    with pytest.raises(PermissionError, match="differs from canonical test data"):
        require_held_out_evaluation_projection(transferred, freeze=freeze)


def test_private_review_generation_and_regeneration(tmp_path, result_evidence):
    review_arguments, result, _, _ = result_evidence
    predictions = review_arguments["predictions"]
    packages = review_arguments["final_packages"]
    review = campaign_control.publish_error_review(
        private_root=tmp_path / "private", **review_arguments
    )
    original_review = review.read_bytes()
    assert (
        campaign_control.publish_error_review(private_root=tmp_path / "private", **review_arguments)
        == review
    )
    campaign_control.validate_error_review(review, **review_arguments)
    assert json.loads(original_review)["prediction_ids"] == [
        item.prediction_id
        for package, item in zip(packages, predictions, strict=True)
        if package.manifest["input"]["family"]["family_id"] == "cxr_labs_gated"
    ]
    review_document = json.loads(original_review)
    assert "error_review_id" not in review_document
    assert "global_result_id" not in review_document
    assert "bundle" not in review_document
    assert "task" not in review_document
    assert review_document["ranking_policy"] == symile_statistics.ERROR_REVIEW_POLICY
    for value in (True, 1.0):
        altered_review = dict(review_document)
        altered_review["error_review_schema_version"] = value
        review.write_text(json.dumps(altered_review), encoding="utf-8")
        with pytest.raises(ManifestBuildError, match="does not rederive"):
            campaign_control.validate_error_review(review, **review_arguments)
    review.write_bytes(original_review)
    review.write_bytes(b"corrupt")
    with pytest.raises(ManifestBuildError, match="does not rederive"):
        campaign_control.validate_error_review(review, **review_arguments)
    recovered = campaign_control.publish_error_review(
        private_root=tmp_path / "private", **review_arguments
    )
    assert recovered == review
    assert recovered.read_bytes() == original_review
    campaign_control.validate_global_result(
        result.directory,
        **review_arguments,
    )
    review.unlink()
    assert (
        campaign_control.publish_error_review(
            private_root=tmp_path / "private", **review_arguments
        ).read_bytes()
        == original_review
    )


def test_private_review_is_preserved_and_validated_after_restore(tmp_path, result_evidence):
    review_arguments, _, _, _ = result_evidence
    review = campaign_control.publish_error_review(
        private_root=tmp_path / "private", **review_arguments
    )
    original_review = review.read_bytes()
    archive = export_and_verify(
        members=[PreservationMember(review, Path("private/error-review/symile") / review.name)],
        export_root=tmp_path / "export",
        backup_root=tmp_path / "backup",
        export_name="private-review",
        restoration_validator=lambda root: campaign_control.validate_error_review(
            root / "private/error-review/symile" / review.name, **review_arguments
        ),
    )
    with zipfile.ZipFile(archive) as exported:
        assert exported.read(f"member-0000/{review.name}") == original_review


def test_global_result_is_closed_to_frozen_packages_and_subject_grouping(result_evidence):
    review_arguments, result, package_by_path, evidence_by_path = result_evidence
    freeze, predictions, packages, test_data = (
        review_arguments[key]
        for key in ("capability", "predictions", "final_packages", "test_data")
    )
    alternate = list(packages)
    alternate[0] = ValidatedFinalPackage(
        packages[0].directory,
        {**packages[0].manifest, "final_package_id": "final-package-" + "9" * 64},
    )
    package_by_path[packages[0].directory] = alternate[0]
    with pytest.raises(ManifestBuildError, match="pre-test freeze"):
        campaign_control.validate_global_result(
            result.directory,
            capability=freeze,
            predictions=predictions,
            final_packages=alternate,
            test_data=test_data,
        )

    package_by_path[packages[0].directory] = packages[0]
    prediction_manifest = {**predictions[0].manifest, "model_package_id": packages[1].package_id}
    changed_prediction = ValidatedPredictionEvidence(
        predictions[0].directory,
        prediction_manifest,
        predictions[0].predictions,
    )
    changed_predictions = (changed_prediction, *predictions[1:])
    evidence_by_path[predictions[0].directory] = changed_prediction
    with pytest.raises(ManifestBuildError, match="prediction lineage"):
        campaign_control.validate_global_result(
            result.directory,
            capability=freeze,
            predictions=changed_predictions,
            final_packages=packages,
            test_data=test_data,
        )

    evidence_by_path[predictions[0].directory] = predictions[0]
    projection = _projection_for_freeze(freeze, grouped=True)
    test_data._bundle = SimpleNamespace(bundle_id=projection.bundle_id)
    test_data._canonical_frame = projection.frame()
    with pytest.raises(ManifestBuildError, match="claims do not rederive"):
        campaign_control.validate_global_result(
            result.directory,
            capability=freeze,
            predictions=predictions,
            final_packages=packages,
            test_data=test_data,
        )


def test_private_review_does_not_depend_on_global_claims(tmp_path, result_evidence):
    review_arguments, result, _, _ = result_evidence
    review = campaign_control.publish_error_review(
        private_root=tmp_path / "private", **review_arguments
    )
    (result.directory / "claims.json").write_text("{}\n", encoding="utf-8")
    campaign_control.validate_error_review(review, **review_arguments)
    with pytest.raises(ManifestBuildError, match="claims contract is invalid"):
        campaign_control.validate_global_result(result.directory, **review_arguments)
