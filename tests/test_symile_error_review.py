from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest
from symile_campaign_test_support import (
    _freeze,
    _projection_for_freeze,
    _synthetic_final_packages,
    _synthetic_test_predictions,
)

import radfusion.training.symile_campaign_control as campaign_control
import radfusion.training.symile_statistics as symile_statistics
from radfusion.data.errors import ManifestBuildError
from radfusion.training.symile_export import SymileExportMember, export_and_verify
from radfusion.training.symile_final_packages import (
    ValidatedFinalPackage,
)
from radfusion.training.symile_test_data import (
    FrozenSymileTestData,
)
from radfusion.utils.private_predictions import (
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
    with pytest.raises(ManifestBuildError, match="conflicts"):
        campaign_control.publish_error_review(private_root=tmp_path / "private", **review_arguments)
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
        members=[SymileExportMember(review, Path("private/error-review/symile") / review.name)],
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
        packages[0].manifest_sha256,
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

    altered_hash = ValidatedFinalPackage(packages[0].directory, packages[0].manifest, "8" * 64)
    alternate[0] = altered_hash
    package_by_path[packages[0].directory] = altered_hash
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
        predictions[0].manifest_sha256,
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
    test_data.frame = projection.frame()
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
    with pytest.raises(ManifestBuildError, match="claims do not rederive"):
        campaign_control.validate_global_result(result.directory, **review_arguments)
