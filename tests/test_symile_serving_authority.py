from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from symile_campaign_test_support import _freeze

import beyondcxr.release.serving as release_serving
import beyondcxr.serving.authority as serving_authority
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.serving.authority import (
    ENSEMBLE_POLICY,
    PRIMARY_FAMILY,
    ServingDistributionIntegrity,
    ValidatedServingAuthority,
    validate_serving_distribution,
)
from beyondcxr.serving.authority import (
    publish_serving_authority as _publish_serving_authority,
)
from beyondcxr.training import symile_campaign_control as campaign_control
from beyondcxr.training.symile_campaign_control import GlobalResultIntegrity
from beyondcxr.training.symile_final_packages import ValidatedFinalPackage


@pytest.fixture
def serving_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate result/package science in tests that own serving-manifest invariants."""

    def validate(directory: str | Path, **kwargs: object) -> GlobalResultIntegrity:
        del kwargs
        path = Path(directory)
        return GlobalResultIntegrity(
            path,
            json.loads((path / "manifest.json").read_bytes()),
            "c" * 64,
            "d" * 64,
        )

    monkeypatch.setattr(serving_authority, "validate_global_result_integrity", validate)
    monkeypatch.setattr(
        serving_authority,
        "validated_pretest_freeze_manifest",
        lambda capability: capability.manifest,
    )


def publish_serving_authority(**kwargs: object):
    return _publish_serving_authority(
        **kwargs,
        global_predictions=(),
        global_test_data=object(),
    )


def _primary_packages(tmp_path: Path, freeze) -> tuple[ValidatedFinalPackage, ...]:
    packages = []
    for seed, frozen in zip((17, 42, 2026), freeze.manifest["final_packages"][8:11], strict=True):
        directory = tmp_path / "packages" / frozen["package_id"]
        directory.mkdir(parents=True)
        packages.append(
            ValidatedFinalPackage(
                directory,
                {
                    "final_package_id": frozen["package_id"],
                    "package_kind": "neural",
                    "execution_scope": "full_development",
                    "seed_policy": seed,
                    "model_state_sha256": f"{seed:064x}",
                    "preprocessor_state_sha256": "a" * 64,
                    "input": {
                        "dataset": {
                            "dataset_id": "symile",
                            "bundle_id": freeze.manifest["bundle_id"],
                            "cohort": "official_train_plus_validation_strict_pneumonia",
                        },
                        "task": dict(freeze.manifest["task"]),
                        "family": {
                            "family_id": PRIMARY_FAMILY,
                            "modalities": ["cxr", "labs"],
                            "parameters": {"use_observedness": True, "modality_count": 2},
                        },
                        "preprocessing": {
                            "cxr_transform_policy": ("torchxrayvision-densenet121-res224-v1"),
                            "lab_policy": "symile-outer-training-right-ecdf-v1",
                        },
                        "training": {"parameters": {}, "loader": {}, "augmentation": {}},
                    },
                },
            )
        )
    return tuple(packages)


def _global_result(
    tmp_path: Path,
    freeze,
    monkeypatch: pytest.MonkeyPatch,
    *,
    digest: str = "9",
):
    result_id = "global-result-" + digest * 64
    directory = tmp_path / result_id
    directory.mkdir(parents=True)
    manifest = {
        "global_result_schema_version": 1,
        "global_result_id": result_id,
        "pretest_freeze_id": freeze.freeze_id,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    integrity = GlobalResultIntegrity(directory, manifest, "c" * 64, "d" * 64)
    monkeypatch.setattr(
        campaign_control,
        "_validate_global_result_artifact",
        lambda *_args, **_kwargs: (integrity, (), object(), (), {}),
    )
    monkeypatch.setattr(campaign_control, "_derive_global_claims", lambda *_args: {})
    return campaign_control.validate_global_result(
        directory,
        capability=freeze,
        predictions=(),
        final_packages=(),
        test_data=object(),
    )


def test_serving_authority_is_deterministic_immutable_and_path_neutral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serving_stubs: None
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    packages = _primary_packages(tmp_path, freeze)
    by_id = {package.package_id: package for package in packages}
    monkeypatch.setattr(
        serving_authority,
        "validate_final_package",
        lambda directory, **kwargs: replace(by_id[Path(directory).name], directory=Path(directory)),
    )
    result = _global_result(tmp_path, freeze, monkeypatch)
    first = publish_serving_authority(
        authority_root=tmp_path / "authorities-a",
        capability=freeze,
        global_result=result,
        final_packages=packages,
    )
    second = publish_serving_authority(
        authority_root=tmp_path / "authorities-a",
        capability=freeze,
        global_result=result,
        final_packages=packages,
    )
    elsewhere = publish_serving_authority(
        authority_root=tmp_path / "authorities-b",
        capability=freeze,
        global_result=result,
        final_packages=packages,
    )
    changed_result = _global_result(tmp_path, freeze, monkeypatch, digest="8")
    scientifically_distinct = publish_serving_authority(
        authority_root=tmp_path / "authorities-d",
        capability=freeze,
        global_result=changed_result,
        final_packages=packages,
    )
    assert first.authority_id == second.authority_id == elsewhere.authority_id
    assert isinstance(first, ServingDistributionIntegrity)
    assert not isinstance(first, ValidatedServingAuthority)
    with pytest.raises(TypeError, match="only by release validation"):
        ValidatedServingAuthority()
    assert scientifically_distinct.authority_id != first.authority_id
    assert first.manifest["ordered_seeds"] == [17, 42, 2026]
    assert first.manifest["ensemble_policy"] == ENSEMBLE_POLICY
    assert [item["package_id"] for item in first.manifest["members"]] == [
        package.package_id for package in packages
    ]
    assert {item.name for item in first.directory.iterdir()} == {"manifest.json", "packages"}
    assert tuple(package.package_id for package in first.packages) == tuple(
        package.package_id for package in packages
    )
    assert all(
        package.directory.parent == first.directory / "packages" for package in first.packages
    )
    assert (
        first.manifest["primary_thresholds"]
        == freeze.development_decision.manifest["primary_thresholds"]
    )
    assert "science_execution" not in first.manifest
    assert "serving_release" not in first.manifest


def test_serving_authority_rejects_invalid_release_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del tmp_path, monkeypatch
    with pytest.raises(ManifestBuildError, match="Serving release provenance"):
        release_serving._validated_release_provenance(
            {"git_commit": "current", "dependency_lock_sha256": "2" * 64}
        )


def test_operationally_distinct_deployments_share_scientific_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_directory = tmp_path / "packages" / ("final-package-" + "1" * 64)
    package_directory.mkdir(parents=True)
    authority = SimpleNamespace(
        authority_id="serving-authority-" + "a" * 64,
        directory=tmp_path / ("serving-authority-" + "a" * 64),
        packages=(SimpleNamespace(directory=package_directory),),
        manifest={
            "members": [{"package_id": "final-package-" + "1" * 64}],
            "primary_thresholds": {"youden_j": 0.4, "target_sensitivity": 0.3},
        },
    )
    execution = SimpleNamespace(
        execution_id="symile-execution-" + "b" * 64,
        manifest={"git_commit": "c" * 40, "dependency_lock_sha256": "d" * 64},
    )
    campaign = SimpleNamespace(
        freeze=SimpleNamespace(freeze_id="pretest-freeze-" + "e" * 64),
        global_result=SimpleNamespace(result_id="global-result-" + "f" * 64),
        predictions=(),
        test_data=object(),
        packages=(),
        execution=execution,
        development_decision=SimpleNamespace(decision_id="development-decision-" + "3" * 64),
    )
    archive = tmp_path / "release.zip"
    archive.write_bytes(b"canonical release")
    campaign.restoration_context = SimpleNamespace(
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()
    )

    def publish_staged_authority(**kwargs: object) -> SimpleNamespace:
        directory = Path(kwargs["authority_root"]) / authority.authority_id
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_bytes(b"authority")
        return SimpleNamespace(**{**authority.__dict__, "directory": directory})

    monkeypatch.setattr(
        release_serving,
        "publish_serving_authority",
        publish_staged_authority,
    )
    binding = SimpleNamespace(canonical_bytes=lambda: b"binding\n")
    monkeypatch.setattr(release_serving, "derive_result_binding", lambda _: binding)
    monkeypatch.setattr(
        release_serving,
        "validate_serving_authority",
        lambda path, *args, **kwargs: SimpleNamespace(
            **{**authority.__dict__, "directory": Path(path)}
        ),
    )
    monkeypatch.setattr(
        release_serving,
        "clean_release_provenance",
        lambda repository: {
            "git_commit": ("1" if Path(repository).name == "first" else "3") * 40,
            "dependency_lock_sha256": ("2" if Path(repository).name == "first" else "4") * 64,
        },
    )

    publications = []
    for name in ("first", "second"):
        candidate = release_serving._prepare_validated_campaign_serving_authority(
            authority_root=tmp_path / name / "staging",
            campaign=campaign,
            release_archive=archive,
            serving_release=release_serving.clean_release_provenance(tmp_path / name),
        )
        publications.append(
            release_serving.publish_accepted_serving_state(
                candidate_authority=candidate.authority.directory,
                candidate_deployment_provenance=candidate.deployment_provenance,
                candidate_result_binding=candidate.result_binding,
                release_archive=archive,
                authority_root=tmp_path / "accepted",
                restored_campaign=campaign,
            )
        )
    first, second = publications

    assert first.authority.authority_id == second.authority.authority_id
    assert first.deployment_provenance != second.deployment_provenance
    assert first.deployment_provenance.is_file()
    assert second.deployment_provenance.is_file()
    document = json.loads(first.deployment_provenance.read_bytes())
    for version in (True, 1.0):
        document["schema_version"] = version
        first.deployment_provenance.write_bytes(serving_authority.canonical_json_bytes(document))
        with pytest.raises(ManifestBuildError, match="provenance contract"):
            serving_authority._validate_deployment_provenance(
                first.deployment_provenance, first.authority.authority_id
            )


@pytest.mark.parametrize(
    "mutation",
    [
        "seed",
        "family",
        "positive_class",
        "threshold",
        "spatial_policy",
        "corruption",
        "float_version",
        "boolean_version",
    ],
)
def test_serving_authority_rejects_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    serving_stubs: None,
    mutation: str,
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    packages = _primary_packages(tmp_path, freeze)
    by_id = {package.package_id: package for package in packages}
    monkeypatch.setattr(
        serving_authority,
        "validate_final_package",
        lambda directory, **kwargs: replace(by_id[Path(directory).name], directory=Path(directory)),
    )
    authority = publish_serving_authority(
        authority_root=tmp_path / "authorities",
        capability=freeze,
        global_result=_global_result(tmp_path, freeze, monkeypatch),
        final_packages=packages,
    )
    path = authority.directory / "manifest.json"
    if mutation == "corruption":
        path.write_bytes(b"not-json")
    else:
        document = json.loads(path.read_bytes())
        if mutation in {"float_version", "boolean_version"}:
            document["serving_authority_schema_version"] = (
                1.0 if mutation == "float_version" else True
            )
        elif mutation == "seed":
            document["ordered_seeds"][0] = 18
        elif mutation == "family":
            document["family"] = "cxr_labs_ecg_gated"
        elif mutation == "positive_class":
            document["positive_class"]["value"] = 0
        elif mutation == "spatial_policy":
            document["preprocessing"]["spatial_policy"] = "wrong"
        else:
            document["primary_thresholds"]["youden_j"] = 2.0
        path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ManifestBuildError):
        validate_serving_distribution(authority.directory)


def test_serving_authority_requires_exact_primary_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serving_stubs: None
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    packages = list(_primary_packages(tmp_path, freeze))
    monkeypatch.setattr(
        serving_authority,
        "validate_final_package",
        lambda directory, **kwargs: next(
            package for package in packages if package.directory == Path(directory)
        ),
    )
    with pytest.raises(ManifestBuildError, match="exact three"):
        publish_serving_authority(
            authority_root=tmp_path / "authorities",
            capability=freeze,
            global_result=_global_result(tmp_path, freeze, monkeypatch),
            final_packages=packages[:2],
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_self_contained_distribution_requires_exact_package_copy_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    serving_stubs: None,
    mutation: str,
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    packages = _primary_packages(tmp_path, freeze)
    by_id = {package.package_id: package for package in packages}
    monkeypatch.setattr(
        serving_authority,
        "validate_final_package",
        lambda directory, **kwargs: replace(by_id[Path(directory).name], directory=Path(directory)),
    )
    authority = publish_serving_authority(
        authority_root=tmp_path / "authorities",
        capability=freeze,
        global_result=_global_result(tmp_path, freeze, monkeypatch),
        final_packages=packages,
    )
    attacked = tmp_path / f"attacked-{mutation}"
    shutil.copytree(authority.directory, attacked)
    package_root = attacked / "packages"
    if mutation == "missing":
        shutil.rmtree(package_root / packages[0].package_id)
    else:
        (package_root / ("final-package-" + "f" * 64)).mkdir()

    with pytest.raises(ManifestBuildError, match="package distribution membership"):
        validate_serving_distribution(attacked, enforce_directory_name=False)


@pytest.mark.parametrize(
    "mutation",
    ["wrong_seed", "duplicate_seed", "family", "task", "bundle", "transform"],
)
def test_serving_authority_rejects_incompatible_package_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    serving_stubs: None,
    mutation: str,
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    packages = list(_primary_packages(tmp_path, freeze))
    if mutation == "wrong_seed":
        packages[0].manifest["seed_policy"] = 18
    elif mutation == "duplicate_seed":
        packages[1].manifest["seed_policy"] = 17
    elif mutation == "family":
        packages[0].manifest["input"]["family"]["family_id"] = "cxr_labs_ecg_gated"
    elif mutation == "task":
        packages[1].manifest["input"]["task"]["task_id"] = "wrong_task"
    elif mutation == "bundle":
        packages[1].manifest["input"]["dataset"]["bundle_id"] = "bundle-" + "9" * 64
    else:
        packages[1].manifest["input"]["preprocessing"]["cxr_transform_policy"] = "wrong"
    by_id = {package.package_id: package for package in packages}
    monkeypatch.setattr(
        serving_authority,
        "validate_final_package",
        lambda directory, **kwargs: by_id[Path(directory).name],
    )
    with pytest.raises(ManifestBuildError):
        publish_serving_authority(
            authority_root=tmp_path / "authorities",
            capability=freeze,
            global_result=_global_result(tmp_path, freeze, monkeypatch),
            final_packages=packages,
        )


def test_serving_authority_rejects_package_and_global_scientific_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serving_stubs: None
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    packages = list(_primary_packages(tmp_path, freeze))
    packages[0] = ValidatedFinalPackage(
        packages[0].directory,
        {**packages[0].manifest, "final_package_id": "final-package-" + "8" * 64},
    )
    by_path = {package.directory: package for package in packages}
    by_id = {package.package_id: package for package in packages}
    monkeypatch.setattr(
        serving_authority,
        "validate_final_package",
        lambda directory, **kwargs: by_path.get(Path(directory), by_id.get(Path(directory).name)),
    )
    with pytest.raises(ManifestBuildError, match="pre-test freeze"):
        publish_serving_authority(
            authority_root=tmp_path / "authorities",
            capability=freeze,
            global_result=_global_result(tmp_path, freeze, monkeypatch),
            final_packages=packages,
        )

    packages = list(_primary_packages(tmp_path / "second", freeze))
    by_id = {package.package_id: package for package in packages}
    monkeypatch.setattr(
        serving_authority,
        "validate_final_package",
        lambda directory, **kwargs: by_id[Path(directory).name],
    )
    result = _global_result(tmp_path / "second", freeze, monkeypatch)
    altered = dict(result.manifest)
    altered["pretest_freeze_id"] = "pretest-freeze-" + "8" * 64
    (result.directory / "manifest.json").write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="changed after certification"):
        publish_serving_authority(
            authority_root=tmp_path / "second" / "authorities",
            capability=freeze,
            global_result=result,
            final_packages=packages,
        )
