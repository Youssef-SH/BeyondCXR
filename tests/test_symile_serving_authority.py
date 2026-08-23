from __future__ import annotations

import json
from pathlib import Path

import pytest
from symile_campaign_test_support import _freeze

import beyondcxr.serving.authority as serving_authority
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.serving.authority import (
    ENSEMBLE_POLICY,
    PRIMARY_FAMILY,
    validate_serving_authority,
)
from beyondcxr.serving.authority import (
    publish_serving_authority as _publish_serving_authority,
)
from beyondcxr.training.symile_campaign_control import ValidatedGlobalResult
from beyondcxr.training.symile_final_packages import ValidatedFinalPackage

_SERVING_RELEASE = {
    "git_commit": "1" * 40,
    "dependency_lock_sha256": "2" * 64,
}


@pytest.fixture(autouse=True)
def _canonical_global_result_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate(directory: str | Path, **kwargs: object) -> ValidatedGlobalResult:
        del kwargs
        path = Path(directory)
        return ValidatedGlobalResult(path, json.loads((path / "manifest.json").read_bytes()))

    monkeypatch.setattr(serving_authority, "validate_global_result", validate)


def publish_serving_authority(**kwargs: object):
    release = kwargs.pop("serving_release", _SERVING_RELEASE)
    return _publish_serving_authority(
        **kwargs,
        global_predictions=(),
        global_test_data=object(),
        serving_release=release,
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
                            "bundle_id": freeze.manifest["bundle"]["bundle_id"],
                            "split_assignment_id": freeze.manifest["bundle"]["split_assignment_id"],
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
                frozen["manifest_sha256"],
            )
        )
    return tuple(packages)


def _global_result(tmp_path: Path, freeze) -> ValidatedGlobalResult:
    result_id = "global-result-" + "9" * 64
    directory = tmp_path / result_id
    directory.mkdir()
    manifest = {
        "global_result_schema_version": 1,
        "global_result_id": result_id,
        "pretest_freeze_id": freeze.freeze_id,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ValidatedGlobalResult(directory, manifest)


def test_serving_authority_is_deterministic_immutable_and_path_neutral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    packages = _primary_packages(tmp_path, freeze)
    by_path = {package.directory: package for package in packages}
    monkeypatch.setattr(
        serving_authority,
        "validate_final_package",
        lambda directory, **kwargs: by_path[Path(directory)],
    )
    result = _global_result(tmp_path, freeze)
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
    changed_release = publish_serving_authority(
        authority_root=tmp_path / "authorities-c",
        capability=freeze,
        global_result=result,
        final_packages=packages,
        serving_release={
            **_SERVING_RELEASE,
            "git_commit": "3" * 40,
        },
    )
    assert first.authority_id == second.authority_id == elsewhere.authority_id
    assert changed_release.authority_id != first.authority_id
    assert first.manifest["ordered_seeds"] == [17, 42, 2026]
    assert first.manifest["ensemble_policy"] == ENSEMBLE_POLICY
    assert [item["package_id"] for item in first.manifest["members"]] == [
        package.package_id for package in packages
    ]
    assert first.manifest["science_execution"] != first.manifest["serving_release"]


def test_serving_authority_rejects_invalid_release_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    packages = _primary_packages(tmp_path, freeze)
    by_path = {package.directory: package for package in packages}
    monkeypatch.setattr(
        serving_authority,
        "validate_final_package",
        lambda directory, **kwargs: by_path[Path(directory)],
    )
    with pytest.raises(ManifestBuildError, match="Serving release provenance"):
        publish_serving_authority(
            authority_root=tmp_path / "authorities",
            capability=freeze,
            global_result=_global_result(tmp_path, freeze),
            final_packages=packages,
            serving_release={"git_commit": "current", "dependency_lock_sha256": "2" * 64},
        )


@pytest.mark.parametrize(
    "mutation",
    ["seed", "family", "positive_class", "threshold", "spatial_policy", "corruption"],
)
def test_serving_authority_rejects_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    packages = _primary_packages(tmp_path, freeze)
    by_path = {package.directory: package for package in packages}
    monkeypatch.setattr(
        serving_authority,
        "validate_final_package",
        lambda directory, **kwargs: by_path[Path(directory)],
    )
    authority = publish_serving_authority(
        authority_root=tmp_path / "authorities",
        capability=freeze,
        global_result=_global_result(tmp_path, freeze),
        final_packages=packages,
    )
    path = authority.directory / "manifest.json"
    if mutation == "corruption":
        path.write_bytes(b"not-json")
    else:
        document = json.loads(path.read_bytes())
        if mutation == "seed":
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
        validate_serving_authority(authority.directory, package_root=packages[0].directory.parent)


def test_serving_authority_requires_exact_primary_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
            global_result=_global_result(tmp_path, freeze),
            final_packages=packages[:2],
        )


@pytest.mark.parametrize(
    "mutation",
    ["wrong_seed", "duplicate_seed", "family", "task", "bundle", "split", "transform"],
)
def test_serving_authority_rejects_incompatible_package_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    elif mutation == "split":
        packages[1].manifest["input"]["dataset"]["split_assignment_id"] = (
            "split-assignment-" + "9" * 64
        )
    else:
        packages[1].manifest["input"]["preprocessing"]["cxr_transform_policy"] = "wrong"
    by_path = {package.directory: package for package in packages}
    monkeypatch.setattr(
        serving_authority,
        "validate_final_package",
        lambda directory, **kwargs: by_path[Path(directory)],
    )
    with pytest.raises(ManifestBuildError):
        publish_serving_authority(
            authority_root=tmp_path / "authorities",
            capability=freeze,
            global_result=_global_result(tmp_path, freeze),
            final_packages=packages,
        )


def test_serving_authority_rejects_package_byte_witness_and_global_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    packages = list(_primary_packages(tmp_path, freeze))
    packages[0] = ValidatedFinalPackage(
        packages[0].directory,
        packages[0].manifest,
        "9" * 64,
    )
    by_path = {package.directory: package for package in packages}
    monkeypatch.setattr(
        serving_authority,
        "validate_final_package",
        lambda directory, **kwargs: by_path[Path(directory)],
    )
    with pytest.raises(ManifestBuildError, match="pre-test freeze"):
        publish_serving_authority(
            authority_root=tmp_path / "authorities",
            capability=freeze,
            global_result=_global_result(tmp_path, freeze),
            final_packages=packages,
        )

    packages = list(_primary_packages(tmp_path / "second", freeze))
    by_path = {package.directory: package for package in packages}
    result = _global_result(tmp_path / "second", freeze)
    result.manifest["pretest_freeze_id"] = "pretest-freeze-" + "8" * 64
    (result.directory / "manifest.json").write_text(json.dumps(result.manifest), encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="global-result provenance"):
        publish_serving_authority(
            authority_root=tmp_path / "second" / "authorities",
            capability=freeze,
            global_result=result,
            final_packages=packages,
        )
