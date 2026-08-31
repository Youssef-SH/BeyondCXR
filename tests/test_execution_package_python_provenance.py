from __future__ import annotations

import platform
from pathlib import Path
from types import SimpleNamespace

import pytest

import beyondcxr.training.rsna_campaign_control as rsna_control
import beyondcxr.training.symile_campaign_control as symile_control
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.symile_final_packages import ValidatedFinalPackage


def _execution(python: str = "3.13.7") -> dict[str, object]:
    return {
        "git_commit": "a" * 40,
        "dependency_lock_sha256": "b" * 64,
        "framework_versions": {
            "python": python,
            "torch": "2.8.0",
            "torchvision": "0.23.0",
            "torchxrayvision": "1.3.5",
        },
        "neural_runtime": {
            "device_type": "cuda",
            "autocast_dtype": "float16",
            "cuda_runtime_version": "12.4",
            "cudnn_version": 9100,
            "gpu_device_name": "GPU",
            "gpu_compute_capability": [8, 6],
        },
        "training_evaluation_runtime": {
            "device_type": "cuda",
            "autocast_dtype": "float16",
            "cuda_runtime_version": "12.4",
            "cudnn_version": 9100,
            "gpu_device_name": "GPU",
            "gpu_compute_capability": [8, 6],
        },
        "training_loader": {"lifecycle": "reused", "num_workers": 2, "pin_memory": True},
        "pretrained_weight": {
            "declared_name": "weight",
            "stable_identifier": "weight-v1",
            "byte_size": 1,
            "sha256": "c" * 64,
        },
    }


def _runtime() -> dict[str, object]:
    return {
        "resolved_device": "cuda",
        "mixed_precision_effective": True,
        "torch_version": "2.8.0",
        "torchvision_version": "0.23.0",
        "torchxrayvision_version": "1.3.5",
        "cuda_runtime_version": "12.4",
        "cudnn_version": 9100,
        "gpu_device_name": "GPU",
        "gpu_compute_capability": [8, 6],
    }


def _loader() -> dict[str, object]:
    return {"lifecycle": "reused", "num_workers": 2, "pin_memory": True}


def test_rsna_tabular_python_must_match_execution() -> None:
    package = {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "dependency_lock_sha256": "b" * 64,
        "python_version": "3.13.8",
    }
    assert not rsna_control._rsna_package_belongs_to_execution(package, _execution())


def test_rsna_neural_python_remains_enforced() -> None:
    package = {
        "source_provenance": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "dependency_lock_sha256": "b" * 64,
            "python_version": "3.13.8",
            "torch_version": "2.8.0",
            "torchvision_version": "0.23.0",
            "torchxrayvision_version": "1.3.5",
        },
        "runtime_provenance": {**_runtime(), "loader_execution": _loader()},
        "model_identity": {"pretrained_weight": _execution()["pretrained_weight"]},
    }
    assert not rsna_control._rsna_package_belongs_to_execution(package, _execution())


@pytest.mark.parametrize(
    ("field", "execution_field"),
    [
        ("torchvision_version", "torchvision"),
        ("torchxrayvision_version", "torchxrayvision"),
    ],
)
def test_rsna_neural_framework_versions_must_match_execution(
    field: str, execution_field: str
) -> None:
    source = {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "dependency_lock_sha256": "b" * 64,
        "python_version": "3.13.7",
        "torch_version": "2.8.0",
        "torchvision_version": "0.23.0",
        "torchxrayvision_version": "1.3.5",
    }
    package = {
        "source_provenance": source,
        "runtime_provenance": {**_runtime(), "loader_execution": _loader()},
        "model_identity": {"pretrained_weight": _execution()["pretrained_weight"]},
    }
    assert rsna_control._rsna_package_belongs_to_execution(package, _execution())
    changed = _execution()
    changed["framework_versions"] = {
        **changed["framework_versions"],
        execution_field: "different",
    }
    assert not rsna_control._rsna_package_belongs_to_execution(package, changed)


@pytest.mark.parametrize("neural", [False, True])
def test_symile_fold_python_must_match_execution(
    monkeypatch: pytest.MonkeyPatch, neural: bool
) -> None:
    manifest = {
        "lineage": {
            "git_commit": "a" * 40,
            "dependency_lock_sha256": "b" * 64,
            "pretrained_weight": None,
        },
        "operational": {
            "python_version": "3.13.8",
            "runtime_provenance": _runtime() if neural else None,
            "loader_execution": _loader() if neural else None,
        },
    }
    package = SimpleNamespace(manifest=manifest)
    manifest["fold_package_id"] = "fold"
    development = SimpleNamespace(
        manifest={"fold_packages": [{"fold_package_id": "fold"}]}, folds=(package,)
    )
    with pytest.raises(ManifestBuildError, match="provenance differs"):
        symile_control._validate_development_execution_provenance(
            development,
            execution=_execution(),
        )


@pytest.mark.parametrize("neural", [False, True])
def test_symile_final_python_must_match_execution(neural: bool) -> None:
    provenance = {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "dependency_lock_sha256": "b" * 64,
        "python_version": "3.13.8",
    }
    if neural:
        provenance.update(runtime_provenance=_runtime(), loader_execution=_loader())
    package = ValidatedFinalPackage(
        Path("package"),
        {
            "package_kind": "neural" if neural else "tabular",
            "execution_provenance": provenance,
            "pretrained_weight_fingerprint": None,
        },
    )
    assert not symile_control.final_package_belongs_to_execution(package, _execution())


@pytest.mark.parametrize("field", ["torchvision", "torchxrayvision"])
def test_symile_neural_framework_versions_must_match_historical_execution(field: str) -> None:
    package = ValidatedFinalPackage(
        Path("package"),
        {
            "package_kind": "neural",
            "execution_provenance": {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "dependency_lock_sha256": "b" * 64,
                "python_version": "3.13.7",
                "runtime_provenance": _runtime(),
                "loader_execution": _loader(),
            },
            "pretrained_weight_fingerprint": None,
        },
    )
    assert symile_control.final_package_belongs_to_execution(package, _execution())
    changed = _execution()
    changed["framework_versions"] = {**changed["framework_versions"], field: "different"}
    assert not symile_control.final_package_belongs_to_execution(package, changed)


def test_historical_python_comparison_does_not_use_current_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "python_version", lambda: "9.9.9")
    package = ValidatedFinalPackage(
        Path("package"),
        {
            "package_kind": "tabular",
            "execution_provenance": {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "dependency_lock_sha256": "b" * 64,
                "python_version": "3.13.7",
            },
            "pretrained_weight_fingerprint": None,
        },
    )
    assert symile_control.final_package_belongs_to_execution(package, _execution("3.13.7"))
