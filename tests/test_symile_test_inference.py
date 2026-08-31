from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from neural_test_support import cpu_runtime
from symile_campaign_test_support import (
    _freeze,
    _projection_for_freeze,
    _synthetic_final_packages,
    _synthetic_held_out_capability,
    _synthetic_test_predictions,
)

import beyondcxr.training.symile_campaign_control as campaign_control
import beyondcxr.training.symile_final_packages as final_packages
import beyondcxr.training.symile_test_inference as test_inference
import beyondcxr.utils.publication as publication
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.preservation import PreservationMember, export_and_verify
from beyondcxr.training.symile_campaign_control import ValidatedTestOpenEvidence
from beyondcxr.training.symile_final_packages import (
    ValidatedFinalPackage,
)
from beyondcxr.training.symile_test_data import (
    RestoredSymileTestData,
    validate_prediction_against_test_projection,
)
from beyondcxr.utils.private_predictions import (
    SYMILE_TEST_INFERENCE_POLICY,
    ValidatedPredictionEvidence,
    build_prediction_table,
)


def test_interrupted_prediction_staging_is_not_evidence_or_exported_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path / "control")
    monkeypatch.setattr(campaign_control, "_require_capability", lambda value: freeze)
    capability = _synthetic_held_out_capability(tmp_path / "capability", freeze)
    for consumer in (campaign_control, test_inference):
        monkeypatch.setattr(
            consumer,
            "require_validated_held_out_capability",
            lambda value: capability,
        )
    private = tmp_path / "private"
    prediction_root = private / "predictions" / "symile" / "test"
    stage = publication.staging_directory(prediction_root / ("prediction-" + "a" * 64))
    partial = stage / "predictions.parquet"
    partial.write_bytes(b"interrupted unpublished bytes")
    arguments = {
        "capability": capability,
        "expected_packages": set(),
        "projection": _projection_for_freeze(freeze, grouped=False),
    }
    assert test_inference.validate_existing_test_predictions(private, **arguments) == {}
    source = tmp_path / "export-source"
    source.mkdir()
    (source / "complete.json").write_text("{}\n")
    abandoned = publication.staging_directory(source / "result")
    (abandoned / "partial.json").write_text("unfinished")
    export_arguments = {
        "members": [PreservationMember(source, Path("private/synthetic"))],
        "export_root": tmp_path / "export",
        "backup_root": tmp_path / "backup",
        "export_name": "synthetic",
    }
    first = export_and_verify(**export_arguments)
    original = first.read_bytes()
    (abandoned / "partial.json").write_text("still unfinished")
    assert export_and_verify(**export_arguments).read_bytes() == original
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["export-manifest.json", "member-0000/complete.json"]
    assert partial.read_bytes() == b"interrupted unpublished bytes"
    assert abandoned.is_dir()
    # A malformed published destination must still fail, not be silently skipped.
    (prediction_root / ("prediction-" + "b" * 64)).mkdir()
    with pytest.raises(ValueError, match="invalid file set"):
        test_inference.validate_existing_test_predictions(private, **arguments)


def test_prediction_evidence_validates_against_ordinary_test_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    monkeypatch.setattr(campaign_control, "_require_capability", lambda value: freeze)
    sample_ids = [f"sample-{index:03d}" for index in range(110)]
    targets = np.asarray([0] * 55 + [1] * 55, dtype=np.int8)
    package_id = str(freeze.manifest["final_packages"][0]["package_id"])
    projection = _projection_for_freeze(freeze, grouped=False)
    evidence = ValidatedPredictionEvidence(
        tmp_path / "prediction",
        {
            "prediction_id": "prediction-" + "1" * 64,
            "dataset_id": "symile",
            "bundle_id": projection.bundle_id,
            "task_id": projection.task_id,
            "label_policy_version": projection.label_policy_version,
            "scope": "test",
            "inference_policy": dict(SYMILE_TEST_INFERENCE_POLICY),
            "authorized_by_freeze_id": freeze.freeze_id,
            "model_package_id": package_id,
        },
        build_prediction_table(sample_ids, targets, np.linspace(-2.0, 2.0, 110)),
    )

    assert (
        validate_prediction_against_test_projection(
            evidence,
            capability=freeze,
            projection=projection,
            expected_package_ids=[package_id],
        )
        is evidence
    )


def test_manually_constructed_preserved_opening_cannot_authorize_restoration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    capability = _synthetic_held_out_capability(tmp_path / "capability", freeze)
    monkeypatch.setattr(campaign_control, "_require_capability", lambda value: freeze)
    for consumer in (campaign_control, test_inference):
        monkeypatch.setattr(
            consumer,
            "require_validated_held_out_capability",
            lambda value: capability,
        )
    packages = _synthetic_final_packages(tmp_path / "packages", freeze)
    projection = _projection_for_freeze(freeze, grouped=False)
    frame = projection.frame()
    frame["sample_id"] = [f"symile:{index:03d}" for index in range(len(frame))]
    projection._test_data._canonical_frame = frame.copy()
    projection = projection._test_data.evaluation_projection()
    data = projection._test_data
    data._canonical_frame = frame.copy()
    evidence = test_inference._publish(
        packages[0].package_id,
        data,
        np.linspace(-1.0, 1.0, len(frame)),
        tmp_path / "private",
        capability,
    )
    opening = ValidatedTestOpenEvidence(
        capability.opening.path,
        tmp_path.resolve(),
        freeze.freeze_id,
        capability.execution.execution_id,
        str(freeze.manifest["bundle_id"]),
        capability.opening.document_bytes,
        campaign_control._PRESERVED_TEST_OPEN_GUARD,
    )
    with pytest.raises(PermissionError, match="authenticated restoration context"):
        RestoredSymileTestData(
            opening,
            freeze,
            SimpleNamespace(bundle_id=str(freeze.manifest["bundle_id"])),
            manifest_root=tmp_path,
        )
    with pytest.raises(PermissionError, match="authenticated restoration context"):
        test_inference.validate_restored_test_prediction_publication(
            evidence,
            freeze=freeze,
            execution=capability.execution,
            opening=opening,
        )


def test_later_opening_does_not_legitimize_prediction_without_publication_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    capability = _synthetic_held_out_capability(tmp_path / "capability", freeze)
    monkeypatch.setattr(campaign_control, "_require_capability", lambda value: freeze)
    for consumer in (campaign_control, test_inference):
        monkeypatch.setattr(
            consumer,
            "require_validated_held_out_capability",
            lambda value: capability,
        )
    package = _synthetic_final_packages(tmp_path / "packages", freeze)[0]
    projection = _projection_for_freeze(freeze, grouped=False)
    frame = projection.frame()
    frame["sample_id"] = [f"symile:{index:03d}" for index in range(len(frame))]
    projection._test_data._canonical_frame = frame.copy()
    projection = projection._test_data.evaluation_projection()
    data = SimpleNamespace(
        frame=frame,
        bundle=SimpleNamespace(bundle_id=projection.bundle_id),
        evaluation_projection=lambda: projection,
    )
    evidence = test_inference._publish(
        package.package_id,
        data,
        np.linspace(-1.0, 1.0, len(frame)),
        tmp_path / "private",
        capability,
    )
    manifest_path = evidence.directory / "manifest.json"
    document = json.loads(manifest_path.read_bytes())
    assert document["held_out_publication"] == {
        "authority_kind": "symile-opened-campaign-v1",
        "execution_id": capability.execution.execution_id,
    }
    document.pop("held_out_publication")
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected field set"):
        test_inference.validate_existing_test_predictions(
            tmp_path / "private",
            capability=capability,
            expected_packages={package.package_id},
            projection=projection,
        )


def test_fourteen_package_predictions_derive_exactly_six_views(tmp_path: Path) -> None:
    packages = []
    predictions = []
    index = 0
    for family, members in (
        ("labs_logistic", 1),
        ("labs_lightgbm", 1),
        ("cxr_densenet", 3),
        ("cxr_labs_concat", 3),
        ("cxr_labs_gated", 3),
        ("cxr_labs_ecg_gated", 3),
    ):
        for member in range(members):
            seed = 42 if members == 1 else (17, 42, 2026)[member]
            package_id = "final-package-" + f"{index:064x}"
            packages.append(
                ValidatedFinalPackage(
                    tmp_path / package_id,
                    {
                        "final_package_id": package_id,
                        "seed_policy": seed,
                        "input": {"family": {"family_id": family}},
                    },
                )
            )
            predictions.append(
                ValidatedPredictionEvidence(
                    tmp_path / f"prediction-{index}",
                    {"prediction_id": "prediction-" + f"{index:064x}"},
                    build_prediction_table(["a", "b"], [0, 1], [-1.0 + member, 1.0 + member]),
                )
            )
            index += 1
    views = campaign_control.predictor_views(packages, predictions)
    assert set(views) == {
        "labs_logistic",
        "labs_lightgbm",
        "cxr_densenet",
        "cxr_labs_concat",
        "cxr_labs_gated",
        "cxr_labs_ecg_gated",
    }

    for index in range(len(predictions) - 3, len(predictions)):
        item = predictions[index]
        predictions[index] = ValidatedPredictionEvidence(
            item.directory,
            item.manifest,
            build_prediction_table(["a", "b"], [1, 0], [-1.0, 1.0]),
        )
    with pytest.raises(ManifestBuildError, match="not exactly aligned"):
        campaign_control.predictor_views(packages, predictions)


def test_partial_prediction_resume_infers_only_missing_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    monkeypatch.setattr(campaign_control, "_require_capability", lambda value: freeze)
    packages = _synthetic_final_packages(tmp_path / "packages", freeze)
    predictions = _synthetic_test_predictions(tmp_path / "predictions", freeze, packages)
    existing = {
        package.package_id: prediction
        for package, prediction in zip(packages[:5], predictions[:5], strict=True)
    }
    family_by_package = {
        package.package_id: package.manifest["input"]["family"]["family_id"] for package in packages
    }
    monkeypatch.setattr(
        test_inference,
        "FINAL_TABULAR_FAMILIES",
        tuple(set(family_by_package.values())),
    )
    monkeypatch.setattr(
        test_inference,
        "load_final_package_config",
        lambda package: SimpleNamespace(
            family_id=family_by_package[package.package_id],
            modalities=(),
            mixed_precision=True,
        ),
    )
    monkeypatch.setattr(test_inference, "load_final_tabular_model", lambda package: package)
    monkeypatch.setattr(test_inference, "test_laboratory_frame", lambda data: data.frame)
    inferred: list[str] = []
    monkeypatch.setattr(
        test_inference,
        "symile_tabular_logits",
        lambda model, frame: np.linspace(-1.0, 1.0, len(frame)),
    )

    def publish(package_id, data, logits, private_root, publication_capability):
        del data, logits, private_root
        assert publication_capability.freeze.freeze_id == freeze.freeze_id
        inferred.append(package_id)
        return predictions[[package.package_id for package in packages].index(package_id)]

    monkeypatch.setattr(test_inference, "_publish", publish)
    data = SimpleNamespace(
        frame=_projection_for_freeze(freeze, grouped=False).frame(),
        bundle=SimpleNamespace(bundle_id=freeze.manifest["bundle_id"]),
    )
    capability = _synthetic_held_out_capability(tmp_path / "capability", freeze)
    with monkeypatch.context() as runtime_context:
        runtime_context.setattr(
            test_inference, "FINAL_TABULAR_FAMILIES", ("labs_logistic", "labs_lightgbm")
        )
        runtime = cpu_runtime()
        test_inference._validate_neural_inference_runtime(capability, runtime)
    resumed = test_inference._infer_all(
        capability,
        packages,
        data,
        tmp_path / "source",
        tmp_path / "private",
        runtime,
        existing,
    )
    assert resumed == predictions
    assert inferred == [package.package_id for package in packages[5:]]


@pytest.mark.parametrize(
    "difference",
    [
        {"device": torch.device("cpu")},
        {"mixed_precision_effective": False},
        {"cuda_runtime_version": "12.5"},
        {"cudnn_version": 9200},
        {"gpu_device_name": "GPU-B"},
        {"gpu_compute_capability": (9, 0)},
    ],
)
def test_cross_runtime_resume_is_rejected_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, difference: dict
) -> None:
    frozen_runtime = {
        "device_type": "cuda",
        "autocast_dtype": "float16",
        "deterministic_algorithms": "enabled_warn_only",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_runtime_version": "12.4",
        "cudnn_version": 9100,
        "gpu_device_name": "GPU-A",
        "gpu_compute_capability": [8, 6],
    }
    freeze = _freeze(tmp_path / "freeze")
    monkeypatch.setattr(campaign_control, "_require_capability", lambda value: freeze)
    capability = _synthetic_held_out_capability(
        tmp_path / "capability", freeze, neural_runtime=frozen_runtime
    )
    packages = _synthetic_final_packages(tmp_path / "packages", freeze)
    family_by_package = {
        package.package_id: package.manifest["input"]["family"]["family_id"] for package in packages
    }
    monkeypatch.setattr(
        test_inference,
        "load_final_package_config",
        lambda package: SimpleNamespace(
            family_id=family_by_package[package.package_id],
            mixed_precision=True,
        ),
    )
    package_by_path = {package.directory: package for package in packages}
    monkeypatch.setattr(
        test_inference, "validate_final_package", lambda path: package_by_path[Path(path)]
    )
    monkeypatch.setattr(
        test_inference,
        "materialize_official_test",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("test access began")),
    )

    runtime = replace(
        cpu_runtime(),
        device=torch.device("cuda"),
        resolved_device="cuda",
        mixed_precision_effective=True,
        cuda_runtime_version="12.4",
        cudnn_version=9100,
        gpu_device_name="GPU-A",
        gpu_compute_capability=(8, 6),
    )
    runtime = replace(runtime, **difference)
    with pytest.raises(ManifestBuildError, match="runtime differs"):
        test_inference.publish_frozen_test_predictions(
            capability=capability,
            final_packages=packages,
            source_root=tmp_path / "source",
            manifest_root=tmp_path / "manifests",
            private_root=tmp_path / "private",
            runtime=runtime,
        )


def test_neural_partial_resume_uses_validated_runtime_and_preserves_existing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from beyondcxr.training.config import load_symile_development_config
    from beyondcxr.training.symile_final_packages import (
        _final_input_projection,
        _final_package_config_from_input,
    )

    freeze = _freeze(tmp_path / "control")
    monkeypatch.setattr(campaign_control, "_require_capability", lambda value: freeze)
    for consumer in (campaign_control, test_inference):
        monkeypatch.setattr(
            consumer,
            "require_validated_held_out_capability",
            lambda value: value,
        )
    capability = _synthetic_held_out_capability(tmp_path / "capability", freeze)
    monkeypatch.setattr(
        test_inference,
        "materialize_official_test",
        lambda capability, operation: operation(),
    )
    packages = _synthetic_final_packages(tmp_path / "packages", freeze)
    missing = packages[2]
    by_path = {package.directory: package for package in packages}
    monkeypatch.setattr(test_inference, "validate_final_package", lambda path: by_path[Path(path)])
    configs = {
        package.package_id: _final_package_config_from_input(
            _final_input_projection(
                load_symile_development_config(
                    "configs/symile_" + package.manifest["input"]["family"]["family_id"] + ".yaml",
                    bundle_id="bundle-" + "a" * 64,
                    cv_assignment_id="cv-assignment-" + "c" * 64,
                )
            ),
        )
        for package in packages
    }
    monkeypatch.setattr(
        test_inference, "load_final_package_config", lambda p: configs[p.package_id]
    )
    runtime = cpu_runtime()
    projection = _projection_for_freeze(freeze, grouped=False)
    frame = projection.frame()
    frame["sample_id"] = [f"symile:{index:03d}" for index in range(len(frame))]
    projection._test_data._canonical_frame = frame.copy()
    projection = projection._test_data.evaluation_projection()
    frame["source_row"] = np.arange(len(frame))
    data = projection._test_data
    data._canonical_frame = frame.copy()
    private = tmp_path / "private"
    existing = [
        test_inference._publish(
            p.package_id, data, np.linspace(-1, 1, len(frame)), private, capability
        )
        for p in packages
        if p is not missing
    ]
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for item in existing
        for path in item.directory.iterdir()
    }
    events = []
    original_configure = test_inference.configure_neural_determinism
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)

    def configure():
        original_configure()
        events.append("determinism")

    def materialize(*args, **kwargs):
        assert events == ["determinism"]
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        events.append("materialize")
        return data

    monkeypatch.setattr(test_inference, "configure_neural_determinism", configure)
    monkeypatch.setattr(test_inference, "FrozenSymileTestData", materialize)
    cxr_store_type = test_inference.FrozenSymileTestCxrStore

    def cxr_store(*args):
        del args
        store = object.__new__(cxr_store_type)
        store._freeze = freeze
        store._bundle_id = str(freeze.manifest["bundle_id"])
        store.canonical_image = lambda row: np.full((8, 8), row / len(frame), dtype=np.float32)
        return store

    monkeypatch.setattr(test_inference, "FrozenSymileTestCxrStore", cxr_store)

    class MeanImage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.ones(()))

        def forward(self, image):
            return image.mean(dim=(1, 2, 3)) * self.scale

    missing.directory.mkdir(parents=True)
    missing.manifest["package_kind"] = "neural"
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "model_state_dict": MeanImage().state_dict(),
            "terminal_training": {
                "stage1_epochs": 2,
                "stage2_epochs": 1,
                "scheduler": None,
                "early_stopping": None,
                "selection": None,
            },
        },
        missing.directory / "model.pt",
    )
    monkeypatch.setattr(
        final_packages, "load_final_package_config", lambda p: configs[p.package_id]
    )
    monkeypatch.setattr(final_packages, "_reconstruct_neural", lambda config: MeanImage())
    original_load_model = test_inference.load_final_neural_model

    def load_model(package):
        assert package is missing
        events.append("load_model")
        return original_load_model(package)

    original_infer = test_inference.deterministic_inference

    def infer(model, loader, **kwargs):
        assert kwargs["runtime"] is runtime
        assert loader.num_workers == 0
        events.append("infer")
        return original_infer(model, loader, **kwargs)

    monkeypatch.setattr(test_inference, "load_final_neural_model", load_model)
    monkeypatch.setattr(test_inference, "deterministic_inference", infer)
    try:
        predictions, observed = test_inference.publish_frozen_test_predictions(
            capability=capability,
            final_packages=packages,
            source_root=tmp_path / "source",
            manifest_root=tmp_path / "manifests",
            private_root=private,
            runtime=runtime,
        )
    finally:
        torch.use_deterministic_algorithms(previous_algorithms, warn_only=previous_warn)
    assert observed is data
    assert events == ["determinism", "materialize", "load_model", "infer"]
    assert len(predictions) == 14
    assert predictions[2].manifest["model_package_id"] == missing.package_id
    for path, original in before.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == original


def test_official_inference_configures_determinism_and_reuses_exact_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    monkeypatch.setattr(campaign_control, "_require_capability", lambda value: freeze)
    runtime = cpu_runtime()
    capability = _synthetic_held_out_capability(tmp_path / "capability", freeze)
    package = SimpleNamespace(package_id="final-package-synthetic")
    events: list[object] = []
    data = SimpleNamespace(evaluation_projection=lambda: object())
    monkeypatch.setattr(test_inference, "_validated_package_membership", lambda *a: (package,))
    monkeypatch.setattr(
        test_inference, "configure_neural_determinism", lambda: events.append("det")
    )
    monkeypatch.setattr(
        test_inference,
        "materialize_official_test",
        lambda capability, operation: (
            events.append("materialize"),
            operation(),
        )[1],
    )
    monkeypatch.setattr(test_inference, "FrozenSymileTestData", lambda *a, **k: data)
    monkeypatch.setattr(
        test_inference, "load_final_package_config", lambda package: SimpleNamespace()
    )
    monkeypatch.setattr(test_inference, "validate_existing_test_predictions", lambda *a, **k: {})

    def infer(*args):
        assert args[-2] is runtime
        events.append("infer")
        return (object(),)

    monkeypatch.setattr(test_inference, "_infer_all", infer)
    predictions, observed_data = test_inference.publish_frozen_test_predictions(
        capability=capability,
        final_packages=(),
        source_root=tmp_path / "source",
        manifest_root=tmp_path / "manifests",
        private_root=tmp_path / "private",
        runtime=runtime,
    )

    assert len(predictions) == 1
    assert observed_data is data
    assert events == ["det", "materialize", "infer"]
