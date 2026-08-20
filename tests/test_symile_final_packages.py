from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
import torch
from neural_test_support import TensorDataset, TinyImageModel, cpu_runtime
from symile_campaign_test_support import (
    _final_provenance,
    _freeze,
    _synthetic_final_packages,
)
from torch import nn

import radfusion.training.neural as neural
import radfusion.training.symile_campaign_control as campaign_control
import radfusion.training.symile_final_packages as final_packages
import radfusion.training.symile_final_training as final_training
from radfusion.data.errors import ManifestBuildError
from radfusion.data.symile_preprocess import LAB_FEATURE_COLUMNS
from radfusion.models.cxr_baseline import PretrainedWeightIdentity
from radfusion.models.symile_tabular import fit_symile_labs_logistic
from radfusion.training.config import load_symile_development_config
from radfusion.training.execution import reused_loader_policy
from radfusion.training.neural import (
    build_terminal_training_loader,
    fit_terminal_two_stage_binary_model,
)
from radfusion.training.symile_families import FINAL_PACKAGE_COUNT
from radfusion.training.symile_final_packages import (
    ValidatedFinalPackage,
    final_training_plan,
    publish_final_neural_package,
    publish_final_tabular_package,
    validate_final_package,
)
from radfusion.training.symile_statistics import (
    final_member_mean_logit_ensemble,
)


@pytest.mark.parametrize("value", [None, {}, {"git_commit": "a" * 40}, {"unexpected": True}])
@pytest.mark.parametrize("neural", [False, True])
def test_final_provenance_rejects_missing_or_partial_fields(value, neural):
    with pytest.raises(ManifestBuildError, match="provenance"):
        final_packages._validate_execution_provenance(value, neural=neural)


def test_final_provenance_rejects_malformed_runtime():
    value = _final_provenance(neural=True)
    value["runtime_provenance"] = {}
    with pytest.raises(ManifestBuildError, match="runtime provenance"):
        final_packages._validate_execution_provenance(value, neural=True)


def test_final_fit_apis_require_and_prevalidate_provenance(monkeypatch) -> None:
    tabular = load_symile_development_config("configs/symile_labs_logistic.yaml")
    neural_config = load_symile_development_config("configs/symile_cxr_densenet.yaml")

    with pytest.raises(TypeError, match="operational"):
        final_training.fit_final_tabular_family(tabular, development=None)
    with pytest.raises(TypeError, match="operational"):
        final_training.fit_final_neural_member(
            neural_config,
            development=None,
            member_seed=17,
            expected_pretrained_scientific_identity=None,
        )

    monkeypatch.setattr(
        final_training,
        "load_symile_development_cohort",
        lambda config: (_ for _ in ()).throw(AssertionError("fitting began")),
    )
    with pytest.raises(ManifestBuildError, match="provenance"):
        final_training.fit_final_tabular_family(tabular, development=None, operational={})
    with pytest.raises(ManifestBuildError, match="provenance"):
        final_training.fit_final_neural_member(
            neural_config,
            development=None,
            member_seed=17,
            expected_pretrained_scientific_identity=None,
            operational={},
        )


def test_final_cardinalities_and_mean_logit_semantics_are_exact() -> None:
    assert FINAL_PACKAGE_COUNT == 14
    rows = []
    for seed, logits in ((17, [-2.0, 2.0]), (42, [0.0, 0.0]), (2026, [2.0, -2.0])):
        for sample, target, logit in zip(("a", "b"), (0, 1), logits, strict=True):
            rows.append(
                {"sample_id": sample, "target": target, "logit": logit, "member_seed": seed}
            )
    result = final_member_mean_logit_ensemble(pd.DataFrame(rows))
    np.testing.assert_allclose(result["logit"], [0.0, 0.0])
    np.testing.assert_allclose(result["probability"], [0.5, 0.5])


def test_final_training_plans_and_logistic_package_are_narrow_and_reconstructable(
    tmp_path: Path,
) -> None:
    assert final_training_plan("labs_logistic", None).budget is None
    assert final_training_plan("labs_lightgbm", 23).budget == 23
    neural = final_training_plan("cxr_labs_gated", 9)
    assert (neural.stage1_epochs, neural.stage2_epochs) == (2, 7)
    with pytest.raises(ManifestBuildError):
        final_training_plan("cxr_labs_gated_no_observedness", 9)

    config = load_symile_development_config("configs/symile_labs_logistic.yaml")
    values = np.arange(400, dtype=np.float64).reshape(8, 50)
    features = pd.DataFrame(values, columns=LAB_FEATURE_COLUMNS[:50])
    for column in LAB_FEATURE_COLUMNS[50:]:
        features[column] = True
    target = np.asarray([0, 1] * 4, dtype=np.int8)
    fit = fit_symile_labs_logistic(
        features,
        target,
        parameters=config.training.parameters,
        selection_metric=config.training.selection_metric,
        lab_policy=str(config.preprocessing["lab_policy"]),
        training_seed=42,
    )
    package = publish_final_tabular_package(
        model_root=tmp_path / "models",
        config=config,
        family_development_id=None,
        plan=final_training_plan("labs_logistic", None),
        model=fit.pipeline,
        operational=_final_provenance(),
    )
    validated = validate_final_package(package.directory)
    assert validated.manifest["execution_scope"] == "full_development"
    assert validated.manifest["family_development_id"] is None
    assert validated.manifest["seed_policy"] == 42
    assert "bundle_manifest_sha256" not in validated.manifest["input"]["dataset"]
    assert validated.manifest["bundle_manifest_sha256"] == config.dataset.bundle_manifest_sha256
    assert "config_source_sha256" not in validated.manifest
    assert "config_semantic_sha256" not in validated.manifest

    for value in (True, 1.0):
        altered = tmp_path / f"altered-manifest-{value!r}"
        shutil.copytree(package.directory, altered)
        manifest_path = altered / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["final_package_schema_version"] = value
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ManifestBuildError, match="manifest contract"):
            validate_final_package(altered, enforce_directory_name=False)

        altered = tmp_path / f"altered-config-{value!r}"
        shutil.copytree(package.directory, altered)
        config_path = altered / "final_fit_config.json"
        packaged_config = json.loads(config_path.read_bytes())
        packaged_config["final_fit_config_schema_version"] = value
        config_path.write_text(json.dumps(packaged_config), encoding="utf-8")
        manifest_path = altered / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["artifacts"] = final_packages._artifact_hashes(altered)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ManifestBuildError, match="configuration differs"):
            validate_final_package(altered, enforce_directory_name=False)


@pytest.mark.parametrize("value", [True, 1.0])
def test_final_neural_checkpoint_requires_exact_integer_schema_version(
    tmp_path: Path, value: object
) -> None:
    path = tmp_path / "model.pt"
    torch.save(
        {
            "checkpoint_schema_version": value,
            "model_state_dict": {"weight": torch.ones(1)},
            "terminal_training": {},
        },
        path,
    )

    with pytest.raises(ManifestBuildError, match="checkpoint contract"):
        final_packages._load_final_neural_checkpoint(path)


def test_final_neural_lifecycle_uses_only_fixed_terminal_epoch_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_symile_development_config("configs/symile_cxr_densenet.yaml")
    assert config.neural is not None
    runtime = cpu_runtime()
    loader = build_terminal_training_loader(
        TensorDataset([0, 1, 0, 1]),
        config=config.neural,
        runtime=runtime,
        seed=17,
        execution=reused_loader_policy(num_workers=0, pin_memory=False),
    )

    stages: list[bool] = []
    original_train = neural.train_one_epoch

    def record_stage(*args: object, **kwargs: object) -> float:
        stages.append(bool(kwargs["warmup"]))
        return original_train(*args, **kwargs)

    monkeypatch.setattr(neural, "train_one_epoch", record_stage)
    result = fit_terminal_two_stage_binary_model(
        TinyImageModel(),
        loader,
        input_keys=("image",),
        config=config.neural,
        runtime=runtime,
        pos_weight=1.0,
        stage1_epochs=1,
        stage2_epochs=1,
        fine_tune_scope="all",
    )

    assert stages == [True, False]
    assert all(value.device.type == "cpu" for value in result.state_dict.values())


def test_narrow_final_cxr_is_compatible_with_full_fusion_ancestry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cxr = load_symile_development_config("configs/symile_cxr_densenet.yaml")
    fusion = load_symile_development_config("configs/symile_cxr_labs_concat.yaml")
    tiny = nn.Linear(2, 1)
    monkeypatch.setattr(final_packages, "_reconstruct_neural", lambda config: nn.Linear(2, 1))
    fingerprint = {
        "declared_name": cxr.family.parameters["weights"],
        "stable_identifier": "synthetic-weight",
        "cache_filename": "synthetic.pt",
        "byte_size": 4,
        "sha256": "a" * 64,
    }
    package = publish_final_neural_package(
        model_root=tmp_path,
        config=cxr,
        family_development_id="development-" + "1" * 64,
        plan=final_training_plan("cxr_densenet", 3),
        seed=17,
        state_dict=tiny.state_dict(),
        lab_preprocessor=None,
        source_cxr_package_id=None,
        pretrained_weight=fingerprint,
        operational=_final_provenance(neural=True),
    )
    reconstructed = final_packages.load_final_package_config(package)
    checkpoint_loads = []
    original_load = torch.load

    def record_load(*args, **kwargs):
        checkpoint_loads.append(args[0])
        return original_load(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", record_load)
        validate_final_package(package.directory)
    assert checkpoint_loads == [package.directory / "model.pt"]
    assert isinstance(reconstructed, final_packages.FinalPackageConfig)
    assert "warmup_epochs" not in reconstructed.training_parameters
    assert not hasattr(reconstructed, "runtime")
    assert not hasattr(reconstructed, "neural")
    assert "warmup_epochs" in fusion.training.parameters

    altered_training = replace(
        fusion,
        training=replace(
            fusion.training,
            parameters=MappingProxyType(
                {**fusion.training.parameters, "warmup_epochs": 1, "fine_tune_epochs": 1}
            ),
            loader=MappingProxyType({**fusion.training.loader, "batch_size": 1}),
            augmentation=MappingProxyType(
                {**fusion.training.augmentation, "rotation_degrees": 0.0}
            ),
        ),
    )
    final_training._validate_final_cxr_source(altered_training, 17, package)

    for config_name in (
        "configs/symile_cxr_labs_concat.yaml",
        "configs/symile_cxr_labs_gated.yaml",
        "configs/symile_cxr_labs_ecg_gated.yaml",
    ):
        final_training._validate_final_cxr_source(
            load_symile_development_config(config_name), 17, package
        )

    invalid_configs = [
        replace(
            fusion,
            dataset=replace(fusion.dataset, bundle_id="bundle-" + "9" * 64),
        ),
        replace(
            fusion,
            task=replace(fusion.task, task_id="different_task"),
        ),
        replace(
            fusion,
            family=replace(
                fusion.family,
                parameters=MappingProxyType(
                    {**fusion.family.parameters, "encoder_name": "different_encoder"}
                ),
            ),
        ),
        replace(
            fusion,
            preprocessing=MappingProxyType(
                {**fusion.preprocessing, "cxr_transform_policy": "different-transform"}
            ),
        ),
    ]
    for invalid in invalid_configs:
        with pytest.raises(ManifestBuildError, match="incompatible"):
            final_training._validate_final_cxr_source(invalid, 17, package)
    with pytest.raises(ManifestBuildError, match="incompatible"):
        final_training._validate_final_cxr_source(fusion, 42, package)

    wrong_family = ValidatedFinalPackage(
        package.directory,
        {
            **package.manifest,
            "input": {
                **package.manifest["input"],
                "family": {**package.manifest["input"]["family"], "family_id": "cxr_labs_concat"},
            },
        },
        package.manifest_sha256,
    )
    with pytest.raises(ManifestBuildError):
        final_training._validate_final_cxr_source(fusion, 17, wrong_family)

    other = publish_final_neural_package(
        model_root=tmp_path,
        config=cxr,
        family_development_id="development-" + "1" * 64,
        plan=final_training_plan("cxr_densenet", 3),
        seed=42,
        state_dict=tiny.state_dict(),
        lab_preprocessor=None,
        source_cxr_package_id=None,
        pretrained_weight=fingerprint,
        operational=_final_provenance(neural=True),
    )
    mixed = ValidatedFinalPackage(package.directory, other.manifest, other.manifest_sha256)
    monkeypatch.setattr(
        final_training,
        "load_final_neural_state",
        lambda source: (_ for _ in ()).throw(AssertionError("mixed state was consumed")),
    )
    with pytest.raises(ManifestBuildError, match="semantic identity"):
        final_training._final_neural_model(fusion, 42, mixed, None)


def test_final_cxr_fit_rejects_pretrained_identity_different_from_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_symile_development_config("configs/symile_cxr_densenet.yaml")
    actual = PretrainedWeightIdentity(
        declared_name="densenet121-res224-chex",
        stable_identifier="identity-b",
        cache_filename="synthetic.pt",
        byte_size=4,
        sha256="b" * 64,
    )
    monkeypatch.setattr(final_training, "fingerprint_pretrained_weights", lambda weights: actual)
    monkeypatch.setattr(final_training, "StandardCxrEncoder", lambda **kwargs: nn.Identity())
    monkeypatch.setattr(
        final_training, "CxrBinaryClassifier", lambda *args, **kwargs: nn.Identity()
    )

    with pytest.raises(ManifestBuildError, match="differs from development authority"):
        final_training._final_neural_model(
            config,
            17,
            None,
            {
                "declared_name": "densenet121-res224-chex",
                "stable_identifier": "identity-a",
                "sha256": "a" * 64,
            },
        )


def test_final_cxr_members_reject_inconsistent_pretrained_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    packages = list(_synthetic_final_packages(tmp_path / "packages", freeze))
    packages[3] = ValidatedFinalPackage(
        packages[3].directory,
        {
            **packages[3].manifest,
            "pretrained_scientific_identity": {
                "declared_name": "densenet121-res224-chex",
                "stable_identifier": "different-weight",
                "sha256": "b" * 64,
            },
        },
        packages[3].manifest_sha256,
    )
    package_by_path = {package.directory: package for package in packages}
    monkeypatch.setattr(
        campaign_control,
        "validate_final_package",
        lambda directory, **kwargs: package_by_path[Path(directory)],
    )

    with pytest.raises(ManifestBuildError, match="inconsistent pretrained identities"):
        campaign_control._validated_packages(packages)
