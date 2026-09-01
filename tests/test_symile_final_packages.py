from __future__ import annotations

import io
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

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
from symile_lifecycle_test_support import prospective_symile_mutation
from torch import nn

import beyondcxr.training.neural as neural
import beyondcxr.training.symile_campaign_control as campaign_control
import beyondcxr.training.symile_final_packages as final_packages
import beyondcxr.training.symile_final_training as final_training
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_preprocess import LAB_FEATURE_COLUMNS
from beyondcxr.models.cxr_baseline import PretrainedWeightIdentity
from beyondcxr.models.symile_tabular import fit_symile_labs_logistic
from beyondcxr.training.config import load_symile_development_config, with_runtime
from beyondcxr.training.execution import reused_loader_policy
from beyondcxr.training.neural import (
    build_terminal_training_loader,
    fit_terminal_two_stage_binary_model,
)
from beyondcxr.training.symile_families import FINAL_PACKAGE_COUNT
from beyondcxr.training.symile_final_packages import (
    ValidatedFinalPackage,
    final_training_plan,
    publish_final_neural_package,
    publish_final_tabular_package,
    validate_final_package,
)
from beyondcxr.training.symile_statistics import (
    final_member_mean_logit_ensemble,
)


def test_final_config_loads_from_immutable_serving_manifest(tmp_path: Path) -> None:
    from beyondcxr.utils.immutable_json import immutable_json

    config = load_symile_development_config(
        "configs/symile_cxr_labs_gated.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
    path = tmp_path / "final_fit_config.json"
    path.write_bytes(final_packages._final_config_bytes(config))
    manifest = {"input": final_packages._final_input_projection(config)}
    plain = ValidatedFinalPackage(tmp_path, manifest)
    frozen = ValidatedFinalPackage(tmp_path, immutable_json(manifest))
    assert final_packages.load_final_package_config(
        frozen
    ) == final_packages.load_final_package_config(plain)
    document = json.loads(path.read_bytes())
    document["family"]["modalities"].reverse()
    path.write_text(json.dumps(document))
    with pytest.raises(ManifestBuildError, match="differs from manifest"):
        final_packages.load_final_package_config(frozen)


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


def test_final_fit_apis_require_and_prevalidate_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tabular = load_symile_development_config(
        "configs/symile_labs_logistic.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
    neural_config = load_symile_development_config(
        "configs/symile_cxr_densenet.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )

    with prospective_symile_mutation(tmp_path, monkeypatch) as mutation:
        with pytest.raises(TypeError, match="operational"):
            final_training.fit_final_tabular_family(
                tabular,
                prospective_mutation=mutation,
                development=None,
            )
        with pytest.raises(TypeError, match="operational"):
            final_training.fit_final_neural_member(
                neural_config,
                prospective_mutation=mutation,
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
            final_training.fit_final_tabular_family(
                tabular,
                prospective_mutation=mutation,
                development=None,
                operational={},
            )
        with pytest.raises(ManifestBuildError, match="provenance"):
            final_training.fit_final_neural_member(
                neural_config,
                prospective_mutation=mutation,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert final_training_plan("labs_logistic", None).budget is None
    assert final_training_plan("labs_lightgbm", 23).budget == 23
    neural = final_training_plan("cxr_labs_gated", 9)
    assert (neural.stage1_epochs, neural.stage2_epochs) == (2, 7)
    with pytest.raises(ManifestBuildError):
        final_training_plan("cxr_labs_gated_no_observedness", 9)

    config = load_symile_development_config(
        "configs/symile_labs_logistic.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
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
    with prospective_symile_mutation(tmp_path, monkeypatch) as mutation:
        package = publish_final_tabular_package(
            prospective_mutation=mutation,
            model_root=tmp_path / "models",
            config=config,
            family_development_id="development-" + "d" * 64,
            plan=final_training_plan("labs_logistic", None),
            model=fit.pipeline,
            operational=_final_provenance(),
        )
        same_science = publish_final_tabular_package(
            prospective_mutation=mutation,
            model_root=tmp_path / "models",
            config=config,
            family_development_id="development-" + "e" * 64,
            plan=final_training_plan("labs_logistic", None),
            model=fit.pipeline,
            operational=_final_provenance(),
        )
        resume_config = with_runtime(config, model_directory=tmp_path / "models")
        monkeypatch.setattr(
            final_training,
            "load_symile_development_cohort",
            lambda config: pytest.fail("Completed final package was fitted again"),
        )
        resumed = final_training.fit_final_tabular_family(
            resume_config,
            prospective_mutation=mutation,
            development=None,
            operational={**_final_provenance(), "mlflow_run_id": "resumed-run"},
        )
        assert resumed.package_id == package.package_id
        manifest_path = package.directory / "manifest.json"
        original = manifest_path.read_bytes()
        changed = json.loads(original)
        changed["execution_provenance"]["git_commit"] = "f" * 40
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ManifestBuildError, match="conflicts with the prospective campaign"):
            final_training.fit_final_tabular_family(
                resume_config,
                prospective_mutation=mutation,
                development=None,
                operational=_final_provenance(),
            )
        manifest_path.write_bytes(original)
        fit.pipeline.named_steps["classifier"].intercept_ += 0.25
        publish_final_tabular_package(
            prospective_mutation=mutation,
            model_root=tmp_path / "models",
            config=config,
            family_development_id=None,
            plan=final_training_plan("labs_logistic", None),
            model=fit.pipeline,
            operational=_final_provenance(),
        )
        with pytest.raises(ManifestBuildError, match="selection is ambiguous"):
            final_training.fit_final_tabular_family(
                resume_config,
                prospective_mutation=mutation,
                development=None,
                operational=_final_provenance(),
            )
    validated = validate_final_package(package.directory)
    assert validated.manifest["execution_scope"] == "full_development"
    assert validated.manifest["family_development_id"] is None
    assert same_science.package_id == package.package_id
    assert validated.manifest["seed_policy"] == 42

    manifest_path = package.directory / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest_path.write_text(
        json.dumps(dict(reversed(list(validated.manifest.items()))), indent=4),
        encoding="utf-8",
    )
    assert manifest_path.read_bytes() != original_manifest
    reformatted = validate_final_package(package.directory, expected_package_id=package.package_id)
    assert reformatted == validated
    manifest_path.write_bytes(original_manifest)

    for member in package.directory.iterdir():
        altered = tmp_path / f"linked-{member.name}"
        shutil.copytree(package.directory, altered)
        linked_member = altered / member.name
        linked_member.unlink()
        linked_member.symlink_to(member)
        with pytest.raises(ManifestBuildError, match="regular files"):
            validate_final_package(altered, enforce_directory_name=False)

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
    config = load_symile_development_config(
        "configs/symile_cxr_densenet.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
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
    from beyondcxr.utils.operational_logging import configured_logging, parse_operational_record

    stream = io.StringIO()
    with configured_logging(stream=stream, context={}):
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
            operational_fields={"family_id": "cxr_densenet", "seed": 17},
        )

    assert stages == [True, False]
    assert all(value.device.type == "cpu" for value in result.state_dict.values())
    records = [parse_operational_record(line) for line in stream.getvalue().splitlines()]
    assert [(record["event"], record["stage"], record["global_epoch"]) for record in records] == [
        ("epoch_started", "warmup", "1"),
        ("epoch_completed", "warmup", "1"),
        ("epoch_started", "fine_tune", "2"),
        ("epoch_completed", "fine_tune", "2"),
    ]
    assert [record["stage_epoch"] for record in records] == ["1", "1", "1", "1"]
    assert all(record["family_id"] == "cxr_densenet" for record in records)
    assert all(record["seed"] == "17" for record in records)
    completed = [record for record in records if record["event"] == "epoch_completed"]
    assert all({"elapsed_s", "training_loss"} <= record.keys() for record in completed)


def test_narrow_final_cxr_is_compatible_with_full_fusion_ancestry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cxr = load_symile_development_config(
        "configs/symile_cxr_densenet.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
    fusion = load_symile_development_config(
        "configs/symile_cxr_labs_concat.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
    tiny = nn.Linear(2, 1)
    monkeypatch.setattr(final_packages, "_reconstruct_neural", lambda config: nn.Linear(2, 1))
    fingerprint = {
        "declared_name": cxr.family.parameters["weights"],
        "stable_identifier": "weights",
        "cache_filename": "weights.pt",
        "byte_size": 1,
        "sha256": "c" * 64,
    }
    with prospective_symile_mutation(tmp_path, monkeypatch) as mutation:
        package = publish_final_neural_package(
            prospective_mutation=mutation,
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
        resume_config = with_runtime(cxr, model_directory=tmp_path, source_root=tmp_path / "source")
        monkeypatch.setattr(
            final_training,
            "load_symile_development_cohort",
            lambda config: pytest.fail("Completed neural member was fitted again"),
        )
        development = SimpleNamespace(
            manifest={
                "family_id": "cxr_densenet",
                "development_id": "development-" + "1" * 64,
                "final_training_budget": 3,
            }
        )
        arguments = {
            "prospective_mutation": mutation,
            "development": development,
            "member_seed": 17,
            "expected_pretrained_scientific_identity": package.manifest[
                "pretrained_scientific_identity"
            ],
            "operational": _final_provenance(),
        }
        assert final_training.fit_final_neural_member(resume_config, **arguments).package_id == (
            package.package_id
        )
        development.manifest["final_training_budget"] = 4
        with pytest.raises(ManifestBuildError, match="conflicts with the prospective campaign"):
            final_training.fit_final_neural_member(resume_config, **arguments)
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
            load_symile_development_config(
                config_name,
                bundle_id="bundle-" + "a" * 64,
                cv_assignment_id="cv-assignment-" + "c" * 64,
            ),
            17,
            package,
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
    )
    with pytest.raises(ManifestBuildError):
        final_training._validate_final_cxr_source(fusion, 17, wrong_family)

    with prospective_symile_mutation(tmp_path, monkeypatch) as mutation:
        other = publish_final_neural_package(
            prospective_mutation=mutation,
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
    mixed = ValidatedFinalPackage(package.directory, other.manifest)
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
    config = load_symile_development_config(
        "configs/symile_cxr_densenet.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
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
    )
    package_by_path = {package.directory: package for package in packages}
    monkeypatch.setattr(
        campaign_control,
        "validate_final_package",
        lambda directory, **kwargs: package_by_path[Path(directory)],
    )

    with pytest.raises(ManifestBuildError, match="inconsistent pretrained identities"):
        campaign_control._validated_packages(packages)
