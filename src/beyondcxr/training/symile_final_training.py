"""Fit final Symile packages on all development admissions without test access."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_preprocess import LAB_FEATURE_COLUMNS, SymileLabEcdfTransformer
from beyondcxr.data.symile_source import DevelopmentReleaseAsset
from beyondcxr.models.cxr_baseline import (
    CxrBinaryClassifier,
    StandardCxrEncoder,
    fingerprint_pretrained_weights,
)
from beyondcxr.models.fusion_concat import initialize_fusion_encoder
from beyondcxr.models.symile_ecg_fusion import build_symile_trimodal_gated_model
from beyondcxr.models.symile_fusion import build_symile_concat_model, build_symile_gated_model
from beyondcxr.models.symile_tabular import (
    fit_final_symile_labs_lightgbm,
    fit_symile_labs_logistic,
)
from beyondcxr.training.config import ConfigError, ExperimentConfig
from beyondcxr.training.device import resolve_device
from beyondcxr.training.execution import reused_loader_policy
from beyondcxr.training.neural import (
    build_terminal_training_loader,
    fit_terminal_two_stage_binary_model,
    seed_neural_runtime,
)
from beyondcxr.training.symile_data import (
    SymileCxrStore,
    SymileNeuralDataset,
    load_symile_development_cohort,
)
from beyondcxr.training.symile_ecg_data import SymileEcgStore, SymileTriModalDataset
from beyondcxr.training.symile_execution import (
    ValidatedSymileProspectiveMutation,
    require_symile_prospective_mutation,
    require_symile_prospective_path,
)
from beyondcxr.training.symile_families import (
    FINAL_NEURAL_MEMBER_SEEDS,
    FINAL_TABULAR_FAMILIES,
    FINAL_TABULAR_SEED,
    SYMILE_ALL_FUSION_FAMILIES,
    SYMILE_ECG_GATED_FAMILY,
)
from beyondcxr.training.symile_final_packages import (
    FinalTrainingPlan,
    ValidatedFinalPackage,
    final_cxr_ancestry_projection,
    final_input_projection_from_development,
    final_training_plan,
    load_final_neural_state,
    publish_final_neural_package,
    publish_final_tabular_package,
    validate_final_fit_provenance,
    validate_final_package,
)
from beyondcxr.utils.package_identity import (
    package_scientific_config_payload,
    pretrained_weight_semantic_identity,
)
from beyondcxr.utils.publication import is_publication_staging_directory
from beyondcxr.utils.symile_publication import ValidatedDevelopmentResult


def fit_final_tabular_family(
    config: ExperimentConfig,
    *,
    prospective_mutation: ValidatedSymileProspectiveMutation,
    development: ValidatedDevelopmentResult | None,
    operational: Mapping[str, object],
) -> ValidatedFinalPackage:
    """Fit one final tabular family on the complete development cohort."""
    require_symile_prospective_mutation(prospective_mutation)
    validate_final_fit_provenance(operational)
    require_symile_prospective_path(
        prospective_mutation,
        config.runtime.model_directory,
        "Symile final-fitting model root",
    )
    family = config.family.family_id
    if family == "labs_logistic":
        existing = _completed_final_fit(
            config,
            prospective_mutation,
            final_training_plan(family, None),
            seed=FINAL_TABULAR_SEED,
            development_id=None,
        )
        if existing is not None:
            return existing
        return _fit_final_logistic(config, operational, prospective_mutation)
    if family != "labs_lightgbm":
        raise ManifestBuildError("Final tabular fitting requires a tabular family")
    if development is None:
        raise ManifestBuildError("Final LightGBM fitting requires its development authority")
    if development.manifest["family_id"] != family:
        raise ManifestBuildError("Final family development authority has the wrong family")
    budget = development.manifest["final_training_budget"]
    plan = final_training_plan(family, budget)
    existing = _completed_final_fit(
        config,
        prospective_mutation,
        plan,
        seed=FINAL_TABULAR_SEED,
        development_id=str(development.manifest["development_id"]),
    )
    if existing is not None:
        return existing
    return _fit_final_lightgbm(config, development, plan, operational, prospective_mutation)


def _fit_final_logistic(
    config: ExperimentConfig,
    operational: Mapping[str, object],
    prospective_mutation: ValidatedSymileProspectiveMutation,
) -> ValidatedFinalPackage:
    data = load_symile_development_cohort(config)
    fit = fit_symile_labs_logistic(
        data.frame.loc[:, LAB_FEATURE_COLUMNS],
        data.frame["target"].to_numpy(dtype=np.int8),
        parameters=config.training.parameters,
        selection_metric=config.training.selection_metric,
        lab_policy=str(config.preprocessing["lab_policy"]),
        training_seed=FINAL_TABULAR_SEED,
    )
    package = publish_final_tabular_package(
        prospective_mutation=prospective_mutation,
        model_root=config.runtime.model_directory,
        config=config,
        family_development_id=None,
        plan=final_training_plan("labs_logistic", None),
        model=fit.pipeline,
        operational=operational,
    )
    return package


def _fit_final_lightgbm(
    config: ExperimentConfig,
    development: ValidatedDevelopmentResult,
    plan: FinalTrainingPlan,
    operational: Mapping[str, object],
    prospective_mutation: ValidatedSymileProspectiveMutation,
) -> ValidatedFinalPackage:
    data = load_symile_development_cohort(config)
    model = fit_final_symile_labs_lightgbm(
        data.frame.loc[:, LAB_FEATURE_COLUMNS],
        data.frame["target"].to_numpy(dtype=np.int8),
        parameters={**config.family.parameters, **config.training.parameters},
        n_estimators=int(plan.budget or 0),
        lab_policy=str(config.preprocessing["lab_policy"]),
        training_seed=FINAL_TABULAR_SEED,
    )
    package = publish_final_tabular_package(
        prospective_mutation=prospective_mutation,
        model_root=config.runtime.model_directory,
        config=config,
        family_development_id=str(development.manifest["development_id"]),
        plan=plan,
        model=model,
        operational=operational,
    )
    return package


def fit_final_neural_member(
    config: ExperimentConfig,
    *,
    prospective_mutation: ValidatedSymileProspectiveMutation,
    development: ValidatedDevelopmentResult,
    member_seed: int,
    expected_pretrained_scientific_identity: Mapping[str, object] | None,
    operational: Mapping[str, object],
    source_cxr_package: ValidatedFinalPackage | None = None,
    development_assets: tuple[DevelopmentReleaseAsset, ...] | None = None,
) -> ValidatedFinalPackage:
    """Fit and publish exactly one final neural member for one explicit seed."""
    require_symile_prospective_mutation(prospective_mutation)
    validate_final_fit_provenance(operational)
    require_symile_prospective_path(
        prospective_mutation,
        config.runtime.model_directory,
        "Symile final-fitting model root",
    )
    family = config.family.family_id
    if family in FINAL_TABULAR_FAMILIES or family == "cxr_labs_gated_no_observedness":
        raise ManifestBuildError("Final neural member fitting requires a final neural family")
    if development.manifest["family_id"] != family:
        raise ManifestBuildError("Final family development authority has the wrong family")
    plan = final_training_plan(family, development.manifest["final_training_budget"])
    if (
        isinstance(member_seed, bool)
        or not isinstance(member_seed, int)
        or member_seed not in FINAL_NEURAL_MEMBER_SEEDS
    ):
        raise ManifestBuildError("Final neural member seed is outside the frozen seed policy")
    if config.neural is None or config.runtime.source_root is None:
        raise ConfigError("Final neural fitting requires source data and neural configuration")
    source_id = None
    if family in SYMILE_ALL_FUSION_FAMILIES:
        if expected_pretrained_scientific_identity is not None or source_cxr_package is None:
            raise ManifestBuildError("Final fusion requires its same-seed CXR source authority")
        source_id = _validate_final_cxr_source(config, member_seed, source_cxr_package).package_id
    elif source_cxr_package is not None:
        raise ManifestBuildError("Final CXR cannot declare a source package")
    existing = _completed_final_fit(
        config,
        prospective_mutation,
        plan,
        seed=member_seed,
        development_id=str(development.manifest["development_id"]),
        source_cxr_package_id=source_id,
        pretrained_identity=expected_pretrained_scientific_identity,
    )
    if existing is not None:
        return existing
    data = load_symile_development_cohort(config)
    runtime = resolve_device(
        config.runtime.device,
        mixed_precision=config.neural.mixed_precision,
        pin_memory_policy=config.runtime.pin_memory_policy,
    )
    execution = reused_loader_policy(
        num_workers=config.runtime.num_workers,
        pin_memory=runtime.pin_memory_effective,
    )
    cxr_store = SymileCxrStore(
        data.bundle,
        config.runtime.source_root,
        development_assets=development_assets,
    )
    ecg_store = (
        SymileEcgStore(
            data.bundle,
            config.runtime.source_root,
            development_assets=development_assets,
        )
        if config.family.family_id == SYMILE_ECG_GATED_FAMILY
        else None
    )
    transformer = None
    labs = None
    if config.family.family_id in SYMILE_ALL_FUSION_FAMILIES:
        transformer = SymileLabEcdfTransformer().fit(data.frame.loc[:, LAB_FEATURE_COLUMNS])
        labs = transformer.transform(data.frame.loc[:, LAB_FEATURE_COLUMNS])
    seed_neural_runtime(member_seed)
    model, source_id, pretrained_weight = _final_neural_model(
        config,
        member_seed,
        source_cxr_package,
        expected_pretrained_scientific_identity,
    )
    model.to(runtime.device)
    dataset = _final_dataset(
        config,
        data.frame,
        cxr_store=cxr_store,
        ecg_store=ecg_store,
        labs=labs,
        seed=member_seed,
    )
    loader = build_terminal_training_loader(
        dataset,
        config=config.neural,
        runtime=runtime,
        seed=member_seed,
        execution=execution,
    )
    result = fit_terminal_two_stage_binary_model(
        model,
        loader,
        input_keys=_input_keys(config.family.family_id),
        config=config.neural,
        runtime=runtime,
        pos_weight=float(config.training.parameters["pos_weight"]),
        stage1_epochs=int(plan.stage1_epochs or 0),
        stage2_epochs=int(plan.stage2_epochs or 0),
        fine_tune_scope=str(config.training.parameters["fine_tune_scope"]),
    )
    publication_operational = {
        **operational,
        "runtime_provenance": runtime.provenance(),
        "loader_execution": {
            **execution.provenance(),
            "batch_size": config.neural.batch_size,
            "drop_last": False,
            "shuffle": not getattr(dataset, "epoch_tagged_requests", False),
            "sampler": (
                "epoch_permutation" if getattr(dataset, "epoch_tagged_requests", False) else None
            ),
            "persistent_workers": execution.persistent_workers,
            "prefetch_factor": execution.prefetch_factor,
            "multiprocessing_context": "spawn" if execution.num_workers > 0 else None,
        },
    }
    return publish_final_neural_package(
        prospective_mutation=prospective_mutation,
        model_root=config.runtime.model_directory,
        config=config,
        family_development_id=str(development.manifest["development_id"]),
        plan=plan,
        seed=member_seed,
        state_dict=result.state_dict,
        lab_preprocessor=transformer,
        source_cxr_package_id=source_id,
        pretrained_weight=pretrained_weight,
        operational=publication_operational,
    )


def _completed_final_fit(
    config: ExperimentConfig,
    prospective_mutation: ValidatedSymileProspectiveMutation,
    plan: FinalTrainingPlan,
    *,
    seed: int,
    development_id: str | None,
    source_cxr_package_id: str | None = None,
    pretrained_identity: Mapping[str, object] | None = None,
) -> ValidatedFinalPackage | None:
    """Reuse the sole validated final fit for this exact prospective coordinate."""
    from beyondcxr.training.symile_campaign_control import final_package_belongs_to_execution

    root = Path(config.runtime.model_directory) / "final" / "packages"
    if not root.exists() and not root.is_symlink():
        return None
    if root.is_symlink() or not root.is_dir():
        raise ManifestBuildError("Symile final package root is invalid")
    expected = {
        "input": final_input_projection_from_development(package_scientific_config_payload(config)),
        "family_development_id": development_id,
        "final_training_budget": plan.budget,
        "final_stage_budgets": (
            None
            if plan.family_id in FINAL_TABULAR_FAMILIES
            else {
                "stage1_epochs": plan.stage1_epochs,
                "stage2_epochs": plan.stage2_epochs,
            }
        ),
        "source_cxr_package_id": source_cxr_package_id,
        "pretrained_scientific_identity": pretrained_identity,
    }
    matches = []
    for directory in root.iterdir():
        if is_publication_staging_directory(directory):
            continue
        package = validate_final_package(directory)
        manifest = package.manifest
        if (
            manifest["input"]["family"]["family_id"] != plan.family_id
            or manifest["seed_policy"] != seed
        ):
            continue
        if any(manifest[key] != value for key, value in expected.items()) or not (
            final_package_belongs_to_execution(
                package,
                prospective_mutation.lifecycle.current_execution.execution.manifest,
            )
        ):
            raise ManifestBuildError("Existing final fit conflicts with the prospective campaign")
        matches.append(package)
    if len(matches) > 1:
        raise ManifestBuildError("Symile final fit selection is ambiguous")
    return matches[0] if matches else None


def _final_neural_model(
    config: ExperimentConfig,
    seed: int,
    source_cxr_package: ValidatedFinalPackage | None,
    expected_pretrained_scientific_identity: Mapping[str, object] | None,
) -> tuple[torch.nn.Module, str | None, Mapping[str, object] | None]:
    family = config.family.family_id
    parameters = config.family.parameters
    if family == "cxr_densenet":
        weights = str(parameters["weights"])
        before = fingerprint_pretrained_weights(weights)
        model = CxrBinaryClassifier(
            StandardCxrEncoder(
                weights=weights,
                expected_embedding_dimension=int(parameters["embedding_dimension"]),
                image_size=int(parameters["image_size"]),
            ),
            embedding_dimension=int(parameters["embedding_dimension"]),
            image_size=int(parameters["image_size"]),
        )
        after = fingerprint_pretrained_weights(weights)
        if before != after:
            raise ManifestBuildError("Final CXR pretrained bytes changed during model construction")
        if pretrained_weight_semantic_identity(before.as_dict()) != (
            expected_pretrained_scientific_identity
        ):
            raise ManifestBuildError(
                "Final CXR pretrained identity differs from development authority"
            )
        return model, None, before.as_dict()
    if expected_pretrained_scientific_identity is not None:
        raise ManifestBuildError("Final fusion cannot declare a pretrained identity authority")
    if source_cxr_package is None:
        raise ManifestBuildError("Final fusion fitting requires its same-seed final CXR package")
    source = _validate_final_cxr_source(config, seed, source_cxr_package)
    if family == "cxr_labs_concat":
        model = build_symile_concat_model(parameters, weights=None)
    elif family == "cxr_labs_gated":
        model = build_symile_gated_model(parameters, weights=None)
    elif family == SYMILE_ECG_GATED_FAMILY:
        model = build_symile_trimodal_gated_model(parameters, weights=None)
    else:
        raise ManifestBuildError("Final neural family is unsupported")
    initialize_fusion_encoder(model, load_final_neural_state(source))
    return model, source.package_id, None


def _validate_final_cxr_source(
    config: ExperimentConfig, seed: int, source: ValidatedFinalPackage
) -> ValidatedFinalPackage:
    expected_id = source.package_id
    validated = validate_final_package(
        source.directory,
        expected_package_id=expected_id,
        enforce_directory_name=False,
    )
    if validated.manifest != source.manifest:
        raise ManifestBuildError("Final CXR source wrapper differs from its validated package")
    manifest = validated.manifest
    family = manifest.get("input", {}).get("family", {}).get("family_id")
    if (
        manifest.get("package_kind") != "neural"
        or family != "cxr_densenet"
        or manifest.get("seed_policy") != seed
        or final_cxr_ancestry_projection(manifest["input"])
        != final_cxr_ancestry_projection(
            final_input_projection_from_development(package_scientific_config_payload(config))
        )
    ):
        raise ManifestBuildError("Final CXR source package is incompatible with fusion")
    return validated


def _final_dataset(
    config: ExperimentConfig,
    frame: object,
    *,
    cxr_store: SymileCxrStore,
    ecg_store: SymileEcgStore | None,
    labs: np.ndarray | None,
    seed: int,
) -> SymileNeuralDataset | SymileTriModalDataset:
    if not hasattr(frame, "sort_values"):
        raise ManifestBuildError("Final development cohort is invalid")
    ordered = frame.sort_values("sample_id", kind="stable").reset_index(drop=True)
    transform = StandardCxrTransform(
        training=True,
        policy_version=str(config.preprocessing["cxr_transform_policy"]),
        image_size=int(config.family.parameters["image_size"]),
        rotation_degrees=config.neural.rotation_degrees if config.neural is not None else 0.0,
        translation_fraction=(
            config.neural.translation_fraction if config.neural is not None else 0.0
        ),
        brightness_jitter=config.neural.brightness_jitter if config.neural is not None else 0.0,
        contrast_jitter=config.neural.contrast_jitter if config.neural is not None else 0.0,
    )
    if config.family.family_id == SYMILE_ECG_GATED_FAMILY:
        if ecg_store is None or labs is None:
            raise ManifestBuildError("Final ECG gated dataset is incomplete")
        return SymileTriModalDataset(
            ordered,
            cxr_store=cxr_store,
            ecg_store=ecg_store,
            transform=transform,
            training_seed=seed,
            labs=labs,
        )
    return SymileNeuralDataset(
        ordered,
        cxr_store=cxr_store,
        transform=transform,
        training_seed=seed,
        labs=labs,
    )


def _input_keys(family: str) -> tuple[str, ...]:
    if family == "cxr_densenet":
        return ("image",)
    if family == SYMILE_ECG_GATED_FAMILY:
        return ("image", "structured", "ecg")
    return ("image", "structured")
