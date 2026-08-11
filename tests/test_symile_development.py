from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
from sklearn.linear_model import LogisticRegression

import radfusion.training.symile_analysis as symile_analysis
import radfusion.training.symile_data as symile_data
import radfusion.training.symile_development as symile_development
import radfusion.utils.symile_publication as symile_publication
from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.errors import ManifestBuildError
from radfusion.data.symile_artifacts import (
    SymileBundlePaths,
    ValidatedSymileBundleReference,
)
from radfusion.data.symile_cv import ValidatedSymileCvReference
from radfusion.data.symile_preprocess import LAB_FEATURE_COLUMNS, SymileLabEcdfTransformer
from radfusion.data.symile_schemas import CV_SCHEMA
from radfusion.models.symile_tabular import (
    fit_symile_labs_lightgbm,
    fit_symile_labs_logistic,
)
from radfusion.training.config import load_symile_development_config
from radfusion.training.symile_analysis import analyze_symile_development
from radfusion.training.symile_data import (
    SymileCxrStore,
    SymileDevelopmentData,
    derive_inner_split,
    load_symile_development,
    materialize_outer_fold,
)
from radfusion.training.symile_development import aggregate_family_folds
from radfusion.utils.symile_publication import (
    OOF_SCHEMA,
    ValidatedFoldPackage,
    build_oof_table,
    neural_checkpoint_document,
    publish_analysis_result,
    publish_development_result,
    publish_fold_package,
    validate_analysis_result,
    validate_development_result,
    validate_fold_package,
)


def _development_frame(rows: int = 50) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for index in range(rows):
        record: dict[str, object] = {
            "sample_id": f"sample-{index:03d}",
            "subject_id": 10_000 + index,
            "hadm_id": 20_000 + index,
            "official_split": "train" if index < rows - 10 else "validation",
            "source_row": index if index < rows - 10 else index - (rows - 10),
            "pneumonia_state": index % 2,
            "age_years": 50,
            "sex": "F" if index % 2 else "M",
            "view_position": "AP" if index % 2 else "PA",
            "target": index % 2,
        }
        for column_index, column in enumerate(LAB_FEATURE_COLUMNS[:50]):
            record[column] = float(index + column_index + 1)
        for column in LAB_FEATURE_COLUMNS[50:]:
            record[column] = True
        records.append(record)
    if rows > 5:
        records[5]["subject_id"] = records[0]["subject_id"]
    return pd.DataFrame(records)


def _assignments(frame: pd.DataFrame) -> pa.Table:
    records = [
        {
            "sample_id": sample_id,
            "repeat_seed": seed,
            "outer_fold": index % 5,
        }
        for seed in (17, 42, 2026)
        for index, sample_id in enumerate(frame["sample_id"])
    ]
    return pa.Table.from_pylist(records, schema=CV_SCHEMA).sort_by(
        [("repeat_seed", "ascending"), ("sample_id", "ascending")]
    )


def _bundle(tmp_path: Path) -> SymileBundlePaths:
    root = tmp_path / "manifests/symile"
    directory = root / "builds" / ("build-" + "a" * 64)
    return SymileBundlePaths(
        "build-" + "a" * 64,
        directory,
        directory / "symile_samples.parquet",
        directory / "symile_labs.parquet",
        directory / "symile_manifest_metadata.json",
        root / "CURRENT",
    )


def _data(tmp_path: Path) -> SymileDevelopmentData:
    frame = _development_frame()
    reference = ValidatedSymileCvReference({}, "b" * 64, _assignments(frame))
    return SymileDevelopmentData(_bundle(tmp_path), "c" * 64, reference, frame)


def test_outer_fold_and_inner_split_are_deterministic_and_patient_isolated(
    tmp_path: Path,
) -> None:
    data = _data(tmp_path)
    outer = materialize_outer_fold(data, repeat_seed=42, outer_fold=2)
    first = derive_inner_split(outer)
    second = derive_inner_split(outer)

    assert len(outer.training) == 40
    assert len(outer.holdout) == 10
    assert not set(outer.training["subject_id"]) & set(outer.holdout["subject_id"])
    assert first.inner_seed == second.inner_seed
    assert first.inner_split_id == second.inner_split_id
    assert np.array_equal(first.training_indices, second.training_indices)
    assert first.inner_split_id.startswith("inner-split-")
    shared = outer.training.index[outer.training["subject_id"] == 10_000].tolist()
    training_roles = set(first.training_indices.tolist())
    validation_roles = set(first.validation_indices.tolist())
    assert len(shared) == 2
    assert set(shared) <= training_roles or set(shared) <= validation_roles


def test_development_loader_requests_only_train_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_symile_development_config("configs/symile_labs_logistic.yaml")
    frame = _development_frame()
    bundle = _bundle(tmp_path)
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(symile_data, "resolve_symile_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        symile_data,
        "validate_symile_bundle_reference",
        lambda *args, **kwargs: ValidatedSymileBundleReference(
            {
                "official_membership": {
                    "official_split_assignment_id": config.dataset.official_split_assignment_id
                }
            },
            config.dataset.bundle_manifest_sha256,
        ),
    )
    monkeypatch.setattr(
        symile_data,
        "validate_symile_cv_reference",
        lambda *args, **kwargs: ValidatedSymileCvReference({}, "d" * 64, _assignments(frame)),
    )

    def read_samples(*args: object, official_splits: tuple[str, ...]) -> pd.DataFrame:
        calls.append(tuple(official_splits))
        assert "test" not in official_splits
        return frame.drop(columns=[*LAB_FEATURE_COLUMNS, "target"])

    monkeypatch.setattr(symile_data, "read_symile_samples", read_samples)
    monkeypatch.setattr(
        symile_data,
        "read_symile_labs",
        lambda *args, **kwargs: frame[["sample_id", *LAB_FEATURE_COLUMNS]],
    )

    loaded = load_symile_development(config, enforce_production_counts=False)

    assert len(loaded.frame) == len(frame)
    assert calls == [("train", "validation")]
    assert set(loaded.frame["official_split"]) == {"train", "validation"}


def test_cxr_store_resolves_rows_and_rejects_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = np.stack([np.full((3, 320, 320), value, dtype=np.float32) for value in (0.2, 0.8)])
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)
    paths: dict[str, Path] = {}
    for split in ("train", "validation"):
        path = tmp_path / f"cxr_{split}.npy"
        np.save(path, ((raw - mean) / std).astype(np.float32))
        paths[split] = path
    monkeypatch.setattr(
        symile_data,
        "authenticate_source_asset",
        lambda *args, official_split, **kwargs: paths[official_split],
    )
    store = SymileCxrStore(_bundle(tmp_path), tmp_path)

    assert store.canonical_image("validation", 1).shape == (320, 320)
    assert np.allclose(store.canonical_image("train", 0), 0.2, atol=2e-6)
    with pytest.raises(ManifestBuildError):
        store.canonical_image("test", 0)


def _lab_preprocessor() -> SymileLabEcdfTransformer:
    return SymileLabEcdfTransformer().fit(_development_frame(2)[list(LAB_FEATURE_COLUMNS)])


def _lineage(family: str) -> dict[str, object]:
    config = load_symile_development_config(f"configs/symile_{family}.yaml")
    neural = family in {"cxr", "concat", "gated", "gated_no_observedness"}
    return {
        "bundle_id": config.dataset.bundle_id,
        "bundle_manifest_sha256": config.dataset.bundle_manifest_sha256,
        "official_split_assignment_id": config.dataset.official_split_assignment_id,
        "cv_assignment_id": config.dataset.cv_assignment_id,
        "cv_manifest_sha256": "e" * 64,
        "task_id": config.dataset.task_id,
        "git_commit": "f" * 40,
        "dependency_lock_sha256": "1" * 64,
        "encoder_identity": (
            {
                "library": "torchxrayvision",
                "architecture": "densenet121",
                "weights": "densenet121-res224-chex",
                "embedding_dimension": 1024,
            }
            if neural
            else None
        ),
        "transform_contract": (
            StandardCxrTransform(
                training=False,
                image_size=int(config.model["image_size"]),
                rotation_degrees=config.image.rotation_degrees,
                translation_fraction=config.image.translation_fraction,
                brightness_jitter=config.image.brightness_jitter,
                contrast_jitter=config.image.contrast_jitter,
            ).contract()
            if neural and config.image is not None
            else None
        ),
        "pretrained_weight": (
            {
                "declared_name": "densenet121-res224-chex",
                "stable_identifier": "https://example.test/weights.pt",
                "cache_filename": "weights.pt",
                "byte_size": 1,
                "sha256": "2" * 64,
            }
            if family == "cxr"
            else None
        ),
    }


def _inner_split(seed: int, fold: int) -> dict[str, object]:
    return {
        "inner_split_id": "inner-split-" + "b" * 64,
        "inner_seed": 1,
        "policy": {
            "policy_version": "symile-inner-stratified-group-five-fold-v1",
            "algorithm": "sklearn.model_selection.StratifiedGroupKFold",
            "n_splits": 5,
            "shuffle": True,
            "generated_validation_fold": 0,
            "group_field": "subject_id",
            "stratification_target": "pneumonia_strict",
            "repeat_seed": seed,
            "outer_fold": fold,
            "inner_seed": 1,
        },
    }


def _publish_family_folds(
    tmp_path: Path,
    family: str,
    *,
    offset: float,
    source_cxr_folds: list[ValidatedFoldPackage] | None = None,
    source_cxr_development_id: str | None = None,
) -> tuple[list[ValidatedFoldPackage], str]:
    config = load_symile_development_config(f"configs/symile_{family}.yaml")
    semantic_hash = symile_development.symile_development_semantic_sha256(config)
    neural_family = family in {"cxr", "concat", "gated", "gated_no_observedness"}
    lineage = _lineage(family)
    folds: list[ValidatedFoldPackage] = []
    for seed_index, seed in enumerate((17, 42, 2026)):
        for fold in range(5):
            indices = range(fold * 4, fold * 4 + 4)
            logits = np.asarray(
                [(-1.0 if index % 2 == 0 else 1.0) + offset + seed_index * 0.1 for index in indices]
            )
            oof = build_oof_table(
                [f"sample-{index:03d}" for index in indices],
                [index % 2 for index in indices],
                logits,
            )
            neural = neural_family
            selection = {
                "metric": "roc_auc" if family != "labs_logistic" else "none",
                "selected_epoch": fold + 1 if neural else None,
                "best_iteration": fold + 1 if family == "labs_lightgbm" else None,
            }
            if neural:
                selected_stage = "warmup" if fold + 1 <= 2 else "fine_tune"
                selection.update(
                    selected_stage=selected_stage,
                    selected_validation_metric=0.8,
                )
                model: object = neural_checkpoint_document(
                    {"encoder.weight": torch.ones((1, 1))},
                    selected_epoch=fold + 1,
                    selected_stage=selected_stage,
                    selected_validation_roc_auc=0.8,
                )
            else:
                features = _development_frame(20)[list(LAB_FEATURE_COLUMNS)]
                targets = np.tile(np.array([0, 1], dtype=np.int8), 10)
                if family == "labs_logistic":
                    fitted = fit_symile_labs_logistic(
                        features,
                        targets,
                        parameters=config.model,
                        repeat_seed=seed,
                    )
                else:
                    fitted = fit_symile_labs_lightgbm(
                        features,
                        targets,
                        parameters=config.model,
                        inner_training_indices=np.arange(16, dtype=np.int64),
                        inner_validation_indices=np.arange(16, 20, dtype=np.int64),
                        repeat_seed=seed,
                    )
                    selection["best_iteration"] = fitted.best_iteration
                model = fitted.pipeline
            fusion = family in {"concat", "gated", "gated_no_observedness"}
            source_fold = (
                next(
                    item
                    for item in source_cxr_folds or ()
                    if item.manifest["repeat_seed"] == seed and item.manifest["outer_fold"] == fold
                )
                if fusion
                else None
            )
            package = publish_fold_package(
                model_root=tmp_path / "models/symile/development",
                family=family,
                repeat_seed=seed,
                outer_fold=fold,
                config_bytes=config.source_bytes,
                config_sha256=config.source_sha256,
                semantic_config_sha256=semantic_hash,
                lineage=lineage,
                inner_split=_inner_split(seed, fold),
                selection=selection,
                oof=oof,
                model=model,
                lab_preprocessor=_lab_preprocessor() if fusion else None,
                training_history=(
                    [
                        {
                            "epoch": epoch,
                            "stage": "warmup" if epoch <= 2 else "fine_tune",
                            "training_loss": 0.5,
                            "validation_metric": 0.8,
                            "encoder_learning_rate": None if epoch <= 2 else 1e-5,
                            "head_learning_rate": 1e-3 if epoch <= 2 else 1e-4,
                        }
                        for epoch in range(1, fold + 2)
                    ]
                    if neural
                    else None
                ),
                source_cxr=(
                    {
                        "development_id": source_cxr_development_id,
                        "fold_package_id": source_fold.manifest["fold_package_id"],
                        "fold_manifest_sha256": source_fold.manifest_sha256,
                    }
                    if fusion
                    else None
                ),
                operational={
                    "mlflow_run_id": f"run-{family}-{seed}-{fold}",
                    "runtime_provenance": {} if neural else None,
                },
            )
            folds.append(package)
    return folds, semantic_hash


def test_fold_package_reconstructs_tabular_pipeline_and_validates_oof(tmp_path: Path) -> None:
    folds, _ = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0)
    first = validate_fold_package(folds[0].directory)

    assert first.oof.schema == OOF_SCHEMA
    assert first.oof.num_rows == 4
    assert len(set(first.oof["sample_id"].to_pylist())) == 4
    assert first.manifest["selection"]["metric"] == "none"


def test_fold_publication_rejects_wrong_model_malformed_state_and_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oof = build_oof_table(["sample-0", "sample-1"], [0, 1], [-1.0, 1.0])
    logistic = load_symile_development_config("configs/symile_labs_logistic.yaml")
    wrong_model = LogisticRegression().fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    with pytest.raises(ManifestBuildError, match="exact fitted pipeline"):
        publish_fold_package(
            model_root=tmp_path / "wrong-tabular",
            family="labs_logistic",
            repeat_seed=17,
            outer_fold=0,
            config_bytes=logistic.source_bytes,
            config_sha256=logistic.source_sha256,
            semantic_config_sha256=symile_development.symile_development_semantic_sha256(logistic),
            lineage=_lineage("labs_logistic"),
            inner_split=_inner_split(17, 0),
            selection={"metric": "none", "selected_epoch": None, "best_iteration": None},
            oof=oof,
            model=wrong_model,
            operational={"mlflow_run_id": "run-wrong", "runtime_provenance": None},
        )

    cxr = load_symile_development_config("configs/symile_cxr.yaml")
    selection = {
        "metric": "roc_auc",
        "selected_epoch": 1,
        "best_iteration": None,
        "selected_stage": "warmup",
        "selected_validation_metric": 0.8,
    }
    malformed = neural_checkpoint_document(
        {"unexpected.weight": torch.ones((1, 1))},
        selected_epoch=1,
        selected_stage="warmup",
        selected_validation_roc_auc=0.8,
    )
    with pytest.raises(ManifestBuildError, match="cannot reconstruct exactly"):
        publish_fold_package(
            model_root=tmp_path / "malformed-neural",
            family="cxr",
            repeat_seed=17,
            outer_fold=0,
            config_bytes=cxr.source_bytes,
            config_sha256=cxr.source_sha256,
            semantic_config_sha256=symile_development.symile_development_semantic_sha256(cxr),
            lineage=_lineage("cxr"),
            inner_split=_inner_split(17, 0),
            selection=selection,
            oof=oof,
            model=malformed,
            training_history=[
                {
                    "epoch": 1,
                    "stage": "warmup",
                    "training_loss": 0.5,
                    "validation_metric": 0.8,
                    "encoder_learning_rate": None,
                    "head_learning_rate": 1e-3,
                }
            ],
            operational={"mlflow_run_id": "run-malformed", "runtime_provenance": {}},
        )

    monkeypatch.setattr(symile_publication, "_validate_neural_reconstruction", lambda *args: None)
    history_selection = {**selection, "selected_epoch": 2}
    history_checkpoint = neural_checkpoint_document(
        {"unexpected.weight": torch.ones((1, 1))},
        selected_epoch=2,
        selected_stage="warmup",
        selected_validation_roc_auc=0.8,
    )
    with pytest.raises(ManifestBuildError, match="selected epoch is absent"):
        publish_fold_package(
            model_root=tmp_path / "bad-history",
            family="cxr",
            repeat_seed=17,
            outer_fold=0,
            config_bytes=cxr.source_bytes,
            config_sha256=cxr.source_sha256,
            semantic_config_sha256=symile_development.symile_development_semantic_sha256(cxr),
            lineage=_lineage("cxr"),
            inner_split=_inner_split(17, 0),
            selection=history_selection,
            oof=oof,
            model=history_checkpoint,
            training_history=[
                {
                    "epoch": 1,
                    "stage": "warmup",
                    "training_loss": 0.5,
                    "validation_metric": 0.8,
                    "encoder_learning_rate": None,
                    "head_learning_rate": 1e-3,
                }
            ],
            operational={"mlflow_run_id": "run-history", "runtime_provenance": {}},
        )


def test_family_completeness_medians_analysis_alignment_and_privacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(symile_publication, "DEVELOPMENT_COUNT", 20)
    monkeypatch.setattr(symile_analysis, "DEVELOPMENT_COUNT", 20)
    monkeypatch.setattr(symile_publication, "_validate_neural_reconstruction", lambda *args: None)
    developments = []
    cxr_folds: list[ValidatedFoldPackage] | None = None
    cxr_development_id: str | None = None
    for family_index, family in enumerate(
        ("labs_logistic", "labs_lightgbm", "cxr", "concat", "gated", "gated_no_observedness")
    ):
        folds, semantic_hash = _publish_family_folds(
            tmp_path,
            family,
            offset=family_index * 0.05,
            source_cxr_folds=cxr_folds,
            source_cxr_development_id=cxr_development_id,
        )
        metrics, selected, budget = aggregate_family_folds(
            family,
            folds,
            expected_sample_ids={f"sample-{index:03d}" for index in range(20)},
        )
        if family == "labs_logistic":
            assert selected is None and budget is None
            with pytest.raises(ManifestBuildError, match="exact 3 x 5 folds"):
                publish_development_result(
                    report_root=tmp_path / "reports/symile/development",
                    model_root=tmp_path / "models/symile/development",
                    family=family,
                    semantic_config_sha256=semantic_hash,
                    folds=[*folds[:-1], folds[-2]],
                    repeat_metrics=metrics,
                    selected_values=selected,
                    median_m6_budget=budget,
                )
            corrupted = {seed: dict(values) for seed, values in metrics.items()}
            corrupted["17"]["roc_auc"] -= 0.01
            with pytest.raises(ManifestBuildError, match="differ from fold OOF"):
                publish_development_result(
                    report_root=tmp_path / "reports/symile/development",
                    model_root=tmp_path / "models/symile/development",
                    family=family,
                    semantic_config_sha256=semantic_hash,
                    folds=folds,
                    repeat_metrics=corrupted,
                    selected_values=selected,
                    median_m6_budget=budget,
                )
        else:
            assert selected is not None and len(selected) == 15
            assert budget == int(np.median(selected))
            if family == "labs_lightgbm":
                with pytest.raises(ManifestBuildError, match="median M6 budget"):
                    publish_development_result(
                        report_root=tmp_path / "reports/symile/development",
                        model_root=tmp_path / "models/symile/development",
                        family=family,
                        semantic_config_sha256=semantic_hash,
                        folds=folds,
                        repeat_metrics=metrics,
                        selected_values=selected,
                        median_m6_budget=budget + 1,
                    )
        result = publish_development_result(
            report_root=tmp_path / "reports/symile/development",
            model_root=tmp_path / "models/symile/development",
            family=family,
            semantic_config_sha256=semantic_hash,
            folds=folds,
            repeat_metrics=metrics,
            selected_values=selected,
            median_m6_budget=budget,
        )
        developments.append(result)
        if family == "cxr":
            cxr_folds = folds
            cxr_development_id = str(result.manifest["development_id"])
    for result in developments:
        validate_development_result(
            result.directory,
            model_root=tmp_path / "models/symile/development",
        )
        assert "sample-" not in (result.directory / "summary.md").read_text(encoding="utf-8")

    analysis_id, analysis_directory = analyze_symile_development(
        [str(result.manifest["development_id"]) for result in reversed(developments)],
        report_root=tmp_path / "reports/symile/development",
        model_root=tmp_path / "models/symile/development",
    )
    analysis = validate_analysis_result(
        analysis_directory,
        report_root=tmp_path / "reports/symile/development",
        model_root=tmp_path / "models/symile/development",
        expected_analysis_id=analysis_id,
    )
    assert set(analysis["family_development_ids"]) == {
        "labs_logistic",
        "labs_lightgbm",
        "cxr",
        "concat",
        "gated",
        "gated_no_observedness",
    }
    assert "sample-" not in (analysis_directory / "summary.md").read_text(encoding="utf-8")
    corrupted_effects = {
        comparison: {seed: dict(values) for seed, values in repeats.items()}
        for comparison, repeats in analysis["paired_effects"].items()
    }
    corrupted_effects["gated_minus_cxr"]["17"]["roc_auc"] += 0.01
    with pytest.raises(ManifestBuildError, match="paired_effects differs"):
        publish_analysis_result(
            report_root=tmp_path / "reports/symile/development",
            model_root=tmp_path / "models/symile/development",
            family_development_ids=analysis["family_development_ids"],
            repeat_metrics=analysis["repeat_metrics"],
            paired_effects=corrupted_effects,
            ensemble_metrics=analysis["ensemble_metrics"],
            observedness_ablation=analysis["observedness_ablation"],
            policy=analysis["policy"],
        )
    corrupted_ensemble = {
        family: dict(metrics) for family, metrics in analysis["ensemble_metrics"].items()
    }
    corrupted_ensemble["gated"]["average_precision"] -= 0.01
    with pytest.raises(ManifestBuildError, match="ensemble_metrics differs"):
        publish_analysis_result(
            report_root=tmp_path / "reports/symile/development",
            model_root=tmp_path / "models/symile/development",
            family_development_ids=analysis["family_development_ids"],
            repeat_metrics=analysis["repeat_metrics"],
            paired_effects=analysis["paired_effects"],
            ensemble_metrics=corrupted_ensemble,
            observedness_ablation=analysis["observedness_ablation"],
            policy=analysis["policy"],
        )

    (analysis_directory / "summary.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="summary differs"):
        validate_analysis_result(
            analysis_directory,
            report_root=tmp_path / "reports/symile/development",
            model_root=tmp_path / "models/symile/development",
        )
    first_development = developments[0]
    (first_development.directory / "summary.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="summary differs"):
        validate_development_result(
            first_development.directory,
            model_root=tmp_path / "models/symile/development",
        )


def test_mlflow_completion_failure_rolls_back_new_fold_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_symile_development_config("configs/symile_labs_logistic.yaml")
    config = type(config)(
        **{
            **config.__dict__,
            "training": MappingProxyType(
                {
                    **dict(config.training),
                    "model_directory": tmp_path / "models/symile/development",
                    "report_directory": tmp_path / "reports/symile/development",
                }
            ),
        }
    )
    data = _data(tmp_path)
    data = SymileDevelopmentData(
        data.bundle,
        config.dataset.bundle_manifest_sha256,
        data.cv_reference,
        data.frame,
    )
    context = symile_development.SymileFoldExecutionContext(
        data,
        None,
        None,
        None,
        "f" * 40,
        "a" * 64,
        {},
        None,
    )

    run_arguments: dict[str, object] = {}

    @contextmanager
    def fake_run(**kwargs: object):
        run_arguments.update(kwargs)
        yield "run-id"

    monkeypatch.setattr(symile_development, "tracked_run", fake_run)
    monkeypatch.setattr(symile_development, "log_source_config", lambda *args: None)
    monkeypatch.setattr(symile_development.mlflow, "log_metrics", lambda *args: None)
    monkeypatch.setattr(
        symile_development.mlflow,
        "set_tags",
        lambda *args: (_ for _ in ()).throw(OSError("ledger failure")),
    )

    def fake_fit(*args: object) -> dict[str, object]:
        outer = args[2]
        assert isinstance(outer, symile_data.SymileOuterFold)
        fitted = fit_symile_labs_logistic(
            outer.training[list(LAB_FEATURE_COLUMNS)],
            outer.training["target"].to_numpy(dtype=np.int8),
            parameters=config.model,
            repeat_seed=outer.repeat_seed,
        )
        return {
            "model": fitted.pipeline,
            "logits": np.resize(np.array([-1.0, 1.0]), len(outer.holdout)),
            "selection": {"metric": "none", "selected_epoch": None, "best_iteration": None},
        }

    monkeypatch.setattr(symile_development, "_fit_outer_fold", fake_fit)

    with pytest.raises(OSError):
        symile_development.execute_symile_outer_fold(
            config,
            context,
            repeat_seed=17,
            outer_fold=0,
        )
    assert not list((tmp_path / "models/symile/development/folds").glob("fold-package-*"))
    tags = run_arguments["tags"]
    assert isinstance(tags, dict)
    assert tags["config_name"] == "symile_labs_logistic"
    assert tags["repeat_seed"] == "17"
    assert "seed" not in tags
    assert "experiment_name" not in tags
