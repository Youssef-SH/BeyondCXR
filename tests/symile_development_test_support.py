"""Mechanical synthetic Symile fixture builders."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.symile_preprocess import LAB_FEATURE_COLUMNS, SymileLabEcdfTransformer
from beyondcxr.data.symile_schemas import CV_SCHEMA
from beyondcxr.models.symile_tabular import (
    fit_symile_labs_lightgbm,
    fit_symile_labs_logistic,
)
from beyondcxr.training.config import (
    load_symile_development_config,
)
from beyondcxr.training.symile_data import (
    derive_inner_seed,
)
from beyondcxr.training.symile_development import CompletedSymileFold
from beyondcxr.utils.private_predictions import (
    build_prediction_table,
    publish_prediction_evidence,
)
from beyondcxr.utils.symile_publication import (
    neural_checkpoint_document,
    publish_fold_package,
)


def _development_frame(rows: int = 50) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for index in range(rows):
        record: dict[str, object] = {
            "sample_id": f"symile:{index:03d}",
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


def _lab_preprocessor() -> SymileLabEcdfTransformer:
    return SymileLabEcdfTransformer().fit(_development_frame(2)[list(LAB_FEATURE_COLUMNS)])


def _lineage(family: str, *, cv_manifest_sha256: str = "e" * 64, config=None) -> dict[str, object]:
    config = config or load_symile_development_config(_family_config_path(family))
    neural = family in {
        "cxr_densenet",
        "cxr_labs_concat",
        "cxr_labs_gated",
        "cxr_labs_gated_no_observedness",
        "cxr_labs_ecg_gated",
    }
    return {
        "bundle_id": config.dataset.bundle_id,
        "bundle_manifest_sha256": config.dataset.bundle_manifest_sha256,
        "split_assignment_id": config.dataset.split_assignment_id,
        "cv_assignment_id": config.dataset.cv_assignment_id,
        "cv_manifest_sha256": cv_manifest_sha256,
        "task_id": config.task.task_id,
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
                image_size=int(config.family.parameters["image_size"]),
                rotation_degrees=config.neural.rotation_degrees,
                translation_fraction=config.neural.translation_fraction,
                brightness_jitter=config.neural.brightness_jitter,
                contrast_jitter=config.neural.contrast_jitter,
            ).contract()
            if neural and config.neural is not None
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
            if family == "cxr_densenet"
            else None
        ),
    }


def _family_config_path(family: str) -> str:
    names = {
        "labs_logistic": "symile_labs_logistic",
        "labs_lightgbm": "symile_labs_lightgbm",
        "cxr_densenet": "symile_cxr_densenet",
        "cxr_labs_concat": "symile_cxr_labs_concat",
        "cxr_labs_gated": "symile_cxr_labs_gated",
        "cxr_labs_gated_no_observedness": "symile_cxr_labs_gated_no_observedness",
        "cxr_labs_ecg_gated": "symile_cxr_labs_ecg_gated",
    }
    return f"configs/{names[family]}.yaml"


def _inner_split(seed: int, fold: int) -> dict[str, object]:
    inner_seed = derive_inner_seed(seed, fold)
    split_digest = hashlib.sha256(f"synthetic-inner-split\0{seed}\0{fold}".encode()).hexdigest()
    return {
        "inner_split_id": "inner-split-" + split_digest,
        "inner_seed": inner_seed,
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
            "inner_seed": inner_seed,
        },
    }


def _neural_history(config, selected_epoch: int) -> list[dict[str, object]]:
    assert config.neural is not None
    final_epoch = max(
        config.neural.warmup_epochs + config.neural.early_stopping_patience,
        selected_epoch + config.neural.early_stopping_patience,
    )
    return [
        {
            "epoch": epoch,
            "stage": "warmup" if epoch <= config.neural.warmup_epochs else "fine_tune",
            "training_loss": 0.5,
            "validation_metric": 0.8 if epoch == selected_epoch else 0.7,
            "encoder_learning_rate": (None if epoch <= config.neural.warmup_epochs else 1e-5),
            "head_learning_rate": 1e-3 if epoch <= config.neural.warmup_epochs else 1e-4,
        }
        for epoch in range(1, final_epoch + 1)
    ]


def _publish_family_folds(
    tmp_path: Path,
    family: str,
    *,
    offset: float,
    source_cxr_folds: list[CompletedSymileFold] | None = None,
    source_cxr_development_id: str | None = None,
    authority: pd.DataFrame | None = None,
    cv_manifest_sha256: str = "e" * 64,
    config=None,
) -> tuple[list[CompletedSymileFold], str]:
    config = config or load_symile_development_config(_family_config_path(family))
    semantic_hash = config.config_semantic_sha256
    neural_family = family in {
        "cxr_densenet",
        "cxr_labs_concat",
        "cxr_labs_gated",
        "cxr_labs_gated_no_observedness",
        "cxr_labs_ecg_gated",
    }
    lineage = _lineage(family, cv_manifest_sha256=cv_manifest_sha256, config=config)
    folds: list[CompletedSymileFold] = []
    for seed_index, seed in enumerate((17, 42, 2026)):
        for fold in range(5):
            if authority is None:
                rows = pd.DataFrame(
                    {
                        "sample_id": [f"symile:{index:03d}" for index in range(fold, 20, 5)],
                        "target": [index % 2 for index in range(fold, 20, 5)],
                    }
                )
            else:
                rows = (
                    authority.loc[
                        (authority["repeat_seed"] == seed) & (authority["outer_fold"] == fold),
                        ["sample_id", "target"],
                    ]
                    .sort_values("sample_id", kind="stable")
                    .reset_index(drop=True)
                )
            logits = np.asarray(
                [
                    (-1.0 if int(target) == 0 else 1.0) + offset + seed_index * 0.1
                    for target in rows["target"]
                ]
            )
            oof = build_prediction_table(
                rows["sample_id"].tolist(),
                rows["target"].tolist(),
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
                        parameters=config.training.parameters,
                        selection_metric=config.training.selection_metric,
                        lab_policy=str(config.preprocessing["lab_policy"]),
                        training_seed=seed,
                    )
                else:
                    fitted = fit_symile_labs_lightgbm(
                        features,
                        targets,
                        parameters={**config.family.parameters, **config.training.parameters},
                        selection_metric=config.training.selection_metric,
                        lab_policy=str(config.preprocessing["lab_policy"]),
                        inner_training_indices=np.arange(16, dtype=np.int64),
                        inner_validation_indices=np.arange(16, 20, dtype=np.int64),
                        training_seed=seed,
                    )
                    selection["best_iteration"] = fitted.best_iteration
                model = fitted.pipeline
            fusion = family in {
                "cxr_labs_concat",
                "cxr_labs_gated",
                "cxr_labs_gated_no_observedness",
                "cxr_labs_ecg_gated",
            }
            source_fold = (
                next(
                    item
                    for item in source_cxr_folds or ()
                    if item.package.manifest["repeat_seed"] == seed
                    and item.package.manifest["outer_fold"] == fold
                )
                if fusion
                else None
            )
            package = publish_fold_package(
                model_root=tmp_path / "models/symile/development",
                family=config.family.family_id,
                repeat_seed=seed,
                outer_fold=fold,
                config_bytes=config.source_bytes,
                config_sha256=config.config_source_sha256,
                config_semantic_sha256=semantic_hash,
                lineage=lineage,
                inner_split=_inner_split(seed, fold),
                selection=selection,
                model=model,
                lab_preprocessor=_lab_preprocessor() if fusion else None,
                training_history=(_neural_history(config, fold + 1) if neural else None),
                source_cxr=(
                    {
                        "development_id": source_cxr_development_id,
                        "fold_package_id": source_fold.package.manifest["fold_package_id"],
                        "fold_manifest_sha256": source_fold.package.manifest_sha256,
                    }
                    if fusion
                    else None
                ),
                operational={
                    "mlflow_run_id": f"run-{family}-{seed}-{fold}",
                    "runtime_provenance": {} if neural else None,
                },
            )
            prediction = publish_prediction_evidence(
                private_root=tmp_path / "private",
                dataset_id="symile",
                model_package_id=package.manifest["fold_package_id"],
                task_id=config.task.task_id,
                bundle_id=config.dataset.bundle_id,
                split_assignment_id=config.dataset.split_assignment_id,
                scope="outer_fold_oof",
                sample_ids=oof["sample_id"].to_pylist(),
                targets=oof["target"].to_pylist(),
                logits=oof["logit"].to_pylist(),
                cv_assignment_id=config.dataset.cv_assignment_id,
                repeat_seed=seed,
                outer_fold=fold,
            )
            folds.append(CompletedSymileFold(package, prediction))
    return folds, semantic_hash
