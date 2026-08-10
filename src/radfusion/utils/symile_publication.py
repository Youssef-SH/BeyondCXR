"""Publish and validate the three immutable Symile M5 artifact levels."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.errors import ManifestBuildError
from radfusion.data.hashing import arrow_ipc_sha256, sha256_file
from radfusion.data.symile_preprocess import (
    LAB_FEATURE_COLUMNS,
    SymileLabEcdfTransformer,
    load_symile_lab_preprocessor,
    save_symile_lab_preprocessor,
    validate_symile_lab_preprocessor,
)
from radfusion.models.cxr_baseline import CxrBinaryClassifier, StandardCxrEncoder
from radfusion.models.symile_fusion import build_symile_concat_model, build_symile_gated_model
from radfusion.models.symile_tabular import symile_tabular_logits
from radfusion.training.config import (
    SYMILE_M5_FAMILIES,
    SymileDevelopmentConfig,
    load_symile_development_config,
    symile_development_semantic_sha256,
)
from radfusion.training.symile_data import DEVELOPMENT_COUNT
from radfusion.utils.privacy import validate_public_reports
from radfusion.utils.publication import staging_directory
from radfusion.utils.skops_io import load_skops, save_skops

FOLD_SCHEMA_VERSION = 1
DEVELOPMENT_SCHEMA_VERSION = 1
ANALYSIS_SCHEMA_VERSION = 1
FOLD_PREFIX = "fold-package-"
DEVELOPMENT_PREFIX = "development-"
ANALYSIS_PREFIX = "analysis-"
FOLD_MANIFEST_FILENAME = "fold_manifest.json"
CONFIG_FILENAME = "resolved_config.yaml"
OOF_FILENAME = "oof_predictions.parquet"
TABULAR_MODEL_FILENAME = "model.skops"
NEURAL_MODEL_FILENAME = "model.pt"
LAB_PREPROCESSOR_FILENAME = "lab_preprocessor.skops"
TRAINING_HISTORY_FILENAME = "training_history.json"
DEVELOPMENT_MANIFEST_FILENAME = "development_manifest.json"
ANALYSIS_MANIFEST_FILENAME = "analysis_manifest.json"
SUMMARY_FILENAME = "summary.md"
OOF_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("target", pa.int8(), nullable=False),
        pa.field("logit", pa.float64(), nullable=False),
        pa.field("probability", pa.float64(), nullable=False),
    ]
)
TABULAR_FAMILIES = frozenset({"labs_logistic", "labs_lightgbm"})
NEURAL_FAMILIES = frozenset({"cxr", "concat", "gated", "gated_no_observedness"})
FUSION_FAMILIES = frozenset({"concat", "gated", "gated_no_observedness"})
_FOLD_FIELDS = {
    "fold_schema_version",
    "fold_package_id",
    "family",
    "repeat_seed",
    "outer_fold",
    "config",
    "lineage",
    "inner_split",
    "selection",
    "source_cxr",
    "artifacts",
    "operational",
}
_LINEAGE_FIELDS = {
    "bundle_id",
    "bundle_manifest_sha256",
    "official_split_assignment_id",
    "cv_assignment_id",
    "cv_manifest_sha256",
    "task_id",
    "git_commit",
    "dependency_lock_sha256",
    "encoder_identity",
    "transform_contract",
    "pretrained_weight",
}
_ANALYSIS_POLICY = {
    "policy_version": "symile-m5-development-analysis-v1",
    "alignment": ["sample_id", "repeat_seed"],
    "metrics": ["roc_auc", "average_precision", "brier_score"],
    "paired_comparisons": {
        "concat_minus_cxr": ["concat", "cxr"],
        "gated_minus_cxr": ["gated", "cxr"],
        "gated_minus_concat": ["gated", "concat"],
    },
    "neural_ensemble": "align three repeat logits; arithmetic mean logits; sigmoid once",
    "uncertainty": "none; folds are not treated as independent replicates",
    "official_test_access": "closed",
}


@dataclass(frozen=True)
class ValidatedFoldPackage:
    """Validated fold package metadata and private OOF content."""

    directory: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    oof: pa.Table


@dataclass(frozen=True)
class ValidatedDevelopmentResult:
    """Validated aggregate family development authority."""

    directory: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str


def build_oof_table(
    sample_ids: Sequence[str],
    targets: Sequence[int] | np.ndarray,
    logits: Sequence[float] | np.ndarray,
) -> pa.Table:
    """Build the canonical minimal private OOF table."""
    ids = [str(value) for value in sample_ids]
    truth = np.asarray(targets)
    scores = np.asarray(logits, dtype=np.float64)
    if (
        not ids
        or len(ids) != len(set(ids))
        or truth.shape != (len(ids),)
        or scores.shape != (len(ids),)
        or not set(truth.tolist()) <= {0, 1}
        or not np.isfinite(scores).all()
    ):
        raise ManifestBuildError("Symile OOF predictions are invalid")
    probabilities = _sigmoid(scores)
    records = sorted(
        zip(ids, truth.astype(np.int8), scores, probabilities, strict=True),
        key=lambda row: row[0],
    )
    return pa.Table.from_pylist(
        [
            {
                "sample_id": sample_id,
                "target": int(target),
                "logit": float(logit),
                "probability": float(probability),
            }
            for sample_id, target, logit, probability in records
        ],
        schema=OOF_SCHEMA,
    )


def neural_checkpoint_document(
    state_dict: Mapping[str, torch.Tensor],
    *,
    selected_epoch: int,
    selected_stage: str,
    selected_validation_roc_auc: float,
) -> dict[str, object]:
    """Create the safe M5 neural checkpoint document."""
    document: dict[str, object] = {
        "checkpoint_schema_version": 1,
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in state_dict.items()
        },
        "selected_epoch": selected_epoch,
        "selected_stage": selected_stage,
        "selection_metric": "roc_auc",
        "selected_validation_metric": selected_validation_roc_auc,
    }
    _validate_neural_checkpoint(document)
    return document


def load_symile_neural_checkpoint(path: str | Path) -> dict[str, object]:
    """Safely load one exact M5 neural checkpoint."""
    try:
        document = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        raise ManifestBuildError("Symile neural checkpoint is unreadable") from exc
    _validate_neural_checkpoint(document)
    return document


def publish_fold_package(
    *,
    model_root: str | Path,
    family: str,
    repeat_seed: int,
    outer_fold: int,
    config_bytes: bytes,
    config_sha256: str,
    semantic_config_sha256: str,
    lineage: Mapping[str, object],
    inner_split: Mapping[str, object],
    selection: Mapping[str, object],
    oof: pa.Table,
    model: object,
    lab_preprocessor: object | None = None,
    training_history: Sequence[Mapping[str, object]] | None = None,
    source_cxr: Mapping[str, object] | None = None,
    operational: Mapping[str, object] | None = None,
) -> ValidatedFoldPackage:
    """Stage, validate, and immutably publish one M5 outer-fold package."""
    _validate_family(family)
    _validate_oof(oof)
    _validate_fold_contents(family, lab_preprocessor, training_history, source_cxr)
    folds_root = Path(model_root) / "folds"
    provisional = folds_root / "fold-package-pending"
    stage = staging_directory(provisional)
    try:
        (stage / CONFIG_FILENAME).write_bytes(config_bytes)
        pq.write_table(oof, stage / OOF_FILENAME, compression="zstd")
        if family in TABULAR_FAMILIES:
            save_skops(model, stage / TABULAR_MODEL_FILENAME)
        else:
            _validate_neural_checkpoint(model)
            torch.save(dict(model), stage / NEURAL_MODEL_FILENAME)
            load_symile_neural_checkpoint(stage / NEURAL_MODEL_FILENAME)
            (stage / TRAINING_HISTORY_FILENAME).write_text(
                json.dumps(list(training_history or ()), indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
        if family in FUSION_FAMILIES:
            if not isinstance(lab_preprocessor, SymileLabEcdfTransformer):
                raise ManifestBuildError("Fusion fold lab preprocessor has the wrong type")
            save_symile_lab_preprocessor(lab_preprocessor, stage / LAB_PREPROCESSOR_FILENAME)
        artifacts = _artifact_declarations(stage, oof)
        identity_payload = _fold_identity_payload(
            family=family,
            repeat_seed=repeat_seed,
            outer_fold=outer_fold,
            semantic_config_sha256=semantic_config_sha256,
            lineage=lineage,
            inner_split=inner_split,
            selection=selection,
            source_cxr=source_cxr,
            artifacts=artifacts,
        )
        fold_id = FOLD_PREFIX + _canonical_sha256(identity_payload)
        document = {
            "fold_schema_version": FOLD_SCHEMA_VERSION,
            "fold_package_id": fold_id,
            "family": family,
            "repeat_seed": repeat_seed,
            "outer_fold": outer_fold,
            "config": {
                "source_sha256": config_sha256,
                "semantic_sha256": semantic_config_sha256,
            },
            "lineage": dict(lineage),
            "inner_split": dict(inner_split),
            "selection": dict(selection),
            "source_cxr": dict(source_cxr) if source_cxr is not None else None,
            "artifacts": artifacts,
            "operational": dict(operational or {}),
        }
        (stage / FOLD_MANIFEST_FILENAME).write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        destination = folds_root / fold_id
        _publish_immutable(stage, destination, validate_fold_package)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return validate_fold_package(destination, expected_fold_package_id=fold_id)


def validate_fold_package(
    directory: str | Path,
    *,
    expected_fold_package_id: str | None = None,
    enforce_directory_name: bool = True,
) -> ValidatedFoldPackage:
    """Validate exact files, identities, safe model state, and OOF content."""
    root = Path(directory)
    _require_physical_directory(root)
    manifest_path = root / FOLD_MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ManifestBuildError("Symile fold manifest is missing")
    manifest_bytes = manifest_path.read_bytes()
    try:
        document = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Symile fold manifest is unreadable") from exc
    _validate_fold_manifest(document)
    fold_id = document["fold_package_id"]
    if expected_fold_package_id is not None and fold_id != expected_fold_package_id:
        raise ManifestBuildError("Symile fold package differs from the expected identity")
    if enforce_directory_name and root.name != fold_id:
        raise ManifestBuildError("Symile fold directory differs from its identity")
    expected_files = _fold_files(document["family"])
    _require_exact_regular_files(root, expected_files)
    artifacts = document["artifacts"]
    if set(artifacts) != expected_files - {FOLD_MANIFEST_FILENAME}:
        raise ManifestBuildError("Symile fold artifact declarations are invalid")
    for filename, declaration in artifacts.items():
        expected_declaration_fields = {"physical_sha256"}
        if filename == OOF_FILENAME:
            expected_declaration_fields |= {"logical_arrow_sha256", "row_count"}
        if (
            not isinstance(declaration, dict)
            or set(declaration) != expected_declaration_fields
            or not _sha256(declaration.get("physical_sha256"))
        ):
            raise ManifestBuildError(f"Symile fold artifact declaration is invalid: {filename}")
        if sha256_file(root / filename) != declaration.get("physical_sha256"):
            raise ManifestBuildError(f"Symile fold physical hash mismatch: {filename}")
    if sha256_file(root / CONFIG_FILENAME) != document["config"]["source_sha256"]:
        raise ManifestBuildError("Symile fold resolved config hash does not match")
    try:
        config = load_symile_development_config(root / CONFIG_FILENAME)
    except (OSError, ValueError, TypeError) as exc:
        raise ManifestBuildError("Symile fold resolved configuration is invalid") from exc
    _validate_fold_config(config, document)
    oof = pq.read_table(root / OOF_FILENAME)
    _validate_oof(oof)
    if (
        artifacts[OOF_FILENAME].get("logical_arrow_sha256") != arrow_ipc_sha256(oof)
        or artifacts[OOF_FILENAME].get("row_count") != oof.num_rows
    ):
        raise ManifestBuildError("Symile fold OOF identity is invalid")
    if document["family"] in TABULAR_FAMILIES:
        _validate_tabular_reconstruction(root, config, document)
    else:
        checkpoint = load_symile_neural_checkpoint(root / NEURAL_MODEL_FILENAME)
        if (
            checkpoint["selected_epoch"] != document["selection"].get("selected_epoch")
            or checkpoint["selected_stage"] != document["selection"].get("selected_stage")
            or checkpoint["selected_validation_metric"]
            != document["selection"].get("selected_validation_metric")
        ):
            raise ManifestBuildError("Symile fold selected epoch does not match checkpoint")
        _validate_neural_reconstruction(config, checkpoint)
        history = _load_json_list(root / TRAINING_HISTORY_FILENAME)
        _validate_training_history(history, document["selection"], config)
    if document["family"] in FUSION_FAMILIES:
        try:
            load_symile_lab_preprocessor(root / LAB_PREPROCESSOR_FILENAME)
        except (OSError, ValueError, TypeError) as exc:
            raise ManifestBuildError("Symile fold lab preprocessor is invalid") from exc
    expected_id = FOLD_PREFIX + _canonical_sha256(_fold_payload_from_manifest(document))
    if expected_id != fold_id:
        raise ManifestBuildError("Symile fold semantic identity is invalid")
    return ValidatedFoldPackage(
        root,
        document,
        hashlib.sha256(manifest_bytes).hexdigest(),
        oof,
    )


def publish_development_result(
    *,
    report_root: str | Path,
    model_root: str | Path,
    family: str,
    semantic_config_sha256: str,
    folds: Sequence[ValidatedFoldPackage],
    repeat_metrics: Mapping[str, Mapping[str, float]],
    selected_values: Sequence[int] | None,
    median_m6_budget: int | None,
) -> ValidatedDevelopmentResult:
    """Publish the complete 15-fold family development authority."""
    _validate_family(family)
    ordered = sorted(
        folds, key=lambda item: (item.manifest["repeat_seed"], item.manifest["outer_fold"])
    )
    coordinates = [(item.manifest["repeat_seed"], item.manifest["outer_fold"]) for item in ordered]
    if coordinates != [(seed, fold) for seed in (17, 42, 2026) for fold in range(5)]:
        raise ManifestBuildError("Symile family development does not contain exact 3 x 5 folds")
    if any(
        item.manifest["family"] != family
        or item.manifest["config"]["semantic_sha256"] != semantic_config_sha256
        for item in ordered
    ):
        raise ManifestBuildError("Symile family fold compatibility is invalid")
    _validate_repeat_metrics(repeat_metrics)
    derived_metrics = _fold_repeat_metrics(ordered)
    if not _nested_metrics_close(repeat_metrics, derived_metrics):
        raise ManifestBuildError("Symile family repeat metrics differ from fold OOF content")
    expected_selected = None
    if family != "labs_logistic":
        field = "best_iteration" if family == "labs_lightgbm" else "selected_epoch"
        expected_selected = [int(item.manifest["selection"][field]) for item in ordered]
    if (list(selected_values) if selected_values is not None else None) != expected_selected:
        raise ManifestBuildError("Symile family selected-budget evidence differs from folds")
    _validate_budget(family, selected_values, median_m6_budget)
    fold_refs = [
        {
            "fold_package_id": item.manifest["fold_package_id"],
            "fold_manifest_sha256": item.manifest_sha256,
            "repeat_seed": item.manifest["repeat_seed"],
            "outer_fold": item.manifest["outer_fold"],
        }
        for item in ordered
    ]
    identity_payload = {
        "development_schema_version": DEVELOPMENT_SCHEMA_VERSION,
        "family": family,
        "semantic_config_sha256": semantic_config_sha256,
        "fold_packages": fold_refs,
        "repeat_metrics": repeat_metrics,
        "selected_values": list(selected_values) if selected_values is not None else None,
        "median_m6_budget": median_m6_budget,
    }
    development_id = DEVELOPMENT_PREFIX + _canonical_sha256(identity_payload)
    document = {"development_id": development_id, **identity_payload, "completeness": "passed"}
    destination = Path(report_root) / "families" / development_id
    stage = staging_directory(destination)
    try:
        (stage / DEVELOPMENT_MANIFEST_FILENAME).write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (stage / SUMMARY_FILENAME).write_text(_development_summary(document), encoding="utf-8")
        validate_public_reports(
            [stage / SUMMARY_FILENAME],
            forbidden_source_values={
                str(sample_id)
                for item in ordered
                for sample_id in item.oof["sample_id"].to_pylist()
            },
        )
        _publish_immutable(
            stage,
            destination,
            lambda path, **kwargs: validate_development_result(
                path,
                model_root=model_root,
                **kwargs,
            ),
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return validate_development_result(
        destination,
        model_root=model_root,
        expected_development_id=development_id,
    )


def validate_development_result(
    directory: str | Path,
    *,
    model_root: str | Path = "models/symile/development",
    expected_development_id: str | None = None,
    enforce_directory_name: bool = True,
) -> ValidatedDevelopmentResult:
    """Validate one aggregate family development authority."""
    root = Path(directory)
    _require_exact_regular_files(root, {DEVELOPMENT_MANIFEST_FILENAME, SUMMARY_FILENAME})
    manifest_bytes = (root / DEVELOPMENT_MANIFEST_FILENAME).read_bytes()
    try:
        document = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Symile development manifest is unreadable") from exc
    expected_fields = {
        "development_schema_version",
        "development_id",
        "family",
        "semantic_config_sha256",
        "fold_packages",
        "repeat_metrics",
        "selected_values",
        "median_m6_budget",
        "completeness",
    }
    if not isinstance(document, dict) or set(document) != expected_fields:
        raise ManifestBuildError("Symile development manifest fields are invalid")
    _validate_family(document["family"])
    if document["development_schema_version"] != DEVELOPMENT_SCHEMA_VERSION:
        raise ManifestBuildError("Symile development schema version is invalid")
    if document["completeness"] != "passed" or len(document["fold_packages"]) != 15:
        raise ManifestBuildError("Symile development completeness declaration is invalid")
    coordinates = [
        (item.get("repeat_seed"), item.get("outer_fold")) for item in document["fold_packages"]
    ]
    if coordinates != [(seed, fold) for seed in (17, 42, 2026) for fold in range(5)]:
        raise ManifestBuildError("Symile development fold references are invalid")
    if any(
        set(item) != {"fold_package_id", "fold_manifest_sha256", "repeat_seed", "outer_fold"}
        or not _identity(item["fold_package_id"], FOLD_PREFIX)
        or not _sha256(item["fold_manifest_sha256"])
        for item in document["fold_packages"]
    ):
        raise ManifestBuildError("Symile development fold lineage is invalid")
    _validate_repeat_metrics(document["repeat_metrics"])
    folds = _resolve_family_folds(document, model_root)
    derived_metrics = _fold_repeat_metrics(folds)
    if not _nested_metrics_close(document["repeat_metrics"], derived_metrics):
        raise ManifestBuildError("Symile development metrics differ from fold evidence")
    selected = None
    if document["family"] != "labs_logistic":
        field = "best_iteration" if document["family"] == "labs_lightgbm" else "selected_epoch"
        selected = [int(item.manifest["selection"][field]) for item in folds]
    if document["selected_values"] != selected:
        raise ManifestBuildError("Symile development budget evidence differs from fold packages")
    _validate_budget(document["family"], selected, document["median_m6_budget"])
    identity_payload = {
        key: document[key] for key in document if key not in {"development_id", "completeness"}
    }
    development_id = DEVELOPMENT_PREFIX + _canonical_sha256(identity_payload)
    if document["development_id"] != development_id:
        raise ManifestBuildError("Symile development identity is invalid")
    if expected_development_id is not None and development_id != expected_development_id:
        raise ManifestBuildError("Symile development differs from the expected identity")
    if enforce_directory_name and root.name != development_id:
        raise ManifestBuildError("Symile development directory differs from its identity")
    expected_summary = _development_summary(document).encode("utf-8")
    if (root / SUMMARY_FILENAME).read_bytes() != expected_summary:
        raise ManifestBuildError("Symile development summary differs from its manifest")
    validate_public_reports(
        [root / SUMMARY_FILENAME],
        forbidden_source_values={
            str(sample_id) for item in folds for sample_id in item.oof["sample_id"].to_pylist()
        },
    )
    return ValidatedDevelopmentResult(root, document, hashlib.sha256(manifest_bytes).hexdigest())


def publish_analysis_result(
    *,
    report_root: str | Path,
    model_root: str | Path,
    family_development_ids: Mapping[str, str],
    repeat_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    paired_effects: Mapping[str, Mapping[str, Mapping[str, float]]],
    ensemble_metrics: Mapping[str, Mapping[str, float]],
    observedness_ablation: Mapping[str, Mapping[str, float]],
    policy: Mapping[str, object],
) -> tuple[str, Path]:
    """Publish one aggregate, privacy-safe cross-family M5 analysis."""
    if set(family_development_ids) != set(SYMILE_M5_FAMILIES):
        raise ManifestBuildError("Symile analysis requires exactly six family authorities")
    payload = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "family_development_ids": dict(sorted(family_development_ids.items())),
        "repeat_metrics": repeat_metrics,
        "paired_effects": paired_effects,
        "ensemble_metrics": ensemble_metrics,
        "observedness_ablation": observedness_ablation,
        "policy": dict(policy),
    }
    analysis_id = ANALYSIS_PREFIX + _canonical_sha256(payload)
    document = {"analysis_id": analysis_id, **payload}
    destination = Path(report_root) / "analyses" / analysis_id
    stage = staging_directory(destination)
    try:
        (stage / ANALYSIS_MANIFEST_FILENAME).write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (stage / SUMMARY_FILENAME).write_text(_analysis_summary(document), encoding="utf-8")
        validate_public_reports([stage / SUMMARY_FILENAME], forbidden_source_values=())
        _publish_immutable(
            stage,
            destination,
            lambda path, **kwargs: validate_analysis_result(
                path,
                report_root=report_root,
                model_root=model_root,
                **kwargs,
            ),
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return analysis_id, destination


def validate_analysis_result(
    directory: str | Path,
    *,
    report_root: str | Path = "reports/symile/development",
    model_root: str | Path = "models/symile/development",
    expected_analysis_id: str | None = None,
    enforce_directory_name: bool = True,
) -> dict[str, Any]:
    """Validate one aggregate cross-family M5 analysis."""
    root = Path(directory)
    _require_exact_regular_files(root, {ANALYSIS_MANIFEST_FILENAME, SUMMARY_FILENAME})
    try:
        document = json.loads((root / ANALYSIS_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestBuildError("Symile analysis manifest is unreadable") from exc
    expected = {
        "analysis_schema_version",
        "analysis_id",
        "family_development_ids",
        "repeat_metrics",
        "paired_effects",
        "ensemble_metrics",
        "observedness_ablation",
        "policy",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise ManifestBuildError("Symile analysis manifest fields are invalid")
    if (
        document["analysis_schema_version"] != ANALYSIS_SCHEMA_VERSION
        or set(document["family_development_ids"]) != set(SYMILE_M5_FAMILIES)
        or any(
            not _identity(value, DEVELOPMENT_PREFIX)
            for value in document["family_development_ids"].values()
        )
    ):
        raise ManifestBuildError("Symile analysis lineage is invalid")
    if document["policy"] != _ANALYSIS_POLICY:
        raise ManifestBuildError("Symile analysis policy is invalid")
    payload = {key: document[key] for key in document if key != "analysis_id"}
    analysis_id = ANALYSIS_PREFIX + _canonical_sha256(payload)
    if document["analysis_id"] != analysis_id:
        raise ManifestBuildError("Symile analysis identity is invalid")
    if expected_analysis_id is not None and analysis_id != expected_analysis_id:
        raise ManifestBuildError("Symile analysis differs from the expected identity")
    if enforce_directory_name and root.name != analysis_id:
        raise ManifestBuildError("Symile analysis directory differs from its identity")
    derived = _derive_analysis(document["family_development_ids"], report_root, model_root)
    for field in (
        "repeat_metrics",
        "paired_effects",
        "ensemble_metrics",
        "observedness_ablation",
    ):
        if not _nested_numeric_close(document[field], derived[field]):
            raise ManifestBuildError(f"Symile analysis {field} differs from family evidence")
    if (root / SUMMARY_FILENAME).read_bytes() != _analysis_summary(document).encode("utf-8"):
        raise ManifestBuildError("Symile analysis summary differs from its manifest")
    validate_public_reports([root / SUMMARY_FILENAME], forbidden_source_values=())
    return document


def _fold_identity_payload(
    *,
    family: str,
    repeat_seed: int,
    outer_fold: int,
    semantic_config_sha256: str,
    lineage: Mapping[str, object],
    inner_split: Mapping[str, object],
    selection: Mapping[str, object],
    source_cxr: Mapping[str, object] | None,
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "fold_schema_version": FOLD_SCHEMA_VERSION,
        "family": family,
        "repeat_seed": repeat_seed,
        "outer_fold": outer_fold,
        "semantic_config_sha256": semantic_config_sha256,
        "lineage": dict(lineage),
        "inner_split": dict(inner_split),
        "selection": dict(selection),
        "source_cxr": dict(source_cxr) if source_cxr is not None else None,
        "artifacts": artifacts,
    }


def _fold_payload_from_manifest(document: Mapping[str, Any]) -> dict[str, object]:
    return _fold_identity_payload(
        family=document["family"],
        repeat_seed=document["repeat_seed"],
        outer_fold=document["outer_fold"],
        semantic_config_sha256=document["config"]["semantic_sha256"],
        lineage=document["lineage"],
        inner_split=document["inner_split"],
        selection=document["selection"],
        source_cxr=document["source_cxr"],
        artifacts=document["artifacts"],
    )


def _artifact_declarations(stage: Path, oof: pa.Table) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in sorted(stage.iterdir()):
        declaration: dict[str, object] = {"physical_sha256": sha256_file(path)}
        if path.name == OOF_FILENAME:
            declaration.update(logical_arrow_sha256=arrow_ipc_sha256(oof), row_count=oof.num_rows)
        result[path.name] = declaration
    return result


def _validate_fold_manifest(document: object) -> None:
    if not isinstance(document, dict) or set(document) != _FOLD_FIELDS:
        raise ManifestBuildError("Symile fold manifest fields are invalid")
    _validate_family(document.get("family"))
    if document.get("fold_schema_version") != FOLD_SCHEMA_VERSION:
        raise ManifestBuildError("Symile fold schema version is invalid")
    if not _identity(document.get("fold_package_id"), FOLD_PREFIX):
        raise ManifestBuildError("Symile fold identity declaration is invalid")
    if document.get("repeat_seed") not in {17, 42, 2026} or document.get("outer_fold") not in range(
        5
    ):
        raise ManifestBuildError("Symile fold coordinates are invalid")
    config = document.get("config")
    if (
        not isinstance(config, dict)
        or set(config) != {"source_sha256", "semantic_sha256"}
        or not all(_sha256(value) for value in config.values())
    ):
        raise ManifestBuildError("Symile fold config identity is invalid")
    if not isinstance(document.get("lineage"), dict) or not isinstance(
        document.get("inner_split"), dict
    ):
        raise ManifestBuildError("Symile fold provenance is invalid")
    lineage = document["lineage"]
    if (
        set(lineage) != _LINEAGE_FIELDS
        or not _identity(lineage.get("bundle_id"), "build-")
        or not _sha256(lineage.get("bundle_manifest_sha256"))
        or not _identity(lineage.get("official_split_assignment_id"), "split-assignment-")
        or not _identity(lineage.get("cv_assignment_id"), "cv-assignment-")
        or not _sha256(lineage.get("cv_manifest_sha256"))
        or lineage.get("task_id") != "pneumonia_strict"
        or not isinstance(lineage.get("git_commit"), str)
        or not lineage["git_commit"]
        or not _sha256(lineage.get("dependency_lock_sha256"))
    ):
        raise ManifestBuildError("Symile fold data/code lineage is invalid")
    if document["family"] in NEURAL_FAMILIES:
        encoder = lineage.get("encoder_identity")
        if (
            not isinstance(encoder, dict)
            or set(encoder) != {"library", "architecture", "weights", "embedding_dimension"}
            or encoder.get("library") != "torchxrayvision"
            or not isinstance(encoder.get("architecture"), str)
            or not isinstance(encoder.get("weights"), str)
            or isinstance(encoder.get("embedding_dimension"), bool)
            or not isinstance(encoder.get("embedding_dimension"), int)
            or not isinstance(lineage.get("transform_contract"), dict)
            or lineage["transform_contract"].get("policy_version")
            != ("torchxrayvision-densenet121-res224-v1")
        ):
            raise ManifestBuildError("Symile neural fold encoder/transform lineage is invalid")
        if document["family"] == "cxr":
            weight = lineage.get("pretrained_weight")
            if (
                not isinstance(weight, dict)
                or set(weight)
                != {
                    "declared_name",
                    "stable_identifier",
                    "cache_filename",
                    "byte_size",
                    "sha256",
                }
                or not isinstance(weight.get("declared_name"), str)
                or not isinstance(weight.get("stable_identifier"), str)
                or not weight["stable_identifier"]
                or not isinstance(weight.get("cache_filename"), str)
                or not weight["cache_filename"]
                or isinstance(weight.get("byte_size"), bool)
                or not isinstance(weight.get("byte_size"), int)
                or weight["byte_size"] <= 0
                or not _sha256(weight.get("sha256"))
            ):
                raise ManifestBuildError("Symile CXR fold lacks pretrained-weight lineage")
        if document["family"] in FUSION_FAMILIES and lineage.get("pretrained_weight") is not None:
            raise ManifestBuildError("Symile fusion fold duplicates pretrained-weight lineage")
    elif any(
        lineage.get(field) is not None
        for field in ("encoder_identity", "transform_contract", "pretrained_weight")
    ):
        raise ManifestBuildError("Symile tabular fold declares neural lineage")
    inner = document["inner_split"]
    if (
        set(inner) != {"inner_split_id", "inner_seed", "policy"}
        or not _identity(inner.get("inner_split_id"), "inner-split-")
        or isinstance(inner.get("inner_seed"), bool)
        or not isinstance(inner.get("inner_seed"), int)
        or not isinstance(inner.get("policy"), dict)
    ):
        raise ManifestBuildError("Symile inner split identity is invalid")
    policy = inner["policy"]
    if (
        set(policy)
        != {
            "policy_version",
            "algorithm",
            "n_splits",
            "shuffle",
            "generated_validation_fold",
            "group_field",
            "stratification_target",
            "repeat_seed",
            "outer_fold",
            "inner_seed",
        }
        or policy.get("policy_version") != "symile-inner-stratified-group-five-fold-v1"
        or policy.get("algorithm") != "sklearn.model_selection.StratifiedGroupKFold"
        or policy.get("n_splits") != 5
        or policy.get("shuffle") is not True
        or policy.get("generated_validation_fold") != 0
        or policy.get("group_field") != "subject_id"
        or policy.get("stratification_target") != "pneumonia_strict"
        or policy.get("repeat_seed") != document["repeat_seed"]
        or policy.get("outer_fold") != document["outer_fold"]
        or policy.get("inner_seed") != inner["inner_seed"]
    ):
        raise ManifestBuildError("Symile inner split policy is invalid")
    if not isinstance(document.get("selection"), dict) or not isinstance(
        document.get("artifacts"), dict
    ):
        raise ManifestBuildError("Symile fold selection or artifacts are invalid")
    selection = document["selection"]
    expected_selection = {"metric", "selected_epoch", "best_iteration"}
    if document["family"] in NEURAL_FAMILIES:
        expected_selection |= {"selected_stage", "selected_validation_metric"}
    if set(selection) != expected_selection:
        raise ManifestBuildError("Symile fold selection fields are invalid")
    if document["family"] == "labs_logistic" and selection != {
        "metric": "none",
        "selected_epoch": None,
        "best_iteration": None,
    }:
        raise ManifestBuildError("Symile Logistic Regression selection is invalid")
    if document["family"] == "labs_lightgbm" and (
        selection.get("metric") != "roc_auc"
        or selection.get("selected_epoch") is not None
        or isinstance(selection.get("best_iteration"), bool)
        or not isinstance(selection.get("best_iteration"), int)
        or selection["best_iteration"] <= 0
    ):
        raise ManifestBuildError("Symile LightGBM selection is invalid")
    if document["family"] in NEURAL_FAMILIES and (
        selection.get("metric") != "roc_auc"
        or isinstance(selection.get("selected_epoch"), bool)
        or not isinstance(selection.get("selected_epoch"), int)
        or not 1 <= selection["selected_epoch"] <= 30
        or selection.get("best_iteration") is not None
        or selection.get("selected_stage") not in {"warmup", "fine_tune"}
        or not _finite(selection.get("selected_validation_metric"))
    ):
        raise ManifestBuildError("Symile neural selection is invalid")
    if document["family"] in FUSION_FAMILIES:
        source = document.get("source_cxr")
        if (
            not isinstance(source, dict)
            or set(source) != {"development_id", "fold_package_id", "fold_manifest_sha256"}
            or not _identity(source.get("development_id"), DEVELOPMENT_PREFIX)
            or not _identity(source.get("fold_package_id"), FOLD_PREFIX)
            or not _sha256(source.get("fold_manifest_sha256"))
        ):
            raise ManifestBuildError("Symile fusion fold lacks source CXR lineage")
    elif document.get("source_cxr") is not None:
        raise ManifestBuildError("Non-fusion Symile fold declares source CXR lineage")
    operational = document.get("operational")
    if (
        not isinstance(operational, dict)
        or set(operational) != {"mlflow_run_id", "runtime_provenance"}
        or not isinstance(operational.get("mlflow_run_id"), str)
        or not operational["mlflow_run_id"]
        or (
            document["family"] in NEURAL_FAMILIES
            and not isinstance(operational.get("runtime_provenance"), dict)
        )
        or (
            document["family"] in TABULAR_FAMILIES
            and operational.get("runtime_provenance") is not None
        )
    ):
        raise ManifestBuildError("Symile fold operational provenance is invalid")


def _validate_oof(table: pa.Table) -> None:
    if table.schema != OOF_SCHEMA or table.num_rows == 0:
        raise ManifestBuildError("Symile OOF schema or row count is invalid")
    frame = table.to_pandas()
    if (
        frame["sample_id"].duplicated().any()
        or frame["sample_id"].tolist() != sorted(frame["sample_id"])
        or set(frame["target"]) != {0, 1}
        or not np.isfinite(frame[["logit", "probability"]].to_numpy()).all()
        or not np.allclose(
            frame["probability"].to_numpy(dtype=np.float64),
            _sigmoid(frame["logit"].to_numpy(dtype=np.float64)),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ManifestBuildError("Symile OOF prediction content is invalid")


def _validate_neural_checkpoint(document: object) -> None:
    fields = {
        "checkpoint_schema_version",
        "model_state_dict",
        "selected_epoch",
        "selected_stage",
        "selection_metric",
        "selected_validation_metric",
    }
    if not isinstance(document, dict) or set(document) != fields:
        raise ManifestBuildError("Symile neural checkpoint fields are invalid")
    state = document["model_state_dict"]
    if (
        document["checkpoint_schema_version"] != 1
        or not isinstance(state, dict)
        or not state
        or any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in state.items()
        )
        or any(not torch.isfinite(value).all() for value in state.values())
        or isinstance(document["selected_epoch"], bool)
        or not isinstance(document["selected_epoch"], int)
        or not 1 <= document["selected_epoch"] <= 30
        or document["selected_stage"] not in {"warmup", "fine_tune"}
        or document["selection_metric"] != "roc_auc"
        or not _finite(document["selected_validation_metric"])
    ):
        raise ManifestBuildError("Symile neural checkpoint is invalid")


def _validate_fold_config(config: SymileDevelopmentConfig, document: Mapping[str, Any]) -> None:
    lineage = document["lineage"]
    inner_policy = document["inner_split"]["policy"]
    encoder_identity = (
        {
            "library": "torchxrayvision",
            "architecture": config.model["encoder_name"],
            "weights": config.model["weights"],
            "embedding_dimension": config.model["embedding_dimension"],
        }
        if config.family in NEURAL_FAMILIES
        else None
    )
    transform_contract = (
        _evaluation_transform_contract(config) if config.family in NEURAL_FAMILIES else None
    )
    pretrained = lineage["pretrained_weight"]
    if (
        config.family != document["family"]
        or config.source_sha256 != document["config"]["source_sha256"]
        or symile_development_semantic_sha256(config) != document["config"]["semantic_sha256"]
        or config.dataset.bundle_id != lineage["bundle_id"]
        or config.dataset.bundle_manifest_sha256 != lineage["bundle_manifest_sha256"]
        or config.dataset.official_split_assignment_id != lineage["official_split_assignment_id"]
        or config.dataset.cv_assignment_id != lineage["cv_assignment_id"]
        or config.dataset.task_id != lineage["task_id"]
        or config.training["selection_metric"] != document["selection"]["metric"]
        or lineage["encoder_identity"] != encoder_identity
        or lineage["transform_contract"] != transform_contract
        or (config.family == "cxr" and pretrained["declared_name"] != config.model["weights"])
        or inner_policy.get("repeat_seed") != document["repeat_seed"]
        or inner_policy.get("outer_fold") != document["outer_fold"]
        or inner_policy.get("inner_seed") != document["inner_split"]["inner_seed"]
    ):
        raise ManifestBuildError("Symile fold configuration and manifest disagree")


def _evaluation_transform_contract(config: SymileDevelopmentConfig) -> dict[str, object]:
    if config.image is None:
        raise ManifestBuildError("Symile neural config lacks image settings")
    return StandardCxrTransform(
        training=False,
        image_size=int(config.model["image_size"]),
        rotation_degrees=config.image.rotation_degrees,
        translation_fraction=config.image.translation_fraction,
        brightness_jitter=config.image.brightness_jitter,
        contrast_jitter=config.image.contrast_jitter,
    ).contract()


def _validate_tabular_reconstruction(
    root: Path,
    config: SymileDevelopmentConfig,
    document: Mapping[str, Any],
) -> None:
    try:
        model = load_skops(root / TABULAR_MODEL_FILENAME)
    except (OSError, ValueError, TypeError) as exc:
        raise ManifestBuildError("Symile tabular model is unreadable") from exc
    if not isinstance(model, Pipeline) or tuple(model.named_steps) != ("preprocess", "classifier"):
        raise ManifestBuildError("Symile tabular fold is not the exact fitted pipeline")
    try:
        preprocessor = validate_symile_lab_preprocessor(model.named_steps["preprocess"])
    except (ValueError, TypeError) as exc:
        raise ManifestBuildError("Symile tabular preprocessing contract is invalid") from exc
    classifier = model.named_steps["classifier"]
    parameters = config.model
    if config.family == "labs_logistic":
        if not isinstance(classifier, LogisticRegression):
            raise ManifestBuildError("Symile Logistic Regression fold has the wrong estimator")
        actual = classifier.get_params(deep=False)
        expected = {
            "l1_ratio": parameters["l1_ratio"],
            "solver": parameters["solver"],
            "C": parameters["C"],
            "max_iter": parameters["max_iter"],
            "class_weight": parameters["class_weight"],
            "random_state": document["repeat_seed"],
        }
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ManifestBuildError("Symile Logistic Regression configuration differs")
    else:
        if not isinstance(classifier, LGBMClassifier):
            raise ManifestBuildError("Symile LightGBM fold has the wrong estimator")
        actual = classifier.get_params(deep=False)
        expected = {
            key: parameters[key]
            for key in (
                "objective",
                "n_estimators",
                "learning_rate",
                "num_leaves",
                "min_child_samples",
                "subsample",
                "subsample_freq",
                "colsample_bytree",
                "reg_lambda",
                "class_weight",
            )
        }
        expected.update(
            random_state=document["repeat_seed"],
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
        )
        best_iteration = classifier.best_iteration_
        if (
            any(actual.get(key) != value for key, value in expected.items())
            or actual.get("scale_pos_weight", 1.0) != 1.0
            or actual.get("is_unbalance", False) is not False
            or best_iteration != document["selection"]["best_iteration"]
        ):
            raise ManifestBuildError("Symile LightGBM configuration or selection differs")
    if getattr(classifier, "n_features_in_", None) != 100:
        raise ManifestBuildError("Symile tabular estimator feature contract is invalid")
    probe = _preprocessor_probe(preprocessor)
    try:
        logits = symile_tabular_logits(model, probe)
    except (TypeError, ValueError) as exc:
        raise ManifestBuildError("Symile tabular fold cannot perform prediction") from exc
    if logits.shape != (1,) or not np.isfinite(logits).all():
        raise ManifestBuildError("Symile tabular fold prediction proof failed")


def _preprocessor_probe(transformer: SymileLabEcdfTransformer) -> pd.DataFrame:
    record: dict[str, object] = {}
    for column, values in zip(
        LAB_FEATURE_COLUMNS[:50], transformer.sorted_observed_values_, strict=True
    ):
        record[column] = float(values[0])
    for column in LAB_FEATURE_COLUMNS[50:]:
        record[column] = True
    return pd.DataFrame([record], columns=LAB_FEATURE_COLUMNS)


def _validate_neural_reconstruction(
    config: SymileDevelopmentConfig, checkpoint: Mapping[str, Any]
) -> None:
    try:
        parameters = config.model
        if config.family == "cxr":
            model: torch.nn.Module = CxrBinaryClassifier(
                StandardCxrEncoder(
                    weights=None,
                    expected_embedding_dimension=int(parameters["embedding_dimension"]),
                    image_size=int(parameters["image_size"]),
                ),
                embedding_dimension=int(parameters["embedding_dimension"]),
                image_size=int(parameters["image_size"]),
            )
        elif config.family == "concat":
            model = build_symile_concat_model(parameters, weights=None)
        else:
            model = build_symile_gated_model(parameters, weights=None)
        state = checkpoint["model_state_dict"]
        if not isinstance(state, Mapping):
            raise ManifestBuildError("Symile neural checkpoint state is invalid")
        incompatible = model.load_state_dict(state, strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ManifestBuildError("Symile neural fold cannot reconstruct exactly") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ManifestBuildError("Symile neural fold state is structurally incomplete")


def _validate_training_history(
    history: Sequence[object],
    selection: Mapping[str, Any],
    config: SymileDevelopmentConfig,
) -> None:
    fields = {
        "epoch",
        "stage",
        "training_loss",
        "validation_metric",
        "encoder_learning_rate",
        "head_learning_rate",
    }
    epochs: list[int] = []
    stages: dict[int, str] = {}
    metrics: dict[int, float] = {}
    if config.image is None:
        raise ManifestBuildError("Symile neural history lacks image configuration")
    if len(history) > config.image.warmup_epochs + config.image.fine_tune_epochs:
        raise ManifestBuildError("Symile training history exceeds the configured lifecycle")
    for ordinal, entry in enumerate(history, start=1):
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ManifestBuildError("Symile training history fields are invalid")
        epoch = entry["epoch"]
        expected_stage = "warmup" if ordinal <= config.image.warmup_epochs else "fine_tune"
        encoder_rate = entry["encoder_learning_rate"]
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch != ordinal
            or entry["stage"] != expected_stage
            or not _finite(entry["training_loss"])
            or float(entry["training_loss"]) < 0.0
            or not _finite(entry["validation_metric"])
            or not 0.0 <= float(entry["validation_metric"]) <= 1.0
            or not _finite(entry["head_learning_rate"])
            or float(entry["head_learning_rate"]) <= 0.0
            or (encoder_rate is not None and not _finite(encoder_rate))
            or (expected_stage == "warmup" and encoder_rate is not None)
            or (expected_stage == "fine_tune" and not _finite(encoder_rate))
            or (
                expected_stage == "fine_tune"
                and _finite(encoder_rate)
                and float(encoder_rate) <= 0.0
            )
        ):
            raise ManifestBuildError("Symile training history content is invalid")
        epochs.append(epoch)
        stages[epoch] = str(entry["stage"])
        metrics[epoch] = float(entry["validation_metric"])
    selected_epoch = selection["selected_epoch"]
    if (
        len(epochs) != len(set(epochs))
        or selected_epoch not in stages
        or stages[selected_epoch] != selection["selected_stage"]
        or not np.isclose(
            metrics.get(selected_epoch, float("nan")),
            selection["selected_validation_metric"],
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise ManifestBuildError("Symile selected epoch is absent or inconsistent in history")


def _validate_fold_contents(
    family: str,
    lab_preprocessor: object | None,
    training_history: Sequence[Mapping[str, object]] | None,
    source_cxr: Mapping[str, object] | None,
) -> None:
    if family in TABULAR_FAMILIES:
        if lab_preprocessor is not None or training_history is not None:
            raise ManifestBuildError("Tabular fold received neural-only artifacts")
    elif not training_history:
        raise ManifestBuildError("Neural fold requires a training history")
    if family in FUSION_FAMILIES:
        if lab_preprocessor is None or source_cxr is None:
            raise ManifestBuildError("Fusion fold requires preprocessing and CXR lineage")
    elif source_cxr is not None:
        raise ManifestBuildError("Non-fusion fold does not accept CXR lineage")


def _fold_files(family: str) -> set[str]:
    common = {CONFIG_FILENAME, FOLD_MANIFEST_FILENAME, OOF_FILENAME}
    if family in TABULAR_FAMILIES:
        return common | {TABULAR_MODEL_FILENAME}
    files = common | {NEURAL_MODEL_FILENAME, TRAINING_HISTORY_FILENAME}
    return files | ({LAB_PREPROCESSOR_FILENAME} if family in FUSION_FAMILIES else set())


def _validate_repeat_metrics(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {"17", "42", "2026"}:
        raise ManifestBuildError("Symile repeat metric set is invalid")
    for metrics in value.values():
        if not isinstance(metrics, Mapping) or set(metrics) != {
            "roc_auc",
            "average_precision",
            "brier_score",
        }:
            raise ManifestBuildError("Symile repeat metric fields are invalid")
        if not all(_finite(metric) and 0.0 <= float(metric) <= 1.0 for metric in metrics.values()):
            raise ManifestBuildError("Symile repeat metrics are invalid")


def _fold_repeat_metrics(
    folds: Sequence[ValidatedFoldPackage],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    canonical_targets: pd.Series | None = None
    for seed in (17, 42, 2026):
        scoped = [item.oof.to_pandas() for item in folds if item.manifest["repeat_seed"] == seed]
        if len(scoped) != 5:
            raise ManifestBuildError("Symile development repeat lacks five OOF folds")
        frame = pd.concat(scoped, ignore_index=True)
        if (
            len(frame) != DEVELOPMENT_COUNT
            or frame["sample_id"].duplicated().any()
            or len(set(frame["sample_id"])) != DEVELOPMENT_COUNT
        ):
            raise ManifestBuildError("Symile development repeat contains duplicate OOF samples")
        indexed_targets = frame.set_index("sample_id")["target"].sort_index()
        if canonical_targets is None:
            canonical_targets = indexed_targets
        elif not indexed_targets.equals(canonical_targets):
            raise ManifestBuildError("Symile development targets differ across repeats")
        truth = frame["target"].to_numpy(dtype=np.int8)
        scores = frame["probability"].to_numpy(dtype=np.float64)
        result[str(seed)] = {
            "roc_auc": float(roc_auc_score(truth, scores)),
            "average_precision": float(average_precision_score(truth, scores)),
            "brier_score": float(brier_score_loss(truth, scores)),
        }
    return result


def _resolve_family_folds(
    document: Mapping[str, Any], model_root: str | Path
) -> list[ValidatedFoldPackage]:
    folds_root = Path(model_root) / "folds"
    folds: list[ValidatedFoldPackage] = []
    baseline_lineage: Mapping[str, Any] | None = None
    baseline_source_sha256: str | None = None
    baseline_source_development: str | None = None
    for reference in document["fold_packages"]:
        package = validate_fold_package(
            folds_root / reference["fold_package_id"],
            expected_fold_package_id=reference["fold_package_id"],
        )
        if (
            package.manifest_sha256 != reference["fold_manifest_sha256"]
            or package.manifest["repeat_seed"] != reference["repeat_seed"]
            or package.manifest["outer_fold"] != reference["outer_fold"]
            or package.manifest["family"] != document["family"]
            or package.manifest["config"]["semantic_sha256"] != document["semantic_config_sha256"]
        ):
            raise ManifestBuildError("Symile development fold reference is inconsistent")
        lineage = package.manifest["lineage"]
        source_sha256 = package.manifest["config"]["source_sha256"]
        source_development = (
            package.manifest["source_cxr"]["development_id"]
            if package.manifest["source_cxr"] is not None
            else None
        )
        frozen = {
            key: lineage[key]
            for key in (
                "bundle_id",
                "bundle_manifest_sha256",
                "official_split_assignment_id",
                "cv_assignment_id",
                "cv_manifest_sha256",
                "task_id",
                "git_commit",
                "dependency_lock_sha256",
                "encoder_identity",
                "transform_contract",
                "pretrained_weight",
            )
        }
        if baseline_lineage is None:
            baseline_lineage = frozen
            baseline_source_sha256 = source_sha256
            baseline_source_development = source_development
        elif frozen != baseline_lineage:
            raise ManifestBuildError("Symile development folds have inconsistent frozen lineage")
        elif (
            source_sha256 != baseline_source_sha256
            or source_development != baseline_source_development
        ):
            raise ManifestBuildError("Symile development folds have inconsistent config lineage")
        folds.append(package)
    coordinates = [(item.manifest["repeat_seed"], item.manifest["outer_fold"]) for item in folds]
    if coordinates != [(seed, fold) for seed in (17, 42, 2026) for fold in range(5)]:
        raise ManifestBuildError("Symile development does not resolve exact unique folds")
    return folds


def _nested_metrics_close(
    left: Mapping[str, Mapping[str, float]],
    right: Mapping[str, Mapping[str, float]],
) -> bool:
    return all(
        np.isclose(left[seed][metric], right[seed][metric], rtol=0.0, atol=1e-15)
        for seed in right
        for metric in right[seed]
    )


def _validate_budget(family: str, selected: object, budget: object) -> None:
    expects = family != "labs_logistic"
    if not expects:
        if selected is not None or budget is not None:
            raise ManifestBuildError("Labs Logistic Regression has no selected budget")
        return
    if (
        not isinstance(selected, list | tuple)
        or len(selected) != 15
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in selected
        )
        or isinstance(budget, bool)
        or not isinstance(budget, int)
        or budget != int(np.median(np.asarray(selected, dtype=np.int64)))
    ):
        raise ManifestBuildError("Symile median M6 budget is invalid")


def _development_summary(document: Mapping[str, Any]) -> str:
    lines = [
        f"# Symile development result: {document['family']}",
        "",
        f"- Development ID: `{document['development_id']}`",
        "- Complete outer folds: 15",
        "- Official test access: closed",
        "",
        "| Repeat | AUROC | Average Precision | Brier |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for seed in (17, 42, 2026):
        metrics = document["repeat_metrics"][str(seed)]
        lines.append(
            f"| {seed} | {metrics['roc_auc']:.6f} | {metrics['average_precision']:.6f} | "
            f"{metrics['brier_score']:.6f} |"
        )
    if document["median_m6_budget"] is not None:
        lines.extend(["", f"- Frozen median M6 budget: {document['median_m6_budget']}"])
    return "\n".join(lines) + "\n"


def _analysis_summary(document: Mapping[str, Any]) -> str:
    lines = [
        "# Symile cross-family development analysis",
        "",
        f"- Analysis ID: `{document['analysis_id']}`",
        "- Evidence: three-repeat patient-grouped OOF development predictions",
        "- Official test access: closed",
        "- Uncertainty: no fold-independence interval or pooled OOF bootstrap",
        "",
        "## Neural mean-logit ensemble",
        "",
        "| Family | AUROC | Average Precision | Brier |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family, metrics in sorted(document["ensemble_metrics"].items()):
        lines.append(
            f"| {family} | {metrics['roc_auc']:.6f} | {metrics['average_precision']:.6f} | "
            f"{metrics['brier_score']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Repeat-level paired effects",
            "",
            "| Comparison | Repeat | Delta AUROC | Delta AP | Delta Brier |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparison, repeats in sorted(document["paired_effects"].items()):
        for seed, metrics in sorted(repeats.items(), key=lambda item: int(item[0])):
            lines.append(
                f"| {comparison} | {seed} | {metrics['roc_auc']:.6f} | "
                f"{metrics['average_precision']:.6f} | {metrics['brier_score']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Observedness ablation",
            "",
            "The manifest records gated minus gated-without-observedness effects for every "
            "repeat and for the mean-logit development ensemble.",
            "",
            "Repeat-to-repeat values are stability diagnostics. They are not independent-fold "
            "confidence intervals.",
        ]
    )
    return "\n".join(lines) + "\n"


def _derive_analysis(
    family_ids: Mapping[str, str],
    report_root: str | Path,
    model_root: str | Path,
) -> dict[str, object]:
    frames: dict[str, pd.DataFrame] = {}
    family_folds: dict[str, dict[tuple[int, int], ValidatedFoldPackage]] = {}
    for family, development_id in family_ids.items():
        result = validate_development_result(
            Path(report_root) / "families" / development_id,
            model_root=model_root,
            expected_development_id=development_id,
        )
        if result.manifest["family"] != family:
            raise ManifestBuildError("Symile analysis family authority is mislabeled")
        fold_frames = []
        resolved = _resolve_family_folds(result.manifest, model_root)
        family_folds[family] = {
            (fold.manifest["repeat_seed"], fold.manifest["outer_fold"]): fold for fold in resolved
        }
        for fold in resolved:
            frame = fold.oof.to_pandas()
            frame["repeat_seed"] = fold.manifest["repeat_seed"]
            fold_frames.append(frame)
        frames[family] = pd.concat(fold_frames, ignore_index=True).sort_values(
            ["repeat_seed", "sample_id"], kind="stable"
        )
    _validate_cross_family_fold_lineage(family_folds, family_ids)
    _validate_cross_family_oof(frames)
    repeat_metrics = {
        family: {
            str(seed): _metrics(frame.loc[frame["repeat_seed"] == seed]) for seed in (17, 42, 2026)
        }
        for family, frame in frames.items()
    }
    comparisons = {
        "concat_minus_cxr": ("concat", "cxr"),
        "gated_minus_cxr": ("gated", "cxr"),
        "gated_minus_concat": ("gated", "concat"),
    }
    paired = {
        name: _paired_effects(frames[left], frames[right])
        for name, (left, right) in comparisons.items()
    }
    neural = ("cxr", "concat", "gated", "gated_no_observedness")
    ensemble = {family: _ensemble_metrics(frames[family]) for family in neural}
    ablation_repeats = _paired_effects(frames["gated"], frames["gated_no_observedness"])
    ablation_ensemble = {
        metric: ensemble["gated"][metric] - ensemble["gated_no_observedness"][metric]
        for metric in ("roc_auc", "average_precision", "brier_score")
    }
    return {
        "repeat_metrics": repeat_metrics,
        "paired_effects": paired,
        "ensemble_metrics": ensemble,
        "observedness_ablation": {
            "repeat_effects": ablation_repeats,
            "ensemble_effect": ablation_ensemble,
        },
    }


def _validate_cross_family_fold_lineage(
    families: Mapping[str, Mapping[tuple[int, int], ValidatedFoldPackage]],
    family_ids: Mapping[str, str],
) -> None:
    coordinates = {(seed, fold) for seed in (17, 42, 2026) for fold in range(5)}
    data_fields = (
        "bundle_id",
        "bundle_manifest_sha256",
        "official_split_assignment_id",
        "cv_assignment_id",
        "cv_manifest_sha256",
        "task_id",
    )
    for coordinate in coordinates:
        packages = [families[family][coordinate] for family in SYMILE_M5_FAMILIES]
        if len({item.manifest["inner_split"]["inner_split_id"] for item in packages}) != 1:
            raise ManifestBuildError("Symile analysis families use different inner splits")
        if any(
            len({item.manifest["lineage"][field] for item in packages}) != 1
            for field in data_fields
        ):
            raise ManifestBuildError("Symile analysis families use different frozen data lineage")
        cxr = families["cxr"][coordinate]
        for family in FUSION_FAMILIES:
            source = families[family][coordinate].manifest["source_cxr"]
            if (
                source["development_id"] != family_ids["cxr"]
                or source["fold_package_id"] != cxr.manifest["fold_package_id"]
                or source["fold_manifest_sha256"] != cxr.manifest_sha256
            ):
                raise ManifestBuildError("Symile fusion analysis uses incompatible CXR lineage")


def _validate_cross_family_oof(frames: Mapping[str, pd.DataFrame]) -> None:
    for seed in (17, 42, 2026):
        reference: pd.Series | None = None
        for family in SYMILE_M5_FAMILIES:
            scoped = frames[family].loc[frames[family]["repeat_seed"] == seed]
            targets = scoped.set_index("sample_id")["target"].sort_index()
            if reference is None:
                reference = targets
            elif not targets.equals(reference):
                raise ManifestBuildError("Symile analysis OOF targets differ across families")


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    truth = frame["target"].to_numpy(dtype=np.int8)
    scores = frame["probability"].to_numpy(dtype=np.float64)
    if len(frame) != DEVELOPMENT_COUNT or set(truth.tolist()) != {0, 1}:
        raise ManifestBuildError("Symile analysis family evidence is incomplete")
    return {
        "roc_auc": float(roc_auc_score(truth, scores)),
        "average_precision": float(average_precision_score(truth, scores)),
        "brier_score": float(brier_score_loss(truth, scores)),
    }


def _paired_effects(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for seed in (17, 42, 2026):
        left_seed = left.loc[left["repeat_seed"] == seed]
        right_seed = right.loc[right["repeat_seed"] == seed]
        aligned = left_seed.merge(
            right_seed,
            on=["sample_id", "repeat_seed"],
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        if (
            len(aligned) != DEVELOPMENT_COUNT
            or not aligned["target_left"].eq(aligned["target_right"]).all()
        ):
            raise ManifestBuildError("Symile analysis paired evidence is misaligned")
        left_metrics = _metrics(
            aligned.rename(columns={"target_left": "target", "probability_left": "probability"})
        )
        right_metrics = _metrics(
            aligned.rename(columns={"target_right": "target", "probability_right": "probability"})
        )
        result[str(seed)] = {
            metric: left_metrics[metric] - right_metrics[metric]
            for metric in ("roc_auc", "average_precision", "brier_score")
        }
    return result


def _ensemble_metrics(frame: pd.DataFrame) -> dict[str, float]:
    logits = frame.pivot(index="sample_id", columns="repeat_seed", values="logit")
    targets = frame.pivot(index="sample_id", columns="repeat_seed", values="target")
    if (
        list(logits.columns) != [17, 42, 2026]
        or len(logits) != DEVELOPMENT_COUNT
        or logits.isna().any().any()
        or list(targets.columns) != [17, 42, 2026]
        or not targets.nunique(axis=1).eq(1).all()
    ):
        raise ManifestBuildError("Symile analysis neural repeat evidence is misaligned")
    probabilities = _sigmoid(logits.to_numpy(dtype=np.float64).mean(axis=1))
    evidence = pd.DataFrame({"target": targets.iloc[:, 0], "probability": probabilities})
    return _metrics(evidence)


def _nested_numeric_close(left: object, right: object) -> bool:
    if isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and set(left) == set(right)
            and all(_nested_numeric_close(left[key], value) for key, value in right.items())
        )
    return (
        _finite(left)
        and _finite(right)
        and np.isclose(float(left), float(right), rtol=0.0, atol=1e-15)
    )


def _publish_immutable(stage: Path, destination: Path, validator: Any) -> None:
    validator(stage, enforce_directory_name=False)
    if destination.exists():
        validator(destination)
        shutil.rmtree(stage)
    else:
        os.replace(stage, destination)


def _require_physical_directory(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ManifestBuildError("Symile artifact path is not a physical directory")


def _require_exact_regular_files(root: Path, expected: set[str]) -> None:
    _require_physical_directory(root)
    entries = list(os.scandir(root))
    if {entry.name for entry in entries} != expected or any(
        entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in entries
    ):
        raise ManifestBuildError("Symile artifact file set is incomplete or unexpected")


def _load_json_list(path: Path) -> list[object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestBuildError("Symile training history is unreadable") from exc
    if not isinstance(value, list) or not value:
        raise ManifestBuildError("Symile training history is invalid")
    return value


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)
    return result


def _validate_family(value: object) -> None:
    if value not in SYMILE_M5_FAMILIES:
        raise ManifestBuildError("Symile M5 family is invalid")


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _identity(value: object, prefix: str) -> bool:
    return (
        isinstance(value, str) and value.startswith(prefix) and _sha256(value.removeprefix(prefix))
    )


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
