"""Publish immutable private prediction-evidence objects."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from beyondcxr.data.hashing import LogicalTableContract, logical_table_sha256, sha256_file
from beyondcxr.data.symile_schemas import (
    LABEL_POLICY_VERSION,
    OUTER_FOLDS,
    REPEAT_SEEDS,
    TASK_ID,
)
from beyondcxr.utils.publication import (
    install_immutable_directory,
    staging_directory,
    validate_path_component,
)
from beyondcxr.utils.semantic_identity import canonical_scientific_id

PREDICTION_SCHEMA_VERSION = 1
PREDICTION_PREFIX = "prediction-"
PREDICTIONS_FILENAME = "predictions.parquet"
PREDICTION_MANIFEST_FILENAME = "manifest.json"
PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("target", pa.int8(), nullable=False),
        pa.field("logit", pa.float64(), nullable=False),
        pa.field("probability", pa.float64(), nullable=False),
    ]
)
_PREDICTION_TABLE_CONTRACT = LogicalTableContract(
    "private.predictions",
    1,
    PREDICTION_SCHEMA,
    ("sample_id",),
    ("sample_id",),
)
_MANIFEST_FIELDS = {
    "prediction_schema_version",
    "prediction_id",
    "dataset_id",
    "model_package_id",
    "task_id",
    "bundle_id",
    "scope",
    "cv_assignment_id",
    "repeat_seed",
    "outer_fold",
    "prediction_table_sha256",
    "row_count",
    "prediction_file_sha256",
}
_HELD_OUT_CONTROL_FIELD = "authorized_by_freeze_id"
_HELD_OUT_PUBLICATION_FIELD = "held_out_publication"
_SYMILE_TEST_SEMANTIC_FIELDS = {"label_policy_version", "inference_policy"}
_HELD_OUT_PUBLICATION_GUARD = object()
SYMILE_TEST_INFERENCE_POLICY = {
    "inference_policy_schema_version": 1,
    "package_scope": "single_package",
    "preprocessing": "package_bound_deterministic_official_test",
    "output": "raw_logit_and_sigmoid_probability",
    "execution": "one_shot",
    "calibration": "none",
    "thresholding": "none",
}


@dataclass(frozen=True)
class ValidatedPredictionEvidence:
    """Validated private logical prediction content and its identities."""

    directory: Path
    manifest: Mapping[str, Any]
    predictions: pa.Table
    created: bool = False

    @property
    def prediction_id(self) -> str:
        """Return the validated semantic prediction identity."""
        return str(self.manifest["prediction_id"])


@dataclass(frozen=True)
class ValidatedHeldOutPublicationAuthority:
    """Live study authority bound to one package and canonical held-out cohort."""

    dataset_id: str
    model_package_id: str
    task_id: str
    bundle_id: str
    freeze_id: str
    sample_ids: tuple[str, ...]
    targets: tuple[int, ...]
    publication_record: Mapping[str, object]
    _authorization: object
    _projection: object | None
    _guard: object


def build_prediction_table(
    sample_ids: Sequence[str],
    targets: Sequence[int] | np.ndarray,
    logits: Sequence[float] | np.ndarray,
) -> pa.Table:
    """Build canonical ordered sample-level prediction content."""
    ids = list(sample_ids)
    truth = np.asarray(targets).reshape(-1)
    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    if (
        not ids
        or any(not isinstance(value, str) or not value for value in ids)
        or len(ids) != len(set(ids))
        or truth.shape != (len(ids),)
        or scores.shape != (len(ids),)
        or set(np.unique(truth).tolist()) - {0, 1}
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Prediction content is invalid")
    probabilities = _sigmoid_array(scores)
    rows = sorted(zip(ids, truth.astype(np.int8), scores, probabilities, strict=True))
    return pa.Table.from_pylist(
        [
            {
                "sample_id": sample_id,
                "target": int(target),
                "logit": float(logit),
                "probability": float(probability),
            }
            for sample_id, target, logit, probability in rows
        ],
        schema=PREDICTION_SCHEMA,
    )


def publish_prediction_evidence(
    *,
    private_root: str | Path,
    dataset_id: str,
    model_package_id: str,
    task_id: str,
    bundle_id: str,
    scope: str,
    sample_ids: Sequence[str],
    targets: Sequence[int] | np.ndarray,
    logits: Sequence[float] | np.ndarray,
    cv_assignment_id: str | None = None,
    repeat_seed: int | None = None,
    outer_fold: int | None = None,
    label_policy_version: str | None = None,
    inference_policy: Mapping[str, object] | None = None,
    authorized_by_freeze_id: str | None = None,
    prospective_mutation: object | None = None,
) -> ValidatedPredictionEvidence:
    """Publish one immutable prediction object under logical scientific identity."""
    if _is_held_out_prediction(dataset_id, scope):
        raise PermissionError(
            "Held-out prediction publication requires a study-specific validated authority"
        )
    return _publish_prediction_evidence(
        private_root=private_root,
        dataset_id=dataset_id,
        model_package_id=model_package_id,
        task_id=task_id,
        bundle_id=bundle_id,
        scope=scope,
        sample_ids=sample_ids,
        targets=targets,
        logits=logits,
        cv_assignment_id=cv_assignment_id,
        repeat_seed=repeat_seed,
        outer_fold=outer_fold,
        label_policy_version=label_policy_version,
        inference_policy=inference_policy,
        authorized_by_freeze_id=authorized_by_freeze_id,
        prospective_mutation=prospective_mutation,
    )


def _publish_held_out_prediction_evidence(
    *,
    publication_authority: ValidatedHeldOutPublicationAuthority,
    **kwargs: Any,
) -> ValidatedPredictionEvidence:
    """Publish held-out evidence under reauthenticated package/cohort authority."""
    _require_held_out_publication_authority(publication_authority, kwargs)
    return _publish_prediction_evidence(
        held_out_authority=publication_authority,
        **kwargs,
    )


def _publish_prediction_evidence(
    *,
    private_root: str | Path,
    dataset_id: str,
    model_package_id: str,
    task_id: str,
    bundle_id: str,
    scope: str,
    sample_ids: Sequence[str],
    targets: Sequence[int] | np.ndarray,
    logits: Sequence[float] | np.ndarray,
    cv_assignment_id: str | None = None,
    repeat_seed: int | None = None,
    outer_fold: int | None = None,
    label_policy_version: str | None = None,
    inference_policy: Mapping[str, object] | None = None,
    authorized_by_freeze_id: str | None = None,
    prospective_mutation: object | None = None,
    held_out_authority: ValidatedHeldOutPublicationAuthority | None = None,
) -> ValidatedPredictionEvidence:
    """Publish one immutable prediction object under logical scientific identity."""
    _validate_prediction_coordinate(
        dataset_id=dataset_id,
        model_package_id=model_package_id,
        task_id=task_id,
        bundle_id=bundle_id,
        scope=scope,
        cv_assignment_id=cv_assignment_id,
        repeat_seed=repeat_seed,
        outer_fold=outer_fold,
        label_policy_version=label_policy_version,
        inference_policy=inference_policy,
        authorized_by_freeze_id=authorized_by_freeze_id,
    )
    if dataset_id == "symile" and scope == "outer_fold_oof":
        from beyondcxr.training.symile_execution import (
            require_symile_prospective_mutation,
            require_symile_prospective_path,
        )

        require_symile_prospective_mutation(prospective_mutation)
        require_symile_prospective_path(
            prospective_mutation,
            private_root,
            "Symile development prediction root",
        )
    elif prospective_mutation is not None:
        raise ValueError("Prospective mutation authority is invalid for this prediction scope")
    if _is_held_out_prediction(dataset_id, scope):
        if held_out_authority is None:
            raise PermissionError("Held-out prediction publication requires live study authority")
        _require_held_out_publication_authority(
            held_out_authority,
            {
                "dataset_id": dataset_id,
                "model_package_id": model_package_id,
                "task_id": task_id,
                "bundle_id": bundle_id,
                "scope": scope,
                "sample_ids": sample_ids,
                "targets": targets,
                "authorized_by_freeze_id": authorized_by_freeze_id,
            },
        )
        held_out_publication = dict(held_out_authority.publication_record)
    elif held_out_authority is not None:
        raise ValueError("Non-held-out predictions cannot declare held-out publication")
    else:
        held_out_publication = None
    table = build_prediction_table(sample_ids, targets, logits)
    _validate_dataset_sample_ids(dataset_id, table["sample_id"].to_pylist())
    logical_hash = logical_table_sha256(table, _PREDICTION_TABLE_CONTRACT)
    semantic = {
        "dataset_id": dataset_id,
        "model_package_id": model_package_id,
        "task_id": task_id,
        "bundle_id": bundle_id,
        "scope": scope,
        "cv_assignment_id": cv_assignment_id,
        "repeat_seed": repeat_seed,
        "outer_fold": outer_fold,
        "prediction_table_sha256": logical_hash,
    }
    if _is_held_out_prediction(dataset_id, scope):
        semantic[_HELD_OUT_CONTROL_FIELD] = authorized_by_freeze_id
    if dataset_id == "symile" and scope == "test":
        if label_policy_version is None or inference_policy is None:
            raise ValueError("Symile test prediction policy is incomplete")
        semantic.update(
            {
                "label_policy_version": label_policy_version,
                "inference_policy": dict(inference_policy),
            }
        )
    prediction_id = canonical_scientific_id(
        PREDICTION_PREFIX, semantic, domain="prediction", schema_version=1
    )
    scope_root = Path(private_root) / "predictions" / dataset_id
    if scope == "outer_fold_oof":
        scope_root /= "oof"
    elif dataset_id == "symile" and scope == "test":
        scope_root /= "test"
    destination = scope_root / prediction_id
    stage = staging_directory(destination)
    try:
        prediction_path = stage / PREDICTIONS_FILENAME
        pq.write_table(table, prediction_path, compression="zstd")
        manifest = {
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
            **semantic,
            "prediction_id": prediction_id,
            "row_count": table.num_rows,
            "prediction_file_sha256": sha256_file(prediction_path),
        }
        if held_out_publication is not None:
            manifest[_HELD_OUT_PUBLICATION_FIELD] = dict(held_out_publication)
        (stage / PREDICTION_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        validate_prediction_evidence(stage, enforce_directory_name=False)
        created = install_immutable_directory(stage, destination, validate_prediction_evidence)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    validated = validate_prediction_evidence(
        destination,
        expected_prediction_id=prediction_id,
        expected_authorizing_freeze_id=authorized_by_freeze_id,
    )
    if validated.manifest.get(_HELD_OUT_CONTROL_FIELD) != authorized_by_freeze_id:
        raise ValueError("Existing prediction evidence has different control provenance")
    return replace(validated, created=created)


def validate_prediction_evidence(
    directory: str | Path,
    *,
    expected_prediction_id: str | None = None,
    expected_model_package_id: str | None = None,
    expected_authorizing_freeze_id: str | None = None,
    enforce_directory_name: bool = True,
) -> ValidatedPredictionEvidence:
    """Validate physical integrity and canonical logical prediction identity."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Prediction evidence must be a physical directory")
    with os.scandir(root) as entries:
        inspected = list(entries)
    if {entry.name for entry in inspected} != {
        PREDICTIONS_FILENAME,
        PREDICTION_MANIFEST_FILENAME,
    } or any(entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in inspected):
        raise ValueError("Prediction evidence has an invalid file set")
    manifest_bytes = (root / PREDICTION_MANIFEST_FILENAME).read_bytes()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Prediction manifest is unreadable") from exc
    if not isinstance(manifest, dict) or set(manifest) != _manifest_fields(manifest):
        raise ValueError("Prediction manifest has an unexpected field set")
    schema_version = manifest["prediction_schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != PREDICTION_SCHEMA_VERSION
    ):
        raise ValueError("Prediction schema version is invalid")
    _validate_prediction_coordinate(
        dataset_id=manifest["dataset_id"],
        model_package_id=manifest["model_package_id"],
        task_id=manifest["task_id"],
        bundle_id=manifest["bundle_id"],
        scope=manifest["scope"],
        cv_assignment_id=manifest["cv_assignment_id"],
        repeat_seed=manifest["repeat_seed"],
        outer_fold=manifest["outer_fold"],
        label_policy_version=manifest.get("label_policy_version"),
        inference_policy=manifest.get("inference_policy"),
        authorized_by_freeze_id=manifest.get(_HELD_OUT_CONTROL_FIELD),
    )
    prediction_path = root / PREDICTIONS_FILENAME
    if sha256_file(prediction_path) != manifest["prediction_file_sha256"]:
        raise ValueError("Prediction file SHA-256 mismatch")
    table = pq.read_table(prediction_path)
    _validate_table(table)
    _validate_dataset_sample_ids(manifest["dataset_id"], table["sample_id"].to_pylist())
    row_count = manifest["row_count"]
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ValueError("Prediction row count is invalid")
    if (
        table.num_rows != row_count
        or logical_table_sha256(table, _PREDICTION_TABLE_CONTRACT)
        != manifest["prediction_table_sha256"]
    ):
        raise ValueError("Prediction logical content differs from its manifest")
    semantic = {
        key: manifest[key]
        for key in _MANIFEST_FIELDS
        if key
        not in {
            "prediction_schema_version",
            "prediction_id",
            "row_count",
            "prediction_file_sha256",
        }
    }
    if _is_held_out_prediction(manifest["dataset_id"], manifest["scope"]):
        semantic[_HELD_OUT_CONTROL_FIELD] = manifest[_HELD_OUT_CONTROL_FIELD]
        _validate_held_out_publication_record(
            manifest["dataset_id"], manifest.get(_HELD_OUT_PUBLICATION_FIELD)
        )
    if manifest["dataset_id"] == "symile" and manifest["scope"] == "test":
        semantic.update({key: manifest[key] for key in _SYMILE_TEST_SEMANTIC_FIELDS})
    prediction_id = canonical_scientific_id(
        PREDICTION_PREFIX, semantic, domain="prediction", schema_version=1
    )
    if manifest["prediction_id"] != prediction_id:
        raise ValueError("Prediction identity differs from logical content")
    if expected_prediction_id is not None and prediction_id != expected_prediction_id:
        raise ValueError("Prediction evidence differs from the expected identity")
    if (
        expected_model_package_id is not None
        and manifest["model_package_id"] != expected_model_package_id
    ):
        raise ValueError("Prediction evidence is bound to a different model package")
    if (
        expected_authorizing_freeze_id is not None
        and manifest.get(_HELD_OUT_CONTROL_FIELD) != expected_authorizing_freeze_id
    ):
        raise ValueError("Prediction evidence is bound to a different scientific freeze")
    if enforce_directory_name and root.name != prediction_id:
        raise ValueError("Prediction directory differs from its semantic identity")
    return ValidatedPredictionEvidence(
        root,
        manifest,
        table,
    )


def _validate_table(table: pa.Table) -> None:
    if table.schema != PREDICTION_SCHEMA or table.num_rows <= 0:
        raise ValueError("Prediction table schema or size is invalid")
    rows = table.to_pylist()
    ids = [row["sample_id"] for row in rows]
    if (
        any(not isinstance(value, str) or not value for value in ids)
        or ids != sorted(ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("Prediction rows are not uniquely ordered")
    for row in rows:
        if (
            row["target"] not in {0, 1}
            or not math.isfinite(row["logit"])
            or not math.isfinite(row["probability"])
            or not math.isclose(
                row["probability"],
                _sigmoid(row["logit"]),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("Prediction row violates its numerical contract")


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _validate_prediction_coordinate(
    *,
    dataset_id: object,
    model_package_id: object,
    task_id: object,
    bundle_id: object,
    scope: object,
    cv_assignment_id: object,
    repeat_seed: object,
    outer_fold: object,
    label_policy_version: object,
    inference_policy: object,
    authorized_by_freeze_id: object,
) -> None:
    validate_path_component(task_id, "prediction task_id")
    _require_identity(bundle_id, ("bundle-",), "bundle")
    if dataset_id == "rsna":
        if task_id != "pneumonia":
            raise ValueError("RSNA prediction task is invalid")
        _require_identity(model_package_id, ("model-package-",), "model package")
        if scope != "test" or any(
            value is not None
            for value in (
                cv_assignment_id,
                repeat_seed,
                outer_fold,
                label_policy_version,
                inference_policy,
            )
        ):
            raise ValueError("Held-out prediction coordinate is invalid")
        _require_identity(authorized_by_freeze_id, ("rsna-freeze-",), "RSNA scientific freeze")
        return
    if dataset_id != "symile":
        raise ValueError("Prediction dataset_id is unsupported")
    if task_id != TASK_ID:
        raise ValueError("Symile prediction task is invalid")
    if scope == "outer_fold_oof":
        _require_identity(model_package_id, ("fold-package-",), "model package")
        if any(
            value is not None
            for value in (
                label_policy_version,
                inference_policy,
                authorized_by_freeze_id,
            )
        ):
            raise ValueError("OOF predictions cannot declare test authorization")
        _require_identity(cv_assignment_id, ("cv-assignment-",), "CV assignment")
        if (
            isinstance(repeat_seed, bool)
            or not isinstance(repeat_seed, int)
            or repeat_seed not in REPEAT_SEEDS
            or isinstance(outer_fold, bool)
            or not isinstance(outer_fold, int)
            or outer_fold not in OUTER_FOLDS
        ):
            raise ValueError("OOF prediction coordinate is invalid")
        return
    if scope == "test":
        _require_identity(model_package_id, ("final-package-",), "model package")
        _require_identity(authorized_by_freeze_id, ("pretest-freeze-",), "pretest freeze")
        inference_schema_version = (
            inference_policy.get("inference_policy_schema_version")
            if isinstance(inference_policy, Mapping)
            else None
        )
        if (
            label_policy_version != LABEL_POLICY_VERSION
            or isinstance(inference_schema_version, bool)
            or not isinstance(inference_schema_version, int)
            or inference_schema_version != 1
            or inference_policy != SYMILE_TEST_INFERENCE_POLICY
        ):
            raise ValueError("Symile test prediction scientific policy is invalid")
        if any(value is not None for value in (cv_assignment_id, repeat_seed, outer_fold)):
            raise ValueError("Symile test prediction coordinate is invalid")
        return
    raise ValueError("Symile prediction scope is invalid")


def _manifest_fields(manifest: Mapping[str, object]) -> set[str]:
    fields = set(_MANIFEST_FIELDS)
    if _is_held_out_prediction(manifest.get("dataset_id"), manifest.get("scope")):
        fields.add(_HELD_OUT_CONTROL_FIELD)
        fields.add(_HELD_OUT_PUBLICATION_FIELD)
    if manifest.get("dataset_id") == "symile" and manifest.get("scope") == "test":
        fields.update(_SYMILE_TEST_SEMANTIC_FIELDS)
    return fields


def _is_held_out_prediction(dataset_id: object, scope: object) -> bool:
    return dataset_id in {"rsna", "symile"} and scope == "test"


def _validate_held_out_publication_record(dataset_id: object, value: object) -> None:
    if dataset_id == "rsna":
        expected = {"authority_kind", "execution_id"}
        prefix = "rsna-execution-"
        kind = "rsna-frozen-campaign-v1"
    elif dataset_id == "symile":
        expected = {"authority_kind", "execution_id"}
        prefix = "symile-execution-"
        kind = "symile-opened-campaign-v1"
    else:
        raise ValueError("Held-out publication dataset is invalid")
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or value.get("authority_kind") != kind
        or not isinstance(value.get("execution_id"), str)
        or not str(value["execution_id"]).startswith(prefix)
        or len(str(value["execution_id"])) != len(prefix) + 64
        or any(
            character not in "0123456789abcdef"
            for character in str(value["execution_id"])[len(prefix) :]
        )
    ):
        raise ValueError("Held-out publication authority is invalid")


def authorize_rsna_held_out_prediction_publication(
    authorization: object,
    *,
    model_package_id: str,
) -> ValidatedHeldOutPublicationAuthority:
    """Bind current RSNA execution/freeze authority to its canonical test cohort."""
    from beyondcxr.training.rsna_campaign_control import require_validated_rsna_held_out
    from beyondcxr.training.rsna_evaluation_result import validate_rsna_model_package
    from beyondcxr.training.rsna_validation_evidence import _derive_canonical_held_out_cohort

    capability = require_validated_rsna_held_out(authorization)
    capability.require(model_package_id)
    package = validate_rsna_model_package(capability.freeze.model_root, model_package_id)
    cohort = _derive_canonical_held_out_cohort(capability.freeze.bundle_directory, package)
    return ValidatedHeldOutPublicationAuthority(
        dataset_id="rsna",
        model_package_id=model_package_id,
        task_id=str(package["task_id"]),
        bundle_id=str(package["bundle_id"]),
        freeze_id=capability.freeze.freeze_id,
        sample_ids=cohort.sample_ids,
        targets=cohort.targets,
        publication_record={
            "authority_kind": "rsna-frozen-campaign-v1",
            "execution_id": capability.current_execution.execution.execution_id,
        },
        _authorization=authorization,
        _projection=None,
        _guard=_HELD_OUT_PUBLICATION_GUARD,
    )


def authorize_symile_held_out_prediction_publication(
    authorization: object,
    *,
    model_package_id: str,
    projection: object,
) -> ValidatedHeldOutPublicationAuthority:
    """Bind current Symile opening authority to one final package and test projection."""
    from beyondcxr.training.symile_campaign_control import (
        require_validated_held_out_capability,
        validated_pretest_freeze_manifest,
    )
    from beyondcxr.training.symile_test_data import require_held_out_evaluation_projection

    capability = require_validated_held_out_capability(authorization)
    held_out = require_held_out_evaluation_projection(projection, freeze=capability.freeze)
    frozen_packages = {
        str(item["package_id"])
        for item in validated_pretest_freeze_manifest(capability.freeze)["final_packages"]
    }
    if model_package_id not in frozen_packages:
        raise PermissionError("Symile held-out publication package is not frozen")
    targets = held_out.prediction_targets()
    return ValidatedHeldOutPublicationAuthority(
        dataset_id="symile",
        model_package_id=model_package_id,
        task_id=held_out.task_id,
        bundle_id=held_out.bundle_id,
        freeze_id=capability.freeze.freeze_id,
        sample_ids=tuple(targets["sample_id"].astype(str)),
        targets=tuple(int(value) for value in targets["target"]),
        publication_record={
            "authority_kind": "symile-opened-campaign-v1",
            "execution_id": capability.execution.execution_id,
        },
        _authorization=authorization,
        _projection=projection,
        _guard=_HELD_OUT_PUBLICATION_GUARD,
    )


def _require_held_out_publication_authority(
    authority: object,
    values: Mapping[str, object],
) -> ValidatedHeldOutPublicationAuthority:
    if (
        not isinstance(authority, ValidatedHeldOutPublicationAuthority)
        or authority._guard is not _HELD_OUT_PUBLICATION_GUARD
    ):
        raise PermissionError("Held-out prediction publication authority is invalid")
    expected = _reauthenticate_held_out_publication_authority(authority)
    if (
        authority.dataset_id != expected.dataset_id
        or authority.model_package_id != expected.model_package_id
        or authority.task_id != expected.task_id
        or authority.bundle_id != expected.bundle_id
        or authority.freeze_id != expected.freeze_id
        or authority.sample_ids != expected.sample_ids
        or authority.targets != expected.targets
        or dict(authority.publication_record) != dict(expected.publication_record)
    ):
        raise PermissionError("Held-out prediction publication authority is stale or forged")
    sample_ids = tuple(str(value) for value in values.get("sample_ids", ()))
    targets = tuple(int(value) for value in np.asarray(values.get("targets", ())).reshape(-1))
    if (
        values.get("scope") != "test"
        or values.get("dataset_id") != authority.dataset_id
        or values.get("model_package_id") != authority.model_package_id
        or values.get("task_id") != authority.task_id
        or values.get("bundle_id") != authority.bundle_id
        or values.get("authorized_by_freeze_id") != authority.freeze_id
        or sample_ids != authority.sample_ids
        or targets != authority.targets
    ):
        raise PermissionError("Held-out prediction publication differs from live authority")
    _validate_held_out_publication_record(authority.dataset_id, authority.publication_record)
    return authority


def _reauthenticate_held_out_publication_authority(
    authority: ValidatedHeldOutPublicationAuthority,
) -> ValidatedHeldOutPublicationAuthority:
    """Re-derive a publication lease from its current study lifecycle authority."""
    if authority.dataset_id == "rsna":
        return authorize_rsna_held_out_prediction_publication(
            authority._authorization,
            model_package_id=authority.model_package_id,
        )
    if authority.dataset_id == "symile" and authority._projection is not None:
        return authorize_symile_held_out_prediction_publication(
            authority._authorization,
            model_package_id=authority.model_package_id,
            projection=authority._projection,
        )
    raise PermissionError("Held-out prediction publication authority is invalid")


def _validate_dataset_sample_ids(dataset_id: str, sample_ids: Sequence[str]) -> None:
    if dataset_id == "rsna":
        valid = all(
            value.startswith("rsna:")
            and len(value) > len("rsna:")
            and not any(character.isspace() for character in value)
            for value in sample_ids
        )
    else:
        valid = all(
            value.startswith("symile:") and value[len("symile:") :].isdigit()
            for value in sample_ids
        )
    if not valid:
        raise ValueError("Prediction sample identity does not match its dataset contract")


def _require_identity(value: object, prefixes: tuple[str, ...], name: str) -> str:
    validate_path_component(value, f"prediction {name}")
    if not isinstance(value, str):
        raise ValueError(f"Prediction {name} identity is invalid")
    if not any(
        value.startswith(prefix)
        and len(value) == len(prefix) + 64
        and all(character in "0123456789abcdef" for character in value[len(prefix) :])
        for prefix in prefixes
    ):
        raise ValueError(f"Prediction {name} identity is invalid")
    return value
