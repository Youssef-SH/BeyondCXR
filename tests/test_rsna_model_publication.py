from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from lightgbm import Booster, LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from beyondcxr.data.rsna_metadata_preprocess import metadata_input_contract
from beyondcxr.training.config import load_experiment_config
from beyondcxr.utils.package_identity import (
    _array,
    _tagged,
    fitted_object_state_sha256,
    package_scientific_config_payload,
    tensor_state_sha256,
)
from beyondcxr.utils.rsna_model_publication import (
    MODEL_PACKAGE_ID_PREFIX,
    REQUIRED_MANIFEST_FIELDS,
    model_package_id,
    publish_model_package,
    threshold_contract,
    validate_published_model,
)
from beyondcxr.utils.skops_io import load_skops, save_skops

_SHA256 = "a" * 64


def _manifest() -> dict[str, object]:
    config = load_experiment_config("configs/rsna_metadata_logistic.yaml")
    return {
        "bundle_id": config.dataset.bundle_id,
        "split_assignment_id": config.dataset.split_assignment_id,
        "task_id": config.task.task_id,
        "positive_class": 1,
        "family_id": config.family.family_id,
        "config_source_sha256": _SHA256,
        "config_semantic_sha256": _SHA256,
        "seed": 42,
        "git_commit": "commit-test",
        "git_dirty": False,
        "dependency_lock_sha256": _SHA256,
        "best_iteration": None,
        "thresholds": {"youden_j": 0.5, "target_sensitivity": 0.3},
        "threshold_contract": threshold_contract(sensitivity_target=0.9),
        "input_contract": metadata_input_contract(),
    }


def _publish(tmp_path: Path):
    features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    classifier = LogisticRegression().fit(features, [0, 0, 1, 1])
    model = Pipeline([("preprocess", "passthrough"), ("classifier", classifier)])
    serialized = save_skops(model, tmp_path / "source.skops")
    config = tmp_path / "source.yaml"
    config.write_bytes(Path("configs/rsna_metadata_logistic.yaml").read_bytes())
    manifest = {
        **_manifest(),
        "config_source_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }
    published = publish_model_package(
        model_root=tmp_path / "models" / "rsna",
        serialized_model_path=serialized,
        source_config_bytes=config.read_bytes(),
        manifest=manifest,
    )
    return features, model, published


def test_model_package_has_exact_reconstructable_artifacts(tmp_path: Path) -> None:
    features, model, published = _publish(tmp_path)

    assert published.package_directory == (
        tmp_path / "models" / "rsna" / "packages" / published.model_package_id
    )
    assert {path.name for path in published.package_directory.iterdir()} == {
        "model.skops",
        "resolved_config.yaml",
        "manifest.json",
    }
    document = validate_published_model(published.package_directory)
    assert set(document) == REQUIRED_MANIFEST_FIELDS
    assert document["model_package_id"].startswith(MODEL_PACKAGE_ID_PREFIX)
    assert document["model_package_id"] == model_package_id(document)
    assert published.created is True
    np.testing.assert_array_equal(
        load_skops(published.model_path).predict_proba(features),
        model.predict_proba(features),
    )


def test_model_package_preserves_exact_serialized_and_config_bytes(tmp_path: Path) -> None:
    _, _, published = _publish(tmp_path)
    assert published.model_path.read_bytes() == (tmp_path / "source.skops").read_bytes()
    assert published.config_path.read_bytes() == (tmp_path / "source.yaml").read_bytes()


def test_model_package_reuses_equivalent_alternate_serialization(tmp_path: Path) -> None:
    _, fitted, published = _publish(tmp_path)
    alternate_serialization = save_skops(fitted, tmp_path / "alternate.skops")
    assert alternate_serialization.read_bytes() != published.model_path.read_bytes()
    repeated = publish_model_package(
        model_root=tmp_path / "models" / "rsna",
        serialized_model_path=alternate_serialization,
        source_config_bytes=published.config_path.read_bytes(),
        manifest={
            **_manifest(),
            "config_source_sha256": hashlib.sha256(published.config_path.read_bytes()).hexdigest(),
        },
    )

    assert repeated.model_package_id == published.model_package_id
    assert repeated.created is False
    assert validate_published_model(repeated.package_directory) == validate_published_model(
        published.package_directory
    )


def test_model_package_id_is_deterministic_and_changes_with_semantic_content(
    tmp_path: Path,
) -> None:
    _, _, published = _publish(tmp_path)
    document = validate_published_model(published.package_directory)

    assert model_package_id(document) == model_package_id(dict(reversed(document.items())))
    changed = {**document, "seed": document["seed"] + 1}
    changed.pop("model_package_id")
    assert model_package_id(changed) != document["model_package_id"]

    unexpected = {**document, "unexpected_identity_input": "ignored"}
    with pytest.raises(ValueError, match="unexpected field set"):
        model_package_id(unexpected)


def test_fitted_state_canonicalization_distinguishes_typed_mapping_keys() -> None:
    assert _tagged({1: "value"}) != _tagged({"1": "value"})
    tagged = _tagged({1: "integer", "1": "string"})
    assert len(tagged["items"]) == 2


def test_fitted_state_canonicalization_preserves_scalar_and_sequence_types() -> None:
    values = [None, False, 0, 0.0, "0", [0], (0,), np.int64(0), np.float32(0.0)]
    representations = {
        json.dumps(_tagged(value), sort_keys=True, separators=(",", ":")) for value in values
    }
    assert len(representations) == len(values) - 2
    assert _tagged(np.int64(0)) == _tagged(0)
    assert _tagged(np.float32(0.0)) == _tagged(0.0)
    assert json.dumps(_tagged(-0.0)) != json.dumps(_tagged(0.0))
    with pytest.raises(ValueError, match="non-finite"):
        _tagged(float("nan"))
    with pytest.raises(TypeError, match="Unsupported"):
        _array(np.asarray([object()], dtype=object))


@pytest.mark.parametrize(
    "value",
    [np.asarray([b"bytes"]), np.asarray([b"bytes"], dtype=object)],
)
def test_fitted_state_canonicalization_rejects_byte_strings(value: np.ndarray) -> None:
    with pytest.raises(TypeError, match="Byte-string|Unsupported fitted-state scalar"):
        _array(value)


def test_fitted_state_unicode_identity_ignores_capacity_but_binds_values() -> None:
    encoder = OneHotEncoder(handle_unknown="ignore").fit(np.asarray([["aa"], ["bb"]], dtype="<U2"))
    wider = deepcopy(encoder)
    wider.categories_[0] = wider.categories_[0].astype("<U128")
    changed = deepcopy(wider)
    changed.categories_[0][1] = "cc"

    assert fitted_object_state_sha256(wider) == fitted_object_state_sha256(encoder)
    assert fitted_object_state_sha256(changed) != fitted_object_state_sha256(encoder)


def test_canonical_fitted_state_hash_binds_coefficients_and_preprocessor_values() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    classifier = LogisticRegression().fit(features, [0, 0, 1, 1])
    changed_classifier = deepcopy(classifier)
    changed_classifier.coef_[0, 0] += 1e-12
    assert fitted_object_state_sha256(changed_classifier) != fitted_object_state_sha256(classifier)
    changed_intercept = deepcopy(classifier)
    changed_intercept.intercept_[0] += 1e-12
    assert fitted_object_state_sha256(changed_intercept) != fitted_object_state_sha256(classifier)
    changed_classes = deepcopy(classifier)
    changed_classes.classes_ = np.asarray([1, 0])
    assert fitted_object_state_sha256(changed_classes) != fitted_object_state_sha256(classifier)

    scaler = StandardScaler().fit(features)
    changed_scaler = deepcopy(scaler)
    changed_scaler.mean_[0] += 1e-12
    assert fitted_object_state_sha256(changed_scaler) != fitted_object_state_sha256(scaler)


def test_canonical_fitted_state_ignores_execution_diagnostics_and_array_layout() -> None:
    features = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    classifier = LogisticRegression(max_iter=100).fit(features, [0, 0, 1, 1])
    baseline = fitted_object_state_sha256(classifier)

    changed = deepcopy(classifier)
    changed.n_iter_ = changed.n_iter_ + 17
    changed.verbose = 10
    changed.n_jobs = 8
    changed.coef_ = np.asfortranarray(changed.coef_)
    assert fitted_object_state_sha256(changed) == baseline

    endian_changed = deepcopy(classifier)
    endian_changed.coef_ = classifier.coef_.astype(">f8")
    endian_changed.intercept_ = classifier.intercept_.astype(">f8")
    assert fitted_object_state_sha256(endian_changed) == baseline

    dtype_changed = deepcopy(classifier)
    dtype_changed.coef_ = classifier.coef_.astype(np.float32)
    assert fitted_object_state_sha256(dtype_changed) != baseline

    scaler = StandardScaler().fit(features)
    scaler_baseline = fitted_object_state_sha256(scaler)
    scaler.copy = False
    scaler.n_samples_seen_ = np.asarray([999, 999])
    assert fitted_object_state_sha256(scaler) == scaler_baseline

    pipeline = Pipeline([("preprocess", StandardScaler()), ("classifier", classifier)]).fit(
        features, [0, 0, 1, 1]
    )
    pipeline_baseline = fitted_object_state_sha256(pipeline)
    reconstructed = deepcopy(pipeline)
    reconstructed.named_steps["preprocess"].copy = False
    reconstructed.named_steps["classifier"].n_iter_ += 5
    reconstructed.named_steps["classifier"].n_jobs = 4
    assert fitted_object_state_sha256(reconstructed) == pipeline_baseline


def test_column_transformer_and_encoder_identity_bind_inference_state() -> None:
    frame = pd.DataFrame({"numeric": [0.0, 1.0, 2.0, 3.0], "category": ["a", "b", "a", "c"]})
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), ["numeric"]),
            (
                "category",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2),
                ["category"],
            ),
        ]
    ).fit(frame)
    baseline = fitted_object_state_sha256(transformer)

    reordered = deepcopy(transformer)
    reordered.transformers_ = list(reversed(reordered.transformers_))
    assert fitted_object_state_sha256(reordered) != baseline

    changed_columns = deepcopy(transformer)
    changed_columns.transformers_[0] = (
        changed_columns.transformers_[0][0],
        changed_columns.transformers_[0][1],
        ["category"],
    )
    assert fitted_object_state_sha256(changed_columns) != baseline

    changed_categories = deepcopy(transformer)
    encoder = changed_categories.transformers_[1][1]
    encoder.categories_[0] = encoder.categories_[0].copy()
    encoder.categories_[0][0] = "changed"
    assert fitted_object_state_sha256(changed_categories) != baseline

    changed_mapping = deepcopy(transformer)
    encoder = changed_mapping.transformers_[1][1]
    encoder._default_to_infrequent_mappings[0] = encoder._default_to_infrequent_mappings[0].copy()
    encoder._default_to_infrequent_mappings[0][0] += 1
    assert fitted_object_state_sha256(changed_mapping) != baseline


def test_tensor_state_identity_binds_names_shapes_dtypes_and_values_not_storage() -> None:
    baseline = {
        "bias": torch.tensor([0.5], dtype=torch.float32),
        "weight": torch.tensor([[1.0, -0.0]], dtype=torch.float32),
    }
    digest = tensor_state_sha256(baseline)
    assert tensor_state_sha256(dict(reversed(list(baseline.items())))) == digest
    assert tensor_state_sha256({key: value.clone() for key, value in baseline.items()}) == digest
    assert tensor_state_sha256({**baseline, "renamed": baseline["weight"]}) != digest
    assert tensor_state_sha256({**baseline, "weight": baseline["weight"].reshape(2, 1)}) != digest
    assert tensor_state_sha256({**baseline, "weight": baseline["weight"].double()}) != digest
    assert tensor_state_sha256({**baseline, "weight": torch.tensor([[1.0, 0.0]])}) != digest


def test_lightgbm_fitted_state_binds_selected_predictor_only() -> None:
    features = np.arange(16, dtype=np.float64).reshape(-1, 1)
    targets = np.asarray([0] * 8 + [1] * 8)
    classifier = LGBMClassifier(
        n_estimators=5,
        min_child_samples=1,
        num_leaves=4,
        random_state=42,
        verbosity=-1,
        n_jobs=1,
    ).fit(features, targets)
    baseline = fitted_object_state_sha256(classifier, selected_iteration=5)

    operational = deepcopy(classifier)
    operational.set_params(n_jobs=8, verbosity=2)
    operational._evals_result = {"validation": {"auc": [0.1, 0.2]}}
    assert fitted_object_state_sha256(operational, selected_iteration=5) == baseline

    model_text = classifier.booster_.model_to_string(num_iteration=5)
    marker = "leaf_value="
    start = model_text.index(marker) + len(marker)
    end = model_text.index(" ", start)
    replacement = str(float(model_text[start:end]) + 0.125)
    changed = deepcopy(classifier)
    changed._Booster = Booster(model_str=model_text[:start] + replacement + model_text[end:])
    assert fitted_object_state_sha256(changed, selected_iteration=5) != baseline

    with pytest.raises(ValueError, match="canonically hashable"):
        fitted_object_state_sha256(classifier)


def test_model_package_identity_uses_package_scoped_config_and_frozen_threshold_state(
    tmp_path: Path,
) -> None:
    _, _, published = _publish(tmp_path)
    document = validate_published_model(published.package_directory)
    baseline = document["model_package_id"]
    changed = json.loads(json.dumps(document))
    changed["thresholds"] = {"youden_j": 0.9, "target_sensitivity": 0.8}
    changed["threshold_contract"]["sensitivity_target"] = 0.75
    assert model_package_id(changed) != baseline

    source_document = yaml.safe_load(
        Path("configs/rsna_metadata_logistic.yaml").read_text(encoding="utf-8")
    )
    source_document["evaluation"]["calibration_bins"] = 20
    changed_path = tmp_path / "calibration.yaml"
    changed_path.write_text(yaml.safe_dump(source_document, sort_keys=False), encoding="utf-8")
    original_config = load_experiment_config("configs/rsna_metadata_logistic.yaml")
    changed_config = load_experiment_config(changed_path)
    assert changed_config.config_semantic_sha256 != original_config.config_semantic_sha256
    assert package_scientific_config_payload(changed_config) == package_scientific_config_payload(
        original_config
    )
    evaluation_changed = json.loads(json.dumps(document))
    evaluation_changed["config_semantic_sha256"] = changed_config.config_semantic_sha256
    assert model_package_id(evaluation_changed) == baseline

    source_changed = json.loads(json.dumps(document))
    source_changed["config_source_sha256"] = "e" * 64
    assert model_package_id(source_changed) == baseline


def test_model_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    _, _, published = _publish(tmp_path)
    document = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    document["model_sha256"] = "tampered"
    published.manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_published_model(published.package_directory)


def test_model_manifest_rejects_archived_config_threshold_target_mismatch(
    tmp_path: Path,
) -> None:
    _, _, published = _publish(tmp_path)
    config_document = yaml.safe_load(published.config_path.read_text(encoding="utf-8"))
    config_document["evaluation"]["sensitivity_target"] = 0.85
    published.config_path.write_text(
        yaml.safe_dump(config_document, sort_keys=False), encoding="utf-8"
    )
    config = load_experiment_config(published.config_path)
    manifest = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    manifest["config_source_sha256"] = hashlib.sha256(
        published.config_path.read_bytes()
    ).hexdigest()
    manifest["config_semantic_sha256"] = config.config_semantic_sha256
    published.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="threshold contract differs from archived configuration"):
        validate_published_model(published.package_directory)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "reordered", "malformed"],
)
def test_model_manifest_rejects_invalid_input_contract(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, _, published = _publish(tmp_path)
    document = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    contract = document["input_contract"]
    features = contract["features"]
    if mutation == "missing":
        features.pop()
    elif mutation == "reordered":
        features[0], features[1] = features[1], features[0]
    else:
        contract["fitted_preprocessing_embedded"] = "yes"
    published.manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_published_model(published.package_directory)


_MISSING_SCHEMA_VERSION = object()


@pytest.mark.parametrize("value", [True, 1.0, "1", None, 0, _MISSING_SCHEMA_VERSION])
def test_model_manifest_requires_integer_schema_version_one(
    tmp_path: Path,
    value: object,
) -> None:
    _, _, published = _publish(tmp_path)
    document = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    if value is _MISSING_SCHEMA_VERSION:
        document.pop("model_package_schema_version")
    else:
        document["model_package_schema_version"] = value
        document["model_package_id"] = model_package_id(document)
    published.manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_published_model(published.package_directory)


def test_model_manifest_rejects_boolean_positive_class(tmp_path: Path) -> None:
    _, _, published = _publish(tmp_path)
    document = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    document["positive_class"] = True
    document["model_package_id"] = model_package_id(document)
    published.manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_published_model(published.package_directory)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("youden_j_policy_version", "unknown"),
        ("target_sensitivity_policy_version", "unknown"),
        ("sensitivity_target", True),
        ("positive_class", True),
    ],
)
def test_model_manifest_rejects_invalid_threshold_contract(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    _, _, published = _publish(tmp_path)
    document = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    document["threshold_contract"][field] = value
    document["model_package_id"] = model_package_id(document)
    published.manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_published_model(published.package_directory)


def test_conflicting_model_publication_retry_is_rejected(tmp_path: Path) -> None:
    _, _, published = _publish(tmp_path)
    with pytest.raises(ValueError):
        publish_model_package(
            model_root=tmp_path / "models" / "rsna",
            serialized_model_path=published.model_path,
            source_config_bytes=published.config_path.read_bytes(),
            manifest={
                **_manifest(),
                "bundle_id": "different",
                "config_source_sha256": hashlib.sha256(
                    published.config_path.read_bytes()
                ).hexdigest(),
            },
        )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("youden_j", True),
        ("youden_j", "0.5"),
        ("youden_j", float("nan")),
        ("youden_j", float("inf")),
        ("youden_j", -0.1),
        ("youden_j", 1.1),
        ("missing", None),
        ("extra", 0.5),
    ],
)
def test_model_manifest_rejects_invalid_thresholds(
    tmp_path: Path,
    mutation: str,
    value: object,
) -> None:
    _, _, published = _publish(tmp_path)
    document = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    thresholds = document["thresholds"]
    if mutation == "missing":
        thresholds.pop("youden_j")
    elif mutation == "extra":
        thresholds["unexpected"] = value
    else:
        thresholds[mutation] = value
    published.manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_published_model(published.package_directory)
