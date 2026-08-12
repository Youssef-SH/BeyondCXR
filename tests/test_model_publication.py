from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from radfusion.data.tabular_preprocess import metadata_input_contract
from radfusion.utils.model_publication import (
    MODEL_PACKAGE_ID_PREFIX,
    REQUIRED_MANIFEST_FIELDS,
    model_package_id,
    publish_model_run,
    threshold_contract,
    validate_published_model,
)
from radfusion.utils.skops_io import load_skops, save_skops

_SHA256 = "a" * 64


def _manifest() -> dict[str, object]:
    return {
        "bundle_id": "bundle-test",
        "split_assignment_id": "assignment-test",
        "task": "pneumonia",
        "positive_class": 1,
        "model": "metadata_logistic",
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
    model = LogisticRegression().fit(features, [0, 0, 1, 1])
    serialized = save_skops(model, tmp_path / "source.skops")
    config = tmp_path / "source.yaml"
    config.write_text("config_version: 1\n", encoding="utf-8")
    manifest = {
        **_manifest(),
        "config_source_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }
    published = publish_model_run(
        model_root=tmp_path / "models" / "rsna",
        mlflow_run_id="run-test",
        serialized_model_path=serialized,
        source_config_bytes=config.read_bytes(),
        manifest=manifest,
    )
    return features, model, published


def test_model_package_is_compact_run_qualified_and_loadable(tmp_path: Path) -> None:
    features, model, published = _publish(tmp_path)

    assert published.run_directory == tmp_path / "models" / "rsna" / "runs" / "run-test"
    assert {path.name for path in published.run_directory.iterdir()} == {
        "model.skops",
        "resolved_config.yaml",
        "model_manifest.json",
    }
    document = validate_published_model(published.run_directory)
    assert set(document) == REQUIRED_MANIFEST_FIELDS
    assert document["model_package_id"].startswith(MODEL_PACKAGE_ID_PREFIX)
    assert document["model_package_id"] == model_package_id(document)
    np.testing.assert_array_equal(
        load_skops(published.model_path).predict_proba(features),
        model.predict_proba(features),
    )


def test_model_package_preserves_exact_serialized_and_config_bytes(tmp_path: Path) -> None:
    _, _, published = _publish(tmp_path)
    assert published.model_path.read_bytes() == (tmp_path / "source.skops").read_bytes()
    assert published.config_path.read_bytes() == (tmp_path / "source.yaml").read_bytes()


def test_model_package_id_is_deterministic_and_changes_with_semantic_content(
    tmp_path: Path,
) -> None:
    _, _, published = _publish(tmp_path)
    document = validate_published_model(published.run_directory)

    assert model_package_id(document) == model_package_id(dict(reversed(document.items())))
    changed = {**document, "seed": document["seed"] + 1}
    changed.pop("model_package_id")
    assert model_package_id(changed) != document["model_package_id"]


def test_model_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    _, _, published = _publish(tmp_path)
    document = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    document["model_sha256"] = "tampered"
    published.manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_published_model(published.run_directory)


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
    document["model_package_id"] = model_package_id(document)
    published.manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_published_model(published.run_directory)


@pytest.mark.parametrize(
    "field",
    ["model_package_schema_version", "positive_class"],
)
def test_model_manifest_rejects_boolean_integer_fields(
    tmp_path: Path,
    field: str,
) -> None:
    _, _, published = _publish(tmp_path)
    document = json.loads(published.manifest_path.read_text(encoding="utf-8"))
    document[field] = True
    document["model_package_id"] = model_package_id(document)
    published.manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_published_model(published.run_directory)


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
        validate_published_model(published.run_directory)


@pytest.mark.parametrize("run_id", ["../run", "a/b", ".", ".."])
def test_model_publication_rejects_unsafe_run_ids(tmp_path: Path, run_id: str) -> None:
    model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
    serialized = save_skops(model, tmp_path / "source.skops")
    config = tmp_path / "source.yaml"
    config.write_text("test\n", encoding="utf-8")
    with pytest.raises(ValueError):
        publish_model_run(
            model_root=tmp_path / "models",
            mlflow_run_id=run_id,
            serialized_model_path=serialized,
            source_config_bytes=config.read_bytes(),
            manifest={
                **_manifest(),
                "config_source_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            },
        )


def test_conflicting_model_publication_retry_is_rejected(tmp_path: Path) -> None:
    _, _, published = _publish(tmp_path)
    with pytest.raises(FileExistsError):
        publish_model_run(
            model_root=tmp_path / "models" / "rsna",
            mlflow_run_id="run-test",
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
    if isinstance(value, float) and not np.isfinite(value):
        identity = dict(document)
        identity.pop("model_package_id")
        with pytest.raises(ValueError):
            model_package_id(identity)
    else:
        document["model_package_id"] = model_package_id(document)
    published.manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_published_model(published.run_directory)
