"""Semantic package configuration and fitted-state hashing."""

from __future__ import annotations

import hashlib
import struct
import unicodedata
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from beyondcxr.training.config import ExperimentConfig, normalized_scientific_config
from beyondcxr.utils.semantic_identity import canonical_json_bytes, semantic_sha256


def package_scientific_config_payload(config: ExperimentConfig) -> dict[str, object]:
    """Project validated configuration meaning owned by a fitted model package."""
    return package_scientific_config_from_normalized(normalized_scientific_config(config))


def package_scientific_config_from_normalized(
    normalized: Mapping[str, object],
) -> dict[str, object]:
    """Project package-owned science from an execution's normalized configuration."""
    required = {
        "config_schema_version",
        "dataset",
        "task",
        "family",
        "preprocessing",
        "training",
        "evaluation",
    }
    if not isinstance(normalized, Mapping) or set(normalized) != required:
        raise ValueError("Normalized scientific configuration is invalid")
    dataset = normalized["dataset"]
    task = normalized["task"]
    family = normalized["family"]
    training = normalized["training"]
    preprocessing = normalized["preprocessing"]
    if (
        not isinstance(dataset, Mapping)
        or set(dataset) != {"dataset_id", "bundle_id", "cv_assignment_id"}
        or not isinstance(task, Mapping)
        or set(task) != {"task_id", "label_policy_version"}
        or not isinstance(family, Mapping)
        or set(family) != {"family_id", "modalities", "parameters"}
        or not isinstance(family["modalities"], list)
        or not isinstance(family["parameters"], Mapping)
        or not isinstance(preprocessing, Mapping)
        or not isinstance(training, Mapping)
        or set(training) != {"selection_metric", "parameters", "loader", "augmentation"}
        or any(
            not isinstance(training[field], Mapping)
            for field in ("parameters", "loader", "augmentation")
        )
    ):
        raise ValueError("Normalized scientific configuration is invalid")
    return {
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "bundle_id": dataset["bundle_id"],
            "cv_assignment_id": dataset["cv_assignment_id"],
        },
        "task": dict(task),
        "family": {
            "family_id": family["family_id"],
            "modalities": list(family["modalities"]),
            "parameters": dict(family["parameters"]),
        },
        "preprocessing": dict(preprocessing),
        "training": {
            "selection_metric": training["selection_metric"],
            "parameters": dict(training["parameters"]),
            "loader": dict(training["loader"]),
            "augmentation": dict(training["augmentation"]),
        },
    }


def pretrained_weight_semantic_identity(weight: Mapping[str, object]) -> dict[str, object]:
    """Project scientific initialization lineage from a full weight fingerprint."""
    required = {"declared_name", "stable_identifier", "cache_filename", "byte_size", "sha256"}
    if set(weight) != required:
        raise ValueError("Pretrained weight fingerprint has an unexpected field set")
    declared_name = weight["declared_name"]
    stable_identifier = weight["stable_identifier"]
    digest = weight["sha256"]
    if (
        not isinstance(declared_name, str)
        or not declared_name
        or not isinstance(stable_identifier, str)
        or not stable_identifier
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Pretrained weight scientific identity is invalid")
    return {
        "declared_name": declared_name,
        "stable_identifier": stable_identifier,
        "sha256": digest,
    }


def fitted_object_state_sha256(value: object, *, selected_iteration: int | None = None) -> str:
    """Hash explicitly supported fitted scientific state directly with SHA-256."""
    try:
        payload = _fitted_state(value, selected_iteration=selected_iteration)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Fitted object state is not canonically hashable") from exc
    return semantic_sha256("fitted-object-state", 1, payload)


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash ordered tensor metadata and exact contiguous CPU logical bytes."""
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Tensor state must be a non-empty mapping")
    normalized: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if not isinstance(key, str) or not key or not isinstance(tensor, torch.Tensor):
            raise ValueError("Tensor state entries are invalid")
        name = unicodedata.normalize("NFC", key)
        if name in normalized:
            raise ValueError("Tensor state names collide after NFC normalization")
        normalized[name] = tensor
    digest = hashlib.sha256()
    _update_framed_hash(
        digest,
        canonical_json_bytes(
            {
                "domain": "tensor-state",
                "schema_version": 1,
                "entry_count": len(state),
            }
        ),
    )
    for key in sorted(normalized):
        tensor = normalized[key]
        value = tensor.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError("Tensor state contains non-finite values")
        array = value.numpy()
        raw = array.tobytes(order="C")
        _update_framed_hash(
            digest,
            canonical_json_bytes(
                {
                    "name": key,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "byte_length": len(raw),
                }
            ),
        )
        _update_framed_hash(digest, raw)
    return digest.hexdigest()


def _update_framed_hash(digest: Any, payload: bytes) -> None:
    """Add one unambiguous length-prefixed byte field to a SHA-256 state."""
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _fitted_state(value: object, *, selected_iteration: int | None = None) -> dict[str, Any]:
    from beyondcxr.data.rsna_metadata_preprocess import RsnaMetadataFeatures
    from beyondcxr.data.symile_preprocess import SymileLabEcdfTransformer

    if isinstance(value, Pipeline):
        return {
            "type": "sklearn.pipeline.Pipeline",
            "steps": [
                [name, _fitted_state(step, selected_iteration=selected_iteration)]
                for name, step in value.steps
            ],
        }
    if isinstance(value, LogisticRegression):
        return {
            "type": "sklearn.linear_model.LogisticRegression",
            "classes": _array(value.classes_),
            "coefficients": _array(value.coef_),
            "intercept": _array(value.intercept_),
            "features_in": _optional_array(getattr(value, "feature_names_in_", None)),
            "n_features_in": int(value.n_features_in_),
        }
    if isinstance(value, LGBMClassifier):
        return _lightgbm_state(value, selected_iteration)
    if isinstance(value, SymileLabEcdfTransformer):
        return {
            "type": "beyondcxr.data.SymileLabEcdfTransformer",
            "sorted_observed_values": [_array(item) for item in value.sorted_observed_values_],
            "missing_replacements": _array(value.missing_replacements_),
            "features_in": _array(value.feature_names_in_),
            "n_features_in": int(value.n_features_in_),
        }
    if isinstance(value, RsnaMetadataFeatures):
        return {
            "type": "beyondcxr.data.RsnaMetadataFeatures",
            "features_in": _array(value.feature_names_in_),
        }
    if isinstance(value, ColumnTransformer):
        return {
            "type": "sklearn.compose.ColumnTransformer",
            "transformers": [
                [
                    name,
                    _fitted_state(transformer, selected_iteration=selected_iteration),
                    _tagged(columns),
                ]
                for name, transformer, columns in value.transformers_
            ],
            "features_in": _array(value.feature_names_in_),
            "n_features_in": int(value.n_features_in_),
        }
    if isinstance(value, SimpleImputer):
        indicator = getattr(value, "indicator_", None)
        return {
            "type": "sklearn.impute.SimpleImputer",
            "statistics": _array(value.statistics_),
            "features_in": _optional_array(getattr(value, "feature_names_in_", None)),
            "n_features_in": int(value.n_features_in_),
            "indicator_features": (None if indicator is None else _array(indicator.features_)),
        }
    if isinstance(value, StandardScaler):
        return {
            "type": "sklearn.preprocessing.StandardScaler",
            "mean": _optional_array(getattr(value, "mean_", None)),
            "scale": _optional_array(getattr(value, "scale_", None)),
            "features_in": _optional_array(getattr(value, "feature_names_in_", None)),
            "n_features_in": int(value.n_features_in_),
        }
    if isinstance(value, OneHotEncoder):
        return {
            "type": "sklearn.preprocessing.OneHotEncoder",
            "categories": [_array(item) for item in value.categories_],
            "drop_indices": _optional_array(getattr(value, "drop_idx_", None)),
            "infrequent_indices": _optional_array_list(getattr(value, "_infrequent_indices", None)),
            "infrequent_mappings": _optional_array_list(
                getattr(value, "_default_to_infrequent_mappings", None)
            ),
            "features_in": _optional_array(getattr(value, "feature_names_in_", None)),
            "n_features_in": int(value.n_features_in_),
        }
    if isinstance(value, FunctionTransformer):
        if value.func is not None or value.inverse_func is not None:
            raise TypeError("Callable FunctionTransformer state is not supported")
        return {
            "type": "sklearn.preprocessing.FunctionTransformer",
            "features_in": _optional_array(getattr(value, "feature_names_in_", None)),
            "n_features_in": (
                None
                if getattr(value, "n_features_in_", None) is None
                else int(value.n_features_in_)
            ),
        }
    if value == "passthrough" or value == "drop":
        return {"type": "sklearn.column-sentinel", "value": value}
    type_name = f"{type(value).__module__}.{type(value).__name__}"
    raise TypeError(f"Unsupported fitted-state type: {type_name}")


def _array(value: object) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError("Fitted state contains non-finite numeric values")
    if array.dtype.kind == "S":
        raise TypeError("Byte-string fitted-state arrays are not supported")
    if array.dtype.kind in "OU":
        data: object = [_tagged(item) for item in array.reshape(-1).tolist()]
        dtype = _canonical_dtype(array.dtype)
    elif array.dtype.kind in "biuf":
        canonical_dtype = array.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
        data = canonical.tobytes(order="C").hex()
        dtype = _canonical_dtype(array.dtype)
    else:
        raise TypeError(f"Unsupported fitted-state array dtype: {array.dtype}")
    return {"dtype": dtype, "shape": list(array.shape), "data": data}


def _optional_array(value: object) -> dict[str, Any] | None:
    return None if value is None else _array(value)


def _optional_array_list(value: object) -> list[dict[str, Any] | None] | None:
    if value is None:
        return None
    return [None if item is None else _array(item) for item in value]


def _tagged_mapping(value: Mapping[object, object]) -> list[list[object]]:
    tagged = [(_tagged(key), _tagged(item)) for key, item in value.items()]
    tagged.sort(key=lambda pair: canonical_json_bytes(pair[0]))
    return [[key, item] for key, item in tagged]


def _tagged(value: object) -> object:
    if value is None:
        return {"type": "none"}
    if isinstance(value, (bool, np.bool_)):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, (int, np.integer)):
        return {"type": "int", "value": int(value)}
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("Fitted state contains a non-finite scalar")
        return {"type": "float", "value": number}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, np.dtype | type):
        return {"type": "dtype_or_type", "value": str(value)}
    if isinstance(value, slice):
        return {
            "type": "slice",
            "start": _tagged(value.start),
            "stop": _tagged(value.stop),
            "step": _tagged(value.step),
        }
    if isinstance(value, Mapping):
        return {"type": "mapping", "items": _tagged_mapping(value)}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "items": [_tagged(item) for item in value]}
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "value": _array(value)}
    raise TypeError(f"Unsupported fitted-state scalar: {type(value).__name__}")


def _canonical_dtype(dtype: np.dtype[Any]) -> str:
    if dtype.kind == "b":
        return "bool"
    if dtype.kind in "iuf":
        return f"{dtype.kind}{dtype.itemsize}"
    if dtype.kind == "U":
        return "unicode"
    if dtype.kind == "O":
        return "object"
    raise TypeError(f"Unsupported fitted-state dtype: {dtype}")


def _lightgbm_state(value: LGBMClassifier, selected_iteration: int | None) -> dict[str, Any]:
    if (
        isinstance(selected_iteration, bool)
        or not isinstance(selected_iteration, int)
        or selected_iteration <= 0
    ):
        raise ValueError("LightGBM fitted-state hashing requires a positive selected iteration")
    model = value.booster_.dump_model(num_iteration=selected_iteration)
    pandas_categorical = model.get("pandas_categorical")
    if pandas_categorical not in (None, "null", []):
        raise ValueError("Categorical LightGBM fitted state is not supported")
    return {
        "type": "lightgbm.LGBMClassifier",
        "classes": _array(value.classes_),
        "n_features_in": int(value.n_features_in_),
        "feature_names": list(value.feature_name_),
        "num_class": int(model["num_class"]),
        "num_tree_per_iteration": int(model["num_tree_per_iteration"]),
        "average_output": bool(model["average_output"]),
        "objective": str(model["objective"]),
        "trees": [_lightgbm_tree(item) for item in model["tree_info"]],
    }


def _lightgbm_tree(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "shrinkage": _finite_float(value["shrinkage"]),
        "structure": _lightgbm_node(value["tree_structure"]),
    }


def _lightgbm_node(value: Mapping[str, Any]) -> dict[str, Any]:
    if "leaf_value" in value:
        result: dict[str, Any] = {"leaf_value": _finite_float(value["leaf_value"])}
        for field in ("leaf_const", "leaf_features", "leaf_coeff"):
            if field in value:
                result[field] = _tagged(value[field])
        return result
    return {
        "split_feature": int(value["split_feature"]),
        "threshold": _tagged(value["threshold"]),
        "decision_type": str(value["decision_type"]),
        "default_left": bool(value["default_left"]),
        "missing_type": str(value["missing_type"]),
        "left_child": _lightgbm_node(value["left_child"]),
        "right_child": _lightgbm_node(value["right_child"]),
    }


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value):
        raise ValueError("LightGBM predictor state contains a non-finite value")
    return float(value)
