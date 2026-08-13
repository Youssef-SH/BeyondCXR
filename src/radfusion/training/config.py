"""Load the single strict version-1 scientific experiment configuration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from radfusion.data.bundle_contract import valid_bundle_id
from radfusion.data.cxr_transforms import CXR_TRANSFORM_POLICY_VERSION, STANDARD_CXR_IMAGE_SIZE
from radfusion.data.rsna_metadata_preprocess import METADATA_INPUT_POLICY_VERSION
from radfusion.data.symile_preprocess import LAB_ECDF_POLICY_VERSION


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the version-1 grammar."""


class _StrictSafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _StrictSafeLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)

MODEL_RANDOMNESS_KEYS = frozenset(
    {
        "random_state",
        "seed",
        "bagging_seed",
        "feature_fraction_seed",
        "data_random_seed",
        "drop_seed",
        "extra_seed",
    }
)

FAMILY_MODALITIES = MappingProxyType(
    {
        ("rsna", "metadata_logistic"): ("metadata",),
        ("rsna", "metadata_lightgbm"): ("metadata",),
        ("rsna", "cxr_densenet"): ("cxr",),
        ("rsna", "cxr_metadata_concat"): ("cxr", "metadata"),
        ("symile", "labs_logistic"): ("labs",),
        ("symile", "labs_lightgbm"): ("labs",),
        ("symile", "cxr_densenet"): ("cxr",),
        ("symile", "cxr_labs_concat"): ("cxr", "labs"),
        ("symile", "cxr_labs_gated"): ("cxr", "labs"),
        ("symile", "cxr_labs_gated_no_observedness"): ("cxr", "labs"),
    }
)

_FAMILY_PARAMETER_FIELDS = MappingProxyType(
    {
        "metadata_logistic": frozenset(),
        "metadata_lightgbm": frozenset(
            {
                "objective",
                "num_leaves",
                "min_child_samples",
            }
        ),
        "labs_logistic": frozenset(),
        "labs_lightgbm": frozenset(
            {
                "objective",
                "num_leaves",
                "min_child_samples",
            }
        ),
        "cxr_metadata_concat": frozenset(
            {
                "encoder_name",
                "weights",
                "image_size",
                "embedding_dimension",
                "image_projection_dimension",
                "structured_hidden_dimension",
                "structured_projection_dimension",
                "fusion_hidden_dimension",
                "dropout",
            }
        ),
        "cxr_labs_concat": frozenset(
            {
                "encoder_name",
                "weights",
                "image_size",
                "embedding_dimension",
                "lab_input_dimension",
                "image_projection_dimension",
                "lab_hidden_dimension",
                "lab_projection_dimension",
                "fusion_hidden_dimension",
                "dropout",
            }
        ),
        "cxr_labs_gated": frozenset(
            {
                "encoder_name",
                "weights",
                "image_size",
                "embedding_dimension",
                "lab_input_dimension",
                "lab_hidden_dimension",
                "lab_core_dimension",
                "latent_dimension",
                "observedness_dimension",
                "gate_hidden_dimension",
                "classifier_hidden_dimension",
                "modality_count",
                "dropout",
                "use_observedness",
            }
        ),
        "cxr_labs_gated_no_observedness": frozenset(
            {
                "encoder_name",
                "weights",
                "image_size",
                "embedding_dimension",
                "lab_input_dimension",
                "lab_hidden_dimension",
                "lab_core_dimension",
                "latent_dimension",
                "observedness_dimension",
                "gate_hidden_dimension",
                "classifier_hidden_dimension",
                "modality_count",
                "dropout",
                "use_observedness",
            }
        ),
    }
)

_TABULAR_TRAINING_FIELDS = MappingProxyType(
    {
        "metadata_logistic": frozenset({"l1_ratio", "solver", "C", "max_iter", "class_weight"}),
        "labs_logistic": frozenset({"l1_ratio", "solver", "C", "max_iter", "class_weight"}),
        "metadata_lightgbm": frozenset(
            {
                "n_estimators",
                "learning_rate",
                "subsample",
                "subsample_freq",
                "colsample_bytree",
                "reg_lambda",
                "class_weighting",
                "early_stopping_rounds",
            }
        ),
        "labs_lightgbm": frozenset(
            {
                "n_estimators",
                "learning_rate",
                "subsample",
                "subsample_freq",
                "colsample_bytree",
                "reg_lambda",
                "class_weight",
                "early_stopping_rounds",
            }
        ),
    }
)

_NEURAL_FAMILIES = frozenset(
    {
        "cxr_densenet",
        "cxr_metadata_concat",
        "cxr_labs_concat",
        "cxr_labs_gated",
        "cxr_labs_gated_no_observedness",
    }
)
_SYMILE_FAMILIES = frozenset(family for dataset, family in FAMILY_MODALITIES if dataset == "symile")


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable scientific dataset and integrity witnesses."""

    dataset_id: str
    bundle_id: str
    bundle_manifest_sha256: str
    split_assignment_id: str
    cv_assignment_id: str | None


@dataclass(frozen=True)
class TaskConfig:
    """Prediction task and label-policy identity."""

    task_id: str
    label_policy_version: str


@dataclass(frozen=True)
class FamilyConfig:
    """Scientific family, modalities, and topology parameters."""

    family_id: str
    modalities: tuple[str, ...]
    parameters: MappingProxyType[str, Any]


@dataclass(frozen=True)
class TrainingConfig:
    """Scientific fitting, loader, augmentation, and selection policies."""

    selection_metric: str
    parameters: MappingProxyType[str, Any]
    loader: MappingProxyType[str, Any]
    augmentation: MappingProxyType[str, Any]


@dataclass(frozen=True)
class EvaluationConfig:
    """Scientific evaluation policy, when applicable."""

    parameters: MappingProxyType[str, Any]

    @property
    def sensitivity_target(self) -> float:
        return float(self.parameters["sensitivity_target"])

    @property
    def calibration_bins(self) -> int:
        return int(self.parameters["calibration_bins"])


@dataclass(frozen=True)
class RuntimeConfig:
    """Operational paths, hardware choice, and execution seed."""

    manifest_directory: Path
    source_root: Path | None
    model_directory: Path
    report_directory: Path
    private_output_directory: Path
    experiment_name: str
    device: str
    num_workers: int
    pin_memory_policy: str
    seed: int | None = None


@dataclass(frozen=True)
class NeuralConfig:
    """Typed derived view of neural scientific training policy."""

    batch_size: int
    mixed_precision: bool
    rotation_degrees: float
    translation_fraction: float
    brightness_jitter: float
    contrast_jitter: float
    optimizer: str
    warmup_epochs: int
    fine_tune_epochs: int
    warmup_head_learning_rate: float
    encoder_learning_rate: float
    head_learning_rate: float
    weight_decay: float
    scheduler_factor: float
    scheduler_patience: int
    scheduler_min_learning_rate: float
    gradient_clip_norm: float
    early_stopping_patience: int
    early_stopping_min_delta: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated science plus separately owned runtime coordinates."""

    config_version: int
    dataset: DatasetConfig
    task: TaskConfig
    family: FamilyConfig
    preprocessing: MappingProxyType[str, Any]
    training: TrainingConfig
    evaluation: EvaluationConfig | None
    neural: NeuralConfig | None
    runtime: RuntimeConfig
    source_path: Path
    source_bytes: bytes
    config_source_sha256: str
    config_semantic_sha256: str


SYMILE_M5_FAMILIES = tuple(sorted(_SYMILE_FAMILIES))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load one strict canonical version-1 experiment YAML."""
    source = Path(path)
    source_bytes, root = _read_yaml(source)
    _keys(
        root,
        required={"config_version", "dataset", "task", "family", "training"},
        optional={"preprocessing", "evaluation"},
        context="config",
    )
    version = _integer(root["config_version"], "config_version")
    if version != 1:
        raise ConfigError(f"Unsupported config_version: {version}")
    dataset = _dataset_config(root["dataset"])
    task = _task_config(root["task"], dataset.dataset_id)
    family = _family_config(root["family"], dataset.dataset_id)
    preprocessing = _preprocessing_config(root.get("preprocessing", {}), dataset, family)
    training = _training_config(root["training"], dataset.dataset_id, family.family_id)
    evaluation = _evaluation_config(root["evaluation"]) if "evaluation" in root else None
    _validate_section_applicability(dataset, family, training, evaluation)
    runtime = _default_runtime(dataset.dataset_id)
    neural = _neural_config(training) if family.family_id in _NEURAL_FAMILIES else None
    semantic = _semantic_payload(
        version, dataset, task, family, preprocessing, training, evaluation
    )
    return ExperimentConfig(
        version,
        dataset,
        task,
        family,
        preprocessing,
        training,
        evaluation,
        neural,
        runtime,
        source,
        source_bytes,
        hashlib.sha256(source_bytes).hexdigest(),
        _canonical_sha256(semantic),
    )


def load_symile_development_config(path: str | Path) -> ExperimentConfig:
    """Load a canonical Symile development-family configuration."""
    config = load_experiment_config(path)
    if config.dataset.dataset_id != "symile":
        raise ConfigError("Symile development requires dataset_id='symile'")
    return config


def with_runtime(
    config: ExperimentConfig,
    *,
    seed: int | None = None,
    manifest_directory: str | Path | None = None,
    source_root: str | Path | None = None,
    model_directory: str | Path | None = None,
    report_directory: str | Path | None = None,
    private_output_directory: str | Path | None = None,
    experiment_name: str | None = None,
    device: str | None = None,
    num_workers: int | None = None,
    pin_memory_policy: str | None = None,
) -> ExperimentConfig:
    """Attach operational destinations and a seed without changing config identity."""
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1
    ):
        raise ConfigError("Runtime seed must be an integer in [0, 2**31 - 1]")
    if num_workers is not None and (
        isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0
    ):
        raise ConfigError("Runtime num_workers must be a non-negative integer")
    if pin_memory_policy is not None and pin_memory_policy not in {"auto", "enabled", "disabled"}:
        raise ConfigError("Runtime pin_memory_policy is unsupported")
    runtime = replace(
        config.runtime,
        seed=seed if seed is not None else config.runtime.seed,
        manifest_directory=(
            Path(manifest_directory)
            if manifest_directory is not None
            else config.runtime.manifest_directory
        ),
        source_root=Path(source_root) if source_root is not None else config.runtime.source_root,
        model_directory=(
            Path(model_directory) if model_directory is not None else config.runtime.model_directory
        ),
        report_directory=(
            Path(report_directory)
            if report_directory is not None
            else config.runtime.report_directory
        ),
        private_output_directory=(
            Path(private_output_directory)
            if private_output_directory is not None
            else config.runtime.private_output_directory
        ),
        experiment_name=experiment_name or config.runtime.experiment_name,
        device=device or config.runtime.device,
        num_workers=num_workers if num_workers is not None else config.runtime.num_workers,
        pin_memory_policy=pin_memory_policy or config.runtime.pin_memory_policy,
    )
    return replace(config, runtime=runtime)


def require_runtime_seed(config: ExperimentConfig) -> int:
    """Return the explicit execution seed or fail before scientific work."""
    if config.runtime.seed is None:
        raise ConfigError("An explicit runtime seed is required")
    return config.runtime.seed


def _dataset_config(value: object) -> DatasetConfig:
    data = _mapping(value, "dataset")
    _keys(
        data,
        required={
            "dataset_id",
            "bundle_id",
            "bundle_manifest_sha256",
            "split_assignment_id",
        },
        optional={"cv_assignment_id"},
        context="dataset",
    )
    dataset_id = _choice(data["dataset_id"], {"rsna", "symile"}, "dataset.dataset_id")
    bundle_id = _text(data["bundle_id"], "dataset.bundle_id")
    if not valid_bundle_id(bundle_id):
        raise ConfigError("dataset.bundle_id is invalid")
    manifest_sha = _sha256(data["bundle_manifest_sha256"], "dataset.bundle_manifest_sha256")
    split_id = _identity(
        data["split_assignment_id"], "split-assignment-", "dataset.split_assignment_id"
    )
    cv_id = None
    if "cv_assignment_id" in data:
        cv_id = _identity(data["cv_assignment_id"], "cv-assignment-", "dataset.cv_assignment_id")
    if dataset_id == "symile" and cv_id is None:
        raise ConfigError("Symile configs require dataset.cv_assignment_id")
    if dataset_id == "rsna" and cv_id is not None:
        raise ConfigError("RSNA configs do not accept dataset.cv_assignment_id")
    return DatasetConfig(dataset_id, bundle_id, manifest_sha, split_id, cv_id)


def _task_config(value: object, dataset_id: str) -> TaskConfig:
    data = _mapping(value, "task")
    _keys(data, required={"task_id", "label_policy_version"}, context="task")
    task = TaskConfig(
        _text(data["task_id"], "task.task_id"),
        _text(data["label_policy_version"], "task.label_policy_version"),
    )
    supported = {
        "rsna": ("pneumonia", "rsna-stage-2-target-v1"),
        "symile": ("pneumonia_strict", "symile-pneumonia-strict-v1"),
    }
    if (task.task_id, task.label_policy_version) != supported[dataset_id]:
        raise ConfigError("Task and label policy are incompatible with dataset")
    return task


def _family_config(value: object, dataset_id: str) -> FamilyConfig:
    data = _mapping(value, "family")
    _keys(data, required={"family_id", "modalities", "parameters"}, context="family")
    family_id = _text(data["family_id"], "family.family_id")
    expected_modalities = FAMILY_MODALITIES.get((dataset_id, family_id))
    if expected_modalities is None:
        raise ConfigError("family.family_id is unsupported for dataset")
    raw_modalities = data["modalities"]
    if not isinstance(raw_modalities, list) or not raw_modalities:
        raise ConfigError("family.modalities must be a non-empty ordered list")
    modalities = tuple(_text(item, "family.modalities") for item in raw_modalities)
    if modalities != expected_modalities:
        raise ConfigError("family.modalities are incompatible with family.family_id")
    parameters = _mapping(data["parameters"], "family.parameters")
    expected_fields = _FAMILY_PARAMETER_FIELDS.get(family_id)
    if family_id == "cxr_densenet":
        expected_fields = frozenset(
            {"encoder_name", "weights", "image_size", "embedding_dimension"}
        )
    if expected_fields is None or set(parameters) != expected_fields:
        raise ConfigError("family.parameters has missing or unknown fields")
    _validate_family_parameters(family_id, parameters)
    return FamilyConfig(family_id, modalities, MappingProxyType(dict(parameters)))


def _preprocessing_config(
    value: object, dataset: DatasetConfig, family: FamilyConfig
) -> MappingProxyType[str, Any]:
    data = _mapping(value, "preprocessing")
    required: set[str] = set()
    if "metadata" in family.modalities:
        required.add("metadata_policy")
    if "cxr" in family.modalities:
        required.add("cxr_transform_policy")
    if "labs" in family.modalities:
        required.add("lab_policy")
    _keys(data, required=required, context="preprocessing")
    expected = {
        "metadata_policy": METADATA_INPUT_POLICY_VERSION,
        "cxr_transform_policy": CXR_TRANSFORM_POLICY_VERSION,
        "lab_policy": LAB_ECDF_POLICY_VERSION,
    }
    if any(_text(data[key], f"preprocessing.{key}") != expected[key] for key in required):
        raise ConfigError("Preprocessing policy is unsupported by the configured implementation")
    if dataset.dataset_id == "rsna" and "lab_policy" in data:
        raise ConfigError("RSNA configurations do not accept Symile laboratory preprocessing")
    return MappingProxyType(dict(data))


def _training_config(value: object, dataset_id: str, family_id: str) -> TrainingConfig:
    data = _mapping(value, "training")
    _keys(
        data,
        required={"selection_metric", "parameters"},
        optional={"loader", "augmentation"},
        context="training",
    )
    selection = _choice(
        data["selection_metric"],
        {"none", "average_precision", "roc_auc"},
        "training.selection_metric",
    )
    parameters = _mapping(data["parameters"], "training.parameters")
    loader = _mapping(data.get("loader", {}), "training.loader")
    augmentation = _mapping(data.get("augmentation", {}), "training.augmentation")
    if family_id in _NEURAL_FAMILIES:
        required = (
            {"class_weighting", "fine_tune_scope"}
            if dataset_id == "rsna"
            else {
                "pos_weight",
                "fine_tune_scope",
            }
        )
        if not required <= set(parameters):
            raise ConfigError("Neural training weighting or fine-tuning policy is missing")
        _validate_neural_training(parameters, loader, augmentation)
        _validate_scientific_training_policy(dataset_id, family_id, parameters)
    else:
        expected = _TABULAR_TRAINING_FIELDS.get(family_id)
        if expected is None or set(parameters) != expected or loader or augmentation:
            raise ConfigError("Tabular training policy field set is invalid")
        _validate_tabular_training(dataset_id, family_id, parameters)
    return TrainingConfig(
        selection,
        MappingProxyType(dict(parameters)),
        MappingProxyType(dict(loader)),
        MappingProxyType(dict(augmentation)),
    )


def _evaluation_config(value: object) -> EvaluationConfig:
    data = _mapping(value, "evaluation")
    required = {"sensitivity_target", "calibration_bins"}
    _keys(data, required=required, context="evaluation")
    target = _number(data["sensitivity_target"], "evaluation.sensitivity_target")
    bins = _integer(data["calibration_bins"], "evaluation.calibration_bins")
    if not 0 < target <= 1 or bins <= 1:
        raise ConfigError("evaluation values are outside supported ranges")
    return EvaluationConfig(MappingProxyType(dict(data)))


def _validate_section_applicability(
    dataset: DatasetConfig,
    family: FamilyConfig,
    training: TrainingConfig,
    evaluation: EvaluationConfig | None,
) -> None:
    if dataset.dataset_id == "rsna" and evaluation is None:
        raise ConfigError("RSNA configs require evaluation policy")
    if dataset.dataset_id == "symile" and evaluation is not None:
        raise ConfigError("Symile development configs do not accept held-out evaluation policy")
    expected = {
        "metadata_logistic": "none",
        "metadata_lightgbm": "average_precision",
        "cxr_metadata_concat": "average_precision",
        "labs_logistic": "none",
        "labs_lightgbm": "roc_auc",
        "cxr_labs_concat": "roc_auc",
        "cxr_labs_gated": "roc_auc",
        "cxr_labs_gated_no_observedness": "roc_auc",
    }.get(family.family_id)
    if family.family_id == "cxr_densenet":
        expected = "average_precision" if dataset.dataset_id == "rsna" else "roc_auc"
    if training.selection_metric != expected:
        raise ConfigError("training.selection_metric is incompatible with family and dataset")


def _validate_family_parameters(family_id: str, values: dict[str, Any]) -> None:
    if any(key in MODEL_RANDOMNESS_KEYS for key in values):
        raise ConfigError("Execution randomness belongs to runtime coordinates")
    integer_fields = {
        "image_size",
        "num_leaves",
        "min_child_samples",
        "modality_count",
    }
    for key, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigError(f"family.parameters.{key} must be finite")
        if key.endswith("dimension") or key in integer_fields:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"family.parameters.{key} must be a positive integer")
    if family_id.startswith("cxr") and (
        values.get("encoder_name") != "densenet121"
        or values.get("weights") != "densenet121-res224-chex"
        or values.get("image_size") != STANDARD_CXR_IMAGE_SIZE
        or values.get("embedding_dimension") != 1024
    ):
        raise ConfigError("Only the frozen standard CXR encoder identity is supported")
    if "dropout" in values and not 0 <= _number(values["dropout"], "family.parameters.dropout") < 1:
        raise ConfigError("family.parameters.dropout must be in [0, 1)")
    if family_id.endswith("lightgbm"):
        if values.get("objective") != "binary":
            raise ConfigError("LightGBM families require the binary objective")
    if family_id in {"cxr_labs_concat", "cxr_labs_gated", "cxr_labs_gated_no_observedness"}:
        if values.get("lab_input_dimension") != 100:
            raise ConfigError("Symile laboratory models require the 100-column lab contract")
    if family_id in {"cxr_labs_gated", "cxr_labs_gated_no_observedness"}:
        expected_observedness = family_id == "cxr_labs_gated"
        if values.get("use_observedness") is not expected_observedness:
            raise ConfigError("Gated-family observedness policy is inconsistent")
        if values.get("modality_count") != 2:
            raise ConfigError("Current gated families require modality_count=2")
        if values.get("observedness_dimension") != 50:
            raise ConfigError("Current gated families require 50 observedness indicators")


def _validate_tabular_training(dataset_id: str, family_id: str, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigError(f"training.parameters.{key} must be finite")
    if family_id.endswith("logistic"):
        if (
            not 0.0 <= _number(values["l1_ratio"], "training.parameters.l1_ratio") <= 1.0
            or values["solver"] != "liblinear"
            or _number(values["C"], "training.parameters.C") <= 0.0
            or _integer(values["max_iter"], "training.parameters.max_iter") <= 0
        ):
            raise ConfigError("Logistic Regression fitting policy is unsupported")
        expected_weight: object = "balanced" if dataset_id == "rsna" else None
        if values["class_weight"] != expected_weight:
            raise ConfigError("Logistic Regression class-weight policy is unsupported")
        return
    for key in {"n_estimators", "subsample_freq", "early_stopping_rounds"}:
        if _integer(values[key], f"training.parameters.{key}") <= 0:
            raise ConfigError(f"training.parameters.{key} must be a positive integer")
    if _number(values["learning_rate"], "training.parameters.learning_rate") <= 0:
        raise ConfigError("training.parameters.learning_rate must be positive")
    for key in {"subsample", "colsample_bytree"}:
        value = _number(values[key], f"training.parameters.{key}")
        if not 0 < value <= 1:
            raise ConfigError(f"training.parameters.{key} must be in (0, 1]")
    if _number(values["reg_lambda"], "training.parameters.reg_lambda") < 0:
        raise ConfigError("training.parameters.reg_lambda must be non-negative")
    if dataset_id == "rsna":
        if values["class_weighting"] != "train_neg_pos_ratio":
            raise ConfigError("RSNA LightGBM weighting policy is unsupported")
    elif values["class_weight"] is not None:
        raise ConfigError("Symile LightGBM does not use class weighting")


def _validate_scientific_training_policy(
    dataset_id: str, family_id: str, values: dict[str, Any]
) -> None:
    del family_id
    if dataset_id == "rsna":
        if values["class_weighting"] != "train_pos_weight":
            raise ConfigError("RSNA neural families require train-derived positive weighting")
        if values["fine_tune_scope"] != "all":
            raise ConfigError("RSNA neural families require full encoder fine-tuning")
    else:
        if _number(values["pos_weight"], "training.parameters.pos_weight") != 1.0:
            raise ConfigError("Symile neural families require pos_weight=1")
        if values["fine_tune_scope"] != "terminal":
            raise ConfigError("Symile neural families require terminal encoder fine-tuning")


def _validate_neural_training(
    parameters: dict[str, Any], loader: dict[str, Any], augmentation: dict[str, Any]
) -> None:
    shared_parameter_fields = {
        "optimizer",
        "mixed_precision",
        "warmup_epochs",
        "fine_tune_epochs",
        "warmup_head_learning_rate",
        "encoder_learning_rate",
        "head_learning_rate",
        "weight_decay",
        "scheduler_factor",
        "scheduler_patience",
        "scheduler_min_learning_rate",
        "gradient_clip_norm",
        "early_stopping_patience",
        "early_stopping_min_delta",
    }
    loader_fields = {"batch_size"}
    augmentation_fields = {
        "rotation_degrees",
        "translation_fraction",
        "brightness_jitter",
        "contrast_jitter",
    }
    policy_fields = (
        {"class_weighting", "fine_tune_scope"}
        if "class_weighting" in parameters
        else {
            "pos_weight",
            "fine_tune_scope",
        }
    )
    parameter_fields = shared_parameter_fields | policy_fields
    if (
        set(parameters) != parameter_fields
        or set(loader) != loader_fields
        or set(augmentation) != augmentation_fields
    ):
        raise ConfigError("Neural training, loader, or augmentation field set is invalid")
    if _text(parameters["optimizer"], "training.parameters.optimizer") != "adamw":
        raise ConfigError("Only AdamW is supported")
    _boolean(parameters["mixed_precision"], "training.parameters.mixed_precision")
    integer_fields = {
        "warmup_epochs",
        "fine_tune_epochs",
        "scheduler_patience",
        "early_stopping_patience",
    }
    positive_integer_fields = {"warmup_epochs", "fine_tune_epochs"}
    for key in integer_fields:
        minimum = 1 if key in positive_integer_fields else 0
        if _integer(parameters[key], f"training.parameters.{key}") < minimum:
            qualifier = "positive" if minimum else "non-negative"
            raise ConfigError(f"training.parameters.{key} must be {qualifier}")
    numeric_fields = shared_parameter_fields - integer_fields - {"optimizer", "mixed_precision"}
    for key in numeric_fields:
        if _number(parameters[key], f"training.parameters.{key}") < 0:
            raise ConfigError(f"training.parameters.{key} must be non-negative")
    for key in {
        "warmup_head_learning_rate",
        "encoder_learning_rate",
        "head_learning_rate",
        "gradient_clip_norm",
    }:
        if _number(parameters[key], f"training.parameters.{key}") <= 0:
            raise ConfigError(f"training.parameters.{key} must be positive")
    scheduler_factor = _number(
        parameters["scheduler_factor"], "training.parameters.scheduler_factor"
    )
    if not 0 < scheduler_factor < 1:
        raise ConfigError("training.parameters.scheduler_factor must be in (0, 1)")
    scheduler_min = _number(
        parameters["scheduler_min_learning_rate"],
        "training.parameters.scheduler_min_learning_rate",
    )
    if scheduler_min >= min(
        _number(parameters["encoder_learning_rate"], "training.parameters.encoder_learning_rate"),
        _number(parameters["head_learning_rate"], "training.parameters.head_learning_rate"),
    ):
        raise ConfigError("scheduler minimum learning rate must be below trainable learning rates")
    if _integer(loader["batch_size"], "training.loader.batch_size") <= 0:
        raise ConfigError("training.loader.batch_size must be positive")
    augmentation_limits = {
        "rotation_degrees": 180.0,
        "translation_fraction": 1.0,
        "brightness_jitter": 1.0,
        "contrast_jitter": 1.0,
    }
    for key, upper in augmentation_limits.items():
        value = _number(augmentation[key], f"training.augmentation.{key}")
        if not 0 <= value <= upper:
            raise ConfigError(f"training.augmentation.{key} must be in [0, {upper:g}]")


def _neural_config(training: TrainingConfig) -> NeuralConfig:
    p, loader, augmentation = training.parameters, training.loader, training.augmentation
    return NeuralConfig(
        batch_size=int(loader["batch_size"]),
        mixed_precision=bool(p["mixed_precision"]),
        rotation_degrees=float(augmentation["rotation_degrees"]),
        translation_fraction=float(augmentation["translation_fraction"]),
        brightness_jitter=float(augmentation["brightness_jitter"]),
        contrast_jitter=float(augmentation["contrast_jitter"]),
        optimizer=str(p["optimizer"]),
        warmup_epochs=int(p["warmup_epochs"]),
        fine_tune_epochs=int(p["fine_tune_epochs"]),
        warmup_head_learning_rate=float(p["warmup_head_learning_rate"]),
        encoder_learning_rate=float(p["encoder_learning_rate"]),
        head_learning_rate=float(p["head_learning_rate"]),
        weight_decay=float(p["weight_decay"]),
        scheduler_factor=float(p["scheduler_factor"]),
        scheduler_patience=int(p["scheduler_patience"]),
        scheduler_min_learning_rate=float(p["scheduler_min_learning_rate"]),
        gradient_clip_norm=float(p["gradient_clip_norm"]),
        early_stopping_patience=int(p["early_stopping_patience"]),
        early_stopping_min_delta=float(p["early_stopping_min_delta"]),
    )


def _default_runtime(dataset_id: str) -> RuntimeConfig:
    if dataset_id == "rsna":
        return RuntimeConfig(
            Path("data/manifests"),
            Path("data/raw/rsna/extracted"),
            Path("models/rsna"),
            Path("reports"),
            Path("private"),
            "radfusion-rsna",
            "auto",
            2,
            "auto",
        )
    return RuntimeConfig(
        Path("data/manifests"),
        Path("data/raw/symile/extracted"),
        Path("models/symile/development"),
        Path("reports/symile/development"),
        Path("private"),
        "radfusion-symile-development",
        "cuda",
        2,
        "enabled",
    )


def _semantic_payload(
    version: int,
    dataset: DatasetConfig,
    task: TaskConfig,
    family: FamilyConfig,
    preprocessing: MappingProxyType[str, Any],
    training: TrainingConfig,
    evaluation: EvaluationConfig | None,
) -> dict[str, Any]:
    return {
        "config_version": version,
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "bundle_id": dataset.bundle_id,
            "split_assignment_id": dataset.split_assignment_id,
            "cv_assignment_id": dataset.cv_assignment_id,
        },
        "task": {"task_id": task.task_id, "label_policy_version": task.label_policy_version},
        "family": {
            "family_id": family.family_id,
            "modalities": list(family.modalities),
            "parameters": dict(family.parameters),
        },
        "preprocessing": dict(preprocessing),
        "training": {
            "selection_metric": training.selection_metric,
            "parameters": dict(training.parameters),
            "loader": dict(training.loader),
            "augmentation": dict(training.augmentation),
        },
        "evaluation": dict(evaluation.parameters) if evaluation is not None else None,
    }


def _read_yaml(source: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        source_bytes = source.read_bytes()
        document = yaml.load(source_bytes.decode("utf-8"), Loader=_StrictSafeLoader)
    except OSError as exc:
        raise ConfigError(f"Experiment config is unreadable: {source}") from exc
    except UnicodeError as exc:
        raise ConfigError(f"Experiment config is not valid UTF-8: {source}") from exc
    except yaml.YAMLError as exc:
        detail = getattr(exc, "problem", None) or str(exc)
        raise ConfigError(f"Experiment config is invalid YAML: {source}: {detail}") from exc
    return source_bytes, _mapping(document, "config")


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{context} must be a mapping with string keys")
    return value


def _keys(
    value: dict[str, Any], *, required: set[str], optional: set[str] | None = None, context: str
) -> None:
    allowed = required | (optional or set())
    if required - set(value) or set(value) - allowed:
        raise ConfigError(f"{context} has missing or unknown fields")


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ConfigError(f"{context} must be a non-empty trimmed string")
    return value


def _choice(value: object, choices: set[str], context: str) -> str:
    text = _text(value, context)
    if text not in choices:
        raise ConfigError(f"{context} is unsupported")
    return text


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{context} must be an integer")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ConfigError(f"{context} must be a finite number")
    return float(value)


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{context} must be boolean")
    return value


def _sha256(value: object, context: str) -> str:
    text = _text(value, context)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ConfigError(f"{context} must be a lowercase SHA-256")
    return text


def _identity(value: object, prefix: str, context: str) -> str:
    text = _text(value, context)
    if not text.startswith(prefix):
        raise ConfigError(f"{context} has an invalid prefix")
    _sha256(text.removeprefix(prefix), context)
    return text


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
