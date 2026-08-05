"""Load and validate typed experiment configuration files."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


class _StrictSafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _StrictSafeLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
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
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


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

_FUSION_MODEL_PARAMETERS = MappingProxyType(
    {
        "encoder_name": "densenet121",
        "weights": "densenet121-res224-chex",
        "image_size": 224,
        "embedding_dimension": 1024,
        "image_projection_dimension": 256,
        "structured_hidden_dimension": 128,
        "structured_projection_dimension": 64,
        "fusion_hidden_dimension": 128,
        "dropout": 0.2,
        "class_weighting": "train_pos_weight",
    }
)


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset adapter and bundle settings."""

    registry_key: str
    manifest_directory: Path
    bundle_id: str
    task_id: str
    dataset_root: Path | None


@dataclass(frozen=True)
class ModelConfig:
    """Registered model and its declarative parameters."""

    registry_key: str
    modality: str
    parameters: MappingProxyType[str, Any]
    fit_parameters: MappingProxyType[str, Any]


@dataclass(frozen=True)
class TrainingConfig:
    """Training seed and generated-output locations."""

    seed: int
    report_directory: Path
    model_directory: Path


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation policies shared by model implementations."""

    sensitivity_target: float
    calibration_bins: int
    latency_warmup_calls: int
    latency_measured_calls: int


@dataclass(frozen=True)
class MLflowConfig:
    """MLflow experiment identity."""

    experiment_name: str


@dataclass(frozen=True)
class ImageConfig:
    """Image augmentation, loading, runtime, and optimization settings."""

    batch_size: int
    num_workers: int
    pin_memory_policy: str
    device: str
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
    """Complete immutable experiment definition."""

    config_version: int
    name: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    mlflow: MLflowConfig
    image: ImageConfig | None
    source_path: Path
    source_bytes: bytes
    source_sha256: str


def image_semantic_config_sha256(config: ExperimentConfig) -> str:
    """Hash the meaning-bearing, path-independent image experiment configuration."""
    return _canonical_sha256(_image_semantic_config(config))


def image_seed_compatibility_sha256(config: ExperimentConfig) -> str:
    """Hash image experiment meaning after excluding only the training seed."""
    payload = _image_semantic_config(config)
    del payload["training"]["seed"]
    return _canonical_sha256(payload)


def fusion_semantic_config_sha256(config: ExperimentConfig) -> str:
    """Hash the meaning-bearing, path-independent fusion experiment configuration."""
    return _canonical_sha256(_fusion_semantic_config(config))


def fusion_seed_compatibility_sha256(config: ExperimentConfig) -> str:
    """Hash fusion experiment meaning after excluding only the training seed."""
    payload = _fusion_semantic_config(config)
    del payload["training"]["seed"]
    return _canonical_sha256(payload)


def _image_semantic_config(config: ExperimentConfig) -> dict[str, Any]:
    return _neural_semantic_config(config, modality="image")


def _fusion_semantic_config(config: ExperimentConfig) -> dict[str, Any]:
    return _neural_semantic_config(config, modality="fusion")


def _neural_semantic_config(
    config: ExperimentConfig,
    *,
    modality: str,
) -> dict[str, Any]:
    if config.model.modality != modality or config.image is None:
        raise ConfigError(f"Semantic {modality} configuration requires {modality} modality")
    image = asdict(config.image)
    del image["num_workers"]
    del image["pin_memory_policy"]
    return {
        "config_version": config.config_version,
        "dataset": {
            "registry_key": config.dataset.registry_key,
            "bundle_id": config.dataset.bundle_id,
            "task_id": config.dataset.task_id,
        },
        "model": {
            "registry_key": config.model.registry_key,
            "modality": config.model.modality,
            "parameters": dict(config.model.parameters),
            "fit_parameters": dict(config.model.fit_parameters),
        },
        "training": {"seed": config.training.seed},
        "evaluation": {
            "sensitivity_target": config.evaluation.sensitivity_target,
            "calibration_bins": config.evaluation.calibration_bins,
        },
        "image": image,
    }


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    source = Path(path)
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
    root = _mapping(document, "config")
    _keys(
        root,
        required={
            "config_version",
            "name",
            "dataset",
            "model",
            "training",
            "evaluation",
            "mlflow",
        },
        optional={"image"},
        context="config",
    )
    config_version = _integer(root["config_version"], "config_version")
    if config_version != 1:
        raise ConfigError(f"Unsupported config_version: {config_version}")
    model = _model_config(root["model"])
    dataset = _dataset_config(root["dataset"])
    if model.modality in {"image", "fusion"}:
        expected_key = "image_densenet" if model.modality == "image" else "fusion_concat"
        if model.registry_key != expected_key:
            raise ConfigError(
                f"{model.modality.capitalize()} experiments require "
                f"model.registry_key={expected_key!r}"
            )
        if "image" not in root:
            raise ConfigError("Neural experiments require an image configuration section")
        if dataset.dataset_root is None:
            raise ConfigError("Neural experiments require dataset.dataset_root")
    else:
        if model.registry_key not in {"metadata_logistic", "metadata_lightgbm"}:
            raise ConfigError("Metadata experiments require a registered metadata model")
        if dataset.dataset_root is not None:
            raise ConfigError("Metadata experiments do not accept dataset.dataset_root")
        if "image" in root:
            raise ConfigError("The image configuration requires a neural modality")
    return ExperimentConfig(
        config_version=config_version,
        name=_text(root["name"], "name"),
        dataset=dataset,
        model=model,
        training=_training_config(root["training"]),
        evaluation=_evaluation_config(root["evaluation"]),
        mlflow=_mlflow_config(root["mlflow"]),
        image=_image_config(root["image"]) if "image" in root else None,
        source_path=source,
        source_bytes=source_bytes,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )


def _dataset_config(value: object) -> DatasetConfig:
    data = _mapping(value, "dataset")
    _keys(
        data,
        required={"registry_key", "manifest_directory", "bundle_id", "task_id"},
        optional={"dataset_root"},
        context="dataset",
    )
    return DatasetConfig(
        registry_key=_path_component(data["registry_key"], "dataset.registry_key"),
        manifest_directory=Path(_text(data["manifest_directory"], "dataset.manifest_directory")),
        bundle_id=_path_component(data["bundle_id"], "dataset.bundle_id"),
        task_id=_text(data["task_id"], "dataset.task_id"),
        dataset_root=(
            Path(_text(data["dataset_root"], "dataset.dataset_root"))
            if "dataset_root" in data
            else None
        ),
    )


def _model_config(value: object) -> ModelConfig:
    data = _mapping(value, "model")
    _keys(
        data,
        required={
            "registry_key",
            "modality",
            "parameters",
            "fit_parameters",
        },
        context="model",
    )
    modality = _text(data["modality"], "model.modality")
    if modality not in {"metadata", "image", "fusion"}:
        raise ConfigError("model.modality must be 'metadata', 'image', or 'fusion'")
    parameters = _mapping(data["parameters"], "model.parameters")
    randomness_conflicts = sorted(MODEL_RANDOMNESS_KEYS & parameters.keys())
    if randomness_conflicts:
        raise ConfigError(
            "model.parameters contains randomness controls reserved for training.seed: "
            f"{randomness_conflicts}"
        )
    if modality == "image":
        _validate_image_model_parameters(parameters)
        if _mapping(data["fit_parameters"], "model.fit_parameters"):
            raise ConfigError("Image models require empty model.fit_parameters")
    elif modality == "fusion":
        _validate_fusion_model_parameters(parameters)
        if _mapping(data["fit_parameters"], "model.fit_parameters"):
            raise ConfigError("Fusion models require empty model.fit_parameters")
    return ModelConfig(
        registry_key=_path_component(data["registry_key"], "model.registry_key"),
        modality=modality,
        parameters=MappingProxyType(parameters),
        fit_parameters=MappingProxyType(_mapping(data["fit_parameters"], "model.fit_parameters")),
    )


def _training_config(value: object) -> TrainingConfig:
    data = _mapping(value, "training")
    _keys(
        data,
        required={"seed", "report_directory", "model_directory"},
        context="training",
    )
    seed = _integer(data["seed"], "training.seed")
    if not 0 <= seed <= 2**31 - 1:
        raise ConfigError("training.seed must be between 0 and 2147483647")
    return TrainingConfig(
        seed=seed,
        report_directory=Path(_text(data["report_directory"], "training.report_directory")),
        model_directory=Path(_text(data["model_directory"], "training.model_directory")),
    )


def _evaluation_config(value: object) -> EvaluationConfig:
    data = _mapping(value, "evaluation")
    _keys(
        data,
        required={
            "sensitivity_target",
            "calibration_bins",
            "latency_warmup_calls",
            "latency_measured_calls",
        },
        context="evaluation",
    )
    sensitivity = _number(data["sensitivity_target"], "evaluation.sensitivity_target")
    bins = _integer(data["calibration_bins"], "evaluation.calibration_bins")
    warmup = _integer(data["latency_warmup_calls"], "evaluation.latency_warmup_calls")
    measured = _integer(data["latency_measured_calls"], "evaluation.latency_measured_calls")
    if not 0 < sensitivity <= 1:
        raise ConfigError("evaluation.sensitivity_target must be in (0, 1]")
    if bins <= 0 or warmup < 0 or measured <= 0:
        raise ConfigError("Evaluation bin and latency call counts are invalid")
    return EvaluationConfig(sensitivity, bins, warmup, measured)


def _mlflow_config(value: object) -> MLflowConfig:
    data = _mapping(value, "mlflow")
    _keys(data, required={"experiment_name"}, context="mlflow")
    return MLflowConfig(
        experiment_name=_text(data["experiment_name"], "mlflow.experiment_name"),
    )


def _validate_image_model_parameters(parameters: dict[str, Any]) -> None:
    required = {
        "encoder_name",
        "weights",
        "image_size",
        "embedding_dimension",
        "class_weighting",
    }
    _keys(parameters, required=required, context="model.parameters")
    if _text(parameters["encoder_name"], "model.parameters.encoder_name") != "densenet121":
        raise ConfigError("The image encoder must be DenseNet121")
    if _text(parameters["weights"], "model.parameters.weights") != "densenet121-res224-chex":
        raise ConfigError("The image encoder must use densenet121-res224-chex weights")
    if _integer(parameters["image_size"], "model.parameters.image_size") != 224:
        raise ConfigError("The image model requires image_size=224")
    if _integer(parameters["embedding_dimension"], "model.parameters.embedding_dimension") != 1024:
        raise ConfigError("The image model requires embedding_dimension=1024")
    if (
        _text(parameters["class_weighting"], "model.parameters.class_weighting")
        != "train_pos_weight"
    ):
        raise ConfigError("The image model requires train_pos_weight class weighting")


def _validate_fusion_model_parameters(parameters: dict[str, Any]) -> None:
    required = set(_FUSION_MODEL_PARAMETERS)
    _keys(parameters, required=required, context="model.parameters")
    encoder = _FUSION_MODEL_PARAMETERS["encoder_name"]
    if _text(parameters["encoder_name"], "model.parameters.encoder_name") != encoder:
        raise ConfigError(f"The fusion image encoder must be {encoder}")
    weights = _FUSION_MODEL_PARAMETERS["weights"]
    if _text(parameters["weights"], "model.parameters.weights") != weights:
        raise ConfigError(f"The fusion image encoder must use {weights} weights")
    expected_dimensions = {
        field: value
        for field, value in _FUSION_MODEL_PARAMETERS.items()
        if field.endswith("dimension") or field == "image_size"
    }
    for field, expected in expected_dimensions.items():
        if _integer(parameters[field], f"model.parameters.{field}") != expected:
            raise ConfigError(f"The fusion model requires {field}={expected}")
    dropout = _FUSION_MODEL_PARAMETERS["dropout"]
    if _number(parameters["dropout"], "model.parameters.dropout") != dropout:
        raise ConfigError(f"The fusion model requires dropout={dropout}")
    class_weighting = _FUSION_MODEL_PARAMETERS["class_weighting"]
    if _text(parameters["class_weighting"], "model.parameters.class_weighting") != class_weighting:
        raise ConfigError(f"The fusion model requires {class_weighting} class weighting")


def fusion_architecture_contract(
    config: ModelConfig,
    *,
    structured_input_dimension: int,
) -> dict[str, int | float]:
    """Derive the fixed concat-fusion architecture from validated configuration."""
    if config.modality != "fusion" or config.registry_key != "fusion_concat":
        raise ConfigError("Concat fusion requires the registered fusion configuration")
    if config.fit_parameters:
        raise ConfigError("Concat fusion does not accept model.fit_parameters")
    _validate_fusion_model_parameters(dict(config.parameters))
    if (
        isinstance(structured_input_dimension, bool)
        or not isinstance(structured_input_dimension, int)
        or structured_input_dimension <= 0
    ):
        raise ConfigError("Fusion structured input dimension must be a positive integer")
    parameters = config.parameters
    return {
        "image_embedding_dimension": parameters["embedding_dimension"],
        "image_projection_dimension": parameters["image_projection_dimension"],
        "structured_input_dimension": structured_input_dimension,
        "structured_hidden_dimension": parameters["structured_hidden_dimension"],
        "structured_projection_dimension": parameters["structured_projection_dimension"],
        "fusion_input_dimension": parameters["image_projection_dimension"]
        + parameters["structured_projection_dimension"],
        "fusion_hidden_dimension": parameters["fusion_hidden_dimension"],
        "dropout": parameters["dropout"],
        "output_dimension": 1,
    }


def fusion_structured_input_conversion_contract() -> dict[str, object]:
    """Return the fixed conversion from fitted metadata output to neural input."""
    return {
        "source_dtype": "float64",
        "tensor_dtype": "torch.float32",
        "layout": "contiguous",
        "finite": True,
        "feature_order": "structured_preprocessor_contract.transformed_feature_names",
    }


def _image_config(value: object) -> ImageConfig:
    data = _mapping(value, "image")
    required = {
        "batch_size",
        "num_workers",
        "pin_memory_policy",
        "device",
        "mixed_precision",
        "rotation_degrees",
        "translation_fraction",
        "brightness_jitter",
        "contrast_jitter",
        "optimizer",
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
    _keys(data, required=required, context="image")
    device = _choice(data["device"], {"auto", "cpu", "cuda"}, "image.device")
    pin_memory = _choice(
        data["pin_memory_policy"],
        {"auto", "enabled", "disabled"},
        "image.pin_memory_policy",
    )
    optimizer = _choice(data["optimizer"], {"adamw"}, "image.optimizer")
    config = ImageConfig(
        batch_size=_integer(data["batch_size"], "image.batch_size"),
        num_workers=_integer(data["num_workers"], "image.num_workers"),
        pin_memory_policy=pin_memory,
        device=device,
        mixed_precision=_boolean(data["mixed_precision"], "image.mixed_precision"),
        rotation_degrees=_number(data["rotation_degrees"], "image.rotation_degrees"),
        translation_fraction=_number(data["translation_fraction"], "image.translation_fraction"),
        brightness_jitter=_number(data["brightness_jitter"], "image.brightness_jitter"),
        contrast_jitter=_number(data["contrast_jitter"], "image.contrast_jitter"),
        optimizer=optimizer,
        warmup_epochs=_integer(data["warmup_epochs"], "image.warmup_epochs"),
        fine_tune_epochs=_integer(data["fine_tune_epochs"], "image.fine_tune_epochs"),
        warmup_head_learning_rate=_number(
            data["warmup_head_learning_rate"], "image.warmup_head_learning_rate"
        ),
        encoder_learning_rate=_number(data["encoder_learning_rate"], "image.encoder_learning_rate"),
        head_learning_rate=_number(data["head_learning_rate"], "image.head_learning_rate"),
        weight_decay=_number(data["weight_decay"], "image.weight_decay"),
        scheduler_factor=_number(data["scheduler_factor"], "image.scheduler_factor"),
        scheduler_patience=_integer(data["scheduler_patience"], "image.scheduler_patience"),
        scheduler_min_learning_rate=_number(
            data["scheduler_min_learning_rate"], "image.scheduler_min_learning_rate"
        ),
        gradient_clip_norm=_number(data["gradient_clip_norm"], "image.gradient_clip_norm"),
        early_stopping_patience=_integer(
            data["early_stopping_patience"], "image.early_stopping_patience"
        ),
        early_stopping_min_delta=_number(
            data["early_stopping_min_delta"], "image.early_stopping_min_delta"
        ),
    )
    _validate_image_ranges(config)
    return config


def _validate_image_ranges(config: ImageConfig) -> None:
    if config.batch_size <= 0 or config.num_workers < 0:
        raise ConfigError("Image batch size and worker count are invalid")
    if config.warmup_epochs <= 0 or config.fine_tune_epochs <= 0:
        raise ConfigError("Image stage epoch counts must be positive")
    if not 0.0 <= config.rotation_degrees <= 180.0:
        raise ConfigError("image.rotation_degrees must be within [0, 180]")
    fractions = (
        config.translation_fraction,
        config.brightness_jitter,
        config.contrast_jitter,
    )
    if any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ConfigError("Image augmentation fractions must be within [0, 1]")
    if any(
        value <= 0.0
        for value in (
            config.warmup_head_learning_rate,
            config.encoder_learning_rate,
            config.head_learning_rate,
            config.gradient_clip_norm,
        )
    ):
        raise ConfigError("Image learning rates and gradient clipping must be positive")
    if any(
        value < 0.0
        for value in (
            config.weight_decay,
            config.scheduler_min_learning_rate,
            config.early_stopping_min_delta,
        )
    ):
        raise ConfigError("Image optimization values must be nonnegative")
    if not 0.0 < config.scheduler_factor < 1.0:
        raise ConfigError("image.scheduler_factor must be within (0, 1)")
    if config.scheduler_min_learning_rate >= min(
        config.encoder_learning_rate, config.head_learning_rate
    ):
        raise ConfigError("Scheduler minimum learning rate must be below fine-tuning rates")
    if config.scheduler_patience < 0 or config.early_stopping_patience < 0:
        raise ConfigError("Image patience values must be nonnegative")


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{context} must be a mapping with string keys")
    return dict(value)


def _keys(
    data: dict[str, Any],
    *,
    required: set[str],
    context: str,
    optional: set[str] | None = None,
) -> None:
    missing = sorted(required - set(data))
    unknown = sorted(set(data) - required - (optional or set()))
    if missing:
        raise ConfigError(f"{context} is missing keys: {missing}")
    if unknown:
        raise ConfigError(f"{context} has unknown keys: {unknown}")


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must be a non-empty string")
    return value


def _path_component(value: object, context: str) -> str:
    text = _text(value, context)
    if (
        text in {".", ".."}
        or Path(text).is_absolute()
        or "/" in text
        or "\\" in text
        or Path(text).name != text
    ):
        raise ConfigError(f"{context} must be one safe path component")
    return text


def _integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{context} must be an integer")
    return value


def _number(value: object, context: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{context} must be finite")
    return number


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{context} must be Boolean")
    return value


def _choice(value: object, choices: set[str], context: str) -> str:
    text = _text(value, context)
    if text not in choices:
        raise ConfigError(f"{context} must be one of {sorted(choices)}")
    return text
