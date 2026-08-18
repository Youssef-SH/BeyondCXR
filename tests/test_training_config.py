from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from radfusion.training.config import (
    FAMILY_MODALITIES,
    ConfigError,
    load_experiment_config,
    require_runtime_seed,
    with_runtime,
)

CONFIG_FILENAMES = (
    "rsna_metadata_logistic.yaml",
    "rsna_metadata_lightgbm.yaml",
    "rsna_cxr_densenet.yaml",
    "rsna_cxr_metadata_concat.yaml",
    "symile_labs_logistic.yaml",
    "symile_labs_lightgbm.yaml",
    "symile_cxr_densenet.yaml",
    "symile_cxr_labs_concat.yaml",
    "symile_cxr_labs_gated.yaml",
    "symile_cxr_labs_gated_no_observedness.yaml",
    "symile_cxr_labs_ecg_gated.yaml",
)


def _document(name: str = "rsna_metadata_logistic.yaml") -> dict[str, object]:
    return yaml.safe_load((Path("configs") / name).read_text(encoding="utf-8"))


def _write(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_config_directory_contains_exactly_the_eleven_canonical_files() -> None:
    observed = tuple(sorted(path.name for path in Path("configs").glob("*.yaml")))
    assert observed == tuple(sorted(CONFIG_FILENAMES))
    assert all(_document(name)["config_schema_version"] == 1 for name in CONFIG_FILENAMES)


def test_supported_configs_load_as_immutable_typed_values() -> None:
    configs = tuple(load_experiment_config(Path("configs") / name) for name in CONFIG_FILENAMES)
    assert all(config.config_schema_version == 1 for config in configs)
    assert configs[0].dataset.dataset_id == "rsna"
    assert configs[0].dataset.bundle_id.startswith("bundle-")
    assert configs[0].family.family_id == "metadata_logistic"
    assert configs[0].family.modalities == ("metadata",)
    assert configs[1].training.parameters["early_stopping_rounds"] == 50
    assert configs[1].evaluation is not None
    assert configs[1].evaluation.calibration_bins == 15
    expected_source = hashlib.sha256(
        Path("configs/rsna_metadata_logistic.yaml").read_bytes()
    ).hexdigest()
    assert configs[0].config_source_sha256 == expected_source
    with pytest.raises(FrozenInstanceError):
        configs[0].config_schema_version = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        configs[0].family.parameters["C"] = 2.0  # type: ignore[index]


def test_family_and_modality_vocabulary_is_exact() -> None:
    configs = tuple(load_experiment_config(Path("configs") / name) for name in CONFIG_FILENAMES)
    observed = {
        (config.dataset.dataset_id, config.family.family_id): config.family.modalities
        for config in configs
    }
    assert observed == FAMILY_MODALITIES
    assert {modality for value in observed.values() for modality in value} == {
        "cxr",
        "metadata",
        "labs",
        "ecg",
    }


def test_yaml_contains_no_runtime_or_execution_coordinate_authority() -> None:
    forbidden = {
        "source_root",
        "manifest_directory",
        "model_directory",
        "report_directory",
        "private_output_directory",
        "tracking_uri",
        "experiment_name",
        "device",
        "seed",
        "num_workers",
        "pin_memory_policy",
    }
    for name in CONFIG_FILENAMES:
        document = _document(name)
        serialized = yaml.safe_dump(document)
        assert not any(f"{field}:" in serialized for field in forbidden)


def test_runtime_seed_is_explicit_and_identity_neutral() -> None:
    config = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    with pytest.raises(ConfigError):
        require_runtime_seed(config)
    seeded = with_runtime(config, seed=17, source_root="/tmp/source", device="cpu")
    assert require_runtime_seed(seeded) == 17
    assert seeded.runtime.source_root == Path("/tmp/source")
    assert seeded.runtime.device == "cpu"
    assert seeded.config_semantic_sha256 == config.config_semantic_sha256
    assert seeded.config_source_sha256 == config.config_source_sha256
    for device in ("auto", "cpu", "cuda"):
        assert with_runtime(config, device=device).runtime.device == device
    with pytest.raises(ConfigError):
        with_runtime(config, device="mps")


@pytest.mark.parametrize("seed", [True, -1, 2**31])
def test_runtime_seed_rejects_invalid_type_or_bounds(seed: object) -> None:
    config = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    with pytest.raises(ConfigError):
        with_runtime(config, seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [0, 2**31 - 1])
def test_runtime_seed_accepts_exact_boundaries(seed: int) -> None:
    config = with_runtime(load_experiment_config("configs/rsna_cxr_densenet.yaml"), seed=seed)
    assert require_runtime_seed(config) == seed


def test_selection_metric_is_canonical_for_all_families() -> None:
    expected = {
        "rsna_metadata_logistic.yaml": "none",
        "rsna_metadata_lightgbm.yaml": "average_precision",
        "rsna_cxr_densenet.yaml": "average_precision",
        "rsna_cxr_metadata_concat.yaml": "average_precision",
        "symile_labs_logistic.yaml": "none",
        "symile_labs_lightgbm.yaml": "roc_auc",
        "symile_cxr_densenet.yaml": "roc_auc",
        "symile_cxr_labs_concat.yaml": "roc_auc",
        "symile_cxr_labs_gated.yaml": "roc_auc",
        "symile_cxr_labs_gated_no_observedness.yaml": "roc_auc",
        "symile_cxr_labs_ecg_gated.yaml": "roc_auc",
    }
    assert {
        name: load_experiment_config(Path("configs") / name).training.selection_metric
        for name in CONFIG_FILENAMES
    } == expected


def test_source_hash_changes_but_semantic_hash_survives_equivalent_yaml(tmp_path: Path) -> None:
    source = Path("configs/rsna_metadata_logistic.yaml")
    original = load_experiment_config(source)
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    equivalent = tmp_path / "equivalent.yaml"
    equivalent.write_text(
        "# representation-only comment\n" + yaml.safe_dump(document, sort_keys=True),
        encoding="utf-8",
    )
    loaded = load_experiment_config(equivalent)
    assert loaded.config_source_sha256 != original.config_source_sha256
    assert loaded.config_semantic_sha256 == original.config_semantic_sha256


def test_semantic_config_identity_is_stable_across_manifest_witnesses(tmp_path: Path) -> None:
    original = load_experiment_config("configs/rsna_metadata_logistic.yaml")
    document = _document()
    document["dataset"]["bundle_manifest_sha256"] = "0" * 64  # type: ignore[index]
    changed = load_experiment_config(_write(tmp_path, document))
    assert changed.dataset.bundle_manifest_sha256 == "0" * 64
    assert changed.config_source_sha256 != original.config_source_sha256
    assert changed.config_semantic_sha256 == original.config_semantic_sha256


def test_structurally_valid_alternate_scientific_value_is_representable(tmp_path: Path) -> None:
    original = load_experiment_config("configs/rsna_metadata_logistic.yaml")
    document = _document()
    document["training"]["parameters"]["C"] = 0.5  # type: ignore[index]
    changed = load_experiment_config(_write(tmp_path, document))
    assert changed.training.parameters["C"] == 0.5
    assert changed.config_semantic_sha256 != original.config_semantic_sha256


def test_logistic_policy_matches_frozen_contract() -> None:
    for name in ("rsna_metadata_logistic.yaml", "symile_labs_logistic.yaml"):
        config = load_experiment_config(Path("configs") / name)
        assert config.training.parameters["solver"] == "liblinear"
        assert config.training.parameters["l1_ratio"] == 0.0


@pytest.mark.parametrize(
    ("num_leaves", "supported"),
    [(2, True), (131072, True), (1, False), (131073, False)],
)
def test_lightgbm_num_leaves_respects_supported_domain(
    tmp_path: Path, num_leaves: int, supported: bool
) -> None:
    document = _document("rsna_metadata_lightgbm.yaml")
    document["family"]["parameters"]["num_leaves"] = num_leaves  # type: ignore[index]
    if supported:
        assert (
            load_experiment_config(_write(tmp_path, document)).family.parameters["num_leaves"]
            == num_leaves
        )
    else:
        with pytest.raises(ConfigError):
            load_experiment_config(_write(tmp_path, document))


def test_equivalent_numeric_config_representations_share_semantic_identity(
    tmp_path: Path,
) -> None:
    integer_form = _document()
    integer_form["training"]["parameters"]["C"] = 1  # type: ignore[index]
    integer_form["training"]["parameters"]["l1_ratio"] = 0  # type: ignore[index]
    integer_form["evaluation"]["sensitivity_target"] = 1  # type: ignore[index]
    integer_config = load_experiment_config(_write(tmp_path, integer_form))

    float_form = _document()
    float_form["training"]["parameters"]["C"] = 1.0  # type: ignore[index]
    float_form["training"]["parameters"]["l1_ratio"] = -0.0  # type: ignore[index]
    float_form["evaluation"]["sensitivity_target"] = 1.0  # type: ignore[index]
    float_config = load_experiment_config(_write(tmp_path, float_form))

    assert integer_config.config_semantic_sha256 == float_config.config_semantic_sha256
    assert isinstance(integer_config.training.parameters["C"], float)
    assert math.copysign(1.0, float_config.training.parameters["l1_ratio"]) == 1.0

    different = _document()
    different["training"]["parameters"]["C"] = 0.5  # type: ignore[index]
    different["evaluation"]["sensitivity_target"] = 1.0  # type: ignore[index]
    assert (
        load_experiment_config(_write(tmp_path, different)).config_semantic_sha256
        != integer_config.config_semantic_sha256
    )


def test_invalid_config_fails_before_execution(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("config_schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_experiment_config(path)


@pytest.mark.parametrize("calibration_bins", [2, 1000])
def test_evaluation_calibration_bins_accept_exact_boundaries(
    tmp_path: Path, calibration_bins: int
) -> None:
    document = _document()
    document["evaluation"]["calibration_bins"] = calibration_bins
    config = load_experiment_config(_write(tmp_path, document))
    assert config.evaluation is not None
    assert config.evaluation.calibration_bins == calibration_bins


@pytest.mark.parametrize("calibration_bins", [1, 1001, True, "15"])
def test_evaluation_calibration_bins_reject_out_of_range_or_mistyped_values(
    tmp_path: Path, calibration_bins: object
) -> None:
    document = _document()
    document["evaluation"]["calibration_bins"] = calibration_bins
    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))


_MISSING_SCHEMA_VERSION = object()


@pytest.mark.parametrize("version", [True, 1.0, "1", None, 0, _MISSING_SCHEMA_VERSION])
def test_only_integer_config_schema_version_one_is_supported(
    tmp_path: Path, version: object
) -> None:
    document = _document()
    if version is _MISSING_SCHEMA_VERSION:
        document.pop("config_schema_version")
    else:
        document["config_schema_version"] = version
    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))


def test_unknown_top_level_and_nested_fields_are_rejected(tmp_path: Path) -> None:
    document = _document()
    document["unexpected"] = True
    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))
    document = _document()
    document["task"]["unexpected"] = True  # type: ignore[index]
    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))


@pytest.mark.parametrize("bundle_id", ["../escape", "a/b", ".", "bundle-not-a-sha256"])
def test_bundle_id_requires_canonical_bundle_identity(tmp_path: Path, bundle_id: str) -> None:
    document = _document()
    document["dataset"]["bundle_id"] = bundle_id  # type: ignore[index]
    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("config_schema_version: 1\nconfig_schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_experiment_config(path)


@pytest.mark.parametrize(
    "seed_key",
    ["random_state", "seed", "bagging_seed", "feature_fraction_seed", "data_random_seed"],
)
def test_model_randomness_controls_are_rejected_in_yaml(tmp_path: Path, seed_key: str) -> None:
    document = _document()
    document["family"]["parameters"][seed_key] = 7  # type: ignore[index]
    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document))
