from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from beyondcxr.training.config import (
    FAMILY_MODALITIES,
    ConfigError,
    load_experiment_config,
    normalized_scientific_config,
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
    assert all(config.dataset.bundle_id is None for config in configs)
    assert all(config.dataset.cv_assignment_id is None for config in configs)
    assert configs[0].family.family_id == "metadata_logistic"
    assert configs[0].family.modalities == ("metadata",)
    assert configs[1].training.parameters["early_stopping_rounds"] == 50
    assert configs[1].evaluation is not None
    assert configs[1].evaluation.calibration_bins == 15
    assert configs[0].source_path == Path("configs/rsna_metadata_logistic.yaml")
    assert configs[0].source_bytes == configs[0].source_path.read_bytes()
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
    assert normalized_scientific_config(seeded) == normalized_scientific_config(config)
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


def test_yaml_representation_does_not_change_normalized_science(tmp_path: Path) -> None:
    source = Path("configs/rsna_metadata_logistic.yaml")
    original = load_experiment_config(source)
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    equivalent = tmp_path / "equivalent.yaml"
    equivalent.write_text(
        "# representation-only comment\n" + yaml.safe_dump(document, sort_keys=True),
        encoding="utf-8",
    )
    loaded = load_experiment_config(equivalent)
    assert loaded.source_bytes != original.source_bytes
    assert normalized_scientific_config(loaded) == normalized_scientific_config(original)


def test_structurally_valid_alternate_scientific_value_is_representable(tmp_path: Path) -> None:
    original = load_experiment_config("configs/rsna_metadata_logistic.yaml")
    document = _document()
    document["training"]["parameters"]["C"] = 0.5  # type: ignore[index]
    changed = load_experiment_config(_write(tmp_path, document))
    assert changed.training.parameters["C"] == 0.5
    assert normalized_scientific_config(changed) != normalized_scientific_config(original)


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

    assert normalized_scientific_config(integer_config) == normalized_scientific_config(
        float_config
    )
    assert isinstance(integer_config.training.parameters["C"], float)
    assert math.copysign(1.0, float_config.training.parameters["l1_ratio"]) == 1.0

    different = _document()
    different["training"]["parameters"]["C"] = 0.5  # type: ignore[index]
    different["evaluation"]["sensitivity_target"] = 1.0  # type: ignore[index]
    assert normalized_scientific_config(
        load_experiment_config(_write(tmp_path, different))
    ) != normalized_scientific_config(integer_config)


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


@pytest.mark.parametrize("version", [True, 1.0, 2, _MISSING_SCHEMA_VERSION])
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
    for section in ("task", "dataset"):
        document = _document()
        document[section]["unexpected"] = True  # type: ignore[index]
        with pytest.raises(ConfigError, match="unknown fields"):
            load_experiment_config(_write(tmp_path, document))


@pytest.mark.parametrize("bundle_id", ["../escape", "a/b", ".", "bundle-not-a-sha256"])
def test_bundle_id_requires_canonical_bundle_identity(tmp_path: Path, bundle_id: str) -> None:
    document = _document()
    with pytest.raises(ConfigError):
        load_experiment_config(_write(tmp_path, document), bundle_id=bundle_id)


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


@pytest.mark.parametrize("field", ["bundle_id", "cv_assignment_id"])
def test_reviewed_yaml_rejects_generated_authority_coordinates(tmp_path: Path, field: str) -> None:
    document = _document("symile_labs_logistic.yaml")
    document["dataset"][field] = "not-a-reviewed-policy"
    with pytest.raises(ConfigError, match="unknown fields"):
        load_experiment_config(_write(tmp_path, document))


def test_same_reviewed_policy_binds_explicit_builder_coordinates() -> None:
    path = Path("configs/symile_labs_logistic.yaml")
    first = load_experiment_config(
        path, bundle_id="bundle-" + "a" * 64, cv_assignment_id="cv-assignment-" + "b" * 64
    )
    second = load_experiment_config(
        path, bundle_id="bundle-" + "c" * 64, cv_assignment_id="cv-assignment-" + "d" * 64
    )
    assert first.source_bytes == second.source_bytes == path.read_bytes()
    assert first.dataset.bundle_id == "bundle-" + "a" * 64
    assert second.dataset.bundle_id == "bundle-" + "c" * 64
    assert first.dataset.cv_assignment_id == "cv-assignment-" + "b" * 64
    assert second.dataset.cv_assignment_id == "cv-assignment-" + "d" * 64
    assert normalized_scientific_config(first) != normalized_scientific_config(second)


def test_unbound_symile_config_cannot_resolve_current(monkeypatch: pytest.MonkeyPatch) -> None:
    from beyondcxr.data.errors import ManifestBuildError
    from beyondcxr.training import symile_data

    def forbidden(*args, **kwargs):
        pytest.fail("Unbound scientific access reached a dataset resolver")

    monkeypatch.setattr(symile_data, "resolve_symile_bundle", forbidden)
    config = load_experiment_config("configs/symile_labs_logistic.yaml")
    with pytest.raises(ManifestBuildError, match="explicit bundle and CV identities"):
        symile_data.load_symile_development_cohort(config)
