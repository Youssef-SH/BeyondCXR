from __future__ import annotations

from pathlib import Path

import yaml

from radfusion.data.symile_schemas import ECG_TENSOR_POLICY_VERSION
from radfusion.training.config import load_symile_development_config

SYMILE_CONFIG_FILENAMES = (
    "symile_labs_logistic.yaml",
    "symile_labs_lightgbm.yaml",
    "symile_cxr_densenet.yaml",
    "symile_cxr_labs_concat.yaml",
    "symile_cxr_labs_gated.yaml",
    "symile_cxr_labs_gated_no_observedness.yaml",
)


def _document(name: str) -> dict[str, object]:
    return yaml.safe_load((Path("configs") / name).read_text(encoding="utf-8"))


def test_exact_six_symile_development_configs_are_strict() -> None:
    configs = tuple(
        load_symile_development_config(Path("configs") / name) for name in SYMILE_CONFIG_FILENAMES
    )
    assert tuple(config.family.family_id for config in configs) == (
        "labs_logistic",
        "labs_lightgbm",
        "cxr_densenet",
        "cxr_labs_concat",
        "cxr_labs_gated",
        "cxr_labs_gated_no_observedness",
    )
    assert all(config.task.task_id == "pneumonia_strict" for config in configs)
    assert len({config.config_semantic_sha256 for config in configs}) == 6


def test_symile_gated_ablation_differs_only_by_family_and_observedness() -> None:
    gated = _document("symile_cxr_labs_gated.yaml")
    ablation = _document("symile_cxr_labs_gated_no_observedness.yaml")
    gated_family = gated["family"]  # type: ignore[index]
    ablation_family = ablation["family"]  # type: ignore[index]
    assert gated_family.pop("family_id") == "cxr_labs_gated"
    assert ablation_family.pop("family_id") == "cxr_labs_gated_no_observedness"
    assert gated_family["parameters"].pop("use_observedness") is True
    assert ablation_family["parameters"].pop("use_observedness") is False
    assert gated == ablation


def test_ecg_gated_config_is_a_separate_exact_three_extension() -> None:
    config = load_symile_development_config("configs/symile_cxr_labs_ecg_gated.yaml")
    assert config.family.modalities == ("cxr", "labs", "ecg")
    assert config.family.parameters["modality_count"] == 3
    assert config.preprocessing["ecg_policy"] == ECG_TENSOR_POLICY_VERSION
