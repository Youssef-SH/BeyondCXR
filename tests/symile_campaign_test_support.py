from __future__ import annotations

import json
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import beyondcxr.training.symile_campaign_control as campaign_control
import beyondcxr.training.symile_final_packages as final_packages
from beyondcxr.training.config import load_symile_development_config
from beyondcxr.training.symile_campaign_control import (
    ValidatedHeldOutCapability,
    ValidatedPretestFreeze,
    ValidatedSymileFrozenCampaign,
    ValidatedTestOpenRecord,
)
from beyondcxr.training.symile_execution import (
    _CURRENT_EXECUTION_GUARD,
    _EXECUTION_GUARD,
    _LIFECYCLE_GUARD,
    ValidatedCurrentSymileExecution,
    ValidatedSymileExecution,
    ValidatedSymileLifecycle,
)
from beyondcxr.training.symile_families import (
    FINAL_NEURAL_MEMBER_SEEDS,
    FINAL_PACKAGE_POLICY,
    FINAL_TABULAR_SEED,
)
from beyondcxr.training.symile_final_packages import ValidatedFinalPackage
from beyondcxr.training.symile_test_data import (
    FrozenSymileTestData,
    HeldOutEvaluationProjection,
)
from beyondcxr.utils.private_predictions import (
    SYMILE_TEST_INFERENCE_POLICY,
    ValidatedPredictionEvidence,
    build_prediction_table,
)
from beyondcxr.utils.semantic_identity import canonical_json_bytes


def _final_provenance(*, neural=False):
    result = {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "dependency_lock_sha256": "b" * 64,
        "python_version": platform.python_version(),
        "mlflow_run_id": "synthetic-final-run",
    }
    if neural:
        result.update(
            runtime_provenance={
                "resolved_device": "cuda",
                "mixed_precision_effective": True,
                "pin_memory_effective": True,
                "torch_version": str(torch.__version__),
                "torchvision_version": "0.23.0",
                "torchxrayvision_version": "1.3.5",
                "cuda_runtime_version": "12.4",
                "cudnn_version": 9100,
                "gpu_device_name": "Synthetic GPU",
                "gpu_compute_capability": [8, 6],
            },
            loader_execution={
                "lifecycle": "reused",
                "num_workers": 2,
                "pin_memory": True,
                "batch_size": 2,
                "drop_last": False,
                "shuffle": False,
                "sampler": "epoch_permutation",
                "persistent_workers": True,
                "prefetch_factor": 2,
                "multiprocessing_context": "spawn",
            },
        )
    return result


def _freeze(tmp_path: Path, suffix: str = "a") -> ValidatedPretestFreeze:
    package_members = []
    for family, policy in FINAL_PACKAGE_POLICY.items():
        seeds = (FINAL_TABULAR_SEED,) if int(policy["members"]) == 1 else FINAL_NEURAL_MEMBER_SEEDS
        for seed in seeds:
            package_members.append(
                {
                    "package_id": "final-package-" + f"{len(package_members):064x}",
                    "family_id": family,
                    "seed": seed,
                }
            )
    document = campaign_control.pretest_freeze_document(
        bundle_id="bundle-" + suffix * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
        task={"task_id": "pneumonia_strict", "label_policy_version": "symile-pneumonia-strict-v1"},
        development_decision_id="development-decision-" + "d" * 64,
        package_members=package_members,
    )
    directory = tmp_path / document["pretest_freeze_id"]
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_bytes(canonical_json_bytes(document))
    raw = (directory / "manifest.json").read_bytes()
    return ValidatedPretestFreeze(
        directory,
        json.loads(raw),
        SimpleNamespace(
            manifest={
                "primary_thresholds": {
                    "youden_j": 0.5,
                    "target_sensitivity": 0.5,
                }
            }
        ),
        (),
        campaign_control._FREEZE_GUARD,
    )


def _synthetic_held_out_capability(
    tmp_path: Path,
    freeze: ValidatedPretestFreeze,
    *,
    neural_runtime: dict[str, object] | None = None,
) -> ValidatedHeldOutCapability:
    """Build non-accessing lifecycle coordinates for inference-unit tests."""
    runtime = neural_runtime or {
        "device_type": "cpu",
        "autocast_dtype": None,
        "deterministic_algorithms": "enabled_warn_only",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_runtime_version": None,
        "cudnn_version": None,
        "gpu_device_name": None,
        "gpu_compute_capability": None,
    }
    control = tmp_path / "project/private/control/symile"
    backup = tmp_path / "backup"
    control.mkdir(parents=True)
    backup.mkdir()
    execution_id = "symile-execution-" + "e" * 64
    execution_directory = control / "executions" / execution_id
    execution_directory.mkdir(parents=True)
    (execution_directory / "manifest.json").write_text("{}\n", encoding="utf-8")
    execution = ValidatedSymileExecution(
        execution_id,
        execution_directory,
        {"neural_runtime": runtime},
        _EXECUTION_GUARD,
    )
    (control / "active-execution.json").write_bytes(
        canonical_json_bytes({"schema_version": 1, "execution_id": execution_id})
    )
    current = ValidatedCurrentSymileExecution(
        execution, tmp_path / "project", _CURRENT_EXECUTION_GUARD
    )
    lifecycle = ValidatedSymileLifecycle(
        current,
        tmp_path / "project",
        tmp_path / "project/private",
        control,
        backup,
        _LIFECYCLE_GUARD,
    )
    document = {
        "schema_version": 1,
        "freeze_id": freeze.freeze_id,
        "execution_id": execution.execution_id,
        "opened_at_utc": "2026-01-01T00:00:00Z",
    }
    raw = canonical_json_bytes(document)
    local = control / "test-open.json"
    external = backup / "control/symile/test-open.json"
    external.parent.mkdir(parents=True)
    local.write_bytes(raw)
    external.write_bytes(raw)
    opening = ValidatedTestOpenRecord(
        local,
        external,
        freeze.freeze_id,
        execution.execution_id,
        raw,
        campaign_control._TEST_OPEN_GUARD,
    )
    return ValidatedHeldOutCapability(
        ValidatedSymileFrozenCampaign(
            freeze,
            lifecycle,
            campaign_control._FROZEN_CAMPAIGN_GUARD,
        ),
        opening,
        campaign_control._HELD_OUT_GUARD,
    )


def _synthetic_final_packages(
    tmp_path: Path,
    freeze: ValidatedPretestFreeze,
    *,
    family_authorities: dict[str, dict[str, object]] | None = None,
) -> tuple[ValidatedFinalPackage, ...]:
    family_members = tuple(
        (
            family,
            (FINAL_TABULAR_SEED,) if int(policy["members"]) == 1 else FINAL_NEURAL_MEMBER_SEEDS,
        )
        for family, policy in FINAL_PACKAGE_POLICY.items()
    )
    flattened = [(family, seed) for family, seeds in family_members for seed in seeds]
    cxr_by_seed = {
        seed: freeze.manifest["final_packages"][index]["package_id"]
        for index, (family, seed) in enumerate(flattened)
        if family == "cxr_densenet"
    }
    packages = []
    index = 0
    for family, seeds in family_members:
        for seed in seeds:
            frozen = freeze.manifest["final_packages"][index]
            packages.append(
                ValidatedFinalPackage(
                    tmp_path / f"package-{index}",
                    {
                        "final_package_id": frozen["package_id"],
                        "package_kind": "tabular" if family.startswith("labs_") else "neural",
                        "execution_scope": "full_development",
                        "seed_policy": seed,
                        "model_state_sha256": f"{index + 1:064x}",
                        "preprocessor_state_sha256": "f" * 64,
                        "input": (
                            family_authorities[family]["final_input"]
                            if family_authorities is not None
                            else {
                                "dataset": {
                                    "dataset_id": "symile",
                                    "bundle_id": freeze.manifest["bundle_id"],
                                    "cohort": ("official_train_plus_validation_strict_pneumonia"),
                                },
                                "task": dict(freeze.manifest["task"]),
                                "family": {"family_id": family},
                            }
                        ),
                        "family_development_id": (
                            None
                            if family == "labs_logistic"
                            else (
                                family_authorities[family]["development_id"]
                                if family_authorities is not None
                                else "development-" + f"{index + 1:064x}"
                            )
                        ),
                        "final_training_budget": (
                            family_authorities[family]["final_training_budget"]
                            if family_authorities is not None
                            else (None if family == "labs_logistic" else 3)
                        ),
                        "pretrained_scientific_identity": (
                            family_authorities[family]["pretrained_scientific_identity"]
                            if family_authorities is not None
                            else (
                                {
                                    "declared_name": "densenet121-res224-chex",
                                    "stable_identifier": "synthetic-weight",
                                    "sha256": "a" * 64,
                                }
                                if family == "cxr_densenet"
                                else None
                            )
                        ),
                        "source_cxr_package_id": (
                            cxr_by_seed[seed]
                            if family not in {"labs_logistic", "labs_lightgbm", "cxr_densenet"}
                            else None
                        ),
                        "execution_provenance": {
                            "git_commit": "f" * 40,
                            "git_dirty": False,
                            "dependency_lock_sha256": "1" * 64,
                        },
                    },
                )
            )
            index += 1
    return tuple(packages)


def _synthetic_final_family_authorities(
    development_ids: dict[str, str],
    freeze: ValidatedPretestFreeze | None = None,
) -> dict[str, dict[str, object]]:
    authorities = {
        family: {
            "development_id": development_ids[family],
            "final_training_budget": None if family == "labs_logistic" else 3,
            "final_input": final_packages._final_input_projection(
                load_symile_development_config(
                    Path("configs") / f"symile_{family}.yaml",
                    bundle_id="bundle-" + "a" * 64,
                    cv_assignment_id="cv-assignment-" + "c" * 64,
                )
            ),
            "pretrained_scientific_identity": (
                {
                    "declared_name": "densenet121-res224-chex",
                    "stable_identifier": "synthetic-weight",
                    "sha256": "a" * 64,
                }
                if family == "cxr_densenet"
                else None
            ),
        }
        for family in FINAL_PACKAGE_POLICY
    }
    if freeze is not None:
        for authority in authorities.values():
            final_input = authority["final_input"]
            final_input["dataset"] = {
                "dataset_id": "symile",
                "bundle_id": freeze.manifest["bundle_id"],
                "cohort": "official_train_plus_validation_strict_pneumonia",
            }
            final_input["task"] = dict(freeze.manifest["task"])
    return authorities


def _synthetic_test_predictions(
    tmp_path: Path,
    freeze: ValidatedPretestFreeze,
    packages: tuple[ValidatedFinalPackage, ...],
) -> tuple[ValidatedPredictionEvidence, ...]:
    sample_ids = [f"sample-{index:03d}" for index in range(110)]
    targets = np.asarray([0] * 55 + [1] * 55, dtype=np.int8)
    return tuple(
        ValidatedPredictionEvidence(
            tmp_path / f"evidence-{index}",
            {
                "prediction_id": "prediction-" + f"{index:064x}",
                "model_package_id": package.package_id,
                "authorized_by_freeze_id": freeze.freeze_id,
                "dataset_id": "symile",
                "bundle_id": freeze.manifest["bundle_id"],
                "task_id": freeze.manifest["task"]["task_id"],
                "label_policy_version": freeze.manifest["task"]["label_policy_version"],
                "scope": "test",
                "inference_policy": dict(SYMILE_TEST_INFERENCE_POLICY),
            },
            build_prediction_table(
                sample_ids,
                targets,
                np.linspace(-2.0 + index / 100, 2.0 + index / 100, 110),
            ),
        )
        for index, package in enumerate(packages)
    )


def _projection_for_freeze(
    freeze: ValidatedPretestFreeze, *, grouped: bool
) -> HeldOutEvaluationProjection:
    sample_ids = [f"sample-{index:03d}" for index in range(110)]
    targets = np.asarray([0] * 55 + [1] * 55, dtype=np.int8)
    subjects = np.repeat(np.arange(1, 56), 2) if grouped else np.arange(1, 111)
    data = object.__new__(FrozenSymileTestData)
    data._bundle = SimpleNamespace(bundle_id=str(freeze.manifest["bundle_id"]))
    data._canonical_frame = pd.DataFrame(
        {"sample_id": sample_ids, "target": targets, "subject_id": subjects}
    )
    data._capability = freeze
    return data.evaluation_projection()


def _repeat_oof(logits: list[float]) -> pd.DataFrame:
    rows = []
    for seed in FINAL_NEURAL_MEMBER_SEEDS:
        for sample_id, target, logit in zip(
            ("a", "b", "c", "d"), (0, 0, 1, 1), logits, strict=True
        ):
            rows.append(
                {
                    "sample_id": sample_id,
                    "target": target,
                    "logit": logit + seed / 100_000,
                    "repeat_seed": seed,
                }
            )
    return pd.DataFrame(rows).sort_values(["sample_id", "repeat_seed"]).reset_index(drop=True)
