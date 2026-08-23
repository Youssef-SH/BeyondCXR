from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import beyondcxr.training.symile_campaign_control as campaign_control
import beyondcxr.training.symile_final_packages as final_packages
import beyondcxr.training.symile_statistics as symile_statistics
from beyondcxr.training.config import load_symile_development_config
from beyondcxr.training.symile_campaign_control import (
    PRETEST_FREEZE_PREFIX,
    ValidatedPretestFreeze,
)
from beyondcxr.training.symile_families import (
    FINAL_NEURAL_MEMBER_SEEDS,
    FINAL_PACKAGE_COUNT,
    FINAL_PACKAGE_POLICY,
    FINAL_TABULAR_SEED,
)
from beyondcxr.training.symile_final_packages import (
    ValidatedFinalPackage,
)
from beyondcxr.training.symile_test_data import (
    HeldOutEvaluationProjection,
)
from beyondcxr.utils.package_identity import canonical_scientific_id
from beyondcxr.utils.private_predictions import (
    SYMILE_TEST_INFERENCE_POLICY,
    ValidatedPredictionEvidence,
    build_prediction_table,
)


def _final_provenance(*, neural=False):
    result = {
        "git_commit": "a" * 40,
        "git_dirty": False,
        "dependency_lock_sha256": "b" * 64,
        "mlflow_run_id": "synthetic-final-run",
    }
    if neural:
        result.update(
            runtime_provenance={"pin_memory_effective": False},
            loader_execution={
                "lifecycle": "reused",
                "num_workers": 0,
                "pin_memory": False,
                "batch_size": 2,
                "drop_last": False,
                "shuffle": False,
                "sampler": "epoch_permutation",
                "persistent_workers": False,
                "prefetch_factor": None,
                "multiprocessing_context": None,
            },
        )
    return result


def _freeze(
    tmp_path: Path, suffix: str = "a", *, neural_runtime: dict[str, object] | None = None
) -> ValidatedPretestFreeze:
    neural_runtime = neural_runtime or {
        "device_type": "cpu",
        "autocast_dtype": None,
        "cuda_runtime_version": None,
        "cudnn_version": None,
        "gpu_device_name": None,
        "gpu_compute_capability": None,
    }
    semantic = {
        "bundle": {
            "bundle_id": "bundle-" + suffix * 64,
            "bundle_manifest_sha256": "b" * 64,
            "split_assignment_id": "split-assignment-" + "c" * 64,
        },
        "task": {
            "task_id": "pneumonia_strict",
            "label_policy_version": "symile-pneumonia-strict-v1",
        },
        "ecg_extension_result_id": "ecg-extension-result-" + "d" * 64,
        "final_packages": [
            {
                "package_id": "final-package-" + f"{index:064x}",
                "manifest_sha256": "e" * 64,
            }
            for index in range(FINAL_PACKAGE_COUNT)
        ],
        "held_out_policy": {
            "metric_policy": symile_statistics.METRIC_POLICY,
            "bootstrap": dict(symile_statistics.BOOTSTRAP_POLICY),
            "neural_inference_runtime": neural_runtime,
        },
        "primary_thresholds": {"youden_j": 0.5, "target_sensitivity": 0.25},
        "science_git_commit": "f" * 40,
        "dependency_lock_sha256": "0" * 64,
    }
    freeze_id = canonical_scientific_id(PRETEST_FREEZE_PREFIX, semantic)
    directory = tmp_path / freeze_id
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "pretest_freeze_schema_version": 1,
                "pretest_freeze_id": freeze_id,
                **semantic,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    raw = (directory / "manifest.json").read_bytes()
    return ValidatedPretestFreeze(
        directory,
        json.loads(raw),
        hashlib.sha256(raw).hexdigest(),
        campaign_control._CAPABILITY_GUARD,
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
                        "seed_policy": seed,
                        "input": (
                            family_authorities[family]["final_input"]
                            if family_authorities is not None
                            else {
                                "dataset": {
                                    "dataset_id": "symile",
                                    "bundle_id": freeze.manifest["bundle"]["bundle_id"],
                                    "split_assignment_id": freeze.manifest["bundle"][
                                        "split_assignment_id"
                                    ],
                                    "cohort": ("official_train_plus_validation_strict_pneumonia"),
                                },
                                "task": dict(freeze.manifest["task"]),
                                "family": {"family_id": family},
                            }
                        ),
                        "bundle_manifest_sha256": freeze.manifest["bundle"][
                            "bundle_manifest_sha256"
                        ],
                        "family_development_id": (
                            family_authorities[family]["development_id"]
                            if family_authorities is not None
                            else (
                                None
                                if family == "labs_logistic"
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
                            "git_commit": freeze.manifest["science_git_commit"],
                            "git_dirty": False,
                            "dependency_lock_sha256": freeze.manifest["dependency_lock_sha256"],
                        },
                    },
                    frozen["manifest_sha256"],
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
            "development_id": None if family == "labs_logistic" else development_ids[family],
            "final_training_budget": None if family == "labs_logistic" else 3,
            "final_input": final_packages._final_input_projection(
                load_symile_development_config(Path("configs") / f"symile_{family}.yaml")
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
                "bundle_id": freeze.manifest["bundle"]["bundle_id"],
                "split_assignment_id": freeze.manifest["bundle"]["split_assignment_id"],
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
                "authorized_by_pretest_freeze_id": freeze.freeze_id,
                "dataset_id": "symile",
                "bundle_id": freeze.manifest["bundle"]["bundle_id"],
                "split_assignment_id": freeze.manifest["bundle"]["split_assignment_id"],
                "task_id": freeze.manifest["task"]["task_id"],
                "label_policy_version": freeze.manifest["task"]["label_policy_version"],
                "scope": "test",
                "inference_policy": dict(SYMILE_TEST_INFERENCE_POLICY),
            },
            "7" * 64,
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
    return HeldOutEvaluationProjection(
        dataset_id="symile",
        bundle_id=str(freeze.manifest["bundle"]["bundle_id"]),
        split_assignment_id=str(freeze.manifest["bundle"]["split_assignment_id"]),
        task_id=str(freeze.manifest["task"]["task_id"]),
        label_policy_version=str(freeze.manifest["task"]["label_policy_version"]),
        scope="test",
        inference_policy=dict(SYMILE_TEST_INFERENCE_POLICY),
        freeze_id=freeze.freeze_id,
        _frame=pd.DataFrame({"sample_id": sample_ids, "target": targets, "subject_id": subjects}),
    )


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
