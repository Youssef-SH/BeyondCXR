"""Fabricate guarded capabilities only for narrow RSNA component tests.

These helpers do not exercise or prove the production authorization producer path.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import beyondcxr.training.rsna_campaign_control as campaign_control
from beyondcxr.training.config import (
    ExperimentConfig,
    normalized_scientific_config,
    require_runtime_seed,
)
from beyondcxr.training.device import (
    full_precision_neural_runtime_policy,
    neural_inference_runtime_policy,
    resolve_device,
)
from beyondcxr.training.rsna_campaign_control import (
    _CURRENT_EXECUTION_GUARD,
    _EXECUTION_GUARD,
    _FREEZE_GUARD,
    _HELD_OUT_GUARD,
    FrozenScientificPackage,
    ValidatedCurrentRsnaExecution,
    ValidatedRsnaExecution,
    ValidatedRsnaFreeze,
    ValidatedRsnaHeldOutCapability,
)


def authorize_rsna_package(
    config: ExperimentConfig,
    package_id: str,
    *,
    monkeypatch: pytest.MonkeyPatch,
    model_path: Path | None = None,
    held_out_sample_ids: Sequence[str] | None = None,
    held_out_targets: Sequence[int] | None = None,
) -> ValidatedRsnaHeldOutCapability:
    """Return a one-member capability matching a component-test package/config pair."""
    return authorize_rsna_packages(
        ((config, package_id),),
        monkeypatch=monkeypatch,
        model_path=model_path,
        held_out_sample_ids=held_out_sample_ids,
        held_out_targets=held_out_targets,
    )


def rsna_test_runtime(config: ExperimentConfig):
    """Resolve the runtime represented by a component-test formal capability."""
    if config.neural is not None:
        return resolve_device(
            config.runtime.device,
            mixed_precision=config.neural.mixed_precision,
            pin_memory_policy=config.runtime.pin_memory_policy,
        )
    return resolve_device("cpu", mixed_precision=False, pin_memory_policy="disabled")


def authorize_rsna_packages(
    members: Sequence[tuple[ExperimentConfig, str]],
    *,
    monkeypatch: pytest.MonkeyPatch,
    model_path: Path | None = None,
    held_out_sample_ids: Sequence[str] | None = None,
    held_out_targets: Sequence[int] | None = None,
) -> ValidatedRsnaHeldOutCapability:
    """Return a fabricated bounded capability for component-test package/config pairs."""
    values = tuple(members)
    if not values:
        raise ValueError("Component-test authorization requires at least one package")
    reference = values[0][0]
    runtime = rsna_test_runtime(reference)
    execution = ValidatedRsnaExecution(
        execution_id="rsna-execution-" + "0" * 64,
        directory=Path("unused-control"),
        manifest={
            "dataset": {
                "dataset_id": reference.dataset.dataset_id,
                "bundle_id": reference.dataset.bundle_id,
            },
            "task": {
                "task_id": reference.task.task_id,
                "label_policy_version": reference.task.label_policy_version,
            },
            "configs": [
                {
                    "family_id": config.family.family_id,
                    "seed": require_runtime_seed(config),
                    "config_path": f"configs/{config.family.family_id}.yaml",
                    "normalized_config": normalized_scientific_config(config),
                }
                for config, _ in values
            ],
            "training_evaluation_runtime": neural_inference_runtime_policy(runtime),
            "localization_runtime": full_precision_neural_runtime_policy(runtime),
        },
        _guard=_EXECUTION_GUARD,
    )
    model_root = reference.runtime.model_directory.resolve()
    packages = tuple(
        FrozenScientificPackage(
            model_package_id=package_id,
            model_path=model_path or model_root / "packages" / package_id,
        )
        for _, package_id in values
    )
    bundle_directory = (
        reference.runtime.manifest_directory / "rsna" / "bundles" / reference.dataset.bundle_id
    ).resolve()
    if held_out_sample_ids is not None or held_out_targets is not None:
        if held_out_sample_ids is None or held_out_targets is None:
            raise ValueError("Synthetic held-out IDs and targets must be supplied together")
        if len(held_out_sample_ids) != len(held_out_targets):
            raise ValueError("Synthetic held-out IDs and targets differ in length")
        bundle_directory.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "sample_id": tuple(held_out_sample_ids),
                    "split_name": ("test",) * len(held_out_sample_ids),
                }
            ),
            bundle_directory / "splits.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    "sample_id": tuple(held_out_sample_ids),
                    "task_id": (reference.task.task_id,) * len(held_out_sample_ids),
                    "label_value": tuple(int(value) for value in held_out_targets),
                }
            ),
            bundle_directory / "labels.parquet",
        )
    freeze = ValidatedRsnaFreeze(
        "rsna-freeze-" + "f" * 64,
        Path("unused-freeze"),
        packages,
        model_root,
        bundle_directory,
        _FREEZE_GUARD,
    )
    monkeypatch.setattr(
        campaign_control,
        "validate_freeze",
        lambda directory, *, model_root, bundle_directory: freeze,
    )

    def validate_component_execution(
        directory: str | Path, *, expected_execution_id: str | None = None
    ) -> ValidatedRsnaExecution:
        if expected_execution_id not in {None, execution.execution_id}:
            raise ValueError("Component-test execution identity differs")
        return execution

    monkeypatch.setattr(
        campaign_control,
        "validate_execution",
        validate_component_execution,
    )
    monkeypatch.setattr(
        campaign_control,
        "validate_rsna_execution_freeze_science",
        lambda execution, freeze: None,
    )
    current = ValidatedCurrentRsnaExecution(
        execution,
        None,  # type: ignore[arg-type]
        _CURRENT_EXECUTION_GUARD,
    )
    capability = ValidatedRsnaHeldOutCapability(freeze, current, _HELD_OUT_GUARD)

    def require_component_capability(value: object) -> ValidatedRsnaHeldOutCapability:
        if value is not capability:
            raise PermissionError("Component test requires its bounded fabricated capability")
        return capability

    monkeypatch.setattr(
        campaign_control,
        "require_validated_rsna_held_out",
        require_component_capability,
    )
    return capability
