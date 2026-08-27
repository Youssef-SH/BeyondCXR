"""Fabricate guarded capabilities only for narrow RSNA component tests.

These helpers do not exercise or prove the production authorization producer path.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

import beyondcxr.training.rsna_campaign_control as campaign_control
from beyondcxr.training.config import ExperimentConfig, require_runtime_seed
from beyondcxr.training.device import (
    full_precision_neural_runtime_policy,
    neural_inference_runtime_policy,
    resolve_device,
)
from beyondcxr.training.rsna_campaign_control import (
    _EXECUTION_GUARD,
    _PACKAGE_FREEZE_GUARD,
    FrozenTrainingPackage,
    ValidatedRsnaExecution,
    ValidatedRsnaPackageFreeze,
)


def authorize_rsna_package(
    config: ExperimentConfig,
    package_id: str,
    *,
    monkeypatch: pytest.MonkeyPatch,
    model_path: Path | None = None,
    report_path: Path | None = None,
) -> ValidatedRsnaPackageFreeze:
    """Return a one-member capability matching a component-test package/config pair."""
    return authorize_rsna_packages(
        ((config, package_id),),
        monkeypatch=monkeypatch,
        model_path=model_path,
        report_path=report_path,
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
    report_path: Path | None = None,
) -> ValidatedRsnaPackageFreeze:
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
            "configs": [
                {
                    "family_id": config.family.family_id,
                    "seed": require_runtime_seed(config),
                    "config_source_sha256": config.config_source_sha256,
                    "config_semantic_sha256": config.config_semantic_sha256,
                }
                for config, _ in values
            ],
            "training_evaluation_runtime": neural_inference_runtime_policy(runtime),
            "localization_runtime": full_precision_neural_runtime_policy(runtime),
        },
        _guard=_EXECUTION_GUARD,
    )
    packages = tuple(
        FrozenTrainingPackage(
            run_id=f"component-test-run-{index}",
            model_package_id=package_id,
            model_path=model_path or Path("unused-model"),
            artifact_directory=report_path or Path("unused-report"),
            report_sha256="0" * 64,
        )
        for index, (_, package_id) in enumerate(values)
    )
    model_root = Path("unused-model-root").resolve()
    report_root = Path("unused-report-root").resolve()
    bundle_directory = Path("unused-bundle").resolve()
    capability = ValidatedRsnaPackageFreeze(
        execution,
        packages,
        model_root,
        report_root,
        bundle_directory,
        _PACKAGE_FREEZE_GUARD,
    )
    monkeypatch.setattr(campaign_control, "_reauthenticate_execution", lambda value: value)
    monkeypatch.setattr(
        campaign_control,
        "validate_package_freeze",
        lambda execution, *, model_root, report_root, bundle_directory: capability,
    )
    return capability
