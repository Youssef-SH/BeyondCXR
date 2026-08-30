from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

import beyondcxr.training.symile_execution as execution_control
from beyondcxr.models.cxr_baseline import PretrainedWeightIdentity
from beyondcxr.training.device import ResolvedDevice
from beyondcxr.training.execution import LoaderExecutionPolicy
from beyondcxr.training.symile_execution import (
    ValidatedSymileProspectiveMutation,
    build_symile_formal_preflight,
    publish_or_validate_symile_execution,
    symile_prospective_mutation,
    validate_current_symile_execution,
    validate_symile_lifecycle,
)
from beyondcxr.training.symile_formal import SYMILE_FORMAL_RUN_PLAN


@contextmanager
def prospective_symile_mutation(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[ValidatedSymileProspectiveMutation]:
    """Yield production-minted unopened authority for isolated publication tests."""
    repository = root
    repository.mkdir(parents=True, exist_ok=True)
    for spec in SYMILE_FORMAL_RUN_PLAN:
        destination = repository / spec.config_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            shutil.copyfile(spec.config_relative, destination)
    monkeypatch.setattr(execution_control, "git_revision", lambda path: ("a" * 40, False))
    monkeypatch.setattr(execution_control, "uv_lock_sha256", lambda path: "b" * 64)
    monkeypatch.setattr(execution_control, "resolve_symile_bundle", lambda *a, **k: object())
    monkeypatch.setattr(execution_control, "validate_symile_cv", lambda *a, **k: object())
    monkeypatch.setattr(
        execution_control,
        "resolve_device",
        lambda *a, **k: ResolvedDevice(
            torch.device("cuda"),
            "cuda",
            "cuda",
            True,
            True,
            True,
            "enabled",
            True,
            "2.8.0",
            "0.23.0",
            "1.3.5",
            "12.4",
            9100,
            "Synthetic GPU",
            0,
            (8, 6),
        ),
    )
    monkeypatch.setattr(
        execution_control,
        "fingerprint_pretrained_weights",
        lambda name: PretrainedWeightIdentity(name, "weights", "weights.pt", 1, "c" * 64),
    )
    monkeypatch.setattr(
        execution_control,
        "reused_loader_policy",
        lambda *, num_workers, pin_memory: LoaderExecutionPolicy("reused", num_workers, pin_memory),
    )
    preflight = build_symile_formal_preflight(
        repository, bundle_id="bundle-" + "a" * 64, cv_assignment_id="cv-assignment-" + "c" * 64
    )
    execution = publish_or_validate_symile_execution(preflight)
    current = validate_current_symile_execution(execution, preflight)
    backup = root.parent / f"{root.name}-prospective-backup"
    backup.mkdir(parents=True, exist_ok=True)
    lifecycle = validate_symile_lifecycle(
        current,
        repository_root=repository,
        private_root=repository / "private",
        backup_root=backup,
    )
    with symile_prospective_mutation(lifecycle) as authority:
        yield authority
