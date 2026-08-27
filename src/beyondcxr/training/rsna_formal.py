"""Canonical coordinates for one validated formal RSNA campaign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from beyondcxr.models.cxr_baseline import PretrainedWeightIdentity
from beyondcxr.training.config import ExperimentConfig
from beyondcxr.training.device import ResolvedDevice
from beyondcxr.training.execution import LoaderExecutionPolicy
from beyondcxr.training.rsna_datasets import RsnaDataset

_FORMAL_PLAN_GUARD = object()
RSNA_RUN_REPORT_RELATIVE = Path("rsna/runs")


def rsna_run_report_root(report_root: str | Path) -> Path:
    """Return the single canonical root for formal RSNA training reports."""
    return Path(report_root) / RSNA_RUN_REPORT_RELATIVE


@dataclass(frozen=True)
class RsnaFormalRunSpec:
    """One position in the immutable formal family/seed matrix."""

    family_id: str
    seed: int
    config_relative: Path


RSNA_FORMAL_RUN_PLAN = (
    RsnaFormalRunSpec("metadata_logistic", 42, Path("configs/rsna_metadata_logistic.yaml")),
    RsnaFormalRunSpec("metadata_lightgbm", 42, Path("configs/rsna_metadata_lightgbm.yaml")),
    RsnaFormalRunSpec("cxr_densenet", 17, Path("configs/rsna_cxr_densenet.yaml")),
    RsnaFormalRunSpec("cxr_densenet", 42, Path("configs/rsna_cxr_densenet.yaml")),
    RsnaFormalRunSpec("cxr_densenet", 2026, Path("configs/rsna_cxr_densenet.yaml")),
    RsnaFormalRunSpec("cxr_metadata_concat", 17, Path("configs/rsna_cxr_metadata_concat.yaml")),
    RsnaFormalRunSpec("cxr_metadata_concat", 42, Path("configs/rsna_cxr_metadata_concat.yaml")),
    RsnaFormalRunSpec("cxr_metadata_concat", 2026, Path("configs/rsna_cxr_metadata_concat.yaml")),
)
RSNA_NEURAL_SEEDS = tuple(
    spec.seed for spec in RSNA_FORMAL_RUN_PLAN if spec.family_id == "cxr_densenet"
)


@dataclass(frozen=True)
class RsnaAuthorityCoordinates:
    dataset_id: str
    bundle_id: str
    bundle_manifest_sha256: str
    split_assignment_id: str
    task_id: str
    label_policy_version: str
    bundle_directory: Path


@dataclass(frozen=True)
class RsnaFormalRoots:
    repository_root: Path
    source_root: Path
    manifest_root: Path
    cache_root: Path
    model_root: Path
    report_root: Path
    private_root: Path
    control_root: Path
    export_root: Path
    backup_root: Path
    tracking_database: Path

    @property
    def tracking_uri(self) -> str:
        """Return the MLflow URI derived from the one formal database path."""
        return f"sqlite:///{self.tracking_database.as_posix()}"


@dataclass(frozen=True)
class ValidatedRsnaPlan:
    """Read-only formal inputs proven coherent before campaign mutation."""

    authority: RsnaAuthorityCoordinates
    roots: RsnaFormalRoots
    runs: tuple[RsnaFormalRunSpec, ...]
    configs: tuple[ExperimentConfig, ...]
    dataset: RsnaDataset
    git_commit: str
    dependency_lock_sha256: str
    pretrained_weight: PretrainedWeightIdentity
    runtime: ResolvedDevice
    training_execution: LoaderExecutionPolicy
    evaluation_execution: LoaderExecutionPolicy
    _guard: object


def require_validated_rsna_plan(value: object) -> ValidatedRsnaPlan:
    """Reject formal plans not constructed by the complete campaign preflight."""
    if not isinstance(value, ValidatedRsnaPlan) or value._guard is not _FORMAL_PLAN_GUARD:
        raise ValueError("A genuine prevalidated RSNA formal plan is required")
    return value
