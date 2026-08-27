"""Execute the formal RSNA campaign from one exact prepublished authority."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO, cast

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.rsna_artifacts import (
    BUNDLES_DIRECTORY,
    validate_bundle_directory,
    validate_bundle_reference,
)
from beyondcxr.data.rsna_audit import generate_rsna_audit
from beyondcxr.data.rsna_cxr_cache import preprocessing_identity, validate_cxr_source_availability
from beyondcxr.models.cxr_baseline import fingerprint_pretrained_weights
from beyondcxr.training.config import (
    ExperimentConfig,
    load_experiment_config,
    require_runtime_seed,
    with_runtime,
)
from beyondcxr.training.device import ResolvedDevice, resolve_device
from beyondcxr.training.execution import (
    LoaderExecutionPolicy,
    one_shot_loader_policy,
    reused_loader_policy,
)
from beyondcxr.training.operational_validation import (
    validate_existing_sqlite_database,
    validate_writable_directory_destination,
    validate_writable_file_destination,
)
from beyondcxr.training.preservation import (
    export_and_verify,
    validate_preservation_paths,
)
from beyondcxr.training.rsna_campaign_control import (
    load_evaluation_record,
    publish_evaluation_record,
    publish_or_validate_execution,
    publish_or_validate_package_freeze,
    validate_package_freeze,
)
from beyondcxr.training.rsna_compare import ComparisonResult, regenerate_comparison
from beyondcxr.training.rsna_datasets import RsnaDataset, prepare_rsna_cxr_cache
from beyondcxr.training.rsna_evaluate import evaluate_model_package
from beyondcxr.training.rsna_formal import (
    _FORMAL_PLAN_GUARD,
    RSNA_FORMAL_RUN_PLAN,
    RsnaAuthorityCoordinates,
    RsnaFormalRoots,
    ValidatedRsnaPlan,
)
from beyondcxr.training.rsna_localize import generate_localization_report
from beyondcxr.training.rsna_preservation import (
    campaign_export_members,
    validate_restored_rsna_campaign,
)
from beyondcxr.training.rsna_registry import get_dataset
from beyondcxr.training.rsna_seed_summary import publish_seed_summary
from beyondcxr.training.rsna_train_cxr import train_cxr_experiment
from beyondcxr.training.rsna_train_fusion import train_fusion_experiment
from beyondcxr.training.rsna_train_metadata import MetadataModelResult, train_metadata_experiment
from beyondcxr.utils.mlflow_utils import (
    discover_repository_root,
    git_revision,
    uv_lock_sha256,
)
from beyondcxr.utils.operational_logging import (
    configure_logging,
    get_operational_logger,
    log_event,
    timed_phase,
)

_MINIMUM_FREE_BYTES = 16 * 1024**3
_LOGGER = get_operational_logger(__name__)


@dataclass(frozen=True)
class CampaignResult:
    """Exact produced identities and final transport artifacts."""

    campaign_id: str
    model_package_ids: tuple[str, ...]
    evaluation_ids: tuple[str, ...]
    training_run_ids: tuple[str, ...]
    evaluation_run_ids: tuple[str, ...]
    archive_path: Path
    checksum_path: Path
    campaign_log_path: Path


class _TeeStream:
    """Write operational records immediately to terminal and durable file."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, value: str) -> int:
        for stream in self._streams:
            if stream.closed:
                continue
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self._streams:
            if stream.closed:
                continue
            stream.flush()


def execute_rsna_campaign(*, backup_root: str | Path) -> CampaignResult:
    """Run or resume one exact RSNA execution after mutation-free preflight."""
    plan = _preflight_rsna_campaign(backup_root=backup_root)
    campaign_id = "campaign-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    campaign_directory = plan.roots.report_root / "rsna" / "campaigns" / campaign_id
    campaign_directory.mkdir(parents=True, exist_ok=False)
    log_path = campaign_directory / "execution.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_stream:
        configure_logging("INFO", stream=_TeeStream(sys.stderr, log_stream))
        campaign_started_at = time.perf_counter()
        log_event(
            _LOGGER,
            "campaign_started",
            campaign_id=campaign_id,
            phase="preparation",
        )
        try:
            execution = publish_or_validate_execution(plan)
            with timed_phase(_LOGGER, "dataset_audit"):
                bundle_id = plan.authority.bundle_id
                generate_rsna_audit(
                    plan.roots.manifest_root,
                    plan.roots.report_root / "rsna" / "audit",
                    bundle_id=bundle_id,
                )
                audit_directory = plan.roots.report_root / "rsna" / "audit" / bundle_id
            cxr_configs = tuple(
                config
                for spec, config in zip(plan.runs, plan.configs, strict=True)
                if spec.family_id == "cxr_densenet"
            )
            fusion_configs = tuple(
                config
                for spec, config in zip(plan.runs, plan.configs, strict=True)
                if spec.family_id == "cxr_metadata_concat"
            )
            cxr_reference = cxr_configs[0]
            transform = _transform(cxr_reference, training=False)
            with timed_phase(_LOGGER, "cxr_cache_preparation"):
                cache = prepare_rsna_cxr_cache(
                    plan.dataset, cxr_reference, transform, cache_root=plan.roots.cache_root
                )
            freeze_path = execution.directory / "package-freeze.json"
            if freeze_path.exists() or freeze_path.is_symlink():
                freeze = validate_package_freeze(
                    execution,
                    model_root=plan.roots.model_root,
                    report_root=plan.roots.report_root,
                    bundle_directory=plan.authority.bundle_directory,
                )
                log_event(_LOGGER, "training_resumed_from_package_freeze", total=8)
            else:
                metadata_training = _train_metadata(plan)
                cxr_training = tuple(
                    train_cxr_experiment(
                        config,
                        tracking_uri=plan.roots.tracking_uri,
                        cache=cache,
                        execution=plan.training_execution,
                        runtime=plan.runtime,
                    )
                    for config in cxr_configs
                )
                fusion_training = tuple(
                    train_fusion_experiment(
                        config,
                        source_cxr_package_id=source.model_package_id,
                        tracking_uri=plan.roots.tracking_uri,
                        cache=cache,
                        execution=plan.training_execution,
                        runtime=plan.runtime,
                    )
                    for config, source in zip(fusion_configs, cxr_training, strict=True)
                )
                completed = (*metadata_training, *cxr_training, *fusion_training)
                freeze = publish_or_validate_package_freeze(
                    plan=plan,
                    execution=execution,
                    results=completed,
                )
            training_results = freeze.packages
            log_event(_LOGGER, "training_boundary_completed", total=len(training_results))
            evaluation_results_list = []
            for result, config in zip(
                training_results,
                plan.configs,
                strict=True,
            ):
                evaluation = load_evaluation_record(
                    execution,
                    package_id=result.model_package_id,
                    report_root=plan.roots.report_root,
                    private_root=plan.roots.private_root,
                    model_root=plan.roots.model_root,
                )
                if evaluation is None:
                    evaluation = evaluate_model_package(
                        result.model_package_id,
                        authorization=freeze,
                        evaluation_config=config,
                        runtime=plan.runtime,
                        tracking_uri=plan.roots.tracking_uri,
                        model_directory=plan.roots.model_root,
                        cache=cache,
                        execution=plan.evaluation_execution,
                        private_output_directory=plan.roots.private_root,
                        report_directory=plan.roots.report_root,
                    )
                    publish_evaluation_record(execution, evaluation)
                evaluation_results_list.append(evaluation)
            evaluation_results = tuple(evaluation_results_list)
            cxr_tests = evaluation_results[2:5]
            fusion_tests = evaluation_results[5:]
            cxr_evaluation_ids = tuple(result.evaluation_id for result in cxr_tests)
            fusion_evaluation_ids = tuple(result.evaluation_id for result in fusion_tests)
            cxr_summary = publish_seed_summary(
                cxr_evaluation_ids,
                output_directory=plan.roots.report_root,
                model_directory=plan.roots.model_root,
                private_directory=plan.roots.private_root,
            )
            fusion_summary = publish_seed_summary(
                fusion_evaluation_ids,
                output_directory=plan.roots.report_root,
                model_directory=plan.roots.model_root,
                private_directory=plan.roots.private_root,
            )
            localization = generate_localization_report(
                cxr_evaluation_ids,
                authorization=freeze,
                output_directory=plan.roots.report_root,
                model_directory=plan.roots.model_root,
                private_directory=plan.roots.private_root,
                cache=cache,
                runtime=plan.runtime,
            )
            comparison = regenerate_comparison(
                tuple(result.evaluation_id for result in evaluation_results),
                output_directory=plan.roots.report_root,
                private_directory=plan.roots.private_root,
                model_directory=plan.roots.model_root,
            )
            with timed_phase(_LOGGER, "output_validation"):
                _validate_outputs(
                    training_results,
                    evaluation_results,
                    audit_directory,
                    cxr_summary.report_directory,
                    fusion_summary.report_directory,
                    localization,
                    comparison,
                    log_path,
                    roots=plan.roots,
                )
            log_event(_LOGGER, "campaign_ready_for_export", phase="export")
            log_stream.flush()
            archive_started_at = time.perf_counter()
            summaries = (cxr_summary, fusion_summary)
            archive = export_and_verify(
                members=campaign_export_members(
                    package_freeze=freeze,
                    evaluations=evaluation_results,
                    summaries=summaries,
                    localization=localization,
                    comparison=comparison,
                    audit_directory=audit_directory,
                    campaign_log=log_path,
                    private_root=plan.roots.private_root,
                ),
                export_root=plan.roots.export_root,
                backup_root=plan.roots.backup_root,
                export_name=f"rsna-{execution.execution_id}-{campaign_id}",
                restoration_validator=validate_restored_rsna_campaign,
            )
            log_event(
                _LOGGER,
                "archive_created",
                elapsed_s=time.perf_counter() - archive_started_at,
            )
            checksum = archive.with_suffix(".zip.sha256")
            log_event(
                _LOGGER,
                "campaign_succeeded",
                campaign_id=campaign_id,
                phase="complete",
                elapsed_s=time.perf_counter() - campaign_started_at,
            )
        except BaseException as exc:
            log_event(
                _LOGGER,
                "campaign_failed",
                campaign_id=campaign_id,
                phase="failed",
                error_type=type(exc).__name__,
                elapsed_s=time.perf_counter() - campaign_started_at,
            )
            raise
    return CampaignResult(
        campaign_id=campaign_id,
        model_package_ids=tuple(result.model_package_id for result in training_results),
        evaluation_ids=tuple(result.evaluation_id for result in evaluation_results),
        training_run_ids=tuple(result.run_id for result in training_results),
        evaluation_run_ids=tuple(result.mlflow_run_id for result in evaluation_results),
        archive_path=archive,
        checksum_path=checksum,
        campaign_log_path=log_path,
    )


def _train_metadata(plan: ValidatedRsnaPlan) -> tuple[MetadataModelResult, MetadataModelResult]:
    configs = tuple(
        config
        for spec, config in zip(plan.runs, plan.configs, strict=True)
        if spec.family_id in {"metadata_logistic", "metadata_lightgbm"}
    )
    if len(configs) != 2:
        raise ManifestBuildError("RSNA formal plan must contain exactly two metadata runs")
    first_config, second_config = configs
    return (
        train_metadata_experiment(first_config, tracking_uri=plan.roots.tracking_uri),
        train_metadata_experiment(second_config, tracking_uri=plan.roots.tracking_uri),
    )


def _preflight_rsna_campaign(*, backup_root: str | Path) -> ValidatedRsnaPlan:
    """Validate every formal input before creating campaign-owned output."""
    repository_root = discover_repository_root()
    if Path.cwd().resolve() != repository_root:
        raise ManifestBuildError("Formal RSNA campaign must run from the repository root")
    _validate_canonical_rsna_layout(repository_root)
    source_root = repository_root / "data/raw/rsna/extracted"
    manifest_root = repository_root / "data/manifests"
    cache_root = repository_root / "data/cache/rsna"
    model_root = repository_root / "models/rsna"
    report_root = repository_root / "reports"
    private_root = repository_root / "private"
    control_root = private_root / "control/rsna"
    export_root = repository_root / "outbox"
    backup_input = Path(backup_root)
    backup = validate_writable_directory_destination(backup_input, "RSNA backup destination")
    if backup.is_relative_to(repository_root):
        raise ManifestBuildError("RSNA backup must reside outside the repository root")
    validate_preservation_paths(
        sources=(
            source_root,
            manifest_root,
            model_root,
            report_root / "rsna",
            private_root,
        ),
        export_root=export_root,
        backup_root=backup,
    )
    tracking_database = repository_root / "mlflow.db"
    for root in (
        cache_root,
        report_root,
        model_root,
        private_root,
        export_root,
    ):
        validate_writable_directory_destination(root, "RSNA campaign directory destination")
    validate_writable_file_destination(
        tracking_database, "RSNA campaign tracking database destination"
    )
    validate_existing_sqlite_database(tracking_database, "RSNA campaign tracking database")
    commit, dirty = git_revision(repository_root)
    if dirty:
        raise ManifestBuildError("Formal RSNA campaign requires a clean Git worktree")
    lock_hash = uv_lock_sha256(repository_root / "uv.lock")
    if not source_root.is_dir():
        raise FileNotFoundError(f"RSNA raw dataset is missing: {source_root}")
    required_raw = (
        source_root / "stage_2_train_images",
        source_root / "stage_2_train_labels.csv",
        source_root / "stage_2_detailed_class_info.csv",
    )
    if not all(path.is_dir() if path.suffix == "" else path.is_file() for path in required_raw):
        raise FileNotFoundError("RSNA raw dataset is incomplete")
    config_paths = tuple(repository_root / spec.config_relative for spec in RSNA_FORMAL_RUN_PLAN)
    if missing := [path for path in config_paths if not path.is_file()]:
        raise FileNotFoundError(f"RSNA campaign configs are missing: {missing}")
    configs = tuple(
        with_runtime(
            load_experiment_config(path),
            seed=spec.seed,
            manifest_directory=manifest_root,
            source_root=source_root,
            model_directory=model_root,
            report_directory=report_root,
            private_output_directory=private_root,
            device="cuda",
        )
        for spec, path in zip(RSNA_FORMAL_RUN_PLAN, config_paths, strict=True)
    )
    _validate_neural_campaign_configs(configs)
    _validate_config_agreement(configs)
    reference = configs[0]
    bundle_directory = (
        reference.runtime.manifest_directory
        / "rsna"
        / BUNDLES_DIRECTORY
        / reference.dataset.bundle_id
    )
    validated = validate_bundle_reference(
        bundle_directory,
        expected_bundle_id=reference.dataset.bundle_id,
        expected_manifest_sha256=reference.dataset.bundle_manifest_sha256,
    )
    manifest = validate_bundle_directory(
        bundle_directory,
        expected_bundle_id=reference.dataset.bundle_id,
    )
    if (
        validated.manifest_sha256 != reference.dataset.bundle_manifest_sha256
        or manifest["membership"]["split"]["split_assignment_id"]
        != reference.dataset.split_assignment_id
    ):
        raise ManifestBuildError("Configured RSNA authority lineage is inconsistent")
    dataset = get_dataset("rsna")
    if not isinstance(dataset, RsnaDataset):
        raise TypeError("RSNA campaign requires the concrete RSNA dataset adapter")
    cxr_reference = next(
        config
        for spec, config in zip(RSNA_FORMAL_RUN_PLAN, configs, strict=True)
        if spec.family_id == "cxr_densenet"
    )
    cache_input = dataset.load_image_cache(cxr_reference)
    if cxr_reference.runtime.source_root is None:
        raise ManifestBuildError("RSNA campaign requires an explicit source root")
    validate_cxr_source_availability(
        cache_input.frame,
        dataset_root=cxr_reference.runtime.source_root,
    )
    if shutil.disk_usage(Path.cwd()).free < _MINIMUM_FREE_BYTES:
        raise OSError("RSNA campaign requires at least 16 GiB of free workspace storage")
    runtime = _required_cuda_runtime(cxr_reference)
    weight = fingerprint_pretrained_weights("densenet121-res224-chex")
    roots = RsnaFormalRoots(
        repository_root=repository_root,
        source_root=source_root,
        manifest_root=manifest_root,
        cache_root=cache_root,
        model_root=model_root,
        report_root=report_root,
        private_root=private_root,
        control_root=control_root,
        export_root=export_root,
        backup_root=backup,
        tracking_database=tracking_database,
    )
    authority = RsnaAuthorityCoordinates(
        dataset_id=reference.dataset.dataset_id,
        bundle_id=reference.dataset.bundle_id,
        bundle_manifest_sha256=reference.dataset.bundle_manifest_sha256,
        split_assignment_id=reference.dataset.split_assignment_id,
        task_id=reference.task.task_id,
        label_policy_version=reference.task.label_policy_version,
        bundle_directory=bundle_directory,
    )
    return ValidatedRsnaPlan(
        authority=authority,
        roots=roots,
        runs=RSNA_FORMAL_RUN_PLAN,
        configs=configs,
        dataset=dataset,
        git_commit=commit,
        dependency_lock_sha256=lock_hash,
        pretrained_weight=weight,
        runtime=runtime,
        training_execution=_training_execution_policy(cxr_reference, runtime),
        evaluation_execution=one_shot_loader_policy(pin_memory=runtime.pin_memory_effective),
        _guard=_FORMAL_PLAN_GUARD,
    )


def _validate_canonical_rsna_layout(repository_root: Path) -> None:
    """Reject redirected, aliased, or mistyped checkout-owned RSNA state."""
    root = repository_root.resolve()
    directory_roots = tuple(
        repository_root / path
        for path in (
            "data/manifests/rsna",
            "data/cache/rsna",
            "models/rsna/packages",
            "reports/rsna/runs",
            "reports/rsna/evaluations",
            "reports/rsna/seed-summaries",
            "reports/rsna/localization",
            "reports/rsna/comparisons",
            "reports/rsna/audit",
            "reports/rsna/campaigns",
            "private/control/rsna",
            "private/predictions/rsna",
            "private/localization",
            "outbox",
            "mlartifacts",
        )
    )
    file_roots = tuple(
        repository_root / name for name in ("mlflow.db", "mlflow.db-wal", "mlflow.db-shm")
    )
    for target in (*directory_roots, *file_roots):
        try:
            relative = target.relative_to(repository_root)
        except ValueError as exc:
            raise ManifestBuildError("Canonical RSNA layout is outside the repository") from exc
        current = repository_root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ManifestBuildError("Canonical RSNA layout contains a symlink redirect")
            if current.exists() and current != target and not current.is_dir():
                raise ManifestBuildError("Canonical RSNA layout has a parent type mismatch")
        if target.exists():
            expects_directory = target in directory_roots
            if (expects_directory and not target.is_dir()) or (
                not expects_directory and not target.is_file()
            ):
                raise ManifestBuildError("Canonical RSNA layout has a root type mismatch")
            if not target.resolve().is_relative_to(root):
                raise ManifestBuildError("Canonical RSNA layout escapes the repository")
    existing = [path for path in (*directory_roots, *file_roots) if path.exists()]
    for index, left in enumerate(existing):
        for right in existing[index + 1 :]:
            if left.samefile(right):
                raise ManifestBuildError("Canonical RSNA layout roots are aliased")


def _validate_neural_campaign_configs(configs: Sequence[ExperimentConfig]) -> None:
    """Validate all six neural configs against one campaign execution contract."""
    cxr = tuple(config for config in configs if config.family.family_id == "cxr_densenet")
    fusions = tuple(
        config for config in configs if config.family.family_id == "cxr_metadata_concat"
    )
    expected_seeds = tuple(
        spec.seed for spec in RSNA_FORMAL_RUN_PLAN if spec.family_id == "cxr_densenet"
    )
    if (
        tuple(require_runtime_seed(config) for config in cxr) != expected_seeds
        or tuple(require_runtime_seed(config) for config in fusions) != expected_seeds
    ):
        raise ValueError("RSNA campaign neural seeds differ from the formal run plan")
    if len({config.config_semantic_sha256 for config in cxr}) != 1:
        raise ValueError("RSNA CXR configurations do not form one scientific family")
    if len({config.config_semantic_sha256 for config in fusions}) != 1:
        raise ValueError("RSNA fusion configurations do not form one scientific family")
    neural_configs = (*cxr, *fusions)
    neural_settings = tuple(config.neural for config in neural_configs)
    if any(neural is None for neural in neural_settings):
        raise ValueError("RSNA campaign requires six complete neural configurations")
    resolved = cast(tuple[Any, ...], neural_settings)
    execution_contracts = {
        (
            neural.batch_size,
            config.runtime.num_workers,
            config.runtime.device,
            neural.mixed_precision,
            config.runtime.pin_memory_policy,
            neural.rotation_degrees,
            neural.translation_fraction,
            neural.brightness_jitter,
            neural.contrast_jitter,
        )
        for config, neural in zip(neural_configs, resolved, strict=True)
    }
    cache_identities = {
        preprocessing_identity(_transform(config, training=False)) for config in neural_configs
    }
    if (
        len(execution_contracts) != 1
        or len(cache_identities) != 1
        or resolved[0].batch_size != 32
        or neural_configs[0].runtime.device == "cpu"
        or neural_configs[0].runtime.pin_memory_policy == "disabled"
    ):
        raise ValueError("RSNA neural configurations do not share the campaign execution contract")


def _validate_config_agreement(configs: Sequence[ExperimentConfig]) -> None:
    coordinates = {
        (
            config.dataset.dataset_id,
            config.dataset.bundle_id,
            config.dataset.bundle_manifest_sha256,
            config.dataset.split_assignment_id,
            config.task.task_id,
            config.task.label_policy_version,
        )
        for config in configs
    }
    observed_matrix = tuple(
        (config.family.family_id, require_runtime_seed(config)) for config in configs
    )
    expected_matrix = tuple((spec.family_id, spec.seed) for spec in RSNA_FORMAL_RUN_PLAN)
    if (
        len(configs) != len(RSNA_FORMAL_RUN_PLAN)
        or len(coordinates) != 1
        or observed_matrix != expected_matrix
    ):
        raise ManifestBuildError("RSNA campaign configurations disagree on authority or task")


def _required_cuda_runtime(config: ExperimentConfig) -> ResolvedDevice:
    neural = config.neural
    if neural is None:
        raise ValueError("RSNA CXR configuration is incomplete")
    runtime = resolve_device(
        "cuda",
        mixed_precision=neural.mixed_precision,
        pin_memory_policy=config.runtime.pin_memory_policy,
    )
    if runtime.device.type != "cuda":
        raise RuntimeError("RSNA campaign neural runtime did not resolve to CUDA")
    return runtime


def _training_execution_policy(
    config: ExperimentConfig, runtime: ResolvedDevice
) -> LoaderExecutionPolicy:
    """Resolve the reviewed persistent policy for epoch-reused loaders."""
    neural = config.neural
    if neural is None:
        raise ValueError("RSNA CXR configuration is incomplete")
    return reused_loader_policy(
        num_workers=config.runtime.num_workers,
        pin_memory=runtime.pin_memory_effective,
    )


def _transform(config: ExperimentConfig, *, training: bool) -> StandardCxrTransform:
    neural = config.neural
    if neural is None:
        raise ValueError("RSNA CXR configuration is incomplete")
    return StandardCxrTransform(
        training=training,
        policy_version=str(config.preprocessing["cxr_transform_policy"]),
        image_size=int(config.family.parameters["image_size"]),
        rotation_degrees=neural.rotation_degrees,
        translation_fraction=neural.translation_fraction,
        brightness_jitter=neural.brightness_jitter,
        contrast_jitter=neural.contrast_jitter,
    )


def _validate_outputs(
    training_results: Sequence[Any],
    evaluation_results: Sequence[Any],
    audit_directory: Path,
    cxr_summary: Path,
    fusion_summary: Path,
    localization: Path,
    comparison: ComparisonResult,
    log_path: Path,
    *,
    roots: RsnaFormalRoots,
) -> None:
    private_root = roots.private_root
    if len(training_results) != 8 or len(evaluation_results) != 8:
        raise ValueError("RSNA campaign requires eight training and eight evaluation results")
    package_ids = tuple(result.model_package_id for result in training_results)
    evaluation_packages = tuple(result.model_package_id for result in evaluation_results)
    if len(set(package_ids)) != 8 or evaluation_packages != package_ids:
        raise ValueError("RSNA evaluation lineage does not match the exact package family")
    if len({result.evaluation_id for result in evaluation_results}) != 8:
        raise ValueError("RSNA evaluation IDs must be unique")
    required = [
        *(Path(result.model_path) for result in training_results),
        *(Path(result.artifact_directory) for result in training_results),
        *(Path(result.artifact_directory) for result in evaluation_results),
        audit_directory,
        cxr_summary,
        fusion_summary,
        localization,
        comparison.directory,
        log_path,
    ]
    for result in evaluation_results:
        private = getattr(result, "private_prediction_directory", None)
        if private is not None:
            required.append(Path(private))
    private_localization = private_root / "localization" / localization.name
    required.append(private_localization)
    if missing := [path for path in required if not path.exists()]:
        raise FileNotFoundError(f"Mandatory RSNA campaign outputs are missing: {missing}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the campaign and print its final transport paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute_rsna_campaign(backup_root=args.backup_root)
    except Exception as exc:
        print(f"RSNA campaign failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "campaign_id": result.campaign_id,
                "model_package_ids": list(result.model_package_ids),
                "evaluation_ids": list(result.evaluation_ids),
                "training_run_ids": list(result.training_run_ids),
                "evaluation_run_ids": list(result.evaluation_run_ids),
                "archive": result.archive_path.as_posix(),
                "checksum": result.checksum_path.as_posix(),
                "campaign_log": result.campaign_log_path.as_posix(),
            },
            indent=2,
        )
    )
    return 0
