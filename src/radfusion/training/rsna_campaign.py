"""Execute the complete authoritative RSNA campaign."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO, cast

import torch

from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.rsna_artifacts import (
    BUNDLES_DIRECTORY,
    build_and_write,
    validate_bundle_directory,
)
from radfusion.data.rsna_audit import generate_rsna_audit
from radfusion.data.rsna_cxr_cache import preprocessing_identity
from radfusion.models.cxr_baseline import ensure_pretrained_weights
from radfusion.training.config import (
    ExperimentConfig,
    load_experiment_config,
    require_runtime_seed,
    with_runtime,
)
from radfusion.training.device import ResolvedDevice, resolve_device
from radfusion.training.execution import (
    LoaderExecutionPolicy,
    one_shot_loader_policy,
    reused_loader_policy,
)
from radfusion.training.rsna_compare import regenerate_comparison
from radfusion.training.rsna_datasets import RsnaDataset, prepare_rsna_cxr_cache
from radfusion.training.rsna_evaluate import evaluate_model_package
from radfusion.training.rsna_localize import generate_localization_report
from radfusion.training.rsna_registry import get_dataset
from radfusion.training.rsna_seed_summary import publish_seed_summary
from radfusion.training.rsna_train_cxr import train_cxr_experiment
from radfusion.training.rsna_train_fusion import train_fusion_experiment
from radfusion.training.rsna_train_metadata import MetadataModelResult, train_metadata_experiment
from radfusion.utils.mlflow_utils import DEFAULT_TRACKING_URI
from radfusion.utils.operational_logging import (
    configure_logging,
    get_operational_logger,
    log_event,
    timed_phase,
)
from radfusion.utils.rsna_model_publication import validate_published_model
from radfusion.utils.rsna_neural_publication import validate_neural_package_metadata

EXPECTED_SEEDS = (17, 42, 2026)
RAW_ROOT = Path("data/raw/rsna/extracted")
MANIFEST_ROOT = Path("data/manifests")
CACHE_ROOT = Path("data/cache/rsna")
REPORT_ROOT = Path("reports")
OUTBOX_ROOT = Path("outbox")
_MINIMUM_FREE_BYTES = 16 * 1024**3
_LOGGER = get_operational_logger(__name__)


@dataclass(frozen=True)
class CampaignConfigs:
    """The eight reviewed experiment definitions in canonical order."""

    metadata_logistic: ExperimentConfig
    metadata_lightgbm: ExperimentConfig
    cxr: tuple[ExperimentConfig, ...]
    fusions: tuple[ExperimentConfig, ...]


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


def execute_rsna_campaign() -> CampaignResult:
    """Run one fresh fail-fast RSNA campaign with direct identity handoff."""
    _require_fresh_output_surface()
    campaign_id = "campaign-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    campaign_directory = REPORT_ROOT / "rsna" / "campaigns" / campaign_id
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
            with timed_phase(_LOGGER, "prerequisite_validation"):
                configs = _validate_prerequisites()
            with timed_phase(_LOGGER, "pretrained_weight_readiness"):
                ensure_pretrained_weights("densenet121-res224-chex")
            with timed_phase(_LOGGER, "manifest_build"):
                written = build_and_write(RAW_ROOT, MANIFEST_ROOT)
                _validate_configured_bundle(configs, written.paths.bundle_id)
            with timed_phase(_LOGGER, "dataset_audit"):
                generate_rsna_audit(MANIFEST_ROOT, REPORT_ROOT / "rsna" / "audit")
                audit_directory = REPORT_ROOT / "rsna" / "audit" / written.paths.bundle_id
            dataset = get_dataset("rsna")
            if not isinstance(dataset, RsnaDataset):
                raise TypeError("RSNA campaign requires the concrete RSNA dataset adapter")
            cxr_reference = configs.cxr[0]
            transform = _transform(cxr_reference, training=False)
            with timed_phase(_LOGGER, "cxr_cache_preparation"):
                cache = prepare_rsna_cxr_cache(
                    dataset, cxr_reference, transform, cache_root=CACHE_ROOT
                )
            runtime = _required_cuda_runtime(cxr_reference)
            training_execution = _training_execution_policy(cxr_reference, runtime)
            evaluation_execution = one_shot_loader_policy(pin_memory=runtime.pin_memory_effective)
            metadata_training = _train_metadata(configs)
            cxr_training = tuple(
                train_cxr_experiment(
                    config,
                    tracking_uri=DEFAULT_TRACKING_URI,
                    cache=cache,
                    execution=training_execution,
                )
                for config in configs.cxr
            )
            fusion_training = tuple(
                train_fusion_experiment(
                    config,
                    source_cxr_package_id=source.model_package_id,
                    tracking_uri=DEFAULT_TRACKING_URI,
                    cache=cache,
                    execution=training_execution,
                )
                for config, source in zip(configs.fusions, cxr_training, strict=True)
            )
            training_results = (*metadata_training, *cxr_training, *fusion_training)
            _validate_frozen_training_packages(training_results)
            log_event(_LOGGER, "training_boundary_completed", total=len(training_results))

            evaluation_results = tuple(
                evaluate_model_package(
                    result.model_package_id,
                    evaluation_config=config,
                    tracking_uri=DEFAULT_TRACKING_URI,
                    cache=cache,
                    execution=evaluation_execution,
                )
                for result, config in zip(
                    training_results,
                    (
                        configs.metadata_logistic,
                        configs.metadata_lightgbm,
                        *configs.cxr,
                        *configs.fusions,
                    ),
                    strict=True,
                )
            )
            cxr_tests = evaluation_results[2:5]
            fusion_tests = evaluation_results[5:]
            cxr_evaluation_ids = tuple(result.evaluation_id for result in cxr_tests)
            fusion_evaluation_ids = tuple(result.evaluation_id for result in fusion_tests)
            cxr_summary = publish_seed_summary(cxr_evaluation_ids, output_directory=REPORT_ROOT)
            fusion_summary = publish_seed_summary(
                fusion_evaluation_ids, output_directory=REPORT_ROOT
            )
            localization = generate_localization_report(
                cxr_evaluation_ids,
                output_directory=REPORT_ROOT,
                cache=cache,
            )
            comparison = regenerate_comparison(
                tuple(result.evaluation_id for result in evaluation_results),
                output_directory=REPORT_ROOT,
            )
            with timed_phase(_LOGGER, "output_validation"):
                _validate_outputs(
                    training_results,
                    evaluation_results,
                    audit_directory,
                    cxr_summary.report_directory,
                    fusion_summary.report_directory,
                    localization,
                    comparison[:2],
                    log_path,
                )
            log_event(_LOGGER, "campaign_ready_for_export", phase="export")
            log_stream.flush()
            archive = OUTBOX_ROOT / f"rsna-results-{campaign_id}.tar.gz"
            archive_started_at = time.perf_counter()
            _write_archive(
                archive,
                bundle_directory=written.paths.bundle_directory,
                current_path=written.paths.current_path,
            )
            log_event(
                _LOGGER,
                "archive_created",
                elapsed_s=time.perf_counter() - archive_started_at,
            )
            checksum = _write_checksum(archive)
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


def _train_metadata(configs: CampaignConfigs) -> tuple[MetadataModelResult, MetadataModelResult]:
    return (
        train_metadata_experiment(configs.metadata_logistic, tracking_uri=DEFAULT_TRACKING_URI),
        train_metadata_experiment(configs.metadata_lightgbm, tracking_uri=DEFAULT_TRACKING_URI),
    )


def _validate_prerequisites() -> CampaignConfigs:
    if not RAW_ROOT.is_dir():
        raise FileNotFoundError(f"RSNA raw dataset is missing: {RAW_ROOT}")
    required_raw = (
        RAW_ROOT / "stage_2_train_images",
        RAW_ROOT / "stage_2_train_labels.csv",
        RAW_ROOT / "stage_2_detailed_class_info.csv",
    )
    if not all(path.is_dir() if path.suffix == "" else path.is_file() for path in required_raw):
        raise FileNotFoundError("RSNA raw dataset is incomplete")
    paths = {
        "metadata_logistic": Path("configs/rsna_metadata_logistic.yaml"),
        "metadata_lightgbm": Path("configs/rsna_metadata_lightgbm.yaml"),
        "cxr": Path("configs/rsna_cxr_densenet.yaml"),
        "fusion": Path("configs/rsna_cxr_metadata_concat.yaml"),
    }
    if missing := [path for path in paths.values() if not path.is_file()]:
        raise FileNotFoundError(f"RSNA campaign configs are missing: {missing}")
    configs = CampaignConfigs(
        with_runtime(load_experiment_config(paths["metadata_logistic"]), seed=42),
        with_runtime(load_experiment_config(paths["metadata_lightgbm"]), seed=42),
        tuple(
            with_runtime(load_experiment_config(paths["cxr"]), seed=seed) for seed in EXPECTED_SEEDS
        ),
        tuple(
            with_runtime(load_experiment_config(paths["fusion"]), seed=seed)
            for seed in EXPECTED_SEEDS
        ),
    )
    _validate_neural_campaign_configs(configs)
    if DEFAULT_TRACKING_URI != "sqlite:///mlflow.db":
        raise RuntimeError("RSNA campaign requires canonical local mlflow.db tracking")
    for root in (
        MANIFEST_ROOT,
        CACHE_ROOT,
        REPORT_ROOT,
        Path("models"),
        Path("private"),
        OUTBOX_ROOT,
    ):
        root.mkdir(parents=True, exist_ok=True)
        if not os.access(root, os.W_OK):
            raise PermissionError(f"RSNA campaign output is not writable: {root}")
    if shutil.disk_usage(Path.cwd()).free < _MINIMUM_FREE_BYTES:
        raise OSError("RSNA campaign requires at least 16 GiB of free workspace storage")
    if not torch.cuda.is_available():
        raise RuntimeError("RSNA campaign requires CUDA")
    return configs


def _validate_neural_campaign_configs(configs: CampaignConfigs) -> None:
    """Validate all six neural configs against one campaign execution contract."""
    if (
        len(configs.cxr) != len(EXPECTED_SEEDS)
        or len(configs.fusions) != len(EXPECTED_SEEDS)
        or tuple(require_runtime_seed(config) for config in configs.cxr) != EXPECTED_SEEDS
        or tuple(require_runtime_seed(config) for config in configs.fusions) != EXPECTED_SEEDS
    ):
        raise ValueError("RSNA campaign requires CXR and fusion seeds 17, 42, and 2026")
    if len({config.config_semantic_sha256 for config in configs.cxr}) != 1:
        raise ValueError("RSNA CXR configurations do not form one scientific family")
    if len({config.config_semantic_sha256 for config in configs.fusions}) != 1:
        raise ValueError("RSNA fusion configurations do not form one scientific family")
    neural_configs = (*configs.cxr, *configs.fusions)
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


def _validate_configured_bundle(configs: CampaignConfigs, bundle_id: str) -> None:
    all_configs = (
        configs.metadata_logistic,
        configs.metadata_lightgbm,
        *configs.cxr,
        *configs.fusions,
    )
    if any(config.dataset.bundle_id != bundle_id for config in all_configs):
        raise ValueError("RSNA experiment config bundle IDs do not match the built bundle")
    validate_bundle_directory(
        MANIFEST_ROOT / "rsna" / BUNDLES_DIRECTORY / bundle_id,
        expected_bundle_id=bundle_id,
    )


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


def _validate_frozen_training_packages(results: Sequence[Any]) -> None:
    for result in results:
        path = Path(result.model_path)
        package = path.parent
        if isinstance(result, MetadataModelResult):
            validate_published_model(package)
        else:
            validate_neural_package_metadata(package)


def _require_fresh_output_surface() -> None:
    """Reject mixed campaigns without deleting inspectable prior outputs."""
    generated = (
        REPORT_ROOT,
        Path("models"),
        Path("private/predictions"),
        Path("private/localization"),
        Path("mlartifacts"),
        Path("mlflow.db"),
        Path("mlflow.db-wal"),
        Path("mlflow.db-shm"),
        OUTBOX_ROOT,
    )
    if existing := [path for path in generated if path.exists()]:
        raise FileExistsError(
            f"RSNA campaign requires a fresh generated-output surface: {existing}"
        )


def _validate_outputs(
    training_results: Sequence[Any],
    evaluation_results: Sequence[Any],
    audit_directory: Path,
    cxr_summary: Path,
    fusion_summary: Path,
    localization: Path,
    comparison: tuple[Path, Path],
    log_path: Path,
) -> None:
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
        *comparison,
        log_path,
        Path("mlflow.db"),
    ]
    for result in evaluation_results:
        private = getattr(result, "private_prediction_directory", None)
        if private is not None:
            required.append(Path(private))
    private_localization = Path("private/localization") / localization.name
    required.append(private_localization)
    if missing := [path for path in required if not path.exists()]:
        raise FileNotFoundError(f"Mandatory RSNA campaign outputs are missing: {missing}")
    _validate_sqlite_integrity(Path("mlflow.db"))


def _write_archive(
    destination: Path,
    *,
    bundle_directory: Path,
    current_path: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"RSNA results archive already exists: {destination}")
    expected_bundle_root = (MANIFEST_ROOT / "rsna" / BUNDLES_DIRECTORY).resolve()
    if (
        bundle_directory.resolve().parent != expected_bundle_root
        or not bundle_directory.name.startswith("bundle-")
        or current_path.resolve().parent != (MANIFEST_ROOT / "rsna").resolve()
        or current_path.name != "CURRENT"
        or current_path.read_text(encoding="utf-8").strip() != bundle_directory.name
    ):
        raise ValueError("Archive bundle lineage does not match the current RSNA bundle")
    roots = [
        Path("models"),
        REPORT_ROOT,
        Path("private/predictions"),
        Path("private/localization"),
        Path("mlartifacts"),
    ]
    database = Path("mlflow.db")
    if missing := [
        path for path in (*roots, database, bundle_directory, current_path) if not path.exists()
    ]:
        raise FileNotFoundError(f"Archive inputs are missing: {missing}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with tempfile.TemporaryDirectory(prefix="radfusion-mlflow-export-") as temporary_directory:
            snapshot = Path(temporary_directory) / database.name
            _snapshot_sqlite_database(database, snapshot)
            with tarfile.open(temporary, "w:gz") as archive:
                archive.add(bundle_directory, arcname=bundle_directory.as_posix(), recursive=True)
                archive.add(current_path, arcname=current_path.as_posix(), recursive=False)
                for root in roots:
                    archive.add(root, arcname=root.as_posix(), recursive=True)
                archive.add(snapshot, arcname=database.as_posix(), recursive=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_checksum(archive: Path) -> Path:
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    destination = archive.with_suffix(archive.suffix + ".sha256")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(f"{digest.hexdigest()}  {archive.name}\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def _snapshot_sqlite_database(source: Path, destination: Path) -> None:
    """Create one consistent snapshot with archive-relative artifact URIs."""
    source_uri = f"file:{source.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as input_database:
        with sqlite3.connect(destination) as output_database:
            input_database.backup(output_database)
            _make_mlflow_snapshot_portable(output_database)
            result = output_database.execute("PRAGMA integrity_check").fetchone()
    if result != ("ok",):
        raise RuntimeError("MLflow SQLite snapshot failed integrity validation")


def _validate_sqlite_integrity(database: Path) -> None:
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        result = connection.execute("PRAGMA integrity_check").fetchone()
    if result != ("ok",):
        raise RuntimeError("MLflow SQLite database failed integrity validation")


def _make_mlflow_snapshot_portable(database: sqlite3.Connection) -> None:
    """Replace machine-absolute MLflow artifact URIs in the exported snapshot."""
    live_root = Path("mlartifacts").resolve().as_uri().rstrip("/")
    portable_root = "file:mlartifacts"
    experiments = database.execute(
        """
        SELECT experiments.experiment_id, experiments.artifact_location,
               COUNT(runs.run_uuid)
        FROM experiments
        LEFT JOIN runs ON runs.experiment_id = experiments.experiment_id
        GROUP BY experiments.experiment_id, experiments.artifact_location
        """
    ).fetchall()
    for experiment_id, location, run_count in experiments:
        if run_count:
            if location != live_root:
                raise ValueError(
                    "Active MLflow experiment artifact location is outside mlartifacts: "
                    f"{location!r}"
                )
            portable_location = portable_root
        else:
            portable_location = f"{portable_root}/experiments/{experiment_id}"
        database.execute(
            "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
            (portable_location, experiment_id),
        )

    live_prefix = live_root + "/"
    for run_uuid, artifact_uri in database.execute(
        "SELECT run_uuid, artifact_uri FROM runs"
    ).fetchall():
        if not isinstance(artifact_uri, str) or not artifact_uri.startswith(live_prefix):
            raise ValueError(
                f"MLflow run {run_uuid!r} artifact URI is outside mlartifacts: {artifact_uri!r}"
            )
        portable_uri = portable_root + "/" + artifact_uri.removeprefix(live_prefix)
        database.execute(
            "UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?",
            (portable_uri, run_uuid),
        )


def main() -> int:
    """Run the campaign and print its final transport paths."""
    try:
        result = execute_rsna_campaign()
    except Exception as exc:
        print(f"RSNA campaign failed: {exc}", file=sys.stderr)
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
