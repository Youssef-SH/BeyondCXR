from __future__ import annotations

import hashlib
import random
import shutil
import sqlite3
import tarfile
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from mlflow.tracking import MlflowClient

from radfusion.data.cxr_cache import CxrCacheIdentity, build_cxr_cache, preprocessing_identity
from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.dicom_loader import DicomRecord
from radfusion.training.config import load_experiment_config
from radfusion.training.datasets import RsnaCachedCalibrationDataset, RsnaDataset
from radfusion.training.execution import LoaderExecutionPolicy
from radfusion.training.rsna_gpu import (
    _LOADER_MEASURED_BATCHES,
    _LOADER_WARMUP_BATCHES,
    CampaignConfigs,
    _calibrate_execution,
    _validate_neural_campaign_configs,
    _validate_outputs,
    _write_archive,
    _write_checksum,
    main,
    run_rsna_gpu_campaign,
)
from radfusion.utils.mlflow_utils import configure_mlflow


def _configs() -> CampaignConfigs:
    seeds = (17, 42, 2026)
    return CampaignConfigs(
        load_experiment_config("configs/metadata_logistic.yaml"),
        load_experiment_config("configs/metadata_lightgbm.yaml"),
        tuple(load_experiment_config(f"configs/image_densenet_seed{seed}.yaml") for seed in seeds),
        tuple(load_experiment_config(f"configs/fusion_concat_seed{seed}.yaml") for seed in seeds),
    )


def test_campaign_hands_exact_ids_across_the_strict_test_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = _configs()
    monkeypatch.chdir(tmp_path)
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu._validate_prerequisites",
        lambda: events.append(("prerequisites", None)) or configs,
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu.ensure_pretrained_weights",
        lambda name: events.append(("weights", name)),
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu.build_and_write",
        lambda *args: (
            events.append(("manifest_build", None))
            or SimpleNamespace(
                paths=SimpleNamespace(
                    bundle_id="bundle",
                    bundle_directory=Path("data/manifests/rsna/builds/bundle"),
                    current_path=Path("data/manifests/rsna/CURRENT"),
                )
            )
        ),
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu._validate_configured_bundle",
        lambda *args: events.append(("manifest_validation", None)),
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu.generate_rsna_audit",
        lambda *args: events.append(("audit", None)),
    )
    monkeypatch.setattr("radfusion.training.rsna_gpu.get_dataset", lambda key: RsnaDataset())
    cache = SimpleNamespace(identity=SimpleNamespace(cache_id="cache-" + "0" * 64))
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu.prepare_rsna_cxr_cache",
        lambda *args, **kwargs: events.append(("cache", None)) or cache,
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu._required_cuda_runtime", lambda config: object()
    )
    policy = LoaderExecutionPolicy(8.0, 4, True, 2, True)
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu._calibrate_execution",
        lambda *args: events.append(("calibrate", None)) or policy,
    )

    def result(run_id: str):
        return SimpleNamespace(
            run_id=run_id,
            model_path=tmp_path / f"{run_id}.model",
            artifact_directory=tmp_path / f"{run_id}.report",
        )

    metadata_ids = iter(("metadata-logistic", "metadata-lightgbm"))
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu.train_configured_experiment",
        lambda *args, **kwargs: (
            events.append(("train", run_id := next(metadata_ids))) or result(run_id)
        ),
    )
    image_ids = iter(("cxr-17", "cxr-42", "cxr-2026"))
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu.train_image_experiment",
        lambda *args, **kwargs: (
            events.append(("train", run_id := next(image_ids))) or result(run_id)
        ),
    )

    def train_fusion(config, *, source_training_run_id, **kwargs):
        run_id = f"fusion-{config.training.seed}"
        events.append(("fusion_source", (run_id, source_training_run_id)))
        events.append(("train", run_id))
        return result(run_id)

    monkeypatch.setattr("radfusion.training.rsna_gpu.train_fusion_experiment", train_fusion)

    def evaluate(run_id, **kwargs):
        events.append(("evaluate", run_id))
        return SimpleNamespace(run_id=f"test-{run_id}", artifact_directory=tmp_path / run_id)

    monkeypatch.setattr("radfusion.training.rsna_gpu.evaluate_training_run", evaluate)

    def summarize(run_ids, **kwargs):
        events.append(("summary", tuple(run_ids)))
        return SimpleNamespace(report_directory=tmp_path / f"summary-{run_ids[0]}")

    monkeypatch.setattr("radfusion.training.rsna_gpu.summarize_seed_runs", summarize)
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu.generate_localization_report",
        lambda run_ids, **kwargs: (
            events.append(("localize", tuple(run_ids))) or tmp_path / "localization"
        ),
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu.regenerate_comparison",
        lambda **kwargs: (
            events.append(("compare", None))
            or (tmp_path / "comparison.csv", tmp_path / "comparison.md", 16)
        ),
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu._validate_frozen_training_packages", lambda r: None
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu._validate_outputs",
        lambda *args: events.append(("validate", None)),
    )

    def archive(path, **_lineage):
        log = (
            Path("reports/rsna/campaigns")
            / next(child.name for child in Path("reports/rsna/campaigns").iterdir())
            / "execution.log"
        )
        archived_log = log.read_text(encoding="utf-8")
        assert "event=campaign_ready_for_export" in archived_log
        assert "event=campaign_succeeded" not in archived_log
        events.append(("archive", None))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"archive")

    monkeypatch.setattr("radfusion.training.rsna_gpu._write_archive", archive)

    campaign = run_rsna_gpu_campaign()

    assert [(kind, value) for kind, value in events if kind == "fusion_source"] == [
        ("fusion_source", ("fusion-17", "cxr-17")),
        ("fusion_source", ("fusion-42", "cxr-42")),
        ("fusion_source", ("fusion-2026", "cxr-2026")),
    ]
    training_positions = [index for index, value in enumerate(events) if value[0] == "train"]
    evaluation_positions = [index for index, value in enumerate(events) if value[0] == "evaluate"]
    assert len(training_positions) == 8
    assert len(evaluation_positions) == 8
    assert max(training_positions) < min(evaluation_positions)
    assert [kind for kind, _ in events[:7]] == [
        "prerequisites",
        "weights",
        "manifest_build",
        "manifest_validation",
        "audit",
        "cache",
        "calibrate",
    ]
    assert [value for kind, value in events if kind == "summary"] == [
        ("test-cxr-17", "test-cxr-42", "test-cxr-2026"),
        ("test-fusion-17", "test-fusion-42", "test-fusion-2026"),
    ]
    assert campaign.archive_path.is_file()
    assert campaign.checksum_path.is_file()
    assert campaign.campaign_log_path.is_file()
    campaign_log = campaign.campaign_log_path.read_text(encoding="utf-8")
    assert "event=campaign_started" in campaign_log
    assert "event=loader_policy_selected" in campaign_log
    assert "event=campaign_succeeded" in campaign_log
    assert [kind for kind, _ in events][-3:] == ["compare", "validate", "archive"]


def test_training_failure_never_crosses_test_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = _configs()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("radfusion.training.rsna_gpu._validate_prerequisites", lambda: configs)
    monkeypatch.setattr("radfusion.training.rsna_gpu.ensure_pretrained_weights", lambda name: None)
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu.build_and_write",
        lambda *args: SimpleNamespace(
            paths=SimpleNamespace(
                bundle_id="bundle",
                bundle_directory=Path("data/manifests/rsna/builds/bundle"),
                current_path=Path("data/manifests/rsna/CURRENT"),
            )
        ),
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu._validate_configured_bundle", lambda *args: None
    )
    monkeypatch.setattr("radfusion.training.rsna_gpu.generate_rsna_audit", lambda *args: None)
    monkeypatch.setattr("radfusion.training.rsna_gpu.get_dataset", lambda key: RsnaDataset())
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu.prepare_rsna_cxr_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu._required_cuda_runtime", lambda config: object()
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu._calibrate_execution",
        lambda *args: LoaderExecutionPolicy(1.0, 0, False, None, False),
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu.train_configured_experiment",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("training failed")),
    )
    evaluations: list[str] = []
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu.evaluate_training_run",
        lambda run_id, **kwargs: evaluations.append(run_id),
    )
    with pytest.raises(RuntimeError):
        run_rsna_gpu_campaign()
    assert evaluations == []
    log_path = next((tmp_path / "reports/rsna/campaigns").glob("*/execution.log"))
    assert "event=campaign_failed" in log_path.read_text(encoding="utf-8")


def test_campaign_rejects_existing_outputs_without_deleting_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    existing = Path("models/rsna/runs/existing/model.pt")
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"partial output")

    with pytest.raises(FileExistsError):
        run_rsna_gpu_campaign()

    assert existing.read_bytes() == b"partial output"


def test_campaign_cli_preserves_process_interrupts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu.run_rsna_gpu_campaign",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        main()


def test_portable_archive_contains_results_and_excludes_sources_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    included = (
        Path("data/manifests/rsna/builds/build-test/metadata.json"),
        Path("data/manifests/rsna/CURRENT"),
        Path("models/rsna/runs/train/model.skops"),
        Path("reports/rsna/campaigns/campaign-test/execution.log"),
        Path("private/predictions/test/predictions.parquet"),
        Path("private/localization/localization-test/summary.json"),
    )
    for path in included:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("result\n", encoding="utf-8")
    included[1].write_text("build-test\n", encoding="utf-8")
    old_bundle = Path("data/manifests/rsna/builds/build-old/metadata.json")
    old_bundle.parent.mkdir(parents=True)
    old_bundle.write_text("old\n", encoding="utf-8")
    configure_mlflow(tracking_uri="sqlite:///mlflow.db", experiment_name="radfusion-rsna")
    source_artifact = Path("resolved_config.yaml")
    source_artifact.write_text("experiment: test\n", encoding="utf-8")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_artifact(source_artifact, artifact_path="config")
    archived_artifact = Path(f"mlartifacts/{run_id}/artifacts/config/resolved_config.yaml")
    for excluded in (
        Path("data/raw/rsna/extracted/stage_2_train_images/a.dcm"),
        Path("data/cache/rsna/cache-test/images.npy"),
    ):
        excluded.parent.mkdir(parents=True, exist_ok=True)
        excluded.write_text("source\n", encoding="utf-8")

    destination = Path("outbox/results.tar.gz").resolve()
    included[3].write_text("event=campaign_ready_for_export\n", encoding="utf-8")
    _write_archive(
        destination,
        bundle_directory=included[0].parent,
        current_path=included[1],
    )
    checksum = _write_checksum(destination)

    with tarfile.open(destination, "r:gz") as archive:
        names = set(archive.getnames())
        database_member = archive.extractfile("mlflow.db")
        assert database_member is not None
        exported_database = tmp_path / "exported-mlflow.db"
        exported_database.write_bytes(database_member.read())
        relocated = tmp_path / "relocated"
        archive.extractall(relocated, filter="data")
    assert all(path.as_posix() in names for path in (*included, archived_artifact))
    assert "mlflow.db" in names
    assert not any(name.startswith("data/raw/") for name in names)
    assert not any(name.startswith("data/cache/") for name in names)
    assert old_bundle.as_posix() not in names
    with tarfile.open(destination, "r:gz") as archive:
        archived_log = archive.extractfile(included[3].as_posix())
        assert archived_log is not None
        log_text = archived_log.read().decode()
    assert "event=campaign_ready_for_export" in log_text
    assert "event=campaign_succeeded" not in log_text
    with sqlite3.connect(exported_database) as database:
        location = database.execute(
            "SELECT artifact_location FROM experiments WHERE name = 'radfusion-rsna'"
        ).fetchone()
        run_uri = database.execute(
            "SELECT artifact_uri FROM runs WHERE run_uuid = ?", (run_id,)
        ).fetchone()
    assert location == ("file:mlartifacts",)
    assert run_uri == (f"file:mlartifacts/{run_id}/artifacts",)
    monkeypatch.chdir(relocated)
    relocated_client = MlflowClient(tracking_uri=f"sqlite:///{relocated / 'mlflow.db'}")
    downloaded = relocated_client.download_artifacts(
        run_id,
        "config/resolved_config.yaml",
        str(relocated / "download"),
    )
    assert Path(downloaded).read_text(encoding="utf-8") == "experiment: test\n"
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    assert checksum.read_text(encoding="utf-8") == f"{digest}  {destination.name}\n"
    shutil.rmtree("models")
    with pytest.raises(FileNotFoundError):
        _write_archive(
            Path("outbox/incomplete.tar.gz"),
            bundle_directory=included[0].parent,
            current_path=included[1],
        )


def test_loader_calibration_uses_8_warmup_and_64_measured_batches() -> None:
    assert _LOADER_WARMUP_BATCHES == 8
    assert _LOADER_MEASURED_BATCHES == 64


def test_loader_calibration_preserves_caller_rng_and_uses_only_train_cache_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_experiment_config("configs/image_densenet_seed42.yaml")
    rows = pd.DataFrame(
        [
            (
                f"rsna:{index:03d}",
                f"patient-{index:03d}",
                f"images/{index:03d}.dcm",
                "train" if index < 4 else ("validation" if index == 4 else "test"),
                1,
                f"{index + 1:x}" * 64,
            )
            for index in range(6)
        ],
        columns=("sample_id", "patient_id", "image_path", "split_name", "byte_size", "sha256"),
    )
    pixels = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)

    def decode(path: Path, **_authentication):
        index = int(path.stem)
        return pixels, DicomRecord(
            path=str(path),
            patient_id=f"patient-{index:03d}",
            patient_age=None,
            patient_sex=None,
            view_position=None,
            rows=8,
            columns=8,
            photometric_interpretation="MONOCHROME2",
        )

    monkeypatch.setattr("radfusion.data.cxr_cache.read_dicom", decode)
    transform = StandardCxrTransform(training=False)
    cache = build_cxr_cache(
        rows,
        dataset_root=tmp_path / "raw",
        cache_root=tmp_path / "cache",
        identity=CxrCacheIdentity(
            bundle_id="build-" + "0" * 64,
            bundle_manifest_sha256="1" * 64,
            source_inventory_file_sha256="2" * 64,
            source_inventory_arrow_sha256="3" * 64,
            preprocessing_sha256=preprocessing_identity(transform),
        ),
        transform=transform,
    )
    monkeypatch.setattr("radfusion.training.rsna_gpu._LOADER_WARMUP_BATCHES", 0)
    monkeypatch.setattr("radfusion.training.rsna_gpu._LOADER_MEASURED_BATCHES", 1)
    monkeypatch.setattr("radfusion.training.rsna_gpu.effective_cpu_capacity", lambda: 2.0)
    monkeypatch.setattr(
        "radfusion.training.rsna_gpu.calibrate_loader_policy",
        lambda benchmark, **kwargs: LoaderExecutionPolicy(
            2.0, 0, False, None, False, ((0, benchmark(0)),)
        ),
    )
    observed_ids: list[tuple[str, ...]] = []

    def record_dataset(sample_ids, **kwargs):
        observed_ids.append(sample_ids)
        return RsnaCachedCalibrationDataset(sample_ids, **kwargs)

    monkeypatch.setattr("radfusion.training.rsna_gpu.RsnaCachedCalibrationDataset", record_dataset)

    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_before = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)

    _calibrate_execution(
        config,
        cache,
        SimpleNamespace(pin_memory_effective=False),
    )

    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    if cuda_before is not None:
        assert all(
            torch.equal(left, right)
            for left, right in zip(torch.cuda.get_rng_state_all(), cuda_before, strict=True)
        )
    assert torch.are_deterministic_algorithms_enabled() is deterministic_before
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only_before
    assert (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark) == cudnn_before
    assert observed_ids == [("rsna:000", "rsna:001", "rsna:002", "rsna:003")]


@pytest.mark.parametrize("mismatch", ["preprocessing", "device", "batch_size", "seed"])
def test_campaign_rejects_neural_config_contract_mismatch(mismatch: str) -> None:
    configs = _configs()
    if mismatch == "preprocessing":
        fusions = tuple(
            replace(
                config,
                model=replace(
                    config.model,
                    parameters=MappingProxyType({**config.model.parameters, "image_size": 225}),
                ),
            )
            for config in configs.fusions
        )
    elif mismatch == "device":
        fusions = tuple(
            replace(config, image=replace(config.image, device="cuda"))
            for config in configs.fusions
        )
    elif mismatch == "seed":
        fusions = (
            replace(
                configs.fusions[0],
                training=replace(configs.fusions[0].training, seed=18),
            ),
            *configs.fusions[1:],
        )
    else:
        fusions = tuple(
            replace(config, image=replace(config.image, batch_size=16))
            for config in configs.fusions
        )

    with pytest.raises(ValueError):
        _validate_neural_campaign_configs(replace(configs, fusions=fusions))


def test_output_validation_rejects_evaluation_lineage_mismatch(tmp_path: Path) -> None:
    training = tuple(
        SimpleNamespace(
            run_id=f"train-{index}",
            model_path=tmp_path / "model",
            artifact_directory=tmp_path / "report",
        )
        for index in range(8)
    )
    evaluations = tuple(
        SimpleNamespace(
            run_id=f"test-{index}",
            training_run_id=f"wrong-{index}",
            artifact_directory=tmp_path / "test-report",
        )
        for index in range(8)
    )
    with pytest.raises(ValueError):
        _validate_outputs(
            training,
            evaluations,
            tmp_path / "audit",
            tmp_path / "cxr-summary",
            tmp_path / "fusion-summary",
            tmp_path / "localization",
            (tmp_path / "comparison.csv", tmp_path / "comparison.md"),
            tmp_path / "execution.log",
        )
