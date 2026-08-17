from __future__ import annotations

import hashlib
import shutil
import sqlite3
import tarfile
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from radfusion.training.config import load_experiment_config, with_runtime
from radfusion.training.rsna_campaign import (
    CampaignConfigs,
    _validate_neural_campaign_configs,
    _validate_outputs,
    _write_archive,
    _write_checksum,
    execute_rsna_campaign,
    main,
)
from radfusion.training.rsna_datasets import RsnaDataset
from radfusion.utils.mlflow_utils import configure_mlflow

_PACKAGE_IDS = {
    name: "model-package-" + character * 64
    for name, character in zip(
        (
            "metadata-logistic",
            "metadata-lightgbm",
            "cxr-17",
            "cxr-42",
            "cxr-2026",
            "fusion-17",
            "fusion-42",
            "fusion-2026",
        ),
        "12345678",
        strict=True,
    )
}


def _evaluation_id(package_id: str) -> str:
    return "evaluation-" + hashlib.sha256(package_id.encode("ascii")).hexdigest()


def _configs() -> CampaignConfigs:
    seeds = (17, 42, 2026)
    return CampaignConfigs(
        with_runtime(load_experiment_config("configs/rsna_metadata_logistic.yaml"), seed=42),
        with_runtime(load_experiment_config("configs/rsna_metadata_lightgbm.yaml"), seed=42),
        tuple(
            with_runtime(load_experiment_config("configs/rsna_cxr_densenet.yaml"), seed=seed)
            for seed in seeds
        ),
        tuple(
            with_runtime(load_experiment_config("configs/rsna_cxr_metadata_concat.yaml"), seed=seed)
            for seed in seeds
        ),
    )


def test_campaign_freezes_complete_candidate_set_before_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = _configs()
    monkeypatch.chdir(tmp_path)
    events: list[tuple[str, object]] = []
    training_policies = []
    evaluation_policies = []
    evaluation_configs = []
    expected_package_ids = tuple(_PACKAGE_IDS.values())
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign._validate_prerequisites",
        lambda: events.append(("prerequisites", None)) or configs,
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign.ensure_pretrained_weights",
        lambda name: events.append(("weights", name)),
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign.build_and_write",
        lambda *args: (
            events.append(("manifest_build", None))
            or SimpleNamespace(
                paths=SimpleNamespace(
                    bundle_id="bundle",
                    bundle_directory=Path("data/manifests/rsna/bundles/bundle-test"),
                    current_path=Path("data/manifests/rsna/CURRENT"),
                )
            )
        ),
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign._validate_configured_bundle",
        lambda *args: events.append(("manifest_validation", None)),
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign.generate_rsna_audit",
        lambda *args: events.append(("audit", None)),
    )
    monkeypatch.setattr("radfusion.training.rsna_campaign.get_dataset", lambda key: RsnaDataset())
    cache = SimpleNamespace(identity=SimpleNamespace(cache_id="cache-" + "0" * 64))
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign.prepare_rsna_cxr_cache",
        lambda *args, **kwargs: events.append(("cache", None)) or cache,
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign._required_cuda_runtime",
        lambda config: SimpleNamespace(pin_memory_effective=True),
    )

    def result(run_id: str):
        return SimpleNamespace(
            run_id=run_id,
            model_package_id=_PACKAGE_IDS[run_id],
            model_path=tmp_path / f"{run_id}.model",
            artifact_directory=tmp_path / f"{run_id}.report",
        )

    metadata_ids = iter(("metadata-logistic", "metadata-lightgbm"))
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign.train_metadata_experiment",
        lambda *args, **kwargs: (
            events.append(("train", run_id := next(metadata_ids))) or result(run_id)
        ),
    )
    cxr_ids = iter(("cxr-17", "cxr-42", "cxr-2026"))

    def train_cxr(*args, **kwargs):
        training_policies.append(kwargs["execution"])
        run_id = next(cxr_ids)
        events.append(("train", run_id))
        return result(run_id)

    monkeypatch.setattr("radfusion.training.rsna_campaign.train_cxr_experiment", train_cxr)

    def train_fusion(config, *, source_cxr_package_id, **kwargs):
        training_policies.append(kwargs["execution"])
        run_id = f"fusion-{config.runtime.seed}"
        events.append(("fusion_source", (run_id, source_cxr_package_id)))
        events.append(("train", run_id))
        return result(run_id)

    monkeypatch.setattr("radfusion.training.rsna_campaign.train_fusion_experiment", train_fusion)

    def evaluate(package_id, **kwargs):
        frozen = [value for kind, value in events if kind == "candidate_set_frozen"]
        assert frozen == [expected_package_ids]
        evaluation_policies.append(kwargs["execution"])
        evaluation_configs.append(kwargs["evaluation_config"])
        events.append(("test_access", package_id))
        return SimpleNamespace(
            mlflow_run_id=f"run-{package_id}",
            evaluation_id=_evaluation_id(package_id),
            artifact_directory=tmp_path / package_id,
        )

    monkeypatch.setattr("radfusion.training.rsna_campaign.evaluate_model_package", evaluate)

    def summarize(run_ids, **kwargs):
        events.append(("summary", tuple(run_ids)))
        return SimpleNamespace(report_directory=tmp_path / f"summary-{run_ids[0]}")

    monkeypatch.setattr("radfusion.training.rsna_campaign.publish_seed_summary", summarize)
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign.generate_localization_report",
        lambda run_ids, **kwargs: (
            events.append(("localize", tuple(run_ids))) or tmp_path / "localization"
        ),
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign.regenerate_comparison",
        lambda *args, **kwargs: (
            events.append(("compare", None))
            or (tmp_path / "comparison.csv", tmp_path / "comparison.md", 16)
        ),
    )

    def validate_candidates(results):
        package_ids = tuple(result.model_package_id for result in results)
        assert package_ids == expected_package_ids
        events.append(("candidate_set_frozen", package_ids))

    monkeypatch.setattr(
        "radfusion.training.rsna_campaign._validate_frozen_training_packages",
        validate_candidates,
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign._validate_outputs",
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

    monkeypatch.setattr("radfusion.training.rsna_campaign._write_archive", archive)

    campaign = execute_rsna_campaign()

    assert [(kind, value) for kind, value in events if kind == "fusion_source"] == [
        ("fusion_source", ("fusion-17", _PACKAGE_IDS["cxr-17"])),
        ("fusion_source", ("fusion-42", _PACKAGE_IDS["cxr-42"])),
        ("fusion_source", ("fusion-2026", _PACKAGE_IDS["cxr-2026"])),
    ]
    training_positions = [index for index, value in enumerate(events) if value[0] == "train"]
    evaluation_positions = [
        index for index, value in enumerate(events) if value[0] == "test_access"
    ]
    freeze_position = next(
        index for index, value in enumerate(events) if value[0] == "candidate_set_frozen"
    )
    assert len(training_positions) == 8
    assert len(evaluation_positions) == 8
    assert max(training_positions) < min(evaluation_positions)
    assert max(training_positions) < freeze_position < min(evaluation_positions)
    assert len(training_policies) == 6
    assert all(policy.lifecycle == "reused" for policy in training_policies)
    assert len(evaluation_policies) == 8
    assert all(policy.lifecycle == "one_shot" for policy in evaluation_policies)
    assert evaluation_configs == [
        configs.metadata_logistic,
        configs.metadata_lightgbm,
        *configs.cxr,
        *configs.fusions,
    ]
    assert [kind for kind, _ in events[:6]] == [
        "prerequisites",
        "weights",
        "manifest_build",
        "manifest_validation",
        "audit",
        "cache",
    ]
    assert [value for kind, value in events if kind == "summary"] == [
        tuple(_evaluation_id(_PACKAGE_IDS[f"cxr-{seed}"]) for seed in (17, 42, 2026)),
        tuple(_evaluation_id(_PACKAGE_IDS[f"fusion-{seed}"]) for seed in (17, 42, 2026)),
    ]
    assert campaign.archive_path.is_file()
    assert campaign.checksum_path.is_file()
    assert campaign.campaign_log_path.is_file()
    campaign_log = campaign.campaign_log_path.read_text(encoding="utf-8")
    assert "event=campaign_started" in campaign_log
    assert "event=campaign_succeeded" in campaign_log
    assert [kind for kind, _ in events][-3:] == ["compare", "validate", "archive"]


def test_training_failure_never_crosses_test_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = _configs()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("radfusion.training.rsna_campaign._validate_prerequisites", lambda: configs)
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign.ensure_pretrained_weights", lambda name: None
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign.build_and_write",
        lambda *args: SimpleNamespace(
            paths=SimpleNamespace(
                bundle_id="bundle",
                bundle_directory=Path("data/manifests/rsna/bundles/bundle-test"),
                current_path=Path("data/manifests/rsna/CURRENT"),
            )
        ),
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign._validate_configured_bundle", lambda *args: None
    )
    monkeypatch.setattr("radfusion.training.rsna_campaign.generate_rsna_audit", lambda *args: None)
    monkeypatch.setattr("radfusion.training.rsna_campaign.get_dataset", lambda key: RsnaDataset())
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign.prepare_rsna_cxr_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign._required_cuda_runtime",
        lambda config: SimpleNamespace(pin_memory_effective=False),
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign.train_metadata_experiment",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("training failed")),
    )
    evaluations: list[str] = []
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign.evaluate_model_package",
        lambda package_id, **kwargs: evaluations.append(package_id),
    )
    with pytest.raises(RuntimeError):
        execute_rsna_campaign()
    assert evaluations == []
    log_path = next((tmp_path / "reports/rsna/campaigns").glob("*/execution.log"))
    assert "event=campaign_failed" in log_path.read_text(encoding="utf-8")


def test_campaign_rejects_existing_outputs_without_deleting_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    existing = Path("models/rsna/packages/model-package-existing/model.pt")
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"partial output")

    with pytest.raises(FileExistsError):
        execute_rsna_campaign()

    assert existing.read_bytes() == b"partial output"


def test_campaign_cli_preserves_process_interrupts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "radfusion.training.rsna_campaign.execute_rsna_campaign",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        main()


def test_portable_archive_contains_results_and_excludes_sources_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    included = (
        Path("data/manifests/rsna/bundles/bundle-test/manifest.json"),
        Path("data/manifests/rsna/CURRENT"),
        Path("models/rsna/packages/model-package-train/model.skops"),
        Path("reports/rsna/campaigns/campaign-test/execution.log"),
        Path("private/predictions/rsna/prediction-test/predictions.parquet"),
        Path("private/localization/localization-test/summary.json"),
    )
    for path in included:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("result\n", encoding="utf-8")
    included[1].write_text("bundle-test\n", encoding="utf-8")
    unrelated_bundle = Path("data/manifests/rsna/bundles/bundle-unrelated/manifest.json")
    unrelated_bundle.parent.mkdir(parents=True)
    unrelated_bundle.write_text("unrelated\n", encoding="utf-8")
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
    assert unrelated_bundle.as_posix() not in names
    with tarfile.open(destination, "r:gz") as archive:
        archived_log = archive.extractfile(included[3].as_posix())
        assert archived_log is not None
        log_text = archived_log.read().decode()
    assert "event=campaign_ready_for_export" in log_text
    assert "event=campaign_succeeded" not in log_text
    with closing(sqlite3.connect(exported_database)) as database:
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


@pytest.mark.parametrize("mismatch", ["preprocessing", "device", "batch_size", "workers", "seed"])
def test_campaign_rejects_neural_config_contract_mismatch(mismatch: str) -> None:
    configs = _configs()
    if mismatch == "preprocessing":
        fusions = tuple(
            replace(
                config,
                family=replace(
                    config.family,
                    parameters=MappingProxyType({**config.family.parameters, "image_size": 225}),
                ),
            )
            for config in configs.fusions
        )
    elif mismatch == "device":
        fusions = tuple(
            replace(config, runtime=replace(config.runtime, device="cpu"))
            for config in configs.fusions
        )
    elif mismatch == "seed":
        fusions = (
            replace(
                configs.fusions[0],
                runtime=replace(configs.fusions[0].runtime, seed=18),
            ),
            *configs.fusions[1:],
        )
    elif mismatch == "batch_size":
        fusions = tuple(
            replace(config, neural=replace(config.neural, batch_size=16))
            for config in configs.fusions
        )
    else:
        fusions = tuple(
            replace(config, runtime=replace(config.runtime, num_workers=4))
            for config in configs.fusions
        )

    with pytest.raises(ValueError):
        _validate_neural_campaign_configs(replace(configs, fusions=fusions))


def test_output_validation_rejects_evaluation_lineage_mismatch(tmp_path: Path) -> None:
    training = tuple(
        SimpleNamespace(
            run_id=f"train-{index}",
            model_package_id=f"model-package-{index}",
            model_path=tmp_path / "model",
            artifact_directory=tmp_path / "report",
        )
        for index in range(8)
    )
    evaluations = tuple(
        SimpleNamespace(
            evaluation_id=f"evaluation-{index}",
            model_package_id=f"wrong-package-{index}",
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
