from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from rsna_validation_evidence_test_support import synthetic_cxr_epoch_history

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.hashing import sha256_file
from beyondcxr.data.rsna_artifacts import build_and_write
from beyondcxr.data.rsna_audit import generate_rsna_audit
from beyondcxr.data.rsna_cxr_cache import SOURCE_AUTHENTICATION_POLICY_VERSION
from beyondcxr.data.rsna_metadata_preprocess import (
    SOURCE_FEATURES,
    build_rsna_preprocessor,
    fitted_rsna_preprocessor_contract,
    metadata_input_contract,
    save_preprocessor,
)
from beyondcxr.data.rsna_splitting import SplitConfig
from beyondcxr.evaluation.metrics import evaluate_operating_point, evaluate_probabilities
from beyondcxr.models.cxr_baseline import PretrainedWeightIdentity
from beyondcxr.training.config import load_experiment_config, with_runtime
from beyondcxr.training.device import ResolvedDevice
from beyondcxr.training.execution import one_shot_loader_policy, reused_loader_policy
from beyondcxr.training.rsna_campaign_control import (
    publish_evaluation_record,
    publish_or_validate_execution,
    publish_or_validate_package_freeze,
)
from beyondcxr.training.rsna_compare import ComparisonResult, regenerate_comparison
from beyondcxr.training.rsna_datasets import (
    CxrRunData,
    FusionRunData,
    RsnaDataset,
    SourceInventoryIdentity,
)
from beyondcxr.training.rsna_evaluation_result import (
    CompletedRsnaEvaluation,
    publish_rsna_evaluation,
)
from beyondcxr.training.rsna_formal import (
    _FORMAL_PLAN_GUARD,
    RSNA_FORMAL_RUN_PLAN,
    RsnaAuthorityCoordinates,
    RsnaFormalRoots,
    ValidatedRsnaPlan,
    rsna_run_report_root,
)
from beyondcxr.training.rsna_interfaces import DatasetLineage
from beyondcxr.training.rsna_localize import (
    LOCALIZATION_EVIDENCE_FILENAME,
    PRIVATE_LOCALIZATION_SCHEMA_VERSION,
    QUALITATIVE_POLICY_VERSION,
    _localization_id,
    _markdown,
    _report_document,
    _write_localization_evidence,
    validate_localization_report,
)
from beyondcxr.training.rsna_preservation import campaign_export_members
from beyondcxr.training.rsna_registry import get_model
from beyondcxr.training.rsna_seed_summary import SeedSummaryResult, publish_seed_summary
from beyondcxr.training.rsna_train_cxr import _manifest as cxr_manifest
from beyondcxr.training.rsna_train_fusion import _manifest as fusion_manifest
from beyondcxr.training.rsna_training_report import (
    CXR_REPORT_LIMITATIONS,
    CXR_REPORT_RUNTIME_FIELDS,
    metrics_document,
    validate_training_report,
    write_run_reports,
)
from beyondcxr.training.rsna_validation_evidence import (
    CanonicalValidationCohort,
    load_canonical_validation_cohort,
    write_validation_evidence,
)
from beyondcxr.utils.private_predictions import publish_prediction_evidence
from beyondcxr.utils.rsna_model_publication import publish_model_package, threshold_contract
from beyondcxr.utils.rsna_neural_publication import (
    checkpoint_document,
    publish_neural_model_package,
    save_neural_checkpoint,
)
from beyondcxr.utils.skops_io import save_skops


@dataclass(frozen=True)
class RealRsnaCampaignClosure:
    plan: ValidatedRsnaPlan
    execution: object
    package_freeze: object
    training_results: tuple[object, ...]
    evaluations: tuple[CompletedRsnaEvaluation, ...]
    summaries: tuple[SeedSummaryResult, ...]
    localization: Path
    comparison: ComparisonResult
    audit_directory: Path
    campaign_log: Path

    @property
    def members(self):
        return campaign_export_members(
            package_freeze=self.package_freeze,
            evaluations=self.evaluations,
            summaries=self.summaries,
            localization=self.localization,
            comparison=self.comparison,
            audit_directory=self.audit_directory,
            campaign_log=self.campaign_log,
            private_root=self.plan.roots.private_root,
        )


def build_real_rsna_campaign_closure(workspace: Path) -> RealRsnaCampaignClosure:
    workspace.mkdir()
    raw_root = _write_raw_source(workspace / "raw")
    manifest_root = workspace / "data/manifests"
    bundle = build_and_write(
        raw_root,
        manifest_root,
        split_config=SplitConfig(train_ratio=0.5, validation_ratio=0.16, test_ratio=0.34),
    )
    bundle_document = json.loads(bundle.paths.metadata_path.read_text(encoding="utf-8"))
    split_id = str(bundle_document["membership"]["split"]["split_assignment_id"])
    manifest_sha256 = sha256_file(bundle.paths.metadata_path)
    configs = _write_configs(
        workspace,
        bundle_id=bundle.paths.bundle_id,
        bundle_manifest_sha256=manifest_sha256,
        split_assignment_id=split_id,
    )
    runtime = ResolvedDevice(
        torch.device("cuda"),
        "cuda",
        "cuda",
        True,
        True,
        True,
        "auto",
        True,
        "torch-test",
        "torchvision-test",
        "torchxrayvision-test",
        "12.0",
        9000,
        "Synthetic CUDA",
        0,
        (8, 0),
    )
    weight = PretrainedWeightIdentity(
        "densenet121-res224-chex",
        "https://example.invalid/densenet121-res224-chex.pt",
        "densenet121-res224-chex.pt",
        1,
        "1" * 64,
    )
    roots = RsnaFormalRoots(
        workspace,
        raw_root,
        manifest_root,
        workspace / "data/cache",
        workspace / "models/rsna",
        workspace / "reports",
        workspace / "private",
        workspace / "private/control/rsna",
        workspace / "outbox",
        workspace.parent / "backup",
        workspace / "mlflow.db",
    )
    plan = ValidatedRsnaPlan(
        RsnaAuthorityCoordinates(
            "rsna",
            bundle.paths.bundle_id,
            manifest_sha256,
            split_id,
            "pneumonia",
            "rsna-stage-2-target-v1",
            bundle.paths.bundle_directory,
        ),
        roots,
        RSNA_FORMAL_RUN_PLAN,
        configs,
        RsnaDataset(),
        "f" * 40,
        "0" * 64,
        weight,
        runtime,
        reused_loader_policy(num_workers=0, pin_memory=True),
        one_shot_loader_policy(pin_memory=True),
        _FORMAL_PLAN_GUARD,
    )
    execution = publish_or_validate_execution(plan)
    validation_cohort = load_canonical_validation_cohort(
        bundle.paths.bundle_directory,
        expected_bundle_id=plan.authority.bundle_id,
        expected_manifest_sha256=plan.authority.bundle_manifest_sha256,
        expected_split_assignment_id=plan.authority.split_assignment_id,
        expected_task_id=plan.authority.task_id,
        expected_label_policy_version=plan.authority.label_policy_version,
    )
    source_inventory = SourceInventoryIdentity(
        source_inventory_arrow_sha256=str(
            bundle_document["artifacts"]["source_inventory.parquet"]["logical_arrow_sha256"]
        ),
        source_inventory_file_sha256=str(
            bundle_document["artifacts"]["source_inventory.parquet"]["physical_file_sha256"]
        ),
    )
    authentication = {
        "policy_version": SOURCE_AUTHENTICATION_POLICY_VERSION,
        "partitions": ["train", "validation", "test"],
        "file_count": 12,
        "source_inventory_arrow_sha256": source_inventory.source_inventory_arrow_sha256,
        "source_inventory_file_sha256": source_inventory.source_inventory_file_sha256,
    }
    lineage = DatasetLineage(
        bundle_id=bundle.paths.bundle_id,
        split_assignment_id=split_id,
        label_policy_version="rsna-stage-2-target-v1",
        task_id="pneumonia",
    )
    features = _metadata_features()
    training_results: list[object] = []
    cxr_by_seed: dict[int, object] = {}
    for index, (spec, config) in enumerate(zip(RSNA_FORMAL_RUN_PLAN, configs, strict=True)):
        if spec.family_id.startswith("metadata_"):
            published = _publish_metadata_package(
                workspace,
                config=config,
                features=features,
                index=index,
                commit=plan.git_commit,
                lock_hash=plan.dependency_lock_sha256,
                validation_cohort=validation_cohort,
            )
        elif spec.family_id == "cxr_densenet":
            published = _publish_cxr_package(
                workspace,
                config=config,
                lineage=lineage,
                bundle_manifest_sha256=manifest_sha256,
                source_inventory=source_inventory,
                source_authentication=authentication,
                plan=plan,
                index=index,
                validation_cohort=validation_cohort,
            )
            cxr_by_seed[spec.seed] = published
        else:
            published = _publish_fusion_package(
                workspace,
                config=config,
                source_package=cxr_by_seed[spec.seed],
                features=features,
                lineage=lineage,
                bundle_manifest_sha256=manifest_sha256,
                source_inventory=source_inventory,
                source_authentication=authentication,
                plan=plan,
                index=index,
                validation_cohort=validation_cohort,
            )
        run_id = f"run-{index:02d}"
        report = _write_training_report(
            rsna_run_report_root(roots.report_root) / run_id,
            run_id=run_id,
            package_id=published.model_package_id,
            family_id=spec.family_id,
            seed=spec.seed,
            package=json.loads(published.manifest_path.read_text(encoding="utf-8")),
            validation_cohort=validation_cohort,
        )
        training_results.append(
            SimpleNamespace(
                run_id=run_id,
                model_package_id=published.model_package_id,
                model_path=published.model_path,
                artifact_directory=report,
            )
        )
    package_freeze = publish_or_validate_package_freeze(
        plan=plan,
        execution=execution,
        results=training_results,
    )
    evaluations = tuple(
        _publish_evaluation(workspace, execution, result, index)
        for index, result in enumerate(training_results)
    )
    family_by_package = {
        result.model_package_id: spec.family_id
        for result, spec in zip(training_results, RSNA_FORMAL_RUN_PLAN, strict=True)
    }
    summaries = tuple(
        publish_seed_summary(
            [
                result.evaluation_id
                for result in evaluations
                if family_by_package[result.model_package_id] == family
            ],
            output_directory=roots.report_root,
            model_directory=roots.model_root,
            private_directory=roots.private_root,
        )
        for family in ("cxr_densenet", "cxr_metadata_concat")
    )
    cxr_package_ids = tuple(cxr_by_seed[seed].model_package_id for seed in (17, 42, 2026))
    test_ids = set(
        pq.read_table(
            bundle.paths.splits_path,
            columns=["sample_id"],
            filters=[("split_name", "=", "test")],
        )
        .to_pandas()["sample_id"]
        .astype(str)
    )
    positive_test_ids = tuple(
        sorted(
            set(
                pq.read_table(
                    bundle.paths.labels_path,
                    columns=["sample_id"],
                    filters=[("task_id", "=", "pneumonia"), ("label_value", "=", 1)],
                )
                .to_pandas()["sample_id"]
                .astype(str)
            )
            & test_ids
        )
    )
    localization = _write_localization(workspace, cxr_package_ids, positive_test_ids)
    comparison = regenerate_comparison(
        [result.evaluation_id for result in evaluations],
        output_directory=roots.report_root,
        private_directory=roots.private_root,
        model_directory=roots.model_root,
    )
    generate_rsna_audit(
        manifest_root,
        roots.report_root / "rsna/audit",
        bundle_id=bundle.paths.bundle_id,
    )
    audit_directory = roots.report_root / "rsna/audit" / bundle.paths.bundle_id
    campaign_log = roots.report_root / "rsna/campaigns" / execution.execution_id / "execution.log"
    campaign_log.parent.mkdir(parents=True)
    campaign_log.write_text("event=campaign_ready_for_export\n", encoding="utf-8")
    return RealRsnaCampaignClosure(
        plan,
        execution,
        package_freeze,
        tuple(training_results),
        evaluations,
        summaries,
        localization,
        comparison,
        audit_directory,
        campaign_log,
    )


def _write_configs(
    workspace: Path,
    *,
    bundle_id: str,
    bundle_manifest_sha256: str,
    split_assignment_id: str,
):
    config_root = workspace / "configs"
    config_root.mkdir()
    loaded = []
    for index, spec in enumerate(RSNA_FORMAL_RUN_PLAN):
        document = yaml.safe_load(spec.config_relative.read_text(encoding="utf-8"))
        document["dataset"].update(
            {
                "bundle_id": bundle_id,
                "bundle_manifest_sha256": bundle_manifest_sha256,
                "split_assignment_id": split_assignment_id,
            }
        )
        path = config_root / f"{index:02d}-{spec.config_relative.name}"
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        loaded.append(with_runtime(load_experiment_config(path), seed=spec.seed))
    return tuple(loaded)


def _metadata_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age_years": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            "age_is_implausible": [False] * 8,
            "sex": ["F", "M", "F", "M", "F", "M", "F", "M"],
            "view_position": ["PA", "AP", "PA", "AP", "PA", "AP", "PA", "AP"],
            "pixel_spacing_row_mm": [0.1] * 8,
            "pixel_spacing_col_mm": [0.1] * 8,
        }
    ).loc[:, SOURCE_FEATURES]


def _publish_metadata_package(
    workspace: Path,
    *,
    config,
    features: pd.DataFrame,
    index: int,
    commit: str,
    lock_hash: str,
    validation_cohort: CanonicalValidationCohort,
):
    targets = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    fit = get_model(config.family.family_id).fit(
        config,
        42,
        features,
        targets,
        features,
        targets,
    )
    best_iteration = fit.derived_parameters.get("best_iteration")
    serialized = save_skops(fit.pipeline, workspace / f"temporary-metadata-{index}.skops")
    evidence = write_validation_evidence(
        workspace / f"temporary-metadata-evidence-{index}.json",
        config=config,
        sample_ids=validation_cohort.sample_ids,
        targets=validation_cohort.targets,
        probabilities=_validation_probabilities(validation_cohort),
    )
    return publish_model_package(
        model_root=workspace / "models/rsna",
        serialized_model_path=serialized,
        source_config_bytes=config.source_bytes,
        validation_evidence_path=evidence,
        manifest={
            "bundle_id": config.dataset.bundle_id,
            "split_assignment_id": config.dataset.split_assignment_id,
            "task_id": config.task.task_id,
            "positive_class": 1,
            "family_id": config.family.family_id,
            "config_source_sha256": config.config_source_sha256,
            "config_semantic_sha256": config.config_semantic_sha256,
            "seed": 42,
            "git_commit": commit,
            "git_dirty": False,
            "dependency_lock_sha256": lock_hash,
            "best_iteration": best_iteration,
            "thresholds": {"youden_j": 0.7, "target_sensitivity": 0.7},
            "threshold_contract": threshold_contract(sensitivity_target=0.9),
            "input_contract": metadata_input_contract(),
        },
    )


def _neural_common(config, plan: ValidatedRsnaPlan):
    neural = config.neural
    assert neural is not None
    transform_kwargs = {
        "image_size": int(config.family.parameters["image_size"]),
        "rotation_degrees": neural.rotation_degrees,
        "translation_fraction": neural.translation_fraction,
        "brightness_jitter": neural.brightness_jitter,
        "contrast_jitter": neural.contrast_jitter,
    }
    runtime = {
        **plan.runtime.provenance(),
        "cxr_cache_id": "cache-" + "2" * 64,
        "loader_execution": {"lifecycle": "reused", "num_workers": 0, "pin_memory": True},
    }
    return (
        StandardCxrTransform(training=True, **transform_kwargs).contract(),
        StandardCxrTransform(training=False, **transform_kwargs).contract(),
        runtime,
    )


def _publish_cxr_package(
    workspace: Path,
    *,
    config,
    lineage: DatasetLineage,
    bundle_manifest_sha256: str,
    source_inventory: SourceInventoryIdentity,
    source_authentication: dict[str, object],
    plan: ValidatedRsnaPlan,
    index: int,
    validation_cohort: CanonicalValidationCohort,
):
    checkpoint = checkpoint_document(
        {"weight": torch.tensor([float(index)])},
        selected_epoch=3,
        selected_stage="fine_tune",
        validation_average_precision=1.0,
    )
    checkpoint_path = save_neural_checkpoint(checkpoint, workspace / f"temporary-cxr-{index}.pt")
    train_transform, evaluation_transform, runtime = _neural_common(config, plan)
    manifest = cxr_manifest(
        config=config,
        cxr_data=CxrRunData(
            pd.DataFrame(),
            pd.DataFrame(),
            lineage,
            bundle_manifest_sha256,
            source_inventory,
        ),
        source_authentication=source_authentication,
        commit=plan.git_commit,
        dirty=False,
        lock_hash=plan.dependency_lock_sha256,
        environment={"environment_python_version": "3.13"},
        runtime=runtime,
        weight_identity=plan.pretrained_weight.as_dict(),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        positive_count=4,
        negative_count=4,
        pos_weight=1.0,
        fit=SimpleNamespace(selected_epoch=3, selected_stage="fine_tune"),
        final_average_precision=1.0,
        thresholds={"youden_j": 0.7, "target_sensitivity": 0.7},
    )
    return publish_neural_model_package(
        model_root=workspace / "models/rsna",
        checkpoint_path=checkpoint_path,
        source_config_bytes=config.source_bytes,
        validation_evidence_path=write_validation_evidence(
            workspace / f"temporary-cxr-evidence-{index}.json",
            config=config,
            sample_ids=validation_cohort.sample_ids,
            targets=validation_cohort.targets,
            probabilities=_validation_probabilities(validation_cohort),
            epoch_history=synthetic_cxr_epoch_history(
                config,
                selected_epoch=3,
                selected_stage="fine_tune",
                selected_validation_average_precision=1.0,
            ),
        ),
        manifest=manifest,
    )


def _publish_fusion_package(
    workspace: Path,
    *,
    config,
    source_package,
    features: pd.DataFrame,
    lineage: DatasetLineage,
    bundle_manifest_sha256: str,
    source_inventory: SourceInventoryIdentity,
    source_authentication: dict[str, object],
    plan: ValidatedRsnaPlan,
    index: int,
    validation_cohort: CanonicalValidationCohort,
):
    preprocessor = build_rsna_preprocessor().fit(features)
    contract = fitted_rsna_preprocessor_contract(preprocessor)
    preprocessor_path = save_preprocessor(
        preprocessor, workspace / f"temporary-fusion-preprocessor-{index}.skops"
    )
    checkpoint = checkpoint_document(
        {"weight": torch.tensor([float(index)])},
        selected_epoch=3,
        selected_stage="fine_tune",
        validation_average_precision=1.0,
    )
    checkpoint_path = save_neural_checkpoint(checkpoint, workspace / f"temporary-fusion-{index}.pt")
    train_transform, evaluation_transform, runtime = _neural_common(config, plan)
    manifest = fusion_manifest(
        config=config,
        data=FusionRunData(
            features,
            features,
            lineage,
            bundle_manifest_sha256,
            source_inventory,
        ),
        source_authentication=source_authentication,
        commit=plan.git_commit,
        dirty=False,
        lock_hash=plan.dependency_lock_sha256,
        environment={"environment_python_version": "3.13"},
        runtime=runtime,
        source_package_id=source_package.model_package_id,
        source_pretrained_weight=plan.pretrained_weight.as_dict(),
        structured_contract=contract,
        preprocessor_sha256=sha256_file(preprocessor_path),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        positive_count=4,
        negative_count=4,
        pos_weight=1.0,
        fit=SimpleNamespace(
            selected_epoch=3,
            selected_stage="fine_tune",
            selected_validation_metric=1.0,
        ),
        thresholds={"youden_j": 0.7, "target_sensitivity": 0.7},
    )
    return publish_neural_model_package(
        model_root=workspace / "models/rsna",
        checkpoint_path=checkpoint_path,
        source_config_bytes=config.source_bytes,
        validation_evidence_path=write_validation_evidence(
            workspace / f"temporary-fusion-evidence-{index}.json",
            config=config,
            sample_ids=validation_cohort.sample_ids,
            targets=validation_cohort.targets,
            probabilities=_validation_probabilities(validation_cohort),
        ),
        manifest=manifest,
        structured_preprocessor_path=preprocessor_path,
    )


def _write_training_report(
    directory: Path,
    *,
    run_id: str,
    package_id: str,
    family_id: str,
    seed: int,
    package: dict[str, object],
    validation_cohort: CanonicalValidationCohort,
) -> Path:
    targets = np.asarray(validation_cohort.targets, dtype=np.int8)
    probabilities = np.asarray(_validation_probabilities(validation_cohort), dtype=np.float64)
    thresholds = {"youden_j": 0.7, "target_sensitivity": 0.7}
    document = metrics_document(
        scope="validation",
        calibration_bins=15,
        sensitivity_target=0.9,
        thresholds=thresholds,
        probability=evaluate_probabilities(targets, probabilities, calibration_bins=15),
        youden=evaluate_operating_point(targets, probabilities, threshold=0.7),
        target_sensitivity=evaluate_operating_point(targets, probabilities, threshold=0.7),
    )
    document["training_package"] = {
        "run_id": run_id,
        "model_package_id": package_id,
        "family_id": family_id,
        "seed": seed,
    }
    selection = package.get("selection")
    if isinstance(selection, dict):
        document["probability_metrics"]["average_precision"] = selection[
            "validation_average_precision"
        ]
    if family_id == "cxr_densenet":
        document["cxr_training"] = {
            "lineage": {
                key: package[key]
                for key in (
                    "bundle_id",
                    "bundle_manifest_sha256",
                    "split_assignment_id",
                    "task_id",
                    "label_policy_version",
                )
            },
            "source_authentication": package["source_authentication"],
            "model_identity": package["model_identity"],
            "input_contract": package["input_contract"],
            "training_transform_contract": package["training_transform_contract"],
            "evaluation_transform_contract": package["evaluation_transform_contract"],
            "class_weighting": package["training_policy"]["class_weight"],
            "runtime": {
                field: package["runtime_provenance"][field] for field in CXR_REPORT_RUNTIME_FIELDS
            },
            "epoch_history": synthetic_cxr_epoch_history(
                with_runtime(
                    load_experiment_config("configs/rsna_cxr_densenet.yaml"),
                    seed=seed,
                ),
                selected_epoch=3,
                selected_stage="fine_tune",
                selected_validation_average_precision=1.0,
            ),
            "selection": package["selection"],
            "limitations": CXR_REPORT_LIMITATIONS,
        }
    write_run_reports(
        directory,
        model_name=family_id,
        targets=targets,
        probabilities=probabilities,
        document=document,
    )
    validate_training_report(
        directory,
        run_id=run_id,
        model_package_id=package_id,
        family_id=family_id,
        seed=seed,
        package=package,
        package_directory=directory.parent.parent.parent.parent
        / "models/rsna/packages"
        / package_id,
        canonical_validation_cohort=validation_cohort,
    )
    return directory


def _validation_probabilities(cohort: CanonicalValidationCohort) -> tuple[float, ...]:
    return tuple(0.7 if target else 0.3 for target in cohort.targets)


def _publish_evaluation(
    workspace: Path,
    execution,
    training_result,
    index: int,
) -> CompletedRsnaEvaluation:
    package = json.loads((training_result.model_path.parent / "manifest.json").read_text())
    sample_ids = tuple(f"rsna:preservation-synthetic-{item}" for item in range(6))
    evidence = publish_prediction_evidence(
        private_root=workspace / "private",
        dataset_id="rsna",
        model_package_id=training_result.model_package_id,
        task_id="pneumonia",
        bundle_id=package["bundle_id"],
        split_assignment_id=package["split_assignment_id"],
        scope="test",
        sample_ids=sample_ids,
        targets=(0, 1, 0, 1, 0, 1),
        logits=(-2.0 + index / 10, 2.0, -1.5, 1.5, -1.0, 1.0),
    )
    validated = publish_rsna_evaluation(
        report_root=workspace / "reports",
        private_root=workspace / "private",
        model_root=workspace / "models/rsna",
        evidence=evidence,
        evaluation_policy={
            "policy_version": "rsna-held-out-evaluation-v1",
            "calibration_bins": 15,
            "threshold_selection": threshold_contract(sensitivity_target=0.9),
        },
        forbidden_source_values=sample_ids,
    )
    result = CompletedRsnaEvaluation(
        validated.manifest["evaluation_id"],
        evidence.prediction_id,
        training_result.model_package_id,
        f"evaluation-run-{index:02d}",
        validated.directory,
        evidence.directory,
        float(validated.manifest["claims"]["probability_metrics"]["average_precision"]),
    )
    publish_evaluation_record(execution, result)
    return result


def _write_localization(
    workspace: Path,
    package_ids: tuple[str, ...],
    positive_sample_ids: tuple[str, ...],
) -> Path:
    report_id = _localization_id(package_ids)
    public = workspace / "reports/rsna/localization" / report_id
    private = workspace / "private/localization" / report_id
    examples = private / "examples"
    public.mkdir(parents=True)
    examples.mkdir(parents=True)
    example_name = "synthetic-overlay.png"
    (examples / example_name).write_bytes(b"synthetic-private-overlay")
    private_members = [{"filename": example_name}]
    members = [
        {
            "seed": seed,
            "positive_test_sample_count": len(positive_sample_ids),
            "localization_evaluated_count": len(positive_sample_ids),
            "zero_heatmap_count": 0,
            "pointing_game_accuracy": 0.5,
            "mean_activation_energy_inside_union": 0.4,
            "qualitative_strata_present": {
                "TP": True,
                "FN": False,
                "FP": False,
                "TN": False,
            },
        }
        for seed in (17, 42, 2026)
    ]
    evidence_members = [
        {
            "model_package_id": package_id,
            "seed": seed,
            "cases": [
                {
                    "sample_id": sample_id,
                    "pointing_game": index % 2,
                    "activation_energy_inside_union": 0.3 if index % 2 else 0.5,
                    "zero_heatmap": False,
                }
                for index, sample_id in enumerate(positive_sample_ids)
            ],
            "qualitative_strata_present": members[index]["qualitative_strata_present"],
        }
        for index, (package_id, seed) in enumerate(zip(package_ids, (17, 42, 2026), strict=True))
    ]
    _write_localization_evidence(
        private / LOCALIZATION_EVIDENCE_FILENAME,
        report_id=report_id,
        model_package_ids=package_ids,
        members=evidence_members,
        expected_positive_sample_ids=positive_sample_ids,
    )
    document = _report_document(report_id, package_ids, members)
    (public / "summary.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (public / "summary.md").write_text(_markdown(document), encoding="utf-8")
    (private / "qualitative_manifest.json").write_text(
        json.dumps(
            {
                "private_localization_schema_version": PRIVATE_LOCALIZATION_SCHEMA_VERSION,
                "report_id": report_id,
                "selection_policy": QUALITATIVE_POLICY_VERSION,
                "examples": private_members,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return validate_localization_report(
        public,
        private_directory=private,
        expected_positive_sample_ids=positive_sample_ids,
        expected_report_id=report_id,
        expected_model_package_ids=package_ids,
    )


def _write_raw_source(root: Path) -> Path:
    images = root / "stage_2_train_images"
    images.mkdir(parents=True)
    labels = []
    classes = []
    for target in (0, 1):
        for index in range(6):
            patient_id = f"patient-{target}-{index}"
            _write_dicom(images / f"{patient_id}.dcm", patient_id, age=35 + target * 20 + index)
            labels.append(
                {
                    "patientId": patient_id,
                    "x": 10 if target else None,
                    "y": 20 if target else None,
                    "width": 30 if target else None,
                    "height": 40 if target else None,
                    "Target": target,
                }
            )
            classes.append(
                {"patientId": patient_id, "class": "Lung Opacity" if target else "Normal"}
            )
    pd.DataFrame(labels).to_csv(root / "stage_2_train_labels.csv", index=False)
    pd.DataFrame(classes).to_csv(root / "stage_2_detailed_class_info.csv", index=False)
    return root


def _write_dicom(path: Path, patient_id: str, *, age: int) -> None:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.PatientID = patient_id
    dataset.PatientAge = f"{age:03d}Y"
    dataset.PatientSex = "F" if age % 2 else "M"
    dataset.ViewPosition = "PA"
    dataset.PixelSpacing = [0.168, 0.168]
    dataset.Rows = 1024
    dataset.Columns = 1024
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.Modality = "CR"
    dataset.BodyPartExamined = "CHEST"
    dataset.save_as(path)
