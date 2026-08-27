from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rsna_preservation_test_support import build_real_rsna_campaign_closure

import beyondcxr.training.rsna_campaign_control as control
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.hashing import sha256_file
from beyondcxr.models.cxr_baseline import PretrainedWeightIdentity
from beyondcxr.training.config import load_experiment_config, with_runtime
from beyondcxr.training.device import ResolvedDevice
from beyondcxr.training.rsna_campaign_control import (
    FrozenTrainingPackage,
    ValidatedRsnaExecution,
    publish_evaluation_record,
    publish_or_validate_execution,
    publish_or_validate_package_freeze,
    require_validated_package_freeze,
    validate_execution,
    validate_execution_closure,
    validate_package_freeze,
)
from beyondcxr.training.rsna_datasets import RsnaDataset
from beyondcxr.training.rsna_evaluation_result import CompletedRsnaEvaluation
from beyondcxr.training.rsna_formal import (
    _FORMAL_PLAN_GUARD,
    RSNA_FORMAL_RUN_PLAN,
    RsnaAuthorityCoordinates,
    RsnaFormalRoots,
    ValidatedRsnaPlan,
    rsna_run_report_root,
)
from beyondcxr.training.rsna_training_report import (
    REQUIRED_REPORT_FILENAMES,
    training_report_sha256,
    write_run_reports,
)
from beyondcxr.training.rsna_validation_evidence import (
    CanonicalValidationCohort,
    evidence_semantic_sha256,
)
from beyondcxr.utils.rsna_model_publication import model_package_id


def _configs():
    return tuple(
        with_runtime(load_experiment_config(spec.config_relative), seed=spec.seed)
        for spec in RSNA_FORMAL_RUN_PLAN
    )


def _plan(tmp_path: Path) -> ValidatedRsnaPlan:
    configs = _configs()
    first = configs[0]
    roots = RsnaFormalRoots(
        tmp_path,
        tmp_path / "source",
        tmp_path / "manifests",
        tmp_path / "cache",
        tmp_path / "models/rsna",
        tmp_path / "reports",
        tmp_path / "private",
        tmp_path / "control",
        tmp_path / "outbox",
        tmp_path / "backup",
        tmp_path / "mlflow.db",
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
        "torch",
        "torchvision",
        "xrv",
        "12.0",
        9000,
        "GPU",
        0,
        (8, 0),
    )
    return ValidatedRsnaPlan(
        RsnaAuthorityCoordinates(
            "rsna",
            first.dataset.bundle_id,
            first.dataset.bundle_manifest_sha256,
            first.dataset.split_assignment_id,
            first.task.task_id,
            first.task.label_policy_version,
            tmp_path / "bundle",
        ),
        roots,
        RSNA_FORMAL_RUN_PLAN,
        configs,
        RsnaDataset(),
        "f" * 40,
        "0" * 64,
        PretrainedWeightIdentity("densenet121-res224-chex", "url", "weights.bin", 1, "1" * 64),
        runtime,
        SimpleNamespace(),
        SimpleNamespace(),
        _FORMAL_PLAN_GUARD,
    )


def test_execution_control_freezes_the_exact_ordered_package_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def validate_report(
        directory,
        *,
        run_id,
        model_package_id,
        family_id,
        seed,
        package,
        package_directory,
        canonical_validation_cohort,
    ):
        del package, package_directory, canonical_validation_cohort
        document = json.loads((Path(directory) / "metrics.json").read_text(encoding="utf-8"))
        if document["training_package"] != {
            "run_id": run_id,
            "model_package_id": model_package_id,
            "family_id": family_id,
            "seed": seed,
        }:
            raise ValueError("Training report package binding is invalid")

    monkeypatch.setattr(control, "validate_training_report", validate_report)
    monkeypatch.setattr(
        control,
        "load_canonical_validation_cohort",
        lambda *args, **kwargs: CanonicalValidationCohort(("sample",), (0,)),
    )
    plan = _plan(tmp_path)
    configs = plan.configs
    execution = publish_or_validate_execution(plan)
    model_root = tmp_path / "models/rsna"
    report_root = tmp_path / "reports"
    package_manifests = {}
    results = []
    for index, config in enumerate(configs):
        package_id = "model-package-" + f"{index + 1:x}" * 64
        package = model_root / "packages" / package_id
        package.mkdir(parents=True)
        model_name = "model.skops" if index < 2 else "model.pt"
        model_path = package / model_name
        model_path.write_bytes(b"model")
        report = rsna_run_report_root(report_root) / f"run-{index}"
        report.mkdir(parents=True)
        for name in REQUIRED_REPORT_FILENAMES:
            (report / name).write_bytes(b"report")
        (report / "metrics.json").write_text(
            json.dumps(
                {
                    "training_package": {
                        "run_id": f"run-{index}",
                        "model_package_id": package_id,
                        "family_id": config.family.family_id,
                        "seed": config.runtime.seed,
                    }
                }
            ),
            encoding="utf-8",
        )
        package_manifests[package_id] = {
            "bundle_id": config.dataset.bundle_id,
            "bundle_manifest_sha256": config.dataset.bundle_manifest_sha256,
            "split_assignment_id": config.dataset.split_assignment_id,
            "task_id": config.task.task_id,
            "label_policy_version": config.task.label_policy_version,
            "family_id": config.family.family_id,
            "config_source_sha256": config.config_source_sha256,
            "config_semantic_sha256": config.config_semantic_sha256,
            "seed": config.runtime.seed,
            **(
                {
                    "model_identity": {"pretrained_weight": plan.pretrained_weight.as_dict()},
                    "runtime_provenance": plan.runtime.provenance(),
                    "source_provenance": {
                        "git_commit": plan.git_commit,
                        "git_dirty": False,
                        "dependency_lock_sha256": plan.dependency_lock_sha256,
                    },
                }
                if config.neural is not None
                else {
                    "git_commit": plan.git_commit,
                    "git_dirty": False,
                    "dependency_lock_sha256": plan.dependency_lock_sha256,
                }
            ),
        }
        results.append(
            SimpleNamespace(
                run_id=f"run-{index}",
                model_package_id=package_id,
                model_path=model_path,
                artifact_directory=report,
            )
        )
    monkeypatch.setattr(
        control,
        "validate_rsna_model_package",
        lambda root, package_id: package_manifests[package_id],
    )

    frozen = publish_or_validate_package_freeze(
        plan=plan,
        execution=execution,
        results=results,
    )

    assert tuple(item.model_package_id for item in frozen.packages) == tuple(
        item.model_package_id for item in results
    )
    assert require_validated_package_freeze(frozen) is frozen

    report_bytes = (frozen.packages[0].artifact_directory / "evaluation_report.md").read_bytes()
    (frozen.packages[0].artifact_directory / "evaluation_report.md").write_bytes(
        report_bytes + b"tampered"
    )
    with pytest.raises(ManifestBuildError, match="training report"):
        frozen.require(frozen.packages[0].model_package_id)
    materialized = False

    def load_pinned_bundle(*args, **kwargs):
        nonlocal materialized
        materialized = True
        raise AssertionError("held-out materialization must not begin")

    monkeypatch.setattr("beyondcxr.training.rsna_datasets._load_pinned_bundle", load_pinned_bundle)
    with pytest.raises(ManifestBuildError, match="training report"):
        RsnaDataset().load_test(
            configs[0],
            authorization=frozen,
            package_id=frozen.packages[0].model_package_id,
        )
    assert materialized is False
    (frozen.packages[0].artifact_directory / "evaluation_report.md").write_bytes(report_bytes)

    package_manifests[frozen.packages[0].model_package_id]["seed"] = 7
    with pytest.raises(ManifestBuildError, match="package freeze output"):
        frozen.require(frozen.packages[0].model_package_id)
    package_manifests[frozen.packages[0].model_package_id]["seed"] = 42

    forged_member = FrozenTrainingPackage(
        "run-forged",
        "model-package-" + "9" * 64,
        frozen.packages[0].model_path,
        frozen.packages[0].artifact_directory,
        frozen.packages[0].report_sha256,
    )
    evaluations = execution.directory / "evaluations"
    evaluations.mkdir()
    for member in (*frozen.packages, forged_member):
        (evaluations / f"{member.model_package_id}.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="canonical authority"):
        validate_execution_closure(
            replace(frozen, packages=(*frozen.packages, forged_member)),
        )

    substituted = replace(
        frozen.packages[0],
        model_package_id="model-package-" + "f" * 64,
    )
    with pytest.raises(ManifestBuildError, match="differs from its canonical authority"):
        require_validated_package_freeze(
            replace(frozen, packages=(substituted, *frozen.packages[1:]))
        )
    with pytest.raises(ManifestBuildError, match="differs from its canonical authority"):
        require_validated_package_freeze(
            replace(
                frozen,
                packages=(frozen.packages[1], frozen.packages[0], *frozen.packages[2:]),
            )
        )

    original_bundle_id = frozen.execution.manifest["dataset"]["bundle_id"]
    replacement_hash = "0" * 64 if original_bundle_id != "bundle-" + "0" * 64 else "1" * 64
    frozen.execution.manifest["dataset"]["bundle_id"] = "bundle-" + replacement_hash
    with pytest.raises(ManifestBuildError, match="execution capability differs"):
        require_validated_package_freeze(frozen)
    frozen.execution.manifest["dataset"]["bundle_id"] = original_bundle_id

    freeze_path = execution.directory / "package-freeze.json"
    canonical_freeze = freeze_path.read_bytes()
    freeze_document = json.loads(canonical_freeze)
    freeze_document["packages"][0]["run_id"] = "run-tampered"
    freeze_path.write_text(
        json.dumps(freeze_document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestBuildError, match="package freeze output"):
        require_validated_package_freeze(frozen)
    freeze_path.write_bytes(canonical_freeze)

    other_plan = replace(plan, git_commit="e" * 40)
    other_execution = publish_or_validate_execution(other_plan)
    shutil.copyfile(
        execution.directory / "package-freeze.json",
        other_execution.directory / "package-freeze.json",
    )
    copied_freeze = json.loads(
        (other_execution.directory / "package-freeze.json").read_text(encoding="utf-8")
    )
    assert copied_freeze["execution_id"] == execution.execution_id
    assert copied_freeze["execution_id"] != other_execution.execution_id
    with pytest.raises(ManifestBuildError, match="package freeze contract is invalid"):
        require_validated_package_freeze(replace(frozen, execution=other_execution))
    with pytest.raises(ManifestBuildError, match="package freeze contract is invalid"):
        validate_package_freeze(
            other_execution,
            model_root=model_root,
            report_root=report_root,
            bundle_directory=plan.authority.bundle_directory,
        )

    package_manifests[results[2].model_package_id]["seed"] = 42
    with pytest.raises(ManifestBuildError, match="package freeze output"):
        validate_package_freeze(
            execution,
            model_root=model_root,
            report_root=report_root,
            bundle_directory=plan.authority.bundle_directory,
        )
    package_manifests[results[2].model_package_id]["seed"] = 17
    report_manifest = results[2].artifact_directory / "metrics.json"
    report_document = json.loads(report_manifest.read_text(encoding="utf-8"))
    report_document["training_package"]["model_package_id"] = "model-package-wrong"
    report_manifest.write_text(json.dumps(report_document), encoding="utf-8")
    with pytest.raises(ValueError, match="package binding"):
        validate_package_freeze(
            execution,
            model_root=model_root,
            report_root=report_root,
            bundle_directory=plan.authority.bundle_directory,
        )

    report_document["training_package"]["model_package_id"] = results[2].model_package_id
    report_manifest.write_text(json.dumps(report_document), encoding="utf-8")
    package_manifests[results[2].model_package_id]["model_identity"]["pretrained_weight"][
        "sha256"
    ] = "2" * 64
    with pytest.raises(ManifestBuildError, match="package freeze output"):
        validate_package_freeze(
            execution,
            model_root=model_root,
            report_root=report_root,
            bundle_directory=plan.authority.bundle_directory,
        )

    package_manifests[results[2].model_package_id]["model_identity"]["pretrained_weight"][
        "sha256"
    ] = plan.pretrained_weight.sha256
    package_manifests[results[2].model_package_id]["runtime_provenance"]["gpu_device_name"] = (
        "different GPU"
    )
    with pytest.raises(ManifestBuildError, match="package freeze output"):
        validate_package_freeze(
            execution,
            model_root=model_root,
            report_root=report_root,
            bundle_directory=plan.authority.bundle_directory,
        )
    package_manifests[results[2].model_package_id]["runtime_provenance"]["gpu_device_name"] = (
        plan.runtime.gpu_device_name
    )

    provenance_tampering = (
        (0, "git_commit", "e" * 40),
        (2, "dependency_lock_sha256", "e" * 64),
        (5, "git_dirty", True),
    )
    for index, field, replacement in provenance_tampering:
        manifest = package_manifests[results[index].model_package_id]
        provenance = manifest if index < 2 else manifest["source_provenance"]
        original = provenance[field]
        provenance[field] = replacement
        with pytest.raises(ManifestBuildError, match="package freeze output"):
            validate_package_freeze(
                execution,
                model_root=model_root,
                report_root=report_root,
                bundle_directory=plan.authority.bundle_directory,
            )
        provenance[field] = original

    freeze_document = json.loads(freeze_path.read_text(encoding="utf-8"))
    misplaced_validation = report_root / "some-other-place" / results[0].run_id
    shutil.copytree(results[0].artifact_directory, misplaced_validation)
    freeze_document["packages"][0]["report_relative"] = misplaced_validation.relative_to(
        report_root
    ).as_posix()
    freeze_path.write_text(
        json.dumps(freeze_document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestBuildError, match="package freeze output"):
        validate_package_freeze(
            execution,
            model_root=model_root,
            report_root=report_root,
            bundle_directory=plan.authority.bundle_directory,
        )
    freeze_path.unlink()
    publish_or_validate_package_freeze(plan=plan, execution=execution, results=results)

    misplaced = report_root / "some-other-place" / results[0].run_id
    shutil.rmtree(misplaced_validation)
    results[0].artifact_directory.rename(misplaced)
    misplaced_result = SimpleNamespace(**{**results[0].__dict__, "artifact_directory": misplaced})
    with pytest.raises(ManifestBuildError, match="report directory"):
        publish_or_validate_package_freeze(
            plan=plan,
            execution=execution,
            results=(misplaced_result, *results[1:]),
        )


def test_package_freeze_rejects_coherently_reidentified_noncanonical_validation_cohort(
    tmp_path: Path,
) -> None:
    closure = build_real_rsna_campaign_closure(tmp_path / "campaign")
    result = closure.training_results[0]
    old_package_id = result.model_package_id
    package_directory = result.model_path.parent
    evidence_path = package_directory / "validation-evidence.json"
    evidence = json.loads(evidence_path.read_bytes())
    evidence["sample_ids"][0] = "rsna:" + "-".join(
        ("00000000", "0000", "4000", "8000", "000000000000")
    )
    evidence_path.write_bytes(
        (
            json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    )

    manifest_path = package_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["validation_evidence_sha256"] = sha256_file(evidence_path)
    manifest["validation_evidence_semantic_sha256"] = evidence_semantic_sha256(evidence)
    manifest["model_package_id"] = model_package_id(manifest)
    new_package_id = manifest["model_package_id"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    renamed_package = package_directory.with_name(new_package_id)
    package_directory.rename(renamed_package)

    report = result.artifact_directory
    report_document = json.loads((report / "metrics.json").read_bytes())
    report_document["training_package"]["model_package_id"] = new_package_id
    write_run_reports(
        report,
        model_name=str(manifest["family_id"]),
        targets=np.asarray(evidence["targets"], dtype=np.int8),
        probabilities=np.asarray(evidence["probabilities"], dtype=np.float64),
        document=report_document,
    )

    freeze_path = closure.execution.directory / "package-freeze.json"
    freeze = json.loads(freeze_path.read_bytes())
    member = next(item for item in freeze["packages"] if item["package_id"] == old_package_id)
    member.update(
        {
            "package_id": new_package_id,
            "model_relative": f"packages/{new_package_id}/model.skops",
            "report_sha256": training_report_sha256(report),
        }
    )
    freeze_path.write_bytes(
        (json.dumps(freeze, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    )

    with pytest.raises(ValueError, match="canonical validation cohort"):
        validate_package_freeze(
            closure.execution,
            model_root=closure.plan.roots.model_root,
            report_root=closure.plan.roots.report_root,
            bundle_directory=closure.plan.authority.bundle_directory,
        )


@pytest.mark.parametrize(
    "model_relative",
    (
        "./packages/{package_id}/model.skops",
        "packages//{package_id}/model.skops",
        "packages/./{package_id}/model.skops",
    ),
)
def test_package_freeze_rejects_lexically_noncanonical_control_paths(
    tmp_path: Path, model_relative: str
) -> None:
    closure = build_real_rsna_campaign_closure(tmp_path / "campaign")
    freeze_path = closure.execution.directory / "package-freeze.json"
    document = json.loads(freeze_path.read_bytes())
    package_id = document["packages"][0]["package_id"]
    document["packages"][0]["model_relative"] = model_relative.format(package_id=package_id)
    freeze_path.write_bytes(
        (
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    )

    with pytest.raises(ManifestBuildError, match="RSNA control path is invalid"):
        validate_package_freeze(
            closure.execution,
            model_root=closure.plan.roots.model_root,
            report_root=closure.plan.roots.report_root,
            bundle_directory=closure.plan.authority.bundle_directory,
        )


def test_package_freeze_rejects_a_valid_execution_from_a_different_plan(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    execution = publish_or_validate_execution(plan)
    different_plan = replace(plan, git_commit="e" * 40)

    with pytest.raises(ManifestBuildError, match="does not match the validated plan"):
        publish_or_validate_package_freeze(
            plan=different_plan,
            execution=execution,
            results=(),
        )


def test_package_freeze_rejects_forged_and_tampered_executions(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    execution = publish_or_validate_execution(plan)
    forged = ValidatedRsnaExecution(
        execution.execution_id,
        execution.directory,
        execution.manifest,
        object(),
    )
    with pytest.raises(PermissionError, match="validated execution"):
        publish_or_validate_package_freeze(plan=plan, execution=forged, results=())
    with pytest.raises(PermissionError, match="validated execution"):
        validate_package_freeze(
            forged,
            model_root=plan.roots.model_root,
            report_root=plan.roots.report_root,
            bundle_directory=plan.authority.bundle_directory,
        )

    tampered = replace(
        execution,
        manifest={**execution.manifest, "git_commit": "e" * 40},
    )
    with pytest.raises(ManifestBuildError, match="canonical authority"):
        validate_package_freeze(
            tampered,
            model_root=plan.roots.model_root,
            report_root=plan.roots.report_root,
            bundle_directory=plan.authority.bundle_directory,
        )


def test_execution_and_evaluation_records_are_append_only(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    execution = publish_or_validate_execution(plan)
    assert validate_execution(execution.directory) == execution
    path = execution.directory / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["configs"][0]["seed"] = 17
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ManifestBuildError):
        validate_execution(execution.directory)

    # Restore the exact control, then prove a conflicting completion cannot replace the first.
    path.unlink()
    execution = publish_or_validate_execution(plan)
    first = CompletedRsnaEvaluation(
        evaluation_id="evaluation-" + "1" * 64,
        prediction_id="prediction-" + "2" * 64,
        model_package_id="model-package-" + "3" * 64,
        mlflow_run_id="run-1",
        artifact_directory=tmp_path / "evaluation",
        private_prediction_directory=tmp_path / "prediction",
        average_precision=0.5,
    )
    publish_evaluation_record(execution, first)
    conflicting = CompletedRsnaEvaluation(
        **{**first.__dict__, "evaluation_id": "evaluation-" + "4" * 64}
    )
    with pytest.raises(ManifestBuildError, match="differs"):
        publish_evaluation_record(execution, conflicting)


def test_execution_control_rejects_a_forged_formal_plan_before_publication(
    tmp_path: Path,
) -> None:
    forged = replace(_plan(tmp_path), _guard=object())

    with pytest.raises(ValueError, match="genuine prevalidated"):
        publish_or_validate_execution(forged)

    assert not (tmp_path / "control").exists()
