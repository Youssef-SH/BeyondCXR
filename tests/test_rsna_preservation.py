from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import beyondcxr.training.rsna_preservation as rsna_preservation
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.campaign_observability import campaign_log_members, campaign_session
from beyondcxr.training.preservation import (
    PreservationMember,
    export_and_verify,
    restore_and_validate_export,
)
from beyondcxr.training.rsna_preservation import (
    campaign_export_members,
    validate_restored_rsna_campaign,
)
from beyondcxr.utils.operational_logging import (
    get_operational_logger,
    log_event,
    parse_operational_record,
)
from beyondcxr.utils.semantic_identity import canonical_json_bytes


@pytest.mark.integration
def test_recursive_preservation_revalidates_bundle_evidence(tmp_path: Path) -> None:
    from rsna_manifest_test_support import write_sources

    from beyondcxr.data.rsna_artifacts import build_and_write, validate_bundle_directory

    bundle = build_and_write(write_sources(tmp_path / "raw"), tmp_path / "manifests")
    relative = Path("data/manifests/rsna/bundles") / bundle.paths.bundle_id
    members = (PreservationMember(bundle.paths.bundle_directory, relative),)

    def validate(root: Path) -> None:
        validate_bundle_directory(root / relative)

    arguments = dict(members=members, backup_root=tmp_path / "backup", export_name="synthetic-rsna")
    export_and_verify(**arguments, export_root=tmp_path / "export", restoration_validator=validate)
    labels = pq.read_table(bundle.paths.labels_path).to_pylist()
    labels[0]["label_value"] = 1 - labels[0]["label_value"]
    pq.write_table(pa.Table.from_pylist(labels), bundle.paths.labels_path)
    arguments["backup_root"] = tmp_path / "altered-backup"
    altered = export_and_verify(**arguments, export_root=tmp_path / "altered-export")
    with pytest.raises((ValueError, ManifestBuildError)):
        restore_and_validate_export(altered, restoration_validator=validate)


class _BoundedFreeze:
    def __init__(
        self,
        *,
        directory: Path,
        model_root: Path,
        bundle_directory: Path,
        packages: tuple[object, ...],
    ) -> None:
        self.freeze_id = "rsna-freeze-" + "f" * 64
        self.directory = directory
        self.model_root = model_root
        self.bundle_directory = bundle_directory
        self.packages = packages

    def __iter__(self):
        return iter(self.packages)


def _marker(directory: Path, name: str = "manifest.json", content: bytes = b"{}\n") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(content)
    return directory


def _bounded_closure(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PreservationMember, ...]:
    bundle_id = "bundle-" + "a" * 64
    execution_id = "rsna-execution-" + "b" * 64
    package_ids = ("model-package-" + "c" * 64, "model-package-" + "d" * 64)
    evaluation_ids = ("evaluation-" + "e" * 64, "evaluation-" + "1" * 64)
    prediction_ids = ("prediction-" + "2" * 64, "prediction-" + "3" * 64)
    families = dict(zip(package_ids, ("cxr_densenet", "cxr_metadata_concat"), strict=True))

    bundle = _marker(root / "data/manifests/rsna/bundles" / bundle_id)
    sample_ids = ["rsna:positive", "rsna:negative", "rsna:training-positive"]
    pq.write_table(
        pa.table({"sample_id": sample_ids, "split_name": ["test", "test", "train"]}),
        bundle / "splits.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "sample_id": sample_ids,
                "task_id": ["pneumonia"] * 3,
                "label_value": pa.array([1, 0, 1], type=pa.int8()),
            }
        ),
        bundle / "labels.parquet",
    )

    execution_directory = _marker(root / "private/control/rsna/executions" / execution_id)
    evaluation_control = root / "private/control/rsna/evaluations" / execution_id
    for package_id in package_ids:
        _marker(evaluation_control, f"{package_id}.json")
    active = root / "private/control/rsna/active-execution.json"
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_bytes(canonical_json_bytes({"schema_version": 1, "execution_id": execution_id}))
    freeze_directory = _marker(root / "private/control/rsna/freezes" / ("rsna-freeze-" + "f" * 64))

    model_root = root / "models/rsna"
    report_root = root / "reports"
    private_root = root / "private"
    packages = []
    training_results = []
    evaluations = []
    for package_id, evaluation_id, prediction_id in zip(
        package_ids, evaluation_ids, prediction_ids, strict=True
    ):
        package = _marker(model_root / "packages" / package_id)
        model_path = package / "model.skops"
        model_path.write_bytes(b"bounded fitted state")
        run = _marker(report_root / "rsna/runs" / package_id)
        evaluation = _marker(report_root / "rsna/evaluations" / evaluation_id)
        prediction = _marker(private_root / "predictions/rsna" / prediction_id)
        packages.append(SimpleNamespace(model_package_id=package_id, model_path=model_path))
        training_results.append(
            SimpleNamespace(
                model_package_id=package_id,
                model_path=model_path,
                artifact_directory=run,
            )
        )
        evaluations.append(
            SimpleNamespace(
                model_package_id=package_id,
                evaluation_id=evaluation_id,
                prediction_id=prediction_id,
                artifact_directory=evaluation,
                private_prediction_directory=prediction,
            )
        )

    summaries = tuple(
        SimpleNamespace(directory=_marker(report_root / "rsna/seed-summaries" / family))
        for family in ("cxr_densenet", "cxr_metadata_concat")
    )
    localization = _marker(report_root / "rsna/localization/bounded-localization")
    _marker(private_root / "localization/bounded-localization")
    comparison = SimpleNamespace(
        directory=_marker(report_root / "rsna/comparisons/comparison"),
        manifest={"evaluation_ids": list(evaluation_ids)},
    )
    audit = _marker(report_root / "rsna/audit" / bundle_id)
    execution_manifest = {
        "dataset": {"dataset_id": "rsna", "bundle_id": bundle_id},
        "task": {
            "task_id": "pneumonia",
            "label_policy_version": "rsna-stage-2-target-v1",
        },
    }

    def execution_for(directory: str | Path, **_: object) -> object:
        return SimpleNamespace(
            execution_id=execution_id,
            directory=Path(directory),
            manifest=execution_manifest,
        )

    def freeze_for(
        directory: str | Path,
        *,
        model_root: str | Path,
        bundle_directory: str | Path,
    ) -> _BoundedFreeze:
        return _BoundedFreeze(
            directory=Path(directory),
            model_root=Path(model_root),
            bundle_directory=Path(bundle_directory),
            packages=tuple(
                SimpleNamespace(
                    model_package_id=package_id,
                    model_path=Path(model_root) / "packages" / package_id / "model.skops",
                )
                for package_id in package_ids
            ),
        )

    def restored_training(
        frozen: _BoundedFreeze,
        *,
        report_root: Path,
        **_: object,
    ) -> tuple[object, ...]:
        return tuple(
            SimpleNamespace(
                model_package_id=item.model_package_id,
                model_path=item.model_path,
                artifact_directory=report_root / "rsna/runs" / item.model_package_id,
            )
            for item in frozen.packages
        )

    def restored_evaluation(
        _execution: object,
        *,
        package_id: str,
        report_root: Path,
        private_root: Path,
        **_: object,
    ) -> object:
        index = package_ids.index(package_id)
        return SimpleNamespace(
            model_package_id=package_id,
            evaluation_id=evaluation_ids[index],
            prediction_id=prediction_ids[index],
            artifact_directory=report_root / "rsna/evaluations" / evaluation_ids[index],
            private_prediction_directory=(
                private_root / "predictions/rsna" / prediction_ids[index]
            ),
        )

    def summary_for(ids: list[str], *, output_directory: Path, **_: object) -> object:
        family = families[package_ids[evaluation_ids.index(ids[0])]]
        return SimpleNamespace(directory=output_directory / "rsna/seed-summaries" / family)

    def audit_for(_manifest_root: Path, output_root: Path, *, bundle_id: str) -> None:
        _marker(output_root / bundle_id)

    monkeypatch.setattr(rsna_preservation, "validate_execution", execution_for)
    monkeypatch.setattr(rsna_preservation, "validate_bundle_reference", lambda *a, **k: None)
    monkeypatch.setattr(
        rsna_preservation,
        "validate_bundle_directory",
        lambda *a, **k: {
            "tasks": {"pneumonia": {"label_policy_version": "rsna-stage-2-target-v1"}}
        },
    )
    monkeypatch.setattr(rsna_preservation, "validate_freeze", freeze_for)
    monkeypatch.setattr(rsna_preservation, "validate_execution_closure", lambda *a, **k: None)
    monkeypatch.setattr(rsna_preservation, "_restored_training_derivatives", restored_training)
    monkeypatch.setattr(
        rsna_preservation,
        "_load_evaluation_record_from_established_context",
        restored_evaluation,
    )
    monkeypatch.setattr(
        rsna_preservation,
        "validate_rsna_model_package",
        lambda _root, package_id: {"family_id": families[package_id]},
    )
    monkeypatch.setattr(rsna_preservation, "publish_seed_summary", summary_for)
    monkeypatch.setattr(
        rsna_preservation,
        "validate_localization_report",
        lambda directory, **kwargs: Path(directory),
    )
    monkeypatch.setattr(
        rsna_preservation,
        "regenerate_comparison",
        lambda ids, *, output_directory, **kwargs: SimpleNamespace(
            directory=Path(output_directory) / "rsna/comparisons/comparison",
            manifest={"evaluation_ids": list(ids)},
        ),
    )
    monkeypatch.setattr("beyondcxr.data.rsna_audit._generate_rsna_audit", audit_for)

    execution = execution_for(execution_directory)
    frozen = freeze_for(
        freeze_directory,
        model_root=model_root,
        bundle_directory=bundle,
    )
    with campaign_session(
        repository_root=root, report_root=report_root / "rsna", study="rsna", operation="campaign"
    ) as session:
        session.bind_execution(execution_id)
        members = campaign_export_members(
            execution=execution,
            freeze=frozen,
            training_results=tuple(training_results),
            evaluations=tuple(evaluations),
            summaries=summaries,
            localization=localization,
            comparison=comparison,
            audit_directory=audit,
            private_root=private_root,
            report_root=report_root,
        )
        log_event(get_operational_logger("fixture"), "campaign_ready_for_export")
        operational = campaign_log_members(
            session.log_path, study="rsna", execution_id=execution_id
        )
        # Freeze the live boundary for the restoration tests, before session exit appends success.
        snapshot = root / "campaign-snapshot"
        snapshot.mkdir()
        for member in operational:
            path = snapshot / member.path.name
            shutil.copyfile(member.path, path)
            members += (PreservationMember(path, member.restore_relative, member.role),)
    return members


def test_prepared_closure_preserves_derivatives_and_exact_operational_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    members = _bounded_closure(tmp_path, monkeypatch)
    reports = [
        member for member in members if member.restore_relative.is_relative_to("reports/rsna/runs")
    ]
    assert len(reports) == 2
    assert all(member.role == "derivative" for member in reports)
    packages = [
        member
        for member in members
        if member.restore_relative.is_relative_to("models/rsna/packages")
    ]
    assert len(packages) == 2
    assert all(member.role == "scientific_authority" for member in packages)
    log = next(member.path for member in members if member.path.name == "execution.log")
    records = [parse_operational_record(line) for line in log.read_text().splitlines()]
    assert [(record["event"], record.get("phase")) for record in records] == [
        ("campaign_started", "preparation"),
        ("campaign_execution_bound", None),
        ("phase_started", "audit_generation"),
        ("phase_completed", "audit_generation"),
        ("campaign_ready_for_export", None),
    ]


def _archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    omit_active_receipt: bool = False,
    mismatch_active_receipt: bool = False,
    add_foreign_member: bool = False,
) -> Path:
    live = tmp_path / "live"
    members = list(_bounded_closure(live, monkeypatch))
    if omit_active_receipt:
        members = [
            member
            for member in members
            if member.restore_relative != Path("private/control/rsna/active-execution.json")
        ]
    if mismatch_active_receipt:
        active = live / "private/control/rsna/active-execution.json"
        active.write_bytes(
            canonical_json_bytes(
                {"schema_version": 1, "execution_id": "rsna-execution-" + "9" * 64}
            )
        )
    if add_foreign_member:
        foreign = _marker(live / "foreign")
        members.append(PreservationMember(foreign, Path("foreign")))
    return export_and_verify(
        members=tuple(members),
        export_root=tmp_path / "export",
        backup_root=tmp_path / "backup",
        export_name="bounded-rsna-closure",
    )


def test_rsna_restoration_composes_domain_validators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive(tmp_path, monkeypatch)

    restore_and_validate_export(
        archive,
        restoration_validator=validate_restored_rsna_campaign,
    )


def test_restoration_passes_bundle_cohort_and_frozen_cxr_members_to_localization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive(tmp_path, monkeypatch)
    calls = []

    def reject_localization(directory: Path, **kwargs: object) -> Path:
        calls.append(directory)
        assert kwargs["expected_positive_sample_ids"] == ("rsna:positive",)
        assert kwargs["expected_model_package_ids"] == ("model-package-" + "c" * 64,)
        raise ValueError("Localization evidence rejected")

    monkeypatch.setattr(rsna_preservation, "validate_localization_report", reject_localization)
    with pytest.raises(ValueError, match="Localization evidence rejected"):
        restore_and_validate_export(
            archive,
            restoration_validator=validate_restored_rsna_campaign,
        )
    assert len(calls) == 1


def test_real_rsna_restoration_validator_rejects_missing_required_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive(tmp_path, monkeypatch, omit_active_receipt=True)

    with pytest.raises(ManifestBuildError, match="active execution receipt"):
        restore_and_validate_export(
            archive,
            restoration_validator=validate_restored_rsna_campaign,
        )


def test_real_rsna_restoration_validator_rejects_foreign_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive(tmp_path, monkeypatch, add_foreign_member=True)

    with pytest.raises(ManifestBuildError, match="outside its exact closure"):
        restore_and_validate_export(
            archive,
            restoration_validator=validate_restored_rsna_campaign,
        )


def test_real_rsna_restoration_validator_rejects_wrong_active_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive(tmp_path, monkeypatch, mismatch_active_receipt=True)

    with pytest.raises(ManifestBuildError, match="active execution receipt is invalid"):
        restore_and_validate_export(
            archive,
            restoration_validator=validate_restored_rsna_campaign,
        )
