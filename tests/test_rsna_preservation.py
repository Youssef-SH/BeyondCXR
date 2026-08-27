from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
import zipfile
from pathlib import Path

import pytest
from rsna_preservation_test_support import build_real_rsna_campaign_closure

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.training.preservation import (
    EXPORT_MANIFEST_FILENAME,
    PreservationMember,
    export_and_verify,
    restore_and_validate_export,
)
from beyondcxr.training.rsna_campaign import _validate_outputs
from beyondcxr.training.rsna_compare import (
    COMPARISON_COLUMNS,
    COMPARISON_POLICY_VERSION,
    COMPARISON_PREFIX,
)
from beyondcxr.training.rsna_localize import _markdown as localization_markdown
from beyondcxr.training.rsna_localize import _report_document as localization_report_document
from beyondcxr.training.rsna_preservation import (
    validate_restored_rsna_campaign,
)
from beyondcxr.training.rsna_training_report import (
    REQUIRED_REPORT_FILENAMES,
    _write_evaluation_report,
)
from beyondcxr.utils.package_identity import canonical_scientific_id


def _directory(path: Path, filename: str = "manifest.json") -> Path:
    path.mkdir(parents=True)
    (path / filename).write_text("{}\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def real_rsna_archive(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("real-rsna-preservation")
    closure = build_real_rsna_campaign_closure(root / "workspace")
    unrelated = _directory(root / "workspace/reports/rsna/campaigns/unrelated")
    validations = []

    def validate(restored: Path):
        validations.append(restored)
        assert not (restored / "reports/rsna/campaigns/unrelated").exists()
        assert not (restored / "mlflow.db").exists()
        assert not (restored / "mlartifacts").exists()
        assert not (restored / "data/manifests/rsna/CURRENT").exists()
        assert not (restored / "raw").exists()
        return validate_restored_rsna_campaign(restored)

    archive = export_and_verify(
        members=closure.members,
        export_root=root / "export",
        backup_root=root / "backup",
        export_name="rsna-real-closure",
        restoration_validator=validate,
    )
    return closure, archive, validations, unrelated


def test_rsna_export_restores_and_recursively_validates_the_real_scientific_closure(
    real_rsna_archive,
) -> None:
    closure, archive, validations, unrelated = real_rsna_archive
    assert len(validations) == 2
    assert archive.is_file()
    assert unrelated.is_dir()
    assert tuple((spec.family_id, spec.seed) for spec in closure.plan.runs) == (
        ("metadata_logistic", 42),
        ("metadata_lightgbm", 42),
        ("cxr_densenet", 17),
        ("cxr_densenet", 42),
        ("cxr_densenet", 2026),
        ("cxr_metadata_concat", 17),
        ("cxr_metadata_concat", 42),
        ("cxr_metadata_concat", 2026),
    )


def test_completed_scientific_closure_does_not_require_or_create_mlflow(
    real_rsna_archive,
) -> None:
    closure, _, _, _ = real_rsna_archive
    database = closure.plan.roots.tracking_database
    assert not database.exists()
    _validate_outputs(
        closure.training_results,
        closure.evaluations,
        closure.audit_directory,
        closure.summaries[0].directory,
        closure.summaries[1].directory,
        closure.localization,
        closure.comparison,
        closure.campaign_log,
        roots=closure.plan.roots,
    )
    assert not database.exists()


@pytest.mark.parametrize(
    "tamper",
    (
        "package-report-lineage",
        "training-metric",
        "training-markdown",
        "training-plot",
        "comparison-membership",
    ),
)
def test_rsna_restoration_rejects_archive_valid_scientific_tampering(
    tmp_path: Path,
    real_rsna_archive,
    tamper: str,
) -> None:
    _, archive, _, _ = real_rsna_archive
    plot_selected = False

    def mutate(path: str, content: bytes) -> bytes | None:
        nonlocal plot_selected
        if tamper == "package-report-lineage" and path.endswith("/metrics.json"):
            document = json.loads(content)
            if document.get("training_package", {}).get("family_id") != "metadata_logistic":
                return None
            document["training_package"]["model_package_id"] = "model-package-" + "0" * 64
            return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
        if tamper == "training-metric" and path.endswith("/metrics.json"):
            document = json.loads(content)
            if document.get("training_package", {}).get("family_id") != "metadata_logistic":
                return None
            document["probability_metrics"]["brier_score"] = 0.999
            return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
        if (
            tamper == "training-markdown"
            and path.endswith("/evaluation_report.md")
            and content.startswith(b"# metadata_logistic validation evaluation")
        ):
            return content + b"tampered report\n"
        if tamper == "training-plot" and path.endswith("/roc_curve.png") and not plot_selected:
            plot_selected = True
            return content + b"tampered plot"
        if tamper == "comparison-membership" and path.endswith("/manifest.json"):
            document = json.loads(content)
            if "comparison_schema_version" not in document:
                return None
            document["evaluation_ids"].reverse()
            return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
        return None

    corrupted = _rewrite_archive_with_valid_hashes(archive, tmp_path / f"{tamper}.zip", mutate)
    restore_and_validate_export(corrupted)
    with pytest.raises((ManifestBuildError, ValueError)):
        restore_and_validate_export(
            corrupted,
            restoration_validator=validate_restored_rsna_campaign,
        )


@pytest.mark.parametrize(
    ("relative_path", "content"),
    (("junk.json", b"{}\n"), ("evaluations/unexpected-package.json", b"{}\n")),
)
def test_rsna_restoration_rejects_extra_execution_control_members(
    tmp_path: Path,
    real_rsna_archive,
    relative_path: str,
    content: bytes,
) -> None:
    _, archive, _, _ = real_rsna_archive
    corrupted = _add_archive_member_with_valid_hashes(
        archive,
        tmp_path / (relative_path.replace("/", "-") + ".zip"),
        restore_prefix="private/control/rsna/executions/",
        relative_path=relative_path,
        content=content,
    )
    restore_and_validate_export(corrupted)
    with pytest.raises(ManifestBuildError, match="membership"):
        restore_and_validate_export(
            corrupted,
            restoration_validator=validate_restored_rsna_campaign,
        )


def test_rsna_restoration_rejects_coherent_duplicate_comparison_membership(
    tmp_path: Path,
    real_rsna_archive,
) -> None:
    _, archive, _, _ = real_rsna_archive
    corrupted = _rewrite_comparison_with_duplicate(
        archive,
        tmp_path / "duplicate-comparison-membership.zip",
    )

    restore_and_validate_export(corrupted)
    with pytest.raises((ManifestBuildError, ValueError)):
        restore_and_validate_export(
            corrupted,
            restoration_validator=validate_restored_rsna_campaign,
        )


def test_rsna_restoration_rejects_coherent_report_freeze_and_archive_rewrite(
    tmp_path: Path,
    real_rsna_archive,
) -> None:
    _, archive, _, _ = real_rsna_archive

    def mutate(manifest, contents):
        report_member = next(
            member
            for member in manifest["members"]
            if member["restore_relative"].startswith("reports/rsna/runs/")
            and json.loads(contents[f"{member['archive_identity']}/metrics.json"])[
                "training_package"
            ]["family_id"]
            == "metadata_logistic"
        )
        identity = report_member["archive_identity"]
        metrics_name = f"{identity}/metrics.json"
        document = json.loads(contents[metrics_name])
        document["probability_metrics"]["brier_score"] = 0.5
        contents[metrics_name] = (
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        with tempfile.TemporaryDirectory() as temporary:
            rendered = Path(temporary) / "evaluation_report.md"
            _write_evaluation_report(rendered, "metadata_logistic", document)
            contents[f"{identity}/evaluation_report.md"] = rendered.read_bytes()
        report_digest = hashlib.sha256()
        files = {item["path"] for item in report_member["files"]}
        assert files == set(REQUIRED_REPORT_FILENAMES)
        for filename in sorted(files):
            name = filename.encode()
            content = contents[f"{identity}/{filename}"]
            report_digest.update(len(name).to_bytes(4, "big"))
            report_digest.update(name)
            report_digest.update(len(content).to_bytes(8, "big"))
            report_digest.update(content)
        control_member = next(
            member
            for member in manifest["members"]
            if member["restore_relative"].startswith("private/control/rsna/executions/")
        )
        freeze_name = f"{control_member['archive_identity']}/package-freeze.json"
        freeze = json.loads(contents[freeze_name])
        report_relative = report_member["restore_relative"].removeprefix("reports/")
        frozen_report = next(
            item for item in freeze["packages"] if item["report_relative"] == report_relative
        )
        frozen_report["report_sha256"] = report_digest.hexdigest()
        contents[freeze_name] = (
            json.dumps(freeze, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()

    corrupted = _rewrite_archive_members_with_valid_hashes(
        archive, tmp_path / "coherent-report-rewrite.zip", mutate
    )
    restore_and_validate_export(corrupted)
    with pytest.raises(ValueError, match="scientific claims differ from validation evidence"):
        restore_and_validate_export(
            corrupted,
            restoration_validator=validate_restored_rsna_campaign,
        )


def test_rsna_restoration_rejects_coherent_localization_summary_rewrite(
    tmp_path: Path,
    real_rsna_archive,
) -> None:
    _, archive, _, _ = real_rsna_archive

    def mutate(manifest, contents):
        member = next(
            item
            for item in manifest["members"]
            if item["restore_relative"].startswith("reports/rsna/localization/")
        )
        identity = member["archive_identity"]
        summary_name = f"{identity}/summary.json"
        document = json.loads(contents[summary_name])
        document["members"][0]["pointing_game_accuracy"] = 0.25
        document["aggregates"] = localization_report_document(
            document["report_id"], document["model_package_ids"], document["members"]
        )["aggregates"]
        contents[summary_name] = (
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        contents[f"{identity}/summary.md"] = localization_markdown(document).encode()

    corrupted = _rewrite_archive_members_with_valid_hashes(
        archive, tmp_path / "coherent-localization-rewrite.zip", mutate
    )
    restore_and_validate_export(corrupted)
    with pytest.raises(ValueError, match="differ from localization evidence"):
        restore_and_validate_export(
            corrupted,
            restoration_validator=validate_restored_rsna_campaign,
        )


def test_rsna_restoration_rejects_coherent_localization_cohort_omission(
    tmp_path: Path,
    real_rsna_archive,
) -> None:
    _, archive, _, _ = real_rsna_archive

    def mutate(manifest, contents):
        private_member = next(
            item
            for item in manifest["members"]
            if item["restore_relative"].startswith("private/localization/")
        )
        evidence_name = f"{private_member['archive_identity']}/localization-evidence.json"
        evidence = json.loads(contents[evidence_name])
        for member in evidence["members"]:
            member["cases"].pop()
        contents[evidence_name] = (
            json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()

        public_member = next(
            item
            for item in manifest["members"]
            if item["restore_relative"].startswith("reports/rsna/localization/")
        )
        identity = public_member["archive_identity"]
        summary_name = f"{identity}/summary.json"
        document = json.loads(contents[summary_name])
        document["members"] = [
            {
                "seed": member["seed"],
                "positive_test_sample_count": len(member["cases"]),
                "localization_evaluated_count": len(member["cases"]),
                "zero_heatmap_count": sum(case["zero_heatmap"] for case in member["cases"]),
                "pointing_game_accuracy": statistics.fmean(
                    case["pointing_game"] for case in member["cases"]
                ),
                "mean_activation_energy_inside_union": statistics.fmean(
                    case["activation_energy_inside_union"] for case in member["cases"]
                ),
                "qualitative_strata_present": member["qualitative_strata_present"],
            }
            for member in evidence["members"]
        ]
        document["aggregates"] = localization_report_document(
            document["report_id"], document["model_package_ids"], document["members"]
        )["aggregates"]
        contents[summary_name] = (
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        contents[f"{identity}/summary.md"] = localization_markdown(document).encode()

    corrupted = _rewrite_archive_members_with_valid_hashes(
        archive, tmp_path / "coherent-localization-cohort-omission.zip", mutate
    )
    restore_and_validate_export(corrupted)
    with pytest.raises(ValueError, match="canonical positive held-out cohort"):
        restore_and_validate_export(
            corrupted,
            restoration_validator=validate_restored_rsna_campaign,
        )


def test_rsna_restoration_rejects_substituted_localization_case_id(
    tmp_path: Path,
    real_rsna_archive,
) -> None:
    _, archive, _, _ = real_rsna_archive

    def mutate(manifest, contents):
        private_member = next(
            item
            for item in manifest["members"]
            if item["restore_relative"].startswith("private/localization/")
        )
        evidence_name = f"{private_member['archive_identity']}/localization-evidence.json"
        evidence = json.loads(contents[evidence_name])
        evidence["members"][0]["cases"][0]["sample_id"] = "rsna:" + "-".join(
            ("00000000", "0000", "4000", "8000", "000000000000")
        )
        contents[evidence_name] = (
            json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()

    corrupted = _rewrite_archive_members_with_valid_hashes(
        archive, tmp_path / "substituted-localization-case-id.zip", mutate
    )
    restore_and_validate_export(corrupted)
    with pytest.raises(ValueError, match="canonical positive held-out cohort"):
        restore_and_validate_export(
            corrupted,
            restoration_validator=validate_restored_rsna_campaign,
        )


def test_rsna_restoration_rejects_valid_external_preservation_member(
    tmp_path: Path,
    real_rsna_archive,
) -> None:
    _, archive, _, _ = real_rsna_archive
    corrupted = _add_external_archive_member_with_valid_hashes(
        archive,
        tmp_path / "external-member.zip",
        restore_relative="private/unrelated",
        relative_path="state.json",
        content=b"{}\n",
    )

    restore_and_validate_export(corrupted)
    with pytest.raises(ManifestBuildError, match="outside its exact closure"):
        restore_and_validate_export(
            corrupted,
            restoration_validator=validate_restored_rsna_campaign,
        )


@pytest.mark.parametrize(
    "member_index",
    range(9),
    ids=(
        "bundle",
        "freeze",
        "package-report",
        "evaluation-prediction",
        "seed-summary",
        "localization",
        "comparison",
        "audit",
        "campaign-log",
    ),
)
def test_rsna_preservation_rejects_corruption_across_closure_families(
    tmp_path: Path, member_index: int
) -> None:
    members = []
    for index, restore in enumerate(
        (
            "data/manifests/rsna/bundles/bundle-test",
            "private/control/rsna/executions/execution-test",
            "models/rsna/packages/package-test",
            "private/predictions/rsna/prediction-test",
            "reports/rsna/seed-summaries/summary-test",
            "reports/rsna/localization/localization-test",
            "reports/rsna/comparisons/comparison-test",
            "reports/rsna/audit/bundle-test",
            "reports/rsna/campaigns/campaign-test",
        )
    ):
        source = tmp_path / "sources" / str(index)
        _directory(source)
        members.append(PreservationMember(source, Path(restore)))
    archive = export_and_verify(
        members=members,
        export_root=tmp_path / "outbox",
        backup_root=tmp_path / "backup",
        export_name="campaign",
    )
    corrupted = tmp_path / "corrupted.zip"
    target = f"member-{member_index:04d}/manifest.json"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(corrupted, "w") as output:
        for info in source.infolist():
            content = source.read(info.filename)
            output.writestr(info, b"corrupted" if info.filename == target else content)
    os.chmod(corrupted, 0o600)
    with pytest.raises(ManifestBuildError, match="hash"):
        restore_and_validate_export(corrupted)


def _rewrite_archive_with_valid_hashes(archive: Path, destination: Path, mutate) -> Path:
    selected = []

    def apply(manifest, contents):
        for member in manifest["members"]:
            for file_document in member["files"]:
                name = f"{member['archive_identity']}/{file_document['path']}"
                replacement = mutate(name, contents[name])
                if replacement is None:
                    continue
                selected.append(name)
                contents[name] = replacement

    result = _rewrite_archive_members_with_valid_hashes(archive, destination, apply)
    if len(selected) != 1:
        raise AssertionError(f"Semantic tamper selector matched {len(selected)} archive files")
    return result


def _rewrite_archive_members_with_valid_hashes(archive: Path, destination: Path, mutate) -> Path:
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        contents = {info.filename: source.read(info.filename) for info in infos}
    manifest = json.loads(contents[EXPORT_MANIFEST_FILENAME])
    mutate(manifest, contents)
    for member in manifest["members"]:
        for file_document in member["files"]:
            name = f"{member['archive_identity']}/{file_document['path']}"
            file_document["sha256"] = hashlib.sha256(contents[name]).hexdigest()
    contents[EXPORT_MANIFEST_FILENAME] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    with zipfile.ZipFile(destination, "w") as output:
        for info in infos:
            output.writestr(info, contents[info.filename])
    os.chmod(destination, 0o600)
    return destination


def _add_archive_member_with_valid_hashes(
    archive: Path,
    destination: Path,
    *,
    restore_prefix: str,
    relative_path: str,
    content: bytes,
) -> Path:
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        contents = {info.filename: source.read(info.filename) for info in infos}
    manifest = json.loads(contents[EXPORT_MANIFEST_FILENAME])
    member = next(
        item for item in manifest["members"] if item["restore_relative"].startswith(restore_prefix)
    )
    name = f"{member['archive_identity']}/{relative_path}"
    member["files"].append({"path": relative_path, "sha256": hashlib.sha256(content).hexdigest()})
    member["files"].sort(key=lambda item: item["path"])
    contents[name] = content
    contents[EXPORT_MANIFEST_FILENAME] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    with zipfile.ZipFile(destination, "w") as output:
        for info in infos:
            output.writestr(info, contents[info.filename])
        output.writestr(name, content)
    os.chmod(destination, 0o600)
    return destination


def _add_external_archive_member_with_valid_hashes(
    archive: Path,
    destination: Path,
    *,
    restore_relative: str,
    relative_path: str,
    content: bytes,
) -> Path:
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        contents = {info.filename: source.read(info.filename) for info in infos}
    manifest = json.loads(contents[EXPORT_MANIFEST_FILENAME])
    identity = f"member-{len(manifest['members']):04d}"
    manifest["members"].append(
        {
            "archive_identity": identity,
            "restore_relative": restore_relative,
            "kind": "directory",
            "files": [{"path": relative_path, "sha256": hashlib.sha256(content).hexdigest()}],
        }
    )
    name = f"{identity}/{relative_path}"
    contents[name] = content
    contents[EXPORT_MANIFEST_FILENAME] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    with zipfile.ZipFile(destination, "w") as output:
        for info in infos:
            output.writestr(info, contents[info.filename])
        output.writestr(name, content)
    os.chmod(destination, 0o600)
    return destination


def _rewrite_comparison_with_duplicate(archive: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        contents = {info.filename: source.read(info.filename) for info in infos}
    manifest = json.loads(contents[EXPORT_MANIFEST_FILENAME])
    comparison_member = next(
        member
        for member in manifest["members"]
        if member["restore_relative"].startswith("reports/rsna/comparisons/comparison-")
    )
    identity = comparison_member["archive_identity"]
    comparison_manifest_name = f"{identity}/manifest.json"
    comparison_document = json.loads(contents[comparison_manifest_name])
    duplicate_id = comparison_document["evaluation_ids"][-1]
    comparison_document["evaluation_ids"].append(duplicate_id)
    comparison_document["row_count"] = len(comparison_document["evaluation_ids"])
    comparison_document["comparison_id"] = canonical_scientific_id(
        COMPARISON_PREFIX,
        {
            "comparison_policy_version": COMPARISON_POLICY_VERSION,
            "evaluation_ids": comparison_document["evaluation_ids"],
            "columns": list(COMPARISON_COLUMNS),
        },
    )
    comparison_member["restore_relative"] = (
        f"reports/rsna/comparisons/{comparison_document['comparison_id']}"
    )
    replacements = {
        comparison_manifest_name: (
            json.dumps(comparison_document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode(),
        f"{identity}/table.csv": _duplicate_last_line(contents[f"{identity}/table.csv"]),
        f"{identity}/table.md": _duplicate_last_line(contents[f"{identity}/table.md"]),
    }
    for file_document in comparison_member["files"]:
        name = f"{identity}/{file_document['path']}"
        if name in replacements:
            contents[name] = replacements[name]
            file_document["sha256"] = hashlib.sha256(contents[name]).hexdigest()
    contents[EXPORT_MANIFEST_FILENAME] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    with zipfile.ZipFile(destination, "w") as output:
        for info in infos:
            output.writestr(info, contents[info.filename])
    os.chmod(destination, 0o600)
    return destination


def _duplicate_last_line(content: bytes) -> bytes:
    lines = content.splitlines(keepends=True)
    return b"".join((*lines, lines[-1]))
