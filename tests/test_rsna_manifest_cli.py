from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pytest
from rsna_manifest_test_support import write_sources as _write_sources

from radfusion.data.rsna_artifacts import (
    CURRENT_FILENAME,
    SOURCE_INVENTORY_FILENAME,
    BuildResult,
    build_rsna_artifacts,
    resolve_bundle,
    write_bundle,
)
from radfusion.data.rsna_audit import REPORT_FILENAMES, generate_rsna_audit
from radfusion.data.rsna_manifest import main
from radfusion.data.rsna_schemas import (
    PNEUMONIA_TASK_ID,
)
from radfusion.utils.privacy import validate_public_reports


def test_cli_success_and_failure_exit_codes(tmp_path: Path) -> None:
    root = _write_sources(tmp_path / "extracted")
    output = tmp_path / "manifests"

    assert main(["--source-root", str(root), "--output-directory", str(output)]) == 0
    current = resolve_bundle(output)
    assert current.current_path.name == CURRENT_FILENAME
    assert current.samples_path.is_file()
    assert current.labels_path.is_file()
    assert current.splits_path.is_file()
    assert current.source_inventory_path.is_file()
    metadata = json.loads(current.metadata_path.read_text(encoding="utf-8"))
    assert metadata["qualification"]["counts"]["samples"] == 2
    assert metadata["qualification"]["counts"]["labels"] == 4
    assert metadata["bundle_manifest_schema_version"] == 1
    split = metadata["membership"]["split"]
    assert split["patient_hash_algorithm"] == "sha256"
    assert split["patient_hash_input_encoding"] == "utf-8"
    assert split["patient_hash_input_template"] == "<seed>\\0<patient_id>"
    assert split["stratification_target"] == "pneumonia"
    assert split["allocation_rule"] == "feasible-minimum-then-largest-remainder-canonical-tiebreak"
    assert split["seed"] == 42
    assert split["ratios"] == [
        {"split_name": "train", "ratio": 0.7},
        {"split_name": "validation", "ratio": 0.15},
        {"split_name": "test", "ratio": 0.15},
    ]
    assert split["split_recipe_id"].startswith("split-recipe-")
    assert split["split_assignment_id"].startswith("split-assignment-")
    assert set(split) == {
        "split_source",
        "split_recipe_id",
        "split_assignment_id",
        "algorithm_version",
        "patient_grouping_rule",
        "patient_target_consistency_rule",
        "ranking_rule",
        "patient_hash_algorithm",
        "patient_hash_input_encoding",
        "patient_hash_input_template",
        "seed",
        "stratification_target",
        "allocation_rule",
        "split_order",
        "ratios",
    }
    assert metadata["qualification"]["counts"]["source_inventory"] == 2
    assert SOURCE_INVENTORY_FILENAME in metadata["artifacts"]
    assert metadata["provenance"]["logical_arrow_runtime"]["pyarrow_version"] == pa.__version__
    assert (
        "cross-version stability is not claimed"
        in metadata["provenance"]["logical_arrow_runtime"]["stability_scope"]
    )
    assert main(["--source-root", str(tmp_path / "missing")]) == 1


@pytest.fixture(scope="session")
def real_rsna_build() -> BuildResult:
    root = Path("data/raw/rsna/extracted")
    if not (root / "stage_2_train_images").is_dir():
        pytest.skip("Local RSNA data is not available")
    return build_rsna_artifacts(root)


@pytest.mark.integration
def test_real_rsna_aggregate_contract(real_rsna_build: BuildResult) -> None:
    assert real_rsna_build.samples.num_rows == 26_684
    assert real_rsna_build.labels.num_rows == 53_368
    assert real_rsna_build.annotations.num_rows == 9_555
    targets = Counter(
        row["label_value"]
        for row in real_rsna_build.labels.to_pylist()
        if row["task_id"] == PNEUMONIA_TASK_ID
    )
    assert targets == {0: 20_672, 1: 6_012}


@pytest.mark.integration
def test_real_rsna_audit_reports_are_aggregate(
    tmp_path: Path, real_rsna_build: BuildResult
) -> None:
    output = tmp_path / "manifests"
    written = write_bundle(real_rsna_build, output)

    report_directory = tmp_path / "reports" / "rsna"
    summary = generate_rsna_audit(output, report_directory)

    audit_directory = report_directory / written.paths.bundle_id
    assert set(path.name for path in audit_directory.iterdir()) == set(REPORT_FILENAMES)
    assert summary["reports"] == list(REPORT_FILENAMES)
    sample_rows = real_rsna_build.samples.slice(0, 10).to_pylist()
    source_values = {
        str(row[field])
        for row in sample_rows
        for field in ("patient_id", "sample_id", "image_id", "image_path")
    }
    source_values.update(Path(row["image_path"]).name for row in sample_rows)
    source_values.update(Path(row["image_path"]).stem for row in sample_rows)
    validate_public_reports(audit_directory.iterdir(), forbidden_source_values=source_values)
