from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from beyondcxr.data.rsna_artifacts import BuildResult, build_rsna_artifacts, write_bundle
from beyondcxr.data.rsna_audit import REPORT_FILENAMES, generate_rsna_audit
from beyondcxr.data.rsna_schemas import PNEUMONIA_TASK_ID
from beyondcxr.utils.privacy import validate_public_reports


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
