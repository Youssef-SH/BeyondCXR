from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from radfusion.data.rsna_audit import (
    REPORT_FILENAMES,
    _age_distribution,
    _bbox_statistics,
    _label_distribution,
    generate_rsna_audit,
)
from radfusion.data.rsna_schemas import RSNA_LABEL_SCHEMA, RSNA_SAMPLE_SCHEMA, RSNA_SPLIT_SCHEMA
from radfusion.data.rsna_splitting import SplitConfig, split_assignment_id


def _audit_bundle(tmp_path: Path) -> SimpleNamespace:
    samples = []
    labels = []
    splits = []
    annotations = []
    for split_name, split_index in (("train", 0), ("validation", 1), ("test", 2)):
        for target in (0, 1):
            name = f"{split_name}-{target}"
            sample_id = f"rsna:{name}"
            samples.append(
                {
                    "sample_id": sample_id,
                    "patient_id": name,
                    "image_id": name,
                    "image_path": f"stage_2_train_images/{name}.dcm",
                    "image_rows": 1024,
                    "image_columns": 1024,
                    "age_years": float(40 + split_index),
                    "age_is_implausible": False,
                    "sex": "F" if target else "M",
                    "view_position": "PA",
                    "pixel_spacing_row_mm": 0.168,
                    "pixel_spacing_col_mm": 0.168,
                }
            )
            labels.append(
                {
                    "sample_id": sample_id,
                    "task_id": "pneumonia",
                    "label_value": target,
                }
            )
            splits.append({"sample_id": sample_id, "split_name": split_name})
            if target:
                annotations.append({"sample_id": sample_id})
    directory = tmp_path / "bundle"
    directory.mkdir()
    paths = {
        "samples_path": directory / "samples.parquet",
        "labels_path": directory / "labels.parquet",
        "annotations_path": directory / "annotations.parquet",
        "splits_path": directory / "splits.parquet",
        "metadata_path": directory / "metadata.json",
    }
    sample_table = pa.Table.from_pylist(
        sorted(samples, key=lambda row: row["sample_id"]), RSNA_SAMPLE_SCHEMA
    )
    label_table = pa.Table.from_pylist(
        sorted(labels, key=lambda row: row["sample_id"]), RSNA_LABEL_SCHEMA
    )
    pq.write_table(sample_table, paths["samples_path"])
    pq.write_table(label_table, paths["labels_path"])
    pq.write_table(pa.Table.from_pylist(annotations), paths["annotations_path"])
    recipe_id = SplitConfig().recipe_id
    assignments = {row["sample_id"]: row["split_name"] for row in splits}
    assignment_id = split_assignment_id(assignments)
    pq.write_table(
        pa.Table.from_pylist(sorted(splits, key=lambda row: row["sample_id"]), RSNA_SPLIT_SCHEMA),
        paths["splits_path"],
    )
    paths["metadata_path"].write_text(
        json.dumps(
            {
                "membership": {
                    "split": {
                        "split_recipe_id": recipe_id,
                        "split_assignment_id": assignment_id,
                    },
                },
                "qualification": {"age_parsing": {"implausible_age_count": 0}},
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(bundle_id="bundle-test", **paths)


def test_label_distribution_uses_task_specific_names() -> None:
    labels = pd.DataFrame(
        [
            {"sample_id": "rsna:a", "task_id": "pneumonia", "label_value": 1},
            {"sample_id": "rsna:a", "task_id": "rsna_class", "label_value": 1},
        ]
    )
    splits = pd.DataFrame([{"sample_id": "rsna:a", "split_name": "train"}])

    report = _label_distribution(labels, splits)
    names = dict(zip(report["task_id"], report["label_name"], strict=False))
    assert names["pneumonia"] == "positive"
    assert names["rsna_class"] == "No Lung Opacity / Not Normal"
    assert set(REPORT_FILENAMES) == {
        "dataset_summary.md",
        "split_summary.md",
        "label_distribution.csv",
        "age_distribution.csv",
        "sex_distribution.csv",
        "view_distribution.csv",
        "image_dimensions.csv",
        "pixel_spacing.csv",
        "bbox_statistics.csv",
        "missingness_report.md",
    }


def test_age_distribution_matches_the_strictly_greater_than_120_policy() -> None:
    frame = pd.DataFrame(
        {
            "split_name": ["train", "train", "validation", "test"],
            "age_years": [119.9, 120.0, 120.1, None],
        }
    )

    report = _age_distribution(frame)
    overall = report.loc[report["scope"] == "overall"].set_index("age_band")["count"]

    assert overall["80-120"] == 2
    assert overall[">120"] == 1
    assert overall["<missing>"] == 1


def test_bbox_statistics_handles_scope_without_positive_samples() -> None:
    frame = pd.DataFrame(
        [
            {"sample_id": "train-positive", "split_name": "train", "target_pneumonia": 1},
            {"sample_id": "train-negative", "split_name": "train", "target_pneumonia": 0},
            {
                "sample_id": "validation-negative",
                "split_name": "validation",
                "target_pneumonia": 0,
            },
            {"sample_id": "test-negative", "split_name": "test", "target_pneumonia": 0},
        ]
    )
    annotations = pd.DataFrame(
        [
            {"sample_id": "train-positive"},
            {"sample_id": "train-positive"},
        ]
    )

    report = _bbox_statistics(frame, annotations)
    validation = report.loc[report["scope"] == "validation"].iloc[0]
    training = report.loc[report["scope"] == "train"].iloc[0]

    assert validation["positive_sample_count"] == 0
    assert validation["annotation_count"] == 0
    assert validation["positive_samples_one_box"] == 0
    assert validation["positive_samples_multiple_boxes"] == 0
    for field in (
        "mean_boxes_per_positive",
        "median_boxes_per_positive",
        "min_boxes_per_positive",
        "max_boxes_per_positive",
    ):
        assert pd.isna(validation[field])
    assert training["positive_sample_count"] == 1
    assert training["annotation_count"] == 2
    assert training["mean_boxes_per_positive"] == 2.0
    assert training["min_boxes_per_positive"] == 2

    first = StringIO()
    second = StringIO()
    report.to_csv(first, index=False, float_format="%.6f", lineterminator="\n")
    report.to_csv(second, index=False, float_format="%.6f", lineterminator="\n")
    assert first.getvalue() == second.getvalue()


def test_audit_publication_owns_only_the_bundle_specific_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _audit_bundle(tmp_path)
    monkeypatch.setattr("radfusion.data.rsna_audit.load_current_bundle", lambda _: bundle)
    output = tmp_path / "reports" / "rsna"
    other_bundle = output / "bundle-other"
    other_bundle.mkdir(parents=True)
    (other_bundle / "keep.txt").write_text("other", encoding="utf-8")

    generate_rsna_audit(tmp_path / "manifests", output)

    destination = output / bundle.bundle_id
    assert {path.name for path in destination.iterdir()} == set(REPORT_FILENAMES)
    assert (other_bundle / "keep.txt").read_text(encoding="utf-8") == "other"
    assert not list(output.parent.glob(".rsna-staging-*"))
    assert not list(output.parent.glob(".rsna-backup-*"))


def test_audit_failure_preserves_previous_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _audit_bundle(tmp_path)
    monkeypatch.setattr("radfusion.data.rsna_audit.load_current_bundle", lambda _: bundle)
    monkeypatch.setattr(
        "radfusion.data.rsna_audit._missingness_report",
        lambda _: (_ for _ in ()).throw(RuntimeError("report failed")),
    )
    output = tmp_path / "reports" / "rsna"
    destination = output / bundle.bundle_id
    destination.mkdir(parents=True)
    (destination / "previous.txt").write_text("complete", encoding="utf-8")

    with pytest.raises(RuntimeError):
        generate_rsna_audit(tmp_path / "manifests", output)

    assert (destination / "previous.txt").read_text(encoding="utf-8") == "complete"
    assert not list(output.glob(".bundle-test-staging-*"))
    assert not list(output.glob(".bundle-test-backup-*"))
