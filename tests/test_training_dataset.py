from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pytest

from radfusion.data.tabular_preprocess import SOURCE_FEATURES
from radfusion.training.config import ConfigError, load_experiment_config
from radfusion.training.datasets import RsnaDataset


def _tables() -> dict[str, pa.Table]:
    samples = []
    labels = []
    splits = []
    for split in ("train", "validation", "test"):
        for index, target in enumerate((0, 1)):
            sample_id = f"rsna:{split}-{index}"
            samples.append(
                {
                    "sample_id": sample_id,
                    "patient_id": f"{split}-{index}",
                    "image_id": f"image-{split}-{index}",
                    "image_path": f"images/{split}-{index}.dcm",
                    "image_rows": 1024,
                    "image_columns": 1024,
                    "age_years": 40.0 + index,
                    "age_is_implausible": False,
                    "sex": "F" if index == 0 else "M",
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
            splits.append({"sample_id": sample_id, "split_name": split})
    return {
        "rsna_samples.parquet": pa.Table.from_pylist(samples),
        "rsna_labels.parquet": pa.Table.from_pylist(labels),
        "rsna_splits.parquet": pa.Table.from_pylist(splits),
    }


def test_dataset_adapter_loads_exact_bundle_and_exposes_only_approved_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_experiment_config("configs/metadata_logistic.yaml").dataset
    config = replace(
        config,
        manifest_directory=tmp_path / "manifests",
        bundle_id="build-pinned",
    )
    tables = _tables()
    validated: list[Path] = []
    reads: list[tuple[str, tuple[str, ...], tuple[tuple[str, str, object], ...]]] = []

    def validate(path, *, expected_bundle_id):
        validated.append(Path(path))
        assert expected_bundle_id == "build-pinned"
        return {
            "split": {"split_assignment_id": "assignment"},
            "tasks": {"pneumonia": {"label_policy_version": "label-v1"}},
        }

    monkeypatch.setattr("radfusion.training.datasets.validate_bundle_directory", validate)

    def read_table(path, *, columns, filters):
        filename = Path(path).name
        normalized_filters = tuple(tuple(item) for item in filters)
        reads.append((filename, tuple(columns), normalized_filters))
        rows = tables[filename].to_pylist()
        for field, operator, value in filters:
            if operator == "=":
                rows = [row for row in rows if row[field] == value]
            elif operator == "in":
                rows = [row for row in rows if row[field] in value]
            else:
                raise AssertionError(f"Unexpected filter operator: {operator}")
        return pa.Table.from_pylist([{column: row[column] for column in columns} for row in rows])

    monkeypatch.setattr("radfusion.training.datasets.pq.read_table", read_table)

    data = RsnaDataset().load_train_validation(config)
    test, lineage = RsnaDataset().load_test(config)

    expected_bundle = tmp_path / "manifests" / "rsna" / "builds" / "build-pinned"
    assert validated == [expected_bundle, expected_bundle]
    assert tuple(data.train.features.columns) == SOURCE_FEATURES
    assert tuple(data.validation.features.columns) == SOURCE_FEATURES
    assert not {
        "sample_id",
        "patient_id",
        "image_id",
        "image_path",
        "target",
        "split_name",
        "bundle_id",
    } & set(data.train.features)
    assert data.lineage.bundle_id == "build-pinned"
    assert lineage == data.lineage
    assert not hasattr(data, "test")
    assert test.sample_ids == ("rsna:test-0", "rsna:test-1")
    assert reads[0] == (
        "rsna_splits.parquet",
        ("sample_id", "split_name"),
        (("split_name", "in", ["train", "validation"]),),
    )
    assert reads[1][0:2] == (
        "rsna_samples.parquet",
        ("sample_id", "patient_id", *SOURCE_FEATURES),
    )
    assert reads[1][2] == (
        (
            "sample_id",
            "in",
            [
                "rsna:train-0",
                "rsna:train-1",
                "rsna:validation-0",
                "rsna:validation-1",
            ],
        ),
    )
    assert reads[2] == (
        "rsna_labels.parquet",
        ("sample_id", "label_value"),
        (
            ("task_id", "=", "pneumonia"),
            (
                "sample_id",
                "in",
                [
                    "rsna:train-0",
                    "rsna:train-1",
                    "rsna:validation-0",
                    "rsna:validation-1",
                ],
            ),
        ),
    )
    assert reads[3][2] == (("split_name", "in", ["test"]),)
    assert reads[4][2] == (("sample_id", "in", ["rsna:test-0", "rsna:test-1"]),)
    assert reads[5][2] == (
        ("task_id", "=", "pneumonia"),
        ("sample_id", "in", ["rsna:test-0", "rsna:test-1"]),
    )


def test_config_cannot_omit_bundle_pin(tmp_path: Path) -> None:
    text = Path("configs/metadata_logistic.yaml").read_text(encoding="utf-8")
    line = next(item for item in text.splitlines() if item.strip().startswith("bundle_id:"))
    path = tmp_path / "unpinned.yaml"
    path.write_text(text.replace(line + "\n", ""), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_experiment_config(path)
