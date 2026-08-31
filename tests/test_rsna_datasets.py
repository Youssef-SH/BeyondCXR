from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
from rsna_authorization_test_support import authorize_rsna_package

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.rsna_metadata_preprocess import SOURCE_FEATURES
from beyondcxr.training.config import ConfigError, load_experiment_config, with_runtime
from beyondcxr.training.rsna_datasets import (
    RsnaDataset,
    _validate_complete_cache_source_frame,
)


def _tables() -> dict[str, pa.Table]:
    samples = []
    labels = []
    splits = []
    inventory = []
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
            inventory.append(
                {
                    "sample_id": sample_id,
                    "relative_path": f"images/{split}-{index}.dcm",
                    "byte_size": 100 + index,
                    "sha256": str(index) * 64,
                }
            )
    return {
        "samples.parquet": pa.Table.from_pylist(samples),
        "labels.parquet": pa.Table.from_pylist(labels),
        "splits.parquet": pa.Table.from_pylist(splits),
        "source_inventory.parquet": pa.Table.from_pylist(inventory),
    }


def test_heldout_reader_rejects_forged_structural_authorization() -> None:
    config = with_runtime(
        load_experiment_config(
            "configs/rsna_metadata_logistic.yaml", bundle_id="bundle-" + "a" * 64
        ),
        seed=42,
    )
    forged = SimpleNamespace(require=lambda package_id: package_id)

    with pytest.raises(PermissionError, match="current-execution capability"):
        RsnaDataset().load_test(
            config,
            authorization=forged,
            package_id="model-package-forged",
        )


def test_dataset_adapter_loads_exact_bundle_and_exposes_only_approved_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = with_runtime(
        load_experiment_config(
            "configs/rsna_metadata_logistic.yaml", bundle_id="bundle-" + "a" * 64
        ),
        seed=42,
        manifest_directory=tmp_path / "manifests",
    )
    tables = _tables()
    validated: list[Path] = []
    reads: list[tuple[str, tuple[str, ...], tuple[tuple[str, str, object], ...]]] = []

    def validate(path, *, expected_bundle_id):
        validated.append(Path(path))
        assert expected_bundle_id == config.dataset.bundle_id
        return {
            "tasks": {
                config.task.task_id: {"label_policy_version": config.task.label_policy_version}
            },
        }

    monkeypatch.setattr("beyondcxr.training.rsna_datasets.validate_bundle_directory", validate)

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

    monkeypatch.setattr("beyondcxr.training.rsna_datasets.pq.read_table", read_table)

    data = RsnaDataset().load_train_validation(config)
    authorization = authorize_rsna_package(config, "model-package-test", monkeypatch=monkeypatch)
    test, lineage = RsnaDataset().load_test(
        config, authorization=authorization, package_id="model-package-test"
    )

    expected_bundle = tmp_path / "manifests" / "rsna" / "bundles" / config.dataset.bundle_id
    assert validated == [expected_bundle] * 2
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
    assert data.lineage.bundle_id == config.dataset.bundle_id
    assert lineage == data.lineage
    assert not hasattr(data, "test")
    assert test.sample_ids == ("rsna:test-0", "rsna:test-1")
    assert reads[0] == (
        "splits.parquet",
        ("sample_id", "split_name"),
        (("split_name", "in", ["train", "validation"]),),
    )
    assert reads[1][0:2] == (
        "samples.parquet",
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
        "labels.parquet",
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


def test_yaml_rejects_bundle_pin_and_unbound_data_access_fails(tmp_path: Path) -> None:
    text = Path("configs/rsna_metadata_logistic.yaml").read_text(encoding="utf-8")
    path = tmp_path / "pinned.yaml"
    path.write_text(
        text.replace("dataset_id: rsna", "dataset_id: rsna\n  bundle_id: bundle-" + "a" * 64),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown fields"):
        load_experiment_config(path)
    config = load_experiment_config("configs/rsna_metadata_logistic.yaml")
    with pytest.raises(ManifestBuildError, match="explicit bundle identity"):
        RsnaDataset().load_lineage(config)


@pytest.mark.parametrize(
    "mismatch",
    ["bundle_id", "task_id", "label_policy"],
)
def test_rsna_adapter_rejects_mismatched_authority_before_rows(
    monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    baseline = load_experiment_config(
        "configs/rsna_metadata_logistic.yaml", bundle_id="bundle-" + "a" * 64
    )
    config = baseline
    if mismatch == "bundle_id":
        config = replace(
            baseline,
            dataset=replace(baseline.dataset, bundle_id="bundle-" + "0" * 64),
        )
    elif mismatch == "task_id":
        config = replace(baseline, task=replace(baseline.task, task_id="unsupported"))
    else:
        config = replace(
            baseline,
            task=replace(baseline.task, label_policy_version="unsupported"),
        )
    metadata = {
        "tasks": {
            baseline.task.task_id: {"label_policy_version": baseline.task.label_policy_version}
        },
    }

    def validate(path, *, expected_bundle_id):
        del path
        if expected_bundle_id != baseline.dataset.bundle_id:
            raise ManifestBuildError("configured bundle differs")
        return metadata

    monkeypatch.setattr(
        "beyondcxr.training.rsna_datasets.validate_bundle_directory",
        validate,
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_datasets.pq.read_table",
        lambda *args, **kwargs: pytest.fail("row access occurred before authority rejection"),
    )

    with pytest.raises(ManifestBuildError):
        RsnaDataset().load_lineage(config)


def test_cache_source_frame_combines_builder_local_inputs_without_task_labels() -> None:
    tables = _tables()
    frame = _validate_complete_cache_source_frame(
        tables["samples.parquet"].select(["sample_id", "patient_id", "image_path"]).to_pandas(),
        tables["splits.parquet"].select(["sample_id", "split_name"]).to_pandas(),
        tables["source_inventory.parquet"]
        .select(["sample_id", "relative_path", "byte_size", "sha256"])
        .to_pandas(),
    )

    assert tuple(frame.columns) == (
        "sample_id",
        "patient_id",
        "image_path",
        "split_name",
        "byte_size",
        "sha256",
    )
