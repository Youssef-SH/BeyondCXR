from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rsna_authorization_test_support import authorize_rsna_package
from rsna_manifest_test_support import (
    build_rsna_artifacts,
    write_bundle,
    write_header,
    write_sources,
)
from rsna_manifest_test_support import (
    tables as _tables,
)

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.hashing import LogicalTableContract, logical_table_sha256, sha256_file
from beyondcxr.data.rsna_artifacts import (
    ANNOTATIONS_FILENAME,
    LABELS_FILENAME,
    SAMPLES_FILENAME,
    SOURCE_INVENTORY_FILENAME,
    SPLITS_FILENAME,
    _artifact_contracts,
    _bundle_id,
    _bundle_identity_payload,
    resolve_bundle,
    validate_bundle_directory,
    validate_bundle_reference,
)
from beyondcxr.data.rsna_schemas import (
    RSNA_ANNOTATION_SCHEMA,
    RSNA_LABEL_SCHEMA,
    RSNA_SAMPLE_SCHEMA,
)
from beyondcxr.training.config import load_experiment_config, with_runtime
from beyondcxr.training.rsna_datasets import RsnaDataset, validate_rsna_cache_source_availability
from beyondcxr.utils.semantic_identity import canonical_json_bytes

_CONTRACTS = {filename: contract for _, filename, contract in _artifact_contracts()}


def _logical_hash(filename: str, table: pa.Table) -> str:
    return logical_table_sha256(table, _CONTRACTS[filename])


def test_parquet_round_trip_is_exact_and_nested_free(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")

    restored_samples = pq.read_table(written.paths.samples_path)
    restored_labels = pq.read_table(written.paths.labels_path)
    restored_annotations = pq.read_table(written.paths.annotations_path)
    restored_inventory = pq.read_table(written.paths.source_inventory_path)
    assert restored_samples.equals(result.samples)
    assert restored_labels.equals(result.labels)
    assert restored_annotations.equals(result.annotations)
    assert restored_inventory.equals(result.source_inventory)
    assert restored_samples.schema == RSNA_SAMPLE_SCHEMA
    assert restored_labels.schema == RSNA_LABEL_SCHEMA
    assert restored_annotations.schema == RSNA_ANNOTATION_SCHEMA
    assert not any(pa.types.is_nested(field.type) for field in restored_annotations.schema)


def test_published_authority_exact_pin_resolves_for_formal_cache_consumption(
    tmp_path: Path,
) -> None:
    source_root = write_sources(tmp_path / "source")
    labels_path = source_root / "stage_2_train_labels.csv"
    classes_path = source_root / "stage_2_detailed_class_info.csv"
    labels = pd.read_csv(labels_path)
    classes = pd.read_csv(classes_path)
    extra_labels = []
    extra_classes = []
    for target in (0, 1):
        for index in (1, 2):
            patient_id = f"target-{target}-{index}"
            write_header(source_root / "stage_2_train_images" / f"{patient_id}.dcm", patient_id)
            extra_labels.append(
                {
                    "patientId": patient_id,
                    "x": 1 if target else None,
                    "y": 2 if target else None,
                    "width": 3 if target else None,
                    "height": 4 if target else None,
                    "Target": target,
                }
            )
            extra_classes.append(
                {
                    "patientId": patient_id,
                    "class": "Lung Opacity" if target else "Normal",
                }
            )
    pd.concat([labels, pd.DataFrame(extra_labels)], ignore_index=True).to_csv(
        labels_path, index=False
    )
    pd.concat([classes, pd.DataFrame(extra_classes)], ignore_index=True).to_csv(
        classes_path, index=False
    )
    result = build_rsna_artifacts(source_root)
    written = write_bundle(result, tmp_path / "manifests")
    baseline = load_experiment_config(
        "configs/rsna_cxr_densenet.yaml", bundle_id="bundle-" + "a" * 64
    )
    config = with_runtime(
        replace(
            baseline,
            dataset=replace(
                baseline.dataset,
                bundle_id=written.paths.bundle_id,
            ),
        ),
        seed=17,
        manifest_directory=tmp_path / "manifests",
        source_root=source_root,
    )

    validate_rsna_cache_source_availability(config)

    wrong = replace(config, dataset=replace(config.dataset, bundle_id="bundle-" + "0" * 64))
    with pytest.raises(ManifestBuildError):
        validate_rsna_cache_source_availability(wrong)


def test_all_rsna_consumers_reject_logically_corrupted_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")
    inventory = pq.read_table(written.paths.source_inventory_path)
    rows = inventory.to_pylist()
    rows[0]["byte_size"] += 1
    pq.write_table(
        pa.Table.from_pylist(rows, schema=inventory.schema),
        written.paths.source_inventory_path,
    )

    metadata = load_experiment_config(
        "configs/rsna_metadata_logistic.yaml", bundle_id="bundle-" + "a" * 64
    )
    metadata = with_runtime(
        replace(metadata, dataset=replace(metadata.dataset, bundle_id=written.paths.bundle_id)),
        seed=42,
        manifest_directory=tmp_path / "manifests",
    )
    image = load_experiment_config("configs/rsna_cxr_densenet.yaml", bundle_id="bundle-" + "a" * 64)
    image = with_runtime(
        replace(image, dataset=replace(image.dataset, bundle_id=written.paths.bundle_id)),
        seed=17,
        manifest_directory=tmp_path / "manifests",
    )
    adapter = RsnaDataset()

    with pytest.raises(ManifestBuildError, match="Logical table hash mismatch"):
        adapter.load_train_validation(metadata)
    with pytest.raises(ManifestBuildError, match="Logical table hash mismatch"):
        validate_rsna_cache_source_availability(image)
    authorization = authorize_rsna_package(
        metadata, "model-package-corruption-test", monkeypatch=monkeypatch
    )
    with pytest.raises(ManifestBuildError, match="Logical table hash mismatch"):
        adapter.load_test(
            metadata,
            authorization=authorization,
            package_id="model-package-corruption-test",
        )


def test_logical_table_hashes_are_deterministic(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    first = write_bundle(result, tmp_path / "first")
    second = write_bundle(result, tmp_path / "second")

    assert first.logical_table_sha256 == second.logical_table_sha256
    assert first.paths.bundle_id == second.paths.bundle_id
    assert first.logical_table_sha256[SAMPLES_FILENAME] == _logical_hash(
        SAMPLES_FILENAME, result.samples
    )


def test_logical_hash_ignores_null_buffers_and_survives_parquet_round_trip(
    tmp_path: Path,
) -> None:
    null_bitmap = pa.py_buffer(b"\x01")
    first = pa.Array.from_buffers(
        pa.int32(),
        2,
        [null_bitmap, pa.py_buffer((1).to_bytes(4, "little") + (7).to_bytes(4, "little"))],
        null_count=1,
    )
    second = pa.Array.from_buffers(
        pa.int32(),
        2,
        [null_bitmap, pa.py_buffer((1).to_bytes(4, "little") + (99).to_bytes(4, "little"))],
        null_count=1,
    )

    assert first.to_pylist() == second.to_pylist() == [1, None]
    first_table = pa.table({"value": first})
    second_table = pa.table({"value": second})
    path = tmp_path / "nulls.parquet"
    pq.write_table(first_table, path)
    contract = LogicalTableContract("test.nulls", 1, first_table.schema, ("value",), ("value",))

    expected = logical_table_sha256(first_table, contract)
    assert expected == logical_table_sha256(second_table, contract)
    assert expected == logical_table_sha256(pq.read_table(path), contract)


def test_semantic_identity_is_independent_of_parquet_encoding(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    zstd = tmp_path / "samples-zstd.parquet"
    snappy = tmp_path / "samples-snappy.parquet"
    pq.write_table(result.samples, zstd, compression="zstd")
    pq.write_table(result.samples, snappy, compression="snappy")

    assert sha256_file(zstd) != sha256_file(snappy)
    zstd_samples = pq.read_table(zstd)
    snappy_samples = pq.read_table(snappy)
    zstd_sample_hash = _logical_hash(SAMPLES_FILENAME, zstd_samples)
    snappy_sample_hash = _logical_hash(SAMPLES_FILENAME, snappy_samples)
    assert zstd_sample_hash == snappy_sample_hash

    zstd_logical_hashes = {
        SAMPLES_FILENAME: zstd_sample_hash,
        LABELS_FILENAME: _logical_hash(LABELS_FILENAME, result.labels),
        ANNOTATIONS_FILENAME: _logical_hash(ANNOTATIONS_FILENAME, result.annotations),
        SPLITS_FILENAME: _logical_hash(SPLITS_FILENAME, result.splits),
        SOURCE_INVENTORY_FILENAME: _logical_hash(
            SOURCE_INVENTORY_FILENAME, result.source_inventory
        ),
    }
    snappy_logical_hashes = {
        SAMPLES_FILENAME: snappy_sample_hash,
        LABELS_FILENAME: _logical_hash(LABELS_FILENAME, result.labels),
        ANNOTATIONS_FILENAME: _logical_hash(ANNOTATIONS_FILENAME, result.annotations),
        SPLITS_FILENAME: _logical_hash(SPLITS_FILENAME, result.splits),
        SOURCE_INVENTORY_FILENAME: _logical_hash(
            SOURCE_INVENTORY_FILENAME, result.source_inventory
        ),
    }
    assert zstd_logical_hashes == snappy_logical_hashes
    assert _bundle_id(zstd_logical_hashes, result.metadata) == _bundle_id(
        snappy_logical_hashes, result.metadata
    )


def test_bundle_id_changes_with_semantic_metadata(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    changed = {
        **result.metadata,
        "dataset": {**result.metadata["dataset"], "release": "semantic-change"},
    }
    logical_hashes = {
        SAMPLES_FILENAME: _logical_hash(SAMPLES_FILENAME, result.samples),
        LABELS_FILENAME: _logical_hash(LABELS_FILENAME, result.labels),
        ANNOTATIONS_FILENAME: _logical_hash(ANNOTATIONS_FILENAME, result.annotations),
        SPLITS_FILENAME: _logical_hash(SPLITS_FILENAME, result.splits),
        SOURCE_INVENTORY_FILENAME: _logical_hash(
            SOURCE_INVENTORY_FILENAME, result.source_inventory
        ),
    }

    assert _bundle_id(logical_hashes, result.metadata) != _bundle_id(logical_hashes, changed)


def test_bundle_manifest_excludes_generation_and_runtime_provenance(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    assert "generation" not in metadata
    assert "provenance" not in metadata
    assert resolve_bundle(output).bundle_id == written.paths.bundle_id


def test_bundle_id_uses_semantic_metadata_and_logical_artifact_hashes(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    arrow_hashes = {
        SAMPLES_FILENAME: "a" * 64,
        LABELS_FILENAME: "b" * 64,
        ANNOTATIONS_FILENAME: "c" * 64,
        SPLITS_FILENAME: "d" * 64,
        SOURCE_INVENTORY_FILENAME: "e" * 64,
    }
    original = _bundle_id(arrow_hashes, result.metadata)
    changed_split = {
        **result.metadata,
        "membership": {"split": {**result.metadata["membership"]["split"], "seed": 43}},
    }

    assert _bundle_id(arrow_hashes, changed_split) != original
    assert _bundle_id({**arrow_hashes, SAMPLES_FILENAME: "f" * 64}, result.metadata) != original


def test_rsna_semantic_payload_uses_abstract_artifact_roles(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    logical_hashes = {
        SAMPLES_FILENAME: "a" * 64,
        LABELS_FILENAME: "b" * 64,
        ANNOTATIONS_FILENAME: "c" * 64,
        SPLITS_FILENAME: "d" * 64,
        SOURCE_INVENTORY_FILENAME: "e" * 64,
    }

    payload = _bundle_identity_payload(logical_hashes, result.metadata)

    assert set(payload["artifacts"]) == {
        "samples",
        "labels",
        "annotations",
        "splits",
        "source_inventory",
    }
    assert ".parquet" not in json.dumps(payload, sort_keys=True)


def test_staging_failure_preserves_current_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    first = write_bundle(result, output)
    current_before = first.paths.current_path.read_text(encoding="utf-8")

    def fail_validation(*args: object, **kwargs: object) -> None:
        raise ManifestBuildError("staged validation failed")

    monkeypatch.setattr("beyondcxr.data.rsna_artifacts.validate_bundle_directory", fail_validation)
    with pytest.raises(ManifestBuildError):
        write_bundle(result, output)
    assert first.paths.current_path.read_text(encoding="utf-8") == current_before
    validate_bundle_directory(
        first.paths.bundle_directory,
        expected_bundle_id=first.paths.bundle_id,
    )


def test_bundle_publication_preserves_concurrent_empty_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import beyondcxr.utils.publication as publication

    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    rename = publication.rename_directory_no_replace
    destinations = []

    def create_competing_directory(source: Path, destination: Path) -> None:
        destination.mkdir()
        destinations.append(destination)
        rename(source, destination)

    monkeypatch.setattr(publication, "rename_directory_no_replace", create_competing_directory)
    with pytest.raises(ManifestBuildError):
        write_bundle(result, output)
    assert len(destinations) == 1
    assert list(destinations[0].iterdir()) == []
    assert not (output / "rsna/CURRENT").exists()


def test_new_bundle_is_validated_once_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result = _tables(tmp_path)
    calls: list[tuple[Path, bool]] = []

    def record_validation(
        directory: str | Path,
        *,
        expected_bundle_id: str | None = None,
        enforce_directory_name: bool = True,
    ) -> dict[str, object]:
        calls.append((Path(directory), enforce_directory_name))
        return validate_bundle_directory(
            directory,
            expected_bundle_id=expected_bundle_id,
            enforce_directory_name=enforce_directory_name,
        )

    monkeypatch.setattr(
        "beyondcxr.data.rsna_artifacts.validate_bundle_directory",
        record_validation,
    )
    written = write_bundle(result, tmp_path / "manifests")

    assert len(calls) == 1
    assert calls[0][1] is False
    resolve_bundle(tmp_path / "manifests")
    assert calls[-1] == (written.paths.bundle_directory, True)


def test_consumer_rejects_bundle_with_hash_mismatch(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")
    written.paths.labels_path.write_bytes(b"corrupt")

    with pytest.raises(ManifestBuildError):
        resolve_bundle(tmp_path / "manifests")


def test_bundle_reference_validation_does_not_materialize_parquet_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")

    monkeypatch.setattr(
        "beyondcxr.data.rsna_artifacts.pq.read_table",
        lambda *args, **kwargs: pytest.fail((args, kwargs, "row materialization")),
    )

    validated = validate_bundle_reference(
        written.paths.bundle_directory,
        expected_bundle_id=written.paths.bundle_id,
    )
    assert validated.manifest["bundle"]["bundle_id"] == written.paths.bundle_id


def test_bundle_reference_rejects_artifact_row_count_mismatch(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"][SAMPLES_FILENAME]["row_count"] += 1
    written.paths.metadata_path.write_bytes(canonical_json_bytes(metadata))

    with pytest.raises(ManifestBuildError):
        validate_bundle_reference(
            written.paths.bundle_directory,
            expected_bundle_id=written.paths.bundle_id,
        )


def test_fresh_reconstruction_needs_no_historical_manifest_witness(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    first = write_bundle(result, tmp_path / "first")
    second = write_bundle(result, tmp_path / "second")
    base = load_experiment_config("configs/rsna_cxr_densenet.yaml", bundle_id="bundle-" + "a" * 64)
    configured = with_runtime(
        replace(
            base,
            dataset=replace(base.dataset, bundle_id=second.paths.bundle_id),
        ),
        manifest_directory=tmp_path / "second",
    )
    assert first.paths.bundle_id == second.paths.bundle_id
    assert first.paths.metadata_path.read_bytes() == second.paths.metadata_path.read_bytes()
    assert RsnaDataset().load_lineage(configured).bundle_id == second.paths.bundle_id


def test_bundle_reference_rejects_wrong_selection_and_malformed_manifest(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    selected = write_bundle(result, tmp_path / "selected")
    with pytest.raises(ManifestBuildError):
        validate_bundle_reference(
            selected.paths.bundle_directory,
            expected_bundle_id="bundle-wrong",
        )

    malformed = write_bundle(result, tmp_path / "malformed")
    malformed.paths.metadata_path.write_bytes(b"not-json")
    with pytest.raises(ManifestBuildError):
        validate_bundle_reference(
            malformed.paths.bundle_directory,
            expected_bundle_id=malformed.paths.bundle_id,
        )


def test_cxr_test_wrong_bundle_fails_before_partition_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result = _tables(tmp_path)
    write_bundle(result, tmp_path / "manifests")
    base = load_experiment_config("configs/rsna_cxr_densenet.yaml", bundle_id="bundle-" + "a" * 64)
    configured = with_runtime(
        replace(base, dataset=replace(base.dataset, bundle_id="bundle-" + "0" * 64)),
        seed=42,
        manifest_directory=tmp_path / "manifests",
        source_root=tmp_path / "raw",
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_datasets._merge_task_tables",
        lambda *args, **kwargs: pytest.fail((args, kwargs, "test partition access")),
    )

    with pytest.raises(ManifestBuildError):
        RsnaDataset().load_cxr_test(
            configured,
            authorization=authorize_rsna_package(
                configured, "model-package-test", monkeypatch=monkeypatch
            ),
            package_id="model-package-test",
        )


def test_bundle_identity_is_independent_of_parquet_reencoding(
    tmp_path: Path,
) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")
    table = pq.read_table(written.paths.labels_path)
    pq.write_table(table, written.paths.labels_path, compression=None)
    validate_bundle_directory(
        written.paths.bundle_directory, expected_bundle_id=written.paths.bundle_id
    )


def test_bundle_reference_lineage_rejects_artifact_only_and_metadata_only_tampering(
    tmp_path: Path,
) -> None:
    _, result = _tables(tmp_path)
    artifact = write_bundle(result, tmp_path / "artifact")
    artifact.paths.labels_path.write_bytes(b"tampered")
    with pytest.raises(ManifestBuildError):
        validate_bundle_reference(
            artifact.paths.bundle_directory,
            expected_bundle_id=artifact.paths.bundle_id,
        )

    manifest = write_bundle(result, tmp_path / "manifest")
    document = json.loads(manifest.paths.metadata_path.read_text(encoding="utf-8"))
    document["source"]["files"][next(iter(document["source"]["files"]))] = "a" * 64
    manifest.paths.metadata_path.write_bytes(canonical_json_bytes(document))
    with pytest.raises(ManifestBuildError):
        validate_bundle_reference(
            manifest.paths.bundle_directory,
            expected_bundle_id=manifest.paths.bundle_id,
        )


def test_consumer_rejects_declared_artifact_hash_tampering(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"][LABELS_FILENAME]["semantic_sha256"] = "0" * 64
    written.paths.metadata_path.write_bytes(canonical_json_bytes(metadata))

    with pytest.raises(ManifestBuildError):
        resolve_bundle(output)


@pytest.mark.parametrize("entry_kind", ["file", "directory", "symlink"])
def test_consumer_rejects_unexpected_bundle_entries(tmp_path: Path, entry_kind: str) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    unexpected = written.paths.bundle_directory / "unexpected"
    if entry_kind == "file":
        unexpected.write_text("unexpected\n", encoding="utf-8")
    elif entry_kind == "directory":
        unexpected.mkdir()
    else:
        unexpected.symlink_to(written.paths.samples_path.name)

    with pytest.raises(ManifestBuildError):
        resolve_bundle(output)


@pytest.mark.parametrize("attribute", ["labels_path", "metadata_path"])
def test_consumer_rejects_required_bundle_artifact_symlinks(tmp_path: Path, attribute: str) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    required_path = getattr(written.paths, attribute)
    external = tmp_path / f"external-{required_path.name}"
    required_path.rename(external)
    required_path.symlink_to(external)

    with pytest.raises(ManifestBuildError):
        resolve_bundle(output)


def test_consumer_rejects_extra_declared_artifact(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"]["unexpected.parquet"] = {}
    written.paths.metadata_path.write_bytes(canonical_json_bytes(metadata))

    with pytest.raises(ManifestBuildError):
        resolve_bundle(output)


@pytest.mark.parametrize("schema_version", [True, 1.0, "1", None, 0])
def test_bundle_manifest_requires_integer_schema_version_one(
    tmp_path: Path, schema_version: object
) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["bundle_manifest_schema_version"] = schema_version
    written.paths.metadata_path.write_bytes(canonical_json_bytes(metadata))
    with pytest.raises(ManifestBuildError):
        resolve_bundle(output)


def test_source_inventory_authenticates_every_dicom_and_detects_tampering(tmp_path: Path) -> None:
    root, result = _tables(tmp_path)
    rows = result.source_inventory.to_pylist()
    assert {row["relative_path"] for row in rows} == {
        "stage_2_train_images/negative.dcm",
        "stage_2_train_images/positive.dcm",
    }
    for row in rows:
        source = root / row["relative_path"]
        assert row["byte_size"] == source.stat().st_size
        assert row["sha256"] == sha256_file(source)

    written = write_bundle(result, tmp_path / "manifests")
    written.paths.source_inventory_path.write_bytes(b"tampered")
    with pytest.raises(ManifestBuildError):
        resolve_bundle(tmp_path / "manifests")


def test_portable_bundle_validation_does_not_access_external_dicoms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")

    def reject_external_access(*args: object, **kwargs: object) -> Path:
        raise AssertionError("portable validation accessed the external dataset")

    monkeypatch.setattr(
        "beyondcxr.data.rsna_artifacts.resolve_image_path",
        reject_external_access,
    )
    validate_bundle_directory(
        written.paths.bundle_directory,
        expected_bundle_id=written.paths.bundle_id,
    )


def test_consumer_rejects_metadata_that_does_not_match_bundle_id(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["membership"]["split"]["seed"] = 43
    written.paths.metadata_path.write_bytes(canonical_json_bytes(metadata))

    with pytest.raises(ManifestBuildError):
        resolve_bundle(output)


@pytest.mark.parametrize(
    "invalid_form",
    [
        "dictionary-ratios",
        "reordered-ratios",
        "duplicate-split-name",
        "unknown-field",
    ],
)
def test_consumer_rejects_noncanonical_split_metadata(
    tmp_path: Path,
    invalid_form: str,
) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    split = metadata["membership"]["split"]
    if invalid_form == "dictionary-ratios":
        split["ratios"] = {"train": 0.7, "validation": 0.15, "test": 0.15}
    elif invalid_form == "reordered-ratios":
        split["ratios"] = [split["ratios"][0], split["ratios"][2], split["ratios"][1]]
    elif invalid_form == "duplicate-split-name":
        split["ratios"][1]["split_name"] = "train"
    else:
        split["unexpected"] = "value"
    written.paths.metadata_path.write_bytes(canonical_json_bytes(metadata))

    with pytest.raises(ManifestBuildError):
        resolve_bundle(output)


def test_explicit_bundle_resolution_does_not_depend_on_current(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    written.paths.current_path.unlink()

    resolved = resolve_bundle(output, bundle_id=written.paths.bundle_id)

    assert resolved.bundle_id == written.paths.bundle_id
