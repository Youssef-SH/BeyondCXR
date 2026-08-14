from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rsna_manifest_test_support import tables as _tables

from radfusion.data.errors import ManifestBuildError
from radfusion.data.hashing import logical_arrow_sha256, sha256_file
from radfusion.data.rsna_artifacts import (
    ANNOTATIONS_FILENAME,
    LABELS_FILENAME,
    SAMPLES_FILENAME,
    SOURCE_INVENTORY_FILENAME,
    SPLITS_FILENAME,
    _bundle_id,
    _bundle_identity_payload,
    resolve_bundle,
    validate_bundle_directory,
    validate_bundle_reference,
    write_bundle,
)
from radfusion.data.rsna_schemas import (
    RSNA_ANNOTATION_SCHEMA,
    RSNA_LABEL_SCHEMA,
    RSNA_SAMPLE_SCHEMA,
)
from radfusion.training.config import load_experiment_config, with_runtime
from radfusion.training.rsna_datasets import RsnaDataset


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


def test_logical_arrow_hashes_are_deterministic(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    first = write_bundle(result, tmp_path / "first")
    second = write_bundle(result, tmp_path / "second")

    assert first.logical_arrow_sha256 == second.logical_arrow_sha256
    assert first.paths.bundle_id == second.paths.bundle_id
    assert first.logical_arrow_sha256[SAMPLES_FILENAME] == logical_arrow_sha256(result.samples)


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

    expected = logical_arrow_sha256(first_table)
    assert expected == logical_arrow_sha256(second_table)
    assert expected == logical_arrow_sha256(pq.read_table(path))


def test_semantic_identity_is_independent_of_parquet_encoding(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    zstd = tmp_path / "samples-zstd.parquet"
    snappy = tmp_path / "samples-snappy.parquet"
    pq.write_table(result.samples, zstd, compression="zstd")
    pq.write_table(result.samples, snappy, compression="snappy")

    assert sha256_file(zstd) != sha256_file(snappy)
    zstd_samples = pq.read_table(zstd)
    snappy_samples = pq.read_table(snappy)
    zstd_sample_hash = logical_arrow_sha256(zstd_samples)
    snappy_sample_hash = logical_arrow_sha256(snappy_samples)
    assert zstd_sample_hash == snappy_sample_hash

    zstd_logical_hashes = {
        SAMPLES_FILENAME: zstd_sample_hash,
        LABELS_FILENAME: logical_arrow_sha256(result.labels),
        ANNOTATIONS_FILENAME: logical_arrow_sha256(result.annotations),
        SPLITS_FILENAME: logical_arrow_sha256(result.splits),
        SOURCE_INVENTORY_FILENAME: logical_arrow_sha256(result.source_inventory),
    }
    snappy_logical_hashes = {
        SAMPLES_FILENAME: snappy_sample_hash,
        LABELS_FILENAME: logical_arrow_sha256(result.labels),
        ANNOTATIONS_FILENAME: logical_arrow_sha256(result.annotations),
        SPLITS_FILENAME: logical_arrow_sha256(result.splits),
        SOURCE_INVENTORY_FILENAME: logical_arrow_sha256(result.source_inventory),
    }
    assert zstd_logical_hashes == snappy_logical_hashes
    assert _bundle_id(zstd_logical_hashes, result.metadata) == _bundle_id(
        snappy_logical_hashes, result.metadata
    )


def test_bundle_id_ignores_nonsemantic_metadata(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    changed = replace(
        result,
        metadata={
            **result.metadata,
            "generation": {"timestamp_utc": "2099-01-01T00:00:00+00:00"},
            "provenance": {
                "tool_versions": {
                    "python": "different",
                    "pandas": "different",
                    "pyarrow": "different",
                    "pydicom": "different",
                }
            },
        },
    )

    first = write_bundle(result, tmp_path / "first")
    second = write_bundle(changed, tmp_path / "second")

    assert first.paths.bundle_id == second.paths.bundle_id


def test_bundle_id_changes_with_semantic_metadata(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    changed = {
        **result.metadata,
        "dataset": {**result.metadata["dataset"], "release": "semantic-change"},
    }
    logical_hashes = {
        SAMPLES_FILENAME: logical_arrow_sha256(result.samples),
        LABELS_FILENAME: logical_arrow_sha256(result.labels),
        ANNOTATIONS_FILENAME: logical_arrow_sha256(result.annotations),
        SPLITS_FILENAME: logical_arrow_sha256(result.splits),
        SOURCE_INVENTORY_FILENAME: logical_arrow_sha256(result.source_inventory),
    }

    assert _bundle_id(logical_hashes, result.metadata) != _bundle_id(logical_hashes, changed)


def test_bundle_validation_ignores_provenance_and_generation_changes(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    original_id = written.paths.bundle_id
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["generation"]["timestamp_utc"] = "2099-01-01T00:00:00+00:00"
    metadata["provenance"]["tool_versions"] = {"python": "different"}
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    assert resolve_bundle(output).bundle_id == original_id


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

    monkeypatch.setattr("radfusion.data.rsna_artifacts.validate_bundle_directory", fail_validation)
    with pytest.raises(ManifestBuildError):
        write_bundle(result, output)
    assert first.paths.current_path.read_text(encoding="utf-8") == current_before
    validate_bundle_directory(
        first.paths.bundle_directory,
        expected_bundle_id=first.paths.bundle_id,
    )


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
        "radfusion.data.rsna_artifacts.validate_bundle_directory",
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
        "radfusion.data.rsna_artifacts.pq.read_table",
        lambda *args, **kwargs: pytest.fail((args, kwargs, "row materialization")),
    )

    validated = validate_bundle_reference(
        written.paths.bundle_directory,
        expected_bundle_id=written.paths.bundle_id,
        expected_manifest_sha256=sha256_file(written.paths.metadata_path),
    )
    assert validated.manifest["bundle"]["bundle_id"] == written.paths.bundle_id
    assert validated.manifest_sha256 == sha256_file(written.paths.metadata_path)


def test_bundle_reference_rejects_artifact_row_count_mismatch(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"][SAMPLES_FILENAME]["row_count"] += 1
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ManifestBuildError):
        validate_bundle_reference(
            written.paths.bundle_directory,
            expected_bundle_id=written.paths.bundle_id,
        )


def test_bundle_reference_allows_new_operational_metadata_for_same_identity(
    tmp_path: Path,
) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")
    published_metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    base = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    configured = with_runtime(
        replace(
            base,
            dataset=replace(
                base.dataset,
                bundle_id=written.paths.bundle_id,
                bundle_manifest_sha256=sha256_file(written.paths.metadata_path),
                split_assignment_id=published_metadata["membership"]["split"][
                    "split_assignment_id"
                ],
            ),
        ),
        manifest_directory=tmp_path / "manifests",
    )
    original_lineage = RsnaDataset().load_lineage(configured)
    original = validate_bundle_reference(
        written.paths.bundle_directory,
        expected_bundle_id=written.paths.bundle_id,
    )
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["generation"]["timestamp_utc"] = "2099-01-01T00:00:00+00:00"
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    regenerated = validate_bundle_reference(
        written.paths.bundle_directory,
        expected_bundle_id=written.paths.bundle_id,
    )

    assert regenerated.manifest["bundle"]["bundle_id"] == written.paths.bundle_id
    assert regenerated.manifest_sha256 != original.manifest_sha256
    refreshed_config = replace(
        configured,
        dataset=replace(
            configured.dataset,
            bundle_manifest_sha256=sha256_file(written.paths.metadata_path),
        ),
    )
    assert RsnaDataset().load_lineage(refreshed_config) == original_lineage


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


def test_cxr_test_manifest_mismatch_fails_before_partition_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")
    base = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    configured = with_runtime(
        replace(base, dataset=replace(base.dataset, bundle_id=written.paths.bundle_id)),
        manifest_directory=tmp_path / "manifests",
        source_root=tmp_path / "raw",
    )
    monkeypatch.setattr(
        "radfusion.training.rsna_datasets._task_frame",
        lambda *args, **kwargs: pytest.fail((args, kwargs, "test partition access")),
    )

    with pytest.raises(ManifestBuildError):
        RsnaDataset().load_cxr_test(
            configured,
            expected_manifest_sha256="0" * 64,
        )


def test_bundle_reference_lineage_rejects_manifest_and_coordinated_artifact_tampering(
    tmp_path: Path,
) -> None:
    _, result = _tables(tmp_path)
    written = write_bundle(result, tmp_path / "manifests")
    expected_manifest_sha256 = sha256_file(written.paths.metadata_path)

    table = pq.read_table(written.paths.labels_path)
    pq.write_table(table, written.paths.labels_path, compression=None)
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"][LABELS_FILENAME]["physical_file_sha256"] = sha256_file(
        written.paths.labels_path
    )
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ManifestBuildError):
        validate_bundle_reference(
            written.paths.bundle_directory,
            expected_bundle_id=written.paths.bundle_id,
            expected_manifest_sha256=expected_manifest_sha256,
        )


def test_bundle_reference_lineage_rejects_artifact_only_and_metadata_only_tampering(
    tmp_path: Path,
) -> None:
    _, result = _tables(tmp_path)
    artifact = write_bundle(result, tmp_path / "artifact")
    artifact_manifest_sha256 = sha256_file(artifact.paths.metadata_path)
    artifact.paths.labels_path.write_bytes(b"tampered")
    with pytest.raises(ManifestBuildError):
        validate_bundle_reference(
            artifact.paths.bundle_directory,
            expected_bundle_id=artifact.paths.bundle_id,
            expected_manifest_sha256=artifact_manifest_sha256,
        )

    manifest = write_bundle(result, tmp_path / "manifest")
    manifest_sha256 = sha256_file(manifest.paths.metadata_path)
    document = json.loads(manifest.paths.metadata_path.read_text(encoding="utf-8"))
    document["source"]["files"][next(iter(document["source"]["files"]))] = "a" * 64
    manifest.paths.metadata_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ManifestBuildError):
        validate_bundle_reference(
            manifest.paths.bundle_directory,
            expected_bundle_id=manifest.paths.bundle_id,
            expected_manifest_sha256=manifest_sha256,
        )


def test_consumer_rejects_declared_artifact_hash_tampering(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    metadata = json.loads(written.paths.metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"][LABELS_FILENAME]["physical_file_sha256"] = "tampered"
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

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
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

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
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
        "radfusion.data.rsna_artifacts.resolve_image_path",
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
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

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
    written.paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestBuildError):
        resolve_bundle(output)


def test_explicit_bundle_resolution_does_not_depend_on_current(tmp_path: Path) -> None:
    _, result = _tables(tmp_path)
    output = tmp_path / "manifests"
    written = write_bundle(result, output)
    written.paths.current_path.unlink()

    resolved = resolve_bundle(output, bundle_id=written.paths.bundle_id)

    assert resolved.bundle_id == written.paths.bundle_id
