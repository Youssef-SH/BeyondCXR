from __future__ import annotations

import base64
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.dicom_loader import DicomRecord
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.hashing import sha256_file
from beyondcxr.data.rsna_cxr_cache import (
    CxrCacheIdentity,
    build_cxr_cache,
    preprocessing_identity,
    validate_cxr_cache,
)
from beyondcxr.training.config import load_experiment_config
from beyondcxr.training.device import resolve_device
from beyondcxr.training.execution import LoaderExecutionPolicy
from beyondcxr.training.neural import EpochPermutationSampler, build_image_loaders
from beyondcxr.training.rsna_datasets import (
    RsnaCachedFusionDataset,
    RsnaCachedImageDataset,
)


class _WorkerDigestDataset:
    """Exercise the real transform in workers without CI shared-memory tensors."""

    epoch_tagged_requests = True

    def __init__(self, dataset: RsnaCachedImageDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, request: int | tuple[int, int]) -> dict[str, str]:
        sample = self.dataset[request]
        encoded = base64.b64encode(sample["image"].numpy().tobytes()).decode("ascii")
        return {"sample_id": sample["sample_id"], "image_bytes": encoded}


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("rsna:a", "patient-a", "images/a.dcm", "train", 0, 1, "a" * 64),
            ("rsna:b", "patient-b", "images/b.dcm", "train", 1, 1, "b" * 64),
            ("rsna:c", "patient-c", "images/c.dcm", "validation", 0, 1, "c" * 64),
            ("rsna:d", "patient-d", "images/d.dcm", "test", 1, 1, "d" * 64),
        ],
        columns=(
            "sample_id",
            "patient_id",
            "image_path",
            "split_name",
            "target",
            "byte_size",
            "sha256",
        ),
    )


def _cache_frame() -> pd.DataFrame:
    return _frame().drop(columns="target")


def _identity(transform: StandardCxrTransform) -> CxrCacheIdentity:
    return CxrCacheIdentity(
        bundle_id="bundle-" + "0" * 64,
        bundle_manifest_sha256="1" * 64,
        source_inventory_file_sha256="2" * 64,
        source_inventory_arrow_sha256="3" * 64,
        preprocessing_sha256=preprocessing_identity(transform),
    )


def _image_frame() -> pd.DataFrame:
    return _frame().loc[
        _frame()["split_name"] == "train",
        ("sample_id", "patient_id", "image_path", "split_name", "target"),
    ]


def _sample_partitions() -> tuple[tuple[str, str], ...]:
    return tuple(zip(_frame()["sample_id"], _frame()["split_name"], strict=True))


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    images = {
        name: np.linspace(0.0, 1.0, (20 + index) * 30, dtype=np.float32).reshape(20 + index, 30)
        for index, name in enumerate(("a", "b", "c", "d"))
    }

    def decode(path: Path, **_authentication):
        name = path.stem
        pixels = images[name]
        return pixels, DicomRecord(
            path=str(path),
            patient_id=f"patient-{name}",
            patient_age=None,
            patient_sex=None,
            view_position=None,
            rows=pixels.shape[0],
            columns=pixels.shape[1],
            photometric_interpretation="MONOCHROME2",
        )

    monkeypatch.setattr("beyondcxr.data.rsna_cxr_cache.read_dicom", decode)
    transform = StandardCxrTransform(training=False)
    built = build_cxr_cache(
        _cache_frame(),
        dataset_root=tmp_path / "raw",
        cache_root=tmp_path / "cache",
        identity=_identity(transform),
        transform=transform,
    )
    return built, images, transform


def test_cache_preserves_deterministic_preprocessing_and_strict_identity(cache) -> None:
    built, images, transform = cache
    assert tuple(built.sample_rows) == tuple(_frame()["sample_id"])
    assert built.images.shape == (4, 1, 224, 224)
    assert built.images.dtype == np.float32
    assert torch.equal(built.image("rsna:a"), transform.deterministic_base(images["a"]))
    with pytest.raises(ManifestBuildError):
        validate_cxr_cache(
            built.directory,
            identity=replace(built.identity, bundle_id="bundle-" + "9" * 64),
            expected_sample_partitions=_sample_partitions(),
        )


def test_deterministic_transform_policy_participates_in_cache_identity(monkeypatch) -> None:
    transform = StandardCxrTransform(training=False)
    original = preprocessing_identity(transform)
    monkeypatch.setattr(
        "beyondcxr.data.cxr_transforms.CXR_TRANSFORM_POLICY_VERSION",
        "changed-deterministic-policy",
    )

    assert preprocessing_identity(transform) != original


def test_cache_identity_rejects_unsupported_source_authentication_policy() -> None:
    identity = _identity(StandardCxrTransform(training=False))

    with pytest.raises(ValueError):
        replace(identity, source_authentication_policy="changed-policy")
    assert set(identity.as_dict()) == {
        "bundle_id",
        "bundle_manifest_sha256",
        "source_inventory_file_sha256",
        "source_inventory_arrow_sha256",
        "preprocessing_sha256",
        "source_authentication_policy",
    }


def test_cache_source_frame_requires_all_three_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "beyondcxr.data.rsna_cxr_cache.read_dicom",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("decoded partial cache")),
    )
    with pytest.raises(ManifestBuildError):
        build_cxr_cache(
            _cache_frame().loc[_cache_frame()["split_name"] == "train"],
            dataset_root=tmp_path / "raw",
            cache_root=tmp_path / "cache",
            identity=_identity(StandardCxrTransform(training=False)),
            transform=StandardCxrTransform(training=False),
        )


def test_cache_source_frame_rejects_task_columns_before_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "beyondcxr.data.rsna_cxr_cache.read_dicom",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("decoded labeled rows")),
    )

    with pytest.raises(ManifestBuildError):
        build_cxr_cache(
            _frame(),
            dataset_root=tmp_path / "raw",
            cache_root=tmp_path / "cache",
            identity=_identity(StandardCxrTransform(training=False)),
            transform=StandardCxrTransform(training=False),
        )


def test_validated_cache_witness_is_immutable(cache) -> None:
    built, _, _ = cache
    with pytest.raises(TypeError):
        built.sample_rows["rsna:a"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        built.sample_partitions["rsna:a"] = "test"  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        built.source_authentication.file_count = 0  # type: ignore[misc]
    assert set(built.source_authentication.as_dict()) == {
        "policy_version",
        "partitions",
        "file_count",
        "source_inventory_arrow_sha256",
        "source_inventory_file_sha256",
    }
    with pytest.raises(ValueError):
        replace(built, sample_rows=dict(built.sample_rows))
    with pytest.raises(ValueError):
        replace(built.source_authentication, file_count=0)


def test_cache_rejects_partial_or_corrupt_publication(cache, tmp_path: Path) -> None:
    built, _, _ = cache
    partial = tmp_path / built.identity.cache_id
    partial.mkdir()
    (partial / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ManifestBuildError):
        validate_cxr_cache(
            partial,
            identity=built.identity,
            expected_sample_partitions=_sample_partitions(),
        )
    writable = np.load(built.directory / "images.npy", mmap_mode="r+")
    writable[0, 0, 0, 0] = 0.5
    writable.flush()
    with pytest.raises(ManifestBuildError):
        validate_cxr_cache(
            built.directory,
            identity=built.identity,
            expected_sample_partitions=_sample_partitions(),
        )


def test_cache_rejects_self_consistent_but_false_sample_partition_mapping(cache) -> None:
    built, _, _ = cache
    index_path = built.directory / "index.parquet"
    table = pq.read_table(index_path)
    index = table.to_pandas()
    index.loc[index["sample_id"] == "rsna:a", "split_name"] = "test"
    index.loc[index["sample_id"] == "rsna:d", "split_name"] = "train"
    pq.write_table(pa.Table.from_pandas(index, schema=table.schema), index_path)
    metadata_path = built.directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["mapping_sha256"] = sha256_file(index_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ManifestBuildError):
        validate_cxr_cache(
            built.directory,
            identity=built.identity,
            expected_sample_partitions=_sample_partitions(),
        )


def test_interrupted_cache_publication_cannot_leave_a_valid_destination(
    cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    built, _, transform = cache
    identity = replace(built.identity, bundle_manifest_sha256="3" * 64)

    def interrupt(stage: Path, destination: Path) -> None:
        del stage, destination
        raise OSError("interrupted publication")

    monkeypatch.setattr("beyondcxr.data.rsna_cxr_cache.publish_directory", interrupt)
    with pytest.raises(OSError):
        build_cxr_cache(
            _cache_frame(),
            dataset_root=built.directory.parent,
            cache_root=built.directory.parent,
            identity=identity,
            transform=transform,
        )
    assert not (built.directory.parent / identity.cache_id).exists()
    assert not any(path.name.startswith(".") for path in built.directory.parent.iterdir())


def test_cache_mmaps_close_on_build_and_validation_failure(
    cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    built, _, transform = cache
    opened_writers: list[np.memmap] = []
    original_open_memmap = np.lib.format.open_memmap

    def tracked_open_memmap(*args, **kwargs):
        images = original_open_memmap(*args, **kwargs)
        opened_writers.append(images)
        return images

    monkeypatch.setattr(np.lib.format, "open_memmap", tracked_open_memmap)
    monkeypatch.setattr(
        "beyondcxr.data.rsna_cxr_cache.read_dicom",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError((args, kwargs))),
    )
    identity = replace(built.identity, bundle_manifest_sha256="4" * 64)
    with pytest.raises(ManifestBuildError):
        build_cxr_cache(
            _cache_frame(),
            dataset_root=built.directory.parent,
            cache_root=built.directory.parent,
            identity=identity,
            transform=transform,
        )
    assert opened_writers
    assert all(mapping._mmap.closed for mapping in opened_writers)

    metadata_path = built.directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["content_sha256"] = "0" * 64
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    opened_readers: list[np.memmap] = []
    original_load = np.load

    def tracked_load(*args, **kwargs):
        images = original_load(*args, **kwargs)
        opened_readers.append(images)
        return images

    monkeypatch.setattr(np, "load", tracked_load)
    with pytest.raises(ManifestBuildError):
        validate_cxr_cache(
            built.directory,
            identity=built.identity,
            expected_sample_partitions=_sample_partitions(),
        )
    assert opened_readers
    assert all(mapping._mmap.closed for mapping in opened_readers)


def test_cache_build_rejects_unsafe_paths_and_decoded_patient_mismatch(
    cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    built, _, transform = cache
    identity = replace(built.identity, bundle_manifest_sha256="4" * 64)
    unsafe = _cache_frame().assign(
        image_path=["../a.dcm", "images/b.dcm", "images/c.dcm", "images/d.dcm"]
    )
    with pytest.raises(ManifestBuildError):
        build_cxr_cache(
            unsafe,
            dataset_root=built.directory.parent,
            cache_root=built.directory.parent,
            identity=identity,
            transform=transform,
        )

    def wrong_patient(path: Path, **_authentication):
        pixels = np.ones((8, 8), dtype=np.float32)
        return pixels, DicomRecord(
            path=str(path),
            patient_id="different-patient",
            patient_age=None,
            patient_sex=None,
            view_position=None,
            rows=8,
            columns=8,
            photometric_interpretation="MONOCHROME2",
        )

    monkeypatch.setattr("beyondcxr.data.rsna_cxr_cache.read_dicom", wrong_patient)
    with pytest.raises(ManifestBuildError):
        build_cxr_cache(
            _cache_frame(),
            dataset_root=built.directory.parent,
            cache_root=built.directory.parent,
            identity=identity,
            transform=transform,
        )


def _policy(workers: int, persistent: bool) -> LoaderExecutionPolicy:
    assert persistent is (workers > 0)
    return LoaderExecutionPolicy("reused", workers, False)


def _dataset_and_loader(cache, workers: int, persistent: bool, seed: int = 42):
    config = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    assert config.neural is not None
    dataset = RsnaCachedImageDataset(
        _image_frame(),
        cache=cache,
        expected_cache_identity=cache.identity,
        partition="train",
        transform=StandardCxrTransform(training=True),
        training_seed=seed,
    )
    loader = build_image_loaders(
        dataset,
        dataset,
        config=replace(config.neural, batch_size=2),
        runtime=resolve_device("cpu", mixed_precision=False, pin_memory_policy="disabled"),
        seed=seed,
        execution=_policy(workers, persistent),
    ).train
    return dataset, loader


def _epochs(cache, seed: int = 42):
    dataset, _ = _dataset_and_loader(cache, 0, False, seed)
    sampler = EpochPermutationSampler(dataset, seed=seed)
    epochs = []
    for _ in range(2):
        requests = tuple(sampler)
        samples = tuple(dataset[request] for request in requests)
        epochs.append(
            (
                tuple(sample["sample_id"] for sample in samples),
                torch.stack(tuple(sample["image"] for sample in samples)),
            )
        )
    return tuple(epochs)


def _loader_epochs(cache, workers: int, persistent: bool, seed: int = 42):
    config = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    assert config.neural is not None
    dataset = _WorkerDigestDataset(
        RsnaCachedImageDataset(
            _image_frame(),
            cache=cache,
            expected_cache_identity=cache.identity,
            partition="train",
            transform=StandardCxrTransform(training=True),
            training_seed=seed,
        )
    )
    loader = build_image_loaders(
        dataset,
        dataset,
        config=replace(config.neural, batch_size=2),
        runtime=resolve_device("cpu", mixed_precision=False, pin_memory_policy="disabled"),
        seed=seed,
        execution=_policy(workers, persistent),
    ).train
    epochs = []
    for _ in range(2):
        sample_ids: list[str] = []
        images: list[np.ndarray] = []
        for batch in loader:
            sample_ids.extend(batch["sample_id"])
            images.extend(
                np.frombuffer(base64.b64decode(value), dtype=np.float32).reshape(1, 224, 224)
                for value in batch["image_bytes"]
            )
        epochs.append((tuple(sample_ids), np.stack(images)))
    return tuple(epochs)


def test_order_and_augmentation_are_worker_and_persistence_independent(cache) -> None:
    built, _, _ = cache
    single = _loader_epochs(built, 0, False)
    one_worker = _loader_epochs(built, 1, True)
    persistent = _loader_epochs(built, 2, True)
    dataset, _ = _dataset_and_loader(built, 0, False)
    epoch_zero = tuple(EpochPermutationSampler(dataset, seed=42, epoch=0))
    repeated_epoch_zero = tuple(EpochPermutationSampler(dataset, seed=42, epoch=0))
    epoch_one = tuple(EpochPermutationSampler(dataset, seed=42, epoch=1))
    assert epoch_zero == repeated_epoch_zero
    assert {epoch for epoch, _ in epoch_zero} == {0}
    assert {epoch for epoch, _ in epoch_one} == {1}
    epoch_zero_images = dict(zip(single[0][0], single[0][1], strict=True))
    epoch_one_images = dict(zip(single[1][0], single[1][1], strict=True))

    assert epoch_zero_images.keys() == epoch_one_images.keys()
    assert any(
        not np.array_equal(epoch_zero_images[sample_id], epoch_one_images[sample_id])
        for sample_id in epoch_zero_images
    )
    for expected, one, reused in zip(single, one_worker, persistent, strict=True):
        assert expected[0] == one[0] == reused[0]
        np.testing.assert_array_equal(expected[1], one[1])
        np.testing.assert_array_equal(expected[1], reused[1])
    assert not np.array_equal(_epochs(built, seed=17)[0][1], single[0][1])


def test_validated_cache_process_transfer_reopens_the_memory_map(cache) -> None:
    built, _, _ = cache

    encoded = pickle.dumps(built)
    restored = pickle.loads(encoded)

    assert len(encoded) < built.images.nbytes
    assert restored.images.filename == built.images.filename
    np.testing.assert_array_equal(restored.images, built.images)


@pytest.mark.parametrize("seed", [True, -1, 1.5])
def test_cached_training_datasets_share_the_nonnegative_integer_seed_contract(cache, seed) -> None:
    built, _, _ = cache
    with pytest.raises(ManifestBuildError):
        RsnaCachedImageDataset(
            _image_frame(),
            cache=built,
            expected_cache_identity=built.identity,
            partition="train",
            transform=StandardCxrTransform(training=True),
            training_seed=seed,
        )


def test_fusion_uses_the_same_cache_contract(cache) -> None:
    built, _, _ = cache
    frame = (
        _frame()
        .loc[_frame()["split_name"] == "train"]
        .assign(
            age_years=40.0,
            age_is_implausible=False,
            sex="F",
            view_position="PA",
            pixel_spacing_row_mm=0.2,
            pixel_spacing_col_mm=0.2,
        )
    )
    columns = (
        "sample_id",
        "patient_id",
        "image_path",
        "age_years",
        "age_is_implausible",
        "sex",
        "view_position",
        "pixel_spacing_row_mm",
        "pixel_spacing_col_mm",
        "split_name",
        "target",
    )
    structured = np.arange(12, dtype=np.float64).reshape(2, 6)[:, ::2]
    assert not structured.flags.c_contiguous
    dataset = RsnaCachedFusionDataset(
        frame.loc[:, columns],
        structured,
        structured_sample_ids=tuple(frame["sample_id"]),
        cache=built,
        expected_cache_identity=built.identity,
        partition="train",
        transform=StandardCxrTransform(training=False),
        training_seed=42,
    )
    sample = dataset[0]
    assert sample["image"].shape == (1, 224, 224)
    assert sample["structured"].shape == (3,)
    assert sample["structured"].dtype == torch.float32
    assert sample["structured"].is_contiguous()
    np.testing.assert_array_equal(
        torch.stack([dataset[0]["structured"], dataset[1]["structured"]]).numpy(),
        structured.astype(np.float32),
    )

    with pytest.raises(ManifestBuildError):
        RsnaCachedFusionDataset(
            frame.loc[:, columns],
            np.ones((2, 3), dtype=np.float32),
            structured_sample_ids=("wrong", *tuple(frame["sample_id"])[1:]),
            cache=built,
            expected_cache_identity=built.identity,
            partition="train",
            transform=StandardCxrTransform(training=False),
            training_seed=42,
        )
    with pytest.raises(ManifestBuildError):
        RsnaCachedFusionDataset(
            frame.loc[:, columns],
            np.asarray([["bad"], ["data"]], dtype=object),
            structured_sample_ids=tuple(frame["sample_id"]),
            cache=built,
            expected_cache_identity=built.identity,
            partition="train",
            transform=StandardCxrTransform(training=False),
            training_seed=42,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bundle_id", "bundle-" + "9" * 64),
        ("bundle_manifest_sha256", "8" * 64),
        ("source_inventory_file_sha256", "7" * 64),
        ("source_inventory_arrow_sha256", "6" * 64),
        ("preprocessing_sha256", "f" * 64),
    ],
)
def test_cached_image_consumer_rejects_incompatible_identity(cache, field: str, value: str) -> None:
    built, _, _ = cache
    with pytest.raises(ManifestBuildError):
        RsnaCachedImageDataset(
            _image_frame(),
            cache=built,
            expected_cache_identity=replace(built.identity, **{field: value}),
            partition="train",
            transform=StandardCxrTransform(training=True),
            training_seed=42,
        )


def test_cached_fusion_consumer_rejects_incompatible_identity(cache) -> None:
    built, _, _ = cache
    frame = (
        _frame()
        .loc[_frame()["split_name"] == "train"]
        .assign(
            age_years=40.0,
            age_is_implausible=False,
            sex="F",
            view_position="PA",
            pixel_spacing_row_mm=0.2,
            pixel_spacing_col_mm=0.2,
        )
    )
    columns = (
        "sample_id",
        "patient_id",
        "image_path",
        "age_years",
        "age_is_implausible",
        "sex",
        "view_position",
        "pixel_spacing_row_mm",
        "pixel_spacing_col_mm",
        "split_name",
        "target",
    )
    with pytest.raises(ManifestBuildError):
        RsnaCachedFusionDataset(
            frame.loc[:, columns],
            np.ones((2, 3), dtype=np.float32),
            structured_sample_ids=tuple(frame["sample_id"]),
            cache=built,
            expected_cache_identity=replace(built.identity, preprocessing_sha256="f" * 64),
            partition="train",
            transform=StandardCxrTransform(training=False),
            training_seed=42,
        )
