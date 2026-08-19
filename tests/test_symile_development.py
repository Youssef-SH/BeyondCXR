from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
from neural_test_support import cpu_runtime
from sklearn.linear_model import LogisticRegression
from symile_development_test_support import (
    _assignments,
    _development_frame,
    _family_config_path,
    _inner_split,
    _lineage,
    _neural_history,
    _publish_family_folds,
)

import radfusion.data.symile_source as symile_source
import radfusion.training.symile_analysis as symile_analysis
import radfusion.training.symile_data as symile_data
import radfusion.training.symile_development as symile_development
import radfusion.training.symile_ecg_data as symile_ecg_data
import radfusion.utils.symile_publication as symile_publication
from radfusion.data.errors import ManifestBuildError
from radfusion.data.symile_artifacts import (
    SymileBundlePaths,
    ValidatedSymileBundleReference,
)
from radfusion.data.symile_cv import ValidatedSymileCvReference
from radfusion.data.symile_preprocess import LAB_FEATURE_COLUMNS
from radfusion.data.symile_schemas import DEVELOPMENT_SPLITS
from radfusion.data.symile_source import EXPECTED_RELEASE_ASSETS, modality_asset_path
from radfusion.models.symile_tabular import (
    fit_symile_labs_logistic,
)
from radfusion.training.config import (
    load_symile_development_config,
    with_runtime,
)
from radfusion.training.execution import reused_loader_policy
from radfusion.training.symile_analysis import analyze_symile_development
from radfusion.training.symile_data import (
    SymileCxrStore,
    SymileDevelopmentCohort,
    SymileDevelopmentData,
    derive_augmentation_seed,
    derive_inner_split,
    load_symile_development,
    materialize_outer_fold,
)
from radfusion.training.symile_development import CompletedSymileFold
from radfusion.training.symile_ecg_data import SymileEcgStore
from radfusion.utils.package_identity import (
    canonical_scientific_id,
    package_scientific_config_payload,
)
from radfusion.utils.private_predictions import (
    PREDICTION_SCHEMA,
    publish_prediction_evidence,
    validate_prediction_evidence,
)
from radfusion.utils.symile_publication import (
    ValidatedDevelopmentResult,
    neural_checkpoint_document,
    publish_development_result,
    publish_fold_package,
    validate_analysis_result,
    validate_development_result,
    validate_fold_package,
)


@pytest.mark.parametrize("defect", [None, "missing", "duplicate", "target", "reference", "family"])
def test_development_repeat_oof_has_one_ordered_complete_authority(tmp_path, defect):
    config = load_symile_development_config("configs/symile_labs_logistic.yaml")
    context = {
        "fit_config": {"family": {"family_id": "labs_logistic"}},
        "task_id": config.task.task_id,
        "bundle_id": config.dataset.bundle_id,
        "split_assignment_id": config.dataset.split_assignment_id,
        "cv_assignment_id": config.dataset.cv_assignment_id,
    }
    references = []
    for index, seed in enumerate((2026, 17, 42)):
        targets = [1, 0] if defect == "target" and seed == 42 else [0, 1]
        evidence = publish_prediction_evidence(
            private_root=tmp_path,
            dataset_id="symile",
            model_package_id="fold-package-" + f"{index:064x}",
            task_id=config.task.task_id,
            bundle_id=config.dataset.bundle_id,
            split_assignment_id=config.dataset.split_assignment_id,
            cv_assignment_id=config.dataset.cv_assignment_id,
            scope="outer_fold_oof",
            repeat_seed=seed,
            outer_fold=0,
            sample_ids=["symile:001", "symile:002"],
            targets=targets,
            logits=[-1.0, 1.0],
        )
        references.append(
            {
                "prediction_id": evidence.prediction_id,
                "prediction_manifest_sha256": evidence.manifest_sha256,
                "fold_package_id": evidence.manifest["model_package_id"],
                "repeat_seed": seed,
                "outer_fold": 0,
            }
        )
    if defect == "missing":
        references.pop()
    elif defect == "duplicate":
        references.append(references[0])
    elif defect == "reference":
        references[0]["fold_package_id"] = "fold-package-" + "9" * 64
    development = ValidatedDevelopmentResult(
        tmp_path,
        {
            "dataset_id": "symile",
            "family_id": "cxr_densenet" if defect == "family" else "labs_logistic",
            "scientific_context": context,
            "fold_packages": references,
        },
        "a" * 64,
    )
    if defect is not None:
        with pytest.raises((ManifestBuildError, ValueError)):
            symile_publication.validated_development_repeat_oof(development, tmp_path)
    else:
        result = symile_publication.validated_development_repeat_oof(development, tmp_path)
        assert list(result.itertuples(index=False, name=None)) == [
            (sample, target, logit, seed)
            for sample, target, logit in (("symile:001", 0, -1.0), ("symile:002", 1, 1.0))
            for seed in (17, 42, 2026)
        ]


def test_neutral_training_seed_accepts_a_nonprotocol_integer() -> None:
    training_seed = 314_159
    epoch = 7
    sample_id = "symile:admission"
    payload = f"radfusion-symile-augmentation\0{training_seed}\0{epoch}\0{sample_id}".encode()
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)

    assert derive_augmentation_seed(training_seed, epoch, sample_id) == expected


def _bundle(tmp_path: Path) -> SymileBundlePaths:
    root = tmp_path / "manifests/symile"
    directory = root / "bundles" / ("bundle-" + "a" * 64)
    return SymileBundlePaths(
        "bundle-" + "a" * 64,
        directory,
        directory / "samples.parquet",
        directory / "labs.parquet",
        directory / "manifest.json",
        root / "CURRENT",
    )


def _data(tmp_path: Path) -> SymileDevelopmentData:
    frame = _development_frame()
    reference = ValidatedSymileCvReference({}, "b" * 64, _assignments(frame))
    return SymileDevelopmentData(
        SymileDevelopmentCohort(_bundle(tmp_path), "c" * 64, frame), reference
    )


@pytest.mark.parametrize("family", ["cxr_densenet", "cxr_labs_concat", "cxr_labs_ecg_gated"])
def test_neural_fold_delivers_inner_partitions_to_shared_training_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    config = load_symile_development_config(f"configs/symile_{family}.yaml")
    data = _data(tmp_path)
    outer = materialize_outer_fold(data, repeat_seed=42, outer_fold=2)
    inner = derive_inner_split(outer)
    runtime = cpu_runtime()
    context = symile_development.SymileFoldExecutionContext(
        data=data,
        cxr_store=SimpleNamespace(
            canonical_image=lambda split, row: np.full((224, 224), 0.5, dtype=np.float32)
        ),
        runtime=runtime,
        loader_execution=reused_loader_policy(num_workers=0, pin_memory=False),
        git_commit="a" * 40,
        dependency_lock_sha256="b" * 64,
        source_cxr_folds={},
        source_cxr_development_id=None,
        ecg_store=SimpleNamespace(
            signal=lambda split, row: np.full((12, 5000), 0.5, dtype=np.float32)
        ),
    )
    monkeypatch.setattr(
        symile_development, "_build_neural_model", lambda *args: (torch.nn.Identity(), None)
    )

    class TrainingReached(Exception):
        pass

    def train(model, training, validation, **kwargs):
        expected_keys = {"image", "target", "sample_id", "patient_id"}
        if family != "cxr_densenet":
            expected_keys.add("structured")
        if family == "cxr_labs_ecg_gated":
            expected_keys.add("ecg")
        for loader, indices in (
            (training, inner.training_indices),
            (validation, inner.validation_indices),
        ):
            assert loader.batch_size == config.neural.batch_size
            observed = []
            for batch in loader:
                assert set(batch) == expected_keys
                observed.extend(batch["sample_id"])
            assert sorted(observed) == sorted(outer.training.iloc[indices]["sample_id"])
        assert set(kwargs["input_keys"]) == expected_keys - {"target", "sample_id", "patient_id"}
        raise TrainingReached

    monkeypatch.setattr(symile_development, "fit_two_stage_binary_model", train)
    with pytest.raises(TrainingReached):
        symile_development._fit_neural_outer_fold(config, context, outer, inner, None)


def test_outer_fold_and_inner_split_are_deterministic_and_patient_isolated(
    tmp_path: Path,
) -> None:
    data = _data(tmp_path)
    outer = materialize_outer_fold(data, repeat_seed=42, outer_fold=2)
    first = derive_inner_split(outer)
    second = derive_inner_split(outer)

    assert len(outer.training) == 40
    assert len(outer.holdout) == 10
    assert not set(outer.training["subject_id"]) & set(outer.holdout["subject_id"])
    assert first.inner_seed == second.inner_seed
    assert first.inner_split_id == second.inner_split_id
    assert np.array_equal(first.training_indices, second.training_indices)
    assert first.inner_split_id.startswith("inner-split-")
    shared = outer.training.index[outer.training["subject_id"] == 10_000].tolist()
    training_roles = set(first.training_indices.tolist())
    validation_roles = set(first.validation_indices.tolist())
    assert len(shared) == 2
    assert set(shared) <= training_roles or set(shared) <= validation_roles


def test_development_loader_requests_only_train_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_symile_development_config("configs/symile_labs_logistic.yaml")
    frame = _development_frame()
    bundle = _bundle(tmp_path)
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(symile_data, "resolve_symile_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        symile_data,
        "validate_symile_bundle_reference",
        lambda *args, **kwargs: ValidatedSymileBundleReference(
            {"membership": {"split_assignment_id": config.dataset.split_assignment_id}},
            config.dataset.bundle_manifest_sha256,
        ),
    )
    monkeypatch.setattr(
        symile_data,
        "validate_symile_cv_reference",
        lambda *args, **kwargs: ValidatedSymileCvReference({}, "d" * 64, _assignments(frame)),
    )

    def read_samples(*args: object, official_splits: tuple[str, ...]) -> pd.DataFrame:
        calls.append(tuple(official_splits))
        assert "test" not in official_splits
        return frame.drop(columns=[*LAB_FEATURE_COLUMNS, "target"])

    monkeypatch.setattr(symile_data, "read_symile_samples", read_samples)
    monkeypatch.setattr(
        symile_data,
        "read_symile_labs",
        lambda *args, **kwargs: frame[["sample_id", *LAB_FEATURE_COLUMNS]],
    )

    loaded = load_symile_development(config, enforce_production_counts=False)

    assert len(loaded.frame) == len(frame)
    assert calls == [("train", "validation")]
    assert set(loaded.frame["official_split"]) == {"train", "validation"}


def _store_checksum_authority(tmp_path, monkeypatch, module, paths):
    entries = {name: "0" * 64 for name in EXPECTED_RELEASE_ASSETS}
    for path in paths.values():
        entries[path.relative_to(tmp_path).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    encoded = "".join(f"{digest} {name}\n" for name, digest in sorted(entries.items())).encode()
    (tmp_path / "SHA256SUMS.txt").write_bytes(encoded)
    reference = SimpleNamespace(
        manifest={"source": {"checksum_manifest_sha256": hashlib.sha256(encoded).hexdigest()}}
    )
    monkeypatch.setattr(
        module, "validate_symile_bundle_reference", lambda *args, **kwargs: reference
    )


def test_cxr_store_resolves_rows_and_rejects_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = np.stack([np.full((3, 320, 320), value, dtype=np.float32) for value in (0.2, 0.8)])
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)
    paths: dict[str, Path] = {}
    for split in ("train", "validation"):
        path = modality_asset_path(tmp_path, split, "cxr")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, ((raw - mean) / std).astype(np.float32))
        paths[split] = path
    _store_checksum_authority(tmp_path, monkeypatch, symile_data, paths)
    store = SymileCxrStore(_bundle(tmp_path), tmp_path)

    assert store.canonical_image("validation", 1).shape == (320, 320)
    assert np.allclose(store.canonical_image("train", 0), 0.2, atol=2e-6)
    with pytest.raises(ManifestBuildError):
        store.canonical_image("test", 0)


def _spawned_store_values(store):
    arrays = store._arrays
    read = store.canonical_image if isinstance(store, SymileCxrStore) else store.signal
    return (
        {split: read(split, 0) for split in arrays},
        {
            split: (str(array.filename), array.mode, array.flags.writeable)
            for split, array in arrays.items()
        },
    )


def _spawned_unpickle_store(encoded: bytes):
    return _spawned_store_values(pickle.loads(encoded))


def _spawned_unpickle_without_source_hashing(encoded: bytes):
    def reject_hash(descriptor: int) -> str:
        del descriptor
        raise AssertionError("spawned workers must not hash established source assets")

    symile_source._sha256_descriptor = reject_hash
    return _spawned_unpickle_store(encoded)


@pytest.mark.parametrize("modality", ["cxr", "ecg"])
def test_development_stores_reopen_readonly_maps_in_spawned_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modality: str
) -> None:
    if modality == "cxr":
        raw = np.full((2, 3, 320, 320), 0.5, dtype=np.float32)
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)
        raw = (raw - mean) / std
        module, store_type = symile_data, SymileCxrStore
    else:
        raw = np.full((2, 1, 5000, 12), 0.25, dtype=np.float32)
        module, store_type = symile_ecg_data, SymileEcgStore
    paths = {
        split: modality_asset_path(tmp_path, split, modality) for split in ("train", "validation")
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, raw)
    _store_checksum_authority(tmp_path, monkeypatch, module, paths)
    store = store_type(_bundle(tmp_path), tmp_path)
    assert len(pickle.dumps(store)) < 4096
    expected_values, expected_maps = _spawned_store_values(store)
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(_spawned_store_values, store) for _ in range(2)]
        results = [future.result(timeout=60) for future in futures]
    for values, maps in results:
        assert maps == expected_maps
        for split, value in values.items():
            np.testing.assert_array_equal(value, expected_values[split])
            assert maps[split] == (str(paths[split]), "r", False)
    changed = raw.copy()
    changed.flat[0] += np.float32(0.01)
    np.save(paths["train"], changed)
    encoded = pickle.dumps(store)
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        future = pool.submit(_spawned_unpickle_store, encoded)
        with pytest.raises(ManifestBuildError, match="changed before worker access"):
            future.result(timeout=60)


@pytest.mark.parametrize("modality", ["cxr", "ecg"])
def test_development_store_full_hashing_is_bounded_to_each_source_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modality: str
) -> None:
    if modality == "cxr":
        raw = np.full((2, 3, 320, 320), 0.5, dtype=np.float32)
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)
        raw = (raw - mean) / std
        module, store_type = symile_data, SymileCxrStore
    else:
        raw = np.full((2, 1, 5000, 12), 0.25, dtype=np.float32)
        module, store_type = symile_ecg_data, SymileEcgStore
    paths = {
        split: modality_asset_path(tmp_path, split, modality) for split in ("train", "validation")
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, raw)
    _store_checksum_authority(tmp_path, monkeypatch, module, paths)
    hashed: list[str] = []
    original = symile_source._sha256_descriptor

    def count_hash(descriptor: int) -> str:
        hashed.append(os.readlink(f"/proc/self/fd/{descriptor}"))
        return original(descriptor)

    monkeypatch.setattr(symile_source, "_sha256_descriptor", count_hash)
    store = store_type(_bundle(tmp_path), tmp_path)
    store_type(_bundle(tmp_path), tmp_path)
    pickle.loads(pickle.dumps(store))
    pickle.loads(pickle.dumps(store))

    assert hashed == [str(paths["train"]), str(paths["validation"])]
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        values, maps = pool.submit(
            _spawned_unpickle_without_source_hashing, pickle.dumps(store)
        ).result(timeout=60)
    assert set(values) == set(DEVELOPMENT_SPLITS)
    assert all(mode == "r" and not writeable for _, mode, writeable in maps.values())


def _cv_authority() -> pd.DataFrame:
    frame = _development_frame(20)
    return (
        _assignments(frame)
        .to_pandas()
        .merge(frame[["sample_id", "target"]], on="sample_id", validate="many_to_one")
    )


@pytest.fixture(autouse=True)
def _synthetic_cv_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    authority = _cv_authority()
    monkeypatch.setattr(symile_publication, "_load_cv_authority", lambda *args: authority.copy())


def test_fold_package_reconstructs_tabular_pipeline_and_validates_oof(tmp_path: Path) -> None:
    folds, _ = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0)
    first = validate_fold_package(folds[0].package.directory)

    assert folds[0].prediction.predictions.schema == PREDICTION_SCHEMA
    assert folds[0].prediction.predictions.num_rows == 4
    assert len(set(folds[0].prediction.predictions["sample_id"].to_pylist())) == 4
    assert first.manifest["selection"]["metric"] == "none"
    assert "predictions.parquet" not in {path.name for path in first.directory.iterdir()}


_MISSING_SCHEMA_VERSION = object()


def test_fold_manifest_requires_integer_schema_version_one(tmp_path: Path) -> None:
    folds, _ = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0)
    manifest_path = folds[0].package.directory / symile_publication.FOLD_MANIFEST_FILENAME
    original = manifest_path.read_bytes()
    try:
        for value in (True, 1.0, "1", None, 0, _MISSING_SCHEMA_VERSION):
            document = json.loads(original.decode("utf-8"))
            if value is _MISSING_SCHEMA_VERSION:
                document.pop("fold_package_schema_version")
            else:
                document["fold_package_schema_version"] = value
            manifest_path.write_text(json.dumps(document), encoding="utf-8")
            with pytest.raises(ManifestBuildError):
                validate_fold_package(folds[0].package.directory)
    finally:
        manifest_path.write_bytes(original)


def test_symile_neural_checkpoint_requires_integer_schema_version_one() -> None:
    original = neural_checkpoint_document(
        {"weight": torch.ones((1, 1))},
        selected_epoch=1,
        selected_stage="warmup",
        selected_validation_roc_auc=0.8,
    )
    for value in (True, 1.0, "1", None, 0, _MISSING_SCHEMA_VERSION):
        document = deepcopy(original)
        if value is _MISSING_SCHEMA_VERSION:
            document.pop("checkpoint_schema_version")
        else:
            document["checkpoint_schema_version"] = value
        with pytest.raises(ManifestBuildError):
            symile_publication._validate_neural_checkpoint(document)


def test_fold_identity_is_stable_across_reproducibility_witnesses() -> None:
    config = load_symile_development_config("configs/symile_cxr_labs_concat.yaml")
    lineage = _lineage("cxr_labs_concat")
    source = {
        "development_id": "development-" + "3" * 64,
        "fold_package_id": "fold-package-" + "4" * 64,
        "fold_manifest_sha256": "5" * 64,
    }

    def identity(candidate_lineage: dict[str, object], candidate_source: dict[str, object]) -> str:
        payload = symile_publication._fold_identity_payload(
            family="cxr_labs_concat",
            repeat_seed=17,
            outer_fold=0,
            fit_config=package_scientific_config_payload(config),
            lineage=candidate_lineage,
            inner_split=_inner_split(17, 0),
            model_state_sha256="6" * 64,
            preprocessor_state_sha256="7" * 64,
            source_cxr=candidate_source,
            selection={"selected_epoch": 3, "selected_stage": "fine_tune"},
        )
        return canonical_scientific_id("fold-package-", payload)

    baseline = identity(lineage, source)
    changed_reproduction = {
        **lineage,
        "git_commit": "changed-documentation-only-commit",
        "dependency_lock_sha256": "8" * 64,
    }
    changed_source_witness = {**source, "fold_manifest_sha256": "9" * 64}

    assert identity(changed_reproduction, source) == baseline
    assert identity(lineage, changed_source_witness) == baseline
    assert identity(lineage, {**source, "fold_package_id": "fold-package-" + "a" * 64}) != baseline
    changed_selection = symile_publication._fold_identity_payload(
        family="cxr_labs_concat",
        repeat_seed=17,
        outer_fold=0,
        fit_config=package_scientific_config_payload(config),
        lineage=lineage,
        inner_split=_inner_split(17, 0),
        model_state_sha256="6" * 64,
        preprocessor_state_sha256="7" * 64,
        source_cxr=source,
        selection={"selected_epoch": 4, "selected_stage": "fine_tune"},
    )
    assert canonical_scientific_id("fold-package-", changed_selection) != baseline

    lightgbm_config = load_symile_development_config("configs/symile_labs_lightgbm.yaml")
    lightgbm_lineage = _lineage("labs_lightgbm")
    lightgbm_payload = symile_publication._fold_identity_payload(
        family="labs_lightgbm",
        repeat_seed=17,
        outer_fold=0,
        fit_config=package_scientific_config_payload(lightgbm_config),
        lineage=lightgbm_lineage,
        inner_split=_inner_split(17, 0),
        model_state_sha256="6" * 64,
        preprocessor_state_sha256="7" * 64,
        source_cxr=None,
        selection={"best_iteration": 3},
    )
    changed_iteration = symile_publication._fold_identity_payload(
        family="labs_lightgbm",
        repeat_seed=17,
        outer_fold=0,
        fit_config=package_scientific_config_payload(lightgbm_config),
        lineage=lightgbm_lineage,
        inner_split=_inner_split(17, 0),
        model_state_sha256="6" * 64,
        preprocessor_state_sha256="7" * 64,
        source_cxr=None,
        selection={"best_iteration": 4},
    )
    assert canonical_scientific_id("fold-package-", changed_iteration) != canonical_scientific_id(
        "fold-package-", lightgbm_payload
    )


@pytest.mark.parametrize(
    ("family", "selection", "source_cxr", "inner_split_is_semantic"),
    [
        ("labs_logistic", {}, None, False),
        ("labs_lightgbm", {"best_iteration": 3}, None, True),
        (
            "cxr_densenet",
            {"selected_epoch": 3, "selected_stage": "fine_tune"},
            None,
            True,
        ),
        (
            "cxr_labs_concat",
            {"selected_epoch": 3, "selected_stage": "fine_tune"},
            {
                "development_id": "development-" + "3" * 64,
                "fold_package_id": "fold-package-" + "4" * 64,
                "fold_manifest_sha256": "5" * 64,
            },
            True,
        ),
    ],
)
def test_fold_identity_binds_inner_split_only_for_selection_families(
    family: str,
    selection: dict[str, object],
    source_cxr: dict[str, object] | None,
    inner_split_is_semantic: bool,
) -> None:
    config = load_symile_development_config(_family_config_path(family))

    def identity(inner_split_id: str) -> str:
        inner = _inner_split(17, 0)
        inner["inner_split_id"] = inner_split_id
        payload = symile_publication._fold_identity_payload(
            family=family,
            repeat_seed=17,
            outer_fold=0,
            fit_config=package_scientific_config_payload(config),
            lineage=_lineage(family),
            inner_split=inner,
            model_state_sha256="6" * 64,
            preprocessor_state_sha256=None if family == "cxr_densenet" else "7" * 64,
            source_cxr=source_cxr,
            selection=selection,
        )
        return canonical_scientific_id("fold-package-", payload)

    baseline = identity("inner-split-" + "a" * 64)
    changed = identity("inner-split-" + "b" * 64)
    assert (changed != baseline) is inner_split_is_semantic


def test_cxr_fold_identity_uses_pretrained_scientific_projection() -> None:
    config = load_symile_development_config("configs/symile_cxr_densenet.yaml")
    lineage = _lineage("cxr_densenet")

    def identity(candidate_lineage: dict[str, object]) -> str:
        payload = symile_publication._fold_identity_payload(
            family="cxr_densenet",
            repeat_seed=17,
            outer_fold=0,
            fit_config=package_scientific_config_payload(config),
            lineage=candidate_lineage,
            inner_split=_inner_split(17, 0),
            model_state_sha256="6" * 64,
            preprocessor_state_sha256=None,
            source_cxr=None,
            selection={"selected_epoch": 3, "selected_stage": "fine_tune"},
        )
        return canonical_scientific_id("fold-package-", payload)

    baseline = identity(lineage)
    materialization_changed = deepcopy(lineage)
    materialization_changed["pretrained_weight"].update(
        {"cache_filename": "alternate.pt", "byte_size": 200}
    )
    assert identity(materialization_changed) == baseline
    for field, value in (
        ("declared_name", "changed-weight"),
        ("stable_identifier", "https://example.invalid/changed.pt"),
        ("sha256", "9" * 64),
    ):
        scientific_changed = deepcopy(lineage)
        scientific_changed["pretrained_weight"][field] = value
        assert identity(scientific_changed) != baseline


def test_fold_identity_tracks_package_scoped_config(
    tmp_path: Path,
) -> None:
    folds, _ = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0)
    manifest = deepcopy(folds[0].package.manifest)
    baseline = canonical_scientific_id(
        "fold-package-", symile_publication._fold_payload_from_manifest(manifest)
    )
    manifest["config"]["config_source_sha256"] = "e" * 64
    assert (
        canonical_scientific_id(
            "fold-package-", symile_publication._fold_payload_from_manifest(manifest)
        )
        == baseline
    )
    manifest["config"]["config_semantic_sha256"] = "f" * 64
    assert (
        canonical_scientific_id(
            "fold-package-", symile_publication._fold_payload_from_manifest(manifest)
        )
        == baseline
    )
    manifest["config"]["fit_config"]["training"]["parameters"]["C"] = 2.0
    assert (
        canonical_scientific_id(
            "fold-package-", symile_publication._fold_payload_from_manifest(manifest)
        )
        != baseline
    )


def test_development_rejects_one_mixed_fold_fit_config(tmp_path: Path) -> None:
    folds, semantic_hash = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0)
    changed_manifest = deepcopy(folds[-1].package.manifest)
    changed_manifest["config"]["fit_config"]["training"]["parameters"]["C"] = 2.0
    mixed_package = replace(folds[-1].package, manifest=changed_manifest)
    with pytest.raises(ManifestBuildError, match="mixed scientific fit configs"):
        publish_development_result(
            report_root=tmp_path / "reports/symile/development",
            model_root=tmp_path / "models/symile/development",
            prediction_root=tmp_path / "private",
            family="labs_logistic",
            config_semantic_sha256=semantic_hash,
            folds=[item.package for item in folds[:-1]] + [mixed_package],
            predictions=[item.prediction for item in folds],
        )


def test_development_rejects_one_mixed_complete_scientific_config(tmp_path: Path) -> None:
    folds, semantic_hash = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0)
    changed_manifest = deepcopy(folds[-1].package.manifest)
    changed_manifest["config"]["config_semantic_sha256"] = "e" * 64
    mixed_package = replace(folds[-1].package, manifest=changed_manifest)
    with pytest.raises(ManifestBuildError, match="mixed scientific configs"):
        publish_development_result(
            report_root=tmp_path / "reports/symile/development",
            model_root=tmp_path / "models/symile/development",
            prediction_root=tmp_path / "private",
            family="labs_logistic",
            config_semantic_sha256=semantic_hash,
            folds=[item.package for item in folds[:-1]] + [mixed_package],
            predictions=[item.prediction for item in folds],
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("dataset_id", "rsna"),
        ("task_id", "other-task"),
        ("bundle_id", "bundle-" + "9" * 64),
        ("split_assignment_id", "split-assignment-" + "9" * 64),
        ("cv_assignment_id", "cv-assignment-" + "9" * 64),
        ("model_package_id", "fold-package-" + "9" * 64),
        ("repeat_seed", 42),
        ("outer_fold", 1),
        ("scope", "test"),
    ],
)
def test_fold_prediction_pair_requires_exact_scientific_lineage(
    tmp_path: Path, field: str, replacement: object
) -> None:
    folds, _ = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0)
    completed = folds[0]
    changed = replace(
        completed.prediction,
        manifest={**completed.prediction.manifest, field: replacement},
    )

    with pytest.raises(ManifestBuildError, match="scientific lineage"):
        symile_publication._validate_fold_prediction_pair(
            completed.package, changed, _cv_authority()
        )


@pytest.mark.parametrize("field", ["inner_seed", "policy.inner_seed"])
def test_fold_manifest_rejects_noncanonical_inner_seed(tmp_path: Path, field: str) -> None:
    folds, _ = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0)
    manifest = deepcopy(folds[0].package.manifest)
    if field == "inner_seed":
        manifest["inner_split"]["inner_seed"] += 1
    else:
        manifest["inner_split"]["policy"]["inner_seed"] += 1

    with pytest.raises(ManifestBuildError, match="inner split policy"):
        symile_publication._validate_fold_manifest(manifest)


def test_fold_prediction_pair_accepts_exact_cv_authority(tmp_path: Path) -> None:
    folds, _ = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0)
    symile_publication._validate_fold_prediction_pair(
        folds[0].package, folds[0].prediction, _cv_authority()
    )


@pytest.mark.parametrize("mutation", ["wrong_fold", "target", "fabricated"])
def test_fold_prediction_pair_rejects_rows_outside_cv_authority(
    tmp_path: Path, mutation: str
) -> None:
    folds, _ = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0)
    completed = folds[0]
    frame = completed.prediction.predictions.to_pandas()
    if mutation == "wrong_fold":
        frame.loc[0, "sample_id"] = "symile:001"
    elif mutation == "target":
        frame.loc[0, "target"] = 1 - int(frame.loc[0, "target"])
    else:
        frame.loc[0, "sample_id"] = "symile:fabricated"
    changed = replace(
        completed.prediction,
        predictions=pa.Table.from_pandas(frame, schema=PREDICTION_SCHEMA, preserve_index=False),
    )

    with pytest.raises(ManifestBuildError, match="immutable CV authority"):
        symile_publication._validate_fold_prediction_pair(
            completed.package, changed, _cv_authority()
        )


def test_development_validation_requires_fold_modalities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(symile_publication, "DEVELOPMENT_COUNT", 20)
    folds, semantic_hash = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0)
    result = publish_development_result(
        report_root=tmp_path / "reports/symile/development",
        model_root=tmp_path / "models/symile/development",
        prediction_root=tmp_path / "private",
        family="labs_logistic",
        config_semantic_sha256=semantic_hash,
        folds=[item.package for item in folds],
        predictions=[item.prediction for item in folds],
    )
    manifest_path = result.directory / symile_publication.DEVELOPMENT_MANIFEST_FILENAME
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["modalities"] = ["cxr"]
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestBuildError, match="modalities"):
        validate_development_result(
            result.directory,
            model_root=tmp_path / "models/symile/development",
            prediction_root=tmp_path / "private",
        )


def test_development_validation_uses_canonical_default_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(symile_publication, "DEVELOPMENT_COUNT", 20)
    folds, semantic_hash = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0)
    result = publish_development_result(
        report_root=tmp_path / "reports/symile/development",
        model_root=tmp_path / "models/symile/development",
        prediction_root=tmp_path / "private",
        family="labs_logistic",
        config_semantic_sha256=semantic_hash,
        folds=[item.package for item in folds],
        predictions=[item.prediction for item in folds],
    )
    monkeypatch.chdir(tmp_path)

    validated = validate_development_result(result.directory)

    assert validated.manifest["development_id"] == result.manifest["development_id"]


def test_fusion_fold_resolves_matching_source_cxr_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(symile_publication, "_validate_neural_reconstruction", lambda *args: None)
    source_folds, _ = _publish_family_folds(tmp_path, "cxr_densenet", offset=0.0)
    fusion_folds, _ = _publish_family_folds(
        tmp_path,
        "cxr_labs_concat",
        offset=0.1,
        source_cxr_folds=source_folds,
        source_cxr_development_id="development-" + "8" * 64,
    )
    fusion = fusion_folds[0].package
    symile_publication._validate_source_cxr_fold(fusion.directory, fusion.manifest)

    missing = deepcopy(fusion.manifest)
    missing["source_cxr"]["fold_package_id"] = "fold-package-" + "9" * 64
    with pytest.raises(ManifestBuildError, match="missing or invalid"):
        symile_publication._validate_source_cxr_fold(fusion.directory, missing)

    incompatible_source = source_folds[1].package
    incompatible = deepcopy(fusion.manifest)
    incompatible["source_cxr"].update(
        fold_package_id=incompatible_source.manifest["fold_package_id"],
        fold_manifest_sha256=incompatible_source.manifest_sha256,
    )
    with pytest.raises(ManifestBuildError, match="incompatible"):
        symile_publication._validate_source_cxr_fold(fusion.directory, incompatible)


def test_fold_publication_rejects_wrong_model_malformed_state_and_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logistic = load_symile_development_config("configs/symile_labs_logistic.yaml")
    wrong_model = LogisticRegression().fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    with pytest.raises(ManifestBuildError, match="exact fitted pipeline"):
        publish_fold_package(
            model_root=tmp_path / "wrong-tabular",
            family="labs_logistic",
            repeat_seed=17,
            outer_fold=0,
            config_bytes=logistic.source_bytes,
            config_sha256=logistic.config_source_sha256,
            config_semantic_sha256=logistic.config_semantic_sha256,
            lineage=_lineage("labs_logistic"),
            inner_split=_inner_split(17, 0),
            selection={"metric": "none", "selected_epoch": None, "best_iteration": None},
            model=wrong_model,
            operational={"mlflow_run_id": "run-wrong", "runtime_provenance": None},
        )

    cxr = load_symile_development_config("configs/symile_cxr_densenet.yaml")
    selection = {
        "metric": "roc_auc",
        "selected_epoch": 1,
        "best_iteration": None,
        "selected_stage": "warmup",
        "selected_validation_metric": 0.8,
    }
    malformed = neural_checkpoint_document(
        {"unexpected.weight": torch.ones((1, 1))},
        selected_epoch=1,
        selected_stage="warmup",
        selected_validation_roc_auc=0.8,
    )
    with pytest.raises(ManifestBuildError, match="cannot reconstruct exactly"):
        publish_fold_package(
            model_root=tmp_path / "malformed-neural",
            family="cxr_densenet",
            repeat_seed=17,
            outer_fold=0,
            config_bytes=cxr.source_bytes,
            config_sha256=cxr.config_source_sha256,
            config_semantic_sha256=cxr.config_semantic_sha256,
            lineage=_lineage("cxr_densenet"),
            inner_split=_inner_split(17, 0),
            selection=selection,
            model=malformed,
            training_history=[
                {
                    "epoch": 1,
                    "stage": "warmup",
                    "training_loss": 0.5,
                    "validation_metric": 0.8,
                    "encoder_learning_rate": None,
                    "head_learning_rate": 1e-3,
                }
            ],
            operational={"mlflow_run_id": "run-malformed", "runtime_provenance": {}},
        )

    monkeypatch.setattr(symile_publication, "_validate_neural_reconstruction", lambda *args: None)
    history_selection = {**selection, "selected_epoch": 2}
    history_checkpoint = neural_checkpoint_document(
        {"unexpected.weight": torch.ones((1, 1))},
        selected_epoch=2,
        selected_stage="warmup",
        selected_validation_roc_auc=0.8,
    )
    with pytest.raises(ManifestBuildError, match="omits configured warmup epochs"):
        publish_fold_package(
            model_root=tmp_path / "bad-history",
            family="cxr_densenet",
            repeat_seed=17,
            outer_fold=0,
            config_bytes=cxr.source_bytes,
            config_sha256=cxr.config_source_sha256,
            config_semantic_sha256=cxr.config_semantic_sha256,
            lineage=_lineage("cxr_densenet"),
            inner_split=_inner_split(17, 0),
            selection=history_selection,
            model=history_checkpoint,
            training_history=[
                {
                    "epoch": 1,
                    "stage": "warmup",
                    "training_loss": 0.5,
                    "validation_metric": 0.8,
                    "encoder_learning_rate": None,
                    "head_learning_rate": 1e-3,
                }
            ],
            operational={"mlflow_run_id": "run-history", "runtime_provenance": {}},
        )


def test_neural_selection_supports_configured_lifecycles_beyond_thirty_epochs() -> None:
    config = load_symile_development_config("configs/symile_cxr_densenet.yaml")
    assert config.neural is not None
    extended = replace(config, neural=replace(config.neural, fine_tune_epochs=40))
    selection = {
        "metric": "roc_auc",
        "selected_epoch": 31,
        "best_iteration": None,
        "selected_stage": "fine_tune",
        "selected_validation_metric": 0.8,
    }
    checkpoint = neural_checkpoint_document(
        {"weight": torch.ones(1)},
        selected_epoch=31,
        selected_stage="fine_tune",
        selected_validation_roc_auc=0.8,
    )
    history = [
        {
            "epoch": epoch,
            "stage": "warmup" if epoch <= extended.neural.warmup_epochs else "fine_tune",
            "training_loss": 0.5,
            "validation_metric": (
                0.8
                if epoch == 31
                else 0.1 + 0.01 * ((epoch - 3) // 4 + 1)
                if epoch >= 3 and (epoch - 3) % 4 == 0
                else 0.1
            ),
            "encoder_learning_rate": (None if epoch <= extended.neural.warmup_epochs else 1e-5),
            "head_learning_rate": 1e-3,
        }
        for epoch in range(1, 37)
    ]

    symile_publication._validate_neural_checkpoint(checkpoint)
    symile_publication._validate_training_history(history, selection, extended)
    with pytest.raises(ManifestBuildError, match="exceeds the configured lifecycle"):
        symile_publication._validate_training_history(history, selection, config)


def test_training_history_rejects_declared_winner_inconsistent_with_replay() -> None:
    config = load_symile_development_config("configs/symile_cxr_densenet.yaml")
    history = _neural_history(config, 3)
    selection = {
        "metric": "roc_auc",
        "selected_epoch": 2,
        "best_iteration": None,
        "selected_stage": "warmup",
        "selected_validation_metric": 0.7,
    }

    with pytest.raises(ManifestBuildError, match="differs from deterministic history"):
        symile_publication._validate_training_history(history, selection, config)


def test_training_history_rejects_unjustified_premature_termination() -> None:
    config = load_symile_development_config("configs/symile_cxr_densenet.yaml")
    history = _neural_history(config, 3)[:-1]
    selection = {
        "metric": "roc_auc",
        "selected_epoch": 3,
        "best_iteration": None,
        "selected_stage": "fine_tune",
        "selected_validation_metric": 0.8,
    }

    with pytest.raises(ManifestBuildError, match="terminates before"):
        symile_publication._validate_training_history(history, selection, config)


def test_training_history_accepts_canonical_early_stopping() -> None:
    config = load_symile_development_config("configs/symile_cxr_densenet.yaml")
    history = _neural_history(config, 3)
    selection = {
        "metric": "roc_auc",
        "selected_epoch": 3,
        "best_iteration": None,
        "selected_stage": "fine_tune",
        "selected_validation_metric": 0.8,
    }

    symile_publication._validate_training_history(history, selection, config)


def test_standalone_analysis_forwards_custom_manifest_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development_ids = [f"development-{index:064x}" for index in range(6)]
    families = (
        "labs_logistic",
        "labs_lightgbm",
        "cxr_densenet",
        "cxr_labs_concat",
        "cxr_labs_gated",
        "cxr_labs_gated_no_observedness",
    )
    manifest_root = tmp_path / "custom-manifests"
    validated_roots: list[str | Path] = []

    def validate(directory: Path, **kwargs: object) -> ValidatedDevelopmentResult:
        validated_roots.append(kwargs["manifest_root"])
        development_id = str(kwargs["expected_development_id"])
        family = families[development_ids.index(development_id)]
        return ValidatedDevelopmentResult(
            directory,
            {"development_id": development_id, "family_id": family},
            "a" * 64,
        )

    published: dict[str, object] = {}

    def publish(**kwargs: object) -> tuple[str, Path]:
        published.update(kwargs)
        return "analysis-" + "b" * 64, tmp_path / "analysis"

    monkeypatch.setattr(symile_analysis, "validate_development_result", validate)
    monkeypatch.setattr(symile_analysis, "publish_analysis_result", publish)

    symile_analysis.analyze_symile_development(
        development_ids,
        report_root=tmp_path / "reports",
        model_root=tmp_path / "models",
        prediction_root=tmp_path / "private",
        manifest_root=manifest_root,
    )

    assert validated_roots == [manifest_root] * 6
    assert published["manifest_root"] == manifest_root


@pytest.fixture(scope="class")
def published_development_scenario(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("published-development")
    patch = pytest.MonkeyPatch()
    patch.setattr(symile_publication, "DEVELOPMENT_COUNT", 20)
    patch.setattr(symile_publication, "_validate_neural_reconstruction", lambda *args: None)
    authority = _cv_authority()
    patch.setattr(symile_publication, "_load_cv_authority", lambda *args: authority.copy())
    developments = []
    family_folds: dict[str, list[CompletedSymileFold]] = {}
    cxr_folds: list[CompletedSymileFold] | None = None
    cxr_development_id: str | None = None
    for family_index, family in enumerate(
        (
            "labs_logistic",
            "labs_lightgbm",
            "cxr_densenet",
            "cxr_labs_concat",
            "cxr_labs_gated",
            "cxr_labs_gated_no_observedness",
        )
    ):
        folds, semantic_hash = _publish_family_folds(
            tmp_path,
            family,
            offset=family_index * 0.05,
            source_cxr_folds=cxr_folds,
            source_cxr_development_id=cxr_development_id,
        )
        family_folds[family] = folds
        canonical_family = str(folds[0].package.manifest["family_id"])
        result = publish_development_result(
            report_root=tmp_path / "reports/symile/development",
            model_root=tmp_path / "models/symile/development",
            prediction_root=tmp_path / "private",
            family=canonical_family,
            config_semantic_sha256=semantic_hash,
            folds=[item.package for item in folds],
            predictions=[item.prediction for item in folds],
        )
        developments.append(result)
        if family == "cxr_densenet":
            cxr_folds = folds
            cxr_development_id = str(result.manifest["development_id"])
    analysis_id, analysis_directory = analyze_symile_development(
        [str(result.manifest["development_id"]) for result in reversed(developments)],
        report_root=tmp_path / "reports/symile/development",
        model_root=tmp_path / "models/symile/development",
        prediction_root=tmp_path / "private",
    )
    try:
        yield {
            "root": tmp_path,
            "developments": developments,
            "family_folds": family_folds,
            "analysis_id": analysis_id,
            "analysis_directory": analysis_directory,
        }
    finally:
        patch.undo()


class TestPublishedDevelopmentScenario:
    def test_cross_family_lineage_ignores_logistic_inner_split_only(
        self, published_development_scenario
    ) -> None:
        scenario = published_development_scenario
        families = {
            family: {
                (item.package.manifest["repeat_seed"], item.package.manifest["outer_fold"]): (
                    item.package
                )
                for item in folds
            }
            for family, folds in scenario["family_folds"].items()
        }
        family_ids = {
            result.manifest["family_id"]: result.manifest["development_id"]
            for result in scenario["developments"]
        }
        coordinate = (17, 0)
        logistic = families["labs_logistic"][coordinate]
        logistic_manifest = deepcopy(logistic.manifest)
        logistic_manifest["inner_split"]["inner_split_id"] = "inner-split-" + "9" * 64
        families["labs_logistic"][coordinate] = replace(logistic, manifest=logistic_manifest)
        symile_publication._validate_cross_family_fold_lineage(families, family_ids)

        lightgbm = families["labs_lightgbm"][coordinate]
        lightgbm_manifest = deepcopy(lightgbm.manifest)
        lightgbm_manifest["inner_split"]["inner_split_id"] = "inner-split-" + "8" * 64
        families["labs_lightgbm"][coordinate] = replace(lightgbm, manifest=lightgbm_manifest)
        with pytest.raises(ManifestBuildError, match="different inner splits"):
            symile_publication._validate_cross_family_fold_lineage(families, family_ids)

    def test_development_completeness_and_final_training_budgets(
        self, published_development_scenario
    ) -> None:
        scenario = published_development_scenario
        tmp_path = scenario["root"]
        developments = scenario["developments"]
        folds = scenario["family_folds"]["labs_logistic"]
        semantic_hash = developments[0].manifest["config_semantic_sha256"]
        with pytest.raises(ManifestBuildError, match="exact 3 x 5 folds"):
            publish_development_result(
                report_root=tmp_path / "reports/symile/development",
                model_root=tmp_path / "models/symile/development",
                prediction_root=tmp_path / "private",
                family="labs_logistic",
                config_semantic_sha256=semantic_hash,
                folds=[item.package for item in [*folds[:-1], folds[-2]]],
                predictions=[item.prediction for item in [*folds[:-1], folds[-2]]],
            )
        for result in developments:
            selected = result.manifest["selection_budget_values"]
            budget = result.manifest["final_training_budget"]
            if result.manifest["family_id"] == "labs_logistic":
                assert selected is None and budget is None
            else:
                assert isinstance(selected, list) and len(selected) == 15
                assert budget == int(np.median(selected))
            validate_development_result(
                result.directory,
                model_root=tmp_path / "models/symile/development",
                prediction_root=tmp_path / "private",
            )

    def test_cross_family_analysis_alignment_and_derived_effects(
        self, published_development_scenario
    ) -> None:
        scenario = published_development_scenario
        tmp_path = scenario["root"]
        analysis = validate_analysis_result(
            scenario["analysis_directory"],
            report_root=tmp_path / "reports/symile/development",
            model_root=tmp_path / "models/symile/development",
            prediction_root=tmp_path / "private",
            expected_analysis_id=scenario["analysis_id"],
        )
        assert set(analysis["family_development_ids"]) == {
            "labs_logistic",
            "labs_lightgbm",
            "cxr_densenet",
            "cxr_labs_concat",
            "cxr_labs_gated",
            "cxr_labs_gated_no_observedness",
        }
        assert set(analysis["paired_effects"]) == {
            "concat_minus_cxr",
            "gated_minus_cxr",
            "gated_minus_concat",
        }
        assert set(analysis["observedness_ablation"]) == {"repeat_effects", "ensemble_effect"}
        assert set(analysis["ensemble_metrics"]) == {
            "cxr_densenet",
            "cxr_labs_concat",
            "cxr_labs_gated",
            "cxr_labs_gated_no_observedness",
        }

    def test_public_summaries_are_private_and_corruption_checked(
        self, published_development_scenario
    ) -> None:
        scenario = published_development_scenario
        tmp_path = scenario["root"]
        developments = scenario["developments"]
        analysis_directory = scenario["analysis_directory"]
        for result in developments:
            assert "symile:" not in (result.directory / "summary.md").read_text(encoding="utf-8")
        assert "symile:" not in (analysis_directory / "summary.md").read_text(encoding="utf-8")

        development_manifest_path = (
            developments[0].directory / symile_publication.DEVELOPMENT_MANIFEST_FILENAME
        )
        original_development_manifest = development_manifest_path.read_bytes()
        try:
            for value in (True, 1.0, "1", None, 0, _MISSING_SCHEMA_VERSION):
                document = json.loads(original_development_manifest.decode("utf-8"))
                if value is _MISSING_SCHEMA_VERSION:
                    document.pop("development_schema_version")
                else:
                    document["development_schema_version"] = value
                development_manifest_path.write_text(json.dumps(document), encoding="utf-8")
                with pytest.raises(ManifestBuildError):
                    validate_development_result(
                        developments[0].directory,
                        model_root=tmp_path / "models/symile/development",
                        prediction_root=tmp_path / "private",
                    )
        finally:
            development_manifest_path.write_bytes(original_development_manifest)

        analysis_manifest_path = analysis_directory / symile_publication.ANALYSIS_MANIFEST_FILENAME
        original_analysis_manifest = analysis_manifest_path.read_bytes()
        try:
            for value in (True, 1.0, "1", None, 0, _MISSING_SCHEMA_VERSION):
                document = json.loads(original_analysis_manifest.decode("utf-8"))
                if value is _MISSING_SCHEMA_VERSION:
                    document.pop("analysis_schema_version")
                else:
                    document["analysis_schema_version"] = value
                analysis_manifest_path.write_text(json.dumps(document), encoding="utf-8")
                with pytest.raises(ManifestBuildError):
                    validate_analysis_result(
                        analysis_directory,
                        report_root=tmp_path / "reports/symile/development",
                        model_root=tmp_path / "models/symile/development",
                        prediction_root=tmp_path / "private",
                    )
        finally:
            analysis_manifest_path.write_bytes(original_analysis_manifest)

        corrupted_analysis = json.loads(original_analysis_manifest.decode("utf-8"))
        corrupted_analysis["paired_effects"]["gated_minus_cxr"]["17"]["roc_auc"] += 0.01
        try:
            analysis_manifest_path.write_text(json.dumps(corrupted_analysis), encoding="utf-8")
            with pytest.raises(ManifestBuildError, match="paired_effects differs"):
                validate_analysis_result(
                    analysis_directory,
                    report_root=tmp_path / "reports/symile/development",
                    model_root=tmp_path / "models/symile/development",
                    prediction_root=tmp_path / "private",
                )
        finally:
            analysis_manifest_path.write_bytes(original_analysis_manifest)

        analysis_summary_path = analysis_directory / "summary.md"
        original_analysis_summary = analysis_summary_path.read_bytes()
        try:
            analysis_summary_path.write_text("tampered\n", encoding="utf-8")
            with pytest.raises(ManifestBuildError, match="summary differs"):
                validate_analysis_result(
                    analysis_directory,
                    report_root=tmp_path / "reports/symile/development",
                    model_root=tmp_path / "models/symile/development",
                    prediction_root=tmp_path / "private",
                )
        finally:
            analysis_summary_path.write_bytes(original_analysis_summary)
        first_development = developments[0]
        development_summary_path = first_development.directory / "summary.md"
        original_development_summary = development_summary_path.read_bytes()
        try:
            development_summary_path.write_text("tampered\n", encoding="utf-8")
            with pytest.raises(ManifestBuildError, match="summary differs"):
                validate_development_result(
                    first_development.directory,
                    model_root=tmp_path / "models/symile/development",
                    prediction_root=tmp_path / "private",
                )
        finally:
            development_summary_path.write_bytes(original_development_summary)


def test_operational_completion_failure_preserves_fold_package_and_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_symile_development_config("configs/symile_labs_logistic.yaml")
    config = with_runtime(
        config,
        model_directory=tmp_path / "models/symile/development",
        report_directory=tmp_path / "reports/symile/development",
        private_output_directory=tmp_path / "private",
    )
    data = _data(tmp_path)
    data = SymileDevelopmentData(
        SymileDevelopmentCohort(
            data.bundle,
            config.dataset.bundle_manifest_sha256,
            data.frame,
        ),
        data.cv_reference,
    )
    context = symile_development.SymileFoldExecutionContext(
        data,
        None,
        None,
        None,
        "f" * 40,
        "a" * 64,
        {},
        None,
    )

    run_arguments: dict[str, object] = {}

    @contextmanager
    def fake_run(**kwargs: object):
        run_arguments.update(kwargs)
        yield "run-id"

    monkeypatch.setattr(symile_development, "tracked_run", fake_run)
    monkeypatch.setattr(symile_development, "log_source_config", lambda *args: None)
    monkeypatch.setattr(symile_development.mlflow, "log_metrics", lambda *args: None)
    monkeypatch.setattr(
        symile_development.mlflow,
        "set_tags",
        lambda *args: (_ for _ in ()).throw(OSError("ledger failure")),
    )

    def fake_fit(*args: object) -> dict[str, object]:
        outer = args[2]
        assert isinstance(outer, symile_data.SymileOuterFold)
        fitted = fit_symile_labs_logistic(
            outer.training[list(LAB_FEATURE_COLUMNS)],
            outer.training["target"].to_numpy(dtype=np.int8),
            parameters=config.training.parameters,
            selection_metric=config.training.selection_metric,
            lab_policy=str(config.preprocessing["lab_policy"]),
            training_seed=outer.repeat_seed,
        )
        return {
            "model": fitted.pipeline,
            "logits": np.resize(np.array([-1.0, 1.0]), len(outer.holdout)),
            "selection": {"metric": "none", "selected_epoch": None, "best_iteration": None},
        }

    monkeypatch.setattr(symile_development, "_fit_outer_fold", fake_fit)

    with pytest.raises(OSError):
        symile_development.execute_symile_outer_fold(
            config,
            context,
            repeat_seed=17,
            outer_fold=0,
        )
    packages = list((tmp_path / "models/symile/development/packages").glob("fold-package-*"))
    predictions = list((tmp_path / "private/predictions/symile/oof").glob("prediction-*"))
    assert len(packages) == len(predictions) == 1
    package = validate_fold_package(packages[0])
    prediction = validate_prediction_evidence(
        predictions[0],
        expected_model_package_id=package.manifest["fold_package_id"],
    )
    assert prediction.manifest["repeat_seed"] == package.manifest["repeat_seed"]
    assert prediction.manifest["outer_fold"] == package.manifest["outer_fold"]
    tags = run_arguments["tags"]
    assert isinstance(tags, dict)
    assert tags["family_id"] == "labs_logistic"
    assert tags["modalities"] == '["labs"]'
    assert tags["repeat_seed"] == "17"
    assert tags["run_complete"] == "false"
    assert set(tags) == {
        "run_kind",
        "evaluation_scope",
        "dataset_id",
        "task_id",
        "label_policy_version",
        "family_id",
        "modalities",
        "bundle_id",
        "bundle_manifest_sha256",
        "split_assignment_id",
        "cv_assignment_id",
        "repeat_seed",
        "outer_fold",
        "inner_split_id",
        "config_source_sha256",
        "config_semantic_sha256",
        "git_commit",
        "git_dirty",
        "dependency_lock_sha256",
        "run_complete",
    }


def test_public_development_cli_rejects_separate_ecg_family(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = SimpleNamespace(family=SimpleNamespace(family_id="cxr_labs_ecg_gated"))
    calls: list[str] = []
    monkeypatch.setattr(symile_development, "load_symile_development_config", lambda path: config)
    monkeypatch.setattr(
        symile_development,
        "run_symile_development",
        lambda *args, **kwargs: calls.append("development"),
    )

    assert symile_development.main(["--config", "unused.yaml"]) == 1
    assert calls == []
    assert "Symile development failed: ConfigError" in capsys.readouterr().err
