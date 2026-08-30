from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import pickle
import platform
import shutil
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
from symile_campaign_test_support import _final_provenance
from symile_development_test_support import (
    _assignments,
    _development_frame,
    _family_config_path,
    _inner_split,
    _lineage,
    _neural_history,
    _publish_family_folds,
)
from symile_lifecycle_test_support import prospective_symile_mutation

import beyondcxr.data.symile_source as symile_source
import beyondcxr.training.symile_data as symile_data
import beyondcxr.training.symile_development as symile_development
import beyondcxr.training.symile_development_reports as symile_development_reports
import beyondcxr.training.symile_ecg_data as symile_ecg_data
import beyondcxr.utils.symile_publication as symile_publication
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_artifacts import (
    SymileBundlePaths,
)
from beyondcxr.data.symile_cv import (
    ValidatedSymileCvReference,
)
from beyondcxr.data.symile_preprocess import LAB_FEATURE_COLUMNS
from beyondcxr.data.symile_schemas import DEVELOPMENT_SPLITS, REPEAT_SEEDS
from beyondcxr.data.symile_source import (
    EXPECTED_RELEASE_ASSETS,
    development_modality_asset_path,
)
from beyondcxr.models.symile_tabular import (
    fit_symile_labs_logistic,
)
from beyondcxr.training.config import (
    load_symile_development_config,
    with_runtime,
)
from beyondcxr.training.execution import reused_loader_policy
from beyondcxr.training.symile_data import (
    SymileCxrStore,
    SymileDevelopmentCohort,
    SymileDevelopmentData,
    derive_augmentation_seed,
    derive_inner_split,
    load_symile_development,
    materialize_outer_fold,
)
from beyondcxr.training.symile_ecg_data import SymileEcgStore
from beyondcxr.utils.package_identity import package_scientific_config_payload
from beyondcxr.utils.private_predictions import (
    PREDICTION_SCHEMA,
    validate_prediction_evidence,
)
from beyondcxr.utils.private_predictions import (
    publish_prediction_evidence as _publish_prediction_evidence,
)
from beyondcxr.utils.semantic_identity import canonical_scientific_id
from beyondcxr.utils.symile_publication import (
    ValidatedDevelopmentResult,
    family_development_identity_payload,
    neural_checkpoint_document,
    validate_fold_package,
)
from beyondcxr.utils.symile_publication import (
    publish_development_result as _publish_development_result,
)
from beyondcxr.utils.symile_publication import (
    publish_fold_package as _publish_fold_package,
)


def test_family_development_identity_excludes_derived_claims() -> None:
    folds = [
        {
            "repeat_seed": seed,
            "outer_fold": fold,
            "fold_package_id": "fold-package-" + f"{seed * 10 + fold:064x}",
            "prediction_id": "prediction-" + f"{seed * 10 + fold + 100:064x}",
        }
        for seed in (0, 1, 2)
        for fold in range(5)
    ]
    payload = family_development_identity_payload(
        dataset_id="symile",
        task_id="pneumonia_strict",
        family_id="labs_lightgbm",
        modalities=["labs"],
        scientific_context={
            "bundle_id": "bundle-" + "a" * 64,
            "cv_assignment_id": "cv-assignment-" + "b" * 64,
            "task_id": "pneumonia_strict",
            "fit_config": {"family": {"family_id": "labs_lightgbm"}},
        },
        fold_packages=folds,
        development_policy=symile_publication.SYMILE_FAMILY_DEVELOPMENT_POLICY,
    )
    assert (
        not {
            "repeat_metrics",
            "selection_budget_values",
            "final_training_budget",
            "completeness",
        }
        & payload.keys()
    )
    first = canonical_scientific_id("development-", payload, domain="development", schema_version=1)
    reordered = {
        **payload,
        "development_policy": dict(reversed(payload["development_policy"].items())),
    }
    second = canonical_scientific_id(
        "development-", reordered, domain="development", schema_version=1
    )
    assert first == second
    changed = {**payload, "package_prediction_pairs": [*payload["package_prediction_pairs"]]}
    changed["package_prediction_pairs"][0] = [
        0,
        0,
        "fold-package-" + "f" * 64,
        "prediction-" + "e" * 64,
    ]
    assert first != canonical_scientific_id(
        "development-", changed, domain="development", schema_version=1
    )


def publish_development_result(**kwargs):
    root = Path(kwargs["report_root"]).parents[2]
    kwargs.setdefault("manifest_root", root / "data/manifests")
    with pytest.MonkeyPatch.context() as monkeypatch:
        with prospective_symile_mutation(root, monkeypatch) as mutation:
            return _publish_development_result(
                prospective_mutation=mutation,
                **kwargs,
            )


def publish_fold_package(**kwargs):
    root = Path(kwargs["model_root"]).parent
    with pytest.MonkeyPatch.context() as monkeypatch:
        with prospective_symile_mutation(root, monkeypatch) as mutation:
            return _publish_fold_package(
                prospective_mutation=mutation,
                **kwargs,
            )


def publish_prediction_evidence(**kwargs):
    private_root = Path(kwargs["private_root"])
    root = private_root.parent if private_root.name == "private" else private_root
    with pytest.MonkeyPatch.context() as monkeypatch:
        with prospective_symile_mutation(root, monkeypatch) as mutation:
            return _publish_prediction_evidence(
                prospective_mutation=mutation,
                **kwargs,
            )


@pytest.mark.parametrize("defect", [None, "missing", "duplicate", "target", "reference", "family"])
def test_development_repeat_oof_has_one_ordered_complete_authority(tmp_path, defect):
    config = load_symile_development_config(
        "configs/symile_labs_logistic.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
    context = {
        "fit_config": {"family": {"family_id": "labs_logistic"}},
        "task_id": config.task.task_id,
        "bundle_id": config.dataset.bundle_id,
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
    payload = f"beyondcxr-symile-augmentation\0{training_seed}\0{epoch}\0{sample_id}".encode()
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
    reference = ValidatedSymileCvReference({}, _assignments(frame))
    return SymileDevelopmentData(SymileDevelopmentCohort(_bundle(tmp_path), frame), reference)


@pytest.mark.parametrize("family", ["cxr_densenet", "cxr_labs_concat", "cxr_labs_ecg_gated"])
def test_neural_fold_delivers_inner_partitions_to_shared_training_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    config = load_symile_development_config(
        f"configs/symile_{family}.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
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
        python_version=platform.python_version(),
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


def test_neural_holdout_establishes_backend_policy_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_symile_development_config(
        "configs/symile_cxr_densenet.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
    data = _data(tmp_path)
    outer = materialize_outer_fold(data, repeat_seed=17, outer_fold=0)
    context = symile_development.SymileFoldExecutionContext(
        data=data,
        cxr_store=SimpleNamespace(
            canonical_image=lambda split, row: np.full((224, 224), 0.5, dtype=np.float32)
        ),
        runtime=cpu_runtime(),
        loader_execution=reused_loader_policy(num_workers=0, pin_memory=False),
        git_commit="a" * 40,
        dependency_lock_sha256="b" * 64,
        python_version=platform.python_version(),
        source_cxr_folds={},
        source_cxr_development_id=None,
    )
    configured = []
    monkeypatch.setattr(
        symile_development, "configure_neural_determinism", lambda: configured.append(True)
    )

    class FixedModel(torch.nn.Module):
        def forward(self, image):
            assert configured == [True]
            assert not self.training
            assert not torch.is_grad_enabled()
            return torch.full((len(image),), 0.25, device=image.device)

    logits = symile_development._infer_neural_holdout(config, context, outer, FixedModel(), None)
    np.testing.assert_array_equal(logits, np.full(len(outer.holdout), 0.25))


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


def test_preopening_scientific_io_is_development_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    import pyarrow.parquet as pq
    from symile_data_test_support import _synthetic_release

    from beyondcxr.data.symile_artifacts import validate_symile_bundle
    from beyondcxr.data.symile_audit import generate_symile_audit
    from beyondcxr.data.symile_cv import publish_symile_cv
    from beyondcxr.training import symile_execution
    from beyondcxr.training.symile_formal import SYMILE_FORMAL_RUN_PLAN

    source = _synthetic_release(tmp_path)
    read_csv, read_table, load_npy = pd.read_csv, pq.read_table, np.load
    decoded = []

    def development_csv(path, *args, **kwargs):
        if isinstance(path, io.BytesIO):
            # The mixed index must be projected BEFORE a CSV decoder sees any bytes.
            raw = path.getvalue()
            assert b"83000" not in raw and b"93000" not in raw
        else:
            assert Path(path).name in {"train.csv", "val.csv"}
        frame = read_csv(path, *args, **kwargs)
        assert not frame["hadm_id"].between(830000, 830009).any()
        decoded.append("csv")
        return frame

    def development_parquet(path, *args, **kwargs):
        table = read_table(path, *args, **kwargs)
        if "official_split" in table.column_names:
            assert set(table["official_split"].to_pylist()) <= {"train", "validation"}
        if "sample_id" in table.column_names:
            assert not any(
                value.startswith("symile:830") for value in table["sample_id"].to_pylist()
            )
        decoded.append("parquet")
        return table

    def development_npy(path, *args, **kwargs):
        assert Path(path).parent.name in {"train", "val"}
        decoded.append("npy")
        return load_npy(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", development_csv)
    monkeypatch.setattr(pq, "read_table", development_parquet)
    monkeypatch.setattr(np, "load", development_npy)
    manifests = tmp_path / "data/manifests"
    bundle, _ = symile_source.publish_symile_source_bundle(
        source, output_directory=manifests, enforce_production_counts=False
    )
    validate_symile_bundle(bundle.bundle_directory)
    cv_id, _ = publish_symile_cv(bundle, manifest_directory=manifests)
    generate_symile_audit(manifests, tmp_path / "reports", bundle_id=bundle.bundle_id)
    config = load_symile_development_config(
        "configs/symile_labs_logistic.yaml", bundle_id=bundle.bundle_id, cv_assignment_id=cv_id
    )
    config = replace(config, runtime=replace(config.runtime, manifest_directory=manifests))
    loaded = load_symile_development(config, enforce_production_counts=False)
    assert len(loaded.frame) == 20
    for spec in SYMILE_FORMAL_RUN_PLAN:
        target = tmp_path / spec.config_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(spec.config_relative, target)
    from beyondcxr.models.cxr_baseline import PretrainedWeightIdentity

    monkeypatch.setattr(symile_execution, "git_revision", lambda root: ("a" * 40, False))
    monkeypatch.setattr(symile_execution, "uv_lock_sha256", lambda path: "b" * 64)
    monkeypatch.setattr(
        symile_execution,
        "resolve_device",
        lambda *a, **k: replace(
            cpu_runtime(),
            device=torch.device("cuda"),
            resolved_device="cuda",
            requested_device="cuda",
            cuda_available=True,
            mixed_precision_requested=True,
            mixed_precision_effective=True,
            pin_memory_requested="enabled",
            pin_memory_effective=True,
            cuda_runtime_version="12.4",
            cudnn_version=9100,
            gpu_device_name="Synthetic GPU",
            gpu_device_index=0,
            gpu_compute_capability=(8, 6),
        ),
    )
    monkeypatch.setattr(
        symile_execution,
        "fingerprint_pretrained_weights",
        lambda name: PretrainedWeightIdentity(name, "weights", "weights.pt", 1, "c" * 64),
    )
    preflight = symile_execution.build_symile_formal_preflight(
        tmp_path, bundle_id=bundle.bundle_id, cv_assignment_id=cv_id
    )
    assert preflight.document["dataset"]["bundle_id"] == bundle.bundle_id
    assert set(decoded) == {"csv", "parquet", "npy"}
    assert not list(tmp_path.rglob("test-open.json"))


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
    return reference.manifest["source"]["checksum_manifest_sha256"]


def test_cxr_store_resolves_rows_and_rejects_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = np.stack([np.full((3, 320, 320), value, dtype=np.float32) for value in (0.2, 0.8)])
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1)
    paths: dict[str, Path] = {}
    for split in ("train", "validation"):
        path = development_modality_asset_path(tmp_path, split, "cxr")
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
        split: development_modality_asset_path(tmp_path, split, modality)
        for split in ("train", "validation")
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
        split: development_modality_asset_path(tmp_path, split, modality)
        for split in ("train", "validation")
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, raw)
    checksum_sha256 = _store_checksum_authority(tmp_path, monkeypatch, module, paths)
    hashed: list[str] = []
    original = symile_source._sha256_descriptor

    def count_hash(descriptor: int) -> str:
        hashed.append(os.readlink(f"/proc/self/fd/{descriptor}"))
        return original(descriptor)

    monkeypatch.setattr(symile_source, "_sha256_descriptor", count_hash)
    assets = tuple(
        symile_source.establish_development_release_asset(
            tmp_path,
            checksum_manifest_sha256=checksum_sha256,
            official_split=split,
            modality=modality,
        )
        for split in DEVELOPMENT_SPLITS
    )
    store = store_type(_bundle(tmp_path), tmp_path, development_assets=assets)
    store_type(_bundle(tmp_path), tmp_path, development_assets=assets)
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
    folds = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0, coordinates=((17, 0),))
    first = validate_fold_package(folds[0].package.directory)

    assert folds[0].prediction.predictions.schema == PREDICTION_SCHEMA
    assert folds[0].prediction.predictions.num_rows == 4
    assert len(set(folds[0].prediction.predictions["sample_id"].to_pylist())) == 4
    prediction = validate_prediction_evidence(folds[0].prediction.directory)
    assert first.manifest == folds[0].package.manifest
    assert prediction.manifest == folds[0].prediction.manifest
    assert prediction.manifest["model_package_id"] == first.manifest["fold_package_id"]
    assert first.manifest["selection"]["metric"] == "none"
    assert "predictions.parquet" not in {path.name for path in first.directory.iterdir()}


_MISSING_SCHEMA_VERSION = object()


def test_fold_manifest_requires_integer_schema_version_one(tmp_path: Path) -> None:
    folds = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0, coordinates=((17, 0),))
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
    config = load_symile_development_config(
        "configs/symile_cxr_labs_concat.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
    lineage = _lineage("cxr_labs_concat")
    source = {
        "development_id": "development-" + "3" * 64,
        "fold_package_id": "fold-package-" + "4" * 64,
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
        return canonical_scientific_id(
            "fold-package-", payload, domain="fold-package", schema_version=1
        )

    baseline = identity(lineage, source)
    changed_reproduction = {
        **lineage,
        "git_commit": "changed-documentation-only-commit",
        "dependency_lock_sha256": "8" * 64,
    }
    assert identity(changed_reproduction, source) == baseline
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
    assert (
        canonical_scientific_id(
            "fold-package-", changed_selection, domain="fold-package", schema_version=1
        )
        != baseline
    )

    lightgbm_config = load_symile_development_config(
        "configs/symile_labs_lightgbm.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
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
    assert canonical_scientific_id(
        "fold-package-", changed_iteration, domain="fold-package", schema_version=1
    ) != canonical_scientific_id(
        "fold-package-", lightgbm_payload, domain="fold-package", schema_version=1
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
    config = load_symile_development_config(
        _family_config_path(family),
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )

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
        return canonical_scientific_id(
            "fold-package-", payload, domain="fold-package", schema_version=1
        )

    baseline = identity("inner-split-" + "a" * 64)
    changed = identity("inner-split-" + "b" * 64)
    assert (changed != baseline) is inner_split_is_semantic


def test_cxr_fold_identity_uses_pretrained_scientific_projection() -> None:
    config = load_symile_development_config(
        "configs/symile_cxr_densenet.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
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
        return canonical_scientific_id(
            "fold-package-", payload, domain="fold-package", schema_version=1
        )

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
    folds = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0, coordinates=((17, 0),))
    manifest = deepcopy(folds[0].package.manifest)
    baseline = canonical_scientific_id(
        "fold-package-",
        symile_publication._fold_payload_from_manifest(manifest),
        domain="fold-package",
        schema_version=1,
    )
    manifest["operational"]["mlflow_run_id"] = "another-run"
    assert (
        canonical_scientific_id(
            "fold-package-",
            symile_publication._fold_payload_from_manifest(manifest),
            domain="fold-package",
            schema_version=1,
        )
        == baseline
    )
    manifest["artifacts"][symile_publication.TABULAR_MODEL_FILENAME]["physical_sha256"] = "f" * 64
    assert (
        canonical_scientific_id(
            "fold-package-",
            symile_publication._fold_payload_from_manifest(manifest),
            domain="fold-package",
            schema_version=1,
        )
        == baseline
    )
    manifest["config"]["fit_config"]["training"]["parameters"]["C"] = 2.0
    assert (
        canonical_scientific_id(
            "fold-package-",
            symile_publication._fold_payload_from_manifest(manifest),
            domain="fold-package",
            schema_version=1,
        )
        != baseline
    )


def test_development_rejects_mixed_fit_config_with_two_lightweight_folds() -> None:
    first = SimpleNamespace(manifest={"config": {"fit_config": {"C": 1.0}}})
    second = SimpleNamespace(manifest={"config": {"fit_config": {"C": 2.0}}})

    with pytest.raises(ManifestBuildError, match="mixed scientific fit configs"):
        symile_publication._require_consistent_development_fit_config((first, second))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("dataset_id", "rsna"),
        ("task_id", "other-task"),
        ("bundle_id", "bundle-" + "9" * 64),
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
    folds = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0, coordinates=((17, 0),))
    completed = folds[0]
    changed = replace(
        completed.prediction,
        manifest={**completed.prediction.manifest, field: replacement},
    )

    with pytest.raises(ManifestBuildError, match="scientific lineage"):
        symile_publication.validate_fold_prediction_pair(
            completed.package, changed, _cv_authority()
        )


@pytest.mark.parametrize("field", ["inner_seed", "policy.inner_seed"])
def test_fold_manifest_rejects_noncanonical_inner_seed(tmp_path: Path, field: str) -> None:
    folds = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0, coordinates=((17, 0),))
    manifest = deepcopy(folds[0].package.manifest)
    if field == "inner_seed":
        manifest["inner_split"]["inner_seed"] += 1
    else:
        manifest["inner_split"]["policy"]["inner_seed"] += 1

    with pytest.raises(ManifestBuildError, match="inner split policy"):
        symile_publication._validate_fold_manifest(manifest)


def test_fold_prediction_pair_accepts_exact_cv_authority(tmp_path: Path) -> None:
    folds = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0, coordinates=((17, 0),))
    symile_publication.validate_fold_prediction_pair(
        folds[0].package, folds[0].prediction, _cv_authority()
    )


@pytest.mark.parametrize("mutation", ["wrong_fold", "target", "fabricated"])
def test_fold_prediction_pair_rejects_rows_outside_cv_authority(
    tmp_path: Path, mutation: str
) -> None:
    folds = _publish_family_folds(tmp_path, "labs_logistic", offset=0.0, coordinates=((17, 0),))
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
        symile_publication.validate_fold_prediction_pair(
            completed.package, changed, _cv_authority()
        )


def test_fusion_fold_resolves_matching_source_cxr_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(symile_publication, "reconstruct_fold_neural_model", lambda *args: None)
    coordinates = ((17, 0), (17, 1))
    source_folds = _publish_family_folds(
        tmp_path, "cxr_densenet", offset=0.0, coordinates=coordinates
    )
    fusion_folds = _publish_family_folds(
        tmp_path,
        "cxr_labs_concat",
        offset=0.1,
        source_cxr_folds=source_folds,
        source_cxr_development_id="development-" + "8" * 64,
        coordinates=coordinates,
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
    )
    with pytest.raises(ManifestBuildError, match="incompatible"):
        symile_publication._validate_source_cxr_fold(fusion.directory, incompatible)


def test_fold_publication_rejects_wrong_model_malformed_state_and_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logistic = load_symile_development_config(
        "configs/symile_labs_logistic.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
    wrong_model = LogisticRegression().fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    with pytest.raises(ManifestBuildError, match="exact fitted pipeline"):
        publish_fold_package(
            model_root=tmp_path / "wrong-tabular",
            family="labs_logistic",
            repeat_seed=17,
            outer_fold=0,
            config_bytes=logistic.source_bytes,
            lineage=_lineage("labs_logistic"),
            inner_split=_inner_split(17, 0),
            selection={"metric": "none", "selected_epoch": None, "best_iteration": None},
            model=wrong_model,
            operational={
                "mlflow_run_id": "run-wrong",
                "python_version": platform.python_version(),
                "runtime_provenance": None,
                "loader_execution": None,
            },
        )

    cxr = load_symile_development_config(
        "configs/symile_cxr_densenet.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
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
            operational={
                "mlflow_run_id": "run-malformed",
                "python_version": platform.python_version(),
                "runtime_provenance": {},
                "loader_execution": {},
            },
        )

    monkeypatch.setattr(symile_publication, "reconstruct_fold_neural_model", lambda *args: None)
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
            operational={
                "mlflow_run_id": "run-history",
                "python_version": platform.python_version(),
                "runtime_provenance": {},
                "loader_execution": {},
            },
        )


def test_neural_selection_supports_configured_lifecycles_beyond_thirty_epochs() -> None:
    config = load_symile_development_config(
        "configs/symile_cxr_densenet.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
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
    config = load_symile_development_config(
        "configs/symile_cxr_densenet.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
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
    config = load_symile_development_config(
        "configs/symile_cxr_densenet.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
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
    config = load_symile_development_config(
        "configs/symile_cxr_densenet.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
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
        )

    published: dict[str, object] = {}

    def publish(report_root, family_ids, derived) -> Path:
        published.update(family_ids)
        return tmp_path / "analysis"

    monkeypatch.setattr(symile_development_reports, "validate_development_result", validate)
    monkeypatch.setattr(symile_development_reports, "_publish_analysis_document", publish)
    monkeypatch.setattr(symile_development_reports, "_analysis_from_developments", lambda *args: {})

    symile_development_reports.analyze_symile_development(
        development_ids,
        report_root=tmp_path / "reports",
        model_root=tmp_path / "models",
        prediction_root=tmp_path / "private",
        manifest_root=manifest_root,
    )

    assert validated_roots == [manifest_root] * 6
    assert set(published.values()) == set(development_ids)


def test_cv_requires_exact_three_by_five_coordinate_set() -> None:
    expected = [(seed, fold) for seed in (17, 42, 2026) for fold in range(5)]
    symile_publication._require_exact_development_coordinates(expected, "invalid coordinates")


def test_development_analysis_rederives_claims_and_checks_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    families = symile_publication.SYMILE_CORE_DEVELOPMENT_FAMILIES
    family_ids = {family: "development-" + f"{index:064x}" for index, family in enumerate(families)}
    evidence = [
        SimpleNamespace(
            manifest={"repeat_seed": seed},
            predictions=pa.table(
                {
                    "sample_id": ["synthetic-a", "synthetic-b", "synthetic-c", "synthetic-d"],
                    "target": [0, 1, 0, 1],
                    "logit": [-np.log(3.0), np.log(3.0), -np.log(3.0), np.log(3.0)],
                    "probability": [0.25, 0.75, 0.25, 0.75],
                }
            ),
        )
        for seed in REPEAT_SEEDS
    ]

    def development(path: Path, **_kwargs: object) -> ValidatedDevelopmentResult:
        family = next(name for name, identity in family_ids.items() if identity == path.name)
        return ValidatedDevelopmentResult(
            path, {"family_id": family, "development_id": path.name}, (), tuple(evidence)
        )

    # Package/CV validation is covered by the publication and coordinate tests. This boundary
    # supplies validated inputs so the report's numerical derivation and byte checks run directly.
    monkeypatch.setattr(symile_development_reports, "validate_development_result", development)
    monkeypatch.setattr(
        symile_development_reports, "_validate_cross_family_fold_lineage", lambda *_args: None
    )
    monkeypatch.setattr(symile_development_reports, "DEVELOPMENT_COUNT", 4)
    roots = {
        "report_root": tmp_path / "reports",
        "model_root": tmp_path / "models",
        "prediction_root": tmp_path / "private",
    }
    directory = symile_development_reports.analyze_symile_development(
        list(family_ids.values()), **roots
    )
    document = symile_development_reports.validate_analysis_result(directory, **roots)
    assert document["paired_effects"]["gated_minus_cxr"]["17"]["roc_auc"] == 0.0
    manifest = directory / symile_development_reports.ANALYSIS_MANIFEST_FILENAME
    original = manifest.read_bytes()
    document["analysis_schema_version"] = True
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="schema version"):
        symile_development_reports.validate_analysis_result(directory, **roots)
    document["analysis_schema_version"] = 1
    document["paired_effects"]["gated_minus_cxr"]["17"]["roc_auc"] = 0.1
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="paired_effects differs"):
        symile_development_reports.validate_analysis_result(directory, **roots)
    manifest.write_bytes(original)
    summary = directory / symile_publication.SUMMARY_FILENAME
    assert "synthetic-a" not in summary.read_text(encoding="utf-8")
    summary.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="summary differs"):
        symile_development_reports.validate_analysis_result(directory, **roots)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong-seed", "wrong-fold"])
def test_cv_rejects_invalid_coordinate_metadata(mutation: str) -> None:
    coordinates = [(seed, fold) for seed in (17, 42, 2026) for fold in range(5)]
    if mutation == "missing":
        coordinates.pop()
    elif mutation == "duplicate":
        coordinates[-1] = coordinates[-2]
    elif mutation == "wrong-seed":
        coordinates[0] = (99, 0)
    else:
        coordinates[0] = (17, 9)

    with pytest.raises(ManifestBuildError, match="invalid coordinates"):
        symile_publication._require_exact_development_coordinates(
            coordinates, "invalid coordinates"
        )


def test_final_training_budget_uses_fifteen_scalar_selection_values() -> None:
    selected = list(range(1, 16))
    symile_publication._validate_budget("cxr_labs_gated", selected, 8)

    with pytest.raises(ManifestBuildError, match="budget"):
        symile_publication._validate_budget("cxr_labs_gated", selected[:-1], 8)
    with pytest.raises(ManifestBuildError, match="budget"):
        symile_publication._validate_budget("cxr_labs_gated", selected, 9)


def _cross_family_coordinate_graph() -> tuple[
    dict[str, dict[tuple[int, int], object]], dict[str, str]
]:
    coordinate = (17, 0)
    cxr_development_id = "development-" + "c" * 64
    cxr_fold_id = "fold-package-" + "d" * 64
    family_ids = {
        family: ("development-" + f"{index:x}" * 64)
        for index, family in enumerate(symile_publication.SYMILE_CORE_DEVELOPMENT_FAMILIES, start=1)
    }
    family_ids["cxr_densenet"] = cxr_development_id
    families: dict[str, dict[tuple[int, int], object]] = {}
    for index, family in enumerate(symile_publication.SYMILE_CORE_DEVELOPMENT_FAMILIES, start=1):
        manifest = {
            "family_id": family,
            "fold_package_id": (
                cxr_fold_id if family == "cxr_densenet" else "fold-package-" + f"{index:x}" * 64
            ),
            "inner_split": {"inner_split_id": "inner-split-" + "a" * 64},
            "lineage": {
                "bundle_id": "bundle-" + "b" * 64,
                "cv_assignment_id": "cv-assignment-" + "e" * 64,
                "task_id": "pneumonia_strict",
            },
            "source_cxr": (
                {
                    "development_id": cxr_development_id,
                    "fold_package_id": cxr_fold_id,
                }
                if family in symile_development_reports.SYMILE_FUSION_FAMILIES
                else None
            ),
        }
        families[family] = {coordinate: SimpleNamespace(manifest=manifest)}
    return families, family_ids


def test_cross_family_coordinate_binding_uses_one_lightweight_coordinate() -> None:
    families, family_ids = _cross_family_coordinate_graph()
    logistic = families["labs_logistic"][(17, 0)]
    logistic.manifest["inner_split"]["inner_split_id"] = "inner-split-" + "9" * 64
    symile_development_reports._validate_cross_family_coordinate_lineage(
        families, family_ids, (17, 0)
    )

    families["labs_lightgbm"][(17, 0)].manifest["inner_split"]["inner_split_id"] = (
        "inner-split-" + "8" * 64
    )
    with pytest.raises(ManifestBuildError, match="different inner splits"):
        symile_development_reports._validate_cross_family_coordinate_lineage(
            families, family_ids, (17, 0)
        )


def test_cross_family_coordinate_rejects_foreign_bundle_and_cxr_source() -> None:
    families, family_ids = _cross_family_coordinate_graph()
    families["labs_lightgbm"][(17, 0)].manifest["lineage"]["bundle_id"] = "bundle-" + "f" * 64
    with pytest.raises(ManifestBuildError, match="frozen data lineage"):
        symile_development_reports._validate_cross_family_coordinate_lineage(
            families, family_ids, (17, 0)
        )

    families, family_ids = _cross_family_coordinate_graph()
    families["cxr_labs_gated"][(17, 0)].manifest["source_cxr"]["fold_package_id"] = (
        "fold-package-" + "f" * 64
    )
    with pytest.raises(ManifestBuildError, match="incompatible CXR lineage"):
        symile_development_reports._validate_cross_family_coordinate_lineage(
            families, family_ids, (17, 0)
        )


def test_operational_completion_failure_preserves_fold_package_and_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_symile_development_config(
        "configs/symile_labs_logistic.yaml",
        bundle_id="bundle-" + "a" * 64,
        cv_assignment_id="cv-assignment-" + "c" * 64,
    )
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
            data.frame,
        ),
        data.cv_reference,
    )
    context = symile_development.SymileFoldExecutionContext(
        data,
        None,
        None,
        None,
        "a" * 40,
        "b" * 64,
        platform.python_version(),
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
            "logits": symile_development.symile_tabular_logits(
                fitted.pipeline, outer.holdout[list(LAB_FEATURE_COLUMNS)]
            ),
            "selection": {"metric": "none", "selected_epoch": None, "best_iteration": None},
        }

    monkeypatch.setattr(symile_development, "_fit_outer_fold", fake_fit)

    with prospective_symile_mutation(tmp_path, monkeypatch) as mutation:
        with pytest.raises(OSError):
            symile_development.execute_symile_outer_fold(
                config,
                context,
                prospective_mutation=mutation,
                repeat_seed=17,
                outer_fold=0,
            )
        monkeypatch.setattr(
            symile_development,
            "_fit_outer_fold",
            lambda *args: pytest.fail("Completed fold was fitted again"),
        )
        arguments = {"prospective_mutation": mutation, "repeat_seed": 17, "outer_fold": 0}
        resumed = symile_development.execute_symile_outer_fold(config, context, **arguments)
        original_id = resumed.prediction.prediction_id
        original_package = resumed.package.manifest["fold_package_id"]
        shutil.rmtree(resumed.prediction.directory)
        recovered = symile_development.execute_symile_outer_fold(config, context, **arguments)
        assert recovered.package.manifest["fold_package_id"] == original_package
        assert recovered.prediction.prediction_id == original_id
        with pytest.raises(ManifestBuildError, match="conflicts with the prospective campaign"):
            symile_development.execute_symile_outer_fold(
                config, replace(context, dependency_lock_sha256="9" * 64), **arguments
            )
        outer = symile_data.materialize_outer_fold(context.data, repeat_seed=17, outer_fold=0)
        extra = _publish_prediction_evidence(
            prospective_mutation=mutation,
            private_root=config.runtime.private_output_directory,
            dataset_id="symile",
            model_package_id=original_package,
            task_id=config.task.task_id,
            bundle_id=config.dataset.bundle_id,
            scope="outer_fold_oof",
            sample_ids=outer.holdout["sample_id"].astype(str).tolist(),
            targets=outer.holdout["target"].to_numpy(dtype=np.int8),
            logits=recovered.prediction.predictions["logit"].to_numpy() + 1.0,
            cv_assignment_id=config.dataset.cv_assignment_id,
            repeat_seed=17,
            outer_fold=0,
        )
        with pytest.raises(ManifestBuildError, match="prediction selection is ambiguous"):
            symile_development.execute_symile_outer_fold(config, context, **arguments)
        shutil.rmtree(extra.directory)
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
        "cv_assignment_id",
        "repeat_seed",
        "outer_fold",
        "inner_split_id",
        "git_commit",
        "git_dirty",
        "dependency_lock_sha256",
        "run_complete",
    }


def test_neural_fold_recovers_missing_oof_from_reconstructed_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = with_runtime(
        load_symile_development_config(
            "configs/symile_cxr_densenet.yaml",
            bundle_id="bundle-" + "a" * 64,
            cv_assignment_id="cv-assignment-" + "c" * 64,
        ),
        model_directory=tmp_path / "models/symile/development",
        private_output_directory=tmp_path / "private",
    )
    data = _data(tmp_path)
    context = symile_development.SymileFoldExecutionContext(
        data=data,
        cxr_store=SimpleNamespace(
            canonical_image=lambda split, row: np.full((224, 224), 0.5, dtype=np.float32)
        ),
        runtime=cpu_runtime(),
        loader_execution=reused_loader_policy(num_workers=0, pin_memory=False),
        git_commit="a" * 40,
        dependency_lock_sha256="b" * 64,
        python_version=platform.python_version(),
        source_cxr_folds={},
        source_cxr_development_id=None,
    )
    outer = materialize_outer_fold(data, repeat_seed=17, outer_fold=0)
    inner = derive_inner_split(outer)

    class SmallEncoder(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images.mean(dim=(1, 2, 3)).unsqueeze(1).expand(-1, 1024)

    monkeypatch.setattr(symile_publication, "StandardCxrEncoder", lambda **kwargs: SmallEncoder())
    model = symile_publication.CxrBinaryClassifier(SmallEncoder())
    with torch.no_grad():
        model.classifier.weight.fill_(0.001)
        model.classifier.bias.fill_(0.25)
    checkpoint = neural_checkpoint_document(
        model.state_dict(),
        selected_epoch=1,
        selected_stage="warmup",
        selected_validation_roc_auc=0.8,
    )
    pretrained = {
        "declared_name": "densenet121-res224-chex",
        "stable_identifier": "weights",
        "cache_filename": "weights.pt",
        "byte_size": 1,
        "sha256": "c" * 64,
    }
    provenance = _final_provenance(neural=True)
    with prospective_symile_mutation(tmp_path, monkeypatch) as mutation:
        package = _publish_fold_package(
            prospective_mutation=mutation,
            model_root=config.runtime.model_directory,
            family="cxr_densenet",
            repeat_seed=17,
            outer_fold=0,
            config_bytes=config.source_bytes,
            lineage=symile_development._fold_lineage(config, context, pretrained),
            inner_split={
                "inner_split_id": inner.inner_split_id,
                "inner_seed": inner.inner_seed,
                "policy": inner.policy,
            },
            selection={
                "metric": "roc_auc",
                "selected_epoch": 1,
                "best_iteration": None,
                "selected_stage": "warmup",
                "selected_validation_metric": 0.8,
            },
            model=checkpoint,
            training_history=_neural_history(config, 1),
            operational={
                key: provenance[key]
                for key in (
                    "mlflow_run_id",
                    "python_version",
                    "runtime_provenance",
                    "loader_execution",
                )
            },
        )
        expected = symile_development._infer_neural_holdout(config, context, outer, model, None)
        monkeypatch.setattr(
            symile_development,
            "_fit_outer_fold",
            lambda *args: pytest.fail("Published neural fold must not be fitted again"),
        )
        recovered = symile_development.execute_symile_outer_fold(
            config, context, prospective_mutation=mutation, repeat_seed=17, outer_fold=0
        )
        assert recovered.package.manifest["fold_package_id"] == package.manifest["fold_package_id"]
        np.testing.assert_array_equal(recovered.prediction.predictions["logit"], expected)
        first_id = recovered.prediction.prediction_id
        shutil.rmtree(recovered.prediction.directory)
        repeated = symile_development.execute_symile_outer_fold(
            config, context, prospective_mutation=mutation, repeat_seed=17, outer_fold=0
        )
        assert repeated.prediction.prediction_id == first_id


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

    assert (
        symile_development.main(["--config", "unused.yaml", "--backup-root", "/unused/backup"]) == 1
    )
    assert calls == []
    assert "Symile development failed: ConfigError" in capsys.readouterr().err


def test_operation_subgroups_reuse_authenticated_family_evidence(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from beyondcxr.training import symile_campaign_control as control

    capability = SimpleNamespace(_operation=None)
    families = tuple(
        ValidatedDevelopmentResult(tmp_path, {"family_id": family}, (), ())
        for family in ("cxr_densenet", "cxr_labs_gated")
    )
    monkeypatch.setattr(control, "require_validated_held_out_capability", lambda value: value)
    scoped = SimpleNamespace(
        _operation=object(), frozen_campaign=SimpleNamespace(developments=families)
    )
    observed = []

    def rows(development, predictions):
        assert predictions is development.predictions
        observed.append(development)
        return development.manifest["family_id"]

    def derive(cxr, cxr_rows, primary_rows, **kwargs):
        assert cxr is families[0]
        assert (cxr_rows, primary_rows) == ("cxr_densenet", "cxr_labs_gated")
        return {"strata": []}

    monkeypatch.setattr(symile_development_reports, "development_repeat_oof", rows)
    monkeypatch.setattr(symile_development_reports, "_focused_subgroups_from_development", derive)
    monkeypatch.setattr(
        symile_development_reports,
        "publish_focused_subgroup_derivative",
        lambda **kwargs: kwargs["summary"],
    )
    result = symile_development_reports.publish_operation_subgroups(
        scoped, report_root=tmp_path, model_root=tmp_path, manifest_root=tmp_path
    )
    assert result == {"strata": []}
    assert observed == list(families)
    with pytest.raises(PermissionError, match="active"):
        symile_development_reports.publish_operation_subgroups(
            capability, report_root=tmp_path, model_root=tmp_path, manifest_root=tmp_path
        )
