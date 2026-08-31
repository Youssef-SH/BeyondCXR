from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from symile_campaign_test_support import _freeze, _synthetic_held_out_capability
from symile_data_test_support import (
    _published_synthetic_release,
    _refresh_checksums,
    _synthetic_release,
)

import beyondcxr.data.symile_source as symile_source
import beyondcxr.training.symile_test_data as symile_test_data
from beyondcxr.data.symile_source import publish_symile_source_bundle
from beyondcxr.training.rsna_validation_evidence import load_canonical_held_out_cohort
from beyondcxr.training.symile_test_data import (
    FrozenSymileTestCxrStore,
    FrozenSymileTestEcgStore,
)
from beyondcxr.utils.private_predictions import publish_prediction_evidence


def test_symile_development_asset_boundary_rejects_held_out_material(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="development modalities"):
        symile_source.development_modality_asset_path(tmp_path, "test", "cxr")
    with pytest.raises(ValueError, match="held-out material"):
        symile_source.DevelopmentReleaseAsset(
            tmp_path,
            "a" * 64,
            "data_npy/test/cxr_test.npy",
            "b" * 64,
            (1, 2, 3, 4, 5),
        )


@pytest.mark.parametrize(
    ("store_type", "shape"),
    [
        (FrozenSymileTestCxrStore, (1, 3, 320, 320)),
        (FrozenSymileTestEcgStore, (1, 1, 5000, 12)),
    ],
    ids=("cxr", "ecg"),
)
def test_symile_held_out_capability_rejects_a_second_valid_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    store_type,
    shape: tuple[int, ...],
) -> None:
    source_a, bundle_a = _published_synthetic_release(tmp_path / "a")
    source_b = _synthetic_release(tmp_path / "b")
    code_path = source_b / "code/process_mimic_data.py"
    code_path.write_text(code_path.read_text(encoding="utf-8") + "# bundle-b\n", encoding="utf-8")
    _refresh_checksums(source_b)
    bundle_b, _ = publish_symile_source_bundle(
        source_b,
        output_directory=tmp_path / "b/manifests",
        enforce_production_counts=False,
    )
    assert bundle_a.bundle_id != bundle_b.bundle_id
    freeze = _freeze(tmp_path / "freeze")
    freeze.manifest["bundle_id"] = bundle_a.bundle_id
    capability = _synthetic_held_out_capability(tmp_path / "capability", freeze)
    monkeypatch.setattr(
        symile_test_data,
        "require_validated_held_out_capability",
        lambda value: capability if value is capability else pytest.fail("foreign capability"),
    )
    monkeypatch.setattr(
        symile_test_data,
        "materialize_official_test",
        lambda value, operation: operation() if value is capability else pytest.fail("foreign"),
    )
    monkeypatch.setattr(
        symile_test_data,
        "validated_pretest_freeze_manifest",
        lambda value: value.manifest,
    )
    real_load = np.load
    header = np.lib.format.open_memmap(
        tmp_path / "valid-header.npy", mode="w+", dtype=np.float32, shape=shape
    )
    del header
    monkeypatch.setattr(
        symile_test_data.np,
        "load",
        lambda path, *args, **kwargs: real_load(tmp_path / "valid-header.npy", *args, **kwargs),
    )

    assert store_type(capability, bundle_a, source_a)._bundle_id == bundle_a.bundle_id
    with pytest.raises(PermissionError, match="differs from the frozen campaign"):
        store_type(capability, bundle_b, source_b)


def test_generic_publisher_cannot_mint_held_out_evidence(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="study-specific validated authority"):
        publish_prediction_evidence(
            private_root=tmp_path,
            dataset_id="symile",
            model_package_id="final-package-" + "a" * 64,
            task_id="pneumonia_strict",
            bundle_id="bundle-" + "b" * 64,
            scope="test",
            sample_ids=("symile:1", "symile:2"),
            targets=(0, 1),
            logits=(-1.0, 1.0),
            authorized_by_freeze_id="pretest-freeze-" + "c" * 64,
        )


def test_rsna_bundle_coordinates_cannot_materialize_held_out_cohort(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        load_canonical_held_out_cohort(
            tmp_path / "bundle",
            model_package_id="model-package-" + "d" * 64,
        )
