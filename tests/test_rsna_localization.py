from __future__ import annotations

import io
import json
import shlex
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torchxrayvision as xrv
from torch import nn

from beyondcxr.data.cxr_transforms import center_crop_geometry
from beyondcxr.evaluation.gradcam import gradcam_heatmaps, standard_cxr_gradcam_target
from beyondcxr.evaluation.localization import (
    deterministic_qualitative_selection,
    localization_metrics,
    transform_rsna_box,
    union_box_mask,
)
from beyondcxr.models.cxr_baseline import CxrBinaryClassifier, StandardCxrEncoder
from beyondcxr.training.config import load_experiment_config, with_runtime
from beyondcxr.training.rsna_datasets import RsnaDataset
from beyondcxr.training.rsna_localize import (
    QUALITATIVE_POLICY_VERSION,
    _evaluate_member,
    _gradcam_indices,
    _localization_id,
    _markdown,
    _positive_localization_metrics,
    _report_document,
    _rsna_localization_dataset,
    _validate_localization_output_boundaries,
    generate_localization_report,
)
from beyondcxr.utils.operational_logging import configure_logging

_PACKAGE_IDS = {
    17: "model-package-" + "1" * 64,
    42: "model-package-" + "2" * 64,
    2026: "model-package-" + "3" * 64,
}


class _GradCamModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU())
        self.classifier = nn.Linear(4, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        return self.classifier(features.mean(dim=(2, 3))).squeeze(1)


def test_localization_rejects_non_rsna_before_dataset_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    non_rsna = replace(config, dataset=replace(config.dataset, dataset_id="symile"))
    monkeypatch.setattr(
        "beyondcxr.training.rsna_localize.get_dataset",
        lambda key: pytest.fail(f"dataset registry accessed for {key}"),
    )

    with pytest.raises(ValueError):
        _rsna_localization_dataset(non_rsna)


def test_standalone_localization_resolves_one_shared_cache(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeds = (17, 42, 2026)
    evaluation_ids_15 = {seed: "evaluation-" + format(seed, "064x") for seed in seeds}
    evaluation_ids_20 = {seed: "evaluation-" + format(seed + 10_000, "064x") for seed in seeds}
    evaluation_seeds = {
        value: seed
        for identities in (evaluation_ids_15, evaluation_ids_20)
        for seed, value in identities.items()
    }
    calibration_bins = {
        **{value: 15 for value in evaluation_ids_15.values()},
        **{value: 20 for value in evaluation_ids_20.values()},
    }
    configs = {seed: load_experiment_config("configs/rsna_cxr_densenet.yaml") for seed in seeds}
    packages = {}
    for seed in seeds:
        package_id = _PACKAGE_IDS[seed]
        packages[package_id] = (
            {"training_policy": {"seed": seed}},
            configs[seed],
        )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_localize.validate_rsna_evaluation",
        lambda path, **kwargs: SimpleNamespace(
            manifest={
                "model_package_id": _PACKAGE_IDS[evaluation_seeds[path.name]],
                "evaluation_policy": {"calibration_bins": calibration_bins[path.name]},
            }
        ),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_localize.validate_neural_package_metadata",
        lambda package: packages[package.name][0],
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_localize.load_experiment_config",
        lambda path: packages[path.parent.name][1],
    )
    monkeypatch.setattr("beyondcxr.training.rsna_localize.get_dataset", lambda key: RsnaDataset())
    prepared: list[object] = []
    shared_cache = object()

    def prepare(*args, **kwargs):
        prepared.append((args, kwargs))
        return shared_cache

    monkeypatch.setattr("beyondcxr.training.rsna_localize.prepare_rsna_cxr_cache", prepare)
    observed_caches: list[object] = []

    def evaluate(package_id, package, manifest, config, *, examples, cache):
        del package_id, package, manifest, examples
        observed_caches.append(cache)
        return {
            "public": {
                "seed": config.runtime.seed,
                "positive_test_sample_count": 1,
                "localization_evaluated_count": 1,
                "zero_heatmap_count": 0,
                "pointing_game_accuracy": 1.0,
                "mean_activation_energy_inside_union": 0.5,
                "qualitative_strata_present": {
                    "TP": True,
                    "FN": False,
                    "FP": False,
                    "TN": True,
                },
            },
            "private_examples": [],
            "forbidden_source_values": set(),
        }

    monkeypatch.setattr("beyondcxr.training.rsna_localize._evaluate_member", evaluate)
    log_stream = io.StringIO()
    configure_logging("INFO", stream=log_stream)

    result = generate_localization_report(
        list(evaluation_ids_15.values()),
        output_directory=tmp_path / "reports",
        model_directory=tmp_path / "models",
        private_directory=tmp_path / "private",
    )

    assert result.is_dir()
    assert len(prepared) == 1
    assert observed_caches == [shared_cache, shared_cache, shared_cache]
    assert "event=phase_started phase=localization" in log_stream.getvalue()
    assert "event=phase_completed" in log_stream.getvalue()
    public_inode = (result / "summary.json").stat().st_ino
    private_inode = (
        (tmp_path / "private" / "localization" / result.name / "qualitative_manifest.json")
        .stat()
        .st_ino
    )

    repeated = generate_localization_report(
        list(reversed(evaluation_ids_20.values())),
        output_directory=tmp_path / "reports",
        model_directory=tmp_path / "models",
        private_directory=tmp_path / "private",
    )
    assert repeated == result
    assert (result / "summary.json").stat().st_ino == public_inode
    assert (
        tmp_path / "private" / "localization" / result.name / "qualitative_manifest.json"
    ).stat().st_ino == private_inode


def test_localization_identity_binds_source_model_packages() -> None:
    changed = [_PACKAGE_IDS[17], _PACKAGE_IDS[42], "model-package-" + "4" * 64]

    assert _localization_id(changed) != _localization_id(list(_PACKAGE_IDS.values()))


def test_localization_member_emits_generic_operation_completion(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    assert config.neural is not None
    config = with_runtime(
        replace(config, runtime=replace(config.runtime, pin_memory_policy="disabled")),
        seed=42,
        device="cpu",
    )
    samples = (
        {
            "image": torch.zeros((1, 224, 224), dtype=torch.float32),
            "target": torch.tensor(1.0),
            "sample_id": "positive",
            "patient_id": "patient-positive",
        },
        {
            "image": torch.zeros((1, 224, 224), dtype=torch.float32),
            "target": torch.tensor(0.0),
            "sample_id": "negative",
            "patient_id": "patient-negative",
        },
    )

    class ControlledDataset:
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int):
            return samples[index]

    class ConstantModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Identity()

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return torch.ones(len(image), device=image.device)

    model = ConstantModel()
    adapter = RsnaDataset()
    localization = SimpleNamespace(
        images=SimpleNamespace(
            lineage=object(),
            bundle_manifest_sha256="b" * 64,
            source_inventory=object(),
            test=object(),
        ),
        dimensions=pd.DataFrame(
            [
                {"sample_id": "positive", "image_rows": 224, "image_columns": 224},
                {"sample_id": "negative", "image_rows": 224, "image_columns": 224},
            ]
        ),
        annotations=pd.DataFrame(
            [
                {
                    "sample_id": "positive",
                    "x": 0.0,
                    "y": 0.0,
                    "width": 224.0,
                    "height": 224.0,
                }
            ]
        ),
    )
    adapter.load_localization_test = lambda *args, **kwargs: localization  # type: ignore[method-assign]
    monkeypatch.setattr("beyondcxr.training.rsna_localize.get_dataset", lambda key: adapter)
    monkeypatch.setattr(
        "beyondcxr.training.rsna_localize.load_validated_neural_checkpoint", lambda *args: {}
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_localize.get_model",
        lambda key: SimpleNamespace(build_architecture=lambda model_config: model),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_localize.strict_load_checkpoint", lambda *args: None
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_localize.standard_cxr_gradcam_target", lambda value: value.encoder
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_localize.expected_rsna_cxr_cache_identity",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_localize.RsnaCachedImageDataset",
        lambda *args, **kwargs: ControlledDataset(),
    )
    monkeypatch.setattr(
        "beyondcxr.training.rsna_localize.gradcam_heatmaps",
        lambda *args, **kwargs: torch.ones((1, 224, 224)),
    )
    monkeypatch.setattr("beyondcxr.training.rsna_localize._write_overlay", lambda *args: None)
    authentication = {"policy_version": "test"}
    cache = SimpleNamespace(
        source_authentication=SimpleNamespace(as_dict=lambda: authentication),
        identity=SimpleNamespace(cache_id="cache-" + "c" * 64),
    )
    examples = tmp_path / "examples"
    examples.mkdir()
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)

    _evaluate_member(
        _PACKAGE_IDS[42],
        tmp_path,
        {
            "bundle_manifest_sha256": "b" * 64,
            "source_authentication": authentication,
            "runtime_provenance": {"cxr_cache_id": cache.identity.cache_id},
            "thresholds": {"youden_j": 0.5},
        },
        config,
        examples=examples,
        cache=cache,
    )

    progress = [
        set(shlex.split(line))
        for line in stream.getvalue().splitlines()
        if "event=operation_progress" in line
    ]
    for operation in ("prediction", "gradcam"):
        assert any(
            {
                f"operation={operation}",
                "seed=42",
                "completed=2",
                "total=2",
                "unit=samples",
            }
            <= record
            for record in progress
        )


def test_gradcam_is_finite_deterministic_cleans_hooks_and_preserves_state() -> None:
    torch.manual_seed(42)
    model = _GradCamModel()
    image = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    first = gradcam_heatmaps(model, image, target_module=model.encoder, output_size=(8, 8))
    second = gradcam_heatmaps(model, image, target_module=model.encoder, output_size=(8, 8))

    assert first.shape == (1, 8, 8)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert not model.encoder._forward_hooks
    assert model.training
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())


def test_gradcam_removes_target_hook_and_restores_mode_on_failure() -> None:
    class InvalidModel(_GradCamModel):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            features = self.encoder(image)
            return features.mean(dim=(2, 3))

    model = InvalidModel()
    with pytest.raises(ValueError):
        gradcam_heatmaps(
            model,
            torch.ones((1, 1, 8, 8)),
            target_module=model.encoder,
            output_size=(8, 8),
        )

    assert not model.encoder._forward_hooks
    assert model.training


def test_standard_cxr_gradcam_target_is_complete_final_spatial_representation() -> None:
    model = CxrBinaryClassifier(StandardCxrEncoder(weights=None)).eval()
    target = standard_cxr_gradcam_target(model)
    captured = []
    handle = target.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.shape)
    )
    try:
        with torch.inference_mode():
            logits = model(torch.zeros((1, 1, 224, 224)))
    finally:
        handle.remove()

    assert tuple(dict(target.named_children()))[-1] == "norm5"
    assert captured == [torch.Size([1, 1024, 7, 7])]
    assert logits.shape == (1,)


@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        (8, 8),
        (9, 9),
        (10, 8),
        (9, 8),
        (10, 9),
        (9, 7),
        (8, 10),
        (9, 10),
        (8, 9),
        (7, 9),
        (8, 9),
        (9, 8),
    ],
)
def test_center_crop_geometry_exactly_matches_torchxrayvision(rows: int, columns: int) -> None:
    pixels = np.arange(rows * columns, dtype=np.float32).reshape(1, rows, columns)
    geometry = center_crop_geometry(rows, columns)
    described = pixels[
        :,
        geometry.offset_y : geometry.offset_y + geometry.crop_size,
        geometry.offset_x : geometry.offset_x + geometry.crop_size,
    ]

    assert np.array_equal(xrv.datasets.XRayCenterCrop()(pixels), described)


def test_center_crop_and_box_mapping_cover_square_clipped_and_outside_geometry() -> None:
    square = center_crop_geometry(1024, 1024)
    wide = center_crop_geometry(100, 200)
    assert (square.crop_size, square.offset_x, square.offset_y) == (1024, 0, 0)
    assert (wide.crop_size, wide.offset_x, wide.offset_y) == (100, 50, 0)

    box = transform_rsna_box(
        {"x": 50.0, "y": 0.0, "width": 100.0, "height": 100.0},
        source_rows=100,
        source_columns=200,
    )
    assert (box.x0, box.y0, box.x1, box.y1) == (0.0, 0.0, 224.0, 224.0)
    assert (
        transform_rsna_box(
            {"x": 0.0, "y": 0.0, "width": 20.0, "height": 20.0},
            source_rows=100,
            source_columns=200,
        )
        is None
    )
    clipped = transform_rsna_box(
        {"x": 40.0, "y": 10.0, "width": 20.0, "height": 30.0},
        source_rows=100,
        source_columns=200,
    )
    assert clipped is not None
    assert clipped.x0 == 0.0
    assert clipped.x1 == pytest.approx(22.4)


@pytest.mark.parametrize(
    "box",
    [
        {"x": -1.0, "y": 0.0, "width": 2.0, "height": 2.0},
        {"x": 0.0, "y": 0.0, "width": 0.0, "height": 2.0},
        {"x": 3.0, "y": 3.0, "width": 2.0, "height": 2.0},
    ],
)
def test_invalid_source_boxes_are_rejected(box: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        transform_rsna_box(box, source_rows=4, source_columns=4)


def test_union_pointing_energy_zero_map_and_row_major_ties() -> None:
    first = transform_rsna_box(
        {"x": 0.0, "y": 0.0, "width": 2.0, "height": 2.0},
        source_rows=4,
        source_columns=4,
        output_size=4,
    )
    second = transform_rsna_box(
        {"x": 3.0, "y": 3.0, "width": 1.0, "height": 1.0},
        source_rows=4,
        source_columns=4,
        output_size=4,
    )
    mask = union_box_mask([first, second], output_size=4)
    heatmap = np.zeros((4, 4))
    heatmap[0, 0] = heatmap[3, 3] = 1.0
    assert localization_metrics(heatmap, mask) == (1, 1.0, False)
    assert localization_metrics(np.zeros((4, 4)), mask) == (0, 0.0, True)


def test_qualitative_selection_is_deterministic_and_does_not_substitute_strata() -> None:
    rows = [
        {"sample_id": "b", "stratum": "TP"},
        {"sample_id": "a", "stratum": "TP"},
        {"sample_id": "c", "stratum": "FN"},
    ]
    assert deterministic_qualitative_selection(rows) == deterministic_qualitative_selection(rows)
    selected = deterministic_qualitative_selection(rows)
    assert selected["TP"] in rows
    assert selected["FN"] == rows[2]
    assert selected["FP"] is None
    assert selected["TN"] is None


def test_gradcam_plan_includes_all_positives_and_only_selected_negatives() -> None:
    rows = [
        {"index": 0, "sample_id": "tp", "target": 1, "stratum": "TP"},
        {"index": 1, "sample_id": "fn", "target": 1, "stratum": "FN"},
        {"index": 2, "sample_id": "fp-selected", "target": 0, "stratum": "FP"},
        {"index": 3, "sample_id": "fp-unused", "target": 0, "stratum": "FP"},
        {"index": 4, "sample_id": "tn-selected", "target": 0, "stratum": "TN"},
        {"index": 5, "sample_id": "tn-unused", "target": 0, "stratum": "TN"},
    ]
    selected = {
        "TP": rows[0],
        "FN": rows[1],
        "FP": rows[2],
        "TN": rows[4],
    }

    assert _gradcam_indices(rows, selected) == (0, 1, 2, 4)


@pytest.mark.parametrize(
    "boxes",
    [
        [],
        [{"x": 0.0, "y": 0.0, "width": 20.0, "height": 20.0}],
    ],
)
def test_positive_without_usable_transformed_box_fails(boxes) -> None:
    with pytest.raises(ValueError):
        _positive_localization_metrics(
            boxes,
            source_rows=100,
            source_columns=200,
            heatmap=np.ones((224, 224), dtype=np.float64),
        )


def test_public_localization_report_states_interpretation_boundary() -> None:
    text = _markdown(
        {
            "members": [
                {
                    "seed": seed,
                    "positive_test_sample_count": 2,
                    "localization_evaluated_count": 2,
                    "zero_heatmap_count": 0,
                    "pointing_game_accuracy": 0.5,
                    "mean_activation_energy_inside_union": 0.4,
                }
                for seed in (17, 42, 2026)
            ],
            "aggregates": {
                "pointing_game_accuracy": {
                    "mean": 0.5,
                    "sample_standard_deviation": 0.0,
                },
                "mean_activation_energy_inside_union": {
                    "mean": 0.4,
                    "sample_standard_deviation": 0.0,
                },
            },
        }
    )

    assert "model-behavior localization sanity analysis" in text
    assert "not a causal explanation" in text
    assert "private workspace" in text


def test_localization_public_private_output_boundary_rejects_leaks_and_symlinks(
    tmp_path,
) -> None:
    public = tmp_path / "reports"
    private = tmp_path / "private"
    examples = private / "examples"
    public.mkdir()
    examples.mkdir(parents=True)
    (examples / "seed-42-example-01-tp.png").write_bytes(b"synthetic")
    members = [{"filename": "seed-42-example-01-tp.png"}]
    package_ids = [_PACKAGE_IDS[seed] for seed in (17, 42, 2026)]
    report_id = _localization_id(package_ids)
    public_members = [
        {
            "seed": seed,
            "positive_test_sample_count": 1,
            "localization_evaluated_count": 1,
            "zero_heatmap_count": 0,
            "pointing_game_accuracy": 1.0,
            "mean_activation_energy_inside_union": 0.5,
            "qualitative_strata_present": {
                "TP": True,
                "FN": False,
                "FP": False,
                "TN": True,
            },
        }
        for seed in (17, 42, 2026)
    ]
    document = _report_document(report_id, package_ids, public_members)
    (public / "summary.json").write_text(
        json.dumps(document, sort_keys=True),
        encoding="utf-8",
    )
    (public / "summary.md").write_text(_markdown(document), encoding="utf-8")
    (private / "qualitative_manifest.json").write_text(
        json.dumps(
            {
                "private_localization_schema_version": 1,
                "report_id": report_id,
                "selection_policy": QUALITATIVE_POLICY_VERSION,
                "examples": members,
            }
        ),
        encoding="utf-8",
    )

    _validate_localization_output_boundaries(
        public,
        private,
        private_members=members,
        forbidden_source_values={"synthetic-sample"},
    )

    (public / "summary.json").write_text(
        '{"localization_schema_version": true}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema version"):
        _validate_localization_output_boundaries(
            public,
            private,
            private_members=members,
            forbidden_source_values={"synthetic-sample"},
        )
    (public / "summary.json").write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    leaked = public / "patient-overlay.png"
    leaked.write_bytes(b"pixels")
    with pytest.raises(ValueError):
        _validate_localization_output_boundaries(
            public,
            private,
            private_members=members,
            forbidden_source_values={"synthetic-sample"},
        )
    leaked.unlink()
    example = examples / "seed-42-example-01-tp.png"
    target = tmp_path / "example.png"
    example.rename(target)
    example.symlink_to(target)
    with pytest.raises(ValueError):
        _validate_localization_output_boundaries(
            public,
            private,
            private_members=members,
            forbidden_source_values={"synthetic-sample"},
        )
