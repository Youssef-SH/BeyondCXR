from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchxrayvision as xrv
from torch import nn

from radfusion.data.cxr_transforms import center_crop_geometry
from radfusion.evaluation.gradcam import gradcam_heatmaps, standard_cxr_gradcam_target
from radfusion.evaluation.localization import (
    deterministic_qualitative_selection,
    localization_metrics,
    transform_rsna_box,
    union_box_mask,
)
from radfusion.models.cxr_baseline import CxrBinaryClassifier, StandardCxrEncoder
from radfusion.training.config import load_experiment_config
from radfusion.training.datasets import RsnaDataset
from radfusion.training.localize import (
    _gradcam_indices,
    _markdown,
    _positive_localization_metrics,
    _rsna_localization_dataset,
    _validate_localization_output_boundaries,
    generate_localization_report,
)


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
    config = load_experiment_config("configs/image_densenet_seed42.yaml")
    dataset = replace(config.dataset, registry_key="symile")
    monkeypatch.setattr(
        "radfusion.training.localize.get_dataset",
        lambda key: pytest.fail(f"dataset registry accessed for {key}"),
    )

    with pytest.raises(ValueError):
        _rsna_localization_dataset(dataset)


def test_standalone_localization_resolves_one_shared_cache(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeds = (17, 42, 2026)
    configs = {
        seed: load_experiment_config(f"configs/image_densenet_seed{seed}.yaml") for seed in seeds
    }
    runs = {}
    for seed in seeds:
        training_id = f"training-{seed}"
        runs[f"test-{seed}"] = SimpleNamespace(
            run_id=f"test-{seed}",
            source_training_run_id=training_id,
            modality="image",
            run_kind="test_evaluation",
            evaluation_scope="test",
            integer_seed=lambda seed=seed: seed,
        )
        runs[training_id] = SimpleNamespace(
            run_id=training_id,
            local_model_path=str(tmp_path / f"seed-{seed}" / "model.pt"),
        )
    monkeypatch.setattr(
        "radfusion.training.localize.configure_mlflow",
        lambda **kwargs: SimpleNamespace(get_run=runs.__getitem__),
    )
    monkeypatch.setattr("radfusion.training.localize.require_completed_run", lambda run: run)
    monkeypatch.setattr(
        "radfusion.training.localize.has_matching_training_parent", lambda *args: True
    )
    monkeypatch.setattr("radfusion.training.localize.git_revision", lambda: ("commit", False))
    monkeypatch.setattr("radfusion.training.localize.uv_lock_sha256", lambda: "a" * 64)
    monkeypatch.setattr(
        "radfusion.training.localize.validate_neural_package_metadata", lambda package: {}
    )
    monkeypatch.setattr(
        "radfusion.training.localize.load_experiment_config",
        lambda path: configs[int(path.parent.name.removeprefix("seed-"))],
    )
    monkeypatch.setattr(
        "radfusion.training.localize.verify_image_training_package", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "radfusion.training.localize.image_seed_compatibility_sha256", lambda config: "compatible"
    )
    monkeypatch.setattr("radfusion.training.localize.get_dataset", lambda key: RsnaDataset())
    prepared: list[object] = []
    shared_cache = object()

    def prepare(*args, **kwargs):
        prepared.append((args, kwargs))
        return shared_cache

    monkeypatch.setattr("radfusion.training.localize.prepare_rsna_cxr_cache", prepare)
    observed_caches: list[object] = []

    def evaluate(test, training, package, manifest, config, *, examples, cache):
        del test, training, package, manifest, examples
        observed_caches.append(cache)
        return {
            "public": {
                "seed": config.training.seed,
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

    monkeypatch.setattr("radfusion.training.localize._evaluate_member", evaluate)

    result = generate_localization_report(
        [f"test-{seed}" for seed in seeds],
        output_directory=tmp_path / "reports",
    )

    assert result.is_dir()
    assert len(prepared) == 1
    assert observed_caches == [shared_cache, shared_cache, shared_cache]


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
    (public / "summary.json").write_text('{"aggregate": true}\n', encoding="utf-8")
    (public / "summary.md").write_text("# Aggregate localization\n", encoding="utf-8")
    (private / "qualitative_manifest.json").write_text(
        '{"sample_id": "synthetic-sample"}\n', encoding="utf-8"
    )
    (examples / "seed-42-example-01-tp.png").write_bytes(b"synthetic")
    members = [{"filename": "seed-42-example-01-tp.png"}]

    _validate_localization_output_boundaries(
        public,
        private,
        private_members=members,
        forbidden_source_values={"synthetic-sample"},
    )

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
