"""Generate an explicit three-seed RSNA CXR Grad-CAM localization report."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import torch
from mlflow.exceptions import MlflowException
from sqlalchemy.exc import SQLAlchemyError

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.evaluation.gradcam import gradcam_heatmaps, standard_cxr_gradcam_target
from radfusion.evaluation.localization import (
    deterministic_qualitative_selection,
    localization_metrics,
    transform_rsna_box,
    union_box_mask,
)
from radfusion.models.cxr_baseline import ImageDenseNetModel
from radfusion.training.completed_runs import has_matching_training_parent, require_completed_run
from radfusion.training.config import (
    DatasetConfig,
    image_seed_compatibility_sha256,
    load_experiment_config,
)
from radfusion.training.datasets import RsnaDataset, RsnaImageDataset
from radfusion.training.device import resolve_device
from radfusion.training.evaluate_image import verify_image_training_package
from radfusion.training.neural import seed_neural_runtime
from radfusion.training.registry import get_dataset, get_model
from radfusion.training.summarize_seeds import EXPECTED_SEEDS
from radfusion.utils.mlflow_utils import (
    DEFAULT_TRACKING_URI,
    configure_mlflow,
    git_revision,
    uv_lock_sha256,
)
from radfusion.utils.neural_publication import (
    CONFIG_FILENAME,
    load_validated_neural_checkpoint,
    strict_load_checkpoint,
    validate_neural_package_metadata,
)
from radfusion.utils.operational_logging import add_logging_argument, configure_logging
from radfusion.utils.privacy import validate_public_reports
from radfusion.utils.private_predictions import private_root_for_reports
from radfusion.utils.publication import publish_directory, staging_directory

LOCALIZATION_POLICY_VERSION = "rsna-gradcam-union-box-v1"
QUALITATIVE_POLICY_VERSION = "sha256-stratum-order-v1"
LOCALIZATION_FILENAMES = frozenset({"summary.json", "summary.md"})
PRIVATE_LOCALIZATION_FILENAMES = frozenset({"qualitative_manifest.json", "examples"})


def generate_localization_report(
    test_run_ids: Sequence[str],
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    output_directory: str | Path = "reports",
) -> Path:
    """Verify three explicit image test runs and publish localization results."""
    run_ids = tuple(test_run_ids)
    if len(run_ids) != 3 or len(set(run_ids)) != 3:
        raise ValueError("Localization requires exactly three distinct test run IDs")
    client = configure_mlflow(tracking_uri=tracking_uri)
    commit, dirty = git_revision()
    lock_hash = uv_lock_sha256()
    members = []
    for test_run_id in run_ids:
        test_run = client.get_run(test_run_id)
        test = require_completed_run(test_run)
        training_run = client.get_run(test.source_training_run_id)
        training = require_completed_run(training_run)
        if (
            test.modality != "image"
            or test.run_kind != "test_evaluation"
            or test.evaluation_scope != "test"
            or not has_matching_training_parent(test, training)
        ):
            raise ValueError("Localization members must be linked completed image test runs")
        package = Path(training.local_model_path).parent
        manifest = validate_neural_package_metadata(package)
        config = load_experiment_config(package / CONFIG_FILENAME)
        verify_image_training_package(
            training_run,
            config,
            manifest,
            training_run_id=training.run_id,
            evaluator_commit=commit,
            evaluator_dirty=dirty,
            evaluator_lock_hash=lock_hash,
        )
        members.append((test, training, package, manifest, config))
    members.sort(key=lambda value: value[0].integer_seed())
    if tuple(value[0].integer_seed() for value in members) != EXPECTED_SEEDS:
        raise ValueError(f"Localization requires seeds {list(EXPECTED_SEEDS)}")
    compatibility = {image_seed_compatibility_sha256(value[4]) for value in members}
    if len(compatibility) != 1:
        raise ValueError("Localization image runs are not scientifically compatible")
    ordered_run_ids = tuple(value[0].run_id for value in members)
    report_id = "localization-" + hashlib.sha256("\0".join(ordered_run_ids).encode()).hexdigest()
    destination = Path(output_directory) / "rsna" / "localization" / report_id
    private_destination = private_root_for_reports(output_directory) / "localization" / report_id
    if destination.exists() or private_destination.exists():
        raise FileExistsError("Localization outputs already exist")
    stage = staging_directory(destination)
    private_stage = staging_directory(private_destination)
    private_published = False
    try:
        examples = private_stage / "examples"
        examples.mkdir(parents=True)
        results = [
            _evaluate_member(
                test,
                training,
                package,
                manifest,
                config,
                examples=examples,
            )
            for test, training, package, manifest, config in members
        ]
        document = _report_document(report_id, [result["public"] for result in results])
        (stage / "summary.json").write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (stage / "summary.md").write_text(_markdown(document), encoding="utf-8")
        private_members = [item for result in results for item in result["private_examples"]]
        (private_stage / "qualitative_manifest.json").write_text(
            json.dumps(
                {
                    "private_localization_schema_version": 1,
                    "report_id": report_id,
                    "selection_policy": QUALITATIVE_POLICY_VERSION,
                    "examples": private_members,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        _validate_localization_output_boundaries(
            stage,
            private_stage,
            private_members=private_members,
            forbidden_source_values={
                value for result in results for value in result["forbidden_source_values"]
            },
        )
        publish_directory(private_stage, private_destination)
        private_published = True
        publish_directory(stage, destination)
    except BaseException:
        if private_published and private_destination.exists():
            shutil.rmtree(private_destination)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if private_stage.exists():
            shutil.rmtree(private_stage)
    return destination


def _evaluate_member(test, training, package, manifest, config, *, examples: Path):
    checkpoint = load_validated_neural_checkpoint(package, manifest)
    builder = cast(ImageDenseNetModel, get_model(config.model.registry_key))
    model = builder.build_architecture(config.model)
    strict_load_checkpoint(model, checkpoint)
    dataset_adapter = _rsna_localization_dataset(config.dataset)
    localization = dataset_adapter.load_localization_test(
        config.dataset,
        expected_manifest_sha256=manifest["bundle_manifest_sha256"],
    )
    image = config.image
    if image is None or config.dataset.dataset_root is None:
        raise ValueError("Localization package configuration is incomplete")
    runtime = resolve_device(
        image.device,
        mixed_precision=False,
        pin_memory_policy=image.pin_memory_policy,
    )
    model.to(runtime.device)
    target = standard_cxr_gradcam_target(model)
    transform = StandardCxrTransform(
        training=False,
        image_size=int(config.model.parameters["image_size"]),
        rotation_degrees=image.rotation_degrees,
        translation_fraction=image.translation_fraction,
        brightness_jitter=image.brightness_jitter,
        contrast_jitter=image.contrast_jitter,
    )
    dataset = RsnaImageDataset(
        localization.images.test,
        dataset_root=config.dataset.dataset_root,
        partition="test",
        transform=transform,
    )
    seed_neural_runtime(config.training.seed)
    model.eval()
    thresholds = manifest["thresholds"]
    threshold = float(thresholds["youden_j"])
    dimensions = localization.dimensions.set_index("sample_id")
    annotations = {
        sample_id: frame.to_dict(orient="records")
        for sample_id, frame in localization.annotations.groupby("sample_id", sort=True)
    }
    prediction_rows: list[dict[str, object]] = []
    forbidden_source_values: set[str] = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        forbidden_source_values.update((sample["sample_id"], sample["patient_id"]))
        image_tensor = sample["image"].unsqueeze(0).to(runtime.device)
        with torch.inference_mode():
            probability = float(torch.sigmoid(model(image_tensor))[0])
        target_value = int(sample["target"].item())
        predicted = int(probability >= threshold)
        stratum = {(1, 1): "TP", (1, 0): "FN", (0, 1): "FP", (0, 0): "TN"}[
            (target_value, predicted)
        ]
        prediction_rows.append(
            {
                "index": index,
                "sample_id": sample["sample_id"],
                "target": target_value,
                "stratum": stratum,
            }
        )
    selected = deterministic_qualitative_selection(
        prediction_rows,
        policy_version=QUALITATIVE_POLICY_VERSION,
    )
    required_indices = _gradcam_indices(prediction_rows, selected)
    positive_test_sample_count = sum(int(row["target"] == 1) for row in prediction_rows)
    positive_pointing: list[int] = []
    positive_energy: list[float] = []
    zero_maps = 0
    private_examples: list[dict[str, object]] = []
    selected_by_sample = {
        str(row["sample_id"]): stratum for stratum, row in selected.items() if row is not None
    }
    for index in required_indices:
        sample = dataset[index]
        image_tensor = sample["image"].unsqueeze(0).to(runtime.device)
        heatmap = gradcam_heatmaps(model, image_tensor, target_module=target)[0].cpu()
        if int(sample["target"].item()) == 1:
            dimension = dimensions.loc[sample["sample_id"]]
            source_boxes = annotations.get(sample["sample_id"], [])
            pointing, energy, zero = _positive_localization_metrics(
                source_boxes,
                source_rows=int(dimension["image_rows"]),
                source_columns=int(dimension["image_columns"]),
                heatmap=heatmap.numpy(),
            )
            positive_pointing.append(pointing)
            positive_energy.append(energy)
            zero_maps += int(zero)
        stratum = selected_by_sample.get(sample["sample_id"])
        if stratum is not None:
            ordinal = ("TP", "FN", "FP", "TN").index(stratum) + 1
            filename = f"seed-{config.training.seed}-example-{ordinal:02d}-{stratum.lower()}.png"
            _write_overlay(examples / filename, sample["image"], heatmap)
            private_examples.append(
                {
                    "seed": config.training.seed,
                    "stratum": stratum,
                    "sample_id": sample["sample_id"],
                    "test_run_id": test.run_id,
                    "training_run_id": training.run_id,
                    "filename": filename,
                }
            )
    localization_evaluated_count = len(positive_pointing)
    if positive_test_sample_count != localization_evaluated_count:
        raise ValueError("Localization did not account for every positive test sample")
    return {
        "public": {
            "seed": config.training.seed,
            "positive_test_sample_count": positive_test_sample_count,
            "localization_evaluated_count": localization_evaluated_count,
            "zero_heatmap_count": zero_maps,
            "pointing_game_accuracy": statistics.fmean(positive_pointing),
            "mean_activation_energy_inside_union": statistics.fmean(positive_energy),
            "qualitative_strata_present": {
                stratum: selected[stratum] is not None for stratum in ("TP", "FN", "FP", "TN")
            },
        },
        "private_examples": private_examples,
        "forbidden_source_values": forbidden_source_values,
    }


def _rsna_localization_dataset(config: DatasetConfig) -> RsnaDataset:
    """Resolve the concrete RSNA adapter required by localization."""
    if config.registry_key != "rsna":
        raise ValueError("Localization supports only the RSNA dataset")
    return cast(RsnaDataset, get_dataset(config.registry_key))


def _gradcam_indices(
    rows: Sequence[Mapping[str, object]],
    selected: Mapping[str, Mapping[str, object] | None],
) -> tuple[int, ...]:
    """Return positives plus only the selected negative qualitative rows."""
    required = {int(row["index"]) for row in rows if row["target"] == 1}
    for stratum in ("FP", "TN"):
        row = selected[stratum]
        if row is not None:
            required.add(int(row["index"]))
    return tuple(sorted(required))


def _positive_localization_metrics(
    source_boxes: Sequence[Mapping[str, object]],
    *,
    source_rows: int,
    source_columns: int,
    heatmap: np.ndarray,
) -> tuple[int, float, bool]:
    """Map every positive sample's boxes and require a nonempty union."""
    if not source_boxes:
        raise ValueError("Positive localization sample has no source annotation")
    transformed = [
        transform_rsna_box(
            row,
            source_rows=source_rows,
            source_columns=source_columns,
        )
        for row in source_boxes
    ]
    boxes = [box for box in transformed if box is not None]
    if not boxes:
        raise ValueError("Positive localization sample has no box after evaluation crop")
    return localization_metrics(heatmap, union_box_mask(boxes))


def _validate_localization_output_boundaries(
    public_stage: Path,
    private_stage: Path,
    *,
    private_members: Sequence[Mapping[str, object]],
    forbidden_source_values: set[str],
) -> None:
    """Enforce aggregate-only public output and regular private qualitative files."""
    public_entries = list(public_stage.iterdir())
    if {path.name for path in public_entries} != LOCALIZATION_FILENAMES or any(
        path.is_symlink() or not path.is_file() for path in public_entries
    ):
        raise ValueError("Public localization report set is invalid")
    private_entries = list(private_stage.iterdir())
    if {path.name for path in private_entries} != PRIVATE_LOCALIZATION_FILENAMES:
        raise ValueError("Private localization artifact set is incomplete")
    manifest_path = private_stage / "qualitative_manifest.json"
    examples = private_stage / "examples"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or examples.is_symlink()
        or not examples.is_dir()
    ):
        raise ValueError("Private localization entries must be physical files and directories")
    expected_examples = {str(item["filename"]) for item in private_members}
    example_entries = list(examples.iterdir())
    if {path.name for path in example_entries} != expected_examples or any(
        path.is_symlink() or not path.is_file() for path in example_entries
    ):
        raise ValueError("Localization qualitative example set is invalid")
    validate_public_reports(
        (public_stage / name for name in ("summary.json", "summary.md")),
        forbidden_source_values=forbidden_source_values,
    )


def _write_overlay(path: Path, image: torch.Tensor, heatmap: torch.Tensor) -> None:
    pixels = image.squeeze(0).numpy().astype(np.float64)
    minimum, maximum = float(pixels.min()), float(pixels.max())
    display = (
        np.zeros_like(pixels) if maximum == minimum else (pixels - minimum) / (maximum - minimum)
    )
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(display, cmap="gray")
    axis.imshow(heatmap.numpy(), cmap="jet", alpha=0.40, vmin=0.0, vmax=1.0)
    axis.axis("off")
    figure.tight_layout(pad=0)
    figure.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def _report_document(report_id: str, members: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("pointing_game_accuracy", "mean_activation_energy_inside_union")
    return {
        "localization_schema_version": 1,
        "report_id": report_id,
        "policy_version": LOCALIZATION_POLICY_VERSION,
        "gradcam_target": "encoder.backbone.features",
        "threshold_policy": "validation_youden_j_seed_specific",
        "qualitative_selection_policy": QUALITATIVE_POLICY_VERSION,
        "interpretation": (
            "Grad-CAM is a model-behavior localization sanity analysis, not a causal "
            "explanation, lesion segmentation, proof of clinical reasoning or clinical "
            "validity, or evidence that the network reasons like a radiologist."
        ),
        "members": members,
        "aggregates": {
            metric: {
                "mean": statistics.fmean(float(member[metric]) for member in members),
                "sample_standard_deviation": statistics.stdev(
                    float(member[metric]) for member in members
                ),
            }
            for metric in metrics
        },
    }


def _markdown(document: dict[str, Any]) -> str:
    aggregate = document["aggregates"]
    pointing = aggregate["pointing_game_accuracy"]
    energy = aggregate["mean_activation_energy_inside_union"]
    lines = [
        "# RSNA CXR localization",
        "",
        "Grad-CAM targets `encoder.backbone.features`, the final 1024-channel spatial feature "
        "sequence consumed by ReLU and adaptive average pooling.",
        "",
        "Grad-CAM is a model-behavior localization sanity analysis. It is not a causal "
        "explanation, lesion segmentation, proof of clinical reasoning or clinical validity, "
        "or evidence that the network reasons like a radiologist.",
        "",
        "| Seed | Positive test | Evaluated | Zero maps | Pointing game | Activation energy |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for member in document["members"]:
        lines.append(
            f"| {member['seed']} | {member['positive_test_sample_count']} | "
            f"{member['localization_evaluated_count']} | {member['zero_heatmap_count']} | "
            f"{member['pointing_game_accuracy']:.6f} | "
            f"{member['mean_activation_energy_inside_union']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Aggregate | Pointing game | Activation energy |",
            "| --- | ---: | ---: |",
            f"| Mean | {pointing['mean']:.6f} | {energy['mean']:.6f} |",
            f"| Sample SD | {pointing['sample_standard_deviation']:.6f} | "
            f"{energy['sample_standard_deviation']:.6f} |",
            "",
            "Qualitative cases are selected mechanically by SHA-256 order within TP, FN, FP, "
            "and TN strata at each seed's validation-frozen Youden threshold. Real-image "
            "overlays and their traceability manifest are stored only in the private workspace; "
            "missing strata remain absent.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-run-ids", nargs=3, required=True)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--output-directory", type=Path, default=Path("reports"))
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one verified three-seed localization report."""
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        destination = generate_localization_report(
            args.test_run_ids,
            tracking_uri=args.tracking_uri,
            output_directory=args.output_directory,
        )
    except (MlflowException, SQLAlchemyError, OSError, ValueError, KeyError) as exc:
        print(f"Localization failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"report_directory": destination.as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
