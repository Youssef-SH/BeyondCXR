"""Generate an explicit three-seed RSNA CXR Grad-CAM localization report."""

from __future__ import annotations

import argparse
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.rsna_cxr_cache import ValidatedCxrCache
from radfusion.evaluation.gradcam import gradcam_heatmaps, standard_cxr_gradcam_target
from radfusion.evaluation.localization import (
    deterministic_qualitative_selection,
    localization_metrics,
    transform_rsna_box,
    union_box_mask,
)
from radfusion.training.config import (
    ExperimentConfig,
    load_experiment_config,
    require_runtime_seed,
    with_runtime,
)
from radfusion.training.device import resolve_device
from radfusion.training.neural import seed_neural_runtime
from radfusion.training.rsna_datasets import (
    RsnaCachedImageDataset,
    RsnaDataset,
    expected_rsna_cxr_cache_identity,
    prepare_rsna_cxr_cache,
)
from radfusion.training.rsna_evaluation_result import validate_rsna_evaluation
from radfusion.training.rsna_interfaces import RsnaCxrModelImplementation
from radfusion.training.rsna_registry import get_dataset, get_model
from radfusion.training.rsna_seed_summary import EXPECTED_SEEDS
from radfusion.utils.operational_logging import (
    CountProgress,
    add_logging_argument,
    configure_logging,
    get_operational_logger,
    timed_phase,
)
from radfusion.utils.package_identity import (
    canonical_scientific_id,
    package_scientific_config_payload,
)
from radfusion.utils.privacy import validate_public_reports
from radfusion.utils.publication import (
    install_immutable_directory,
    staging_directory,
    validate_path_component,
)
from radfusion.utils.rsna_neural_publication import (
    CONFIG_FILENAME,
    load_validated_neural_checkpoint,
    strict_load_checkpoint,
    validate_neural_package_metadata,
)

LOCALIZATION_POLICY_VERSION = "rsna-gradcam-union-box-v1"
LOCALIZATION_SCHEMA_VERSION = 1
PRIVATE_LOCALIZATION_SCHEMA_VERSION = 1
QUALITATIVE_POLICY_VERSION = "sha256-stratum-order-v1"
GRADCAM_TARGET = "encoder.backbone.features"
LOCALIZATION_THRESHOLD_POLICY = "validation_youden_j_seed_specific"
LOCALIZATION_FILENAMES = frozenset({"summary.json", "summary.md"})
PRIVATE_LOCALIZATION_FILENAMES = frozenset({"qualitative_manifest.json", "examples"})
_LOGGER = get_operational_logger(__name__)


def generate_localization_report(
    evaluation_ids: Sequence[str],
    *,
    output_directory: str | Path = "reports",
    model_directory: str | Path = "models/rsna",
    private_directory: str | Path = "private",
    cache: ValidatedCxrCache | None = None,
) -> Path:
    """Verify three explicit CXR evaluation objects and publish localization results."""
    identities = tuple(evaluation_ids)
    if len(identities) != 3 or len(set(identities)) != 3:
        raise ValueError("Localization requires exactly three distinct evaluation IDs")
    members = []
    for evaluation_id in identities:
        validate_path_component(evaluation_id, "evaluation ID")
        evaluation = validate_rsna_evaluation(
            Path(output_directory) / "rsna/evaluations" / evaluation_id,
            private_root=private_directory,
            model_root=model_directory,
            expected_evaluation_id=evaluation_id,
        )
        package_id = str(evaluation.manifest["model_package_id"])
        package = Path(model_directory) / "packages" / package_id
        manifest = validate_neural_package_metadata(package)
        seed = manifest["training_policy"]["seed"]
        config = with_runtime(load_experiment_config(package / CONFIG_FILENAME), seed=seed)
        if config.family.family_id != "cxr_densenet":
            raise ValueError("Localization requires CXR-only model packages")
        members.append((evaluation_id, package_id, package, manifest, config))
    members.sort(key=lambda value: require_runtime_seed(value[4]))
    if tuple(require_runtime_seed(value[4]) for value in members) != EXPECTED_SEEDS:
        raise ValueError(f"Localization requires seeds {list(EXPECTED_SEEDS)}")
    package_configs = {
        json.dumps(
            package_scientific_config_payload(value[4]), sort_keys=True, separators=(",", ":")
        )
        for value in members
    }
    if len(package_configs) != 1:
        raise ValueError("Localization CXR evaluations are not scientifically compatible")
    reference_config = members[0][4]
    reference_neural = reference_config.neural
    if reference_neural is None:
        raise ValueError("Localization package configuration is incomplete")
    reference_dataset = _rsna_localization_dataset(reference_config)
    reference_transform = StandardCxrTransform(
        training=False,
        policy_version=str(reference_config.preprocessing["cxr_transform_policy"]),
        image_size=int(reference_config.family.parameters["image_size"]),
        rotation_degrees=reference_neural.rotation_degrees,
        translation_fraction=reference_neural.translation_fraction,
        brightness_jitter=reference_neural.brightness_jitter,
        contrast_jitter=reference_neural.contrast_jitter,
    )
    resolved_cache = cache or prepare_rsna_cxr_cache(
        reference_dataset,
        reference_config,
        reference_transform,
    )
    ordered_package_ids = tuple(value[1] for value in members)
    report_id = _localization_id(ordered_package_ids)
    destination = Path(output_directory) / "rsna" / "localization" / report_id
    private_destination = Path(private_directory) / "localization" / report_id
    stage = staging_directory(destination)
    private_stage = staging_directory(private_destination)
    try:
        with timed_phase(_LOGGER, "localization"):
            examples = private_stage / "examples"
            examples.mkdir(parents=True)
            results = [
                _evaluate_member(
                    package_id,
                    package,
                    manifest,
                    config,
                    examples=examples,
                    cache=resolved_cache,
                )
                for _evaluation_id, package_id, package, manifest, config in members
            ]
            public_members = [result["public"] for result in results]
            forbidden_source_values = {
                value for result in results for value in result["forbidden_source_values"]
            }
            document = _report_document(
                report_id,
                ordered_package_ids,
                public_members,
            )
            (stage / "summary.json").write_text(
                json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            (stage / "summary.md").write_text(_markdown(document), encoding="utf-8")
            private_members = [item for result in results for item in result["private_examples"]]
            (private_stage / "qualitative_manifest.json").write_text(
                json.dumps(
                    {
                        "private_localization_schema_version": PRIVATE_LOCALIZATION_SCHEMA_VERSION,
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
                forbidden_source_values=forbidden_source_values,
                expected_report_id=report_id,
                expected_public_members=public_members,
            )
            install_immutable_directory(
                private_stage,
                private_destination,
                lambda candidate, **_: _validate_localization_output_boundaries(
                    stage,
                    Path(candidate),
                    private_members=private_members,
                    forbidden_source_values=forbidden_source_values,
                    expected_report_id=report_id,
                    expected_public_members=public_members,
                ),
            )
            install_immutable_directory(
                stage,
                destination,
                lambda candidate, **_: _validate_localization_output_boundaries(
                    Path(candidate),
                    private_destination,
                    private_members=private_members,
                    forbidden_source_values=forbidden_source_values,
                    expected_report_id=report_id,
                    expected_public_members=public_members,
                ),
            )
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if private_stage.exists():
            shutil.rmtree(private_stage)
    return destination


def _evaluate_member(
    package_id,
    package,
    manifest,
    config,
    *,
    examples: Path,
    cache: ValidatedCxrCache,
):
    checkpoint = load_validated_neural_checkpoint(package, manifest)
    builder = cast(RsnaCxrModelImplementation, get_model(config.family.family_id))
    model = builder.build_architecture(config.family)
    strict_load_checkpoint(model, checkpoint)
    dataset_adapter = _rsna_localization_dataset(config)
    localization = dataset_adapter.load_localization_test(
        config,
        expected_manifest_sha256=manifest["bundle_manifest_sha256"],
    )
    neural = config.neural
    if neural is None or config.runtime.source_root is None:
        raise ValueError("Localization package configuration is incomplete")
    runtime = resolve_device(
        config.runtime.device,
        mixed_precision=False,
        pin_memory_policy=config.runtime.pin_memory_policy,
    )
    model.to(runtime.device)
    target = standard_cxr_gradcam_target(model)
    transform = StandardCxrTransform(
        training=False,
        policy_version=str(config.preprocessing["cxr_transform_policy"]),
        image_size=int(config.family.parameters["image_size"]),
        rotation_degrees=neural.rotation_degrees,
        translation_fraction=neural.translation_fraction,
        brightness_jitter=neural.brightness_jitter,
        contrast_jitter=neural.contrast_jitter,
    )
    expected_cache_identity = expected_rsna_cxr_cache_identity(
        lineage=localization.images.lineage,
        bundle_manifest_sha256=localization.images.bundle_manifest_sha256,
        source_inventory=localization.images.source_inventory,
        transform=transform,
    )
    if cache.source_authentication.as_dict() != manifest["source_authentication"]:
        raise ValueError("Validated cache source authentication differs from CXR package")
    if manifest["runtime_provenance"]["cxr_cache_id"] != cache.identity.cache_id:
        raise ValueError("Validated cache identity differs from the CXR package")
    dataset = RsnaCachedImageDataset(
        localization.images.test,
        cache=cache,
        expected_cache_identity=expected_cache_identity,
        partition="test",
        transform=transform,
        training_seed=require_runtime_seed(config),
    )
    seed = require_runtime_seed(config)
    seed_neural_runtime(seed)
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
    prediction_progress = CountProgress(
        _LOGGER,
        "operation_progress",
        total=len(dataset),
        unit="samples",
        count_interval=250,
        fields={"seed": seed, "operation": "prediction"},
    )
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
        prediction_progress.update(index + 1)
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
    gradcam_progress = CountProgress(
        _LOGGER,
        "operation_progress",
        total=len(required_indices),
        unit="samples",
        count_interval=50,
        fields={"seed": seed, "operation": "gradcam"},
    )
    for completed, index in enumerate(required_indices, start=1):
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
            filename = f"seed-{seed}-example-{ordinal:02d}-{stratum.lower()}.png"
            _write_overlay(examples / filename, sample["image"], heatmap)
            private_examples.append(
                {
                    "seed": seed,
                    "stratum": stratum,
                    "sample_id": sample["sample_id"],
                    "model_package_id": package_id,
                    "filename": filename,
                }
            )
        gradcam_progress.update(completed)
    localization_evaluated_count = len(positive_pointing)
    if positive_test_sample_count != localization_evaluated_count:
        raise ValueError("Localization did not account for every positive test sample")
    return {
        "public": {
            "seed": seed,
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


def _rsna_localization_dataset(config: ExperimentConfig) -> RsnaDataset:
    """Resolve the concrete RSNA adapter required by localization."""
    if config.dataset.dataset_id != "rsna":
        raise ValueError("Localization supports only the RSNA dataset")
    return cast(RsnaDataset, get_dataset(config.dataset.dataset_id))


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
    expected_report_id: str | None = None,
    expected_public_members: Sequence[Mapping[str, object]] | None = None,
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
    public_document = json.loads((public_stage / "summary.json").read_text(encoding="utf-8"))
    private_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_schema_version(
        public_document,
        "localization_schema_version",
        LOCALIZATION_SCHEMA_VERSION,
    )
    _require_schema_version(
        private_document,
        "private_localization_schema_version",
        PRIVATE_LOCALIZATION_SCHEMA_VERSION,
    )
    if not isinstance(private_document, dict) or set(private_document) != {
        "private_localization_schema_version",
        "report_id",
        "selection_policy",
        "examples",
    }:
        raise ValueError("Private localization manifest fields are invalid")
    if private_document["examples"] != list(private_members):
        raise ValueError("Private localization examples differ from the validated evidence")
    expected_public_fields = {
        "localization_schema_version",
        "report_id",
        "model_package_ids",
        "policy_version",
        "gradcam_target",
        "threshold_policy",
        "qualitative_selection_policy",
        "interpretation",
        "members",
        "aggregates",
    }
    if not isinstance(public_document, dict) or set(public_document) != expected_public_fields:
        raise ValueError("Public localization manifest fields are invalid")
    model_package_ids = public_document["model_package_ids"]
    if (
        not isinstance(model_package_ids, list)
        or len(model_package_ids) != 3
        or not all(isinstance(value, str) for value in model_package_ids)
        or len(set(model_package_ids)) != 3
        or any(
            validate_path_component(value, "localization model package ID") != value
            or not value.startswith("model-package-")
            or len(value) != len("model-package-") + 64
            or any(character not in "0123456789abcdef" for character in value[14:])
            for value in model_package_ids
        )
    ):
        raise ValueError("Public localization model-package lineage is invalid")
    computed_report_id = _localization_id(model_package_ids)
    if expected_report_id is not None and expected_report_id != public_document["report_id"]:
        raise ValueError("Localization output differs from the requested identity")
    if (
        public_document["report_id"] != computed_report_id
        or private_document.get("report_id") != computed_report_id
        or public_document["policy_version"] != LOCALIZATION_POLICY_VERSION
        or public_document["gradcam_target"] != GRADCAM_TARGET
        or public_document["threshold_policy"] != LOCALIZATION_THRESHOLD_POLICY
        or public_document["qualitative_selection_policy"] != QUALITATIVE_POLICY_VERSION
        or private_document.get("selection_policy") != QUALITATIVE_POLICY_VERSION
    ):
        raise ValueError("Localization identity or policy is invalid")
    members = public_document["members"]
    if (
        not isinstance(members, list)
        or len(members) != 3
        or [member.get("seed") for member in members] != list(EXPECTED_SEEDS)
    ):
        raise ValueError("Public localization members are invalid")
    if expected_public_members is not None and members != list(expected_public_members):
        raise ValueError("Public localization members differ from the validated results")
    expected_aggregates = _report_document(computed_report_id, model_package_ids, members)[
        "aggregates"
    ]
    if public_document["aggregates"] != expected_aggregates:
        raise ValueError("Public localization aggregates differ from their members")
    if (public_stage / "summary.md").read_text(encoding="utf-8") != _markdown(public_document):
        raise ValueError("Public localization Markdown differs from its manifest")
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


def _report_document(
    report_id: str,
    model_package_ids: Sequence[str],
    members: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = ("pointing_game_accuracy", "mean_activation_energy_inside_union")
    return {
        "localization_schema_version": LOCALIZATION_SCHEMA_VERSION,
        "report_id": report_id,
        "model_package_ids": list(model_package_ids),
        "policy_version": LOCALIZATION_POLICY_VERSION,
        "gradcam_target": GRADCAM_TARGET,
        "threshold_policy": LOCALIZATION_THRESHOLD_POLICY,
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


def _localization_id(model_package_ids: Sequence[str]) -> str:
    return canonical_scientific_id(
        "localization-",
        {
            "model_package_ids": list(model_package_ids),
            "policy_version": LOCALIZATION_POLICY_VERSION,
            "gradcam_target": GRADCAM_TARGET,
            "threshold_policy": LOCALIZATION_THRESHOLD_POLICY,
            "qualitative_selection_policy": QUALITATIVE_POLICY_VERSION,
        },
    )


def _require_schema_version(
    document: object,
    field: str,
    expected: int,
) -> None:
    if not isinstance(document, dict):
        raise ValueError("Localization manifest must be a JSON object")
    value = document.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"Localization field {field} must be schema version {expected}")


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
    parser.add_argument("--evaluation-ids", nargs=3, required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("reports"))
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one verified three-seed localization report."""
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        destination = generate_localization_report(
            args.evaluation_ids,
            output_directory=args.output_directory,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Localization failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"report_directory": destination.as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
