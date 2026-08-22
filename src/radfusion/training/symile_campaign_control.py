"""Compact pre-test firewall and global-result authority for the Symile campaign."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.calibration import calibration_curve

from radfusion.data.errors import ManifestBuildError
from radfusion.data.symile_schemas import LABEL_POLICY_VERSION, TASK_ID
from radfusion.training.symile_ecg_extension_result import (
    ValidatedEcgExtensionResult,
    require_validated_ecg_extension_result,
)
from radfusion.training.symile_families import (
    FINAL_NEURAL_MEMBER_SEEDS,
    FINAL_PACKAGE_COUNT,
    FINAL_PACKAGE_POLICY,
    FINAL_TABULAR_SEED,
    SYMILE_ECG_GATED_FAMILY,
)
from radfusion.training.symile_final_packages import ValidatedFinalPackage, validate_final_package
from radfusion.training.symile_statistics import (
    BOOTSTRAP_POLICY,
    ERROR_REVIEW_POLICY,
    METRIC_POLICY,
    cluster_bootstrap_effect,
    deterministic_error_cases,
    final_member_mean_logit_ensemble,
    headline_metric_names,
    operating_point_metrics,
    raw_probability_metrics,
)
from radfusion.utils.package_identity import canonical_scientific_id
from radfusion.utils.private_predictions import (
    SYMILE_TEST_INFERENCE_POLICY,
    ValidatedPredictionEvidence,
    validate_prediction_evidence,
)
from radfusion.utils.publication import (
    install_immutable_directory,
    publish_bytes_no_replace,
    staging_directory,
)

if TYPE_CHECKING:
    from radfusion.training.symile_test_data import (
        FrozenSymileTestData,
        HeldOutEvaluationProjection,
    )

PRETEST_FREEZE_PREFIX = "pretest-freeze-"
GLOBAL_RESULT_PREFIX = "global-result-"
_CAPABILITY_GUARD = object()
_TEST_OPEN_GUARD = object()
GLOBAL_EFFECT_COMPARISONS = (
    ("primary", "cxr_labs_gated", "cxr_densenet"),
    ("concat_vs_cxr", "cxr_labs_concat", "cxr_densenet"),
    ("gated_vs_concat", "cxr_labs_gated", "cxr_labs_concat"),
    ("ecg_vs_gated", SYMILE_ECG_GATED_FAMILY, "cxr_labs_gated"),
)
RELIABILITY_POLICY = {"bins": 10, "strategy": "uniform", "role": "descriptive"}


@dataclass(frozen=True)
class ValidatedPretestFreeze:
    directory: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    _guard: object

    @property
    def freeze_id(self) -> str:
        return str(self.manifest["pretest_freeze_id"])


@dataclass(frozen=True)
class ValidatedTestOpenRecord:
    path: Path
    freeze_id: str
    document_sha256: str
    _guard: object


@dataclass(frozen=True)
class ValidatedGlobalResult:
    directory: Path
    manifest: Mapping[str, Any]

    @property
    def result_id(self) -> str:
        return str(self.manifest["global_result_id"])


def publish_pretest_freeze(
    *,
    control_root: str | Path,
    bundle: Mapping[str, str],
    task: Mapping[str, str],
    ecg_extension_result: ValidatedEcgExtensionResult,
    final_packages: Sequence[ValidatedFinalPackage],
    neural_inference_runtime: Mapping[str, object],
    science_git_commit: str,
    dependency_lock_sha256: str,
) -> ValidatedPretestFreeze:
    development = require_validated_ecg_extension_result(ecg_extension_result)
    packages = _validated_packages(
        final_packages,
        expected_git_commit=science_git_commit,
        expected_dependency_lock_sha256=dependency_lock_sha256,
        final_family_authorities=development.manifest["final_family_authorities"],
    )
    semantic = {
        "bundle": dict(bundle),
        "task": dict(task),
        "ecg_extension_result_id": development.result_id,
        "final_packages": [
            {"package_id": package.package_id, "manifest_sha256": package.manifest_sha256}
            for package in packages
        ],
        "held_out_policy": {
            "metric_policy": METRIC_POLICY,
            "bootstrap": BOOTSTRAP_POLICY,
            "neural_inference_runtime": dict(neural_inference_runtime),
        },
        "primary_thresholds": development.manifest["primary_thresholds"],
        "science_git_commit": science_git_commit,
        "dependency_lock_sha256": dependency_lock_sha256,
    }
    freeze_id = canonical_scientific_id(PRETEST_FREEZE_PREFIX, semantic)
    document = {
        "pretest_freeze_schema_version": 1,
        "pretest_freeze_id": freeze_id,
        **semantic,
    }
    destination = Path(control_root) / "freezes" / freeze_id

    def validator(path: Path, **kwargs: object) -> ValidatedPretestFreeze:
        return validate_pretest_freeze(
            path,
            ecg_extension_result=development,
            final_packages=packages,
            **kwargs,
        )

    _publish_json_object(destination, document, validator)
    return validator(destination)


def validate_pretest_freeze(
    directory: str | Path,
    *,
    ecg_extension_result: ValidatedEcgExtensionResult,
    final_packages: Sequence[ValidatedFinalPackage],
    enforce_directory_name: bool = True,
) -> ValidatedPretestFreeze:
    root, raw, document = _validate_pretest_freeze_document(
        directory, enforce_directory_name=enforce_directory_name
    )
    development = require_validated_ecg_extension_result(ecg_extension_result)
    packages = _validated_packages(
        final_packages,
        expected_git_commit=str(document["science_git_commit"]),
        expected_dependency_lock_sha256=str(document["dependency_lock_sha256"]),
        final_family_authorities=development.manifest["final_family_authorities"],
    )
    expected_packages = [
        {"package_id": package.package_id, "manifest_sha256": package.manifest_sha256}
        for package in packages
    ]
    if (
        document["ecg_extension_result_id"] != development.result_id
        or document["final_packages"] != expected_packages
        or document["primary_thresholds"] != development.manifest["primary_thresholds"]
    ):
        raise ManifestBuildError("Pre-test freeze referenced authorities are invalid")
    package_dataset = packages[0].manifest["input"]["dataset"]
    package_task = packages[0].manifest["input"]["task"]
    expected_bundle = {
        "bundle_id": package_dataset["bundle_id"],
        "bundle_manifest_sha256": packages[0].manifest["bundle_manifest_sha256"],
        "split_assignment_id": package_dataset["split_assignment_id"],
    }
    if document["bundle"] != expected_bundle or document["task"] != package_task:
        raise ManifestBuildError("Pre-test freeze data or task authority is invalid")
    return ValidatedPretestFreeze(
        root, document, hashlib.sha256(raw).hexdigest(), _CAPABILITY_GUARD
    )


def _validate_pretest_freeze_document(
    directory: str | Path,
    *,
    enforce_directory_name: bool = True,
) -> tuple[Path, bytes, dict[str, Any]]:
    root, raw, document = _read_single_json(directory)
    required = {
        "pretest_freeze_schema_version",
        "pretest_freeze_id",
        "bundle",
        "task",
        "ecg_extension_result_id",
        "final_packages",
        "held_out_policy",
        "primary_thresholds",
        "science_git_commit",
        "dependency_lock_sha256",
    }
    schema_version = document.get("pretest_freeze_schema_version")
    if (
        set(document) != required
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ManifestBuildError("Pre-test freeze contract is invalid")
    _validate_freeze_fields(document)
    semantic = {
        key: document[key]
        for key in required - {"pretest_freeze_schema_version", "pretest_freeze_id"}
    }
    expected = canonical_scientific_id(PRETEST_FREEZE_PREFIX, semantic)
    if document["pretest_freeze_id"] != expected or (
        enforce_directory_name and root.name != expected
    ):
        raise ManifestBuildError("Pre-test freeze identity is invalid")
    return root, raw, document


def create_or_validate_test_open_record(
    *, control_root: str | Path, capability: ValidatedPretestFreeze
) -> ValidatedTestOpenRecord:
    capability = _require_capability(capability)
    root = Path(control_root)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ManifestBuildError("Official-test control root is invalid")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "test-open.json"
    meaning = {
        "test_open_schema_version": 1,
        "freeze_id": capability.freeze_id,
        "bundle_id": capability.manifest["bundle"]["bundle_id"],
        "split_assignment_id": capability.manifest["bundle"]["split_assignment_id"],
        "science_git_commit": capability.manifest["science_git_commit"],
        "dependency_lock_sha256": capability.manifest["dependency_lock_sha256"],
    }
    if not path.exists():
        document = {
            **meaning,
            "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        encoded = (
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
        publish_bytes_no_replace(path, encoded)
    raw, observed = _read_test_open_record(path)
    schema_version = observed.get("test_open_schema_version")
    if (
        set(observed) != {*meaning, "created_at_utc"}
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
        or {key: observed[key] for key in meaning} != meaning
        or not _valid_utc_timestamp(observed["created_at_utc"])
    ):
        raise ManifestBuildError("Official-test opening is already bound to another freeze")
    return ValidatedTestOpenRecord(
        path,
        capability.freeze_id,
        hashlib.sha256(raw).hexdigest(),
        _TEST_OPEN_GUARD,
    )


def _read_test_open_record(path: Path) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ManifestBuildError("Official-test opening record is invalid")
    try:
        raw = path.read_bytes()
        observed = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestBuildError("Official-test opening record is malformed") from exc
    if not isinstance(observed, dict):
        raise ManifestBuildError("Official-test opening record is malformed")
    return raw, observed


def materialize_official_test[T](
    capability: ValidatedPretestFreeze,
    record: ValidatedTestOpenRecord,
    operation: Callable[[], T],
) -> T:
    capability = _require_capability(capability)
    _require_test_open_record(record, capability)
    return operation()


def validated_pretest_freeze_manifest(
    capability: ValidatedPretestFreeze,
) -> Mapping[str, Any]:
    return _require_capability(capability).manifest


def publish_global_result(
    *,
    report_root: str | Path,
    capability: ValidatedPretestFreeze,
    predictions: Sequence[ValidatedPredictionEvidence],
    final_packages: Sequence[ValidatedFinalPackage],
    test_data: FrozenSymileTestData,
) -> ValidatedGlobalResult:
    capability = _require_capability(capability)
    packages = _validated_frozen_packages(capability, final_packages)
    projection = _frozen_evaluation_projection(capability, test_data)
    evidence = _validated_predictions(capability, packages, predictions, projection)
    claims = _derive_global_claims(capability, packages, evidence, projection)
    semantic = {
        "pretest_freeze_id": capability.freeze_id,
        "prediction_ids": [item.prediction_id for item in evidence],
        "evaluation_policy": _global_evaluation_policy(),
    }
    result_id = canonical_scientific_id(GLOBAL_RESULT_PREFIX, semantic)
    document = {"global_result_schema_version": 1, "global_result_id": result_id, **semantic}
    destination = Path(report_root) / "global-results" / result_id
    stage = staging_directory(destination)
    try:
        (stage / "manifest.json").write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (stage / "claims.json").write_text(
            json.dumps(claims, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (stage / "reliability.svg").write_text(
            _reliability_svg(claims["reliability_curves"]),
            encoding="utf-8",
        )

        def validator(path: Path, **kwargs: object) -> ValidatedGlobalResult:
            return validate_global_result(
                path,
                capability=capability,
                predictions=evidence,
                final_packages=packages,
                test_data=test_data,
                **kwargs,
            )

        install_immutable_directory(stage, destination, validator)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return validate_global_result(
        destination,
        capability=capability,
        predictions=evidence,
        final_packages=packages,
        test_data=test_data,
    )


def validate_global_result(
    directory: str | Path,
    *,
    capability: ValidatedPretestFreeze,
    predictions: Sequence[ValidatedPredictionEvidence],
    final_packages: Sequence[ValidatedFinalPackage],
    test_data: FrozenSymileTestData,
    enforce_directory_name: bool = True,
) -> ValidatedGlobalResult:
    capability = _require_capability(capability)
    root = Path(directory)
    if (
        root.is_symlink()
        or not root.is_dir()
        or {item.name for item in root.iterdir()}
        != {
            "manifest.json",
            "claims.json",
            "reliability.svg",
        }
    ):
        raise ManifestBuildError("Global-result directory is invalid")
    document = json.loads((root / "manifest.json").read_bytes())
    required = {
        "global_result_schema_version",
        "global_result_id",
        "pretest_freeze_id",
        "prediction_ids",
        "evaluation_policy",
    }
    schema_version = (
        document.get("global_result_schema_version") if isinstance(document, dict) else None
    )
    if (
        set(document) != required
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
        or len(document["prediction_ids"]) != FINAL_PACKAGE_COUNT
        or document["pretest_freeze_id"] != capability.freeze_id
        or document["evaluation_policy"] != _global_evaluation_policy()
    ):
        raise ManifestBuildError("Global-result contract is invalid")
    semantic = {
        key: document[key]
        for key in required - {"global_result_schema_version", "global_result_id"}
    }
    expected = canonical_scientific_id(GLOBAL_RESULT_PREFIX, semantic)
    if document["global_result_id"] != expected or (
        enforce_directory_name and root.name != expected
    ):
        raise ManifestBuildError("Global-result identity is invalid")
    packages = _validated_frozen_packages(capability, final_packages)
    projection = _frozen_evaluation_projection(capability, test_data)
    evidence = _validated_predictions(capability, packages, predictions, projection)
    if document["prediction_ids"] != [item.prediction_id for item in evidence]:
        raise ManifestBuildError("Global-result prediction order is invalid")
    expected_claims = _derive_global_claims(capability, packages, evidence, projection)
    observed_claims = json.loads((root / "claims.json").read_bytes())
    if observed_claims != expected_claims:
        raise ManifestBuildError("Global-result claims do not rederive from prediction evidence")
    if (root / "reliability.svg").read_text(encoding="utf-8") != _reliability_svg(
        expected_claims["reliability_curves"]
    ):
        raise ManifestBuildError("Global-result reliability plot does not rederive from claims")
    return ValidatedGlobalResult(root, document)


def publish_error_review(
    *,
    private_root: str | Path,
    capability: ValidatedPretestFreeze,
    predictions: Sequence[ValidatedPredictionEvidence],
    final_packages: Sequence[ValidatedFinalPackage],
    test_data: FrozenSymileTestData,
) -> Path:
    """Publish a restricted regenerable derivative, never a global-result authority."""
    document = _error_review_document(capability, predictions, final_packages, test_data)
    encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = Path(private_root) / "error-review" / "symile" / f"{document['error_review_id']}.json"
    publish_bytes_no_replace(path, encoded)
    if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
        raise ManifestBuildError("Private error review conflicts with existing content")
    return path


def validate_error_review(
    path: str | Path,
    *,
    capability: ValidatedPretestFreeze,
    predictions: Sequence[ValidatedPredictionEvidence],
    final_packages: Sequence[ValidatedFinalPackage],
    test_data: FrozenSymileTestData,
) -> None:
    """Independently rederive private cases from recursively validated evidence."""
    expected = _error_review_document(capability, predictions, final_packages, test_data)
    source = Path(path)
    encoded = (json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if (
        source.is_symlink()
        or not source.is_file()
        or source.name != f"{expected['error_review_id']}.json"
        or source.read_bytes() != encoded
    ):
        raise ManifestBuildError("Private error review does not rederive from evidence")


def _error_review_document(capability, predictions, final_packages, test_data):
    packages = _validated_frozen_packages(capability, final_packages)
    projection = _frozen_evaluation_projection(capability, test_data)
    evidence = _validated_predictions(capability, packages, predictions, projection)
    primary = _predictor_views(packages, evidence)["cxr_labs_gated"]
    threshold = capability.manifest["primary_thresholds"]["youden_j"]
    semantic = {
        "pretest_freeze_id": capability.freeze_id,
        "predictor": "cxr_labs_gated",
        "prediction_ids": [
            item.prediction_id
            for package, item in zip(packages, evidence, strict=True)
            if package.manifest["input"]["family"]["family_id"] == "cxr_labs_gated"
        ],
        "operating_point": {"name": "youden_j", "threshold": threshold},
        "ranking_policy": ERROR_REVIEW_POLICY,
        "cases": deterministic_error_cases(
            primary[["sample_id", "target", "probability"]], threshold
        ),
    }
    return {
        "error_review_schema_version": 1,
        "error_review_id": canonical_scientific_id("error-review-", semantic),
        **semantic,
    }


def _validated_predictions(
    capability: ValidatedPretestFreeze,
    packages: Sequence[ValidatedFinalPackage],
    predictions: Sequence[ValidatedPredictionEvidence],
    projection: HeldOutEvaluationProjection,
) -> tuple[ValidatedPredictionEvidence, ...]:
    from radfusion.training.symile_test_data import validate_prediction_against_test_projection

    if len(predictions) != FINAL_PACKAGE_COUNT:
        raise ManifestBuildError("Global result requires fourteen prediction objects")
    values = tuple(
        validate_prediction_evidence(
            item.directory,
            expected_prediction_id=item.prediction_id,
            enforce_directory_name=False,
        )
        for item in predictions
    )
    frozen_package_ids = tuple(package.package_id for package in packages)
    values = tuple(
        validate_prediction_against_test_projection(
            item,
            capability=capability,
            projection=projection,
            expected_package_ids=frozen_package_ids,
        )
        for item in values
    )
    for package, evidence in zip(packages, values, strict=True):
        if (
            evidence.manifest["model_package_id"] != package.package_id
            or evidence.manifest.get("authorized_by_pretest_freeze_id") != capability.freeze_id
        ):
            raise ManifestBuildError("Global-result prediction lineage is invalid")
    if len({item.prediction_id for item in values}) != FINAL_PACKAGE_COUNT:
        raise ManifestBuildError("Global-result prediction membership contains duplicates")
    return values


def _global_evaluation_policy() -> dict[str, object]:
    return {
        "metric_policy": METRIC_POLICY,
        "predictor_views": list(FINAL_PACKAGE_POLICY),
        "neural_ensemble": (
            f"ordered-seed-{'-'.join(str(seed) for seed in FINAL_NEURAL_MEMBER_SEEDS)}"
            "-mean-logit-then-sigmoid-v1"
        ),
        "raw_probability_metrics": [
            *headline_metric_names(),
            "calibration_slope",
            "calibration_intercept",
        ],
        "paired_effects": [name for name, _, _ in GLOBAL_EFFECT_COMPARISONS],
        "bootstrap": dict(BOOTSTRAP_POLICY),
        "primary_operating_points": ["youden_j", "target_sensitivity"],
        "reliability_plot": dict(RELIABILITY_POLICY),
    }


def _derive_global_claims(
    capability: ValidatedPretestFreeze,
    packages: Sequence[ValidatedFinalPackage],
    predictions: Sequence[ValidatedPredictionEvidence],
    projection: HeldOutEvaluationProjection,
) -> dict[str, object]:
    views = _predictor_views(packages, predictions)
    subjects = _validated_projection_subjects(capability, projection, views["labs_logistic"])
    effects: dict[str, object] = {}
    for name, candidate, comparator in GLOBAL_EFFECT_COMPARISONS:
        frame = (
            views[candidate]
            .merge(
                views[comparator][["sample_id", "probability"]],
                on="sample_id",
                suffixes=("_candidate", "_comparator"),
            )
            .merge(subjects, on="sample_id", validate="one_to_one")
            .rename(
                columns={
                    "probability_candidate": "candidate",
                    "probability_comparator": "comparator",
                }
            )
        )
        effects[name] = {
            metric: cluster_bootstrap_effect(
                frame, candidate="candidate", comparator="comparator", metric=metric
            )
            for metric in headline_metric_names()
        }
    primary = views["cxr_labs_gated"]
    operating_points = {
        name: operating_point_metrics(primary["target"], primary["probability"], float(threshold))
        for name, threshold in capability.manifest["primary_thresholds"].items()
    }
    return {
        "predictor_views": {
            family: raw_probability_metrics(frame["target"], frame["probability"])
            for family, frame in views.items()
        },
        "reliability_curves": {
            family: _reliability_curve(frame["target"], frame["probability"])
            for family, frame in views.items()
        },
        "paired_effects": effects,
        "primary_operating_points": operating_points,
    }


def _reliability_curve(targets: pd.Series, probabilities: pd.Series) -> dict[str, list[float]]:
    observed, predicted = calibration_curve(
        targets,
        probabilities,
        n_bins=int(RELIABILITY_POLICY["bins"]),
        strategy=str(RELIABILITY_POLICY["strategy"]),
    )
    return {
        "mean_predicted_probability": [float(value) for value in predicted],
        "observed_positive_fraction": [float(value) for value in observed],
    }


def _reliability_svg(curves: object) -> str:
    if not isinstance(curves, Mapping) or set(curves) != set(FINAL_PACKAGE_POLICY):
        raise ManifestBuildError("Reliability-curve membership is invalid")
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="600" viewBox="0 0 720 600">',
        '<rect width="720" height="600" fill="white"/>',
        '<text x="300" y="30" text-anchor="middle" font-family="sans-serif" '
        'font-size="18">Raw reliability curves (descriptive)</text>',
        '<line x1="70" y1="530" x2="530" y2="70" stroke="#888" stroke-dasharray="5 5"/>',
        '<line x1="70" y1="530" x2="530" y2="530" stroke="black"/>',
        '<line x1="70" y1="530" x2="70" y2="70" stroke="black"/>',
        '<text x="300" y="575" text-anchor="middle" font-family="sans-serif" '
        'font-size="14">Mean predicted probability</text>',
        '<text x="18" y="300" text-anchor="middle" transform="rotate(-90 18 300)" '
        'font-family="sans-serif" font-size="14">Observed positive fraction</text>',
    ]
    for index, family in enumerate(FINAL_PACKAGE_POLICY):
        curve = curves[family]
        if not isinstance(curve, Mapping) or set(curve) != {
            "mean_predicted_probability",
            "observed_positive_fraction",
        }:
            raise ManifestBuildError("Reliability-curve schema is invalid")
        predicted = curve["mean_predicted_probability"]
        observed = curve["observed_positive_fraction"]
        if (
            not isinstance(predicted, list)
            or not isinstance(observed, list)
            or not predicted
            or len(predicted) != len(observed)
        ):
            raise ManifestBuildError("Reliability-curve values are invalid")
        points = " ".join(
            f"{70 + 460 * float(x):.3f},{530 - 460 * float(y):.3f}"
            for x, y in zip(predicted, observed, strict=True)
        )
        color = colors[index]
        lines.extend(
            (
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>',
                f'<line x1="555" y1="{90 + 28 * index}" x2="575" '
                f'y2="{90 + 28 * index}" stroke="{color}" stroke-width="2"/>',
                f'<text x="582" y="{95 + 28 * index}" font-family="sans-serif" '
                f'font-size="12">{family}</text>',
            )
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _predictor_views(
    packages: Sequence[ValidatedFinalPackage],
    predictions: Sequence[ValidatedPredictionEvidence],
) -> dict[str, pd.DataFrame]:
    grouped: dict[str, list[pd.DataFrame]] = {}
    for package, evidence in zip(packages, predictions, strict=True):
        family = str(package.manifest["input"]["family"]["family_id"])
        frame = evidence.predictions.to_pandas()
        frame["member_seed"] = package.manifest["seed_policy"]
        grouped.setdefault(family, []).append(frame)
    if set(grouped) != set(FINAL_PACKAGE_POLICY):
        raise ManifestBuildError("Global result requires exactly six predictor views")
    views = {
        family: (
            frames[0].loc[:, ["sample_id", "target", "logit", "probability"]]
            if len(frames) == 1
            else final_member_mean_logit_ensemble(
                pd.concat(frames, ignore_index=True).loc[
                    :, ["sample_id", "target", "logit", "member_seed"]
                ]
            )
        )
        for family, frames in grouped.items()
    }
    reference = views["labs_logistic"].loc[:, ["sample_id", "target"]].reset_index(drop=True)
    if any(
        not frame.loc[:, ["sample_id", "target"]].reset_index(drop=True).equals(reference)
        for frame in views.values()
    ):
        raise ManifestBuildError("Global-result predictor views are not exactly aligned")
    return views


def _validated_projection_subjects(
    capability: ValidatedPretestFreeze,
    projection: HeldOutEvaluationProjection,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    try:
        frame = projection.frame()
        lineage = (
            projection.dataset_id,
            projection.bundle_id,
            projection.split_assignment_id,
            projection.task_id,
            projection.label_policy_version,
            projection.scope,
            projection.inference_policy,
            projection.freeze_id,
        )
    except (AttributeError, TypeError) as exc:
        raise ManifestBuildError("Global-result held-out projection is invalid") from exc
    frozen_bundle = capability.manifest["bundle"]
    frozen_task = capability.manifest["task"]
    if (
        lineage
        != (
            "symile",
            frozen_bundle["bundle_id"],
            frozen_bundle["split_assignment_id"],
            frozen_task["task_id"],
            frozen_task["label_policy_version"],
            "test",
            SYMILE_TEST_INFERENCE_POLICY,
            capability.freeze_id,
        )
        or not isinstance(frame, pd.DataFrame)
        or tuple(frame.columns) != ("sample_id", "target", "subject_id")
        or frame.empty
        or frame.isna().any().any()
        or frame["sample_id"].duplicated().any()
        or not pd.api.types.is_integer_dtype(frame["subject_id"].dtype)
    ):
        raise ManifestBuildError("Global-result held-out projection is invalid")
    ordered = frame.sort_values("sample_id", kind="stable").reset_index(drop=True)
    if not ordered.loc[:, ["sample_id", "target"]].equals(
        reference.loc[:, ["sample_id", "target"]].reset_index(drop=True)
    ):
        raise ManifestBuildError(
            "Global-result held-out projection differs from prediction evidence"
        )
    return ordered.loc[:, ["sample_id", "subject_id"]]


def _frozen_evaluation_projection(
    capability: ValidatedPretestFreeze,
    test_data: FrozenSymileTestData,
) -> HeldOutEvaluationProjection:
    from radfusion.training.symile_test_data import FrozenSymileTestData

    if (
        not isinstance(test_data, FrozenSymileTestData)
        or getattr(test_data, "_capability", None) is not capability
    ):
        raise ManifestBuildError(
            "Global result requires official test data authorized by the same pre-test freeze"
        )
    return test_data.evaluation_projection()


def _validated_frozen_packages(
    capability: ValidatedPretestFreeze,
    packages: Sequence[ValidatedFinalPackage],
) -> tuple[ValidatedFinalPackage, ...]:
    values = _validated_packages(packages)
    observed = [
        {"package_id": package.package_id, "manifest_sha256": package.manifest_sha256}
        for package in values
    ]
    if observed != capability.manifest["final_packages"]:
        raise ManifestBuildError("Global result packages differ from the pre-test freeze")
    return values


def _validated_packages(
    packages: Sequence[ValidatedFinalPackage],
    *,
    expected_git_commit: str | None = None,
    expected_dependency_lock_sha256: str | None = None,
    final_family_authorities: Mapping[str, object] | None = None,
) -> tuple[ValidatedFinalPackage, ...]:
    values = tuple(
        validate_final_package(
            package.directory,
            expected_package_id=package.package_id,
            enforce_directory_name=False,
        )
        for package in packages
    )
    by_family: dict[str, list[ValidatedFinalPackage]] = {}
    for package in values:
        family = package.manifest["input"]["family"]["family_id"]
        by_family.setdefault(family, []).append(package)
        if final_family_authorities is not None:
            authority = final_family_authorities.get(family)
            if (
                not isinstance(authority, Mapping)
                or package.manifest["family_development_id"] != authority.get("development_id")
                or package.manifest["final_training_budget"]
                != authority.get("final_training_budget")
                or package.manifest["input"] != authority.get("final_input")
                or package.manifest["pretrained_scientific_identity"]
                != authority.get("pretrained_scientific_identity")
            ):
                raise ManifestBuildError(
                    "Final package differs from its development-derived family authority"
                )
        if expected_git_commit is not None:
            provenance = package.manifest["execution_provenance"]
            if (
                provenance.get("git_commit") != expected_git_commit
                or provenance.get("dependency_lock_sha256") != expected_dependency_lock_sha256
                or provenance.get("git_dirty") is not False
            ):
                raise ManifestBuildError("Final package execution provenance differs from freeze")
    expected = {family: int(policy["members"]) for family, policy in FINAL_PACKAGE_POLICY.items()}
    if (
        len(values) != FINAL_PACKAGE_COUNT
        or {family: len(items) for family, items in by_family.items()} != expected
    ):
        raise ManifestBuildError("Pre-test freeze requires the exact fourteen-package authority")
    expected_order = [
        family
        for family, policy in FINAL_PACKAGE_POLICY.items()
        for _ in range(int(policy["members"]))
    ]
    if [package.manifest["input"]["family"]["family_id"] for package in values] != expected_order:
        raise ManifestBuildError("Final packages are not in canonical family order")
    datasets = [package.manifest["input"]["dataset"] for package in values]
    tasks = [package.manifest["input"]["task"] for package in values]
    manifest_hashes = [package.manifest["bundle_manifest_sha256"] for package in values]
    if (
        any(dataset != datasets[0] for dataset in datasets[1:])
        or any(task != tasks[0] for task in tasks[1:])
        or any(value != manifest_hashes[0] for value in manifest_hashes[1:])
    ):
        raise ManifestBuildError("Final packages do not share one dataset and task authority")
    cxr_by_seed = {
        package.manifest["seed_policy"]: package.package_id for package in by_family["cxr_densenet"]
    }
    expected_seeds = set(FINAL_NEURAL_MEMBER_SEEDS)
    if set(cxr_by_seed) != expected_seeds:
        raise ManifestBuildError("Final CXR membership has invalid seeds")
    cxr_pretrained = [
        package.manifest["pretrained_scientific_identity"] for package in by_family["cxr_densenet"]
    ]
    if cxr_pretrained[0] is None or any(
        identity != cxr_pretrained[0] for identity in cxr_pretrained[1:]
    ):
        raise ManifestBuildError("Final CXR members have inconsistent pretrained identities")
    for family, items in by_family.items():
        if len(items) == 1:
            if items[0].manifest["seed_policy"] != FINAL_TABULAR_SEED:
                raise ManifestBuildError("Final tabular seed policy is invalid")
            continue
        seeds = [item.manifest["seed_policy"] for item in items]
        if seeds != list(FINAL_NEURAL_MEMBER_SEEDS):
            raise ManifestBuildError("Final neural members are not in canonical seed order")
        if family != "cxr_densenet" and any(
            item.manifest["source_cxr_package_id"] != cxr_by_seed[item.manifest["seed_policy"]]
            for item in items
        ):
            raise ManifestBuildError("Final fusion package lacks same-seed CXR ancestry")
    return values


def _require_capability(value: ValidatedPretestFreeze) -> ValidatedPretestFreeze:
    if not isinstance(value, ValidatedPretestFreeze) or value._guard is not _CAPABILITY_GUARD:
        raise ManifestBuildError("A genuine ValidatedPretestFreeze is required")
    root, raw, document = _validate_pretest_freeze_document(
        value.directory, enforce_directory_name=False
    )
    if (
        root != value.directory
        or document != value.manifest
        or hashlib.sha256(raw).hexdigest() != value.manifest_sha256
    ):
        raise ManifestBuildError("Validated pre-test freeze changed after recursive validation")
    return value


def _require_test_open_record(
    value: ValidatedTestOpenRecord, capability: ValidatedPretestFreeze
) -> ValidatedTestOpenRecord:
    if (
        not isinstance(value, ValidatedTestOpenRecord)
        or value._guard is not _TEST_OPEN_GUARD
        or value.freeze_id != capability.freeze_id
        or value.path.name != "test-open.json"
    ):
        raise ManifestBuildError("Official-test access requires the same-freeze test-open record")
    validated = create_or_validate_test_open_record(
        control_root=value.path.parent,
        capability=capability,
    )
    if (
        validated.path != value.path
        or validated.freeze_id != value.freeze_id
        or validated.document_sha256 != value.document_sha256
    ):
        raise ManifestBuildError("Official-test test-open record changed after validation")
    return value


def _valid_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo == UTC


def _validate_freeze_fields(document: Mapping[str, Any]) -> None:
    bundle = document["bundle"]
    task = document["task"]
    packages = document["final_packages"]
    thresholds = document["primary_thresholds"]
    held_out = document["held_out_policy"]
    neural_runtime = (
        held_out.get("neural_inference_runtime") if isinstance(held_out, Mapping) else None
    )
    runtime_keys = {
        "device_type",
        "autocast_dtype",
        "cuda_runtime_version",
        "cudnn_version",
        "gpu_device_name",
        "gpu_compute_capability",
    }
    device_type = neural_runtime.get("device_type") if isinstance(neural_runtime, Mapping) else None
    autocast_dtype = (
        neural_runtime.get("autocast_dtype") if isinstance(neural_runtime, Mapping) else None
    )
    cuda_values = (
        tuple(neural_runtime.get(key) for key in runtime_keys - {"device_type", "autocast_dtype"})
        if isinstance(neural_runtime, Mapping)
        else ()
    )
    cudnn_version = (
        neural_runtime.get("cudnn_version") if isinstance(neural_runtime, Mapping) else None
    )
    capability = (
        neural_runtime.get("gpu_compute_capability")
        if isinstance(neural_runtime, Mapping)
        else None
    )
    if (
        not isinstance(bundle, Mapping)
        or set(bundle) != {"bundle_id", "bundle_manifest_sha256", "split_assignment_id"}
        or not _identity(bundle["bundle_id"], "bundle-")
        or not _sha256(bundle["bundle_manifest_sha256"])
        or not _identity(bundle["split_assignment_id"], "split-assignment-")
        or not isinstance(task, Mapping)
        or task
        != {
            "task_id": TASK_ID,
            "label_policy_version": LABEL_POLICY_VERSION,
        }
        or not _identity(document["ecg_extension_result_id"], "ecg-extension-result-")
        or not isinstance(packages, list)
        or len(packages) != FINAL_PACKAGE_COUNT
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"package_id", "manifest_sha256"}
            or not _identity(item["package_id"], "final-package-")
            or not _sha256(item["manifest_sha256"])
            for item in packages
        )
        or len({item["package_id"] for item in packages}) != FINAL_PACKAGE_COUNT
        or not isinstance(thresholds, Mapping)
        or set(thresholds) != {"youden_j", "target_sensitivity"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in thresholds.values()
        )
        or not isinstance(neural_runtime, Mapping)
        or set(neural_runtime) != runtime_keys
        or device_type not in {"cpu", "cuda"}
        or autocast_dtype not in {None, "float16"}
        or (
            device_type == "cpu"
            and (autocast_dtype is not None or any(value is not None for value in cuda_values))
        )
        or (
            device_type == "cuda"
            and (
                not isinstance(neural_runtime["cuda_runtime_version"], str)
                or not neural_runtime["cuda_runtime_version"]
                or not isinstance(neural_runtime["gpu_device_name"], str)
                or not neural_runtime["gpu_device_name"]
                or isinstance(cudnn_version, bool)
                or (cudnn_version is not None and not isinstance(cudnn_version, int))
                or not isinstance(capability, list)
                or len(capability) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in capability
                )
            )
        )
        or held_out
        != {
            "metric_policy": METRIC_POLICY,
            "bootstrap": BOOTSTRAP_POLICY,
            "neural_inference_runtime": dict(neural_runtime),
        }
        or not _git_commit(document["science_git_commit"])
        or not _sha256(document["dependency_lock_sha256"])
    ):
        raise ManifestBuildError("Pre-test freeze fields are invalid")


def _identity(value: object, prefix: str) -> bool:
    return isinstance(value, str) and value.startswith(prefix) and _sha256(value[len(prefix) :])


def _sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _publish_json_object(
    destination: Path,
    document: Mapping[str, object],
    validator: Callable[[Path], object],
) -> None:
    stage = staging_directory(destination)
    try:
        (stage / "manifest.json").write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        install_immutable_directory(stage, destination, validator)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _read_single_json(directory: str | Path) -> tuple[Path, bytes, dict[str, Any]]:
    root = Path(directory)
    if (
        root.is_symlink()
        or not root.is_dir()
        or {path.name for path in root.iterdir()} != {"manifest.json"}
    ):
        raise ManifestBuildError("Scientific control object directory is invalid")
    raw = (root / "manifest.json").read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ManifestBuildError("Scientific control manifest is invalid")
    return root, raw, document
