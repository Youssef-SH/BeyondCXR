from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rsna_validation_evidence_test_support import write_synthetic_validation_evidence
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from beyondcxr.data.rsna_metadata_preprocess import (
    SOURCE_FEATURES,
    build_rsna_preprocessor,
    metadata_input_contract,
)
from beyondcxr.training import rsna_compare, rsna_evaluation_result
from beyondcxr.training.config import load_experiment_config
from beyondcxr.training.rsna_compare import (
    COMPARISON_COLUMNS,
    COMPARISON_POLICY_VERSION,
    _write_comparison,
)
from beyondcxr.training.rsna_compare import (
    _comparison_records as _raw_comparison_records,
)
from beyondcxr.training.rsna_compare import (
    regenerate_comparison as _regenerate_comparison,
)
from beyondcxr.training.rsna_compare import (
    validate_comparison as _validate_comparison,
)
from beyondcxr.training.rsna_evaluation_result import (
    _publish_rsna_evaluation,
)
from beyondcxr.training.rsna_evaluation_result import (
    validate_rsna_evaluation as _validate_rsna_evaluation,
)
from beyondcxr.utils.private_predictions import (
    ValidatedHeldOutPublicationAuthority,
    _publish_held_out_prediction_evidence,
    validate_prediction_evidence,
)
from beyondcxr.utils.rsna_model_publication import publish_model_package, threshold_contract
from beyondcxr.utils.skops_io import save_skops

_FREEZE_ID = "rsna-freeze-" + "f" * 64
_TEST_FREEZE = object()
_TEST_EXECUTION = object()


def regenerate_comparison(*args, **kwargs):
    return _regenerate_comparison(
        *args,
        freeze=_TEST_FREEZE,
        execution=_TEST_EXECUTION,
        **kwargs,
    )


def validate_comparison(*args, **kwargs):
    return _validate_comparison(
        *args,
        freeze=_TEST_FREEZE,
        execution=_TEST_EXECUTION,
        **kwargs,
    )


def _comparison_records(*args, **kwargs):
    return _raw_comparison_records(
        *args,
        freeze=_TEST_FREEZE,
        execution=_TEST_EXECUTION,
        **kwargs,
    )


def validate_rsna_evaluation(*args, **kwargs):
    return _validate_rsna_evaluation(
        *args,
        freeze=_TEST_FREEZE,
        execution=_TEST_EXECUTION,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _isolate_derivative_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "beyondcxr.utils.private_predictions._require_held_out_publication_authority",
        lambda authority, values: authority,
    )
    monkeypatch.setattr(
        rsna_evaluation_result,
        "validate_rsna_held_out_prediction",
        lambda evidence, **kwargs: validate_prediction_evidence(
            evidence.directory,
            expected_prediction_id=evidence.prediction_id,
            expected_model_package_id=str(evidence.manifest["model_package_id"]),
            expected_authorizing_freeze_id=_FREEZE_ID,
        ),
    )


def _package(tmp_path: Path, *, variant: int = 0):
    config_path = Path("configs/rsna_metadata_logistic.yaml")
    config = load_experiment_config(config_path, bundle_id="bundle-" + "a" * 64)
    features = pd.DataFrame(
        [
            [40.0, False, "M", "PA", 0.17, 0.17],
            [55.0, False, "F", "AP", 0.18, 0.18],
            [65.0, False, "M", "PA", 0.19, 0.19],
            [75.0, False, "F", "AP", 0.20, 0.20],
        ],
        columns=SOURCE_FEATURES,
    )
    targets = np.asarray([0, variant % 2, 1, 1])
    pipeline = Pipeline(
        [
            ("preprocess", build_rsna_preprocessor()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    ).fit(features, targets)
    serialized = save_skops(pipeline, tmp_path / f"source-{variant}.skops")
    config_bytes = config_path.read_bytes()
    return publish_model_package(
        model_root=tmp_path / "models/rsna",
        serialized_model_path=serialized,
        source_config_bytes=config_bytes,
        validation_evidence_path=write_synthetic_validation_evidence(
            tmp_path / f"validation-evidence-{variant}.json", config_path, seed=42
        ),
        manifest={
            "bundle_id": config.dataset.bundle_id,
            "task_id": config.task.task_id,
            "positive_class": 1,
            "family_id": config.family.family_id,
            "seed": 42,
            "git_commit": "test",
            "git_dirty": False,
            "dependency_lock_sha256": "a" * 64,
            "best_iteration": None,
            "thresholds": {"youden_j": 0.7, "target_sensitivity": 0.7},
            "threshold_contract": threshold_contract(sensitivity_target=0.9),
            "input_contract": metadata_input_contract(),
        },
    )


def _evaluation(
    tmp_path: Path,
    *,
    logits: tuple[float, ...],
    variant: int = 0,
):
    package = _package(tmp_path, variant=variant)
    package_manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
    sample_ids = tuple(f"rsna:sample-{index}" for index in range(len(logits)))
    targets = tuple(index % 2 for index in range(len(logits)))
    authority = ValidatedHeldOutPublicationAuthority(
        "rsna",
        package.model_package_id,
        "pneumonia",
        str(package_manifest["bundle_id"]),
        _FREEZE_ID,
        sample_ids,
        targets,
        {
            "authority_kind": "rsna-frozen-campaign-v1",
            "execution_id": "rsna-execution-" + "0" * 64,
        },
        object(),
        None,
        object(),
    )
    evidence = _publish_held_out_prediction_evidence(
        publication_authority=authority,
        private_root=tmp_path / "private",
        dataset_id="rsna",
        model_package_id=package.model_package_id,
        task_id="pneumonia",
        bundle_id=package_manifest["bundle_id"],
        scope="test",
        sample_ids=sample_ids,
        targets=targets,
        logits=logits,
        authorized_by_freeze_id=_FREEZE_ID,
    )
    return _publish_rsna_evaluation(
        report_root=tmp_path / "reports",
        private_root=tmp_path / "private",
        model_root=tmp_path / "models/rsna",
        evidence=evidence,
        freeze=_TEST_FREEZE,
        execution=_TEST_EXECUTION,
        evaluation_policy={
            "policy_version": "rsna-held-out-evaluation-v1",
            "calibration_bins": 15,
            "threshold_selection": threshold_contract(sensitivity_target=0.9),
        },
        forbidden_source_values=sample_ids,
    )


def test_comparison_consumes_explicit_evaluation_identities(tmp_path: Path) -> None:
    first = _evaluation(
        tmp_path,
        logits=(-2.0, 1.0, -1.0, 2.0),
    )
    second = _evaluation(
        tmp_path,
        logits=(-1.5, 0.5, -0.5, 1.5),
        variant=1,
    )

    result = regenerate_comparison(
        [second.manifest["evaluation_id"], first.manifest["evaluation_id"]],
        output_directory=tmp_path / "reports",
        private_directory=tmp_path / "private",
        model_directory=tmp_path / "models/rsna",
    )
    frame = pd.read_csv(result.csv_path)

    assert result.row_count == 2
    assert tuple(frame.columns) == COMPARISON_COLUMNS
    assert frame["evaluation_id"].tolist() == sorted(
        [first.manifest["evaluation_id"], second.manifest["evaluation_id"]]
    )
    assert set(frame["model_package_id"]) == {
        first.manifest["model_package_id"],
        second.manifest["model_package_id"],
    }
    assert result.markdown_path.read_text(encoding="utf-8").startswith(
        "# Evaluation comparison\n\n| dataset_id |"
    )


def test_comparison_regeneration_is_deterministic(tmp_path: Path) -> None:
    evaluation = _evaluation(
        tmp_path,
        logits=(-2.0, 1.0, -1.0, 2.0),
    )
    arguments = {
        "output_directory": tmp_path / "reports",
        "private_directory": tmp_path / "private",
        "model_directory": tmp_path / "models/rsna",
    }
    result = regenerate_comparison([evaluation.manifest["evaluation_id"]], **arguments)
    first_csv = result.csv_path.read_bytes()
    first_markdown = result.markdown_path.read_bytes()
    regenerate_comparison([evaluation.manifest["evaluation_id"]], **arguments)
    assert result.csv_path.read_bytes() == first_csv
    assert result.markdown_path.read_bytes() == first_markdown


def test_comparison_cli_consumes_evaluation_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    evaluation = _evaluation(tmp_path, logits=(-2.0, 1.0, -1.0, 2.0))
    execution_directory = tmp_path / "execution"
    freeze_directory = tmp_path / "freeze"
    bundle_directory = tmp_path / "bundle"
    model_directory = tmp_path / "models/rsna"
    validated = []

    def execution(path: Path):
        assert path == execution_directory
        validated.append("execution")
        return _TEST_EXECUTION

    def freeze(path: Path, **kwargs):
        assert path == freeze_directory
        assert kwargs == {"model_root": model_directory, "bundle_directory": bundle_directory}
        validated.append("freeze")
        return _TEST_FREEZE

    monkeypatch.setattr(rsna_compare, "validate_execution", execution)
    monkeypatch.setattr(rsna_compare, "validate_freeze", freeze)
    assert (
        rsna_compare.main(
            [
                "--evaluation-ids",
                evaluation.manifest["evaluation_id"],
                "--output-directory",
                str(tmp_path / "reports"),
                "--private-directory",
                str(tmp_path / "private"),
                "--model-directory",
                str(model_directory),
                "--execution-directory",
                str(execution_directory),
                "--freeze-directory",
                str(freeze_directory),
                "--bundle-directory",
                str(bundle_directory),
            ]
        )
        == 0
    )
    assert validated == ["execution", "freeze"]
    assert "Wrote 1 rows" in capsys.readouterr().out
    rendered = pd.read_csv(tmp_path / "reports/rsna/comparisons/comparison/table.csv")
    assert rendered["evaluation_id"].tolist() == [evaluation.manifest["evaluation_id"]]


@pytest.mark.parametrize(
    "tamper",
    (
        "schema-version",
        "policy-version",
        "row-count",
        "evaluation-order",
        "csv-rendering",
        "markdown-rendering",
    ),
)
def test_comparison_validation_rejects_contract_tampering(tmp_path: Path, tamper: str) -> None:
    first = _evaluation(tmp_path, logits=(-2.0, 1.0, -1.0, 2.0))
    second = _evaluation(tmp_path, logits=(-1.5, 0.5, -0.5, 1.5), variant=1)
    result = regenerate_comparison(
        [first.manifest["evaluation_id"], second.manifest["evaluation_id"]],
        output_directory=tmp_path / "reports",
        private_directory=tmp_path / "private",
        model_directory=tmp_path / "models/rsna",
    )
    manifest_path = result.directory / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tamper == "schema-version":
        document["comparison_schema_version"] = True
    elif tamper == "policy-version":
        document["comparison_policy_version"] = 2
    elif tamper == "row-count":
        document["row_count"] = True
    elif tamper == "evaluation-order":
        document["evaluation_ids"].reverse()
    elif tamper == "csv-rendering":
        result.csv_path.write_bytes(result.csv_path.read_bytes() + b"\n")
    else:
        result.markdown_path.write_bytes(result.markdown_path.read_bytes() + b"\n")
    if tamper not in {"csv-rendering", "markdown-rendering"}:
        manifest_path.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError):
        validate_comparison(
            result.directory,
            report_root=tmp_path / "reports",
            private_root=tmp_path / "private",
            model_root=tmp_path / "models/rsna",
            enforce_directory_name=False,
        )


@pytest.mark.parametrize("membership", ("empty", "duplicate"))
def test_comparison_validation_rejects_invalid_producer_membership(
    tmp_path: Path, membership: str
) -> None:
    first = _evaluation(tmp_path, logits=(-2.0, 1.0, -1.0, 2.0))
    second = _evaluation(tmp_path, logits=(-1.5, 0.5, -0.5, 1.5), variant=1)
    result = regenerate_comparison(
        [first.manifest["evaluation_id"], second.manifest["evaluation_id"]],
        output_directory=tmp_path / "reports",
        private_directory=tmp_path / "private",
        model_directory=tmp_path / "models/rsna",
    )
    identities = (
        ()
        if membership == "empty"
        else (
            first.manifest["evaluation_id"],
            second.manifest["evaluation_id"],
            second.manifest["evaluation_id"],
        )
    )
    records = _comparison_records(
        identities,
        output_directory=tmp_path / "reports",
        private_directory=tmp_path / "private",
        model_directory=tmp_path / "models/rsna",
    )
    ordered_ids = [str(record["evaluation_id"]) for record in records]
    document = {
        "comparison_schema_version": 1,
        "comparison_policy_version": COMPARISON_POLICY_VERSION,
        "evaluation_ids": ordered_ids,
        "columns": list(COMPARISON_COLUMNS),
        "row_count": len(records),
    }
    _write_comparison(result.directory, document, records)

    with pytest.raises(ValueError, match="manifest contract"):
        validate_comparison(
            result.directory,
            report_root=tmp_path / "reports",
            private_root=tmp_path / "private",
            model_root=tmp_path / "models/rsna",
            enforce_directory_name=False,
        )


def test_comparison_regenerates_corrupt_cached_view(tmp_path: Path) -> None:
    evaluation = _evaluation(tmp_path, logits=(-2.0, 1.0, -1.0, 2.0))
    arguments = {
        "output_directory": tmp_path / "reports",
        "private_directory": tmp_path / "private",
        "model_directory": tmp_path / "models/rsna",
    }
    result = regenerate_comparison([evaluation.manifest["evaluation_id"]], **arguments)
    expected = result.csv_path.read_bytes()
    result.csv_path.write_bytes(b"corrupt")

    recovered = regenerate_comparison([evaluation.manifest["evaluation_id"]], **arguments)

    assert recovered.directory == result.directory
    assert recovered.csv_path.read_bytes() == expected
    assert set(recovered.manifest) == {
        "comparison_schema_version",
        "comparison_policy_version",
        "evaluation_ids",
        "columns",
        "row_count",
    }


@pytest.mark.parametrize(
    "evaluation_ids",
    [[], ["evaluation-a", "evaluation-a"]],
)
def test_comparison_rejects_missing_or_duplicate_identities(
    tmp_path: Path, evaluation_ids: list[str]
) -> None:
    with pytest.raises(ValueError):
        regenerate_comparison(
            evaluation_ids,
            output_directory=tmp_path / "reports",
            private_directory=tmp_path / "private",
        )


def test_comparison_rejects_tampered_evaluation_claims(tmp_path: Path) -> None:
    result = _evaluation(
        tmp_path,
        logits=(-2.0, 1.0, -1.0, 2.0),
    )
    manifest_path = result.directory / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["claims"]["probability_metrics"]["average_precision"] = 0.0
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        regenerate_comparison(
            [result.manifest["evaluation_id"]],
            output_directory=tmp_path / "reports",
            private_directory=tmp_path / "private",
            model_directory=tmp_path / "models/rsna",
        )


def test_evaluation_validation_rejects_tampered_derivative(tmp_path: Path) -> None:
    result = _evaluation(
        tmp_path,
        logits=(-2.0, 1.0, -1.0, 2.0),
    )
    derivative = result.directory / "derivatives" / "metrics.json"
    derivative.write_bytes(derivative.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="derivative"):
        validate_rsna_evaluation(
            result.directory,
            private_root=tmp_path / "private",
            model_root=tmp_path / "models/rsna",
        )


@pytest.mark.parametrize("child", ["manifest.json", "derivatives"])
def test_evaluation_validation_rejects_child_symlinks(tmp_path: Path, child: str) -> None:
    result = _evaluation(tmp_path, logits=(-2.0, 1.0, -1.0, 2.0))
    path = result.directory / child
    target = tmp_path / f"physical-{child.replace('.', '-')}"
    path.rename(target)
    path.symlink_to(target, target_is_directory=target.is_dir())

    with pytest.raises(ValueError, match="artifact set"):
        validate_rsna_evaluation(
            result.directory,
            private_root=tmp_path / "private",
            model_root=tmp_path / "models/rsna",
        )


_MISSING_SCHEMA_VERSION = object()


@pytest.mark.parametrize("value", [True, 1.0, 2, _MISSING_SCHEMA_VERSION])
def test_evaluation_requires_integer_schema_version_one(tmp_path: Path, value: object) -> None:
    result = _evaluation(tmp_path, logits=(-2.0, 1.0, -1.0, 2.0))
    manifest_path = result.directory / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value is _MISSING_SCHEMA_VERSION:
        document.pop("evaluation_schema_version")
    else:
        document["evaluation_schema_version"] = value
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_rsna_evaluation(
            result.directory,
            private_root=tmp_path / "private",
            model_root=tmp_path / "models/rsna",
        )


def test_evaluation_validation_renders_only_outside_immutable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _evaluation(tmp_path, logits=(-2.0, 1.0, -1.0, 2.0))
    original_write = rsna_evaluation_result.write_run_reports
    workspaces: list[Path] = []

    def tracked_write(directory: Path, **kwargs: object) -> None:
        assert directory != result.directory
        assert result.directory not in directory.parents
        workspaces.append(directory)
        original_write(directory, **kwargs)

    monkeypatch.setattr(rsna_evaluation_result, "write_run_reports", tracked_write)
    validate_rsna_evaluation(
        result.directory,
        private_root=tmp_path / "private",
        model_root=tmp_path / "models/rsna",
    )
    assert len(workspaces) == 1
    assert not workspaces[0].exists()


def test_evaluation_validation_requires_the_actual_model_package(tmp_path: Path) -> None:
    result = _evaluation(tmp_path, logits=(-2.0, 1.0, -1.0, 2.0))
    package_id = result.manifest["model_package_id"]
    package = tmp_path / "models/rsna/packages" / package_id
    package.rename(tmp_path / "unavailable-package")

    with pytest.raises(ValueError):
        validate_rsna_evaluation(
            result.directory,
            private_root=tmp_path / "private",
            model_root=tmp_path / "models/rsna",
        )


def test_evaluation_rederives_claims_from_identified_evidence(tmp_path: Path) -> None:
    original = _evaluation(
        tmp_path,
        logits=(-2.0, 1.0, -1.0, 2.0),
    )
    repeated = _evaluation(
        tmp_path,
        logits=(-2.0, 1.0, -1.0, 2.0),
    )
    assert original.created is True
    assert repeated.created is False
    assert repeated.manifest["evaluation_id"] == original.manifest["evaluation_id"]

    manifest_path = original.directory / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = document["evaluation_id"]
    document["claims"]["probability_metrics"]["roc_auc"] = 0.25
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_rsna_evaluation(
            original.directory,
            private_root=tmp_path / "private",
            model_root=tmp_path / "models/rsna",
            expected_evaluation_id=identity,
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("evaluation_policy", "calibration_bins", True),
        ("evaluation_policy", "calibration_bins", "15"),
        ("evaluation_policy", "calibration_bins", 1),
        ("threshold_selection", "sensitivity_target", 1),
        ("threshold_selection", "sensitivity_target", float("nan")),
        ("threshold_selection", "sensitivity_target", float("inf")),
        ("threshold_selection", "youden_j_policy_version", "changed-policy"),
        ("threshold_selection", "positive_class", 0),
        ("thresholds", "youden_j", True),
        ("thresholds", "youden_j", -0.1),
        ("thresholds", "target_sensitivity", float("inf")),
    ],
)
def test_evaluation_validation_rejects_malformed_scientific_policy(
    tmp_path: Path, section: str, field: str, value: object
) -> None:
    result = _evaluation(tmp_path, logits=(-2.0, 1.0, -1.0, 2.0))
    manifest_path = result.directory / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if section == "threshold_selection":
        document["evaluation_policy"][section][field] = value
    else:
        document[section][field] = value
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_rsna_evaluation(
            result.directory,
            private_root=tmp_path / "private",
            model_root=tmp_path / "models/rsna",
        )
