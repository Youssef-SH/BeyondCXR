"""Generate the two aggregate privacy-safe Symile audit outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_artifacts import (
    read_symile_labs,
    read_symile_samples,
    resolve_symile_bundle,
    strict_pneumonia_rows,
)
from beyondcxr.data.symile_schemas import DEVELOPMENT_SPLITS, LAB_ITEM_IDS, LAB_NAMES
from beyondcxr.utils.operational_logging import (
    add_logging_argument,
    configure_logging,
    get_operational_logger,
    timed_phase,
)
from beyondcxr.utils.privacy import validate_public_reports
from beyondcxr.utils.publication import publish_directory, staging_directory

AUDIT_FILENAME = "symile_audit.md"
MISSINGNESS_FILENAME = "laboratory_missingness.csv"
REPORT_FILENAMES = (AUDIT_FILENAME, MISSINGNESS_FILENAME)
_LOGGER = get_operational_logger(__name__)


def generate_symile_audit(
    manifest_directory: str | Path = "data/manifests",
    output_directory: str | Path = "reports/symile/audit",
    *,
    bundle_id: str | None = None,
) -> dict[str, object]:
    """Resolve one immutable bundle once and publish its aggregate audit."""
    with timed_phase(_LOGGER, "audit_bundle_loading"):
        bundle = resolve_symile_bundle(
            manifest_directory,
            bundle_id=bundle_id,
            full_validation=True,
        )
        samples = read_symile_samples(bundle, official_splits=DEVELOPMENT_SPLITS)
        labs = read_symile_labs(bundle, sample_ids=samples["sample_id"].astype(str).tolist())
        metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
    destination = Path(output_directory) / bundle.bundle_id
    stage = staging_directory(destination)
    try:
        missingness = _laboratory_missingness(samples, labs)
        missingness.to_csv(
            stage / MISSINGNESS_FILENAME,
            index=False,
            float_format="%.6f",
            lineterminator="\n",
        )
        (stage / AUDIT_FILENAME).write_text(
            _audit_markdown(bundle.bundle_id, metadata, samples),
            encoding="utf-8",
        )
        if {path.name for path in stage.iterdir()} != set(REPORT_FILENAMES):
            raise ManifestBuildError("Symile audit output set is incomplete or unexpected")
        validate_public_reports(
            [stage / name for name in REPORT_FILENAMES],
            forbidden_source_values={
                str(value)
                for column in ("sample_id", "subject_id", "hadm_id")
                for value in samples[column]
            },
        )
        publish_directory(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {
        "bundle_id": bundle.bundle_id,
        "report_directory": destination.as_posix(),
        "reports": list(REPORT_FILENAMES),
    }


def _laboratory_missingness(samples: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    frame = samples[["sample_id", "official_split"]].merge(
        labs, on="sample_id", validate="one_to_one"
    )
    scopes = [(split, frame.loc[frame["official_split"] == split]) for split in DEVELOPMENT_SPLITS]
    scopes.append(("development", frame))
    records: list[dict[str, object]] = []
    for scope, scoped in scopes:
        for item_id in LAB_ITEM_IDS:
            observed = int(scoped[f"lab_{item_id}_observed"].sum())
            total = len(scoped)
            records.append(
                {
                    "scope": scope,
                    "item_id": item_id,
                    "lab_name": LAB_NAMES[item_id],
                    "observed_count": observed,
                    "missing_count": total - observed,
                    "total_count": total,
                    "missing_rate": (total - observed) / total,
                }
            )
    return pd.DataFrame.from_records(records)


def _audit_markdown(
    bundle_id: str,
    metadata: dict[str, object],
    samples: pd.DataFrame,
) -> str:
    counts = metadata["qualification"]["membership_counts"]
    strict = _development_strict_counts(samples)
    asset_count = len(metadata["source"]["source_assets"])
    lines = [
        "# Symile-MIMIC data audit",
        "",
        f"- Dataset release: Symile-MIMIC {metadata['dataset']['release']}",
        f"- Bundle ID: `{bundle_id}`",
        "- Development membership: intrinsic to the bundle samples table",
        f"- Authenticated official source assets: {asset_count:,}",
        "- Source authentication: passed",
        "",
        "## Development membership",
        "",
        f"- Train admissions: {counts['train_admissions']:,}",
        f"- Validation admissions: {counts['validation_admissions']:,}",
        "- Development admission and patient overlap: 0",
        "- Complete source reconciliation and test qualification require authorized opening.",
        "",
        "## Strict pneumonia",
        "",
        "| Scope | Eligible | Positive | Negative |",
        "| --- | ---: | ---: | ---: |",
    ]
    for scope in ("train", "validation", "development"):
        entry = strict[scope]
        lines.append(
            f"| {scope} | {entry['eligible']:,} | {entry['positive']:,} | {entry['negative']:,} |"
        )
    lines.extend(
        [
            "",
            (
                "The task represents a report-derived Pneumonia finding. A source state of 1 is "
                "positive, 0 is negative, and -1 or missing is excluded. Confirmed "
                "infectious-pneumonia diagnosis lies outside the endpoint definition."
            ),
            "",
            "## Development modality qualification",
            "",
            "- Common split/source-row alignment: passed",
            (
                "- CXR shape, float32 dtype, finite values, inverse ImageNet normalization, "
                "and repeated-grayscale reconstruction: passed"
            ),
            "- ECG shape, float32 dtype, finite values, and [-1, 1] range: passed",
            "- Raw laboratory values and explicit observedness: passed",
            "- Official percentile and missing-imputation conformance: passed",
            "- Laboratory item ordering and names: passed",
            "- AP/PA eligibility: passed",
            "- CXR >24 to <=72 hour timing represented in the release: directly verified",
            "- ECG +/-24 hour timing represented in the release: directly verified",
            (
                "- Laboratory timing is evidenced by the authenticated official reference "
                "implementation. The downloaded release contains selected laboratory values "
                "without raw laboratory-event timestamps."
            ),
            "",
            "## Aggregate demographics",
            "",
            "| Split | Admissions | Patients | Median age | Female | Male | AP | PA |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in DEVELOPMENT_SPLITS:
        scoped = samples.loc[samples["official_split"] == split]
        lines.append(
            f"| {split} | {len(scoped):,} | {scoped['subject_id'].nunique():,} | "
            f"{scoped['age_years'].median():.1f} | {(scoped['sex'] == 'F').sum():,} | "
            f"{(scoped['sex'] == 'M').sum():,} | {(scoped['view_position'] == 'AP').sum():,} | "
            f"{(scoped['view_position'] == 'PA').sum():,} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            (
                "- Official CXR, ECG, percentile, and missingness tensors serve as authenticated "
                "source/conformance artifacts."
            ),
            (
                "- Laboratory percentile tensors and `labs_means.json` serve as conformance "
                "references. Supervised imputation is fitted later under the leakage-controlled "
                "modeling protocol."
            ),
            (
                "- The audit is aggregate. Patient-level identities, paths, modality values, "
                "and laboratory rows are excluded."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _development_strict_counts(samples: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in DEVELOPMENT_SPLITS:
        eligible = strict_pneumonia_rows(samples.loc[samples["official_split"] == split].copy())
        positive = int((eligible["target"] == 1).sum())
        result[split] = {
            "eligible": len(eligible),
            "positive": positive,
            "negative": len(eligible) - positive,
        }
    result["development"] = {
        key: sum(result[split][key] for split in DEVELOPMENT_SPLITS)
        for key in ("eligible", "positive", "negative")
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-directory", type=Path, default=Path("data/manifests"))
    parser.add_argument("--output-directory", type=Path, default=Path("reports/symile/audit"))
    parser.add_argument("--bundle-id", default=None)
    add_logging_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        summary = generate_symile_audit(
            args.manifest_directory,
            args.output_directory,
            bundle_id=args.bundle_id,
        )
    except (ManifestBuildError, OSError, ValueError, KeyError) as exc:
        print(f"Symile audit failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
