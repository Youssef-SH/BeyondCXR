# Architecture

## Terminology

- **Dataset:** an external source collection with a stable logical identity.
- **Build:** one execution of bundle construction. A successful build publishes one bundle.
- **Bundle:** an immutable, validated set of typed artifacts stored under `build-<sha256>`.
- **Manifest:** `rsna_manifest_metadata.json`, the bundle document that declares identity,
  contents, hashes, policies, and provenance. The `rsna-manifest` command runs a build.

The lifecycle is: source dataset → build execution → immutable bundle → validated consumers.
`CURRENT` selects the active bundle; published bundles remain immutable.

## Component boundaries

| Component | Responsibility |
| --- | --- |
| Source adapter | Parse source tables, discover DICOM files, and extract selected headers |
| Artifact builder | Normalize typed records and construct patient-level assignments |
| Validator | Enforce schemas, relationships, paths, identities, and ordering |
| Bundle publisher | Publish immutable bundles and update `CURRENT` atomically |
| Audit generator | Produce aggregate dataset reports from a validated bundle |
| Dataset mapping | Resolve the built-in adapter for a pinned bundle |
| Model mapping | Resolve a built-in metadata or image model adapter |
| Tabular runner | Fit a metadata model and select operating thresholds on validation |
| Neural runner | Authenticate source images, train one seed, and select one validation state |
| Test evaluator | Verify a completed package before applying it to the test partition |
| Seed summarizer | Validate three explicit linked RSNA image test runs and report aggregate statistics |
| Evaluation utilities | Compute probabilities, metrics, thresholds, latency, and plots |

Dataset adapters isolate source-specific behavior. Training reads validated bundle records through
the dataset mapping. Image consumers also authenticate permitted external DICOM bytes against the
bundle inventory before decoding. Model adapters own estimator or neural architecture construction.

The manifest owns dataset, task, split, source, and artifact lineage. Parquet tables contain
row-level facts, while audits contain derived descriptions. `CURRENT` selects a bundle for
interactive commands; experiment configs pin an exact bundle ID.

Image configs pin the semantic bundle ID. Validation computes the observed bundle-manifest SHA-256
and verifies its physical, logical, semantic, split, and source contracts before partition reads.
Training freezes that exact identity in the package; linked evaluation requires the same
bundle-manifest SHA-256 before test access.

Model packages under `models/` and complete reports under `reports/` are the authoritative
physical outputs. MLflow stores the run ledger and references to those outputs; see
[`training.md`](training.md) for the experiment artifact contract.

Operational records are transient and do not participate in scientific state, MLflow records,
published artifacts, or semantic identities.

## Data flow

```text
RSNA source files
    → dataset adapter
    → validated tables and manifest
    → immutable bundle
    → audit, metadata or image training, or explicit test evaluation
    → bundle-qualified audits or run-qualified experiment outputs
```

Artifact schemas are defined in [`data_contract.md`](data_contract.md). Experiment composition is
defined in [`training.md`](training.md). Reconstruction and evaluation protocols are defined in
[`reproducibility.md`](reproducibility.md).
