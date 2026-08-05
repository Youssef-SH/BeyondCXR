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
| CXR cache | Publish validated deterministic images before live stochastic augmentation |
| Dataset mapping | Resolve the built-in adapter for a pinned bundle |
| Model mapping | Resolve a built-in metadata, image, or fusion model adapter |
| Tabular runner | Fit a metadata model and select operating thresholds on validation |
| Neural runner | Consume a validated CXR cache, train one seed, and select one validation state |
| Test evaluator | Verify a completed package before applying it to the test partition |
| Seed summarizer | Validate three explicit compatible image or fusion test runs and report aggregate statistics |
| Localization evaluator | Generate three-seed CXR Grad-CAM and aggregate box-localization reports |
| RSNA campaign | Execute the complete ordered GPU workflow and pass exact run identities in process |
| Private analysis store | Retain aligned neural predictions and real-image localization overlays outside public outputs |
| Evaluation utilities | Compute probabilities, metrics, thresholds, latency, and plots |

Dataset adapters isolate source-specific behavior. Training reads validated bundle records through
the dataset mapping. Cache construction authenticates and decodes raw DICOM bytes; image and fusion
consumers verify that cache's exact identity and authorized sample coverage. Model adapters own
estimator or neural architecture construction.

The manifest owns dataset, task, split, source, and artifact lineage. Parquet tables contain
row-level facts, while audits contain derived descriptions. `CURRENT` selects a bundle for
interactive commands; experiment configs pin an exact bundle ID.

Image and fusion experiment configs pin the semantic bundle ID. Validation computes the observed
bundle-manifest SHA-256 and verifies its physical, logical, semantic, split, and source contracts
before partition reads. Training freezes that exact identity in the package; linked evaluation
requires the same bundle-manifest SHA-256 before test access.

Model packages under `models/` and complete reports under `reports/` are the authoritative
physical outputs. MLflow stores the run ledger and references to those outputs; see
[`training.md`](training.md) for the experiment artifact contract.

Patient-level neural predictions and real-image localization examples are written under the ignored
`private/` workspace. Public reports contain aggregate results only; see [`privacy.md`](privacy.md).

Selected execution facts such as loader policy are persisted as runtime provenance; transient
operational measurements remain outside scientific identity.

The canonical `make rsna-gpu` campaign writes one durable execution log, validates required output
completeness and lineage, and exports the portable scientific/provenance surface with a checksum.
The raw dataset and disposable deterministic image cache remain outside that archive. Detailed
execution and ownership contracts are in [`training.md`](training.md).

Campaign preparation authenticates and deterministically preprocesses image bytes from every bundle
partition into an immutable cache. Its identity binds bundle, source-inventory, and preprocessing
identity; validation separately proves the exact sample-to-partition index and cached image-content
digest. Cache derivation identity and source authentication are package-bound; the mapping and
content hashes validate the disposable cache locally. Cache-backed consumers reopen no raw DICOMs.
This label-free step fits no statistics or models; within the canonical campaign, task-bearing
training completes before held-out evaluation.

## Data flow

```text
RSNA source files
    → dataset adapter
    → validated tables and manifest
    → immutable bundle
    → audit and deterministic CXR cache
    → metadata, image, or fusion training
    → explicit test evaluation and post-test analysis
    → bundle-qualified audits or run-qualified experiment outputs
    → validated portable campaign archive
```

Artifact schemas are defined in [`data_contract.md`](data_contract.md). Experiment composition is
defined in [`training.md`](training.md). Reconstruction and evaluation protocols are defined in
[`reproducibility.md`](reproducibility.md).
