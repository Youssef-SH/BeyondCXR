# Architecture

## Terminology

- **Dataset:** an external source collection with a stable logical identity.
- **Bundle:** an immutable, validated set of typed artifacts stored under `bundle-<sha256>`.
- **Manifest:** an artifact's `manifest.json`, declaring its identity, contents, integrity,
  and applicable policies or provenance. Bundle, package, prediction, and result manifests
  have distinct contracts.

The lifecycle is: authenticated source → immutable bundle publication → validated scientific
consumer. `CURRENT` supports optional interactive discovery; scientific consumers pin explicit
immutable bundle IDs.

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
| Model mapping | Resolve a built-in metadata, CXR, or fusion model adapter |
| RSNA tabular runner | Fit a metadata model and select operating thresholds on validation |
| RSNA neural runner | Consume a validated CXR cache, train one seed, and select one validation state |
| RSNA test evaluator | Verify a completed package and explicit compatible evaluation config before applying them to the test partition |
| RSNA seed summarizer | Validate three package-compatible CXR or fusion evaluations and report complete aggregate claims |
| RSNA localization evaluator | Generate three-seed CXR Grad-CAM and aggregate box-localization reports |
| RSNA campaign | Execute the ordered CUDA-required campaign and pass exact package and evaluation identities in process |
| Symile development data | Resolve the pinned bundle and CV reference, then expose only official train and validation rows and authenticated CXR tensors |
| Symile fold runner | Execute one frozen outer fit, use inner selection where applicable, and publish private OOF evidence |
| Symile development aggregator | Validate all 15 fold packages and publish repeat metrics and the median final-training budget |
| Symile analysis | Align six explicit family authorities and publish paired and mean-logit ensemble development evidence |
| Symile ECG extension result | Combine core analysis and internal ECG development; derive primary thresholds and reference family-owned final budgets |
| Symile final fitting | Fit terminal full-development state and publish independently reconstructable final packages |
| Symile campaign control | Validate the pre-test freeze and same-freeze opening; derive the global result and private error review |
| Symile held-out inference | Reuse the frozen numerical runtime and publish only missing package-bound predictions |
| Symile preservation | Certify a restored archive before publishing its local ZIP, checksum, and external backup |
| Private analysis store | Retain aligned private prediction evidence and real-image localization overlays outside public outputs |
| Evaluation utilities | Compute probabilities, metrics, thresholds, latency, and plots |

Dataset adapters isolate source-specific behavior. Training reads validated bundle records through
the dataset mapping. Cache construction authenticates and decodes raw DICOM bytes; CXR and fusion
consumers verify that cache's exact identity and authorized sample coverage. Model adapters own
estimator or neural architecture construction.

The bundle manifest owns dataset, task, split, source, and artifact lineage. Parquet tables contain
row-level facts, while audits contain derived descriptions. `CURRENT` selects a bundle for
interactive commands; experiment configs pin an exact bundle ID.

Image and fusion experiment configs pin the semantic bundle ID. Validation computes the observed
bundle-manifest SHA-256 and verifies its physical, logical, semantic, split, and source contracts
before partition reads. Training freezes that exact identity in the package; linked evaluation
requires the same bundle-manifest SHA-256 before test access.

Model packages under `models/` and complete reports under `reports/` are the authoritative
physical outputs. MLflow stores the run ledger and references to those outputs; see
[`training.md`](training.md) for the experiment artifact contract.

Sample-level private prediction evidence and real-image localization examples are written under the
ignored `private/` workspace. Public reports contain aggregate results only; see
[`privacy.md`](privacy.md).

Selected execution facts such as loader policy are persisted as runtime provenance; transient
operational measurements remain outside scientific identity.

The canonical `make rsna-campaign` workflow writes one durable execution log, validates required output
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
    → metadata, CXR, or fusion training
    → explicit test evaluation and post-test analysis
    → bundle-qualified audits or operational experiment outputs
    → validated portable campaign archive
```

The common bundle envelope and RSNA artifact schemas are defined in
[`data_contract.md`](data_contract.md). Dataset-specific source contracts are described in
[`datasets/rsna.md`](datasets/rsna.md) and [`datasets/symile.md`](datasets/symile.md). Experiment
composition is defined in [`training.md`](training.md). Reconstruction and evaluation protocols
are defined in [`reproducibility.md`](reproducibility.md).

Symile follows a dataset-specific development and terminal campaign path. Development accessors
expose no official-test data:

```text
pinned Symile bundle + pinned 3 x 5 CV assignment
    → official train + validation strict-pneumonia rows
    → deterministic inner split per repeat and outer fold for selection families
    → immutable fold package + separate private OOF prediction evidence
    → complete family development authority
    → explicit six-family aggregate analysis
    → ECG extension result, also consuming internal three-modality development
    → full-development fitting of fourteen immutable final packages
    → validated pre-test freeze → atomic same-freeze test-open
    → fourteen package-bound raw prediction evidences → six predictor views
    → global result and separate private error-review derivative
    → recursively certified export and external backup
```

Fold packages contain only fitted model, preprocessing, configuration, and selection state. The
minimal patient-level OOF fields (`sample_id`, target, logit, and probability) live in separate
ignored prediction objects under `private/`. Family and cross-family reports contain aggregate
values only. Fold validation reconstructs the configured fitted estimator or neural architecture,
strict-loads safe state, and checks preprocessing and selection witnesses. Family validation
records the deterministic inner split for every fold. Because labs Logistic Regression fits the
complete outer-training fold without inner selection, that record is audit-only for its package
identity. Selection-family package identities bind `inner_split_id` because their selected fitted
state depends on that partition. Family validation re-derives repeat metrics and final-budget
witnesses from all 15 folds; cross-family validation
re-derives paired and ensemble claims from the six family authorities. Each Markdown summary is a
deterministic rendering of its manifest. Development performs no threshold selection, calibration,
final full-development fitting, or official-test evaluation.

Neural checkpoints are independently reconstructable persisted documents inside their containing
packages. Each checkpoint therefore owns schema version `1` and is validated independently from
the outer package manifest.

The terminal campaign consumes these development authorities without changing their exact-six
public surface or the core two-modality gate. Its separate ECG family uses an exact-three gate.
Official-test materialization requires both `ValidatedPretestFreeze` and its atomic same-freeze
test-open record. The freeze binds all final packages, development-derived primary thresholds,
evaluation policy, numerical inference runtime, and execution provenance. An opened resume cannot
retrain or change that authority; it completes missing post-open work only.

The global result owns aggregate held-out claims. Private error review is a regenerable derivative,
not a prerequisite for validating those claims. Preservation includes both, along with their
recursive authority closure. See [`reproducibility.md`](reproducibility.md) for runtime matching,
execution, and backup requirements.
