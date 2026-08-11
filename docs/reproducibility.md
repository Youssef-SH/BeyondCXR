# Reproducibility

## Environment

The project targets Python 3.13. `uv.lock` defines the dependency environment. A paid GPU campaign
host installs the locked runtime environment:

```bash
uv sync --locked --no-dev
```

Contributors install the development group before running quality gates:

```bash
uv sync --locked --group dev
uv run pre-commit install
```

Each training run records the lock-file SHA-256, Python version, operating system, CPU architecture
and model, and relevant library versions. Neural runs also record PyTorch, TorchVision, and
TorchXRayVision versions and requested and effective device policies.

## Rebuild the RSNA bundle

Place Stage 2 source data under `data/raw/rsna/extracted/`, then run:

```bash
make rsna-manifest
```

Custom paths and split recipes are available through the CLI:

```bash
uv run python -m radfusion.data.rsna_manifest \
  --dataset-root /approved/local/rsna/extracted \
  --output-directory /approved/local/manifests \
  --split-seed 42 \
  --train-ratio 0.70 \
  --validation-ratio 0.15 \
  --test-ratio 0.15
```

The build records both source CSV SHA-256 values and authenticates every labeled DICOM against the
source inventory. It validates the complete staged bundle before publishing it and updating
`CURRENT`.

The default split recipe groups samples by patient and stratifies on the binary challenge target.
Within each target stratum, patients are ordered by SHA-256 of the UTF-8 bytes
`<seed>\0<patient_id>`, with patient ID as the collision tie-break. Allocation guarantees one
patient per positive-ratio destination when feasible, then applies largest remainder in canonical
train, validation, test order. For smaller strata, patients fill the highest-ratio destinations,
with canonical order breaking equal-ratio ties.

The split recipe ID hashes only the algorithm version, seed, stratification target, and ordered
ratios. The algorithm version binds patient grouping, SHA-256 ranking, UTF-8 input encoding,
patient-ID collision tie-breaking, allocation, and canonical split order.

The split assignment ID hashes the canonical sorted `(sample_id, split_name)` mapping. It is stable
for the same assignment and changes when any assignment changes.

Logical Arrow hashes cover each artifact's exact schema and canonical ordered values. Canonical
null handling keeps these hashes stable across valid Parquet round trips. Logical hashes
participate in the semantic bundle ID.

Physical Parquet hashes cover serialized file bytes and detect corruption. They can differ across
valid encodings of the same logical tables and do not participate in bundle identity. The exact
identity and acceptance rules are defined in [`data_contract.md`](data_contract.md).

## Generate audits and experiments

```bash
make rsna-gpu
```

The command runs the complete ordered RSNA campaign and transfers exact run identities directly
between training, evaluation, summary, and localization functions. It crosses the test boundary
only after both metadata packages and all six neural packages have frozen.

Audits are published under `reports/rsna/audit/<bundle-id>/`. Rebuilding one audit replaces only
that bundle-qualified audit directory.

Experiment configs pin the exact bundle ID and one training seed. Preprocessing is fitted on
training data. Validation selects the LightGBM stopping point and both operating thresholds.
`make evaluate` verifies those choices against the source training run before applying them to
test in a separate linked run.

Image experiment configurations pin the semantic bundle ID. Validation computes the observed
bundle-manifest SHA-256 and verifies the manifest's physical, logical, semantic, split, and source
contracts. Training records the hash in its package and an MLflow parameter; linked test evaluation
requires the same bundle-manifest SHA-256 and accesses only its authorized task-bearing partition.

Training runs record the Git commit and dirty status, exact configuration bytes and hash,
dependency-lock hash, environment, dataset identity, and model lineage. Formal test evaluation
requires a package produced from the evaluator's clean Git commit and matching dependency lock.
Test-evaluation runs record their own code and lock provenance and link the verified model package
to its source training run.

The campaign materializes each source DICOM's bytes once, authenticates them against the bundle
inventory, and decodes the same in-memory bytes while building an identity-addressed deterministic
CXR cache. Cache identity binds the bundle, manifest, source inventory, preprocessing, and
authentication policy. Validation separately proves the exact sample-to-partition mapping and
cached image-content digest. Cache-backed training, evaluation, fusion, and localization verify
those contracts without reopening raw DICOMs. The memory-mapped float32 cache is derived and
disposable.

Cache derivation identity and source-authentication provenance are package-bound. Mapping and
image-content hashes provide local cache-integrity checks and remain outside model semantic identity
and the model package ID.

Cache preparation may read authenticated image bytes from every partition, but reads no task
labels and fits no statistics, thresholds, or models. Official training reads only train and
validation task rows. Within the canonical campaign, test labels become available to experiment
consumers only after all eight training packages freeze.

Training permutations derive from SHA-256 of a fixed namespace, training seed, and epoch.
Augmentation seeds derive from a separate namespace, training seed, epoch, and sample ID. Epoch is
carried in sampler requests, so worker startup, persistence, assignment, and prefetch timing cannot
change ordering or augmentation. The cache stores the resized canonical image before stochastic
augmentation and XRV intensity normalization, preserving the established operation order.

Image training fingerprints the materialized pretrained weight file immediately before and after
model construction and requires exact equality. The selected CPU checkpoint may come from
head-only warm-up or full fine-tuning. Explicit test evaluation verifies the package, checkpoint,
code, lock, and model structure before loading test rows and applies the validation thresholds
unchanged. It reconstructs the architecture without reading or downloading the original
pretrained cache.

Train and validation loaders use the configured reused-loader worker count, which is 2 in the
canonical RSNA configurations. Positive-worker loaders use `spawn`, persistent workers, and
prefetch factor 2. One-shot test inference is synchronous and uses no worker processes,
persistence, prefetch, or multiprocessing context. The actual lifecycle-specific loader policy is
runtime provenance rather than a semantic compatibility input.

Epoch records contain the learning rates used for that epoch. CUDA runtime, cuDNN, GPU identity,
device index, and compute capability are recorded as runtime provenance and excluded from the model
package ID.

Neural package identity is an exact provenance identity over meaning-bearing image configuration
and package lineage, including the selected checkpoint and observed bundle-manifest SHA-256.
Runtime provenance, operational paths, and training-run ID remain outside this identity; the
archived YAML is validated separately by its exact byte hash.

Each training invocation runs one explicit seed. The RSNA image seed summarizer accepts only
explicit linked test runs for seeds 17, 42, and 2026, verifies their stored compatibility, and
reports individual metrics with arithmetic means and sample standard deviations. Thresholds remain
seed-specific, and no favorable seed is selected as canonical.

The same summary contract applies to fusion runs and additionally verifies the fixed fusion and
structured-preprocessing contracts and compatible same-seed source CXR families. Fusion source run
IDs are supplied at execution time; exact source package and checkpoint identities are package
lineage.

RSNA localization reconstructs each of the three verified image packages, targets the final
DenseNet spatial feature sequence, and maps bounding boxes through the evaluation center crop and
resize. Pointing-game ties use row-major order. Activation energy is the fraction of nonnegative
heatmap mass inside the union box mask; a zero heatmap receives zero for both metrics. Qualitative
cases are the first SHA-256-ranked member of each TP, FN, FP, and TN stratum under the seed-specific
validation Youden threshold. All positives contribute to quantitative localization; Grad-CAM is
computed for only the selected negative examples. Aggregate output is public-safe, while real-image
overlays remain in the ignored private workspace.

Each image and fusion test evaluation also publishes an aligned, validated private prediction table
under `private/predictions/`. The table supports later aligned analyses and is excluded from public
reports and MLflow artifacts.

## Reproduce Symile development evidence

The six Symile configs pin the frozen bundle and 3-by-5 patient-grouped CV assignment. Rebuilding
the M4 artifacts is not part of development execution. Run each family explicitly; fusion families
also receive the completed CXR development identity:

```bash
make symile-develop CONFIG=configs/symile_labs_logistic.yaml
make symile-develop CONFIG=configs/symile_labs_lightgbm.yaml
make symile-develop CONFIG=configs/symile_cxr.yaml
make symile-develop CONFIG=configs/symile_concat.yaml \
  SOURCE_CXR_DEVELOPMENT_ID=development-<sha256>
make symile-develop CONFIG=configs/symile_gated.yaml \
  SOURCE_CXR_DEVELOPMENT_ID=development-<sha256>
make symile-develop CONFIG=configs/symile_gated_no_observedness.yaml \
  SOURCE_CXR_DEVELOPMENT_ID=development-<sha256>
```

Each successful invocation produces 15 immutable fold packages and one complete family development
authority. Every repeat must contain one OOF prediction for each of the 2,368 strict-pneumonia
development admissions. Neural packages retain selected-epoch histories; LightGBM packages retain
`best_iteration`. The family manifest records the ordered 15 values and their eighth ordered value
as the frozen median final-training budget.

After all six family identities exist, publish the aggregate analysis:

```bash
make symile-analyze DEVELOPMENT_IDS="<six explicit development IDs>"
```

Identity validation covers exact config bytes, semantic configuration, bundle and CV lineage,
inner-split identity, reconstructable fitted state, fitted ECDF and feature contracts, validated
neural history, OOF logical and physical hashes, and exact fold coverage. Family authorities
recompute each repeat metric and median budget from their 15 folds. Cross-family authorities
recompute paired effects and mean-logit ensemble metrics from the six families, and summary text
must equal the deterministic manifest rendering.
MLflow records one `training`/`oof` run per fold and sets `run_complete=true` only after immutable
fold publication and ledger updates succeed. Paths, MLflow run IDs, timestamps, and runtime
hardware remain outside semantic artifact identity.

The development reader requests only official train and validation rows and authenticates only the
corresponding CXR tensors. It has no official-test accessor. Development produces no thresholds,
calibration, final full-development model, or held-out-test statistic.

## Probability and operating-point metrics

Models expose class labels and probabilities. Evaluation locates the column labeled `1` and
rejects invalid class or probability contracts.

Average precision is computed with `average_precision_score`. Probability metrics also include
ROC-AUC and Brier score. Threshold-dependent precision, recall, specificity, F1, and confusion
counts are grouped by operating point:

- the Youden-J threshold maximizes validation sensitivity minus false-positive rate;
- the target-sensitivity threshold is the highest validation threshold meeting the configured
  sensitivity, which is 0.90 in the experiment configs.

Both policies enumerate every finite ROC threshold and choose the highest threshold among ties or
qualifying candidates. The thresholds are applied unchanged to test probabilities. They are
benchmark operating points, not clinical optima.

## Calibration

Expected calibration error uses the configured number of equal-width bins over `[0, 1]`. Bins are
lower-inclusive; the final bin includes 1. Empty bins contribute zero. Each non-empty bin
contributes its sample fraction times the absolute difference between mean predicted probability
and observed positive fraction. Experiment configs use 15 bins, and the plot uses the same count
and strategy.

Calibration slope and intercept come from an L2 logistic regression of the target on predicted
log-odds. Probabilities are clipped to `[1e-6, 1 - 1e-6]` before the logit transform. The fit uses
`C=1e6`, the `lbfgs` solver, 2,000 maximum iterations, and an intercept.

Calibration statistics describe raw class-weighted model outputs. They are point estimates;
bootstrap resampling is not used.

## Latency and model size

Latency covers the complete preprocessing and probability pipeline on CPU. It is the median of
1,000 individually timed single-sample calls after 100 warm-up calls. The benchmark always uses
the first sample in deterministic order for the evaluated partition. Results are reported in
milliseconds and remain dependent on hardware and system load.

Model size is the exact serialized `model.skops` or `model.pt` byte count divided by 1,048,576 and
reported as MiB. The CPU latency protocol applies to metadata pipelines; neural comparison records
do not report that tabular latency measurement.

## Artifact lineage

Bundle identity and acceptance rules are defined in [`data_contract.md`](data_contract.md).
Model lineage, artifact ownership, and comparison-table roles are defined in
[`training.md`](training.md).

MLflow records exact loaded training configs, resolved parameters, scalar metrics, provenance,
lineage, completion state, and project-output references. Linked test runs add evaluator
provenance, test metrics, and their source package and training-run relationship.

Operational logs describe the current execution and may contain environment-dependent elapsed
times and heartbeat timing. They are not inputs to reproducibility, provenance, MLflow lineage,
bundle identity, or model package identity.

The full campaign log is stored under `reports/rsna/campaigns/<campaign-id>/execution.log`. After
final output completeness and lineage validation plus an MLflow database integrity check, the
campaign writes `outbox/rsna-results-<campaign-id>.tar.gz` and its `.sha256` checksum. The archive
contains the campaign's exact bundle, finished scientific/provenance outputs, and selected private
prediction and localization outputs. Raw data, unrelated bundles, the derived CXR cache,
environments, and temporary files are excluded.

The archived copy of the execution log ends at `campaign_ready_for_export`. The live project-owned
log then records archive creation and `campaign_succeeded`; the checksum verifies archive integrity
captured at the ready-for-export boundary.

## Quality gates

```bash
uv lock --check
uv sync --locked --group dev
make check
make pre-commit
git diff --check
```

The default suite uses synthetic data. Local RSNA integration tests run when the source dataset is
available:

```bash
uv run pytest -m integration
```

Generated bundles, reports, models, and MLflow state can be deliberately deleted with
`make purge-generated` and rebuilt. Raw datasets remain user-managed external inputs.
