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

Use a Git checkout on Linux or WSL2 for scientific execution. Make assumes the checkout root;
from another directory use `make -C /path/to/checkout <target>`. The wheel contains importable
Python modules; configs, the Git revision, lock, and private data authorities are checkout/operator
inputs. Installing the wheel or unpacking an sdist without Git provenance does not reproduce a study.

## Rebuild the RSNA bundle

Place Stage 2 source data under `data/raw/rsna/extracted/`, then run:

```bash
make rsna-manifest
```

Custom paths and split recipes are available through the CLI:

```bash
uv run python -m radfusion.data.rsna_manifest \
  --source-root /approved/local/rsna/extracted \
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

## Generate RSNA audits and experiments

```bash
make rsna-campaign
```

The command runs the complete ordered RSNA campaign and transfers exact model-package and
evaluation identities directly between training, evaluation, summary, and localization functions.
It crosses the test boundary only after both metadata packages and all six neural packages have
frozen.

Audits are published under `reports/rsna/audit/<bundle-id>/`. Rebuilding one audit replaces only
that bundle-qualified audit directory.

Experiment configs pin the exact bundle ID; the execution seed is a separate runtime coordinate.
Preprocessing is fitted on
training data. Validation selects the LightGBM stopping point and both operating thresholds.
`make rsna-evaluate PACKAGE_ID=... CONFIG=...` verifies fitted choices and frozen thresholds from
the explicit model package, verifies that the explicit config has identical package-scoped fitting
semantics, and takes downstream calibration policy from that config before accessing test.

CXR experiment configurations pin the semantic bundle ID. Validation computes the observed
bundle-manifest SHA-256 and verifies the manifest's physical, logical, semantic, split, and source
contracts. Training records the hash in its package and an MLflow parameter; linked test evaluation
requires the same bundle-manifest SHA-256 and accesses only its authorized task-bearing partition.

Training runs record the Git commit and dirty status, exact configuration bytes and hash,
dependency-lock hash, environment, dataset identity, and model lineage. Git and lock identities are
reproducibility witnesses outside scientific package identity. Formal test evaluation
validates the package, prediction, and evaluation scientific contracts without requiring the
evaluator to share the producer's Git or lock witness. Test-evaluation runs record their own code
and lock provenance; MLflow run linkage remains operational provenance rather than scientific
authority.

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

CXR training fingerprints the materialized pretrained weight file immediately before and after
model construction and requires exact equality. The selected CPU checkpoint may come from
head-only warm-up or full fine-tuning. Explicit test evaluation verifies the package, checkpoint,
and model structure before loading test rows and applies the package-bound validation thresholds
unchanged. Producer and evaluator code/lock identities remain separately recorded reproducibility
witnesses. Evaluation reconstructs the architecture without reading or downloading the original
pretrained cache.

Train and validation loaders use the configured reused-loader worker count, which is 2 in the
canonical RSNA configurations. Positive-worker loaders use `spawn`, persistent workers, and
prefetch factor 2. One-shot test inference is synchronous and uses no worker processes,
persistence, prefetch, or multiprocessing context. The actual lifecycle-specific loader policy is
runtime provenance outside semantic identity.

Epoch records contain the learning rates used for that epoch. CUDA runtime, cuDNN, GPU identity,
device index, and compute capability are recorded as runtime provenance and excluded from the model
package ID.

Neural package identity binds the package-scoped configuration projection, semantic source-package
lineage, and the canonical selected tensor/preprocessor state. The complete
`config_semantic_sha256` remains a validated configuration witness but does not rename fitted state
when only downstream evaluation policy changes. Serialized checkpoint bytes, observed
bundle-manifest SHA-256, `config_source_sha256`, Git, lock, runtime, paths, and MLflow runs are
validated separately as integrity, reproducibility, or operational witnesses.
Frozen validation-derived thresholds and their selection contract are selected scientific package
state and therefore participate in the package identity. Held-out evaluation derives its nested
threshold-selection policy from that validated package contract and receives calibration policy
from an explicit compatible evaluation config; equal numeric thresholds cannot mask a
sensitivity-target, policy-version, or positive-class mismatch.

Each training invocation runs one explicit seed. The RSNA CXR seed summarizer accepts only
explicit evaluation IDs for seeds 17, 42, and 2026, validates each package-to-prediction-to-
evaluation chain, proves that the packages share one seed-neutral scientific family definition,
and publishes a reusable `seed-summary-<sha256>` result. Its validator re-derives all six
probability/calibration metrics and both operating-point metric/count surfaces, plus arithmetic
means and sample standard deviations for non-threshold values. Thresholds remain seed-specific,
and no favorable seed is selected as canonical.

The same summary contract applies to fusion runs and additionally verifies the fixed fusion and
structured-preprocessing contracts and compatible same-seed source CXR families. Fusion receives
the source package ID as its scientific lineage authority.

RSNA localization reconstructs each of the three verified CXR packages, targets the final
DenseNet spatial feature sequence, and maps bounding boxes through the evaluation center crop and
resize. Pointing-game ties use row-major order. Activation energy is the fraction of nonnegative
heatmap mass inside the union box mask; a zero heatmap receives zero for both metrics. Qualitative
cases are the first SHA-256-ranked member of each TP, FN, FP, and TN stratum under the seed-specific
validation Youden threshold. All positives contribute to quantitative localization; Grad-CAM is
computed for only the selected negative examples. Aggregate output is public-safe, while real-image
overlays remain in the ignored private workspace.

Each RSNA test evaluation also publishes an aligned, validated private prediction
object under `private/predictions/rsna/prediction-<sha256>/`. Its logical content and package
lineage define its identity; serialized Parquet bytes are an integrity witness.

## Reproduce Symile development evidence

The six Symile configs pin the frozen bundle and 3-by-5 patient-grouped CV assignment. Rebuilding
the data-foundation artifacts is not part of development execution. Run each family explicitly;
fusion families
also receive the completed CXR development identity:

```bash
make symile-develop CONFIG=configs/symile_labs_logistic.yaml
make symile-develop CONFIG=configs/symile_labs_lightgbm.yaml
make symile-develop CONFIG=configs/symile_cxr_densenet.yaml
make symile-develop CONFIG=configs/symile_cxr_labs_concat.yaml \
  SOURCE_CXR_DEVELOPMENT_ID=development-<sha256>
make symile-develop CONFIG=configs/symile_cxr_labs_gated.yaml \
  SOURCE_CXR_DEVELOPMENT_ID=development-<sha256>
make symile-develop CONFIG=configs/symile_cxr_labs_gated_no_observedness.yaml \
  SOURCE_CXR_DEVELOPMENT_ID=development-<sha256>
```

Each successful invocation produces 15 immutable fold packages and one complete family development
authority. Every repeat must contain one OOF prediction for each of the 2,368 strict-pneumonia
development admissions. Neural packages retain selected-epoch histories; LightGBM packages retain
`best_iteration`. The family manifest records the 15 selection budgets in deterministic
fold-coordinate order and records their numeric median as the frozen final-training budget.

After all six family identities exist, publish the aggregate analysis:

```bash
make symile-analyze DEVELOPMENT_IDS="<six explicit development IDs>"
```

Identity validation covers exact config bytes, fit-relevant configuration, bundle and CV lineage,
inner-split identity, reconstructable fitted state, fitted ECDF and feature contracts, validated
neural history, separate OOF logical evidence, and exact fold coverage. Family authorities
recompute each repeat metric and median final-training budget from 15 package/prediction pairs. Cross-family authorities
recompute paired effects and mean-logit ensemble metrics from the six families, and summary text
must equal the deterministic manifest rendering.
MLflow records one `training`/`oof` run per fold and sets `run_complete=true` after immutable
package and prediction publication plus operational completion bookkeeping succeeds. Complete
family development publication and validation subsequently revalidate exact package/prediction
membership against the pinned complete CV authority. Paths, MLflow run IDs, timestamps, and
runtime hardware remain outside semantic artifact identity.

The development reader requests only official train and validation rows and authenticates only the
corresponding CXR tensors. It has no official-test accessor. Development produces no thresholds,
calibration, final full-development model, or held-out-test statistic.

## Reproduce the Symile campaign

This is formal scientific execution, not a smoke test. Run it only after the certified science
commit is sealed and the checkout is clean. Formal Symile execution has not occurred.
Transfer the authorized frozen bundle and CV artifact with their original manifest bytes into
`data/manifests/symile/`; all seven Symile configs pin those exact authorities. Rebuilding equivalent
tables alone does not guarantee the pinned manifest-byte hash. The campaign validates the pinned
authorities and never substitutes `CURRENT` or regenerates them.

Place the complete authorized Symile-MIMIC 1.0.0 source at the supplied source root. Before neural
development, materialize the fixed public CXR initialization through the existing acquisition helper:

```bash
uv run --locked --no-dev python -c 'from radfusion.models.cxr_baseline import ensure_pretrained_weights; ensure_pretrained_weights()'
```

Preserve that exact weight file throughout development and final fitting. Unlike the RSNA campaign,
Symile fitting fingerprints an existing cache entry rather than acquiring missing weights. For the
formal GPU run use the locked runtime environment and `DEVICE=cuda`. `BACKUP_ROOT` must name a
separate approved persistent destination outside the resolved repository root (including symlink
targets). The formal command derives the manifest, model, report, private, MLflow, and export roots
from the checkout and does not accept overrides for them.

The formal campaign is automated: a validated immutable freeze is followed immediately by its
atomic same-freeze test-open record. The freeze is an ordering and lineage boundary, not a manual
approval pause. There is no separate public test-opening command.

The primary Symile comparison is gated CXR/labs versus CXR-only, with AUROC primary and Average
Precision and Brier score secondary. Effects are candidate minus comparator: positive AUROC/AP
favors the candidate, while negative Brier favors the candidate. These declarations are validated
in the frozen evaluation policy.

Private FP/FN selections use the primary gated raw probability and frozen development Youden-J
threshold. Content-addressed JSON derivatives under `private/error-review/symile/` are recursively
validated and preserved in export/backup. They contain sample identifiers and must remain private.
Their absence or corruption does not invalidate the global result; a conflicting derivative is
rejected, not overwritten, and an absent derivative is regenerated on resume.

The only public held-out campaign command is:

```bash
make symile-campaign SOURCE_ROOT=path/to/private/symile/source DEVICE=cuda \
  BACKUP_ROOT=path/to/approved/persistent/backup
```

It validates the exact-six core development authorities, executes the separate ECG development
stage, creates exactly 14 full-development packages, and requires a validated freeze plus the
atomic same-freeze test-open record before the canonical accessor can materialize official-test
state. It then publishes 14 raw prediction evidences, derives six predictor views and one global
result, and preserves the result through validated export and backup. Both `outbox/` and the
separately configured backup location contain private scientific state and remain outside version
control.
The deterministic archive contains an exact V1 preservation manifest mapping every included
authority to its canonical restore path and hashing every archived file. Both the local archive and
external backup are created and validated with owner-only read/write permissions (`0600`).
When a valid `test-open.json` already exists, `symile-campaign` revalidates and resumes the exact
opened freeze without development or final retraining, completing only missing post-open work.

### Numerical runtime on resume

Before test opening, all neural packages must agree on inference precision. The campaign resolves
one effective runtime and freezes exactly `device_type`, `autocast_dtype`, `cuda_runtime_version`,
`cudnn_version`, `gpu_device_name`, and `gpu_compute_capability`. CPU execution uses `None` for
CUDA-only fields. On an opened resume, the resolved runtime must match all six fields before any
official-test materialization. Already-published immutable predictions must not be combined with
new predictions from a different numerical CUDA environment.

GPU device index, requested device string, hostname, pin-memory setting, MLflow run identity, and
timestamps are not part of this runtime projection. Dependency/library versions remain bound by
the frozen dependency-lock authority. The same validated runtime object is used for inference;
deterministic PyTorch/cuDNN backend settings are explicitly established on fresh and resumed runs.

## Probability and operating-point metrics

Models expose class labels and probabilities. Evaluation locates the column labeled `1` and
rejects invalid class or probability contracts.

Average precision is computed with `average_precision_score`. Probability metrics also include
ROC-AUC and Brier score. Threshold-dependent precision, recall, specificity, F1, and confusion
counts are grouped by operating point:

- the Youden-J threshold maximizes sensitivity minus false-positive rate;
- the target-sensitivity threshold is the highest threshold meeting sensitivity 0.90.

RSNA derives both thresholds from validation under its evaluation config. Symile derives them only
for the primary gated ensemble from the three-repeat mean-logit development OOF prediction.

Both policies enumerate every finite ROC threshold and choose the highest threshold among ties or
qualifying candidates. The thresholds are applied unchanged to test probabilities. They are
benchmark operating points, not clinical optima.

## Calibration

RSNA expected calibration error uses the configured number of equal-width bins over `[0, 1]`. Bins are
lower-inclusive; the final bin includes 1. Empty bins contribute zero. Each non-empty bin
contributes its sample fraction times the absolute difference between mean predicted probability
and observed positive fraction. Experiment configs use 15 bins, and the plot uses the same count
and strategy. Symile reports no ECE scalar and uses a descriptive ten-bin uniform reliability curve.

Calibration slope and intercept come from an L2 logistic regression of the target on predicted
log-odds. Probabilities are clipped to `[1e-6, 1 - 1e-6]` before the logit transform. The fit uses
`C=1e6`, the `lbfgs` solver, 2,000 maximum iterations, and an intercept.

Calibration statistics describe raw probabilities, not fitted recalibration. RSNA outputs use its
class-weighting policies; Symile uses no class weighting. Calibration slope/intercept are point
estimates. Symile's paired held-out AUROC, Average Precision, and Brier differences separately use
2,000 subject-cluster bootstrap resamples; development OOF has no bootstrap confidence intervals.

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
lineage, completion state, and project-output references. Held-out evaluation runs add evaluator
provenance, test metrics, and the authoritative source model-package and evaluation identities.

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

The default suite uses synthetic fixtures and the public pydicom test fixture bundled with the
locked pydicom package. Local integration tests are excluded by default; the CUDA autocast case skips
when CUDA is unavailable. Local RSNA integration tests run when the source dataset is
available:

```bash
uv run pytest -m integration
```

`make purge-generated` is a destructive reset for generated bundles, reports, models, and MLflow
state, not a preservation operation. It refuses an existing canonical Symile test-open record.
Preserve complete scientific evidence and backups before cleanup; raw datasets remain external
inputs. The required operator-supplied `BACKUP_ROOT` must be a separately approved persistent
destination outside the resolved repository root, including after resolving symlinks.
