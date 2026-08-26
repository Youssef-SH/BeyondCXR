# Reproducibility

BeyondCXR distinguishes exact artifact/result reproduction from experimental replication.
Reproduction validates preserved fitted state and deterministically regenerates aggregate outputs.
Replication retrains the prespecified procedure and records any environment-dependent variation.

Wheels and source distributions package the software only. Exact study reproduction requires the
provenance-linked Git checkout, its `uv.lock`, the frozen configurations, and validated scientific
authorities; installing a distribution archive alone does not reconstruct the study.

## Environment

The project targets Python 3.13. `pyproject.toml` declares direct dependencies and `uv.lock` fixes
the complete graph.

```bash
uv sync --locked --group dev --extra serving
uv run pre-commit install
```

Scientific execution records Git and lock identities, exact configs, dataset and split lineage,
seeds, model lineage, and runtime facts. Git and lock witnesses support reproduction; scientific
object identity remains content- and policy-based. Use a Linux or WSL2 Git checkout for campaign
execution.

## Data authorities

Dataset builds authenticate source releases, stage complete typed artifacts, validate them, and
publish an immutable content-addressed directory. Logical Arrow hashes cover ordered values and
schema. Physical hashes authenticate serialized bytes. Split assignment IDs cover canonical
sample-to-split mappings.

```bash
make rsna-manifest SOURCE_ROOT=/approved/rsna
make rsna-audit BUNDLE_ID=bundle-SHA256

make symile-manifest SOURCE_ROOT=/approved/symile
make symile-audit BUNDLE_ID=bundle-SHA256
make symile-cv BUNDLE_ID=bundle-SHA256
```

Scientific configs pin exact bundle, bundle-manifest, split, and applicable CV identities. A
semantically equivalent reconstruction with different required manifest bytes does not satisfy
that pinned execution state.

## Scientific execution

The supporting RSNA campaign is one ordered command:

```bash
make rsna-campaign
```

It validates source inventory, deterministic cache, all packages, held-out evaluations, seed
summaries, localization outputs, MLflow integrity, and the portable archive.

### Supporting RSNA reproducibility

Bundle qualification authenticates every source DICOM byte. The deterministic image cache stores
the decoded, spatially standardized pre-augmentation CXR representation. Cache preparation may
process image bytes from every partition, but it reads no task labels, fits no model statistics,
selects no thresholds, and fits no model. All eight RSNA packages freeze before held-out
evaluation begins.

CXR fitting uses TorchXRayVision's `densenet121-res224-chex` initialization. The local
initialization file is fingerprinted immediately before and after model construction; this records
the exact local bytes without claiming an official upstream cryptographic digest. Held-out
reconstruction builds the architecture with `weights=None` and loads packaged trained state, so it
does not reacquire or depend on the original pretrained cache.

Sample order derives from seed and epoch; stochastic augmentation derives from seed, epoch, and
sample identity. Worker assignment, lifetime, and prefetch timing therefore do not change the
scientific realization. Seed summaries require exactly three compatible seed-specific evaluations,
preserve each member result, and report the arithmetic mean and sample standard deviation. They
select no seed and produce no averaged model.

Localization reports the pointing game, which asks whether the row-major first maximum of the
Grad-CAM heatmap lies inside the union of reference boxes, and activation energy, the fraction of
nonnegative heatmap mass inside that union. A zero heatmap receives zero for both metrics.

Primary Symile development runs each prespecified family over the pinned 3 × 5 CV assignment.
Fusion families receive the explicit compatible CXR family identity. Their development results
feed the aggregate analysis described in [training](training.md).

Formal Symile execution uses the sealed science-execution commit in a clean checkout:

```bash
uv sync --locked --no-dev
make symile-campaign \
  SOURCE_ROOT=/approved/symile/source \
  BACKUP_ROOT=/approved/persistent/backup
```

The campaign validates CXR/labs development, executes the secondary ECG analysis, fits and
reconstructs all 14 final packages, freezes the complete pre-test state, opens the exact freeze,
publishes 14 predictions and one global result, and creates a checksummed export plus external
backup. The backup root must resolve outside the repository.

## Test firewall and resume

Development accessors expose official train and validation data only. Final fitting also remains
test-blind. Held-out test materialization requires a genuine validated freeze capability and its
same-freeze test-open record.

The opening record binds the freeze, bundle, split, formal Git commit, and dependency lock. Once it
exists, the campaign can regenerate only missing deterministic post-open objects under that exact
freeze. It cannot rerun development, alter thresholds, retrain final packages, or adopt a different
numerical inference runtime.

For CUDA resume, the effective device type, autocast dtype, CUDA runtime, cuDNN version, GPU name,
and compute capability must match the freeze. CPU execution records the corresponding null CUDA
fields. This boundary prevents silent numerical-runtime drift in mean-logit predictions.

## Artifact and result reproduction

After formal execution, `make results` validates a preservation export and derives
`results/symile/binding.json` from its restored scientific authorities. The binding names the
sealed science-execution commit, data/split/CV identities, development identities, pre-test freeze,
exact package and prediction membership, and global result. It contains no paths, copied metrics,
preservation witnesses, byte hashes, or serving coordinates.

Generate the public result surface from the preserved campaign archive:

```bash
make results \
  ARTIFACT_ROOT=/approved/preserved/global-result-SHA256.zip
```

The command validates the export manifest, restores into a temporary directory, recursively runs
the campaign validators, compares every bound identity, constructs an aggregate-only public
projection, and publishes the fixed result surface and bounded README/model-card content.

Verify committed derivatives byte-for-byte without changing them:

```bash
make reproduce \
  ARTIFACT_ROOT=/approved/preserved/global-result-SHA256.zip
```

Reproduction fails on wrong lineage, corrupted evidence, unexpected output membership, changed
bytes, private fields, or stale document regions. Prediction rows never enter the public output
tree.

The recorded dependency environment defines byte-level reproducibility for CSV, Markdown, JSON,
and SVG derivatives. SVG metadata omits timestamps and uses a fixed hash salt. Reproduction outside the supported environment is experimental
replication, not a byte-stability claim.

After the result-bearing release candidate is committed, publish and exercise the runtime-mounted
serving control against that known code state:

```bash
make serving-authority \
  ARTIFACT_ROOT=/approved/preserved/global-result-SHA256.zip \
  AUTHORITY_ROOT=/approved/private/serving-authorities
```

The serving authority records the release-candidate commit. Publication is idempotent when an
existing authority has the same identity, membership, and bytes; divergent existing content is
rejected. The private runtime control is not copied into tracked result files, avoiding circular
Git provenance.

## Experimental replication

Replication uses the same authenticated releases, immutable bundle and CV assignments, configs,
seeds, preprocessing, selection rules, terminal budgets, and campaign procedure. It records the
replication runtime and produces independently validated objects. GPU state and metrics may
differ across supported environments; differences are reported rather than hidden behind a fixed
metric tolerance.

Replication does not change the original result binding. It is a separate scientific execution.

## Evaluation definitions

Symile reports AUROC, Average Precision, and Brier score on raw probabilities. Ten-bin uniform
reliability curves are descriptive; no post-hoc recalibration is applied. The primary paired test
effect is gated minus CXR. Positive AUROC or Average Precision effects favor gated fusion;
negative Brier effects favor gated fusion.

Calibration slope and intercept are descriptive point estimates derived from raw predicted
probabilities. Probabilities are clipped to `[1e-6, 1 - 1e-6]`, transformed to log-odds, and used
as the single feature in an L2 Logistic Regression with `C=1e6`, solver `lbfgs`, `max_iter=2000`,
and an intercept. This assesses raw probabilities; it does not fit or apply recalibration.

Held-out uncertainty uses a subject-cluster bootstrap with 2,000 accepted resamples and the
prespecified linear 2.5th/97.5th percentile interval. Repeated-CV evidence is reported by repeat
and as the prespecified mean-logit ensemble without a bootstrap confidence interval.

The primary ensemble's Youden-J threshold maximizes sensitivity minus false-positive rate. Its
target-sensitivity threshold is the highest threshold meeting sensitivity 0.90. Both enumerate
finite ROC thresholds and choose the highest threshold on ties. They are derived from aggregate
development OOF evidence and applied unchanged to test probabilities; they are research operating
points, not clinical decision rules.

Supporting RSNA probability reports use expected calibration error with the configured equal-width
bins over `[0, 1]`; the final bin includes 1, empty bins contribute zero, and non-empty bins are
weighted by sample fraction times the absolute confidence-to-observed-frequency gap. The RSNA
experiment configurations use 15 bins. Calibration slope and intercept, where reported, use the same raw-
probability procedure described above. RSNA latency covers the complete CPU preprocessing and
probability path: 100 warm-up calls followed by the median of 1,000 individually timed calls on the
deterministic first sample. It depends on hardware and system load. Model size is the exact
serialized model byte count divided by 1,048,576 MiB; latency is reported only where the existing
RSNA reporting contract defines it.

## Serving reproduction

Serving starts from one explicit validated authority and the three bound primary gated packages:

```bash
uv sync --locked --extra serving
make symile-serve \
  AUTHORITY=/approved/serving-authority-SHA256 \
  PACKAGE_ROOT=/approved/final/packages
```

Startup recursively validates package identities, byte witnesses, fitted preprocessing, task,
ensemble policy, and provenance. It reconstructs neural models with `weights=None`; no scientific
model is downloaded. Docker deployment mounts restricted authorities and packages read-only. See
[controlled serving](serving.md).

## Quality and release checks

Contributor gates are:

```bash
make lock-check
make lint
make format-check
make test
make release-check
make pre-commit
uv build
```

`make release-check` performs the CI-safe documentation, privacy, Docker-context, restricted-file,
and result-membership checks. Final release acceptance additionally requires a clean fresh
checkout, exact reproduction, wheel/sdist inspection, staged host-side authority construction and
in-process ASGI exercise, Docker image-hygiene inspection, and containerized reconstruction plus
`/health`, `/model-info`, and synthetic `/predict` checks against read-only authority and package
mounts.

After the result-bearing release candidate is committed, run that complete gate with:

```bash
make release-check FINAL=1 \
  ARTIFACT_ROOT=/approved/preserved/global-result-SHA256.zip \
  AUTHORITY_ROOT=/approved/private/serving-authorities
```

The final mode clones the clean candidate locally, installs the recorded dependency environment,
runs quality gates, builds and inspects distributions, reproduces public results, stages and
host-smoke-tests the private serving authority, performs both Docker acceptance stages against the
staged authority, confirms the clone remains clean, and only then atomically publishes the final
authority. Normal CI uses the fast mode only.

Generated bundles, packages, predictions, reports, caches, MLflow state, and campaign exports are
private local artifacts. See [privacy](privacy.md) and the [data statement](data_statement.md).
