# Reproducibility

## Environment

The project targets Python 3.13. `uv.lock` defines the dependency environment.

```bash
uv sync --locked
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
make rsna-audit
make train CONFIG=configs/metadata_logistic.yaml
make train CONFIG=configs/metadata_lightgbm.yaml
make train CONFIG=configs/image_densenet_seed42.yaml
make evaluate RUN_ID=<training-run-id>
make summarize-seeds TEST_RUN_IDS="<test17> <test42> <test2026>"
make compare
```

`<training-run-id>` denotes the run ID printed by training.

Audits are published under `reports/rsna/audit/<bundle-id>/`. Rebuilding one audit replaces only
that bundle-qualified audit directory.

Executable configs pin the exact bundle ID and one training seed. Preprocessing is fitted on
training data. Validation selects the LightGBM stopping point and both operating thresholds.
`make evaluate` verifies those choices against the source training run before applying them to
test in a separate linked run.

Image experiment configurations pin the semantic bundle ID. Validation computes the observed
bundle-manifest SHA-256 and verifies the manifest's physical, logical, semantic, split, and source
contracts. Training records the hash in its package and an MLflow parameter; linked test evaluation
requires the same bundle-manifest SHA-256 while authenticating only its authorized partition.

Training runs record the Git commit and dirty status, exact configuration bytes and hash,
dependency-lock hash, environment, dataset identity, and model lineage. Formal test evaluation
requires a package produced from the evaluator's clean Git commit and matching dependency lock.
Test-evaluation runs record their own code and lock provenance and link the verified model package
to its source training run.

Image training authenticates only train and validation DICOMs against the bundle source inventory.
It seeds Python, NumPy, PyTorch, DataLoader shuffling, workers, and deterministic kernels once per
run. Image training fingerprints the materialized pretrained weight file immediately before and
after model construction and requires exact equality. The selected CPU checkpoint may come from
head-only warm-up or full fine-tuning. Explicit test evaluation verifies the package, checkpoint,
code, lock, and model structure before loading test rows, then authenticates only test DICOMs and
applies the validation thresholds unchanged. It reconstructs the architecture without reading or
downloading the original pretrained cache.

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

## Probability and operating-point metrics

Models expose class labels and probabilities. Evaluation locates the column labeled `1` and
rejects invalid class or probability contracts.

Average precision is computed with `average_precision_score`. Probability metrics also include
ROC-AUC and Brier score. Threshold-dependent precision, recall, specificity, F1, and confusion
counts are grouped by operating point:

- the Youden-J threshold maximizes validation sensitivity minus false-positive rate;
- the target-sensitivity threshold is the highest validation threshold meeting the configured
  sensitivity, which is 0.90 in the executable configs.

Both policies enumerate every finite ROC threshold and choose the highest threshold among ties or
qualifying candidates. The thresholds are applied unchanged to test probabilities. They are
benchmark operating points, not clinical optima.

## Calibration

Expected calibration error uses the configured number of equal-width bins over `[0, 1]`. Bins are
lower-inclusive; the final bin includes 1. Empty bins contribute zero. Each non-empty bin
contributes its sample fraction times the absolute difference between mean predicted probability
and observed positive fraction. Executable configs use 15 bins, and the plot uses the same count
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

## Quality gates

```bash
uv lock --check
uv sync --locked
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
