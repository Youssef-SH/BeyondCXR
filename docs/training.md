# Experiments

Each experiment is defined by one strict YAML file:

```yaml
config_version: 1
name: metadata_logistic_regression

dataset:
  registry_key: rsna
  manifest_directory: data/manifests
  bundle_id: build-<sha256>
  task_id: pneumonia

model:
  registry_key: metadata_logistic
  parameters: {}
  fit_parameters: {}

training:
  seed: 42
  report_directory: reports
  model_directory: models/rsna

evaluation:
  sensitivity_target: 0.90
  calibration_bins: 15
  latency_warmup_calls: 100
  latency_measured_calls: 1000

mlflow:
  experiment_name: radfusion-rsna
```

The loader rejects missing, unknown, and duplicate keys. Each experiment names an exact immutable
bundle ID. `training.seed` is the single randomness authority.

All experiment configs use schema version 1. Metadata configs are `configs/metadata_logistic.yaml`
and `configs/metadata_lightgbm.yaml`. Image configs are `configs/image_densenet_seed17.yaml`,
`configs/image_densenet_seed42.yaml`, and `configs/image_densenet_seed2026.yaml`. Fusion configs use
the corresponding `configs/fusion_concat_seed*.yaml` files.

## Feature boundary

The RSNA adapter exposes these model features:

- `age_years`
- `age_is_implausible`
- `sex`
- `view_position`
- `pixel_spacing_row_mm`
- `pixel_spacing_col_mm`

Preprocessing rejects additional columns. Sample IDs, patient IDs, image paths, targets,
partitions, and lineage remain separate from the feature frame. Imputation, categories,
missingness indicators, scaling, and class weighting are derived from training data.
The model package records this ordered input contract, its type categories and missing-value
semantics, and the policy version. The fitted preprocessing pipeline is embedded in
`model.skops`.

## Execution

Contributor setup uses `uv sync --locked --group dev`; a paid GPU campaign host uses
`uv sync --locked --no-dev`.

```bash
make rsna-gpu
```

This is the authoritative full RSNA workflow. It prepares the configured pretrained weight,
publishes and audits the bundle, builds the deterministic image cache,
completes all eight training packages before test access, evaluates all eight packages, runs seed
summaries, localization, and comparison, validates the output surface, and publishes a portable
archive with a checksum. Producer results pass exact run IDs directly to
dependent consumers in memory. The command requires a fresh generated-output surface and preserves
partial outputs on failure; `make purge-generated` is the explicit destructive reset.

The lower-level `make train`, `make evaluate`, `make summarize-seeds`, `make localize`, and
`make compare` commands remain available for targeted inspection and debugging of immutable runs.
Within `make rsna-gpu`, held-out evaluation begins only after all eight training packages have
frozen; lower-level explicit evaluation verifies only the selected immutable run and package.

Training validates the complete pinned bundle, then performs projected and filtered reads for
train and validation only. It fits preprocessing and the estimator on train, uses validation for
LightGBM early stopping, selects both operating thresholds on validation, and publishes the fitted
model.

Test evaluation is a separate run. Before reading test data, it validates the package and its
semantic ID, lineage to the source training run, configuration and model hashes, fitted input
contract, validation-derived choices, and LightGBM best iteration. Formal evaluation accepts
packages from clean training commits and requires the evaluator to use the same clean Git commit
and dependency lock. It then reads only test and applies the verified choices unchanged.

The built-in dataset and model adapters are held in immutable mappings. One tabular runner owns
metadata training, one shared neural core owns two-stage optimization, and narrow image and fusion
orchestrators own their data and publication boundaries. The explicit evaluator owns test
evaluation. Dispatch is determined by `model.modality`.

## Image training

The image configuration defines one seed, the fixed TorchXRayVision DenseNet121 encoder, source
dataset root, augmentation, optimization stages, and device policy.
Each invocation trains one seed. Training validates the pinned bundle, loads only train and
validation rows, and verifies their coverage by the validated deterministic CXR cache before
constructing the model.

The image configuration pins the immutable semantic bundle ID. Bundle validation computes the
observed bundle-manifest SHA-256 and verifies its physical, logical, semantic, split, and source
contracts. Training freezes that exact identity in the model package and copies it to an MLflow
parameter; linked evaluation requires the same bundle-manifest SHA-256 before test access.
Cache derivation identity and source-authentication provenance are package-bound. Mapping and
image-content hashes validate the disposable cache locally and do not participate in model
semantic identity or the model package ID.

### Pretrained weight file and provenance

The campaign materializes `densenet121-res224-chex` through TorchXRayVision's supported acquisition
path. Image training requires the URL-derived cache entry to be a regular non-symlink file,
fingerprints it immediately before and after model construction, and requires exact equality. This
establishes local run provenance but does not independently authenticate the file against an
official upstream digest. Test evaluation reconstructs with `weights=None` and loads only the
packaged trained state.

Training performs head-only warm-up followed by full fine-tuning. Validation Average Precision
selects the retained state across both stages and controls fine-tuning scheduling and early
stopping. Final deterministic validation inference freezes the Youden-J and target-sensitivity
thresholds. Campaign preparation may authenticate and deterministically cache image bytes from all
partitions without reading task labels, fitting statistics, selecting thresholds, or fitting a
model. Official training consumes only train and validation task rows; held-out labels and test
evaluation remain unavailable inside the canonical campaign until all eight training packages are
frozen.
Epoch history records the learning rates used during each epoch, before scheduling the next epoch.

The deterministic CXR cache stores float32 `[1, 224, 224]` images after DICOM decoding,
MONOCHROME handling, center crop, and resize, immediately before stochastic augmentation. Training
sample order is a stable function of seed and epoch. Augmentation is a stable function of seed,
epoch, and sample ID, so worker count, worker lifetime, and prefetch timing do not change the
scientific realization.

Train and validation loaders use the configured reused-loader worker count, which is 2 in the
canonical RSNA configurations. Positive-worker loaders use `spawn`, persistent workers, and
prefetch factor 2. One-shot test inference is synchronous and uses no worker processes,
persistence, prefetch, or multiprocessing context. The actual lifecycle-specific policy is
recorded as runtime provenance outside semantic experiment compatibility. Batch size, augmentation,
optimization, AMP, and all other numerical policies remain scientific configuration.

Image packages contain:

```text
models/rsna/runs/<training-run-id>/
  model.pt
  resolved_config.yaml
  model_manifest.json
```

`model.pt` is a validated CPU tensor state dictionary with selection metadata. The package manifest
binds the checkpoint, path-independent experiment meaning, archived configuration, dataset and
split lineage, cache-construction source authentication, pretrained-weight byte identity,
transform contracts, training policy, selected validation state, and frozen thresholds. The
evaluator validates this package, reconstructs the DenseNet architecture without loading the
original TorchXRayVision cache, and strictly loads the complete state before reading test rows.

## Tracking and outputs

### Operational progress

Instrumented entrypoints emit lifecycle records and rate-limited aggregate progress to stderr;
their final machine-readable result remains on stdout. The full campaign also writes the same
records to `reports/rsna/campaigns/<campaign-id>/execution.log`. `epoch_throughput` operational
records include separate training and validation elapsed time, batches per second, and samples per
second. The bookkeeping uses host clocks and aggregate counters without per-batch logging or extra
CUDA synchronization. Localization emits one timed phase and bounded aggregate prediction and
Grad-CAM progress per seed without sample identifiers.

Model packages under `models/` and complete reports under `reports/` are the authoritative
physical outputs. MLflow stores the run ledger, status, parameters, scalar metrics, provenance,
lineage, completion state, references to project-owned outputs, and the exact loaded training
configuration. Run metadata lives in `mlflow.db`; training configuration artifacts live under
`mlartifacts/`. Inspect local runs with:

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db
```

Evaluation reports contain scientific results and device/software facts. The actual cache identity
and loader execution policy are durable MLflow run parameters.

A training run logs the exact loaded YAML before dataset access and records resolved split and
label-policy lineage before fitting. Training and test-evaluation runs become complete only after
their local run-qualified outputs are published. Training runs publish model packages; test runs
link to their source training runs and publish test reports.

Local training packages contain:

```text
models/rsna/runs/<training-run-id>/
  model.skops
  resolved_config.yaml
  model_manifest.json
```

Image packages use the same hierarchy with `model.pt` in place of `model.skops`.

Fusion packages add `structured_preprocessor.skops`. Fusion training receives an explicit
same-seed image training-run ID at execution time, verifies that its clean Git revision and
dependency lock match the fusion execution, initializes only the image encoder, and records the
source package, checkpoint, semantic config, Git, and dependency-lock lineage. The runtime run ID
is lineage rather than scientific YAML. The train-fitted structured preprocessor and its exact
ordered feature contract are part of the package. Fusion evaluation reconstructs from that package
and applies its frozen validation thresholds to test.

The manifest records `model_package_schema_version` and a deterministic `model_package_id`. Image
package identity includes the selected checkpoint and observed bundle-manifest SHA-256, making it
an exact provenance identity. Runtime provenance, operational paths, and training-run ID remain
outside this identity; the archived configuration retains exact byte-hash validation.

Dirty training runs may publish traceable packages, but those packages are ineligible for formal
test evaluation. Test-evaluation runs record their own Git and dependency provenance, the model
package ID, and their source training run.

Each training or test-evaluation run publishes aggregate reports under
`reports/rsna/runs/<run-id>/`:

```text
metrics.json
evaluation_report.md
confusion_summary.md
roc_curve.png
precision_recall_curve.png
calibration_curve.png
confusion_matrix_youden_j.png
confusion_matrix_target_sensitivity.png
```

Publication requires exactly this set after privacy validation. `make compare` deterministically
regenerates `reports/model_comparison_table.csv` and `.md` from complete, finite MLflow records.
Rows include modality, task, and model package identity. Image and fusion rows are published only
for verified test-evaluation runs; failed, unfinished, and incomplete runs are excluded.

`make summarize-seeds` accepts exactly three explicit compatible image or fusion test-run IDs for
seeds 17, 42, and 2026. It publishes deterministic JSON, CSV, and Markdown under
`reports/<dataset>/seed-summaries/<report-id>/`, preserving every seed-specific metric and reporting
arithmetic means and sample standard deviations for applicable results. Thresholds remain
seed-specific; the summary selects no canonical seed and creates no averaged model.

`make localize` accepts the three explicit image test-run IDs and evaluates Grad-CAM against the
union of RSNA boxes for every positive test sample. Public output contains per-seed and aggregate
pointing-game and activation-energy results. Mechanically selected real-image overlays and their
internal traceability manifest are published only under `private/localization/`.

Image and fusion test evaluators publish one ordered Parquet table under
`private/predictions/<dataset>/<test-run-id>/`. It binds each test row's sample and private patient
keys, target, logit, probability, split, seed, training run, evaluation run, and model package.
These patient-level tables are neither public reports nor MLflow artifacts.

The training, evaluation, and comparison CLIs default to `sqlite:///mlflow.db` and accept
`--tracking-uri` when an isolated local SQLite database is required.

Metric definitions and experimental protocols are documented in
[`reproducibility.md`](reproducibility.md).
