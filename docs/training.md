# Experiments

Each experiment is defined by one strict YAML file:

```yaml
config_schema_version: 1

dataset:
  dataset_id: rsna
  bundle_id: bundle-<sha256>
  bundle_manifest_sha256: <sha256>
  split_assignment_id: split-assignment-<sha256>

task:
  task_id: pneumonia
  label_policy_version: rsna-stage-2-target-v1

family:
  family_id: metadata_logistic
  modalities: [metadata]
  parameters: {}

preprocessing:
  metadata_policy: rsna-metadata-input-v1

training:
  selection_metric: none
  parameters:
    l1_ratio: 0.0
    solver: liblinear
    C: 1.0
    max_iter: 2000
    class_weight: balanced

evaluation:
  sensitivity_target: 0.90
  calibration_bins: 15

```

The loader rejects missing, unknown, duplicate, mistyped, non-finite, and incompatible values.
Each experiment names an exact immutable bundle and exact manifest-byte integrity witness.
Filesystem paths, MLflow destinations, hardware destinations, execution seeds, worker counts, and
pin-memory policies are runtime coordinates and do not appear in YAML.

Family parameters describe estimator/model topology. Preprocessing identifiers own implemented
scientific transforms. Fitting, optimization, loss weighting, iteration, early-stopping, loader,
and augmentation policy belong to `training`. `selection_metric` is `none` when fitting performs
no model selection. RSNA evaluation config owns sensitivity and calibration policy; Symile's
campaign owns its fixed held-out policy and development-derived primary thresholds. Latency benchmarking uses
the operational defaults of 100 warm-up calls and 1,000 measured calls; those counts are outside
scientific configuration identity.
LightGBM runs quietly with operational `verbosity=-1`; logging verbosity is not scientific
configuration.

All configuration and generated scientific-object schemas use version 1. RSNA uses one YAML for
each of metadata Logistic Regression, metadata LightGBM, CXR DenseNet, and CXR-metadata concat.
Symile uses one YAML for each of labs Logistic Regression, labs LightGBM, CXR DenseNet, CXR-labs
concat, CXR-labs gated, and the gated no-observedness ablation, plus the campaign-internal
`symile_cxr_labs_ecg_gated.yaml` extension. The canonical filenames are listed
in the repository `configs/` directory. Manual RSNA training receives `SEED` explicitly; the
authoritative campaign owns its fixed family-by-seed matrix.

## RSNA feature boundary

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

## RSNA execution

Contributor setup uses `uv sync --locked --group dev`; a paid GPU campaign host uses
`uv sync --locked --no-dev`.

```bash
make rsna-campaign
```

This is the authoritative full RSNA workflow. It prepares the configured pretrained weight,
publishes and audits the bundle, builds the deterministic image cache,
completes all eight training packages before test access, evaluates all eight packages, runs seed
summaries, localization, and comparison, validates the output surface, and publishes a portable
archive with a checksum. Producer results pass exact package and evaluation IDs directly to
dependent consumers in memory. The command requires a fresh generated-output surface and preserves
partial outputs on failure; `make purge-generated` is the explicit destructive reset.

The lower-level `make rsna-train`, `make rsna-evaluate`, `make rsna-summarize`,
`make rsna-localize`, and `make rsna-compare` commands operate on explicit immutable objects.
Within `make rsna-campaign`, held-out evaluation begins only after all eight training packages have
frozen; lower-level evaluation requires both the selected immutable package and an explicit
compatible canonical config.

Training validates the complete pinned bundle, then performs projected and filtered reads for
train and validation only. It fits preprocessing and the estimator on train, uses validation for
LightGBM early stopping, selects both operating thresholds on validation, and publishes the fitted
model.

Test evaluation is an operational run around a scientific object chain. Before reading test data,
it validates the package semantic ID, integrity witnesses, fitted input contract,
validation-derived choices, LightGBM best iteration, and package-scoped compatibility of the
explicit evaluation config. The package supplies fitted state and frozen thresholds; the explicit
config supplies downstream evaluation policy. Evaluation then reads only test, publishes private
prediction evidence, and derives an immutable aggregate evaluation. Git and dependency-lock facts
remain reproducibility witnesses outside scientific package identity.

The built-in dataset and model adapters are held in immutable mappings. One tabular runner owns
metadata training, one shared neural core owns two-stage optimization, and narrow CXR and fusion
orchestrators own their data and publication boundaries. The explicit evaluator owns test
evaluation. Dispatch is determined by the canonical family ID and ordered modalities.

## RSNA CXR training

The CXR configuration defines the fixed TorchXRayVision DenseNet121 encoder, augmentation, and
optimization stages; runtime owns the source root, device, and explicit execution seed. Each
invocation trains one seed. Training validates the pinned bundle, loads only train and
validation rows, and verifies their coverage by the validated deterministic CXR cache before
constructing the model.

The CXR configuration pins the immutable semantic bundle ID. Bundle validation computes the
observed bundle-manifest SHA-256 and verifies its physical, logical, semantic, split, and source
contracts. Training freezes that exact identity in the model package and copies it to an MLflow
parameter; linked evaluation requires the same bundle-manifest SHA-256 before test access.
Cache derivation identity and source-authentication provenance are package-bound. Mapping and
image-content hashes validate the disposable cache locally and do not participate in model
semantic identity or the model package ID.

### Pretrained weight file and provenance

The campaign materializes `densenet121-res224-chex` through TorchXRayVision's supported acquisition
path. CXR training requires the URL-derived cache entry to be a regular non-symlink file,
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

Train and validation loaders use the runtime-owned reused-loader worker count, which defaults to 2
for RSNA and Symile. Positive-worker loaders use `spawn`, persistent workers, and
prefetch factor 2. One-shot test inference is synchronous and uses no worker processes,
persistence, prefetch, or multiprocessing context. The actual lifecycle-specific policy is
recorded as runtime provenance outside scientific semantic identity. Batch size, augmentation,
optimization, and AMP remain scientific configuration. Worker count and pin-memory policy are
runtime/reproducibility coordinates; prefetch factor and persistent-worker use are derived
execution behavior.

CXR packages contain:

```text
models/rsna/packages/model-package-<sha256>/
  model.pt
  resolved_config.yaml
  manifest.json
```

`model.pt` is a validated CPU tensor state dictionary with selection metadata. The package manifest
binds the checkpoint, path-independent experiment meaning, archived configuration, dataset and
split lineage, cache-construction source authentication, pretrained-weight byte identity,
transform contracts, training policy, selected validation state, and frozen thresholds. The
evaluator validates this package, reconstructs the DenseNet architecture without loading the
original TorchXRayVision cache, and strictly loads the complete state before reading test rows.

## Symile development-only repeated cross-validation

Six strict family configs define the implemented Symile development surface:

```text
configs/symile_labs_logistic.yaml
configs/symile_labs_lightgbm.yaml
configs/symile_cxr_densenet.yaml
configs/symile_cxr_labs_concat.yaml
configs/symile_cxr_labs_gated.yaml
configs/symile_cxr_labs_gated_no_observedness.yaml
```

Every config pins the immutable Symile bundle, observed bundle-manifest hash, official split
assignment, CV assignment, and strict-pneumonia task. One invocation executes all three repeats and
five outer folds. Repeat seeds and folds come from the CV artifact and are not CLI controls.

For each outer fold, the runner derives one deterministic patient-grouped inner split shared by all
families. The 50 lab ECDFs and missing replacements are fitted on complete outer training and are
applied unchanged to inner training, inner validation, and outer OOF. Logistic Regression fits on
complete outer training without inner selection, so its recorded inner split is audit-only and is
absent from fold-package semantic identity. LightGBM and neural families select on inner validation
AUROC, so their `inner_split_id` is semantic; they are not refitted after selection. Symile neural
fine-tuning exposes only DenseNet `denseblock4` and `norm5`. RSNA neural training fine-tunes the
full encoder and selects fitted state by validation Average Precision.

Fusion runs require the explicit complete CXR development authority. Each fusion fold loads only
the selected CXR fold package with the same repeat, outer fold, inner split, data, and transform
lineage, and initializes only its encoder. Git and lock remain reproducibility witnesses. Both gated variants use the same
architecture and initialization; the ablation replaces observedness inputs with zeros at the model
boundary while preserving the 100-dimensional lab preprocessing contract.

```bash
make symile-develop CONFIG=configs/symile_cxr_densenet.yaml
make symile-develop \
  CONFIG=configs/symile_cxr_labs_concat.yaml \
  SOURCE_CXR_DEVELOPMENT_ID=development-<sha256>
make symile-analyze \
  DEVELOPMENT_IDS="<labs-lr> <labs-lgbm> <cxr> <concat> <gated> <gated-no-observedness>"
```

The analyzer validates the six families without trusting argument order. It reports per-repeat
AUROC, Average Precision, and Brier score; the prespecified paired effects; and neural ensemble
point estimates computed by aligning the three repeat logits, taking their arithmetic mean, and
applying sigmoid once. Repeated folds are not treated as independent replicates and no pooled OOF
bootstrap is performed.

Artifacts have three levels:

```text
models/symile/development/packages/fold-package-<sha256>/
  resolved_config.yaml
  manifest.json
  model.skops                         # lab families
  model.pt                            # neural families
  lab_preprocessor.skops              # fusion families
  training_history.json               # neural families

private/predictions/symile/oof/prediction-<sha256>/
  predictions.parquet
  manifest.json

reports/symile/development/families/development-<sha256>/
  manifest.json
  summary.md

reports/symile/development/analyses/analysis-<sha256>/
  manifest.json
  summary.md
```

The OOF Parquet is restricted patient-level evidence and remains ignored. Public summaries are
aggregate and privacy validated. The public development layer remains exact-six and exposes only
official train and validation.

### Symile terminal full-development fitting

The campaign-internal ECG family uses the same repeated-CV selection procedure with its separate
three-modality gate. Its development result and the core analysis feed the ECG extension result,
which derives primary gated operating thresholds and references the family-owned final budgets.
It does not turn ECG into a seventh public core-development family.

Each neural family's final epoch budget is the median selected one-based epoch across its fifteen
folds. For budget `B`, terminal fitting runs `min(B, 2)` head-only epochs and `max(B - 2, 0)`
terminal-encoder fine-tuning epochs. It uses all development admissions, fixed within-stage learning
rates, and no validation split, scheduler, early stopping, or checkpoint selection. The terminal
state is published, not a newly selected state.

Labs Logistic Regression fits once with seed 42 and no iterative budget. Labs LightGBM fits once
with seed 42 and the family median `best_iteration` as exact `n_estimators`, without early stopping.
CXR, concat, gated, and ECG-gated each fit seeds 17, 42, and 2026, giving fourteen final packages.
The observedness ablation remains development-only. Every fusion member initializes from the exact
validated same-seed final CXR package, matching dataset/bundle, task, encoder, and CXR transform;
fusion terminal optimizer, loader, and augmentation settings are not CXR ancestry requirements.

Final packages live under `models/symile/final/packages/final-package-<sha256>/`. Their packaged
`final_fit_config.json` describes only fit-used full-development semantics, not CV or downstream
evaluation policy. Neural checkpoints contain terminal state and stage budgets. The campaign
validates all packages before freezing held-out execution; execution and resume requirements are
documented in [`reproducibility.md`](reproducibility.md).

## Tracking and outputs

### Operational progress

Instrumented entrypoints emit lifecycle records and rate-limited aggregate progress to stderr;
their final machine-readable result remains on stdout. The RSNA campaign also writes the same
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
label-policy lineage before fitting. Operational runs become complete only after their scientific
objects are published. Training runs publish model packages; evaluation runs record package,
prediction, and evaluation IDs without defining those identities.

Local training packages contain:

```text
models/rsna/packages/model-package-<sha256>/
  model.skops
  resolved_config.yaml
  manifest.json
```

CXR packages use the same hierarchy with `model.pt` in place of `model.skops`.

Fusion packages add `structured_preprocessor.skops`. Fusion training receives an explicit
same-seed CXR package ID, verifies its scientific data/task/family/seed contracts, initializes
only the CXR encoder, and records integrity and reproducibility witnesses separately. The
train-fitted structured preprocessor and its exact
ordered feature contract are part of the package. Fusion evaluation reconstructs from that package
and applies its frozen validation thresholds to test.

The manifest records `model_package_schema_version` and a deterministic `model_package_id`. CXR
package identity binds the package-scoped scientific configuration projection, canonical fitted
model/preprocessor state, and the frozen validation-derived threshold state and threshold-selection
contract. The archived complete config and `config_semantic_sha256` remain validated witnesses;
downstream evaluation policy comes from the explicit compatible evaluation config. Serialized
bytes, `config_source_sha256`, bundle-manifest hashes, Git, lock, runtime, paths, and MLflow runs
remain integrity, reproducibility, or operational witnesses outside semantic identity.

Each held-out evaluation publishes aggregate derivatives under
`reports/rsna/evaluations/evaluation-<sha256>/derivatives/`:

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

Publication requires exactly this set after privacy validation. `make rsna-compare` deterministically
regenerates `reports/model_comparison_table.csv` and `.md` from explicit validated evaluation IDs.
The evaluation identity binds the model package, private prediction evidence, held-out scope,
evaluation policy, and frozen thresholds. Aggregate claims and report renderings are re-derived
from that authority and do not create duplicate identity inputs.

`make rsna-summarize` accepts exactly three explicit compatible CXR or fusion evaluation IDs for
seeds 17, 42, and 2026. It publishes deterministic JSON, CSV, and Markdown under
`reports/rsna/seed-summaries/seed-summary-<sha256>/`, preserving every seed-specific metric and
reporting arithmetic means and sample standard deviations for all non-threshold probability,
calibration, operating-point, and confusion-count results. Each member retains all six probability
and calibration metrics plus both complete operating points. The semantic ID binds the exact three
validated evaluation authorities, their common package-level scientific family context, and the
summary policy; validation re-derives the aggregate claims. Thresholds remain seed-specific and are never
averaged; the summary selects no canonical seed and creates no averaged model. Fusion summaries
also validate each same-seed source CXR package and compare the source packages' seed-neutral
scientific family contexts. Validation reconstructs deterministic CSV and Markdown renderings in
an external temporary directory and never writes beneath the immutable summary directory.

`make rsna-localize` accepts three explicit CXR evaluation IDs, validates them as authorization for
the held-out localization lifecycle, and resolves their source CXR model packages. Localization
content identity binds the canonical seed-ordered model package IDs together with the localization,
Grad-CAM target, threshold, and qualitative-selection policies. Evaluation-only policy such as
calibration binning is not localization scientific meaning. Public output contains per-seed and
aggregate pointing-game and activation-energy results re-derived from the per-seed members.
Mechanically selected real-image overlays and their internal traceability manifest are published
only under `private/localization/`.

Test evaluators publish one canonical ordered Parquet table under
`private/predictions/rsna/prediction-<sha256>/`. It binds logical sample-level target, logit, and
probability content to one model package, task, split, and scope. These patient-level tables are
neither public reports nor MLflow artifacts.
`RuntimeConfig.private_output_directory` is the single root authority for this private publication.

The RSNA training and evaluation CLIs default to `sqlite:///mlflow.db` and accept `--tracking-uri`
when an isolated local SQLite database is required. RSNA comparison accepts explicit evaluation IDs,
validates those scientific evaluation authorities, and deterministically regenerates its CSV and
Markdown views without MLflow discovery. MLflow remains operational provenance.

Symile development also accepts a tracking URI. The formal Symile campaign instead owns the
checkout's canonical `mlflow.db` and artifact roots; it exposes no tracking or output-root override.

Metric definitions and experimental protocols are documented in
[`reproducibility.md`](reproducibility.md).
