# RadFusion-Clinical

RadFusion-Clinical is a reproducible machine-learning benchmark and experimentation framework for
prediction of report-derived Pneumonia findings from chest radiographs and admission physiology.
It contains an RSNA imaging benchmark and an authenticated Symile-MIMIC multimodal development
track.

> Research and educational prototype. Clinical decision-making lies outside its intended use.

## Implemented capabilities

- Validated joins across RSNA labels, classes, and DICOM images
- DICOM header extraction and aggregate data-quality reporting
- Typed samples, labels, bounding-box annotations, and patient-disjoint splits
- SHA-256 source inventory for every labeled DICOM
- Content-addressed immutable bundles with exact schemas and integrity validation
- Authenticated Symile-MIMIC 1.0.0 source qualification, two-table bundle, aggregate audit, and
  immutable repeated-CV assignments
- Development-only Symile repeated-CV execution for fixed labs, CXR, concat, gated, and
  observedness-ablation families, with immutable OOF evidence and aggregate analysis
- Metadata preprocessing fitted on the training split and fixed Logistic Regression and LightGBM
  baselines
- A TorchXRayVision DenseNet121 CXR baseline with deterministic per-seed training
- Same-byte raw-DICOM authentication and decoding during deterministic cache construction
- Observed bundle-manifest SHA-256 lineage for CXR training and linked evaluation
- Separate immutable model packages, private prediction evidence, and aggregate evaluations
- Explicit RSNA CXR and fusion three-seed summaries with individual, mean, and sample-SD results
- Fixed CXR-metadata concat fusion with same-seed CXR package initialization
- Post-training three-seed Grad-CAM localization against RSNA bounding boxes
- Immutable content-addressed metadata and neural model packages
- Ruff, pytest, pre-commit, and continuous-integration checks

## Setup

The project requires Python 3.13 and [uv](https://docs.astral.sh/uv/). Data-acquisition tooling is
available through the optional `acquisition` dependency group.

Scientific execution requires this Git checkout, its configs, the recorded lock, and authorized
local data artifacts. The wheel provides importable code, not a standalone study reproduction.
Run Make commands from the checkout root, or use `make -C /path/to/checkout <target>`.

```bash
uv sync --locked --group dev
uv run pre-commit install
```

On a paid GPU host, install only the locked runtime dependencies required by the campaign:

```bash
uv sync --locked --no-dev
```

## Data prerequisite

Obtain the RSNA Pneumonia Detection Challenge data under its original access terms and extract it
to `data/raw/rsna/extracted/`. The required filenames and directory layout are documented in
[`data/README.md`](data/README.md).

Symile data commands require the credentialed Symile-MIMIC 1.0.0 release under
`data/raw/symile/extracted/`. Restricted source and generated patient-level bundles remain local.

Symile supervised development is restricted to the frozen official train and validation cohorts.
The implementation can execute six fixed families over the immutable 3-by-5 patient-grouped CV
assignment and publish complete OOF evidence. The official Symile test remains closed to this
development lifecycle.

## Commands

```bash
make rsna-campaign   # run and export the complete authoritative RSNA campaign
make symile-manifest # authenticate Symile-MIMIC and publish its immutable bundle
make symile-audit    # publish the bundle-qualified aggregate Symile audit
make symile-cv       # publish the bundle-bound immutable repeated-CV assignments
make symile-develop CONFIG=configs/symile_cxr_densenet.yaml
make symile-analyze DEVELOPMENT_IDS="<six explicit development IDs>"
make symile-campaign SOURCE_ROOT=path/to/private/symile/source BACKUP_ROOT=path/to/approved/persistent/backup

# Lower-level explicit scientific commands
make rsna-inspect FILE=path/to/image.dcm
make rsna-manifest SOURCE_ROOT=data/raw/rsna/extracted
make rsna-audit BUNDLE_ID=bundle-...
make rsna-train CONFIG=configs/rsna_metadata_logistic.yaml SEED=42
make rsna-train CONFIG=configs/rsna_cxr_densenet.yaml SEED=42
make rsna-evaluate PACKAGE_ID=model-package-... CONFIG=configs/rsna_metadata_logistic.yaml
make rsna-compare EVALUATION_IDS="<evaluation-id> ..."
make rsna-summarize EVALUATION_IDS="<evaluation-id> <evaluation-id> <evaluation-id>"
make rsna-localize EVALUATION_IDS="<evaluation-id> <evaluation-id> <evaluation-id>"
make clean           # remove tool caches and interrupted-publication staging state
make purge-generated # destructive reset; refuses an opened Symile campaign
make check           # lock consistency, Ruff checks, and unit/contract tests
make pre-commit      # run repository hooks against all tracked files
```

`symile-audit` and `symile-cv` resolve `data/manifests/symile/CURRENT` once for interactive use.
Pass `BUNDLE_ID=bundle-...` to select an immutable bundle explicitly. Published CV artifacts bind
the resolved immutable bundle ID. Later scientific configurations pin both immutable identities.

`symile-develop` runs one configured family across all 15 frozen outer folds; repeat and fold are
not user controls. Concat and gated families additionally require
`SOURCE_CXR_DEVELOPMENT_ID=development-...`. `symile-analyze` accepts exactly one explicit complete
development ID for each of the six families and validates family membership independently of CLI
ordering. Neither command exposes or evaluates the official Symile test.

`symile-campaign` is the sole public held-out campaign command. It preserves the exact-six core
development analysis, runs the separate exact-three ECG family internally, fits 14 final packages,
and constructs the compact freeze/test-open firewall. Fourteen raw package-bound predictions
produce six predictor views and one global result. Formal Symile execution has not occurred, and
the official test remains untouched.
If a valid `test-open.json` already exists, the command revalidates and resumes that exact freeze,
performs no development or final retraining, and completes only missing post-open work.
Resume requires the same frozen numerical inference runtime; see the
[runtime requirements](docs/reproducibility.md#numerical-runtime-on-resume).

After the raw dataset is in place, `make rsna-campaign` owns pretrained-weight readiness, bundle and
audit generation, deterministic image caching, all eight
training runs, all eight package-bound evaluations, both seed summaries, localization, comparison,
final validation, and export. It requires no operator-supplied IDs. The campaign log is written to
`reports/rsna/campaigns/<campaign-id>/execution.log`; the portable archive and checksum are written
to `outbox/rsna-results-<campaign-id>.tar.gz` and `.tar.gz.sha256`.

Deterministic cache preparation authenticates and preprocesses image bytes for the complete pinned
bundle without reading task labels or fitting population statistics. Training consumes only train
and validation task rows. Within the canonical campaign, held-out evaluation begins only after all
eight packages are frozen.

Model packages are stored under `models/`, generated reports under `reports/`, MLflow metadata in
`mlflow.db`, and small MLflow training-configuration artifacts under `mlartifacts/`.

Instrumented manifest, audit, training, evaluation, and comparison commands emit operational
records to stderr while preserving machine-readable stdout; see the training guide for capture
examples.

Every experiment is declared by a validated YAML file under `configs/`. See
[`docs/training.md`](docs/training.md) for the training workflow.

CXR and fusion training execute one configured seed per invocation. Cache preparation
materializes each source DICOM's bytes once, authenticates them, and decodes the same in-memory
bytes. Neural consumers use a validated cache whose validation proves its identity, exact
sample-to-partition mapping, and content integrity without reopening raw DICOMs. CXR training
fingerprints the pretrained weight file immediately before and after model construction and
requires exact equality.
`make rsna-evaluate` verifies the selected immutable package and the explicitly supplied compatible
evaluation config before accessing test data. The package owns fitted state and frozen thresholds;
the explicit config owns downstream evaluation policy. Neural evaluation reconstructs the model
without the pretrained-weight cache.

Fusion training requires an explicit same-seed CXR package at execution time. The committed
fusion YAML remains a stable scientific definition; the resulting package records the exact source
CXR package and embeds its train-fitted structured preprocessor.

The RSNA campaign owns the CXR and fusion seeds 17, 42, and 2026; seed is an execution
coordinate rather than YAML content. Evaluations for one modality can be
summarized only by supplying all three evaluation IDs explicitly. The summary retains individual results
and reports their mean and sample standard deviation without selecting a canonical seed or
averaging models.

Neural test evaluation writes aligned sample-level predictions only to the ignored `private/`
workspace. Localization reports under `reports/` contain aggregate metrics and methodology;
real-image Grad-CAM overlays and their traceability manifest remain under `private/`.

## Cleaning generated artifacts

Run `make clean` for development caches and temporary publication state. It preserves completed
bundles, derived CXR caches, reports, model packages, and experiment history. Run
`make purge-generated` to remove generated outputs, including the derived image cache. It refuses
when the canonical Symile `test-open.json` exists, preserving the opened freeze and its evidence.
Both commands preserve raw source datasets under `data/raw/`.
Formal Symile execution requires `BACKUP_ROOT` to point to a separately approved persistent
destination outside the resolved repository root and generic repository cleanup ownership.
The formal command owns the canonical manifest, model, report, private, MLflow, and `outbox/`
locations in the checkout. Preserve expensive scientific evidence separately before any reset.

## Repository layout

```text
src/radfusion/data/        ingestion, splits, audits, schemas, validation, and hashing
src/radfusion/models/      fixed estimator definitions
src/radfusion/training/    reusable training entry points
src/radfusion/evaluation/  metrics and aggregate evaluation plots
configs/                   experiment definitions
tests/                     unit, contract, and local integration tests
docs/                      architecture, data contracts, privacy, and reproducibility
data/                      ignored local inputs and generated artifacts
outbox/                    portable campaign archives and checksums
scripts/                   small inspection utilities
```

## Privacy boundary

Keep DICOMs, patient-level bundle artifacts, credentials, and real patient examples outside
version control. See [`docs/privacy.md`](docs/privacy.md).

## Limitations

- Implemented scope covers the RSNA Stage 2 metadata, CXR, fusion, and localization campaign plus
  the Symile-MIMIC data foundation, core repeated-CV workflow, ECG extension, final fitting,
  pre-test controls, and held-out evaluation infrastructure. No formal Symile development or
  held-out results are reported here.
- Benchmark targets are radiology-derived findings. Confirmed clinical diagnosis lies outside the
  endpoint definition.

See [`docs/architecture.md`](docs/architecture.md) for system structure,
[`docs/data_contract.md`](docs/data_contract.md) for the shared bundle and RSNA artifact contract, and
[`docs/reproducibility.md`](docs/reproducibility.md) for reconstruction details. RSNA-specific
facts are documented in [`docs/datasets/rsna.md`](docs/datasets/rsna.md); Symile-specific facts are
documented in [`docs/datasets/symile.md`](docs/datasets/symile.md).
