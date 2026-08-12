# RadFusion-Clinical

RadFusion-Clinical is a reproducible machine-learning benchmark and experimentation framework for
prediction of report-derived Pneumonia findings from chest radiographs and admission physiology.
It contains the completed RSNA imaging benchmark and an authenticated Symile-MIMIC multimodal data
layer.

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
- A TorchXRayVision DenseNet121 image baseline with deterministic per-seed training
- Same-byte raw-DICOM authentication and decoding during deterministic cache construction
- Observed bundle-manifest SHA-256 lineage for image training and linked evaluation
- Separate validation and explicit test-evaluation runs with MLflow lineage
- Explicit RSNA image and fusion three-seed summaries with individual, mean, and sample-SD results
- Fixed image-metadata concat fusion with same-seed CXR package initialization
- Post-training three-seed Grad-CAM localization against RSNA bounding boxes
- Immutable run-qualified metadata and neural model packages
- Ruff, pytest, pre-commit, and continuous-integration checks

## Setup

The project requires Python 3.13 and [uv](https://docs.astral.sh/uv/). Data-acquisition tooling is
available through the optional `acquisition` dependency group.

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
make rsna-gpu        # run and export the complete authoritative RSNA campaign
make symile-manifest # authenticate Symile-MIMIC and publish its immutable bundle
make symile-audit    # publish the bundle-qualified aggregate Symile audit
make symile-cv       # publish the bundle-bound immutable repeated-CV assignments
make symile-develop CONFIG=configs/symile_cxr_densenet.yaml
make symile-analyze DEVELOPMENT_IDS="<six explicit development IDs>"

# Lower-level inspection and debugging commands
make rsna-manifest   # publish an RSNA bundle
make rsna-audit      # generate reports under reports/rsna/audit/<bundle-id>
make train CONFIG=configs/rsna_metadata_logistic.yaml SEED=42
make train CONFIG=configs/rsna_cxr_densenet.yaml SEED=42
make evaluate RUN_ID=<training-run-id>
make compare         # regenerate CSV and Markdown comparison views from MLflow
make clean           # remove tool caches and interrupted-publication staging state
make purge-generated # deliberately remove all reproducible generated outputs
make check           # lock consistency, Ruff checks, and unit/contract tests
make pre-commit      # run repository hooks against all tracked files
make inspect FILE=path/to/image.dcm
```

`symile-audit` and `symile-cv` resolve `data/manifests/symile/CURRENT` once for interactive use.
Pass `BUNDLE_ID=bundle-...` to select an immutable bundle explicitly. Published CV artifacts bind
the resolved immutable bundle ID. Later scientific configurations pin both immutable identities.

`symile-develop` runs one configured family across all 15 frozen outer folds; repeat and fold are
not user controls. Concat and gated families additionally require
`SOURCE_CXR_DEVELOPMENT_ID=development-...`. `symile-analyze` accepts exactly one explicit complete
development ID for each of the six families and validates family membership independently of CLI
ordering. Neither command exposes or evaluates the official Symile test.

After the raw dataset is in place, `make rsna-gpu` owns pretrained-weight readiness, bundle and
audit generation, deterministic image caching, all eight
training runs, all eight linked evaluations, both seed summaries, localization, comparison, final
validation, and export. It requires no operator-supplied run IDs. The campaign log is written to
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

Image and fusion training execute one configured seed per invocation. Cache preparation
materializes each source DICOM's bytes once, authenticates them, and decodes the same in-memory
bytes. Neural consumers use a validated cache whose validation proves its identity, exact
sample-to-partition mapping, and content integrity without reopening raw DICOMs. Image training
fingerprints the pretrained weight file immediately before and after model construction and
requires exact equality.
`make evaluate` verifies the selected immutable package before accessing test data and reconstructs
the model without the pretrained-weight cache.

Fusion training requires an explicit same-seed image training run at execution time. The committed
fusion YAML remains a stable scientific definition; the resulting package records the exact source
CXR package and embeds its train-fitted structured preprocessor.

The RSNA campaign owns the image and fusion seeds 17, 42, and 2026; seed is an execution
coordinate rather than YAML content. Linked test runs for one modality can be
summarized only by supplying all three run IDs explicitly. The summary retains individual results
and reports their mean and sample standard deviation without selecting a canonical seed or
averaging models.

Neural test evaluation writes aligned sample-level predictions only to the ignored `private/`
workspace. Localization reports under `reports/` contain aggregate metrics and methodology;
real-image Grad-CAM overlays and their traceability manifest remain under `private/`.

## Cleaning generated artifacts

Run `make clean` for development caches and temporary publication state. It preserves completed
bundles, derived CXR caches, reports, model packages, and experiment history. Run
`make purge-generated` to remove reproducible outputs, including the derived image cache. Both
commands preserve raw source datasets under `data/raw/`.

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

- Implemented scope covers the completed RSNA Stage 2 metadata, image, fusion, and localization
  campaign plus the authenticated Symile-MIMIC data layer and development-only repeated-CV
  lifecycle. No formal Symile development results or held-out-test results are reported here.
- Benchmark targets are radiology-derived findings. Confirmed clinical diagnosis lies outside the
  endpoint definition.

See [`docs/architecture.md`](docs/architecture.md) for system structure,
[`docs/data_contract.md`](docs/data_contract.md) for the RSNA artifact contract, and
[`docs/reproducibility.md`](docs/reproducibility.md) for reconstruction details. RSNA-specific
facts are documented in [`docs/datasets/rsna.md`](docs/datasets/rsna.md).
