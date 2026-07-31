# RadFusion-Clinical

RadFusion-Clinical is a reproducible machine-learning benchmark and experimentation framework for
radiographic pneumonia prediction, centered on the RSNA Pneumonia Detection Challenge. It provides
deterministic data preparation, patient-disjoint evaluation, and reproducible metadata and image
baselines.

> This is a research and educational prototype. It is not a medical device and must not be used for clinical decision-making.

## Implemented capabilities

- Validated joins across RSNA labels, classes, and DICOM images
- DICOM header extraction and aggregate data-quality reporting
- Typed samples, labels, bounding-box annotations, and patient-disjoint splits
- SHA-256 source inventory for every labeled DICOM
- Content-addressed immutable bundles with exact schemas and integrity validation
- Metadata preprocessing fitted on the training split and fixed Logistic Regression and LightGBM
  baselines
- A TorchXRayVision DenseNet121 image baseline with deterministic single-seed training
- Partition-scoped authentication of external DICOM bytes before image access
- Observed bundle-manifest SHA-256 lineage for image training and linked evaluation
- Separate validation and explicit test-evaluation runs with MLflow lineage
- Explicit RSNA image three-seed summaries with individual, mean, and sample-SD results
- Immutable run-qualified metadata and neural model packages
- Ruff, pytest, pre-commit, and continuous-integration checks

## Setup

The project requires Python 3.13 and [uv](https://docs.astral.sh/uv/). Data-acquisition tooling is
available through the optional `acquisition` dependency group.

```bash
uv sync --locked
uv run pre-commit install
```

## Data prerequisite

Obtain the RSNA Pneumonia Detection Challenge data under its original access terms and extract it
to `data/raw/rsna/extracted/`. The required filenames and directory layout are documented in
[`data/README.md`](data/README.md).

## Commands

```bash
make rsna-manifest   # publish an RSNA bundle
make rsna-audit      # generate reports under reports/rsna/audit/<bundle-id>
make train CONFIG=configs/metadata_logistic.yaml
make train CONFIG=configs/metadata_lightgbm.yaml
make train CONFIG=configs/image_densenet_seed42.yaml
make evaluate RUN_ID=<training-run-id>
make summarize-seeds TEST_RUN_IDS="<test17> <test42> <test2026>"
make compare         # regenerate CSV and Markdown comparison views from MLflow
make clean           # remove caches and interrupted-publication staging state
make purge-generated # deliberately remove all reproducible generated outputs
make check           # lock consistency, Ruff checks, and unit/contract tests
make pre-commit      # run repository hooks against all tracked files
make inspect FILE=path/to/image.dcm
```

`<training-run-id>` denotes the run ID printed by `make train`. Model packages are stored under
`models/`, generated reports under `reports/`, MLflow metadata in `mlflow.db`, and small MLflow
training-configuration artifacts under `mlartifacts/`.

Instrumented manifest, audit, training, evaluation, and comparison commands emit operational
records to stderr while preserving machine-readable stdout; see the training guide for capture
examples.

Every executable experiment is declared by a validated YAML file under `configs/`. See
[`docs/training.md`](docs/training.md) for the training workflow.

Image training executes one configured seed per invocation. It reads and authenticates only train
and validation DICOMs, fingerprints the pretrained weight file immediately before and after model
construction, and requires exact equality. It packages exact bundle and run lineage.
`make evaluate` verifies the selected immutable package before accessing test data and reconstructs
the model without the pretrained-weight cache.

The image configs lock seeds 17, 42, and 2026. Their linked test runs can be summarized only by
supplying all three run IDs explicitly. The summary retains individual results and reports their
mean and sample standard deviation without selecting a canonical seed or averaging models.

## Cleaning generated artifacts

Run `make clean` for disposable caches and temporary publication state. It preserves completed
bundles, reports, model packages, and experiment history. Run `make purge-generated` to remove
those reproducible outputs deliberately. Both commands preserve raw source datasets under
`data/raw/`.

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
scripts/                   small inspection utilities
```

## Privacy boundary

Keep DICOMs, patient-level bundle artifacts, credentials, and real patient examples outside
version control. See [`docs/privacy.md`](docs/privacy.md).

## Limitations

- Implemented scope covers the labeled RSNA Stage 2 training set and metadata and image baselines.
- Image experiment code is complete; scientific image results require independent GPU runs and are
  not reported here.
- The labels are derived from public radiology-labeling pipelines and are not equivalent to
  confirmed clinical diagnosis.

See [`docs/architecture.md`](docs/architecture.md) for system structure,
[`docs/data_contract.md`](docs/data_contract.md) for artifact contracts, and
[`docs/reproducibility.md`](docs/reproducibility.md) for reconstruction details. RSNA-specific
facts are documented in [`docs/datasets/rsna.md`](docs/datasets/rsna.md).
