# BeyondCXR

BeyondCXR measures the incremental predictive value of admission physiology beyond chest
radiography. Symile-MIMIC is the primary multimodal study; RSNA Stage 2 provides supporting CXR,
metadata, and localization qualification. The datasets have different endpoints, so RSNA is not
external validation of the Symile predictor.

> Research use only. BeyondCXR is not intended for clinical decision-making, is not a medical
> device, does not replace physician judgment, and predicts a radiology-derived Pneumonia finding
> rather than confirmed infectious pneumonia.

## Workflow

BeyondCXR requires Python 3.13, [uv](https://docs.astral.sh/uv/), and an approved persistent backup
location. From a fresh checkout:

```bash
make setup BACKUP_ROOT=/approved/persistent/backup
make rsna

# Place the authorized private Symile release at data/raw/symile/extracted.
make symile

# Review the immutable pre-test freeze before crossing the held-out boundary.
make symile-test
make release
```

The first release run prepares public results for review. Commit the results and final release
metadata following [release preparation](docs/reproducibility.md#preservation-and-release), then
rerun `make release` for acceptance.

`make rsna` reuses a valid local RSNA release and acquires an absent source through the Kaggle CLI,
using the operator's standard Kaggle credentials. Symile is restricted and is never downloaded by
the project. One-time noncanonical source roots can be recorded during setup with
`RSNA_SOURCE_ROOT=...` and `SYMILE_SOURCE_ROOT=...`.

Optional actions are:

```bash
make serve
make check
```

`make serve` starts the sole validated installed serving authority on CPU at `127.0.0.1:8000`.
`make check` runs the dependency-lock, lint, formatting, and default test gates. See
[reproducibility](docs/reproducibility.md#developer-checks) for integration checks.

`make symile` stops at the pre-test freeze. Only `make symile-test` opens the official held-out
phase. See the [model card](docs/model_card.md) for the primary predictor's development,
evaluation, and limitations.

<!-- BEYONDCXR_RESULTS_START -->
Awaiting formal Symile execution. The release workflow replaces this bounded region only after the
preserved campaign and complete scientific evidence chain validate.
<!-- BEYONDCXR_RESULTS_END -->

## Documentation

- [Architecture](docs/architecture.md)
- [Data contract](docs/data_contract.md)
- [Training and lifecycle](docs/training.md)
- [Reproducibility](docs/reproducibility.md)
- [RSNA source guide](docs/datasets/rsna.md)
- [Symile source guide](docs/datasets/symile.md)
- [Controlled serving](docs/serving.md)
- [Data statement](docs/data_statement.md)
- [Privacy](docs/privacy.md)

Patient-level data, generated bundles, predictions, fitted packages, credentials, and operational
state remain outside Git. Repository code is licensed under [Apache-2.0](LICENSE); that license
does not grant rights to restricted datasets, pretrained weights, or trained artifacts.
