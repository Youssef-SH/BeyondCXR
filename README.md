# BeyondCXR

**Measuring the incremental value of admission physiology beyond chest radiography.**

> Research use only. BeyondCXR is not intended for clinical decision-making, is not a medical
> device, does not replace physician judgment, and predicts a radiology-derived Pneumonia finding
> rather than confirmed infectious pneumonia.

BeyondCXR asks whether admission physiology provides predictive information beyond CXR for a
strict report-derived Pneumonia endpoint.

## Study design

| Dataset | Scientific role | What it contributes |
| --- | --- | --- |
| Symile-MIMIC | Primary multimodal incremental-value study | CXR, 50 laboratory measurements, ECG, repeated grouped cross-validation, and a patient-disjoint held-out test |
| RSNA Stage 2 | Supporting CXR qualification | DICOM ingestion, image modeling, metadata fusion, localization, and supporting imaging evaluation |

The datasets have different endpoints and separate evidence chains. RSNA is not external
validation of the Symile predictor.

## Prespecified comparisons

- Primary: CXR + labs missingness-aware gated fusion versus CXR-only.
- Architecture control: CXR + labs gated fusion versus simple concatenation.
- Secondary modality: CXR + labs + ECG gated fusion versus CXR + labs gated fusion.

## Evaluation

Development uses three five-fold patient-grouped repeats. Neural ensembles use seeds 17, 42, and
2026, average aligned raw logits, and apply sigmoid once. AUROC is primary; Average Precision and
Brier score are key secondary measures. Held-out effects use paired subject-level bootstrap
intervals, and the primary operating points are derived from development evidence.

Held-out test access occurs only after all required model packages validate, the pre-test
freeze validates, and its atomic same-freeze opening record exists. The held-out test is internal
testing within the patient-disjoint Symile source population.

## Results

<!-- BEYONDCXR_RESULTS_START -->
Awaiting formal Symile execution. The release command replaces this bounded region only after the
preserved campaign, result binding, and complete scientific evidence chain validate.
<!-- BEYONDCXR_RESULTS_END -->

## Evidence and reproducibility

At the model/evaluation layer, claims follow:

```text
model package → prediction evidence → scientific result
```

Public results are aggregate derivatives of that evidence rather than scientific authorities.
MLflow records operational provenance only. See
[reproducibility](https://github.com/Youssef-SH/BeyondCXR/blob/main/docs/reproducibility.md),
[architecture](https://github.com/Youssef-SH/BeyondCXR/blob/main/docs/architecture.md), and the
[data contract](https://github.com/Youssef-SH/BeyondCXR/blob/main/docs/data_contract.md).

## Controlled research serving

The API loads one serving authority for the ordered seed-17/42/2026 primary gated ensemble. It
accepts a frontal JPEG, AP/PA validation metadata, and exactly 50 laboratory keys. `/predict`
returns one raw probability and no diagnosis or thresholded decision. See
[controlled serving](https://github.com/Youssef-SH/BeyondCXR/blob/main/docs/serving.md) and the
[synthetic request](https://github.com/Youssef-SH/BeyondCXR/tree/main/examples/symile_serving/).

## Getting started

BeyondCXR requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --group dev --extra serving
make check
```

Scientific execution requires authorized local data. Procedures and commands are documented in
[training](https://github.com/Youssef-SH/BeyondCXR/blob/main/docs/training.md) and
[reproducibility](https://github.com/Youssef-SH/BeyondCXR/blob/main/docs/reproducibility.md).

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/beyondcxr/data/` | Source qualification, bundles, splits, and preprocessing |
| `src/beyondcxr/models/` | Estimator and fusion architectures |
| `src/beyondcxr/training/` | Training, evaluation, scientific objects, and campaigns |
| `src/beyondcxr/serving/` | Controlled research inference |
| `src/beyondcxr/release/` | Aggregate result rendering and release validation |
| `configs/` | Experiment configurations |
| `results/` | Public aggregate result derivatives |

Documentation: [data statement](https://github.com/Youssef-SH/BeyondCXR/blob/main/docs/data_statement.md),
[model card](https://github.com/Youssef-SH/BeyondCXR/blob/main/docs/model_card.md),
[privacy](https://github.com/Youssef-SH/BeyondCXR/blob/main/docs/privacy.md), and
[serving](https://github.com/Youssef-SH/BeyondCXR/blob/main/docs/serving.md).

## Data, citation, and license

Patient-level data, generated bundles, predictions, trained packages, and operational state remain
outside Git. Dataset access and citations are documented in the
[data statement](https://github.com/Youssef-SH/BeyondCXR/blob/main/docs/data_statement.md); project citation metadata is in
[`CITATION.cff`](CITATION.cff), and third-party attribution is in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Repository code is licensed under [Apache-2.0](LICENSE). The code license does not grant rights to
restricted datasets, pretrained weights, or trained artifacts.
