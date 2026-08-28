# Public aggregate results

This directory contains public aggregate results for the primary Symile study. Supporting RSNA CXR
results remain reproducible through the RSNA campaign and aggregate-report workflow; they are
separate evidence and not external validation of the Symile predictor.

`make rsna-campaign` produces and validates the existing RSNA aggregate authorities under
`reports/rsna/evaluations/`, `reports/rsna/seed-summaries/`, and `reports/rsna/localization/`, with
the content-addressed deterministic comparison under `reports/rsna/comparisons/`.

Before formal execution no Symile result files exist here. After execution, `make results`
restores and validates one explicit preservation export and writes the result binding,
five aggregate CSVs, five matching Markdown tables, and three scientific SVGs under `symile/`.
Synthetic outputs remain temporary test data.

No file here may contain patient-level identifiers, prediction rows, private paths, restricted
source material, or manually copied metrics.
