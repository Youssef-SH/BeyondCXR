# Public aggregate results

This directory contains public aggregate results for the primary Symile study. Supporting RSNA CXR
results remain reproducible through the RSNA campaign and aggregate-report workflow; they are
separate evidence and not external validation of the Symile predictor.

`make rsna` produces and validates the RSNA evaluation and localization authorities under
`reports/rsna/evaluations/` and `reports/rsna/localization/`. Deterministic seed summaries and
comparisons are rendered under `reports/rsna/seed-summaries/` and `reports/rsna/comparisons/`.

Before formal execution no Symile result files exist here. After `make symile-test`, `make release`
resolves and validates the sole completed Symile preservation archive and writes the result binding,
five aggregate CSVs, five matching Markdown tables, and three scientific SVGs under `symile/`.
Synthetic outputs remain temporary test data.

No file here may contain patient-level identifiers, prediction rows, private paths, restricted
source material, or manually copied metrics.
