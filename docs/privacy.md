# Privacy and data handling

The version-controlled repository contains code, documentation, configuration, and synthetic
fixtures. Raw and generated patient-level data remains in ignored local storage under the terms of
its source dataset.

## Version-control boundary

Keep these materials outside version control:

- DICOM images and other medical images;
- source tables containing patient-level rows;
- generated bundles and source inventories;
- model and experiment artifacts;
- clinical-note or laboratory rows;
- credentials, access tokens, and private keys;
- screenshots containing patient-level values.

Review staged files against this policy before each commit. `.gitignore` provides baseline
filtering.

## Reports

Dataset audits and model evaluation reports contain aggregate counts, distributions, quality
findings, and metrics. Before publication, checks reject source patient IDs, sample IDs, image
names and paths, UUID-shaped identifiers, and DICOM UID-shaped values. These reports are published
under `reports/`. Comparison views are generated from aggregate MLflow metrics and lineage.

Public examples and test fixtures use synthetic data. External transfer of raw or derived data
requires the dataset terms and project data-handling policy to permit the destination and use.

## Private analysis workspace

The ignored `private/` directory stores patient-level neural prediction tables and real-image
localization overlays. Prediction tables contain sample and private patient keys, targets, logits,
probabilities, and exact run/package lineage. Localization manifests contain the internal sample
identity needed to trace mechanically selected overlays.

Only aggregate localization metrics, counts, and methodological text are published under
`reports/`. Real images, overlays, sample identities, patient identities, row-level predictions,
and private artifact paths are excluded from public reports and MLflow artifacts. Private outputs
remain subject to the source dataset's access and transfer terms.
