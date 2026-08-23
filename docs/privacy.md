# Privacy and data handling

The version-controlled repository contains code, documentation, configuration, and synthetic
fixtures. Raw and generated patient-level data remains in ignored local storage under the terms of
its source dataset.

## Version-control boundary

Keep these materials outside version control:

- DICOM images and other medical images;
- source tables containing patient-level rows;
- generated bundles and source inventories;
- derived image caches;
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
under `reports/`. Comparison views are derived from explicit validated evaluation identities;
MLflow is an operational ledger, not comparison authority.

Public examples and test fixtures use synthetic data. External transfer of raw or derived data
requires the dataset terms and project data-handling policy to permit the destination and use.

## Private analysis workspace

The ignored `private/` directory stores sample-level prediction evidence and real-image
localization overlays. Prediction evidence contains sample keys, targets, logits, probabilities,
and exact package/scientific lineage. Localization manifests contain the internal sample
identity needed to trace mechanically selected overlays.

Only aggregate localization metrics, counts, and methodological text are published under
`reports/`. Real images, overlays, sample identities, patient identities, row-level predictions,
and private artifact paths are excluded from public reports and MLflow artifacts. Private outputs
remain subject to the source dataset's access and transfer terms.

Campaign archives under the ignored `outbox/` directory contain private prediction and localization
outputs as well as aggregate outputs and provenance. They remain controlled research artifacts and
may be transferred only to destinations permitted by the source dataset terms and project policy.

Symile backups at the required operator-supplied `BACKUP_ROOT` have the same restricted status.
The destination must be separately approved, persistent, and outside the resolved repository root.
Local MLflow databases and `mlartifacts/` are private operational state, not public exports.
Handled scientific-command failures report exception types without copying exception text into
stderr. The interactive `rsna-inspect` utility deliberately displays source metadata and pixels:
use it only in an authorized private session and do not publish its output or screenshots.

The research serving process is stateless with respect to patients. It keeps no upload archive,
request or response archive, patient database, or prediction cache. BeyondCXR retains no submitted
image, laboratory input, patient identifier, missingness state, or prediction after request
processing, and operational handling does not log patient-level inputs or outputs. The project
command disables Uvicorn access logging; deployments must apply the same restriction to external
proxies and platforms. Public Docker images contain only distributable code and the locked runtime.
Restricted serving authorities and trained packages are mounted read-only at runtime unless their
distribution is separately authorized. Public request examples are explicitly synthetic.
