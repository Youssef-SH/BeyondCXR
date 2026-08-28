# Symile-MIMIC source guide

Symile-MIMIC is BeyondCXR's primary multimodal scientific dataset. The complete dataset role,
cohort, access, timing, missingness, transfer, and limitation statement is in
[`data_statement.md`](../data_statement.md); this page records source-facing operational details.

BeyondCXR uses the credentialed Symile-MIMIC 1.0.0 release as the synchronized multimodal source
for strict-pneumonia development. Access is governed by the PhysioNet credentialed-data license,
data-use agreement, and required training. Source data and patient-level derivatives remain
outside version control.

The source contract authenticates `SHA256SUMS.txt`, the admission index, official train,
validation, and test retrieval files, laboratory references, and split-aligned CXR, ECG,
laboratory, missingness, and identifier tensors. Classification membership consists of the 10,000
official train admissions, 750 official validation admissions, and 464 positive-query test
admissions. The remaining 408 admissions in the 11,622-admission source index are recorded by the
aggregate audit and are not added to classification membership.

The immutable bundle contains `samples.parquet`, `labs.parquet`, and `manifest.json`. Large source
tensors remain external and are authenticated against the pinned source-release inventory. The
strict endpoint includes explicit `Pneumonia = 1` and `Pneumonia = 0` rows and excludes uncertain
or missing values.

Development combines only official train and validation strict-pneumonia rows. It consumes
the pinned patient-grouped three-repeat by five-fold CV assignment and has no official-test data
accessor. Every outer-fold model package publishes separate private OOF prediction evidence; the
family development result and six-family analysis contain aggregate claims only.

Run the data operators with explicit source or bundle coordinates:

```bash
make symile-manifest SOURCE_ROOT=data/raw/symile/extracted
make symile-audit BUNDLE_ID=bundle-...
make symile-cv BUNDLE_ID=bundle-...
```

`CURRENT` is available only for intentional interactive discovery when `BUNDLE_ID` is omitted.
Formal development and campaign execution instead require the exact config-pinned bundle and CV
directories, including the original manifest bytes. The formal campaign validates both authorities
and all referenced external source assets before development; it does not regenerate either
authority.
