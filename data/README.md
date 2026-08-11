# Local data workspace

This directory holds local source data and generated patient-level artifacts. Git tracks this file
and directory placeholders; local data content is ignored.

## RSNA source and artifact layout

The RSNA source and generated artifacts use this layout:

```text
data/
  raw/
    rsna/
      archive/
        rsna-pneumonia-detection-challenge.zip
      extracted/
        stage_2_train_labels.csv
        stage_2_detailed_class_info.csv
        stage_2_sample_submission.csv
        stage_2_train_images/
        stage_2_test_images/
  interim/
  processed/
  manifests/
    rsna/
      CURRENT
      builds/
        build-<sha256>/
          rsna_samples.parquet
          rsna_labels.parquet
          rsna_annotations.parquet
          rsna_splits.parquet
          rsna_source_inventory.parquet
          rsna_manifest_metadata.json
  cache/
    rsna/
      cache-<sha256>/
        images.npy
        index.parquet
        metadata.json
```

Bundle construction requires `stage_2_train_labels.csv`,
`stage_2_detailed_class_info.csv`, and `stage_2_train_images/`. The submission CSV and unlabeled
test images are shown for completeness and remain source-only.

After accepting the competition terms and configuring the Kaggle CLI outside the repository, the
archive can be downloaded with the command in
[`docs/datasets/rsna.md`](../docs/datasets/rsna.md). Extract it into the layout above and run:

```bash
make rsna-manifest
```

The CXR cache is a disposable memory-mapped deterministic derivative used by neural workflows.

## Symile-MIMIC source and artifact layout

The Symile-MIMIC source and generated artifacts use this layout:

```text
data/
  raw/symile/extracted/             # official restricted Symile-MIMIC 1.0.0 release
  manifests/symile/
    CURRENT
    builds/build-<sha256>/
      symile_samples.parquet
      symile_labs.parquet
      symile_manifest_metadata.json
    cv_assignments/cv-assignment-<sha256>/
      symile_cv_assignments.parquet
      symile_cv_manifest.json
```

Run `make symile-manifest`, `make symile-audit`, and `make symile-cv` in that order. `CURRENT` is
an interactive bundle selector. Durable consumers pin immutable bundle and CV assignment
identities. Official CXR, ECG, lab-percentile, missingness, and identifier NPY arrays remain
external restricted source assets authenticated through the bundle-bound release checksum manifest.

Supervised development reads only strict-pneumonia rows from official train and validation and
consumes the immutable CV assignment without regenerating it. Its patient-level OOF tables are
stored in ignored model packages rather than in `data/manifests/`. The official test rows are not
available through the development data layer.

Keep raw images, source CSVs, generated bundle and cache artifacts, and credentials outside version
control. These files contain patient-level information even when public identifiers are
deidentified.
