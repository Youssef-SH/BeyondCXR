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
      bundles/
        bundle-<sha256>/
          manifest.json
          samples.parquet
          labels.parquet
          annotations.parquet
          splits.parquet
          source_inventory.parquet
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
make rsna-manifest SOURCE_ROOT=data/raw/rsna/extracted
```

The CXR cache is a disposable memory-mapped deterministic derivative used by neural workflows.

## Symile-MIMIC source and artifact layout

The Symile-MIMIC source and generated artifacts use this layout:

```text
data/
  raw/symile/extracted/             # official restricted Symile-MIMIC 1.0.0 release
  manifests/symile/
    CURRENT
    bundles/bundle-<sha256>/
      manifest.json
      samples.parquet
      labs.parquet
    cv/cv-assignment-<sha256>/
      manifest.json
      assignments.parquet
```

Run `make symile-manifest`, `make symile-audit`, and `make symile-cv` in that order. `CURRENT` is
an interactive bundle selector. Durable consumers pin immutable bundle and CV assignment
identities. Official CXR, ECG, lab-percentile, missingness, and identifier NPY arrays remain
external restricted source assets authenticated through the bundle-bound release checksum manifest.

Supervised development reads only strict-pneumonia rows from official train and validation and
consumes the immutable CV assignment without regenerating it. Each fold model package has a
separate patient-level OOF prediction object under the ignored `private/` tree. The official test
rows are not available through the development data layer.

Keep raw images, source CSVs, generated bundle and cache artifacts, and credentials outside version
control. These files contain patient-level information even when public identifiers are
deidentified.
