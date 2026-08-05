# Local data workspace

This directory holds local source data and generated patient-level artifacts. Git tracks this file
and directory placeholders; local data content is ignored.

Expected RSNA layout:

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
Keep raw images, source CSVs, generated bundle and cache artifacts, and credentials outside version
control. These files contain patient-level information even when public identifiers are
deidentified.
