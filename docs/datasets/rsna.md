# RSNA Pneumonia Detection Challenge

## Source and access

The adapter uses the Stage 2 files from the RSNA Pneumonia Detection Challenge. Access is through
the competition host under the dataset's terms. Source CSV files and DICOM images remain in
user-managed local storage.

The optional acquisition dependency group provides the Kaggle CLI so you can download the
competition data after accepting the rules and configuring authentication outside this repository:

```bash
uv run --group acquisition kaggle competitions download \
  -c rsna-pneumonia-detection-challenge \
  -p data/raw/rsna/archive
```

Extract the archive into `data/raw/rsna/extracted/`.

## Verified local layout

```text
stage_2_train_labels.csv
stage_2_detailed_class_info.csv
stage_2_sample_submission.csv
stage_2_train_images/    26,684 labeled DICOMs
stage_2_test_images/      3,000 unlabeled DICOMs
```

There are 29,684 DICOM files in total. The bundle contains the 26,684 images with training labels.
The 3,000 competition test images remain source-only because targets are unavailable.

## Labels

`stage_2_train_labels.csv` contains `patientId, x, y, width, height, Target`. It has 30,227 rows for
26,684 images. Repeated positive rows represent multiple boxes. The normalized annotation artifact
contains 9,555 boxes for 6,012 positive samples. The remaining 20,672 samples are negative and have
no annotation rows.

`stage_2_detailed_class_info.csv` provides one consistent class per source identifier:

- `Normal`
- `No Lung Opacity / Not Normal`
- `Lung Opacity`

The class is encoded as the auxiliary `rsna_class` task in the label artifact. `Target=1` pairs
with `Lung Opacity`; `Target=0` pairs with either remaining class.

The challenge target is derived from the public competition labeling process. It is not equivalent
to microbiologically confirmed clinical pneumonia.

## Patient split

The labeled cohort receives a deterministic 70/15/15 train, validation, and test assignment,
stratified on the pneumonia target at the patient level with seed 42. The split artifact is part of
the validated bundle. The manifest records the recipe and exact assignment identities. Audit
reports contain aggregate counts only.

## DICOM characteristics

All 26,684 labeled headers were read with pixel decoding disabled. Verified aggregate properties:

- 1024 × 1024 pixels
- `MONOCHROME2`
- one transfer syntax
- AP or PA view
- F or M patient sex
- 18 distinct pixel-spacing pairs

Patient age, sex, projection, pixel spacing, and image dimensions enter the sample artifact.
Photometric interpretation, transfer syntax, bit depth, compression, modality, body part, and UID
consistency are source-quality evidence.

The bundle records each source CSV's SHA-256 and authenticates every labeled DICOM against the
source inventory. Metadata models consume validated DICOM header fields. SOP Instance UID is
required and unique across labeled samples.

Metadata values and missingness, including pixel spacing, can encode demographic, workflow,
acquisition, and equipment associations. The metadata-only baselines quantify those associations
only within the internal patient-disjoint RSNA challenge holdout.

## Known data-quality findings

RSNA age values use bare integers interpreted as years under the dataset contract. Standard DICOM
day, week, month, and year forms are also supported. The bundle preserves five observed values
above 120 years and marks them as implausible. Missing and malformed values remain distinct
source-quality outcomes and produce null sample ages.
