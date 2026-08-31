# RSNA Stage 2 source guide

RSNA Stage 2 provides BeyondCXR's supporting CXR qualification. It establishes DICOM ingestion,
image modeling, metadata fusion, and localization; it is not external validation of the primary
Symile predictor. Dataset roles and transfer constraints are in the [data statement](../data_statement.md).

## Source readiness

Accept the competition rules and configure standard Kaggle CLI authentication. Then run:

```bash
make rsna
```

The workflow first validates the configured source. A valid source is reused without network
access. If the configured source is absent, it downloads the official competition archive into a
staging directory, rejects unsafe paths, links, duplicate or unexpected members, validates the
release layout, and installs it atomically. An existing invalid or conflicting source is never
overwritten.

The canonical source root is `data/raw/rsna/extracted`. A one-time alternate root can be recorded
with `make setup RSNA_SOURCE_ROOT=/approved/rsna/source`.

## Verified release layout

```text
stage_2_train_labels.csv
stage_2_detailed_class_info.csv
stage_2_sample_submission.csv
stage_2_train_images/    26,684 labeled DICOMs
stage_2_test_images/      3,000 unlabeled DICOMs
```

The bundle contains the 26,684 labeled images. Competition test images remain source-only because
targets are unavailable. Exact upstream files and DICOMs are authenticated; the scientific bundle
identity is derived from logical cohort, label, annotation, policy, and split content rather than
the source root or physical Parquet encoding.

## Cohort, labels, and split

`stage_2_train_labels.csv` contains `patientId, x, y, width, height, Target`. Repeated positive rows
represent multiple boxes. `stage_2_detailed_class_info.csv` supplies `Normal`,
`No Lung Opacity / Not Normal`, or `Lung Opacity`; target/class compatibility is validated.

The label CSV has 30,227 rows for 26,684 images. The normalized annotation evidence contains 9,555
boxes for 6,012 positive samples; the remaining 20,672 samples are negative and have no annotation
rows. The class is retained as the auxiliary `rsna_class` task. `Target=1` pairs with
`Lung Opacity`; `Target=0` pairs with either remaining class.

The labeled cohort receives the prespecified patient-disjoint 70/15/15 train, validation, and test
assignment with seed 42. The exact assignment is intrinsic to the RSNA bundle. The challenge target
comes from the public competition labeling process and is not a confirmed clinical diagnosis.

## Source qualification

All labeled DICOM headers are qualified without pixel decoding. The reviewed release contains
1024-by-1024 `MONOCHROME2` images, one transfer syntax, AP or PA views, F or M patient sex values,
and 18 observed pixel-spacing pairs. Patient age, sex, projection, pixel spacing, and dimensions
enter the sample evidence. SOP Instance UID is required and unique. Photometric interpretation,
transfer syntax, bit depth, compression, modality, body part, and UID consistency remain
source-quality evidence rather than model claims.

Bare integer ages are interpreted as years under the dataset contract; standard DICOM day, week,
month, and year forms are also supported. The source contains five observed ages above 120 years,
which are retained and marked implausible. Missing and malformed values remain distinct
source-quality outcomes and yield null sample ages.

Metadata and missingness can encode demographic, workflow, acquisition, and equipment
associations. Metadata-only baselines quantify those associations only within the internal
patient-disjoint RSNA challenge holdout.

The completed workflow builds or reuses the exact bundle, produces aggregate audits, materializes
the reviewed pretrained initialization, performs formal execution, freezes packages before
held-out access, evaluates and localizes, and preserves the recursively validated campaign to the
configured external backup.
