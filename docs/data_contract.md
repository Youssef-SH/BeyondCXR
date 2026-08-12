# RSNA data contract

The common bundle manifest schema version `1` defines the current development contract. Bundle IDs
and declared hashes identify exact content.

An RSNA bundle contains five Parquet artifacts and one JSON manifest. Column order, Arrow types,
nullability, and deterministic row order are contractual.

## Samples

File: `samples.parquet`. One row per labeled image, sorted by `sample_id`.

| Field | Arrow type | Nullable | Definition |
| --- | --- | --- | --- |
| `sample_id` | `string` | no | Unique key, `rsna:<image_id>` |
| `patient_id` | `string` | no | Patient grouping key |
| `image_id` | `string` | no | Source image identifier and filename stem |
| `image_path` | `string` | no | Dataset-root-relative POSIX path |
| `image_rows` | `int32` | no | Positive DICOM row count |
| `image_columns` | `int32` | no | Positive DICOM column count |
| `age_years` | `float64` | yes | Parsed finite DICOM age in years |
| `age_is_implausible` | `bool` | no | Whether the parsed age is outside 0–120 years |
| `sex` | `string` | yes | `F` or `M` |
| `view_position` | `string` | yes | `AP` or `PA` |
| `pixel_spacing_row_mm` | `float64` | yes | Positive DICOM row spacing |
| `pixel_spacing_col_mm` | `float64` | yes | Positive DICOM column spacing |

Missing or malformed ages are null and are not marked implausible. Parsed out-of-range ages remain
stored and are marked implausible. Pixel spacing is either a positive finite pair or two nulls.

`image_path` has the form `stage_2_train_images/<image_id>.dcm`. Construction verifies that it
resolves beneath the raw dataset root and agrees with the discovered source file.

## Labels

File: `labels.parquet`. One row per sample and task, sorted by
`(sample_id, task_id)`.

| Field | Arrow type | Nullable | Definition |
| --- | --- | --- | --- |
| `sample_id` | `string` | no | Foreign key to samples |
| `task_id` | `string` | no | `pneumonia` or `rsna_class` |
| `label_value` | `int8` | no | Task-specific value |

Every sample has both tasks. `pneumonia` uses 0 for negative and 1 for positive. `rsna_class` uses
0 for `Normal`, 1 for `No Lung Opacity / Not Normal`, and 2 for `Lung Opacity`. A positive binary
target pairs with class 2; a negative target pairs with class 0 or 1. Task sources, status
semantics, allowed values, exclusions, and label-policy versions are declared once in the
manifest.

## Annotations

File: `annotations.parquet`. One row per positive bounding box, sorted by
`annotation_id`.

| Field | Arrow type | Nullable | Definition |
| --- | --- | --- | --- |
| `sample_id` | `string` | no | Foreign key to samples |
| `annotation_id` | `string` | no | Deterministic box key |
| `x` | `float64` | no | Left coordinate in source pixels |
| `y` | `float64` | no | Top coordinate in source pixels |
| `width` | `float64` | no | Positive width in source pixels |
| `height` | `float64` | no | Positive height in source pixels |

Coordinates are finite and unique within a sample. Each positive sample has at least one box;
negative samples have none. Validation checks `x + width <= image_columns` and
`y + height <= image_rows` using the sample artifact, so portable validation does not need the
source DICOMs.

## Splits

File: `splits.parquet`. One row per sample, sorted by `sample_id`.

| Field | Arrow type | Nullable | Definition |
| --- | --- | --- | --- |
| `sample_id` | `string` | no | Foreign key to samples |
| `split_name` | `string` | no | `train`, `validation`, or `test` |

Validation requires complete unique sample coverage and one split per patient. It recomputes the
assignments from the manifest recipe and verifies class presence when cohort size permits.
Split-wide lineage is stored once in the manifest.

## Source inventory

File: `source_inventory.parquet`. One row per sample, sorted by `sample_id`.

| Field | Arrow type | Nullable | Definition |
| --- | --- | --- | --- |
| `sample_id` | `string` | no | Foreign key to samples |
| `relative_path` | `string` | no | Portable path beneath the raw dataset root |
| `byte_size` | `int64` | no | Positive source file size |
| `sha256` | `string` | no | SHA-256 of exact source DICOM bytes |

Construction authenticates every inventory row against the external DICOM. Portable validation
checks the inventory schema, relationships, paths, sizes, hashes, and artifact integrity; it does
not rehash the external dataset.

## Manifest

File: `manifest.json`.

The manifest owns bundle-level data: dataset and release identity, task definitions, split
lineage, source CSV hashes, artifact hashes, essential counts, source-quality evidence, privacy
classification, bundle identity, and runtime provenance. Recomputable distributions belong to the
bundle-qualified audit report.

Split metadata includes `split_source`, `split_recipe_id`, `split_assignment_id`,
`algorithm_version`, seed, stratification target, and ordered train/validation/test ratios. It
also records the algorithm mechanics needed to interpret the version.

## Bundle acceptance

A loader accepts a bundle only when:

1. the directory contains exactly the manifest and five required Parquet files;
2. each entry is a regular, non-symlink file;
3. the manifest schema, tasks, and split algorithm are supported;
4. the manifest and directory name declare the requested bundle ID;
5. every artifact has its exact Arrow schema;
6. every declared logical and physical hash matches;
7. the semantic bundle ID recomputes exactly;
8. all cross-table, split, annotation, and source-inventory invariants pass.

Samples, splits, and inventory rows are ordered by `sample_id`; labels by
`(sample_id, task_id)`; annotations by `annotation_id`. Unexpected columns and entries are
rejected.

## Identity and publication

The logical Arrow hash covers canonical ordered typed content and the exact schema. It participates
in bundle identity and is independent of Parquet encoding and null-buffer representation.

The physical Parquet SHA-256 covers serialized bytes. It detects corruption or replacement but
does not participate in bundle identity.

Bundle identity covers the existing identity-policy and schema versions, dataset and source
release, task definitions, split source and compact recipe metadata, authoritative CSV hashes, and
the five ordered logical artifact hashes. It excludes physical hashes, generated summaries,
timestamps, commands, paths, tool diagnostics, and `CURRENT`.

Publication validates a complete sibling staging directory once, atomically renames it to its
immutable `bundle-<sha256>` directory, then atomically updates `CURRENT`. Independent consumers
perform complete validation when loading a bundle. Scientific runs should record an explicit
bundle ID; `CURRENT` is the interactive selection pointer.
