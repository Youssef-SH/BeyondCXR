# Data bundle contract

The common bundle manifest uses schema version `1`. Bundle IDs identify semantic scientific
content; source hashes authenticate exact upstream bytes.

Both datasets publish immutable `bundle-<sha256>` directories with `manifest.json` and typed
Parquet artifacts. Their manifests use exactly this common envelope:

```text
bundle_manifest_schema_version
dataset
tasks
membership
source
modalities
privacy
bundle
artifacts
qualification
```

Arrow artifacts declare `semantic_sha256` and `row_count`. Dataset-specific scientific sections
remain explicit.

`CURRENT` is an optional interactive discovery pointer. Campaigns receive explicit semantic
bundle IDs from dataset builders; reviewed YAML contains scientific policy without generated
authority pins. The canonical patient-level RSNA split uses 70% train, 15% validation, and 15% test with seed 42; its policy and exact assignment are intrinsic to the RSNA
bundle, with no separate split-assignment authority. Symile's official train, validation, and test
partition is likewise intrinsic to its bundle. Symile's development design is a separate immutable
three-repetition, five-fold, patient-grouped CV authority under `data/manifests/symile/cv/`, using
repeat seeds 17, 42, and 2026.

## RSNA bundle

An RSNA bundle contains five Parquet artifacts and one JSON manifest. Column order, Arrow types,
nullability, and deterministic row order are contractual.

### Samples

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

### Labels

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

### Annotations

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

### Splits

File: `splits.parquet`. One row per sample, sorted by `sample_id`.

| Field | Arrow type | Nullable | Definition |
| --- | --- | --- | --- |
| `sample_id` | `string` | no | Foreign key to samples |
| `split_name` | `string` | no | `train`, `validation`, or `test` |

Validation requires complete unique sample coverage and one split per patient. It recomputes the
assignments from the manifest recipe and verifies class presence when cohort size permits.
Split-wide lineage is stored once in the manifest.

### Source inventory

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

### Manifest

File: `manifest.json`.

The manifest owns bundle-level data: dataset and release identity, task definitions, intrinsic
partition lineage, source authentication and qualification evidence, artifact hashes, essential
counts, privacy classification, and bundle identity. Recomputable distributions belong to the
bundle-qualified audit report. Formal execution separately owns Git and dependency-lock witnesses,
runtime and framework facts, hardware and CUDA policy, workers, determinism settings, and other
operational execution provenance.

Split metadata includes `split_source`, `split_recipe_id`, `algorithm_version`, seed,
stratification target, and ordered train/validation/test ratios. It also records the algorithm
mechanics needed to interpret the version.

### Bundle acceptance

A loader accepts a bundle only when:

1. the directory contains exactly the manifest and five required Parquet files;
2. each entry is a regular, non-symlink file;
3. the manifest schema, tasks, and split algorithm are supported;
4. the manifest and directory name declare the requested bundle ID;
5. every artifact has its exact Arrow schema;
6. every declared semantic hash matches the canonical typed table content;
7. the semantic bundle ID recomputes exactly;
8. all cross-table, split, annotation, and source-inventory invariants pass.

Samples, splits, and inventory rows are ordered by `sample_id`; labels by
`(sample_id, task_id)`; annotations by `annotation_id`. Unexpected columns and entries are
rejected.

## Identity and publication

Canonical JSON uses UTF-8, NFC-normalized string keys and values, lexicographically sorted object
keys, compact separators, and exactly one trailing newline. List order is preserved. Floats must
be finite; negative zero is normalized. Booleans remain distinct from integers. Unsupported types
and keys that collide after Unicode normalization fail rather than being coerced. Semantic hashes
frame the domain, schema version, and payload explicitly.

The logical table contract declares its schema name/version, ordered columns and scalar types,
nullability, canonical sort key, and uniqueness constraints. Validation rejects incorrect columns,
types, nulls, duplicate keys, and non-finite values before hashing.

The logical table hash covers canonical ordered typed content, row count, and scientific schema. Its type-aware
scalar encoding is independent of Arrow IPC serialization, chunking, Parquet encoding, file
metadata, Arrow field/schema metadata, source paths, and null-buffer representation.

RSNA bundle identity covers dataset and source release, task and interpretation policies, split source
and compact recipe metadata, authoritative CSV hashes, and the five ordered semantic artifact
hashes. Representation-only schema discriminators, generated summaries, timestamps, commands,
paths, tool diagnostics, and `CURRENT` are excluded.

Publication validates a complete sibling staging directory once, atomically renames it to its
immutable `bundle-<sha256>` directory, then atomically updates `CURRENT`. Independent consumers
perform complete validation when loading a bundle. Formal campaigns receive the builder-returned
bundle ID; the RSNA split assignment is part of that bundle's validated scientific content. Formal campaigns resolve
only the selected directory and never consult `CURRENT`.

Symile source-asset paths are canonical names within the authenticated official release, not local
filesystem locations. Relocating the complete source root leaves semantic bundle identity
unchanged.

## Symile bundle and CV assignment

A Symile bundle contains `manifest.json`, development-only `samples.parquet` and `labs.parquet`,
and opaque authenticated copies of `symile_mimic_data.csv` and `test.csv`. The source copies are
hashed and transported without held-out scientific decoding before opening. The samples table
has one row per official train or validation admission, ordered by `sample_id`:

| Field | Arrow type | Nullable | Role |
| --- | --- | --- | --- |
| `sample_id` | `string` | no | Internal model key |
| `subject_id` | `int64` | no | Patient grouping key |
| `hadm_id` | `int64` | no | Admission alignment key |
| `official_split` | `string` | no | Train or validation |
| `source_row` | `int64` | no | Split-aligned tensor row |
| `pneumonia_state` | `int8` | yes | Source label state |
| `age_years` | `int16` | no | Prespecified subgroup attribute |
| `sex` | `string` | no | Prespecified subgroup attribute |
| `view_position` | `string` | no | AP/PA audit attribute |

The laboratory table starts with `sample_id`, followed by 50 nullable `float64`
`lab_<item_id>_value` columns and 50 non-null `bool` `lab_<item_id>_observed` columns in the
manifest-declared item order. Observedness must equal value presence; observed values must be
finite.

The manifest binds official membership, strict-label policy, source checksum inventory, modality
alignment, temporal and view contracts, laboratory order, semantic artifact hashes,
development qualification, and the semantic bundle identity. Development validation checks exact
admission and patient isolation, tensor-row domains, AP/PA membership, and label domains without
decoding the opaque sources. Authorized held-out reconstruction validates test rows, labels,
laboratories and full source reconciliation against the same source identity. Live held-out
execution additionally qualifies the authenticated test tensors before inference.
Strict-task counts are derived from sample evidence within the authorized analysis scope.

The separate CV directory contains `manifest.json` and `assignments.parquet`. Assignment rows have
`sample_id:string`, `repeat_seed:int32`, and `outer_fold:int8`, ordered by repeat then sample. The
identity binds the Symile bundle, strict task, patient grouping, five-fold stratified-group policy,
repeat seeds 17/42/2026, and logical assignment content. Every eligible development admission
appears exactly once per repeat, and a patient occupies one outer fold within a repeat.
