# Architecture

BeyondCXR separates source qualification, scientific execution, public reporting, and serving.
Scientific consumers receive explicit validated object identities; MLflow records operational
provenance.

## Study structure

Symile-MIMIC carries the primary multimodal study. Its prespecified families test the incremental
value of laboratory information beyond CXR, missingness-aware fusion, and the further contribution
of ECG. RSNA Stage 2 supplies the supporting DICOM, image-modeling, metadata-fusion, and
localization foundation. The datasets have
different endpoints and separate evidence chains.

```text
authenticated sources
        ↓
immutable dataset bundle + split authority
        ↓
model package → prediction evidence → scientific result
        ↓
validated preservation export
        ├── aggregate public results
        └── controlled serving
```

## Data boundary

Source adapters authenticate dataset-specific files and construct typed artifacts. Bundle
validators enforce schemas, ordering, relationships, source witnesses, split isolation, logical
content hashes, physical byte hashes, and semantic identity. Scientific consumers pin immutable
bundle and split identities. `CURRENT` is limited to interactive dataset operations.

RSNA packages labeled DICOM metadata, labels, boxes, patient-disjoint splits, and a source
inventory. A deterministic authenticated CXR cache separates raw DICOM decoding from model
training. Symile packages the lean admission spine and raw laboratory values while retaining
authenticated references to the large CXR and ECG tensors. Its separate CV assignment artifact fixes all
repeat/fold assignments.

Exact schemas and identity rules live in the [data contract](data_contract.md). Dataset roles,
access, and limitations live in the [data statement](data_statement.md).

## Model and evaluation evidence

At the model/evaluation layer, scientific claims follow a three-role evidence chain:

| Object | Responsibility |
| --- | --- |
| Model package | Reconstructable fitted model and preprocessing state |
| Prediction evidence | Immutable package-bound sample-level outputs |
| Scientific result | Aggregate claims rederived from validated prediction evidence |

Semantic identity records scientific meaning. The sealed science-execution Git commit is a public
code-state and scientific coordinate in the result binding. Integrity witnesses authenticate
serialized bytes; the dependency lock, MLflow, paths, device, runtime, and ordinary execution
facts are reproducibility or operational provenance. Validators keep those roles distinct.

Dataset bundles, split and CV assignments, pre-test controls, and preservation exports are
distinct data and control objects that support this evidence chain.

Patient-level prediction evidence remains under the ignored `private/` workspace. Public reports
contain aggregate derivatives only.

## Primary Symile workflow

Development consumes official train and validation rows only:

```text
pinned bundle + pinned 3 × 5 grouped CV assignment
        ↓
fold package + separate private OOF evidence
        ↓
family development result
        ↓
prespecified CXR/labs analysis + secondary ECG analysis
        ↓
14 terminal full-development packages
```

The prespecified CXR/labs families share one repeated grouped cross-validation assignment. The
secondary ECG family uses the same development design. Family results rederive repeat metrics and
terminal budgets from all 15 fold coordinates; model definitions and fitting rules are documented
in [training](training.md).

Held-out test materialization requires a fully validated pre-test freeze and its atomic
same-freeze opening record. The freeze binds all 14 packages, the two primary development-derived
thresholds, held-out policy, numerical inference runtime, and formal execution provenance.

```text
validated freeze + same-freeze opening
        ↓
14 package-bound raw predictions
        ↓
six fixed predictor views
        ↓
one global result + private error review
        ↓
validated export, backup, and restoration
```

An opened campaign can complete missing deterministic post-open work; it cannot retrain or alter
the freeze. The global result owns held-out claims. Private error review is regenerable and remains
outside the public result surface.

## Supporting RSNA workflow

The RSNA campaign authenticates source DICOM bytes, builds the deterministic CXR cache, fits
metadata and three-seed neural families, freezes validation-derived thresholds, evaluates explicit
packages on the held-out partition, and publishes aggregate evaluation, seed-summary, comparison,
and localization results. Private aligned predictions and real-image overlays remain outside Git.

## Release and serving

The release layer restores the preserved Symile campaign, validates its scientific evidence,
derives the path-neutral result binding, and constructs an aggregate-only public projection.
Reproduction derives the same projection from the same preserved evidence. The tracked outputs are
deterministic derivatives, not scientific authorities.

Serving uses a separate immutable deployment control. It binds the ordered seed-17, seed-42, and
seed-2026 primary CXR-plus-labs gated packages, the mean-logit policy, preprocessing, global-result
lineage, thresholds as metadata, and science/release provenance. Startup validates and reconstructs
all members. Runtime inference averages raw logits and applies sigmoid once.

See [training](training.md), [reproducibility](reproducibility.md), [privacy](privacy.md), and
[controlled serving](serving.md) for the operational contracts.
