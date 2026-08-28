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
bundle and split identities plus the exact manifest-byte witness. Authority publication is a
separate lifecycle step. `CURRENT` is limited to interactive dataset operations and is never read
by a formal campaign.

RSNA packages labeled DICOM metadata, labels, boxes, patient-disjoint splits, and a source
inventory. A deterministic authenticated CXR cache separates raw DICOM decoding from model
training. Symile packages the lean admission spine and raw laboratory values while retaining
authenticated references to the large CXR and ECG tensors. Its separate CV assignment artifact fixes all
repeat/fold assignments.

Formal RSNA preflight constructs one immutable plan containing every root, exact authority
coordinate, ordered family/seed member, loaded config witness, source revision, dependency lock,
pretrained-weight fingerprint, and the training/evaluation and full-precision localization
runtime witnesses. Every later phase consumes that plan;
no phase re-resolves a root or mutable authority selector. The complete package/report freeze is
the capability required by each held-out accessor and by localization.
Each immutable model package contains the ordered validation sample IDs, their targets, their
probabilities, and the minimal selection history needed to reconstruct validation claims. The
package identity binds this scientific evidence; report and plot bytes are deterministic
derivatives. Package-freeze validation independently rederives canonical validation membership and
labels from the pinned bundle before accepting package evidence. The freeze binds ordered package
membership and report coordinates, while preservation hashes witness transport bytes.

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

The RSNA bundle and split are published before formal execution and every RSNA config pins that
same bundle ID, manifest SHA-256, and split-assignment ID. A read-only campaign preflight validates
the complete bundle, raw-source availability, configuration agreement, Git and lock provenance,
CUDA contract, storage, and external backup destination before creating campaign output.

The campaign then authenticates source DICOM bytes while building or validating the deterministic
CXR cache, fits metadata and three-seed neural families, and publishes an immutable eight-package
freeze before held-out access. Package-bound completion records make held-out evaluation
append-only and resumable. Aggregate evaluations, seed summaries, comparison, and localization
are content-addressed or validated derivatives. The preservation ZIP contains an explicit
scientific closure—not repository roots—and is restored and recursively validated before its
checksum and external backup are accepted.
Private localization state includes compact per-case numerical evidence from the completed
Grad-CAM procedure. Restoration recomputes public member and aggregate statistics from that
evidence without requiring the original DICOM files.

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
