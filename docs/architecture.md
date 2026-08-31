# Architecture

BeyondCXR separates scientific meaning, supporting evidence, operational execution, and transport
integrity. Symile-MIMIC is the primary multimodal study; RSNA supplies supporting CXR, metadata,
and localization qualification. The studies have different endpoints and separate evidence chains.

## Authority and evidence

| Layer | Responsibility | Examples |
| --- | --- | --- |
| Scientific identity | Identify canonical scientific content | Bundle, CV, package, prediction, freeze, result |
| Scientific evidence | Support identities and claims through recursive validation | Cohort tables, fitted state, predictions, localization evidence |
| Operational provenance | Record the reviewed implementation and runtime | Formal execution, MLflow records, deployment receipt |
| Transport integrity | Authenticate exact preserved bytes | Archive membership, member hashes, backup checksum |

Validators reconstruct semantic content and recompute its identity. Scientific identity is
independent of paths, timestamps, runtime hardware, and serializer encoding. Source-byte hashes
authenticate upstream inputs; preservation hashes authenticate archived bytes.

```text
authenticated sources
        ↓
dataset bundle + separate Symile development CV
        ↓
fitted packages + prediction evidence
        ↓
validated scientific results
        ↓
preservation archive
        ├── deterministic public results
        └── self-contained serving authority
```

## Dataset authorities

The RSNA bundle owns cohort, labels, annotations, source inventory, and the exact patient-disjoint
split. The Symile bundle owns cohort, laboratory values, authenticated modality references, and
official partitions. Symile's repeated development CV is a separate bundle-bound authority.

The workflow captures builder-returned bundle and CV IDs and passes them explicitly to consumers.
`CURRENT` is limited to interactive dataset operations. Exact schemas, canonicalization, and
validation rules belong to the [data contract](data_contract.md); source qualification belongs to
the [RSNA](datasets/rsna.md) and [Symile](datasets/symile.md) guides.

## Packages, decisions, and results

Packages identify fitted model and preprocessing state, scientific configuration, initialization,
and supporting validation evidence. RSNA packages own their validation-derived thresholds.
Symile family development authorities own complete fold and out-of-fold evidence; one development
decision binds ordered family membership, final fitting budgets, and primary thresholds.

Prediction evidence binds package, cohort, inference policy, logical values, and the authorizing
scientific freeze for held-out work. Evaluations and the Symile global result rederive aggregate
claims from that evidence. Localization identifies numerical Grad-CAM evidence and its policy.
Audits, comparisons, seed summaries, and private error reviews are regenerable views.

See [training](training.md) for model development, fitting, and evaluation definitions.

## Freeze, execution, and held-out access

A scientific freeze fixes complete package membership and held-out policies. Formal execution
records the reviewed Git commit, dependency lock, run plan, and numerical runtime separately.
Execution is established before fitting; the freeze is published after the required packages exist.

| Study | Before held-out access | Operator boundary |
| --- | --- | --- |
| RSNA | Eight-package scientific freeze joined to current formal execution | `make rsna` continues through held-out evaluation and localization |
| Symile | Development decision, 14 final packages, scientific freeze, current formal execution, and matching local/external opening receipts | `make symile` stops at freeze; `make symile-test` opens or resumes held-out work |

Live held-out capabilities authenticate these relationships before source access. A freeze alone
is not a live capability. Restored historical evidence supports validation and reproduction without
granting live source access. After opening, Symile may complete missing predictions, results,
derivatives, and preservation; development and fitting cannot resume. Recovery and receipt rules
are documented in [reproducibility](reproducibility.md).

## Preservation, release, and serving

Preservation archives an explicit scientific closure and validates its restored content. Release
selects exactly one valid completed Symile archive, derives aggregate public results, independently
reproduces their bytes, and completes acceptance checks before publishing accepted serving state.
The result binding is an index of scientific coordinates; public tables and figures are derivatives.

Serving authority contains the ordered primary gated package copies under their original package
IDs, with inference and input/output policy. Deployment provenance records operational release
facts separately. Startup authenticates the distribution against the preserved campaign and
release evidence; inference averages the three raw logits and applies sigmoid once.

See [reproducibility](reproducibility.md#preservation-and-release) for preservation and release,
[serving](serving.md) for deployment and the API, and [privacy](privacy.md) for distribution limits.
Make and the Python workflow compose these domain operations without owning scientific identities.
