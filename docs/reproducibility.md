# Reproducibility

BeyondCXR separates scientific identity, scientific evidence, operational execution, and transport
integrity. A fresh checkout can reconstruct the same semantic authorities from the same
authenticated source and reviewed policies without possessing generated manifests from another
machine.

Exact reproduction validates preserved fitted state and deterministically regenerates aggregate
outputs. Experimental replication retrains the prespecified procedure and records
environment-dependent variation. A wheel or source distribution contains software only; study
reconstruction also requires the provenance-linked checkout, lockfile, reviewed configurations,
authenticated data authorities, and preserved evidence.

## Machine setup

Follow the [README workflow](../README.md#workflow) for the normal sequence.

Setup installs the locked development, acquisition, and serving environments and writes the ignored
`.beyondcxr.local.toml`. That file contains only the persistent backup root and optional source
roots. It cannot set scientific hyperparameters, identifiers, thresholds, runtime policy, workers,
or credentials. The backup root has no implicit default. Interactive setup prompts when it is missing;
noninteractive setup requires `make setup BACKUP_ROOT=/approved/persistent/backup`.
Explicit setup arguments take precedence over saved values; source roots otherwise use the
canonical locations in the dataset guides.

RSNA may be acquired through the standard Kaggle client only when the configured source root is
absent. Symile is always operator supplied. Neither dataset is downloaded by `make setup`.

## Semantic reconstruction

Canonical scientific identities use strict canonical JSON, explicit domain/schema separation,
logical table values, and semantic tensor/fitted state. They exclude timestamps, machine paths,
physical serializer bytes, Git, runtime hardware, archive coordinates, and deployment coordinates
by construction. Operational execution and transport records have their own identities.

RSNA source inventory, cohort, labels, annotations, policies, and exact split assignment form one
bundle authority. The Symile bundle owns authenticated release membership and official partitions;
repeated development CV remains a separate bundle-bound authority. Source byte hashes authenticate
upstream material but generated manifest bytes are not scientific lineage.

Packages bind normalized scientific configuration, source authorities, fitted state, preprocessing
state, initialization identity, lineage, and validation evidence. Predictions bind packages,
cohorts, inference policy, logical values, and held-out freeze authorization where applicable.
Evaluation, localization, and the global result are rederived from certified evidence.

For RSNA, the deterministic image cache contains decoded, spatially standardized,
pre-augmentation CXR representations. Cache preparation may process bytes from every partition,
but it reads no task labels, fits no statistics, selects no thresholds, and fits no model. Every
source DICOM byte is authenticated before use. Model reconstruction uses packaged fitted state
with `weights=None`, so held-out evaluation and restoration do not reacquire the pretrained
initialization.

## Freeze, execution, and held-out chronology

The scientific freeze records the complete prospectively fixed scientific plan. The formal
execution record separately binds Git, lock, run plan, runtime/framework coordinates, deterministic
settings, hardware policy, workers, and pretrained materialization. Operational changes can alter
execution identity without altering scientific identity.

Symile held-out access requires the sole valid freeze, the active matching execution, and identical
local and persistent opening receipts. If one receipt is missing, it is recovered from the other;
conflicts fail closed. Before opening, source hashing and opaque copying authenticate transport;
only development records are decoded and semantically validated. After opening, the same bundle
supplies authenticated test membership and labels for full reconciliation, while live source
tensors undergo held-out qualification before prediction. Restored historical evidence authorizes
reconstruction from the authenticated archive without live-source access.
After opening, development, decision replacement, fitting, package-set
changes, freeze replacement, and execution replacement are prohibited. Missing predictions,
results, deterministic derivatives, and preservation may resume.

## Preservation and release

Completed archives use stable names under `outbox/`:

- RSNA: `rsna-<freeze-id>-<execution-id>.zip`;
- Symile: `symile-<global-result-id>-<execution-id>.zip`.

A completed archive is validated and reused. Its external backup is an exact copy of the primary,
not an independently rebuilt compressed archive.

Preservation enumerates a bounded campaign closure. It records canonical restore paths and exact
member hashes, creates a safe deterministic ZIP structure, publishes atomically, copies the primary
archive exactly to the configured backup, and validates byte equality. Archive and member hashes
protect transport; restoration re-establishes scientific truth by invoking the same domain
validators used for live state.

Release accepts zero, one, or multiple recursively valid completed Symile campaigns as failure,
success, or ambiguity respectively. It never selects by timestamp, mutable pointer, filename order,
or newest state. The accepted campaign is restored, recursively validated, rendered into an
independent public result tree, reproduced again, and compared byte-for-byte. Privacy, package,
license, serving, and container/API checks complete before the accepted serving state is published.

Checkout cleanliness covers all tracked changes and non-ignored untracked files in the project
source directories (`src`, `tests`, `configs`, `docs`, `examples`, `scripts`, `data`, `results`, `.github`) and
root build/documentation files. Release inspection uses the same ownership boundary, including
staged additions and rename destinations. Other untracked workspace material is outside this
boundary; ignored generated state is not release content.

Serving distribution and startup requirements are documented in [serving](serving.md).

On its first run, `make release` derives missing public results and the binding, then independently
reproduces them. Existing results must reproduce exactly; a mismatch fails without overwriting them.
Final acceptance requires a clean committed checkout. Review the generated results and bounded
README/model-card regions, update the package, citation and root lockfile version to `1.0.0`, and
replace the changelog `Unreleased` heading with `1.0.0 — YYYY-MM-DD`. Commit these permitted release
changes, then rerun `make release`. The command validates the committed results and completes
serving and container acceptance. No separate result-generation command is required. Development
remains at `0.1.0` until this post-experiment release preparation.

## Guarantees and limits

The repository guarantees deterministic semantic identity and recursive authority validation,
strict chronology, explicit operational provenance, and exact preservation transport checks.
Deterministic algorithms and reviewed runtime settings reduce execution variance, but the project
does not claim bit-identical training across arbitrary GPUs, CUDA stacks, or framework versions.
Reproduction of public results starts from preserved certified evidence rather than retraining.

## Experimental replication

Replication uses the same authenticated releases, immutable bundle and CV assignments, reviewed
configurations, seeds, preprocessing, selection rules, terminal budgets, and campaign procedure.
It records a new execution environment and produces independently validated objects. GPU state and
metrics may differ across supported environments; those differences are reported rather than
hidden behind a fixed metric tolerance. Replication does not change the original result binding.

## Developer checks

Routine developer validation is:

```bash
make check
```

It runs the lock, Ruff, formatting, and default pytest gates. The bounded integration suite is:

```bash
uv run pytest -o addopts= -q -m integration
```

Tests validate software invariants with bounded synthetic examples. Integration tests exercise
model serialization, DICOM, archive restoration and API boundaries with tiny inputs. Exact
scientific membership is checked as metadata; the scientific workflow alone executes the
full repeated-CV and final-package topology. Neither test command opens official held-out
data or runs the production bootstrap budget.
