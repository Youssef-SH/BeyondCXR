# Symile-MIMIC source guide

Symile-MIMIC is BeyondCXR's primary multimodal dataset. Access is governed by the PhysioNet
credentialed-data license, data-use agreement, and required training. Source data and patient-level
derivatives remain outside version control. The complete cohort and limitation statement is in the
[data statement](../data_statement.md).

## Source readiness

BeyondCXR never downloads Symile. Place the authorized Symile-MIMIC 1.0.0 release at the canonical
root and run the pre-test workflow:

```bash
# Place the release at data/raw/symile/extracted.
make symile
```

A one-time alternate root can be recorded with
`make setup SYMILE_SOURCE_ROOT=/approved/symile/source`. Missing or invalid source state fails with
an operator action; there is no acquisition fallback.

## Verified release layout

The source contract authenticates `SHA256SUMS.txt`, the admission index, official train,
validation, and test-retrieval files, laboratory references, and split-aligned modality tensors.
The immutable bundle contains development sample and laboratory evidence and opaque authenticated
copies of the mixed source index and test CSV. Large source tensors remain externally authenticated.
Only known development admissions are projected from the mixed index before CSV decoding.

## Cohort, labels, and split

Classification membership contains the 10,000 official train admissions, 750 official validation
admissions, and 464 positive-query official test admissions. The remaining 408 admissions in the
11,622-admission source index are reconciled after authorized opening and are not added to
classification membership. The strict endpoint admits only explicit `Pneumonia = 1` and
`Pneumonia = 0` rows; uncertain and missing values are excluded.

Development combines only eligible official train and validation admissions. Repeated development
CV is a separate authority containing three patient-grouped, class-stratified five-fold repeats
with seeds 17, 42, and 2026. The workflow validates both authorities explicitly; mutable `CURRENT`
markers do not select formal scientific state.

Model development and fold-specific fitting are described in [training](../training.md).
Aggregate reports contain no patient-level rows.

## Source qualification

Pre-opening qualification verifies official transport checksums and development CSV and NPY
schemas, tensor shapes and dtypes, identifier alignment, patient and admission isolation, AP/PA
membership and modality contracts. It does not decode test or retrieval scientific content.
Full source reconciliation, held-out semantic qualification and retrieval checks require authorized
opening; the pre-opening audit reports only development qualification. The laboratory timing policy is authenticated from the
official source and reference-code contract because raw parent laboratory events are outside the
release scope.

## Held-out boundary

`make symile` establishes formal execution, then completes development, the unified development
decision, full-development fitting, all final packages, and the immutable scientific freeze. It then stops and states
that official test data have not been opened.

After reviewing the freeze, the explicit action

```bash
make symile-test
```

validates the same freeze and execution, publishes or recovers matching local and external opening
receipts, and only then permits held-out predictions, global-result derivation, deterministic
derivatives, preservation, backup, restoration, and recursive certification. Resume after opening
cannot return to development, fitting, package selection, freeze creation, or execution creation.
