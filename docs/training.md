# Training and scientific execution

Reviewed configurations declare dataset choice, models, preprocessing, fitting, augmentation,
and evaluation policies. Canonical builders compute bundle and CV identities; the workflow passes
the exact returned coordinates to campaign preflight. YAML contains no generated authority pins.
Parsers reject unknown, missing, duplicate, mistyped,
non-finite, and contradictory values. Machine paths, Git, dependency lock, runtime, hardware, and
worker policy are operational provenance rather than scientific identity.

## Execution boundary

The normal operator sequence is documented in the [README](../README.md). Make delegates to a
small composition layer; scientific state transitions, validation, and immutable publication
remain owned by the domain lifecycle modules. Formal workflows use explicit authority identities,
never `CURRENT`, directory ordering, timestamps, or newest-file rules.

## Symile development and decision

Development combines eligible official train and validation admissions. One immutable CV authority
defines three patient-grouped, class-stratified five-fold repeats. Each family result validates all
15 package/evidence coordinates and owns its development evidence and terminal budget. The unified
development decision owns complete ordered family membership, selection policy, selected final
families and budgets, family-authority mapping, and primary thresholds.

For each repeat and outer fold, the outer holdout is reserved for out-of-fold inference.
Laboratory ECDFs and missing-value replacements fit on the complete outer-training cohort. A
deterministic patient-grouped inner assignment is shared by selection families. Labs Logistic
Regression fits the complete outer-training fold without inner selection. Labs LightGBM uses
inner-validation AUROC for early stopping and retains `best_iteration`; neural families use the
same metric for checkpoint selection, scheduling, and early stopping. Selected fold states are not
refitted before outer-holdout inference.

The prespecified development families are:

```text
Labs Logistic Regression
Labs LightGBM
CXR DenseNet121
CXR + labs concat
CXR + labs missingness-aware gated
CXR + labs gated with observedness inputs zeroed
CXR + labs + ECG gated
```

The two CXR/labs gated variants share architecture and initialization; observedness masking is
their only scientific difference. The ECG family uses a fixed 12-lead residual encoder and a
three-modality gate.

Neural families use the reviewed TorchXRayVision DenseNet121 initialization. Fusion folds initialize
only from the corresponding CXR fold. Terminal models fit all development admissions at the fixed
decision-owned budget without validation selection or early stopping. Fourteen validated final
packages are complete before the pre-test freeze is published.

The standard CXR branch uses `densenet121-res224-chex` and a 1,024-dimensional representation.
Neural fitting uses AdamW, weight decay `1e-4`, global-norm clipping at 1.0, batch size 32, no class
reweighting, and two stages:

| Stage | Trainable state | Maximum development epochs | Learning rate |
| --- | --- | ---: | --- |
| 1 | CXR classification head and all applicable non-CXR components | 2 | `1e-3` |
| 2 | Denseblock4, norm5, family head, and all applicable non-CXR components | 28 | CXR `1e-5`; other `1e-4` |

The scheduler halves learning rates after two non-improving validation epochs. Early stopping uses
patience five and minimum AUROC improvement `1e-4`. Terminal fitting uses the decision-owned fixed
epoch budget and fixed within-stage rates. Laboratory inputs are 50 fold-fitted right-rank ECDF
values followed by 50 observedness indicators. The two-modal gate combines 256-dimensional CXR and
laboratory representations feature-wise.

`make symile` performs source qualification, bundle and CV publication/reuse, aggregate audit,
pretrained-weight readiness, mutation-free formal preflight, execution publication, development,
decision, final fitting, package publication, and scientific freeze. It does not open official test
data.

`make symile-test` is the deliberate irreversible action. The canonical lifecycle validates the
active execution and sole freeze, coordinates matching local and persistent opening receipts, and
issues the held-out capability. Only then can held-out source reconciliation, semantic and modality
qualification, package-bound predictions, the global result,
deterministic derivatives, and preservation proceed.

## Supporting RSNA campaign

RSNA includes metadata Logistic Regression and LightGBM, three CXR members, three same-seed
CXR-plus-metadata members, and aggregate Grad-CAM localization. Metadata preprocessing fits on
training only. The exact 70/15/15 patient-disjoint split with seed 42 belongs to the bundle.

Neural training uses a two-stage head-only/full-encoder policy and validation Average Precision
selection. Each fusion member initializes from the corresponding same-seed CXR package.
Validation derives package- and seed-specific operating points. The three compatible seed
evaluations retain every member result and are summarized by arithmetic mean and sample standard
deviation; the summary selects no best seed and produces no averaged RSNA model.

`make rsna` validates or safely acquires the source, builds/reuses the exact dataset authority,
produces aggregate audits, prepares the pretrained initialization, performs formal preflight,
publishes the execution record, creates the deterministic CXR cache, trains or resumes packages,
freezes the package set, performs held-out evaluation and localization, and preserves and restores
the completed campaign. Once frozen, resume cannot retrain or change package membership.

## Runtime, resume, and publication

Formal neural execution requires the reviewed CUDA contract and fixed worker policy. Execution
records bind clean Git, the dependency lock, normalized run plan and configs, runtime/framework
coordinates, deterministic settings, and pretrained materialization. These facts do not alter
scientific identities.

Training seeds Python, NumPy, and PyTorch. Sample order derives from seed and epoch; augmentation
derives from seed, epoch, and sample identity. Deterministic PyTorch algorithms are requested and
cuDNN benchmarking is disabled. The project records the effective CUDA stack, GPU coordinates,
autocast, loader behavior, and framework versions but does not claim bit-identical training across
arbitrary supported hardware.

Scientific objects publish through validated staging and immutable no-replace installation.
Identical existing objects are recursively validated and reused; conflicting objects fail closed.
Campaign locks coordinate only local multi-object transitions. MLflow records operational
provenance and references, while authority remains in the validated bundle, package, prediction,
result, freeze, and execution graph.

## Evaluation definitions

Symile reports AUROC, Average Precision, and Brier score on raw probabilities. Ten-bin uniform
reliability curves are descriptive; no post-hoc recalibration is applied. The primary paired test
effect is gated minus CXR. Positive AUROC or Average Precision effects favor gated fusion; negative
Brier effects favor gated fusion.

Calibration slope and intercept are descriptive point estimates. Probabilities are clipped to
`[1e-6, 1 - 1e-6]`, transformed to log-odds, and used as the single feature in L2 Logistic
Regression with `C=1e6`, solver `lbfgs`, `max_iter=2000`, and an intercept. This assesses raw
probabilities and does not fit or apply recalibration.

Held-out uncertainty uses a subject-cluster bootstrap with 2,000 accepted resamples and the
prespecified linear 2.5th/97.5th percentile interval. Repeated-CV evidence is reported by repeat and
as the prespecified mean-logit ensemble without a bootstrap confidence interval.

The primary ensemble's Youden-J threshold maximizes sensitivity minus false-positive rate. Its
target-sensitivity threshold is the highest threshold meeting sensitivity 0.90. Both enumerate
finite ROC thresholds and select the highest threshold on ties. They are derived once from
aggregate development out-of-fold evidence and applied unchanged to held-out probabilities; they
are research operating points, not clinical decision rules.

Supporting RSNA probability reports use expected calibration error with 15 equal-width bins over
`[0, 1]`; the final bin includes 1, empty bins contribute zero, and non-empty bins are weighted by
sample fraction times absolute confidence-to-observed-frequency difference. RSNA calibration
slope and intercept use the same raw-probability procedure. Latency covers the complete CPU
preprocessing and probability path: 100 warm-up calls followed by the median of 1,000 individually
timed calls on the deterministic first sample. Model size is exact serialized model bytes divided
by 1,048,576, reported in MiB.

Localization reports the pointing game, where the row-major first heatmap maximum must fall inside
the union of reference boxes, and activation energy, the fraction of nonnegative heatmap mass
inside that union. A zero heatmap receives zero for both metrics.
