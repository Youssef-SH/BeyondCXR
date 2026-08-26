# Training and scientific execution

Experiment configurations declare the prespecified dataset, model, preprocessing, fitting, and
evaluation settings consumed by versioned code, policies, and model contracts. Runtime arguments
provide operational settings such as paths, devices, workers, and seeds. Config loaders reject
unknown, missing, duplicate, mistyped, non-finite, and contradictory values.

## Configuration ownership

Each config pins its dataset, semantic bundle ID, bundle-manifest byte witness, split identity,
task, family, preprocessing, fit policy, scientific batch size, augmentation, and evaluation
policy. Symile configs also pin the CV assignment. Runtime paths, MLflow destinations, hardware,
worker counts, and execution seeds remain outside scientific configuration identity.

The complete semantic config identifies the experiment. Model-package identity uses the narrower
fit projection, so downstream reporting policy cannot rename identical fitted state. Exact config
bytes remain an integrity witness.

Experiment configurations are in [`configs/`](../configs/). They define the prespecified model
families and hyperparameters.

## Primary Symile development

Development combines eligible official train and validation admissions. One immutable assignment
defines three patient-grouped, class-stratified five-fold repeats with seeds 17, 42, and 2026.
Every family uses the same outer coordinates.

For a repeat and outer fold:

1. The outer holdout is reserved for OOF inference.
2. Laboratory ECDFs and missing replacements fit on the complete outer-training cohort.
3. A deterministic patient-grouped inner assignment is shared by selection families.
4. The fitted fold package is published and reconstructed.
5. A separate immutable OOF prediction object is published.

Labs Logistic Regression fits the complete outer-training fold with no inner selection. Labs
LightGBM uses inner-validation AUROC for early stopping and retains `best_iteration`. Neural
families use inner-validation AUROC for checkpoint selection, scheduling, and early stopping.
Selected LightGBM and neural states are not refitted within an outer fold.

The prespecified CXR/labs development families are:

```text
Labs Logistic Regression
Labs LightGBM
CXR DenseNet121
CXR + labs concat
CXR + labs missingness-aware gated
CXR + labs gated with observedness inputs zeroed
```

The secondary ECG family uses a three-modality CXR-plus-labs-plus-ECG gate. Fusion folds initialize
only from the corresponding CXR fold at the same repeat and outer fold. The two gated variants
share architecture and initialization; observedness masking is their only scientific difference.

Each family result validates all 15 package/evidence coordinates and rederives repeat metrics and
its applicable terminal budget. The cross-family analysis aligns OOF logits, reports every repeat,
and computes fixed mean-logit ensemble point estimates. It defines no repeated-CV confidence
interval.

Run one development family with:

```bash
make symile-develop CONFIG=configs/symile_labs_logistic.yaml
make symile-develop CONFIG=configs/symile_cxr_densenet.yaml
make symile-develop CONFIG=configs/symile_cxr_labs_gated.yaml \
  SOURCE_CXR_DEVELOPMENT_ID=development-SHA256
```

Aggregate the six explicit development identities with:

```bash
make symile-analyze DEVELOPMENT_IDS="<six development IDs>"
```

## Symile full-development fitting and held-out campaign

Terminal budgets are the median selected epoch or `best_iteration` across a family's 15 outer
fits. Terminal models fit all development admissions without validation selection or early stopping.
The package set contains one Labs Logistic Regression, one Labs LightGBM, and three neural members
for each of CXR, concat, two-modal gated, and ECG-gated: 14 packages in total.

The campaign publishes and validates every package before creating the pre-test freeze. Official
test access then requires both the opaque validated freeze capability and its atomic same-freeze
opening record. Fourteen raw package-bound predictions form six fixed views. Neural views average
the ordered seed-17/42/2026 logits and apply sigmoid once.

The global result rederives raw probability metrics, primary operating points, paired subject-level
bootstrap effects, and descriptive reliability. Primary thresholds come only from aggregate
development OOF evidence. Private false-positive/false-negative review is a regenerable derivative.

Formal execution uses one command and requires an approved external backup destination:

```bash
make symile-campaign \
  SOURCE_ROOT=/approved/symile/source \
  BACKUP_ROOT=/approved/persistent/backup
```

This command is reserved for the sealed science-execution commit in a clean checkout. It resumes only the bound
opened freeze and never substitutes a mutable bundle or package pointer.

## Symile neural policy

The standard CXR branch is TorchXRayVision DenseNet121 with the
`densenet121-res224-chex` initialization and a 1,024-dimensional representation. Neural fitting
uses AdamW, weight decay `1e-4`, global-norm clipping at 1.0, batch size 32, no class reweighting,
and two stages:

| Stage | Trainable state | Maximum development epochs | Learning rate |
| --- | --- | ---: | --- |
| 1 | Non-CXR components | 2 | `1e-3` |
| 2 | Denseblock4, norm5, and non-CXR components | 28 | CXR `1e-5`; other `1e-4` |

The scheduler halves learning rates after two non-improving validation epochs; early stopping uses
patience five and minimum AUROC improvement `1e-4`. Terminal fitting uses the fixed epoch
budget and fixed within-stage rates.

Laboratory inputs are 50 fold-fitted right-rank ECDF values followed by 50 observedness indicators.
The two-modal gate combines 256-dimensional CXR and lab representations feature-wise. The ECG
family adds a fixed 12-lead residual encoder and a three-modality gate. Architecture
details are enforced by configuration and model contract tests.

## Supporting RSNA campaign

RSNA establishes the medical-imaging foundation with metadata Logistic Regression and LightGBM,
three CXR members, three same-seed CXR-plus-metadata concat members, and aggregate Grad-CAM
localization. Metadata preprocessing fits on training only. Neural training uses a two-stage
head-only/full-encoder policy and validation Average Precision selection.

Each fusion member initializes from the corresponding same-seed CXR package. Validation derives
the operating points for each applicable package, so thresholds remain package- and seed-specific.
The three compatible seed evaluations retain every member result and are summarized by their
arithmetic mean and sample standard deviation. The summary selects no canonical best seed and
produces no averaged RSNA model.

The campaign authenticates every source DICOM byte, constructs the deterministic CXR cache,
freezes all eight packages before held-out evaluation, and passes explicit identities between
training, evaluation, summaries, comparison, and localization:

```bash
make rsna-campaign
```

RSNA package and evaluation details remain in the [RSNA dataset guide](datasets/rsna.md) and
[reproducibility guide](reproducibility.md).

## Determinism, publication, and tracking

Training seeds Python, NumPy, and PyTorch. Sample order derives from seed and epoch; augmentation
derives from seed, epoch, and sample identity. Deterministic PyTorch algorithms are requested and
cuDNN benchmarking is disabled. Runtime provenance records the effective device, CUDA stack,
mixed precision, loader behavior, and library versions.

Packages and results publish through sibling staging and exclusive no-overwrite installation.
Safe neural packages contain CPU tensor state; supported tabular packages use allowlisted skops
types. A completed operational run writes `run_complete=true` only after owned scientific outputs
publish and validate. MLflow records provenance and references; scientific authority follows the
validated package, prediction, and result chain.

The default local roots are `models/` for fitted packages, `reports/` for aggregate results,
`private/` for sample-level prediction and error-review evidence, `data/manifests/` for dataset and
split authorities, `mlflow.db` and `mlartifacts/` for operational tracking, and `outbox/` for
restricted campaign exports. These locations are storage conventions, not discovery mechanisms:
scientific consumers receive explicit immutable identities and validate them recursively.
