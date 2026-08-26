# Model card — primary CXR + labs gated ensemble

> Research use only. BeyondCXR is not intended for clinical decision-making, is not a medical
> device, does not replace physician judgment, and predicts a radiology-derived Pneumonia finding
> rather than confirmed infectious pneumonia.

## Model and task

The served predictor is the ordered three-member CXR-plus-laboratories missingness-aware gated
ensemble for `pneumonia_strict`. Positive means `Pneumonia = 1`; explicit zero is negative;
uncertain and missing labels are excluded.

<!-- BEYONDCXR_RESULTS_START -->
Awaiting formal Symile execution. Bound scientific identities, result evidence, thresholds, and
provenance are inserted only from validated scientific state.
<!-- BEYONDCXR_RESULTS_END -->

## Intended and excluded uses

The model supports controlled research into multimodal prediction and reproducibility. It is not
intended for diagnosis, triage, treatment, patient management, autonomous decisions, prospective
deployment, or replacement of professional judgment.

## Inputs and preprocessing

The input is one frontal CXR from the declared Symile input domain, submitted as JPEG, plus exactly
50 raw laboratory values. AP/PA is validation metadata. Null laboratories are unobserved; each package
applies its fitted full-development ECDF and observedness policy.

The image follows deterministic Symile evaluation spatial preprocessing and the package-bound
deterministic TorchXRayVision evaluation transform. The DenseNet121 encoder uses the declared
`densenet121-res224-chex` initialization during fitting; serving reconstructs package state without
fetching upstream weights.

## Architecture, development, and full-development fitting

CXR and laboratory encoders emit 256-dimensional representations. A learned feature-wise gate
combines both modalities using laboratory observedness. Members use seeds 17, 42, and 2026;
inference averages raw logits in that order and applies sigmoid once.

Development uses one immutable patient-grouped CV assignment comprising three five-fold repeats.
Neural selection uses deterministic inner validation. Terminal epoch budgets are medians of the 15
development selections; ensemble members then fit all development admissions for that fixed
budget. Test access requires all required packages, the pre-test freeze, and its same-freeze
opening record.

## Output and evaluation

`POST /predict` returns one raw probability and no label. Development-derived thresholds appear in
`/model-info` as research metadata. No post-hoc recalibration is applied; raw probability
reliability is assessed descriptively.

Development reports repeat-level and mean-logit OOF AUROC, Average Precision, and Brier score
without repeated-CV confidence intervals. Held-out test evaluation reports raw probability metrics,
descriptive reliability, and paired subject-level bootstrap uncertainty. Development-only
prespecified subgroup metrics are reported for supported strata; unsupported predefined strata
remain explicitly marked unavailable.

Held-out evaluation is internal testing within the Symile source population. No external
validation is claimed.

## Limitations, privacy, and distribution

The cohort is retrospective and credentialed; the held-out test size limits confirmatory precision;
subgroup analyses are development-only, prespecified, and support-gated; missingness may encode care
processes; and performance may not transfer across populations or acquisition systems. Learned
gates are not causal explanations. The service retains no request or prediction. Dataset agreements
govern trained-artifact distribution independently of the code license.

## Reproducibility

The generated result region records the bound scientific coordinates together with preservation
and reproducibility witnesses, the pretrained scientific weight identity and materialization
witnesses, and the development subgroup characterization. The private serving authority separately
records release-candidate provenance. See [reproducibility](reproducibility.md) and
[controlled serving](serving.md). RSNA is separate supporting project context, not evidence for
this served predictor.
