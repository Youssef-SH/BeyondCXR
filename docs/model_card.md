# Model card — primary CXR + labs gated ensemble

> Research use only. BeyondCXR is not intended for clinical decision-making, is not a medical
> device, does not replace physician judgment, and predicts a radiology-derived Pneumonia finding
> rather than confirmed infectious pneumonia.

## Model and task

The served predictor is the ordered three-member CXR-plus-laboratories missingness-aware gated
ensemble for `pneumonia_strict`. Positive means `Pneumonia = 1`; explicit zero is negative;
uncertain and missing labels are excluded.

<!-- BEYONDCXR_RESULTS_START -->
### Development OOF evidence

- CXR AUROC: 0.718
- Primary gated AUROC: 0.723
- Gated minus CXR ΔAUROC: 0.005

### Development subgroup characterization

- Prespecified support requires n ≥ 100, positives ≥ 20, and negatives ≥ 20.
- age:18-49: n=348, positives=172, negatives=176; gated minus CXR ΔAUROC=0.003, ΔAP=0.005, ΔBrier=-0.007.
- age:50-64: n=586, positives=249, negatives=337; gated minus CXR ΔAUROC=-0.001, ΔAP=-0.013, ΔBrier=-0.001.
- age:65-79: n=764, positives=337, negatives=427; gated minus CXR ΔAUROC=0.011, ΔAP=0.018, ΔBrier=-0.002.
- age:80-plus: n=670, positives=346, negatives=324; gated minus CXR ΔAUROC=0.005, ΔAP=0.005, ΔBrier=-0.002.
- observed_labs:1-29: n=601, positives=233, negatives=368; gated minus CXR ΔAUROC=0.003, ΔAP=0.014, ΔBrier=-0.003.
- observed_labs:30-34: n=606, positives=287, negatives=319; gated minus CXR ΔAUROC=0.007, ΔAP=0.010, ΔBrier=-0.004.
- observed_labs:35-40: n=731, positives=371, negatives=360; gated minus CXR ΔAUROC=0.006, ΔAP=0.003, ΔBrier=-0.004.
- observed_labs:41-50: n=430, positives=213, negatives=217; gated minus CXR ΔAUROC=-0.000, ΔAP=-0.008, ΔBrier=0.003.
- sex:F: n=1102, positives=513, negatives=589; gated minus CXR ΔAUROC=0.011, ΔAP=0.016, ΔBrier=-0.004.
- sex:M: n=1266, positives=591, negatives=675; gated minus CXR ΔAUROC=0.000, ΔAP=-0.002, ΔBrier=-0.001.
- view:AP: n=1894, positives=912, negatives=982; gated minus CXR ΔAUROC=0.007, ΔAP=0.005, ΔBrier=-0.003.
- view:PA: n=474, positives=192, negatives=282; gated minus CXR ΔAUROC=-0.002, ΔAP=0.012, ΔBrier=-0.001.

### Held-out test evidence

- CXR AUROC: 0.781
- Primary AUROC: 0.777
- Primary Average Precision: 0.809
- Primary Brier score: 0.199
- Calibration slope: 1.196
- Calibration intercept: 0.534
- Gated minus CXR ΔAUROC: -0.003 (95% CI -0.018 to 0.009)
- Development Youden-J threshold (metadata): 0.447978
- Development 90%-sensitivity threshold (metadata): 0.263

### Validated scientific coordinates

- Ordered primary packages: `final-package-b47f6f5f415c0a01bc186dd067e98d2ef8fc45db8f71498d26034d3c2832523e`, `final-package-92c1f120581feafd739eb42de9801319e953e3b5d7ee63f26b967a80216fcb1a`, `final-package-d7c14dfd314bc9282f1688e51317d169d35258540cc5e698a00c4cb4daaf4607`
- Bundle: `bundle-e3d4f56869a95e7ca3032b8fa9c8023a038b1692232d20e6a353c1d9b7ba1df8`
- CV assignment: `cv-assignment-23d3c8385a2072e79d8658e242944eb82e22f141aa62584b396e5cb8bf3b90f3`
- Pre-test freeze: `pretest-freeze-d33575d53edb1b96321ad9d729a6982edb345592eb62c975aab981f17672a6cb`
- Global result: `global-result-a70e02bd66a173eb7f1c89db29ba857f4f89302706fb193cdb3cc692b9a0d4e0`
- Science Git commit: `441e238f3febfa5225b8a3e61b466bca4d2b9cf6`

### Pretrained scientific weight

- Declared identity: `densenet121-res224-chex`
- Stable identifier: `https://github.com/mlmed/torchxrayvision/releases/download/v1/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt`
- Scientific content SHA-256: `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4`
- Seed 17 materialization: `chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt`, 28381998 bytes, SHA-256 `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4`
- Seed 42 materialization: `chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt`, 28381998 bytes, SHA-256 `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4`
- Seed 2026 materialization: `chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt`, 28381998 bytes, SHA-256 `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4`

### Reproducibility and fixity witnesses

- Science lock SHA-256: `c747143c8f1ce9c6a1dc007d71c610efc1bd7a7c377b855835403f50118c9ebd`
- Export-manifest SHA-256: `86d8e2d50dad8bb76468102644c4b6428cad94a2b8d9b0220112d1b700d698a7`

Serving uses the ordered three-member mean-logit ensemble. Thresholds are research metadata and are not emitted as decisions by `/predict`.
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

CXR and laboratory representations are projected into a shared 256-dimensional fusion space. A learned feature-wise gate
combines both modalities using laboratory observedness. Members use seeds 17, 42, and 2026;
inference averages raw logits in that order and applies sigmoid once.

Development uses one immutable patient-grouped CV assignment comprising three five-fold repeats.
Neural selection uses deterministic inner validation. Terminal epoch budgets are medians of the 15
development selections; ensemble members then fit all development admissions for that fixed
budget. Test access requires all required packages, the pre-test freeze, matching execution,
and identical local and persistent external opening records.

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

The CXR-plus-laboratories predictor is evaluated in a cohort that also requires ECG availability;
its results do not establish performance across all admissions with CXR and laboratory data.

The cohort is retrospective and credentialed; the held-out test size limits confirmatory precision;
subgroup analyses are development-only, prespecified, and support-gated; missingness may encode care
processes; and performance may not transfer across populations or acquisition systems. Learned
gates are not causal explanations. The service retains no request or prediction. Dataset agreements
govern trained-artifact distribution independently of the code license.

## Reproducibility

The generated result region records scientific coordinates, validated aggregate evidence, and
separately labeled operational and transport witnesses.
Preservation authenticates private evidence bytes, formal execution records implementation and
runtime provenance, and the deployment receipt records release-candidate provenance. See
[reproducibility](reproducibility.md) and
[controlled serving](serving.md). RSNA is separate supporting project context, not evidence for
this served predictor.
