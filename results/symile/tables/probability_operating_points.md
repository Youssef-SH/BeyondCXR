Raw probability quality and primary gated operating points.

| Section | Model / operating point | Brier score | Calibration slope | Calibration intercept | Threshold | Precision | Sensitivity | Specificity | F1 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Development OOF raw probability | CXR | 0.214 | 1.121 | 0.050 | — | — | — | — | — | — | — | — | — |
| Held-out descriptive raw probability | CXR | 0.201 | 1.293 | 0.557 | — | — | — | — | — | — | — | — | — |
| Development OOF raw probability | CXR + labs gated | 0.212 | 0.929 | 0.004 | — | — | — | — | — | — | — | — | — |
| Held-out descriptive raw probability | CXR + labs gated | 0.199 | 1.196 | 0.534 | — | — | — | — | — | — | — | — | — |
| Held-out behavior at development-derived operating point | Youden-J | — | — | — | 0.448 | 0.792 | 0.655 | 0.808 | 0.717 | 42 | 10 | 20 | 38 |
| Held-out behavior at development-derived operating point | 90% sensitivity target | — | — | — | 0.263 | 0.636 | 0.845 | 0.462 | 0.726 | 24 | 28 | 9 | 49 |

*Thresholds were derived from development OOF evidence and applied unchanged to held-out probabilities. No post-hoc recalibration; /predict emits no thresholded decision.*
