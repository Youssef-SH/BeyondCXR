Held-out test performance and paired effects.

| Evidence | Model / comparison | AUROC | Average Precision | Brier score | ΔAUROC | ΔAUROC 95% CI lower | ΔAUROC 95% CI upper | ΔAP | ΔAP 95% CI lower | ΔAP 95% CI upper | ΔBrier | ΔBrier 95% CI lower | ΔBrier 95% CI upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Predictor | Labs Logistic Regression | 0.626 | 0.665 | 0.242 | — | — | — | — | — | — | — | — | — |
| Predictor | Labs LightGBM | 0.680 | 0.688 | 0.231 | — | — | — | — | — | — | — | — | — |
| Predictor | CXR | 0.781 | 0.823 | 0.201 | — | — | — | — | — | — | — | — | — |
| Predictor | CXR + labs concat | 0.778 | 0.811 | 0.199 | — | — | — | — | — | — | — | — | — |
| Predictor | CXR + labs gated | 0.777 | 0.809 | 0.199 | — | — | — | — | — | — | — | — | — |
| Predictor | CXR + labs + ECG gated | 0.776 | 0.816 | 0.202 | — | — | — | — | — | — | — | — | — |
| Paired effect | CXR + labs concat − CXR | — | — | — | -0.003 | -0.018 | 0.012 | -0.012 | -0.047 | 0.021 | -0.002 | -0.008 | 0.004 |
| Paired effect | CXR + labs gated − CXR — PRIMARY | — | — | — | -0.003 | -0.018 | 0.009 | -0.014 | -0.049 | 0.017 | -0.002 | -0.007 | 0.004 |
| Paired effect | CXR + labs gated − CXR + labs concat | — | — | — | -0.000 | -0.006 | 0.005 | -0.002 | -0.010 | 0.002 | 0.000 | -0.001 | 0.001 |
| Paired effect | CXR + labs + ECG gated − CXR + labs gated | — | — | — | -0.001 | -0.014 | 0.012 | 0.007 | -0.015 | 0.035 | 0.003 | -0.001 | 0.006 |

*Effects are candidate minus comparator. Intervals are paired subject-level bootstrap 95% CIs; negative ΔBrier favors the candidate.*
