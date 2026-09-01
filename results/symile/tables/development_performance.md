Repeated-CV development performance.

| Evidence | Model / comparison | Repeat seed | AUROC | Average Precision | Brier score | ΔAUROC | ΔAP | ΔBrier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OOF repeat | Labs Logistic Regression | 17 | 0.624 | 0.566 | 0.240 | — | — | — |
| OOF repeat | Labs Logistic Regression | 42 | 0.627 | 0.576 | 0.238 | — | — | — |
| OOF repeat | Labs Logistic Regression | 2026 | 0.625 | 0.571 | 0.239 | — | — | — |
| OOF repeat | Labs LightGBM | 17 | 0.588 | 0.543 | 0.245 | — | — | — |
| OOF repeat | Labs LightGBM | 42 | 0.598 | 0.550 | 0.243 | — | — | — |
| OOF repeat | Labs LightGBM | 2026 | 0.596 | 0.552 | 0.243 | — | — | — |
| OOF repeat | CXR | 17 | 0.710 | 0.670 | 0.217 | — | — | — |
| OOF repeat | CXR | 42 | 0.717 | 0.673 | 0.215 | — | — | — |
| OOF repeat | CXR | 2026 | 0.719 | 0.682 | 0.214 | — | — | — |
| Mean-logit OOF ensemble | CXR | — | 0.718 | 0.679 | 0.214 | — | — | — |
| OOF repeat | CXR + labs concat | 17 | 0.719 | 0.682 | 0.214 | — | — | — |
| OOF repeat | CXR + labs concat | 42 | 0.719 | 0.682 | 0.214 | — | — | — |
| OOF repeat | CXR + labs concat | 2026 | 0.716 | 0.678 | 0.214 | — | — | — |
| Mean-logit OOF ensemble | CXR + labs concat | — | 0.723 | 0.688 | 0.212 | — | — | — |
| OOF repeat | CXR + labs gated | 17 | 0.714 | 0.673 | 0.215 | — | — | — |
| OOF repeat | CXR + labs gated | 42 | 0.725 | 0.687 | 0.212 | — | — | — |
| OOF repeat | CXR + labs gated | 2026 | 0.717 | 0.677 | 0.214 | — | — | — |
| Mean-logit OOF ensemble | CXR + labs gated | — | 0.723 | 0.686 | 0.212 | — | — | — |
| OOF repeat | CXR + labs gated without explicit observedness indicators | 17 | 0.713 | 0.670 | 0.216 | — | — | — |
| OOF repeat | CXR + labs gated without explicit observedness indicators | 42 | 0.717 | 0.677 | 0.215 | — | — | — |
| OOF repeat | CXR + labs gated without explicit observedness indicators | 2026 | 0.717 | 0.674 | 0.214 | — | — | — |
| Mean-logit OOF ensemble | CXR + labs gated without explicit observedness indicators | — | 0.721 | 0.680 | 0.213 | — | — | — |
| ECG extension OOF repeat | CXR + labs + ECG gated | 17 | 0.710 | 0.657 | 0.217 | — | — | — |
| ECG extension OOF repeat | CXR + labs + ECG gated | 42 | 0.717 | 0.671 | 0.215 | — | — | — |
| ECG extension OOF repeat | CXR + labs + ECG gated | 2026 | 0.705 | 0.632 | 0.220 | — | — | — |
| ECG extension mean-logit OOF ensemble | CXR + labs + ECG gated | — | 0.717 | 0.660 | 0.215 | — | — | — |
| OOF repeat paired effect | CXR + labs concat − CXR | 17 | — | — | — | 0.008 | 0.011 | -0.003 |
| OOF repeat paired effect | CXR + labs concat − CXR | 42 | — | — | — | 0.002 | 0.009 | -0.000 |
| OOF repeat paired effect | CXR + labs concat − CXR | 2026 | — | — | — | -0.003 | -0.004 | 0.000 |
| Mean-logit OOF paired effect | CXR + labs concat − CXR | — | — | — | — | 0.005 | 0.009 | -0.002 |
| OOF repeat paired effect | CXR + labs gated − CXR — PRIMARY | 17 | — | — | — | 0.004 | 0.003 | -0.001 |
| OOF repeat paired effect | CXR + labs gated − CXR — PRIMARY | 42 | — | — | — | 0.008 | 0.014 | -0.003 |
| OOF repeat paired effect | CXR + labs gated − CXR — PRIMARY | 2026 | — | — | — | -0.002 | -0.004 | 0.000 |
| Mean-logit OOF paired effect | CXR + labs gated − CXR — PRIMARY | — | — | — | — | 0.005 | 0.006 | -0.003 |
| OOF repeat paired effect | CXR + labs gated − CXR + labs concat | 17 | — | — | — | -0.004 | -0.009 | 0.002 |
| OOF repeat paired effect | CXR + labs gated − CXR + labs concat | 42 | — | — | — | 0.006 | 0.005 | -0.003 |
| OOF repeat paired effect | CXR + labs gated − CXR + labs concat | 2026 | — | — | — | 0.000 | -0.001 | -0.000 |
| Mean-logit OOF paired effect | CXR + labs gated − CXR + labs concat | — | — | — | — | 0.000 | -0.002 | -0.000 |
| OOF repeat paired effect | CXR + labs + ECG gated − CXR + labs gated | 17 | — | — | — | -0.005 | -0.016 | 0.002 |
| OOF repeat paired effect | CXR + labs + ECG gated − CXR + labs gated | 42 | — | — | — | -0.007 | -0.016 | 0.004 |
| OOF repeat paired effect | CXR + labs + ECG gated − CXR + labs gated | 2026 | — | — | — | -0.012 | -0.045 | 0.006 |
| Mean-logit OOF paired effect | CXR + labs + ECG gated − CXR + labs gated | — | — | — | — | -0.007 | -0.025 | 0.003 |

*Point estimates only; no repeated-CV confidence intervals are defined.*
