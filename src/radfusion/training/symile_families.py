"""Canonical Symile core-development family vocabulary."""

SYMILE_CORE_DEVELOPMENT_FAMILIES = (
    "labs_logistic",
    "labs_lightgbm",
    "cxr_densenet",
    "cxr_labs_concat",
    "cxr_labs_gated",
    "cxr_labs_gated_no_observedness",
)
SYMILE_TABULAR_FAMILIES = frozenset({"labs_logistic", "labs_lightgbm"})
SYMILE_FUSION_FAMILIES = frozenset(
    {"cxr_labs_concat", "cxr_labs_gated", "cxr_labs_gated_no_observedness"}
)
SYMILE_NEURAL_FAMILIES = frozenset({"cxr_densenet", *SYMILE_FUSION_FAMILIES})
