"""Canonical Symile development and final-package family vocabulary."""

from types import MappingProxyType

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

SYMILE_ECG_GATED_FAMILY = "cxr_labs_ecg_gated"
SYMILE_ECG_FUSION_FAMILIES = frozenset({SYMILE_ECG_GATED_FAMILY})
SYMILE_ALL_FUSION_FAMILIES = SYMILE_FUSION_FAMILIES | SYMILE_ECG_FUSION_FAMILIES
SYMILE_ALL_NEURAL_FAMILIES = SYMILE_NEURAL_FAMILIES | SYMILE_ECG_FUSION_FAMILIES
FINAL_NEURAL_MEMBER_SEEDS = (17, 42, 2026)
FINAL_TABULAR_SEED = 42
FINAL_PACKAGE_POLICY = MappingProxyType(
    {
        "labs_logistic": MappingProxyType(
            {"package_kind": "tabular", "members": 1, "fusion": False}
        ),
        "labs_lightgbm": MappingProxyType(
            {"package_kind": "tabular", "members": 1, "fusion": False}
        ),
        "cxr_densenet": MappingProxyType(
            {"package_kind": "neural", "members": len(FINAL_NEURAL_MEMBER_SEEDS), "fusion": False}
        ),
        "cxr_labs_concat": MappingProxyType(
            {"package_kind": "neural", "members": len(FINAL_NEURAL_MEMBER_SEEDS), "fusion": True}
        ),
        "cxr_labs_gated": MappingProxyType(
            {"package_kind": "neural", "members": len(FINAL_NEURAL_MEMBER_SEEDS), "fusion": True}
        ),
        SYMILE_ECG_GATED_FAMILY: MappingProxyType(
            {"package_kind": "neural", "members": len(FINAL_NEURAL_MEMBER_SEEDS), "fusion": True}
        ),
    }
)
FINAL_NEURAL_FAMILIES = tuple(
    family for family, policy in FINAL_PACKAGE_POLICY.items() if policy["package_kind"] == "neural"
)
FINAL_TABULAR_FAMILIES = tuple(
    family for family, policy in FINAL_PACKAGE_POLICY.items() if policy["package_kind"] == "tabular"
)
FINAL_FUSION_FAMILIES = tuple(
    family for family, policy in FINAL_PACKAGE_POLICY.items() if policy["fusion"]
)
FINAL_PACKAGE_COUNT = sum(int(policy["members"]) for policy in FINAL_PACKAGE_POLICY.values())
