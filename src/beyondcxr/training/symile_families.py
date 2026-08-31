"""Canonical Symile development, package, and held-out predictor vocabulary."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from beyondcxr.data.errors import ManifestBuildError

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
DEVELOPMENT_FAMILY_ORDER = (*SYMILE_CORE_DEVELOPMENT_FAMILIES, SYMILE_ECG_GATED_FAMILY)
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
SINGLE_MEMBER_REDUCTION = "single_member"
NEURAL_MEMBER_REDUCTION = "mean_member_logits_then_sigmoid"


def final_predictor_reduction_policy() -> list[dict[str, object]]:
    """Return the package-independent six-view reduction contract."""
    return [
        {
            "family_id": family,
            "member_seeds": (
                [FINAL_TABULAR_SEED]
                if int(policy["members"]) == 1
                else list(FINAL_NEURAL_MEMBER_SEEDS)
            ),
            "reduction": (
                SINGLE_MEMBER_REDUCTION if int(policy["members"]) == 1 else NEURAL_MEMBER_REDUCTION
            ),
        }
        for family, policy in FINAL_PACKAGE_POLICY.items()
    ]


def final_predictor_view_policy(
    packages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return the exact six predictor views over canonical final-package membership."""
    expected_families = [
        family
        for family, policy in FINAL_PACKAGE_POLICY.items()
        for _ in range(int(policy["members"]))
    ]
    if (
        len(packages) != FINAL_PACKAGE_COUNT
        or any(set(item) != {"package_id", "family_id", "seed"} for item in packages)
        or [item["family_id"] for item in packages] != expected_families
    ):
        raise ManifestBuildError("Symile final packages do not define canonical predictor views")
    views: list[dict[str, object]] = []
    offset = 0
    for reduction in final_predictor_reduction_policy():
        family = str(reduction["family_id"])
        policy = FINAL_PACKAGE_POLICY[family]
        count = int(policy["members"])
        members = packages[offset : offset + count]
        offset += count
        expected_seeds = reduction["member_seeds"]
        if [item["seed"] for item in members] != expected_seeds:
            raise ManifestBuildError("Symile predictor-view seed membership is invalid")
        views.append(
            {
                "family_id": family,
                "member_package_ids": [item["package_id"] for item in members],
                "member_seeds": expected_seeds,
                "reduction": reduction["reduction"],
            }
        )
    return views
