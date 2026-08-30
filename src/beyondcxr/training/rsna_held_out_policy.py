"""Prospectively fixed scientific policies for RSNA held-out work."""

RSNA_PREDICTION_POLICY = {
    "scope": "test",
    "package_scope": "single_package",
    "output": "raw_probability",
}
LOCALIZATION_POLICY_VERSION = "rsna-gradcam-union-box-v1"
LOCALIZATION_THRESHOLD_POLICY = "validation_youden_j_seed_specific"
QUALITATIVE_POLICY_VERSION = "sha256-stratum-order-v1"
GRADCAM_TARGET = "encoder.backbone.features"


def localization_policy() -> dict[str, str]:
    """Return the exact prospectively fixed localization policy."""
    return {
        "policy_version": LOCALIZATION_POLICY_VERSION,
        "threshold_policy": LOCALIZATION_THRESHOLD_POLICY,
        "qualitative_selection_policy": QUALITATIVE_POLICY_VERSION,
        "target": GRADCAM_TARGET,
        "cohort": "canonical_positive_held_out_cases",
    }
