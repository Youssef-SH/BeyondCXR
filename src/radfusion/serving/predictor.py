"""Strict reconstruction and inference for the ordered primary gated ensemble."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from radfusion.data.cxr_transforms import StandardCxrTransform
from radfusion.data.errors import ManifestBuildError
from radfusion.serving.authority import (
    ENSEMBLE_POLICY,
    ValidatedServingAuthority,
    require_serving_authority,
    validate_serving_authority,
)
from radfusion.serving.preprocessing import ValidatedServingInput
from radfusion.training.symile_final_packages import (
    load_final_lab_preprocessor,
    load_final_neural_model,
    load_final_package_config,
)
from radfusion.training.symile_statistics import sigmoid


class SymileServingPredictor:
    """Loaded, immutable three-member research predictor."""

    def __init__(self, authority: ValidatedServingAuthority, *, device: str = "cpu") -> None:
        self.authority = require_serving_authority(authority)
        if device not in {"cpu", "cuda"}:
            raise ManifestBuildError("Serving device must be explicitly cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise ManifestBuildError("Requested serving CUDA device is unavailable")
        self.device = torch.device(device)
        self._models = tuple(
            load_final_neural_model(package).to(self.device).eval()
            for package in self.authority.packages
        )
        self._preprocessors = tuple(
            load_final_lab_preprocessor(package) for package in self.authority.packages
        )
        config = load_final_package_config(self.authority.packages[0])
        self.transform = StandardCxrTransform(
            training=False,
            policy_version=str(config.preprocessing["cxr_transform_policy"]),
            image_size=int(config.family_parameters["image_size"]),
            rotation_degrees=float(config.augmentation["rotation_degrees"]),
            translation_fraction=float(config.augmentation["translation_fraction"]),
            brightness_jitter=float(config.augmentation["brightness_jitter"]),
            contrast_jitter=float(config.augmentation["contrast_jitter"]),
        )

    @classmethod
    def load(
        cls,
        authority_path: str | Path,
        *,
        package_root: str | Path,
        device: str = "cpu",
    ) -> SymileServingPredictor:
        authority = validate_serving_authority(authority_path, package_root=package_root)
        return cls(authority, device=device)

    def predict(self, request: ValidatedServingInput) -> float:
        image = request.image.unsqueeze(0).to(self.device)
        logits: list[float] = []
        with torch.inference_mode():
            for model, preprocessor in zip(self._models, self._preprocessors, strict=True):
                structured_array = preprocessor.transform(request.laboratory_frame)
                structured = torch.from_numpy(np.asarray(structured_array, dtype=np.float32)).to(
                    self.device
                )
                output = model(image, structured)
                if output.shape != (1,) or not torch.isfinite(output).all():
                    raise RuntimeError("Serving model produced an invalid raw logit")
                logits.append(float(output.item()))
        probability = mean_logit_probability(logits)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError("Serving ensemble produced an invalid probability")
        return probability

    def model_info(self) -> dict[str, object]:
        manifest = self.authority.manifest
        return {
            "task": manifest["task"]["task_id"],
            "positive_class": dict(manifest["positive_class"]),
            "serving_authority_id": self.authority.authority_id,
            "model_package_ids": [item["package_id"] for item in manifest["members"]],
            "seeds": list(manifest["ordered_seeds"]),
            "family": manifest["family"],
            "ensemble_policy": ENSEMBLE_POLICY,
            "input_contract": dict(manifest["input_contract"]),
            "preprocessing": dict(manifest["preprocessing"]),
            "operating_thresholds": dict(manifest["primary_thresholds"]),
            "global_result_id": manifest["global_result"]["global_result_id"],
            "science_git_commit": manifest["science_execution"]["git_commit"],
            "serving_release_git_commit": manifest["serving_release"]["git_commit"],
            "warning": manifest["warning"],
        }


def mean_logit_probability(logits: list[float] | tuple[float, ...]) -> float:
    """Apply the frozen serving ensemble arithmetic to exactly three finite logits."""
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Serving ensemble requires exactly three finite raw logits")
    return float(sigmoid([float(values.mean())])[0])
