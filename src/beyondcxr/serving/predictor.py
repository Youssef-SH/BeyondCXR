"""Strict reconstruction and inference for the ordered primary gated ensemble."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.data.symile_preprocess import SymileLabEcdfTransformer
from beyondcxr.serving.authority import (
    ENSEMBLE_POLICY,
    ValidatedServingAuthority,
    load_serving_authority,
    require_serving_authority,
)
from beyondcxr.serving.preprocessing import ValidatedServingInput
from beyondcxr.training.symile_final_packages import (
    FinalPackageConfig,
    ValidatedFinalNeuralRuntime,
    load_final_lab_preprocessor,
    load_final_neural_model,
    load_final_package_config,
    require_final_neural_runtime,
)
from beyondcxr.training.symile_statistics import sigmoid
from beyondcxr.utils.immutable_json import immutable_json, json_projection

if TYPE_CHECKING:
    from beyondcxr.training.symile_campaign import RestoredCampaignClosure


class SymileServingPredictor:
    """Loaded, immutable three-member research predictor."""

    def __init__(self, authority: ValidatedServingAuthority, *, device: str = "cpu") -> None:
        validated = require_serving_authority(authority)
        models = tuple(load_final_neural_model(package) for package in validated.packages)
        preprocessors = tuple(
            load_final_lab_preprocessor(package) for package in validated.packages
        )
        config = load_final_package_config(validated.packages[0])
        self._initialize(
            validated, models, preprocessors, config, device=device, release_inputs=None
        )

    def _initialize(
        self,
        authority: ValidatedServingAuthority,
        models: tuple[torch.nn.Module, ...],
        preprocessors: tuple[SymileLabEcdfTransformer, ...],
        config: FinalPackageConfig,
        *,
        device: str,
        release_inputs: tuple[Path, Path, Path] | None,
    ) -> None:
        self.authority = require_serving_authority(authority)
        if device not in {"cpu", "cuda"}:
            raise ManifestBuildError("Serving device must be explicitly cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise ManifestBuildError("Requested serving CUDA device is unavailable")
        self.device = torch.device(device)
        self._models = tuple(model.to(self.device).eval() for model in models)
        self._preprocessors = preprocessors
        self._active = True
        self._process_id = os.getpid()
        self._authority_directory = self.authority.directory.resolve()
        self._release_inputs = release_inputs
        self.transform = StandardCxrTransform(
            training=False,
            policy_version=str(config.preprocessing["cxr_transform_policy"]),
            image_size=int(config.family_parameters["image_size"]),
            rotation_degrees=float(config.augmentation["rotation_degrees"]),
            translation_fraction=float(config.augmentation["translation_fraction"]),
            brightness_jitter=float(config.augmentation["brightness_jitter"]),
            contrast_jitter=float(config.augmentation["contrast_jitter"]),
        )
        self._model_info = immutable_json(serving_model_info(self.authority))

    @classmethod
    def from_validated_runtime(
        cls,
        authority: ValidatedServingAuthority,
        members: tuple[ValidatedFinalNeuralRuntime, ...],
        *,
        deployment_provenance: str | Path,
        result_binding: str | Path,
        release_archive: str | Path,
        device: str = "cpu",
    ) -> SymileServingPredictor:
        """Create a predictor from the fitted objects validated for this authority."""
        validated = require_serving_authority(authority)
        members = tuple(require_final_neural_runtime(member) for member in members)
        if tuple(member.package.package_id for member in members) != tuple(
            package.package_id for package in validated.packages
        ):
            raise ManifestBuildError("Serving runtime members differ from the authority")
        predictor = object.__new__(cls)
        predictor._initialize(
            validated,
            tuple(member.model for member in members),
            tuple(member.lab_preprocessor for member in members),
            members[0].config,
            device=device,
            release_inputs=tuple(
                Path(path).resolve()
                for path in (deployment_provenance, result_binding, release_archive)
            ),
        )
        return predictor

    @classmethod
    def load(
        cls,
        authority_path: str | Path,
        *,
        deployment_provenance: str | Path,
        result_binding: str | Path,
        release_archive: str | Path,
        device: str = "cpu",
        restored_campaign: RestoredCampaignClosure | None = None,
    ) -> SymileServingPredictor:
        authority, members = load_serving_authority(
            authority_path,
            deployment_provenance=deployment_provenance,
            result_binding=result_binding,
            release_archive=release_archive,
            restored_campaign=restored_campaign,
        )
        return cls.from_validated_runtime(
            authority,
            members,
            deployment_provenance=deployment_provenance,
            result_binding=result_binding,
            release_archive=release_archive,
            device=device,
        )

    def close(self) -> None:
        """Revoke this process-local runtime at the end of its ownership lifetime."""
        self._active = False

    def require_authority(self, authority_path: str | Path) -> None:
        if not self._active or self._process_id != os.getpid():
            raise ManifestBuildError("Serving predictor lifetime has ended")
        if Path(authority_path).resolve() != self._authority_directory:
            raise ManifestBuildError("Serving predictor belongs to a different authority")

    def require_release_inputs(
        self,
        authority_path: str | Path,
        deployment_provenance: str | Path,
        result_binding: str | Path,
        release_archive: str | Path,
    ) -> None:
        self.require_authority(authority_path)
        observed = tuple(
            Path(path).resolve()
            for path in (deployment_provenance, result_binding, release_archive)
        )
        if self._release_inputs is None or observed != self._release_inputs:
            raise ManifestBuildError("Serving predictor belongs to different release evidence")

    def _require_active(self) -> None:
        if not self._active or self._process_id != os.getpid():
            raise ManifestBuildError("Serving predictor lifetime has ended")

    def predict(self, request: ValidatedServingInput) -> float:
        self._require_active()
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
        self._require_active()
        return json_projection(self._model_info)


def serving_model_info(authority: ValidatedServingAuthority) -> dict[str, object]:
    """Project one validated serving authority onto the public model-info contract."""
    validated = require_serving_authority(authority)
    manifest = validated.manifest
    return {
        "task": manifest["task"]["task_id"],
        "positive_class": dict(manifest["positive_class"]),
        "serving_authority_id": validated.authority_id,
        "model_package_ids": [item["package_id"] for item in manifest["members"]],
        "seeds": list(manifest["ordered_seeds"]),
        "family": manifest["family"],
        "ensemble_policy": ENSEMBLE_POLICY,
        "input_contract": dict(manifest["input_contract"]),
        "preprocessing": dict(manifest["preprocessing"]),
        "operating_thresholds": dict(manifest["primary_thresholds"]),
        "global_result_id": manifest["global_result"]["global_result_id"],
        "warning": manifest["warning"],
    }


def mean_logit_probability(logits: list[float] | tuple[float, ...]) -> float:
    """Average exactly three finite member logits and apply sigmoid once."""
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Serving ensemble requires exactly three finite raw logits")
    return float(sigmoid([float(values.mean())])[0])
