"""Define the standard TorchXRayVision CXR encoder and binary model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn

from radfusion.data.hashing import sha256_file
from radfusion.training.config import FamilyConfig

CXR_TRAINABILITY_SCOPES = frozenset({"frozen", "terminal", "all"})


@dataclass(frozen=True)
class PretrainedWeightIdentity:
    """Local byte identity of one materialized pretrained weight file."""

    declared_name: str
    stable_identifier: str
    cache_filename: str
    byte_size: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        """Return serializable pretrained-weight provenance."""
        return asdict(self)


class StandardCxrEncoder(nn.Module):
    """Expose pooled DenseNet121 features through a stable encoding boundary."""

    def __init__(
        self,
        *,
        weights: str | None = "densenet121-res224-chex",
        expected_embedding_dimension: int = 1024,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if weights not in {None, "densenet121-res224-chex"}:
            raise ValueError("Standard CXR encoder received an unsupported weight identity")
        if expected_embedding_dimension != 1024 or image_size != 224:
            raise ValueError("Standard CXR encoder requires dimensions 1024 and 224")
        self.weights = weights
        self.embedding_dimension = expected_embedding_dimension
        self.image_size = image_size
        import torchxrayvision as xrv

        self.backbone = xrv.models.DenseNet(weights=weights)
        self.backbone.op_threshs = None

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return one 1024-dimensional pooled embedding per image."""
        _validate_images(images, self.image_size)
        embeddings = self.backbone.features2(images)
        _validate_embeddings(embeddings, len(images), self.embedding_dimension)
        return embeddings

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return pooled embeddings."""
        return self.encode(images)


def set_cxr_encoder_trainability(encoder: nn.Module, scope: str) -> None:
    """Apply the frozen, terminal-block, or full DenseNet trainability contract."""
    if not isinstance(encoder, nn.Module) or scope not in CXR_TRAINABILITY_SCOPES:
        raise ValueError("CXR encoder trainability scope is invalid")
    selected = 0
    for name, parameter in encoder.named_parameters():
        trainable = scope == "all" or (
            scope == "terminal"
            and (
                name.startswith("backbone.features.denseblock4.")
                or name.startswith("backbone.features.norm5.")
            )
        )
        parameter.requires_grad = trainable
        selected += int(trainable)
    if scope != "frozen" and selected == 0:
        raise ValueError("CXR encoder trainability scope selected no parameters")


def set_cxr_encoder_training_mode(encoder: nn.Module, scope: str) -> None:
    """Keep frozen DenseNet modules in evaluation mode during terminal fine-tuning."""
    if scope == "all":
        encoder.train()
        return
    if scope == "frozen":
        encoder.eval()
        return
    if scope != "terminal":
        raise ValueError("CXR encoder training-mode scope is invalid")
    modules = dict(encoder.named_modules())
    try:
        denseblock4 = modules["backbone.features.denseblock4"]
        norm5 = modules["backbone.features.norm5"]
    except KeyError as exc:
        raise ValueError("CXR encoder does not expose the terminal DenseNet modules") from exc
    encoder.eval()
    denseblock4.train()
    norm5.train()


class CxrBinaryClassifier(nn.Module):
    """Combine a reusable CXR encoder with a separate one-logit head."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        embedding_dimension: int = 1024,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("CXR encoder must be a torch.nn.Module")
        if isinstance(embedding_dimension, bool) or not isinstance(embedding_dimension, int):
            raise ValueError("embedding_dimension must be an integer")
        if isinstance(image_size, bool) or not isinstance(image_size, int):
            raise ValueError("image_size must be an integer")
        if embedding_dimension != 1024 or image_size != 224:
            raise ValueError("CXR model requires dimensions 1024 and 224")
        self.encoder = encoder
        self.embedding_dimension = embedding_dimension
        self.image_size = image_size
        self.classifier = nn.Linear(embedding_dimension, 1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return one validated embedding per image."""
        _validate_images(images, self.image_size)
        encode = getattr(self.encoder, "encode", None)
        embeddings = encode(images) if callable(encode) else self.encoder(images)
        _validate_embeddings(embeddings, len(images), self.embedding_dimension)
        return embeddings

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one raw binary logit per image."""
        return self.classifier(self.encode(images)).squeeze(1)

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters for head-only warm-up."""
        set_cxr_encoder_trainability(self.encoder, "frozen")

    def unfreeze_encoder(self) -> None:
        """Restore encoder parameter trainability."""
        set_cxr_encoder_trainability(self.encoder, "all")


class ImageDenseNetModel:
    """Build the fixed image-only DenseNet121 classifier."""

    def __init__(
        self,
        encoder_factory: Callable[..., nn.Module] = StandardCxrEncoder,
    ) -> None:
        self._encoder_factory = encoder_factory

    def build(self, config: FamilyConfig) -> CxrBinaryClassifier:
        """Build an unfitted classifier without selecting a runtime device."""
        return self._build(config, weights=str(config.parameters["weights"]))

    def build_architecture(self, config: FamilyConfig) -> CxrBinaryClassifier:
        """Build the package architecture without loading upstream pretrained bytes."""
        return self._build(config, weights=None)

    def _build(self, config: FamilyConfig, *, weights: str | None) -> CxrBinaryClassifier:
        if config.family_id != "cxr_densenet" or config.modalities != ("cxr",):
            raise ValueError("Image DenseNet requires the registered image model configuration")
        parameters = dict(config.parameters)
        expected = {
            "encoder_name",
            "weights",
            "image_size",
            "embedding_dimension",
        }
        if set(parameters) != expected:
            raise ValueError(f"Image DenseNet parameters must be exactly {sorted(expected)}")
        if parameters["encoder_name"] != "densenet121":
            raise ValueError("Image DenseNet requires encoder_name='densenet121'")
        if parameters["weights"] != "densenet121-res224-chex":
            raise ValueError("Image DenseNet requires the fixed pretrained weights")
        if parameters["image_size"] != 224 or parameters["embedding_dimension"] != 1024:
            raise ValueError("Image DenseNet requires dimensions 224 and 1024")
        encoder = self._encoder_factory(
            weights=weights,
            expected_embedding_dimension=parameters["embedding_dimension"],
            image_size=parameters["image_size"],
        )
        if not isinstance(encoder, nn.Module):
            raise TypeError("Image encoder factory must return a torch.nn.Module")
        return CxrBinaryClassifier(
            encoder,
            embedding_dimension=parameters["embedding_dimension"],
            image_size=parameters["image_size"],
        )


def fingerprint_pretrained_weights(
    weights: str = "densenet121-res224-chex",
) -> PretrainedWeightIdentity:
    """Fingerprint the local TorchXRayVision weight file for a declared weight name."""
    import torchxrayvision as xrv

    try:
        stable_identifier = str(xrv.models.model_urls[weights]["weights_url"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown TorchXRayVision weight identity: {weights!r}") from exc
    cache_filename = Path(urlparse(stable_identifier).path).name
    if not cache_filename:
        raise ValueError("TorchXRayVision weight URL does not name a cache file")
    cache_path = Path(xrv.utils.get_cache_dir()).expanduser() / cache_filename
    if cache_path.is_symlink():
        raise ValueError(
            f"Pretrained weight file must be a regular non-symlink file: {cache_filename}"
        )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Pretrained weight file must be materialized before formal training: {cache_filename}"
        )
    if not cache_path.is_file():
        raise ValueError(f"Pretrained weight path must be a regular file: {cache_filename}")
    return PretrainedWeightIdentity(
        declared_name=weights,
        stable_identifier=stable_identifier,
        cache_filename=cache_filename,
        byte_size=cache_path.stat().st_size,
        sha256=sha256_file(cache_path),
    )


def ensure_pretrained_weights(
    weights: str = "densenet121-res224-chex",
) -> PretrainedWeightIdentity:
    """Materialize weights through TorchXRayVision and return strict local identity."""
    try:
        return fingerprint_pretrained_weights(weights)
    except FileNotFoundError:
        import torchxrayvision as xrv

        xrv.models.DenseNet(weights=weights)
    return fingerprint_pretrained_weights(weights)


def _validate_images(images: object, image_size: int) -> None:
    if not isinstance(images, torch.Tensor):
        raise TypeError("CXR model input must be a torch.Tensor")
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, image_size, image_size):
        raise ValueError(f"CXR model input must have shape N x 1 x {image_size} x {image_size}")
    if images.shape[0] == 0:
        raise ValueError("CXR model input batch must be non-empty")
    if not images.is_floating_point():
        raise ValueError("CXR model input must contain floating-point values")


def _validate_embeddings(embeddings: object, batch_size: int, dimension: int) -> None:
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("CXR encoder output must be a torch.Tensor")
    if not embeddings.is_floating_point():
        raise ValueError("CXR encoder output must contain floating-point values")
    if embeddings.shape != (batch_size, dimension):
        raise ValueError(
            f"CXR encoder output must have shape ({batch_size}, {dimension}), "
            f"received {tuple(embeddings.shape)}"
        )
