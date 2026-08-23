from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pytest
import torch
import torchxrayvision as xrv
from torch import nn

from beyondcxr.models.cxr_baseline import (
    CxrBinaryClassifier,
    CxrDenseNetModel,
    StandardCxrEncoder,
    ensure_pretrained_weights,
    fingerprint_pretrained_weights,
    set_cxr_encoder_trainability,
    set_cxr_encoder_training_mode,
)
from beyondcxr.training.config import load_experiment_config
from beyondcxr.training.rsna_registry import MODELS, get_model


class _TinyEncoder(nn.Module):
    def __init__(self, *, expected_embedding_dimension: int = 1024, **_: object) -> None:
        super().__init__()
        self.projection = nn.Linear(1, expected_embedding_dimension)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(2, 3)))


class _FixedOutputEncoder(nn.Module):
    def __init__(self, output: object) -> None:
        super().__init__()
        self.output = output

    def encode(self, images: torch.Tensor) -> object:
        del images
        return self.output


def test_cxr_classifier_exposes_raw_logits_embeddings_and_freezing() -> None:
    model = CxrBinaryClassifier(_TinyEncoder())
    images = torch.ones((3, 1, 224, 224), dtype=torch.float32)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.fill_(-2.0)

    assert model.encoder is not model.classifier
    assert model.encode(images).shape == (3, 1024)
    assert torch.equal(model(images), torch.full((3,), -2.0))
    model.freeze_encoder()
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    model.unfreeze_encoder()
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


def test_standard_encoder_terminal_trainability_is_exact() -> None:
    encoder = StandardCxrEncoder(weights=None)

    set_cxr_encoder_trainability(encoder, "terminal")

    expected = {
        name
        for name, _ in encoder.named_parameters()
        if (
            name.startswith("backbone.features.denseblock4.")
            or name.startswith("backbone.features.norm5.")
        )
    }
    trainable = {name for name, parameter in encoder.named_parameters() if parameter.requires_grad}
    assert trainable == expected
    set_cxr_encoder_training_mode(encoder, "terminal")
    modules = dict(encoder.named_modules())
    terminal_modules = {
        name
        for name in modules
        if name == "backbone.features.denseblock4"
        or name.startswith("backbone.features.denseblock4.")
        or name == "backbone.features.norm5"
        or name.startswith("backbone.features.norm5.")
    }
    assert terminal_modules
    assert {name for name, module in modules.items() if module.training} == terminal_modules
    set_cxr_encoder_trainability(encoder, "all")
    set_cxr_encoder_training_mode(encoder, "all")
    assert all(parameter.requires_grad for parameter in encoder.parameters())
    assert all(module.training for module in encoder.modules())


def test_cxr_registry_builds_with_injected_encoder_without_weights() -> None:
    config = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    model = CxrDenseNetModel(encoder_factory=_TinyEncoder).build(config.family)

    assert get_model("cxr_densenet") is MODELS["cxr_densenet"]
    assert tuple(MODELS).count("cxr_densenet") == 1
    assert model.encode(torch.ones((2, 1, 224, 224))).shape == (2, 1024)
    assert model(torch.ones((2, 1, 224, 224))).shape == (2,)


@pytest.mark.parametrize(
    ("images", "exception"),
    [
        ([1.0], TypeError),
        (torch.ones((0, 1, 224, 224)), ValueError),
        (torch.ones((1, 224, 224)), ValueError),
        (torch.ones((1, 3, 224, 224)), ValueError),
        (torch.ones((1, 1, 223, 224)), ValueError),
        (torch.ones((1, 1, 224, 224), dtype=torch.int64), ValueError),
    ],
)
def test_cxr_classifier_rejects_invalid_inputs(
    images: object,
    exception: type[Exception],
) -> None:
    model = CxrBinaryClassifier(_TinyEncoder())
    with pytest.raises(exception):
        model(images)  # type: ignore[arg-type]


def test_cxr_classifier_rejects_wrong_embedding_shape() -> None:
    model = CxrBinaryClassifier(_TinyEncoder(expected_embedding_dimension=1023))
    with pytest.raises(ValueError):
        model.encode(torch.ones((1, 1, 224, 224)))


@pytest.mark.parametrize(
    ("output", "exception"),
    [
        ([1.0] * 1024, TypeError),
        (torch.ones((1, 1024), dtype=torch.int64), ValueError),
    ],
)
def test_cxr_classifier_rejects_invalid_encoder_outputs(
    output: object,
    exception: type[Exception],
) -> None:
    model = CxrBinaryClassifier(_FixedOutputEncoder(output))

    with pytest.raises(exception):
        model.encode(torch.ones((1, 1, 224, 224)))


def test_cxr_classifier_requires_module_encoder() -> None:
    with pytest.raises(TypeError):
        CxrBinaryClassifier(object())  # type: ignore[arg-type]


def test_cxr_builder_rejects_non_module_factory_output() -> None:
    config = load_experiment_config("configs/rsna_cxr_densenet.yaml")

    with pytest.raises(TypeError):
        CxrDenseNetModel(encoder_factory=lambda **_: object()).build(config.family)


def test_pretrained_weight_identity_fingerprints_materialized_file_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "weights.pt"
    weight_path = tmp_path / filename
    weight_path.write_bytes(b"weight-bytes")
    monkeypatch.setitem(
        xrv.models.model_urls,
        "synthetic-weights",
        {"weights_url": f"https://example.invalid/{filename}"},
    )
    monkeypatch.setattr(xrv.utils, "get_cache_dir", lambda: str(tmp_path))

    identity = fingerprint_pretrained_weights("synthetic-weights")

    assert identity.declared_name == "synthetic-weights"
    assert identity.cache_filename == filename
    assert identity.byte_size == len(b"weight-bytes")
    assert len(identity.sha256) == 64
    weight_path.unlink()
    with pytest.raises(FileNotFoundError):
        fingerprint_pretrained_weights("synthetic-weights")


def test_pretrained_weight_readiness_uses_the_supported_xrv_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "weight.pt"
    monkeypatch.setattr(xrv.utils, "get_cache_dir", lambda: str(tmp_path))
    monkeypatch.setitem(
        xrv.models.model_urls,
        "synthetic-weights",
        {"weights_url": f"https://example.invalid/{filename}"},
    )
    requested: list[str] = []

    def acquire(*, weights: str):
        requested.append(weights)
        (tmp_path / filename).write_bytes(b"materialized")
        return object()

    monkeypatch.setattr(xrv.models, "DenseNet", acquire)

    identity = ensure_pretrained_weights("synthetic-weights")

    assert requested == ["synthetic-weights"]
    assert identity.declared_name == "synthetic-weights"
    assert identity.byte_size == len(b"materialized")


def test_pretrained_weight_identity_rejects_unknown_and_unnamed_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError):
        fingerprint_pretrained_weights("unknown-weight")

    monkeypatch.setattr(xrv.utils, "get_cache_dir", lambda: str(tmp_path))
    monkeypatch.setitem(
        xrv.models.model_urls,
        "unnamed-weight",
        {"weights_url": "https://example.invalid/"},
    )
    with pytest.raises(ValueError):
        fingerprint_pretrained_weights("unnamed-weight")


@pytest.mark.parametrize("entry_kind", ["symlink", "directory"])
def test_pretrained_weight_identity_rejects_nonregular_cache_entries(
    entry_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "weight.pt"
    cache_path = tmp_path / filename
    if entry_kind == "symlink":
        target = tmp_path / "target.pt"
        target.write_bytes(b"weight")
        cache_path.symlink_to(target)
    else:
        cache_path.mkdir()

    monkeypatch.setattr(xrv.utils, "get_cache_dir", lambda: str(tmp_path))
    monkeypatch.setitem(
        xrv.models.model_urls,
        "invalid-weight",
        {"weights_url": f"https://example.invalid/{filename}"},
    )
    with pytest.raises(ValueError):
        fingerprint_pretrained_weights("invalid-weight")


def test_modified_pretrained_bytes_change_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "weight.pt"
    path.write_bytes(b"first")
    monkeypatch.setattr(xrv.utils, "get_cache_dir", lambda: str(tmp_path))
    monkeypatch.setitem(
        xrv.models.model_urls,
        "mutable-weight",
        {"weights_url": "https://example.invalid/weight.pt"},
    )
    first = fingerprint_pretrained_weights("mutable-weight")
    path.write_bytes(b"second")
    second = fingerprint_pretrained_weights("mutable-weight")

    assert first.sha256 != second.sha256
    assert first.byte_size != second.byte_size


def test_evaluation_architecture_is_built_without_pretrained_cache_access() -> None:
    observed_weights = []

    def factory(**kwargs):
        observed_weights.append(kwargs["weights"])
        return _TinyEncoder(**kwargs)

    config = load_experiment_config("configs/rsna_cxr_densenet.yaml")
    model = CxrDenseNetModel(encoder_factory=factory).build_architecture(config.family)

    assert isinstance(model, CxrBinaryClassifier)
    assert observed_weights == [None]


# This warning comes from TorchXRayVision's serialized pretrained weight file.
# BeyondCXR model.pt packages will store explicit state dictionaries rather than pickled modules.
@pytest.mark.filterwarnings(
    "ignore:source code of class .* has changed.*:torch.serialization.SourceChangeWarning"
)
@pytest.mark.integration
def test_cached_standard_encoder_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    weights = "densenet121-res224-chex"
    url = xrv.models.model_urls[weights]["weights_url"]
    cache_path = Path(xrv.utils.get_cache_dir()).expanduser() / Path(urlparse(url).path).name
    if not cache_path.is_file():
        pytest.skip(f"TorchXRayVision weights are not cached at {cache_path}")

    def reject_download(*_: object, **__: object) -> None:
        raise AssertionError("Cached encoder offline check attempted a network download")

    monkeypatch.setattr(xrv.utils, "download", reject_download)
    encoder = StandardCxrEncoder().eval()
    with torch.inference_mode():
        embedding = encoder.encode(torch.zeros((1, 1, 224, 224), dtype=torch.float32))
    assert embedding.shape == (1, 1024)
    assert torch.isfinite(embedding).all()
