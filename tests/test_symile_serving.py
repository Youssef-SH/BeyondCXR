from __future__ import annotations

import io
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from symile_campaign_test_support import _freeze, _synthetic_held_out_capability

import beyondcxr.serving.cli as serving_cli
import beyondcxr.serving.predictor as serving_predictor
import beyondcxr.training.symile_test_data as symile_test_data
from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.data.errors import ManifestBuildError
from beyondcxr.serving.api import MAX_REQUEST_BODY_BYTES, create_app
from beyondcxr.serving.authority import LAB_KEYS, RESEARCH_WARNING, SERVING_SPATIAL_POLICY
from beyondcxr.serving.predictor import SymileServingPredictor, mean_logit_probability
from beyondcxr.serving.preprocessing import (
    SERVING_CANONICAL_SIZE,
    ServingInputError,
    decode_jpeg_once,
    parse_laboratories,
    serving_canonical_image,
    serving_tensor_from_canonical_image,
    validate_view_position,
    validated_serving_input,
)
from beyondcxr.training.symile_test_data import FrozenSymileNeuralTestDataset


def _jpeg(*, rows: int = 360, columns: int = 480) -> bytes:
    pattern = np.indices((rows, columns)).sum(axis=0) % 2
    image = Image.fromarray((pattern * 255).astype(np.uint8), mode="L")
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def _labs(*, missing: str | None = None) -> str:
    return json.dumps(
        {key: None if key == missing else float(index) for index, key in enumerate(LAB_KEYS)}
    )


def test_model_info_uses_immutable_startup_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = object.__new__(SymileServingPredictor)
    predictor._active = True
    predictor._process_id = os.getpid()
    predictor._model_info = MappingProxyType(
        {"serving_authority_id": "serving-authority-" + "a" * 64, "seeds": (17, 42, 2026)}
    )
    monkeypatch.setattr(
        serving_predictor,
        "serving_model_info",
        lambda *_args: (_ for _ in ()).throw(AssertionError("filesystem revalidation")),
    )

    observed = predictor.model_info()
    observed["seeds"].append(999)

    assert predictor.model_info()["seeds"] == [17, 42, 2026]


def test_preloaded_predictor_is_bound_to_exact_release_inputs_and_revocable(
    tmp_path: Path,
) -> None:
    authority = tmp_path / "authority"
    provenance = tmp_path / "deployment.json"
    binding = tmp_path / "binding.json"
    archive = tmp_path / "release.zip"
    predictor = object.__new__(SymileServingPredictor)
    predictor._active = True
    predictor._process_id = os.getpid()
    predictor._authority_directory = authority.resolve()
    predictor._release_inputs = tuple(path.resolve() for path in (provenance, binding, archive))

    predictor.require_release_inputs(authority, provenance, binding, archive)
    with pytest.raises(ManifestBuildError, match="different authority"):
        predictor.require_release_inputs(tmp_path / "other", provenance, binding, archive)
    with pytest.raises(ManifestBuildError, match="different release evidence"):
        predictor.require_release_inputs(authority, tmp_path / "other.json", binding, archive)
    predictor._process_id = -1
    with pytest.raises(ManifestBuildError, match="lifetime has ended"):
        predictor.require_release_inputs(authority, provenance, binding, archive)
    predictor._process_id = os.getpid()
    predictor.close()
    with pytest.raises(ManifestBuildError, match="lifetime has ended"):
        predictor.require_release_inputs(authority, provenance, binding, archive)


_SPATIAL_CASES = (
    (360, 480, (320, 426), (53, 0)),
    (480, 360, (426, 320), (0, 53)),
    (361, 479, (320, 424), (52, 0)),
    (479, 361, (424, 320), (0, 52)),
    (360, 481, (320, 427), (54, 0)),
)


def _spatial_oracle(decoded: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    height, width = decoded.shape
    if width <= height:
        resized_width = SERVING_CANONICAL_SIZE
        resized_height = int(SERVING_CANONICAL_SIZE * height / width)
    else:
        resized_height = SERVING_CANONICAL_SIZE
        resized_width = int(SERVING_CANONICAL_SIZE * width / height)
    left = int(round((resized_width - SERVING_CANONICAL_SIZE) / 2.0))
    top = int(round((resized_height - SERVING_CANONICAL_SIZE) / 2.0))
    resized = Image.fromarray(decoded, mode="L").resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    cropped = resized.crop(
        (
            left,
            top,
            left + SERVING_CANONICAL_SIZE,
            top + SERVING_CANONICAL_SIZE,
        )
    )
    reference = np.ascontiguousarray(
        np.asarray(cropped, dtype=np.float32) / np.float32(255.0),
        dtype=np.float32,
    )
    return reference, (resized_height, resized_width), (left, top)


@pytest.mark.parametrize("value", ["", "LATERAL", "UNKNOWN", "ap", None])
def test_reject_unknown_or_non_frontal_view(value: object) -> None:
    with pytest.raises(ServingInputError, match="invalid_view_position"):
        validate_view_position(value)


@pytest.mark.parametrize("rows,columns,expected_resize,expected_crop", _SPATIAL_CASES)
def test_serving_spatial_policy_exactly_matches_independent_oracle(
    rows: int,
    columns: int,
    expected_resize: tuple[int, int],
    expected_crop: tuple[int, int],
) -> None:
    decoded = decode_jpeg_once(_jpeg(rows=rows, columns=columns))
    reference, resized_shape, crop_origin = _spatial_oracle(decoded)
    production = serving_canonical_image(decoded)
    assert resized_shape == expected_resize
    assert min(resized_shape) == SERVING_CANONICAL_SIZE
    assert crop_origin == expected_crop
    assert reference.shape == production.shape == (320, 320)
    assert reference.dtype == production.dtype == np.float32
    assert reference.flags.c_contiguous and production.flags.c_contiguous
    assert np.isfinite(reference).all() and np.isfinite(production).all()
    assert reference.min() >= 0.0 and production.min() >= 0.0
    assert reference.max() <= 1.0 and production.max() <= 1.0
    assert np.array_equal(reference, production)


@pytest.mark.parametrize("shape", [(10_000, 1), (1, 10_000)], ids=["portrait", "landscape"])
def test_pathological_spatial_resize_is_rejected_before_allocation(
    shape: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_resize(*args: object, **kwargs: object) -> None:
        raise AssertionError("Image.resize must not run for pathological aspect ratios")

    monkeypatch.setattr(Image.Image, "resize", forbidden_resize)
    with pytest.raises(ServingInputError, match="invalid_image_content"):
        serving_canonical_image(np.zeros(shape, dtype=np.uint8))


def test_canonical_image_exactly_matches_frozen_evaluation_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoded = decode_jpeg_once(_jpeg(rows=361, columns=479))
    canonical = serving_canonical_image(decoded)
    transform = StandardCxrTransform(training=False)
    freeze = _freeze(tmp_path / "synthetic-freeze")
    capability = _synthetic_held_out_capability(tmp_path / "capability", freeze)
    monkeypatch.setattr(symile_test_data, "materialize_official_test", lambda *args: None)
    monkeypatch.setattr(symile_test_data, "STRICT_PNEUMONIA_TEST_ADMISSIONS", 2)
    monkeypatch.setattr(
        symile_test_data, "validated_pretest_freeze_manifest", lambda value: value.manifest
    )
    frame = pd.DataFrame(
        {
            "sample_id": ["synthetic-0", "synthetic-1"],
            "subject_id": [1, 2],
            "source_row": [0, 1],
            "target": [0, 1],
        }
    )
    data = object.__new__(symile_test_data.FrozenSymileTestData)
    data._bundle = SimpleNamespace(bundle_id=str(freeze.manifest["bundle_id"]))
    data._canonical_frame = frame
    data._capability = freeze
    store = object.__new__(symile_test_data.FrozenSymileTestCxrStore)
    store._freeze = freeze
    store._bundle_id = str(freeze.manifest["bundle_id"])
    store.canonical_image = lambda row: canonical
    reference_dataset = FrozenSymileNeuralTestDataset(
        capability,
        data,
        cxr_store=store,
        transform=transform,
    )
    reference = reference_dataset[0]["image"]
    serving = serving_tensor_from_canonical_image(canonical, transform)
    assert reference.dtype == serving.dtype == torch.float32
    assert reference.shape == serving.shape == (1, 224, 224)
    assert reference.is_contiguous() and serving.is_contiguous()
    assert torch.equal(reference, serving)


def test_neural_test_dataset_rejects_foreign_modality_composition(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = _freeze(tmp_path / "freeze")
    capability = _synthetic_held_out_capability(tmp_path / "capability", freeze)
    monkeypatch.setattr(symile_test_data, "materialize_official_test", lambda *args: None)
    monkeypatch.setattr(symile_test_data, "STRICT_PNEUMONIA_TEST_ADMISSIONS", 2)
    monkeypatch.setattr(
        symile_test_data, "validated_pretest_freeze_manifest", lambda value: value.manifest
    )
    data = object.__new__(symile_test_data.FrozenSymileTestData)
    data._bundle = SimpleNamespace(bundle_id=str(freeze.manifest["bundle_id"]))
    data._canonical_frame = pd.DataFrame(
        {
            "sample_id": ["sample-0", "sample-1"],
            "subject_id": [1, 2],
            "source_row": [0, 1],
            "target": [0, 1],
        }
    )
    data._capability = freeze

    def store(store_type, *, bundle_id: str):
        value = object.__new__(store_type)
        value._freeze = freeze
        value._bundle_id = bundle_id
        return value

    authorized = str(freeze.manifest["bundle_id"])
    foreign = "bundle-" + "f" * 64
    cxr = store(symile_test_data.FrozenSymileTestCxrStore, bundle_id=authorized)
    ecg = store(symile_test_data.FrozenSymileTestEcgStore, bundle_id=authorized)
    transform = StandardCxrTransform(training=False)
    assert (
        len(
            FrozenSymileNeuralTestDataset(
                capability,
                data,
                cxr_store=cxr,
                ecg_store=ecg,
                transform=transform,
            )
        )
        == 2
    )
    with pytest.raises(PermissionError, match="another campaign"):
        FrozenSymileNeuralTestDataset(
            capability,
            data,
            cxr_store=store(symile_test_data.FrozenSymileTestCxrStore, bundle_id=foreign),
            transform=transform,
        )
    with pytest.raises(PermissionError, match="another campaign"):
        FrozenSymileNeuralTestDataset(
            capability,
            data,
            cxr_store=cxr,
            ecg_store=store(symile_test_data.FrozenSymileTestEcgStore, bundle_id=foreign),
            transform=transform,
        )


@pytest.mark.parametrize("media", ["image/png", "application/dicom", "application/octet-stream"])
def test_reject_non_jpeg_media(media: str) -> None:
    with pytest.raises(ServingInputError, match="invalid_media_type"):
        validated_serving_input(
            image_content=_jpeg(),
            image_media_type=media,
            view_position="AP",
            labs_json=_labs(),
            transform=StandardCxrTransform(training=False),
        )


@pytest.mark.parametrize("format_name", ["PNG", "TIFF", "WEBP", "BMP", "GIF"])
def test_reject_non_jpeg_content(format_name: str) -> None:
    output = io.BytesIO()
    Image.new("L", (32, 32), 127).save(output, format=format_name)
    with pytest.raises(ServingInputError, match="invalid_image_content"):
        decode_jpeg_once(output.getvalue())


def test_labs_require_exact_keys_and_strict_values() -> None:
    complete = json.loads(_labs())
    for invalid in (
        {key: value for key, value in complete.items() if key != LAB_KEYS[0]},
        {**complete, "unknown": 1.0},
        {**complete, LAB_KEYS[0]: True},
        {**complete, LAB_KEYS[0]: "1.0"},
        {**complete, LAB_KEYS[0]: float("nan")},
        {**complete, LAB_KEYS[0]: float("inf")},
    ):
        with pytest.raises(ServingInputError):
            parse_laboratories(json.dumps(invalid))
    frame, missing = parse_laboratories(_labs(missing=LAB_KEYS[7]))
    assert tuple(frame.columns)[-50:] == tuple(
        f"lab_{key.removeprefix('lab_')}_observed" for key in LAB_KEYS
    )
    assert missing == (LAB_KEYS[7],)
    duplicate = _labs()[:-1] + f', "{LAB_KEYS[0]}": 1.0}}'
    with pytest.raises(ServingInputError, match="duplicate_lab_keys"):
        parse_laboratories(duplicate)


@pytest.mark.parametrize(
    "document",
    [
        '{"lab": ' + "9" * 5_000 + "}",
        "[" * 2_000 + "0" + "]" * 2_000,
    ],
)
def test_pathological_labs_json_has_stable_error(document: str) -> None:
    with pytest.raises(ServingInputError, match="invalid_labs_json"):
        parse_laboratories(document)


def test_huge_valid_json_integer_has_stable_lab_value_error() -> None:
    laboratories = json.loads(_labs())
    laboratories[LAB_KEYS[0]] = 10**400
    with pytest.raises(ServingInputError, match="invalid_lab_value"):
        parse_laboratories(json.dumps(laboratories))


def test_image_media_type_matches_immutable_authority() -> None:
    accepted = validated_serving_input(
        image_content=_jpeg(),
        image_media_type="image/jpeg",
        view_position="AP",
        labs_json=_labs(),
        transform=StandardCxrTransform(training=False),
    )
    assert accepted.image.shape == (1, 224, 224)
    with pytest.raises(ServingInputError, match="invalid_media_type"):
        validated_serving_input(
            image_content=_jpeg(),
            image_media_type="image/jpg",
            view_position="AP",
            labs_json=_labs(),
            transform=StandardCxrTransform(training=False),
        )


def test_raw_probability_is_sigmoid_of_mean_logit() -> None:
    logits = (-4.0, 0.0, 2.0)
    observed = mean_logit_probability(logits)
    expected = 1.0 / (1.0 + np.exp(2.0 / 3.0))
    probability_average = float(np.mean(1.0 / (1.0 + np.exp(-np.asarray(logits)))))
    assert observed == pytest.approx(expected)
    assert observed != pytest.approx(probability_average)


def test_view_position_is_validation_only() -> None:
    transform = StandardCxrTransform(training=False)
    ap = validated_serving_input(
        image_content=_jpeg(),
        image_media_type="image/jpeg",
        view_position="AP",
        labs_json=_labs(),
        transform=transform,
    )
    pa = validated_serving_input(
        image_content=_jpeg(),
        image_media_type="image/jpeg",
        view_position="PA",
        labs_json=_labs(),
        transform=transform,
    )
    assert torch.equal(ap.image, pa.image)
    assert ap.laboratory_frame.equals(pa.laboratory_frame)
    assert not hasattr(ap, "view_position")


def test_cli_disables_uvicorn_access_log(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    application = object()
    observed: dict[str, object] = {}
    monkeypatch.setattr(serving_cli, "create_app", lambda **kwargs: application)

    with pytest.raises(SystemExit) as help_exit:
        serving_cli.main(["--help"])
    assert help_exit.value.code == 0
    assert "phase=serving_cli" not in capsys.readouterr().err

    def run(app: object, **kwargs: object) -> None:
        observed["application"] = app
        observed.update(kwargs)

    monkeypatch.setattr(serving_cli.uvicorn, "run", run)
    result = serving_cli.main(
        [
            "--authority",
            "/synthetic/authority",
            "--deployment-provenance",
            "/synthetic/deployment.json",
            "--result-binding",
            "/synthetic/binding.json",
            "--release-archive",
            "/synthetic/release.zip",
            "--device",
            "cpu",
            "--host",
            "127.0.0.2",
            "--port",
            "8123",
        ]
    )
    assert result == 0
    assert observed == {
        "application": application,
        "host": "127.0.0.2",
        "port": 8123,
        "log_config": None,
        "access_log": False,
    }


class _FakePredictor:
    def __init__(self) -> None:
        self.authority = SimpleNamespace(
            authority_id="serving-authority-" + "a" * 64,
            manifest={
                "task": {"task_id": "pneumonia_strict"},
                "warning": RESEARCH_WARNING,
            },
        )
        self.transform = StandardCxrTransform(training=False)

    def model_info(self):
        return {
            "task": "pneumonia_strict",
            "serving_authority_id": self.authority.authority_id,
            "seeds": [17, 42, 2026],
            "operating_thresholds": {"youden_j": 0.5, "target_sensitivity": 0.25},
            "warning": RESEARCH_WARNING,
        }

    def predict(self, request):
        del request
        return mean_logit_probability([-4.0, 0.0, 2.0])

    def close(self) -> None:
        pass


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "beyondcxr.serving.api.SymileServingPredictor.load",
        lambda *args, **kwargs: _FakePredictor(),
    )
    application = create_app(
        authority_path="/explicitly-synthetic/authority",
        deployment_provenance="/explicitly-synthetic/deployment.json",
        result_binding="/explicitly-synthetic/binding.json",
        release_archive="/explicitly-synthetic/release.zip",
    )
    transport = httpx.ASGITransport(app=application)
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as value:
            yield value


def _multipart(*, image: bytes | None = None, view: str = "AP", labs: str | None = None):
    return [
        ("image", ("synthetic.jpg", image or _jpeg(), "image/jpeg")),
        ("view_position", (None, view)),
        ("labs", (None, labs or _labs())),
    ]


@pytest.mark.anyio
async def test_health_endpoint(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "serving_authority_id": "serving-authority-" + "a" * 64,
    }


@pytest.mark.anyio
async def test_service_startup_fails_closed_without_authority() -> None:
    application = create_app()
    with pytest.raises(RuntimeError, match="deployment provenance"):
        async with application.router.lifespan_context(application):
            pass


@pytest.mark.anyio
async def test_model_info_endpoint_exposes_thresholds_only_as_metadata(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["seeds"] == [17, 42, 2026]
    assert set(response.json()["operating_thresholds"]) == {
        "youden_j",
        "target_sensitivity",
    }
    assert response.json()["warning"] == RESEARCH_WARNING


def test_spatial_policy_identifier_is_explicit() -> None:
    assert SERVING_SPATIAL_POLICY == ("symile-jpeg-spatial-short-side-320-center-crop-bilinear-v1")


@pytest.mark.anyio
async def test_predict_with_fake_jpg_returns_one_raw_probability_and_warning(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post("/predict", files=_multipart(labs=_labs(missing=LAB_KEYS[3])))
    assert response.status_code == 200
    document = response.json()
    assert set(document) == {
        "task",
        "probability",
        "missing_labs",
        "serving_authority_id",
        "warning",
    }
    assert document["probability"] == pytest.approx(mean_logit_probability([-4.0, 0.0, 2.0]))
    assert document["missing_labs"] == [LAB_KEYS[3]]
    assert document["warning"] == RESEARCH_WARNING
    assert not {"decision", "positive", "threshold"} & set(document)


@pytest.mark.anyio
async def test_oversized_raw_request_is_rejected_before_inference(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_inference(*args: object, **kwargs: object) -> float:
        raise AssertionError("inference must not run for an oversized request")

    monkeypatch.setattr(_FakePredictor, "predict", forbidden_inference)
    response = await client.post(
        "/predict",
        content=b"x" * (MAX_REQUEST_BODY_BYTES + 1),
        headers={"content-type": "multipart/form-data; boundary=oversized"},
    )
    assert response.status_code == 413
    assert response.json() == {"error": "request_body_too_large"}


@pytest.mark.anyio
async def test_streamed_oversized_request_is_rejected_before_inference(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_inference(*args: object, **kwargs: object) -> float:
        raise AssertionError("inference must not run for an oversized request")

    async def oversized_multipart() -> AsyncIterator[bytes]:
        prefix = (
            b"--streamed\r\n"
            b'Content-Disposition: form-data; name="image"; filename="synthetic.jpg"\r\n'
            b"Content-Type: image/jpeg\r\n\r\n"
        )
        yield prefix + b"x" * (MAX_REQUEST_BODY_BYTES + 1)

    monkeypatch.setattr(_FakePredictor, "predict", forbidden_inference)
    response = await client.post(
        "/predict",
        content=oversized_multipart(),
        headers={"content-type": "multipart/form-data; boundary=streamed"},
    )
    assert response.status_code == 413
    assert response.json() == {"error": "request_body_too_large"}


@pytest.mark.anyio
async def test_api_normalizes_pathological_labs_json(client: httpx.AsyncClient) -> None:
    huge_integer = '{"lab": ' + "9" * 5_000 + "}"
    response = await client.post("/predict", files=_multipart(labs=huge_integer))
    assert response.status_code == 400
    assert response.json() == {"error": "invalid_labs_json"}


@pytest.mark.anyio
async def test_api_normalizes_huge_valid_json_integer(client: httpx.AsyncClient) -> None:
    laboratories = json.loads(_labs())
    laboratories[LAB_KEYS[0]] = 10**400
    response = await client.post("/predict", files=_multipart(labs=json.dumps(laboratories)))
    assert response.status_code == 400
    assert response.json() == {"error": "invalid_lab_value"}


@pytest.mark.parametrize(
    "filename,content,media",
    [
        ("synthetic.dcm", b"DICM" + b"0" * 32, "application/dicom"),
        ("synthetic.npy", b"\x93NUMPY", "application/octet-stream"),
        ("synthetic.jpg", b"not-a-jpeg", "image/jpeg"),
    ],
)
@pytest.mark.anyio
async def test_api_rejects_non_jpeg_or_malformed_image(
    client: httpx.AsyncClient, filename: str, content: bytes, media: str
) -> None:
    fields = _multipart()
    fields[0] = ("image", (filename, content, media))
    response = await client.post("/predict", files=fields)
    assert response.status_code in {400, 415}
    assert response.json()["error"] in {"invalid_media_type", "invalid_image_content"}


@pytest.mark.anyio
async def test_api_rejects_invalid_view_and_laboratory_contract(
    client: httpx.AsyncClient,
) -> None:
    invalid_view = await client.post("/predict", files=_multipart(view="LATERAL"))
    assert invalid_view.status_code == 400
    assert invalid_view.json() == {"error": "invalid_view_position"}

    missing = json.loads(_labs())
    missing.pop(LAB_KEYS[0])
    invalid_labs = await client.post("/predict", files=_multipart(labs=json.dumps(missing)))
    assert invalid_labs.status_code == 400
    assert invalid_labs.json() == {"error": "missing_lab_keys"}


@pytest.mark.anyio
async def test_api_rejects_unknown_or_duplicate_multipart_fields(
    client: httpx.AsyncClient,
) -> None:
    extra = [*_multipart(), ("unknown", (None, "value"))]
    duplicate = [*_multipart(), ("labs", (None, _labs()))]
    assert (await client.post("/predict", files=extra)).json() == {
        "error": "invalid_multipart_fields"
    }
    assert (await client.post("/predict", files=duplicate)).json() == {
        "error": "invalid_multipart_fields"
    }


@pytest.mark.anyio
async def test_serving_lifespan_observes_startup_and_cleans_up_on_interruption(monkeypatch):
    from beyondcxr.utils.operational_logging import configured_logging

    predictor = SimpleNamespace(closed=False)

    def close() -> None:
        predictor.closed = True

    predictor.close = close
    monkeypatch.setattr(SymileServingPredictor, "load", lambda *args, **kwargs: predictor)
    application = create_app(
        authority_path="synthetic",
        deployment_provenance="synthetic",
        result_binding="synthetic",
        release_archive="synthetic",
    )
    stream = io.StringIO()
    with configured_logging(stream=stream, context={}):
        with pytest.raises(RuntimeError, match="PRIVATE"):
            async with application.router.lifespan_context(application):
                assert application.state.predictor is predictor
                raise RuntimeError("PRIVATE")
    assert application.state.predictor is None
    assert predictor.closed
    text = stream.getvalue()
    assert "phase=serving_startup" in text
    assert "event=serving_ready" in text
    assert "event=serving_stopped" in text
    assert "PRIVATE" not in text
