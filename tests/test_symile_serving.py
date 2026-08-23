from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from symile_campaign_test_support import _freeze

import beyondcxr.serving.cli as serving_cli
from beyondcxr.data.cxr_transforms import StandardCxrTransform
from beyondcxr.serving.api import MAX_REQUEST_BODY_BYTES, create_app
from beyondcxr.serving.authority import LAB_KEYS, RESEARCH_WARNING, SERVING_SPATIAL_POLICY
from beyondcxr.serving.predictor import mean_logit_probability
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
from beyondcxr.training.symile_campaign_control import create_or_validate_test_open_record
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


def test_canonical_image_exactly_matches_frozen_evaluation_path(tmp_path) -> None:
    decoded = decode_jpeg_once(_jpeg(rows=361, columns=479))
    canonical = serving_canonical_image(decoded)
    transform = StandardCxrTransform(training=False)
    freeze = _freeze(tmp_path / "synthetic-freeze")
    record = create_or_validate_test_open_record(
        control_root=tmp_path / "synthetic-control",
        capability=freeze,
    )
    frame = pd.DataFrame(
        {"sample_id": ["synthetic-sample"], "subject_id": [1], "source_row": [0], "target": [0]}
    )
    store = SimpleNamespace(canonical_image=lambda row: canonical)
    reference_dataset = FrozenSymileNeuralTestDataset(
        freeze,
        record,
        frame,
        cxr_store=store,
        transform=transform,
    )
    reference = reference_dataset[0]["image"]
    serving = serving_tensor_from_canonical_image(canonical, transform)
    assert reference.dtype == serving.dtype == torch.float32
    assert reference.shape == serving.shape == (1, 224, 224)
    assert reference.is_contiguous() and serving.is_contiguous()
    assert torch.equal(reference, serving)


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


def test_cli_disables_uvicorn_access_log(monkeypatch: pytest.MonkeyPatch) -> None:
    application = object()
    observed: dict[str, object] = {}
    monkeypatch.setattr(serving_cli, "create_app", lambda **kwargs: application)

    def run(app: object, **kwargs: object) -> None:
        observed["application"] = app
        observed.update(kwargs)

    monkeypatch.setattr(serving_cli.uvicorn, "run", run)
    result = serving_cli.main(
        [
            "--authority",
            "/synthetic/authority",
            "--package-root",
            "/synthetic/packages",
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
        package_root="/explicitly-synthetic/packages",
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
    with pytest.raises(RuntimeError, match="authority and package root"):
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
