"""Minimal research-only FastAPI surface for the sealed Symile ensemble."""

from __future__ import annotations

from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from radfusion.serving.predictor import SymileServingPredictor
from radfusion.serving.preprocessing import (
    MAX_JPEG_BYTES,
    ServingInputError,
    validated_serving_input,
)

MAX_REQUEST_BODY_BYTES = 21 * 1024 * 1024


class _RequestBodyTooLarge(Exception):
    """Signal that the raw request exceeded the serving transport contract."""


class _RequestBodyLimitMiddleware:
    """Enforce a streaming ASGI request limit without buffering the body."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        content_length = next(
            (
                value
                for name, value in scope.get("headers", ())
                if name.lower() == b"content-length"
            ),
            None,
        )
        try:
            declared_size = int(content_length) if content_length is not None else None
        except ValueError:
            declared_size = None
        if declared_size is not None and declared_size > self.max_body_bytes:
            await self._reject(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=413,
            content={"error": "request_body_too_large"},
        )
        await response(scope, receive, send)


def create_app(
    *,
    authority_path: str | Path | None = None,
    package_root: str | Path | None = None,
    device: str = "cpu",
) -> FastAPI:
    """Create one app whose lifespan fails unless its complete authority loads."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if authority_path is None or package_root is None:
            raise RuntimeError("A serving authority and package root are required")
        loaded = SymileServingPredictor.load(
            authority_path,
            package_root=package_root,
            device=device,
        )
        app.state.predictor = loaded
        yield
        app.state.predictor = None

    application = FastAPI(
        title="RadFusion-Clinical Research Serving",
        version="1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    application.add_middleware(
        _RequestBodyLimitMiddleware,
        max_body_bytes=MAX_REQUEST_BODY_BYTES,
    )

    @application.exception_handler(ServingInputError)
    async def invalid_input(_: Request, exc: ServingInputError) -> JSONResponse:
        status = 415 if exc.category == "invalid_media_type" else 400
        return JSONResponse(status_code=status, content={"error": exc.category})

    @application.get("/health")
    async def health(request: Request) -> dict[str, object]:
        loaded = request.app.state.predictor
        return {
            "status": "ready",
            "serving_authority_id": loaded.authority.authority_id,
        }

    @application.get("/model-info")
    async def model_info(request: Request) -> dict[str, object]:
        return request.app.state.predictor.model_info()

    @application.post("/predict")
    async def predict(request: Request) -> JSONResponse:
        content_type = request.headers.get("content-type", "")
        if not content_type.lower().startswith("multipart/form-data;"):
            raise ServingInputError("invalid_media_type")
        try:
            async with request.form(max_files=1, max_fields=3) as form:
                items = list(form.multi_items())
                counts = Counter(key for key, _ in items)
                if set(counts) != {"image", "view_position", "labs"} or any(
                    count != 1 for count in counts.values()
                ):
                    raise ServingInputError("invalid_multipart_fields")
                values = dict(items)
                image = values["image"]
                view_position = values["view_position"]
                labs = values["labs"]
                if (
                    not isinstance(image, UploadFile)
                    or not isinstance(view_position, str)
                    or not isinstance(labs, str)
                ):
                    raise ServingInputError("invalid_multipart_fields")
                image_content = await image.read(MAX_JPEG_BYTES + 1)
                image_media_type = image.content_type
        except (ServingInputError, _RequestBodyTooLarge):
            raise
        except Exception as exc:
            raise ServingInputError("invalid_multipart") from exc
        loaded = request.app.state.predictor
        validated = validated_serving_input(
            image_content=image_content,
            image_media_type=image_media_type,
            view_position=view_position,
            labs_json=labs,
            transform=loaded.transform,
        )
        probability = loaded.predict(validated)
        manifest = loaded.authority.manifest
        return JSONResponse(
            content={
                "task": manifest["task"]["task_id"],
                "probability": probability,
                "missing_labs": list(validated.missing_labs),
                "serving_authority_id": loaded.authority.authority_id,
                "warning": manifest["warning"],
            }
        )

    return application
