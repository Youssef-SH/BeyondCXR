"""Exercise the mounted serving application through its HTTP boundary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from beyondcxr.release.serving import synthetic_serving_request, validate_serving_api_responses
from beyondcxr.serving.authority import validate_serving_authority
from beyondcxr.serving.predictor import serving_model_info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args(argv)
    expected_model_info = serving_model_info(
        validate_serving_authority(args.authority, package_root=args.package_root)
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "beyondcxr.serving.cli",
            "--authority",
            str(args.authority),
            "--package-root",
            str(args.package_root),
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
        ]
    )
    try:
        health = _request("GET", "/health")
        model = _request("GET", "/model-info")
        image, labs = synthetic_serving_request()
        prediction = _request("POST", "/predict", _multipart(image, labs))
        validate_serving_api_responses(
            health[0],
            health[1],
            model[0],
            model[1],
            prediction[0],
            prediction[1],
            args.authority.name,
            expected_model_info=expected_model_info,
            expected_missing_labs=[key for key, value in labs.items() if value is None],
        )
    finally:
        _stop_process(process)
    return 0


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.terminate()
    except OSError:
        return
    try:
        process.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
        process.wait()
    except OSError:
        pass


def _request(method: str, path: str, body: bytes | None = None) -> tuple[int, object]:
    request = urllib.request.Request(f"http://127.0.0.1:8765{path}", data=body, method=method)
    if body is not None:
        request.add_header("Content-Type", "multipart/form-data; boundary=beyondcxr")
    for _ in range(100):
        try:
            with urllib.request.urlopen(request, timeout=1) as response:
                return response.status, json.loads(response.read())
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("Serving application did not become ready")


def _multipart(image: bytes, labs: dict[str, float | None]) -> bytes:
    boundary = b"--beyondcxr\r\n"
    return b"".join(
        (
            boundary
            + b'Content-Disposition: form-data; name="image"; filename="synthetic.jpg"\r\n'
            + b"Content-Type: image/jpeg\r\n\r\n"
            + image
            + b"\r\n",
            boundary + b'Content-Disposition: form-data; name="view_position"\r\n\r\nAP\r\n',
            boundary
            + b'Content-Disposition: form-data; name="labs"\r\n\r\n'
            + json.dumps(labs).encode()
            + b"\r\n",
            b"--beyondcxr--\r\n",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
