# Controlled research serving

BeyondCXR serving is a research-only execution surface for the predeclared three-member
CXR-plus-laboratories missingness-aware gated ensemble. Research use only. BeyondCXR is not
intended for clinical decision-making, is not a medical device, does not replace physician
judgment, and predicts a radiology-derived Pneumonia finding rather than confirmed infectious
pneumonia.

## Predictor and authority

The service loads one self-contained `serving-authority-<sha256>` distribution containing the
ordered primary packages for seeds 17, 42, and 2026. Their original package IDs identify fitted
state. The serving authority binds input/output policy, reduction, thresholds, and global-result
lineage; deployment provenance records operational release facts separately.

Startup recursively validates an explicit preservation archive and authenticates the distribution
against that campaign, its result binding, and deployment receipt. It does not select models by
performance, MLflow, or recency.

Each member is reconstructed with the final-package validator and safe package-contained state;
serving does not fetch pretrained weights. For every request, the three raw logits are averaged in
the prespecified seed order and sigmoid is applied once. Probabilities are not averaged,
calibrated, thresholded, or used to choose a member. ECG is not served because the prespecified
primary model is the CXR-plus-labs gated ensemble, irrespective of observed results.

`make release` publishes accepted serving state after scientific validation and host/container
acceptance. See [reproducibility](reproducibility.md#preservation-and-release) for release selection
and prerequisites.

## API

The v1 surface has exactly three endpoints:

- `GET /health` reports readiness only after the authority and all members load.
- `GET /model-info` reports public-safe model metadata captured at authenticated startup. Thresholds are
  informational research metadata.
- `POST /predict` returns one raw research probability and never a diagnostic label.

`POST /predict` requires `multipart/form-data` with exactly one occurrence of `image`,
`view_position`, and `labs`. The image must decode as a nonempty single-frame JPEG, independent of
its filename. PNG, DICOM, NPY, TIFF, WebP, BMP, GIF, malformed content, and other formats are
rejected. `view_position` must be `AP` or `PA`; it is validation metadata and is never a predictor.

`labs` is a JSON object containing exactly the 50 keys `lab_<item_id>` in the declared Symile item
set. Incoming order is irrelevant. A finite JSON number means observed and `null` means unobserved.
Booleans, strings, NaN, infinities, missing keys, and additional keys are rejected. The response
lists missing keys in canonical order.

The locked serving decoder and runtime convert the JPEG once to grayscale. The authority-bound
spatial policy preserves aspect ratio, sets the shorter side to 320 pixels, floors the proportional
longer-side dimension, uses Pillow bilinear resampling, and applies a deterministic centered
320-square crop whose origin uses Python's nearest-even `round`. The resulting contiguous `float32`
image in `[0, 1]` then passes through the package-bound deterministic TorchXRayVision evaluation
transform to obtain a contiguous `float32` `1 × 224 × 224` tensor.

There is no augmentation, random crop, new normalization, fitted calibration, or request-time lab
fitting. Each package's immutable full-development ECDF preprocessor transforms the raw labs.

## Running locally

Publish and validate the release, then run its sole installed accepted serving state:

```bash
make release
make serve
```

The normal service defaults to CPU on `127.0.0.1:8000`. Missing accepted state instructs the
operator to run the release workflow; multiple valid states fail as ambiguous. Expert operational
overrides remain available through `python -m beyondcxr.serving.cli` with explicit authority,
deployment-provenance, result-binding, and release-archive paths. Those coordinates and runtime
choices do not alter scientific serving identity.

The authority distribution contains the exact selected package copies under their original package
IDs. Runtime independently validates those copies, the complete serving authority, and the canonical
release archive relationship. The original restored campaign package directory is not a runtime
input. Missing or invalid authority or release evidence fails startup. CPU and explicit CUDA
operation are supported; device selection does not change authority identity. See the
[synthetic checkerboard request](../examples/symile_serving/) for transport examples.

## Docker

BeyondCXR project content in the public image is limited to distributable project files;
dependencies are installed from the recorded environment. The image contains no dataset, medical
image, waveform, real laboratory row, prediction, credential, agreement, MLflow database, or
trained package. Unless separate distribution authorization exists, mount the self-contained
authority and its release evidence read-only:

```bash
docker build -t beyondcxr-serving .
docker run --rm -p 8000:8000 \
  --mount type=bind,src=/approved/private/serving-authority-SHA256,\
dst=/artifacts/serving-authority-SHA256,readonly \
  --mount type=bind,src=/approved/private/deployment.json,dst=/artifacts/deployment.json,readonly \
  --mount type=bind,src=/approved/private/binding.json,dst=/artifacts/binding.json,readonly \
  --mount type=bind,src=/approved/private/symile-release.zip,dst=/artifacts/release.zip,readonly \
  beyondcxr-serving \
  --authority /artifacts/serving-authority-SHA256 \
  --deployment-provenance /artifacts/deployment.json \
  --result-binding /artifacts/binding.json \
  --release-archive /artifacts/release.zip --host 0.0.0.0 --port 8000
```

The container downloads no scientific model. An absent or invalid mount terminates startup.

Release acceptance exercises `/health`, `/model-info`, and a synthetic `/predict` request both
in-process and inside the built container before publishing accepted serving state.

## Privacy and failures

The service has no patient database, upload archive, request archive, response archive, or
prediction cache. BeyondCXR retains no submitted image, laboratory input, patient identifier,
missingness state, or prediction after request processing. Operational handling does not log
patient-level inputs or outputs. Stable client errors name only categories such as invalid media,
image, view, JSON, keys, or numeric values; private paths, package contents, stack traces, and
credentials are not returned.

The Uvicorn access log is disabled so request targets and routine request metadata are not emitted
by the project command. Operators must preserve that boundary in any external proxy or platform.

Trained packages and serving authorities remain restricted unless their source agreements and
project policy explicitly permit distribution. Synthetic examples demonstrate transport only and
must never be presented as clinical patients.
