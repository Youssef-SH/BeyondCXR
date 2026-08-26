# Controlled research serving

BeyondCXR serving is a research-only execution surface for the predeclared three-member
CXR-plus-laboratories missingness-aware gated ensemble. Research use only. BeyondCXR is not
intended for clinical decision-making, is not a medical device, does not replace physician
judgment, and predicts a radiology-derived Pneumonia finding rather than confirmed infectious
pneumonia.

## Predictor and authority

The service loads one explicit immutable `serving-authority-<sha256>` control. The authority binds
the ordered final packages for seeds 17, 42, and 2026, their byte and fitted-state witnesses, the
Symile bundle and split, the strict task, preprocessing contracts, development-derived thresholds,
global-result provenance, sealed science-execution commit and lock witnesses, and distinct serving-
release commit and lock witnesses. Both provenance records participate in authority identity. The
science record comes from the validated pre-test freeze; the serving-release record is supplied
explicitly when the authority is published. The authority is a deployment control, not a fourth
scientific object. It never discovers models through MLflow, performance ranking, directory globs,
or a mutable latest pointer.

Each member is reconstructed with the final-package validator and safe package-contained state;
serving does not fetch pretrained weights. For every request, the three raw logits are averaged in
the prespecified seed order and sigmoid is applied once. Probabilities are not averaged,
calibrated, thresholded, or used to choose a member. ECG is not served because the prespecified
primary model is the CXR-plus-labs gated ensemble, irrespective of observed results.

Publishing a serving authority requires the validated Symile global result, its 14 bound
package-level prediction evidences and held-out projection, all 14 final packages, and explicit
serving-release provenance. Global-result validation completes before the primary gated packages
for seeds 17, 42, and 2026 are selected. Re-publication succeeds only when an existing authority
has the same identity, file membership, and bytes; divergent content is rejected.

## API

The v1 surface has exactly three endpoints:

- `GET /health` reports readiness only after the authority and all members load.
- `GET /model-info` reports public-safe authority, package, seed, preprocessing, threshold, and
  provenance metadata. Thresholds are informational research metadata.
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

Install the serving extra and supply explicit operational paths:

```bash
uv sync --locked --extra serving
make symile-serve \
  AUTHORITY=/approved/private/serving-authority-SHA256 \
  PACKAGE_ROOT=/approved/private/models/symile/final/packages \
  DEVICE=cpu HOST=127.0.0.1 PORT=8000
```

Missing or invalid authority state fails startup. CPU and explicit CUDA operation are supported;
device selection does not change authority identity. See the
[synthetic checkerboard request](../examples/symile_serving/) for transport examples.

## Docker

BeyondCXR project content in the public image is limited to distributable project files;
dependencies are installed from the recorded environment. The image contains no dataset, medical
image, waveform, real laboratory row, prediction, credential, agreement, MLflow database, or
trained package. Unless separate distribution authorization exists, mount the authority and
packages read-only:

```bash
docker build -t beyondcxr-serving .
docker run --rm -p 8000:8000 \
  --mount type=bind,src=/approved/private/serving-authority-SHA256,\
dst=/artifacts/serving-authority-SHA256,readonly \
  --mount type=bind,src=/approved/private/packages,dst=/artifacts/packages,readonly \
  beyondcxr-serving \
  --authority /artifacts/serving-authority-SHA256 \
  --package-root /artifacts/packages --host 0.0.0.0 --port 8000
```

The container downloads no scientific model. An absent or invalid mount terminates startup.

Final release acceptance stages and reconstructs the candidate authority and exercises the three
endpoints through an in-process ASGI request. It then checks image licensing, package import, and
absence of restricted roots. A separate container run mounts the staged authority and restored
final-package tree read-only, reconstructs the predictor inside the image, and exercises `/health`,
`/model-info`, and the synthetic checkerboard `/predict` request. After every final check succeeds,
the authority is atomically published to its final location. The request must return the mounted
authority identity, the research warning, and a finite probability in `[0, 1]`.

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
