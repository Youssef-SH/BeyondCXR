# Synthetic serving request

This example is artificial and exists only to demonstrate the research API schema. It is not a
medical image or patient record and must not be interpreted clinically.

Generate the checkerboard JPEG and complete 50-key laboratory JSON:

```bash
uv run --extra serving python examples/symile_serving/generate_synthetic_request.py
```

After starting the accepted service with `make serve`, submit them as multipart form data:

```bash
curl --fail-with-body http://127.0.0.1:8000/predict \
  -F image=@synthetic-serving-request/synthetic-checkerboard.jpg\;type=image/jpeg \
  -F view_position=AP \
  -F 'labs=<synthetic-serving-request/synthetic-labs.json'
```

The response contains a raw research probability, missing-lab keys, the serving-authority ID, and
the non-clinical warning. It never contains a thresholded decision.

Representative schema only (the probability and authority placeholder are not scientific
results):

```json
{
  "task": "pneumonia_strict",
  "probability": 0.5,
  "missing_labs": ["lab_<item_id>"],
  "serving_authority_id": "serving-authority-<sha256>",
  "warning": "Research use only. BeyondCXR is not intended for clinical decision-making, is not a medical device, does not replace physician judgment, and predicts a radiology-derived Pneumonia finding rather than confirmed infectious pneumonia."
}
```

See [serving](../../docs/serving.md) for startup, input validation, and deployment requirements.
