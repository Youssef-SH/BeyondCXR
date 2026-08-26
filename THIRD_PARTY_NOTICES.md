# Third-party notices

## Software

BeyondCXR installs, but does not vendor, the packages declared in `pyproject.toml` and resolved in
`uv.lock`. Their upstream licenses remain applicable. Principal runtime components include PyTorch,
TorchVision, TorchXRayVision, scikit-learn, skops, LightGBM, NumPy, pandas, Matplotlib, PyArrow,
pydicom, PyYAML, SQLAlchemy, FastAPI, Uvicorn, and MLflow. See each upstream package's license
metadata for its applicable terms.

Built wheels and source distributions retain the repository license and package metadata.
Dependencies remain governed by their own upstream licenses; consult installed package metadata and
the upstream projects for applicable terms.

## Pretrained weight identity

Training uses TorchXRayVision's `densenet121-res224-chex` registry entry, identified upstream as a
DenseNet121 state trained on CheXpert and distributed from the TorchXRayVision release. BeyondCXR
records the stable upstream URL and exact local byte hash used for scientific initialization. The
weight file is not committed, included in source distributions, or embedded in the serving image.
This notice does not assert redistribution rights for that file.

## Datasets and trained artifacts

RSNA challenge data, Symile-MIMIC, MIMIC-CXR-JPG, MIMIC-IV, MIMIC-IV-ECG, and PhysioNet resources
are not distributed here. Their licenses, rules, credentialing, training, citation, and data-use
requirements apply independently. Dataset citations and access obligations are listed in the
[data statement](https://github.com/Youssef-SH/BeyondCXR/blob/main/docs/data_statement.md). The
repository's Apache-2.0 code license grants no rights
to restricted datasets or trained artifacts, and source-data access does not establish permission
to distribute fitted packages.
