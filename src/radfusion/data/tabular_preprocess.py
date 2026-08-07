"""Build train-fitted preprocessing for RSNA DICOM metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from radfusion.data.errors import ManifestBuildError
from radfusion.utils.skops_io import load_skops, save_skops

CONTINUOUS_FEATURES = (
    "age_model_years",
    "pixel_spacing_row_mm",
    "pixel_spacing_col_mm",
)
CATEGORICAL_FEATURES = ("sex", "view_position")
BINARY_FEATURES = (
    "age_is_implausible",
    "age_years_missing",
    "sex_missing",
    "view_position_missing",
    "pixel_spacing_row_mm_missing",
    "pixel_spacing_col_mm_missing",
)
SOURCE_FEATURES = (
    "age_years",
    "age_is_implausible",
    "sex",
    "view_position",
    "pixel_spacing_row_mm",
    "pixel_spacing_col_mm",
)
METADATA_INPUT_POLICY_VERSION = "rsna-metadata-input-v1"
_SOURCE_FEATURE_CONTRACT = {
    "age_years": ("numeric", "allowed"),
    "age_is_implausible": ("boolean", "forbidden"),
    "sex": ("categorical_string", "allowed"),
    "view_position": ("categorical_string", "allowed"),
    "pixel_spacing_row_mm": ("numeric", "allowed"),
    "pixel_spacing_col_mm": ("numeric", "allowed"),
}


def metadata_input_contract() -> dict[str, Any]:
    """Return the exact raw-input contract for serialized metadata pipelines."""
    return {
        "policy_version": METADATA_INPUT_POLICY_VERSION,
        "features": [
            {
                "name": name,
                "type_category": _SOURCE_FEATURE_CONTRACT[name][0],
                "missing_values": _SOURCE_FEATURE_CONTRACT[name][1],
            }
            for name in SOURCE_FEATURES
        ],
        "fitted_preprocessing_embedded": True,
    }


class RsnaMetadataFeatures(BaseEstimator, TransformerMixin):
    """Derive bounded age and explicit metadata missingness indicators."""

    def fit(self, features: pd.DataFrame, target: object = None) -> RsnaMetadataFeatures:
        """Validate the input columns."""
        self._validate_columns(features)
        self.feature_names_in_ = np.asarray(SOURCE_FEATURES, dtype=object)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Derive model-ready metadata columns from a copy of the input."""
        self._validate_columns(features)
        transformed = features.loc[:, SOURCE_FEATURES].copy()
        age = pd.to_numeric(transformed["age_years"], errors="coerce")
        transformed["age_model_years"] = age.clip(lower=0.0, upper=120.0)
        transformed["age_is_implausible"] = transformed["age_is_implausible"].astype("int8")
        for column in CATEGORICAL_FEATURES:
            values = transformed[column].astype(object)
            transformed[column] = values.where(values.notna(), None)
        for column in (
            "age_years",
            "sex",
            "view_position",
            "pixel_spacing_row_mm",
            "pixel_spacing_col_mm",
        ):
            transformed[f"{column}_missing"] = transformed[column].isna().astype("int8")
        return transformed.loc[:, [*CONTINUOUS_FEATURES, *CATEGORICAL_FEATURES, *BINARY_FEATURES]]

    @staticmethod
    def _validate_columns(features: pd.DataFrame) -> None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("RSNA metadata preprocessing requires a pandas DataFrame")
        if len(features.columns) != len(set(features.columns)):
            raise ValueError("RSNA metadata contains duplicate columns")
        missing = sorted(set(SOURCE_FEATURES) - set(features.columns))
        if missing:
            raise ValueError(f"RSNA metadata is missing required columns: {missing}")
        unexpected = sorted(set(features.columns) - set(SOURCE_FEATURES))
        if unexpected:
            raise ValueError(f"RSNA metadata contains unexpected columns: {unexpected}")
        if tuple(features.columns) != SOURCE_FEATURES:
            raise ValueError("RSNA metadata columns are not in the required order")


def validate_metadata_pipeline(model: object) -> Pipeline:
    """Validate that a loaded estimator embeds the fitted metadata input pipeline."""
    if not isinstance(model, Pipeline):
        raise ValueError("Serialized metadata model must be a scikit-learn pipeline")
    preprocessor = model.named_steps.get("preprocess")
    if not isinstance(preprocessor, Pipeline):
        raise ValueError("Serialized metadata model is missing its preprocessing pipeline")
    validate_fitted_rsna_preprocessor(preprocessor)
    return model


def build_rsna_preprocessor() -> Pipeline:
    """Return an unfitted reusable RSNA metadata preprocessing pipeline."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    missing_values=None,
                    strategy="constant",
                    fill_value="<missing>",
                ),
            ),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
            ),
        ]
    )
    columns = ColumnTransformer(
        [
            ("continuous", numeric, list(CONTINUOUS_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ("binary", "passthrough", list(BINARY_FEATURES)),
        ],
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    ).set_output(transform="pandas")
    return Pipeline([("metadata", RsnaMetadataFeatures()), ("columns", columns)])


def fit_rsna_preprocessor(samples: pa.Table, splits: pa.Table) -> Pipeline:
    """Fit preprocessing exclusively on samples assigned to the training split."""
    sample_frame = samples.to_pandas()
    split_frame = splits.to_pandas()
    assignments = split_frame.loc[split_frame["split_name"] == "train", ["sample_id"]]
    training = sample_frame.merge(assignments, on="sample_id", validate="one_to_one")
    if training.empty:
        raise ManifestBuildError("Cannot fit metadata preprocessing without training samples")
    training = training.loc[:, SOURCE_FEATURES]
    pipeline = build_rsna_preprocessor()
    pipeline.fit(training)
    return validate_fitted_rsna_preprocessor(pipeline)


def validate_fitted_rsna_preprocessor(preprocessor: object) -> Pipeline:
    """Validate the exact fitted RSNA metadata preprocessing pipeline."""
    if not isinstance(preprocessor, Pipeline) or tuple(preprocessor.named_steps) != (
        "metadata",
        "columns",
    ):
        raise ValueError("RSNA preprocessor must contain exactly metadata and columns steps")
    metadata = preprocessor.named_steps["metadata"]
    columns = preprocessor.named_steps["columns"]
    if not isinstance(metadata, RsnaMetadataFeatures) or not isinstance(columns, ColumnTransformer):
        raise ValueError("RSNA preprocessor contains invalid top-level steps")
    fitted_names = getattr(metadata, "feature_names_in_", None)
    if fitted_names is None or tuple(fitted_names.tolist()) != SOURCE_FEATURES:
        raise ValueError("RSNA preprocessor has an invalid fitted input contract")
    try:
        check_is_fitted(columns)
    except NotFittedError as exc:
        raise ValueError("RSNA preprocessor is not fitted") from exc
    expected_transformers = (
        ("continuous", tuple(CONTINUOUS_FEATURES)),
        ("categorical", tuple(CATEGORICAL_FEATURES)),
        ("binary", tuple(BINARY_FEATURES)),
    )
    observed_transformers = tuple(
        (name, tuple(feature_names)) for name, _, feature_names in columns.transformers
    )
    if observed_transformers != expected_transformers:
        raise ValueError("RSNA preprocessor has an invalid column assignment")
    numeric = columns.named_transformers_.get("continuous")
    categorical = columns.named_transformers_.get("categorical")
    binary = next(transformer for name, transformer, _ in columns.transformers if name == "binary")
    if not isinstance(numeric, Pipeline) or tuple(numeric.named_steps) != ("impute", "scale"):
        raise ValueError("RSNA preprocessor has an invalid continuous pipeline")
    if not isinstance(categorical, Pipeline) or tuple(categorical.named_steps) != (
        "impute",
        "encode",
    ):
        raise ValueError("RSNA preprocessor has an invalid categorical pipeline")
    numeric_imputer = numeric.named_steps["impute"]
    scaler = numeric.named_steps["scale"]
    categorical_imputer = categorical.named_steps["impute"]
    encoder = categorical.named_steps["encode"]
    if (
        not isinstance(numeric_imputer, SimpleImputer)
        or numeric_imputer.strategy != "median"
        or numeric_imputer.add_indicator
        or not isinstance(scaler, StandardScaler)
        or not scaler.with_mean
        or not scaler.with_std
    ):
        raise ValueError("RSNA preprocessor continuous semantics are invalid")
    if (
        not isinstance(categorical_imputer, SimpleImputer)
        or categorical_imputer.missing_values is not None
        or categorical_imputer.strategy != "constant"
        or categorical_imputer.fill_value != "<missing>"
        or categorical_imputer.add_indicator
        or not isinstance(encoder, OneHotEncoder)
        or encoder.handle_unknown != "ignore"
        or encoder.sparse_output
        or encoder.dtype != np.float64
        or encoder.drop is not None
        or encoder.min_frequency is not None
        or encoder.max_categories is not None
    ):
        raise ValueError("RSNA preprocessor categorical semantics are invalid")
    if (
        binary != "passthrough"
        or columns.remainder != "drop"
        or columns.sparse_threshold != 0.0
        or columns.verbose_feature_names_out is not True
    ):
        raise ValueError("RSNA preprocessor output semantics are invalid")
    transformed_names = tuple(str(name) for name in columns.get_feature_names_out())
    if (
        not transformed_names
        or len(transformed_names) != len(set(transformed_names))
        or any(not name for name in transformed_names)
    ):
        raise ValueError("RSNA preprocessor transformed feature names are invalid")
    return preprocessor


def transformed_rsna_feature_names(preprocessor: object) -> tuple[str, ...]:
    """Return the fitted transformed feature names in deterministic order."""
    fitted = validate_fitted_rsna_preprocessor(preprocessor)
    columns = fitted.named_steps["columns"]
    return tuple(str(name) for name in columns.get_feature_names_out())


def fitted_rsna_preprocessor_contract(preprocessor: object) -> dict[str, Any]:
    """Describe the fitted dense RSNA metadata representation."""
    feature_names = transformed_rsna_feature_names(preprocessor)
    return {
        "input": metadata_input_contract(),
        "transformed_feature_names": list(feature_names),
        "transformed_dimension": len(feature_names),
        "output_structure": "dense",
        "output_dtype": "float64",
    }


def transform_rsna_metadata(preprocessor: object, features: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted RSNA preprocessor and validate its dense finite output."""
    fitted = validate_fitted_rsna_preprocessor(preprocessor)
    transformed = fitted.transform(features)
    columns = fitted.named_steps["columns"]
    expected_names = tuple(str(name) for name in columns.get_feature_names_out())
    if not isinstance(transformed, pd.DataFrame) or tuple(transformed.columns) != expected_names:
        raise ValueError("RSNA preprocessor returned an invalid transformed feature order")
    try:
        values = transformed.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("RSNA preprocessor returned nonnumeric values") from exc
    if values.ndim != 2 or values.shape[1] != len(expected_names) or not np.isfinite(values).all():
        raise ValueError("RSNA preprocessor returned an invalid dense finite matrix")
    return transformed


def save_preprocessor(preprocessor: Pipeline, path: str | Path) -> Path:
    """Serialize a fitted preprocessing pipeline with skops."""
    return save_skops(validate_fitted_rsna_preprocessor(preprocessor), path)


def load_preprocessor(path: str | Path) -> Pipeline:
    """Load a fitted preprocessing pipeline from a trusted skops artifact."""
    preprocessor = load_skops(path)
    return validate_fitted_rsna_preprocessor(preprocessor)
