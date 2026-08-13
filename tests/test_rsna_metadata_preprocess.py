from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import skops.io as sio

from radfusion.data.rsna_metadata_preprocess import (
    SOURCE_FEATURES,
    RsnaMetadataFeatures,
    build_rsna_preprocessor,
    fit_rsna_preprocessor,
    fitted_rsna_preprocessor_contract,
    load_preprocessor,
    save_preprocessor,
    transform_rsna_metadata,
    transformed_rsna_feature_names,
    validate_fitted_rsna_preprocessor,
)
from radfusion.data.rsna_schemas import RSNA_SAMPLE_SCHEMA, RSNA_SPLIT_SCHEMA


def _tables() -> tuple[pa.Table, pa.Table]:
    samples = []
    splits = []
    values = (
        ("train-a", 10.0, "F", "PA", 0.1, "train"),
        ("train-b", 20.0, None, "AP", 0.2, "train"),
        ("validation", 100.0, "M", "LL", 5.0, "validation"),
    )
    for name, age, sex, view, spacing, split_name in values:
        sample_id = f"rsna:{name}"
        samples.append(
            {
                "sample_id": sample_id,
                "patient_id": name,
                "image_id": name,
                "image_path": f"stage_2_train_images/{name}.dcm",
                "image_rows": 1024,
                "image_columns": 1024,
                "age_years": age,
                "age_is_implausible": False,
                "sex": sex,
                "view_position": view,
                "pixel_spacing_row_mm": spacing,
                "pixel_spacing_col_mm": spacing,
            }
        )
        splits.append(
            {
                "sample_id": sample_id,
                "split_name": split_name,
            }
        )
    return (
        pa.Table.from_pylist(samples, RSNA_SAMPLE_SCHEMA),
        pa.Table.from_pylist(splits, RSNA_SPLIT_SCHEMA),
    )


def test_preprocessor_fits_statistics_and_categories_on_training_only(tmp_path) -> None:
    samples, splits = _tables()
    preprocessor = fit_rsna_preprocessor(samples, splits)
    columns = preprocessor.named_steps["columns"]
    numeric = columns.named_transformers_["continuous"]
    categorical = columns.named_transformers_["categorical"]

    assert numeric.named_steps["impute"].statistics_[0] == 15.0
    categories = categorical.named_steps["encode"].categories_
    assert "<missing>" in categories[0]
    assert None not in categories[0]
    assert "M" not in categories[0]
    assert "LL" not in categories[1]
    transformed = preprocessor.transform(samples.to_pandas().loc[:, SOURCE_FEATURES])
    assert np.isfinite(transformed.to_numpy()).all()

    destination = save_preprocessor(preprocessor, tmp_path / "preprocessor.skops")
    restored = load_preprocessor(destination)
    np.testing.assert_allclose(
        restored.transform(samples.to_pandas().loc[:, SOURCE_FEATURES]),
        transformed,
    )


def test_fitted_preprocessor_contract_and_transform_are_deterministic() -> None:
    samples, splits = _tables()
    preprocessor = fit_rsna_preprocessor(samples, splits)
    features = samples.to_pandas().loc[:, SOURCE_FEATURES]
    first = transform_rsna_metadata(preprocessor, features)
    second = transform_rsna_metadata(preprocessor, features)
    names = transformed_rsna_feature_names(preprocessor)
    contract = fitted_rsna_preprocessor_contract(preprocessor)

    assert tuple(first.columns) == names
    assert names == (
        "continuous__age_model_years",
        "continuous__pixel_spacing_row_mm",
        "continuous__pixel_spacing_col_mm",
        "categorical__sex_<missing>",
        "categorical__sex_F",
        "categorical__view_position_AP",
        "categorical__view_position_PA",
        "binary__age_is_implausible",
        "binary__age_years_missing",
        "binary__sex_missing",
        "binary__view_position_missing",
        "binary__pixel_spacing_row_mm_missing",
        "binary__pixel_spacing_col_mm_missing",
    )
    assert contract["transformed_feature_names"] == list(names)
    assert contract["transformed_dimension"] == len(names)
    assert contract["output_structure"] == "dense"
    assert contract["output_dtype"] == "float64"
    assert first["categorical__sex_<missing>"].tolist() == [0.0, 1.0, 0.0]
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
    assert first.to_numpy().dtype == np.float64
    assert np.isfinite(first.to_numpy()).all()


def test_preprocessor_handles_unseen_categories_and_missing_values() -> None:
    samples, splits = _tables()
    preprocessor = fit_rsna_preprocessor(samples, splits)
    features = pd.DataFrame(
        [[None, False, "M", "LL", None, None]],
        columns=SOURCE_FEATURES,
    )

    transformed = transform_rsna_metadata(preprocessor, features)

    assert transformed.filter(like="categorical__").to_numpy().tolist() == [[0.0, 0.0, 0.0, 0.0]]
    assert transformed["binary__age_years_missing"].tolist() == [1]
    assert transformed["binary__pixel_spacing_row_mm_missing"].tolist() == [1]
    assert np.isfinite(transformed.to_numpy()).all()


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
def test_categorical_missing_markers_share_one_representation(missing: object) -> None:
    training = pd.DataFrame(
        [
            [10.0, False, "F", "PA", 0.1, 0.1],
            [20.0, False, None, None, 0.2, 0.2],
        ],
        columns=SOURCE_FEATURES,
    )
    preprocessor = build_rsna_preprocessor().fit(training)
    features = pd.DataFrame(
        [[30.0, False, missing, missing, 0.15, 0.15]],
        columns=SOURCE_FEATURES,
    )

    transformed = transform_rsna_metadata(preprocessor, features)

    assert transformed["categorical__sex_<missing>"].tolist() == [1.0]
    assert transformed["categorical__view_position_<missing>"].tolist() == [1.0]
    assert transformed["binary__sex_missing"].tolist() == [1]
    assert transformed["binary__view_position_missing"].tolist() == [1]


def test_preprocessor_serialization_rejects_unfitted_or_malformed_state(tmp_path) -> None:
    unfitted = build_rsna_preprocessor()
    destination = tmp_path / "unfitted.skops"
    with pytest.raises(ValueError):
        save_preprocessor(unfitted, destination)
    assert not destination.exists()

    sio.dump(unfitted, destination)
    with pytest.raises(ValueError):
        load_preprocessor(destination)

    samples, splits = _tables()
    malformed = fit_rsna_preprocessor(samples, splits)
    categorical = malformed.named_steps["columns"].named_transformers_["categorical"]
    categorical.named_steps["encode"].handle_unknown = "error"
    with pytest.raises(ValueError):
        validate_fitted_rsna_preprocessor(malformed)


@pytest.mark.parametrize(
    ("component", "attribute", "value"),
    [
        ("columns", "remainder", "passthrough"),
        ("columns", "sparse_threshold", 0.5),
        ("columns", "verbose_feature_names_out", False),
        ("numeric_imputer", "strategy", "mean"),
        ("numeric_imputer", "add_indicator", True),
        ("scaler", "with_mean", False),
        ("categorical_imputer", "add_indicator", True),
        ("categorical_imputer", "missing_values", np.nan),
        ("encoder", "drop", "first"),
        ("encoder", "min_frequency", 2),
        ("encoder", "max_categories", 2),
    ],
)
def test_preprocessor_rejects_meaning_changing_sklearn_settings(
    component: str,
    attribute: str,
    value: object,
) -> None:
    samples, splits = _tables()
    preprocessor = fit_rsna_preprocessor(samples, splits)
    columns = preprocessor.named_steps["columns"]
    numeric = columns.named_transformers_["continuous"]
    categorical = columns.named_transformers_["categorical"]
    components = {
        "columns": columns,
        "numeric_imputer": numeric.named_steps["impute"],
        "scaler": numeric.named_steps["scale"],
        "categorical_imputer": categorical.named_steps["impute"],
        "encoder": categorical.named_steps["encode"],
    }
    setattr(components[component], attribute, value)

    with pytest.raises(ValueError):
        validate_fitted_rsna_preprocessor(preprocessor)


def test_preprocessor_transform_rejects_nonfinite_output() -> None:
    samples, splits = _tables()
    preprocessor = fit_rsna_preprocessor(samples, splits)
    numeric = preprocessor.named_steps["columns"].named_transformers_["continuous"]
    numeric.named_steps["scale"].scale_[0] = np.nan

    with pytest.raises(ValueError):
        transform_rsna_metadata(
            preprocessor,
            samples.to_pandas().loc[:, SOURCE_FEATURES],
        )


def test_metadata_features_clip_age_and_add_missingness() -> None:
    frame = pd.DataFrame(
        {
            "age_years": [155.0, None],
            "age_is_implausible": [True, False],
            "sex": ["F", None],
            "view_position": ["PA", None],
            "pixel_spacing_row_mm": [0.1, None],
            "pixel_spacing_col_mm": [0.1, None],
        }
    )
    transformed = RsnaMetadataFeatures().fit_transform(frame)

    assert transformed["age_model_years"].iloc[0] == 120.0
    assert transformed["age_is_implausible"].tolist() == [1, 0]
    assert transformed["age_years_missing"].tolist() == [0, 1]
    assert transformed["sex_missing"].tolist() == [0, 1]


def test_metadata_features_reject_unapproved_columns() -> None:
    frame = pd.DataFrame(
        {
            "age_years": [50.0],
            "age_is_implausible": [False],
            "sex": ["F"],
            "view_position": ["PA"],
            "pixel_spacing_row_mm": [0.1],
            "pixel_spacing_col_mm": [0.1],
            "patient_id": ["poison"],
        }
    )

    with np.testing.assert_raises_regex(ValueError, "unexpected columns"):
        RsnaMetadataFeatures().fit_transform(frame)


def test_metadata_features_reject_reordered_and_duplicate_columns() -> None:
    frame = pd.DataFrame(
        [[50.0, False, "F", "PA", 0.1, 0.1]],
        columns=SOURCE_FEATURES,
    )
    reordered = frame.loc[:, tuple(reversed(SOURCE_FEATURES))]
    duplicate = frame.copy()
    duplicate.columns = (*SOURCE_FEATURES[:-1], SOURCE_FEATURES[0])

    with np.testing.assert_raises_regex(ValueError, "required order"):
        RsnaMetadataFeatures().fit_transform(reordered)
    with np.testing.assert_raises_regex(ValueError, "duplicate"):
        RsnaMetadataFeatures().fit_transform(duplicate)
