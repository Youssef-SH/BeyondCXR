"""Fit and apply the frozen outer-fold Symile laboratory ECDF transform."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from radfusion.data.symile_schemas import LAB_ITEM_IDS
from radfusion.utils.skops_io import load_skops, save_skops

LAB_ECDF_POLICY_VERSION = "symile-outer-training-right-ecdf-v1"
LAB_VALUE_COLUMNS = tuple(f"lab_{item}_value" for item in LAB_ITEM_IDS)
LAB_OBSERVED_COLUMNS = tuple(f"lab_{item}_observed" for item in LAB_ITEM_IDS)
LAB_FEATURE_COLUMNS = LAB_VALUE_COLUMNS + LAB_OBSERVED_COLUMNS
LAB_TRANSFORMED_DIMENSION = 100


class SymileLabEcdfTransformer(TransformerMixin, BaseEstimator):
    """Transform raw labs with outer-training right-rank ECDFs and observedness."""

    def fit(self, values: pd.DataFrame, targets: object = None) -> SymileLabEcdfTransformer:
        """Fit one ECDF and missing replacement per laboratory."""
        del targets
        frame = _validated_frame(values)
        sorted_values: list[np.ndarray] = []
        replacements: list[float] = []
        for value_column, observed_column in zip(
            LAB_VALUE_COLUMNS, LAB_OBSERVED_COLUMNS, strict=True
        ):
            raw = frame[value_column].to_numpy(dtype=np.float64)
            observed = frame[observed_column].to_numpy(dtype=bool)
            if np.any(observed & ~np.isfinite(raw)):
                raise ValueError("Observed Symile laboratory values must be finite")
            selected = np.sort(raw[observed])
            if not len(selected):
                raise ValueError("Every laboratory requires an observed outer-training value")
            ranks = np.searchsorted(selected, selected, side="right") / len(selected)
            replacement = float(np.mean(ranks))
            if not np.isfinite(replacement):
                raise ValueError("Symile laboratory missing replacement must be finite")
            sorted_values.append(selected)
            replacements.append(replacement)
        self.sorted_observed_values_ = tuple(sorted_values)
        self.missing_replacements_ = np.asarray(replacements, dtype=np.float64)
        self.n_features_in_ = LAB_TRANSFORMED_DIMENSION
        self.feature_names_in_ = np.asarray(LAB_FEATURE_COLUMNS, dtype=object)
        return self

    def transform(self, values: pd.DataFrame) -> np.ndarray:
        """Apply the fitted transform without changing fitted state."""
        check_is_fitted(
            self,
            attributes=("sorted_observed_values_", "missing_replacements_"),
        )
        frame = _validated_frame(values)
        transformed = np.empty((len(frame), len(LAB_ITEM_IDS)), dtype=np.float64)
        indicators = frame.loc[:, LAB_OBSERVED_COLUMNS].to_numpy(dtype=np.float64)
        for index, (value_column, observed_column, fitted, replacement) in enumerate(
            zip(
                LAB_VALUE_COLUMNS,
                LAB_OBSERVED_COLUMNS,
                self.sorted_observed_values_,
                self.missing_replacements_,
                strict=True,
            )
        ):
            raw = frame[value_column].to_numpy(dtype=np.float64)
            observed = frame[observed_column].to_numpy(dtype=bool)
            if np.any(observed & ~np.isfinite(raw)):
                raise ValueError("Observed Symile laboratory values must be finite")
            column = np.full(len(frame), replacement, dtype=np.float64)
            column[observed] = np.searchsorted(fitted, raw[observed], side="right") / len(fitted)
            transformed[:, index] = column
        result = np.concatenate((transformed, indicators), axis=1)
        if result.shape != (len(frame), LAB_TRANSFORMED_DIMENSION) or not np.isfinite(result).all():
            raise ValueError("Symile laboratory transformation produced invalid features")
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the frozen transformed feature order."""
        del input_features
        return np.asarray(
            tuple(f"lab_{item}_ecdf" for item in LAB_ITEM_IDS) + LAB_OBSERVED_COLUMNS,
            dtype=object,
        )


def save_symile_lab_preprocessor(transformer: SymileLabEcdfTransformer, path: str | Path) -> Path:
    """Safely serialize one fitted Symile lab transform."""
    check_is_fitted(transformer, attributes=("sorted_observed_values_", "missing_replacements_"))
    return save_skops(transformer, path)


def load_symile_lab_preprocessor(path: str | Path) -> SymileLabEcdfTransformer:
    """Load and validate one fitted Symile lab transform."""
    value = load_skops(path)
    return validate_symile_lab_preprocessor(value)


def validate_symile_lab_preprocessor(value: object) -> SymileLabEcdfTransformer:
    """Validate one fitted transformer and its exact 100-feature contract."""
    if not isinstance(value, SymileLabEcdfTransformer):
        raise TypeError("Symile laboratory preprocessor has an unexpected type")
    check_is_fitted(value, attributes=("sorted_observed_values_", "missing_replacements_"))
    fitted = value.sorted_observed_values_
    replacements = np.asarray(value.missing_replacements_)
    if (
        not isinstance(fitted, tuple)
        or len(fitted) != len(LAB_ITEM_IDS)
        or replacements.shape != (len(LAB_ITEM_IDS),)
        or not np.isfinite(replacements).all()
        or getattr(value, "n_features_in_", None) != LAB_TRANSFORMED_DIMENSION
        or tuple(getattr(value, "feature_names_in_", ())) != LAB_FEATURE_COLUMNS
        or tuple(value.get_feature_names_out())
        != tuple(f"lab_{item}_ecdf" for item in LAB_ITEM_IDS) + LAB_OBSERVED_COLUMNS
    ):
        raise TypeError("Symile laboratory preprocessor contract is invalid")
    for observed in fitted:
        array = np.asarray(observed)
        if (
            array.ndim != 1
            or not len(array)
            or not np.issubdtype(array.dtype, np.floating)
            or not np.isfinite(array).all()
            or np.any(array[1:] < array[:-1])
        ):
            raise TypeError("Symile laboratory fitted ECDF state is invalid")
    return value


def _validated_frame(values: object) -> pd.DataFrame:
    if not isinstance(values, pd.DataFrame):
        raise TypeError("Symile laboratory features must be a pandas DataFrame")
    if tuple(values.columns) != LAB_FEATURE_COLUMNS or values.columns.duplicated().any():
        raise ValueError("Symile laboratory columns must match the exact frozen order")
    for value_column, observed_column in zip(LAB_VALUE_COLUMNS, LAB_OBSERVED_COLUMNS, strict=True):
        observed = values[observed_column]
        boolean = observed.map(lambda value: isinstance(value, (bool, np.bool_)))
        if observed.isna().any() or not boolean.all():
            raise ValueError("Symile laboratory observedness must be Boolean and non-null")
        raw = pd.to_numeric(values[value_column], errors="coerce").to_numpy(dtype=np.float64)
        mask = observed.to_numpy(dtype=bool)
        if np.any(mask != ~np.isnan(raw)):
            raise ValueError("Symile laboratory observedness differs from raw nullity")
    return values
