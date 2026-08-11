from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from radfusion.data.symile_preprocess import (
    LAB_FEATURE_COLUMNS,
    LAB_OBSERVED_COLUMNS,
    LAB_VALUE_COLUMNS,
    SymileLabEcdfTransformer,
    load_symile_lab_preprocessor,
    save_symile_lab_preprocessor,
)


def _lab_frame(first_values: list[float | None]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_index, first in enumerate(first_values):
        row: dict[str, object] = {}
        for index, (value_column, observed_column) in enumerate(
            zip(LAB_VALUE_COLUMNS, LAB_OBSERVED_COLUMNS, strict=True)
        ):
            value = first if index == 0 else float(row_index + index + 1)
            row[value_column] = value
            row[observed_column] = value is not None
        rows.append(row)
    return pd.DataFrame(rows, columns=LAB_FEATURE_COLUMNS)


def test_right_ecdf_extrema_ties_missing_replacement_and_order() -> None:
    training = _lab_frame([1.0, 2.0, 2.0, None])
    transformer = SymileLabEcdfTransformer().fit(training)
    evaluation = _lab_frame([0.0, 2.0, 3.0, None])

    transformed = transformer.transform(evaluation)

    assert transformed.shape == (4, 100)
    assert transformed[:, 0].tolist() == pytest.approx([0.0, 1.0, 1.0, 7.0 / 9.0])
    assert transformed[:, 50].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert transformer.get_feature_names_out()[0] == "lab_50802_ecdf"
    assert transformer.get_feature_names_out()[50] == "lab_50802_observed"
    one_observed = SymileLabEcdfTransformer().fit(_lab_frame([4.0]))
    assert one_observed.transform(_lab_frame([4.0, 5.0]))[:, 0].tolist() == [1.0, 1.0]


def test_zero_observed_or_nonfinite_observed_value_fails() -> None:
    with pytest.raises(ValueError):
        SymileLabEcdfTransformer().fit(_lab_frame([None, None]))

    invalid = _lab_frame([1.0])
    invalid.loc[0, LAB_VALUE_COLUMNS[0]] = np.inf
    with pytest.raises(ValueError):
        SymileLabEcdfTransformer().fit(invalid)


def test_fitted_transform_is_unchanged_and_round_trips(tmp_path: Path) -> None:
    training = _lab_frame([1.0, 2.0, 3.0])
    outer_holdout = _lab_frame([100.0, None])
    transformer = SymileLabEcdfTransformer().fit(training)
    before = tuple(value.copy() for value in transformer.sorted_observed_values_)
    expected = transformer.transform(outer_holdout)
    path = save_symile_lab_preprocessor(transformer, tmp_path / "preprocessor.skops")
    restored = load_symile_lab_preprocessor(path)

    assert all(
        np.array_equal(left, right)
        for left, right in zip(before, transformer.sorted_observed_values_, strict=True)
    )
    assert np.array_equal(restored.transform(outer_holdout), expected)
