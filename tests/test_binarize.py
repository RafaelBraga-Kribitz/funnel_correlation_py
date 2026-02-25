"""
Unit tests for funnel_correlation_py.binarize
"""

import numpy as np
import pandas as pd
import pytest

from funnel_correlation_py.binarize import (
    binarize,
    _check_data_types,
    _check_missing,
    _fix_low_cardinality_numeric,
    _fix_high_skew_numeric,
    _lump_rare_categories,
    _drop_zero_variance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df() -> pd.DataFrame:
    """Minimal mixed-type DataFrame: 1 numeric + 1 categorical column."""
    rng = np.random.default_rng(seed=0)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=100),
            "job": rng.choice(["admin", "tech", "sales", "other"], size=100),
        }
    )


@pytest.fixture
def binary_target_df() -> pd.DataFrame:
    """DataFrame suitable for a full binarize->correlate test."""
    rng = np.random.default_rng(seed=1)
    n = 200
    return pd.DataFrame(
        {
            "age": rng.integers(20, 65, size=n),
            "balance": rng.uniform(-500, 10000, size=n),
            "job": rng.choice(["admin", "tech", "sales"], size=n),
            "target": rng.choice(["yes", "no"], size=n),
        }
    )


# ---------------------------------------------------------------------------
# binarize() – happy path
# ---------------------------------------------------------------------------

class TestBinarizeHappyPath:
    def test_output_is_dataframe(self, small_df):
        result = binarize(small_df)
        assert isinstance(result, pd.DataFrame)

    def test_output_is_all_integers(self, small_df):
        result = binarize(small_df)
        assert all(result.dtypes == int), "All columns must be int"

    def test_output_values_are_zero_or_one(self, small_df):
        result = binarize(small_df)
        unique_vals = set(result.values.flatten())
        assert unique_vals.issubset({0, 1}), f"Unexpected values: {unique_vals}"

    def test_column_separator_is_double_underscore(self, small_df):
        result = binarize(small_df)
        assert all("__" in col for col in result.columns), (
            "Every output column must contain '__'"
        )

    def test_n_bins_parameter(self, small_df):
        result_4 = binarize(small_df[["age"]], n_bins=4)
        result_2 = binarize(small_df[["age"]], n_bins=2)
        # More bins -> equal or more columns
        assert result_4.shape[1] >= result_2.shape[1]

    def test_one_hot_false_drops_reference_level(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"] * 10})
        full = binarize(df, one_hot=True)
        dummy = binarize(df, one_hot=False)
        assert dummy.shape[1] == full.shape[1] - 1

    def test_boolean_column_handled(self):
        df = pd.DataFrame({"flag": [True, False, True, False] * 5})
        result = binarize(df)
        assert result.shape[1] >= 1

    def test_zero_variance_columns_dropped(self):
        df = pd.DataFrame({"age": [25] * 50, "job": ["admin"] * 50})
        result = binarize(df)
        # Everything is constant -> no varying columns
        assert result.shape[1] == 0 or all(result.nunique() > 1)

    def test_thresh_infreq_lumps_rare_levels(self):
        """A level appearing in <1% of rows should become '-OTHER'."""
        categories = ["common"] * 95 + ["rare_1"] + ["rare_2"] + ["common"] * 3
        df = pd.DataFrame({"cat": categories})
        result = binarize(df, thresh_infreq=0.02)
        other_cols = [c for c in result.columns if "-OTHER" in c]
        assert len(other_cols) >= 1, "Expected an '-OTHER' column for rare levels"


# ---------------------------------------------------------------------------
# binarize() – error cases
# ---------------------------------------------------------------------------

class TestBinarizeErrors:
    def test_raises_on_missing_values(self):
        df = pd.DataFrame({"age": [25, np.nan, 30]})
        with pytest.raises(ValueError, match="Missing Values"):
            binarize(df)

    def test_raises_on_datetime_column(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2021-01-01", "2022-06-15"])})
        with pytest.raises(TypeError, match="Unacceptable Columns"):
            binarize(df)


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------

class TestCheckDataTypes:
    def test_passes_on_allowed_types(self):
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        _check_data_types(df, fun_name="test")  # should not raise

    def test_raises_on_datetime(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2021-01-01"])})
        with pytest.raises(TypeError):
            _check_data_types(df, fun_name="test")


class TestCheckMissing:
    def test_passes_when_no_nulls(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        _check_missing(df, fun_name="test")  # should not raise

    def test_raises_when_null_present(self):
        df = pd.DataFrame({"x": [1, None, 3]})
        with pytest.raises(ValueError, match="Missing Values"):
            _check_missing(df, fun_name="test")


class TestFixLowCardinalityNumeric:
    def test_converts_low_cardinality_column(self):
        df = pd.DataFrame({"flag": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
        result = _fix_low_cardinality_numeric(df, thresh=6)
        # 2 unique values < 6 -> should become categorical
        assert result["flag"].dtype.name in ("category", "object")

    def test_keeps_high_cardinality_column(self):
        df = pd.DataFrame({"age": list(range(20, 70))})
        result = _fix_low_cardinality_numeric(df, thresh=6)
        assert pd.api.types.is_numeric_dtype(result["age"])


class TestFixHighSkewNumeric:
    def test_converts_highly_skewed_column(self):
        # All values are 0 except one -> quantiles collapse
        data = [0] * 98 + [100, 200]
        df = pd.DataFrame({"skewed": data})
        result = _fix_high_skew_numeric(df, unique_limit=2)
        assert result["skewed"].dtype.name in ("category", "object")


class TestLumpRareCategories:
    def test_lumps_rare_levels(self):
        cats = ["a"] * 80 + ["b"] * 15 + ["rare"] * 5
        df = pd.DataFrame({"cat": cats})
        result = _lump_rare_categories(df, thresh=0.06, other_name="-OTHER")
        assert "-OTHER" in result["cat"].values

    def test_keeps_common_levels(self):
        cats = ["a"] * 80 + ["b"] * 20
        df = pd.DataFrame({"cat": cats})
        result = _lump_rare_categories(df, thresh=0.05, other_name="-OTHER")
        assert set(result["cat"].unique()) == {"a", "b"}

    def test_no_lumping_when_thresh_zero(self):
        cats = ["a"] * 99 + ["z"]
        df = pd.DataFrame({"cat": cats})
        result = _lump_rare_categories(df, thresh=0.0, other_name="-OTHER")
        assert "z" in result["cat"].values


class TestDropZeroVariance:
    def test_drops_constant_column(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [0, 1, 0]})
        result = _drop_zero_variance(df)
        assert "a" not in result.columns
        assert "b" in result.columns
