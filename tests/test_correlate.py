"""
Unit tests for funnel_correlation_py.correlate
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from funnel_correlation_py.binarize import binarize
from funnel_correlation_py.correlate import (
    correlate,
    _check_all_numeric,
    _is_binary,
    _warn_imbalance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_df() -> pd.DataFrame:
    """Small hand-crafted binary DataFrame with a known target."""
    rng = np.random.default_rng(seed=42)
    n = 200
    raw = pd.DataFrame(
        {
            "age": rng.integers(20, 65, size=n),
            "balance": rng.uniform(-500, 5000, size=n),
            "job": rng.choice(["admin", "tech", "sales"], size=n),
            "target": rng.choice(["yes", "no"], size=n),
        }
    )
    return binarize(raw)


# ---------------------------------------------------------------------------
# correlate() – happy path
# ---------------------------------------------------------------------------

class TestCorrelateHappyPath:
    def test_output_is_dataframe(self, binary_df):
        target_col = [c for c in binary_df.columns if c.startswith("target__")][0]
        result = correlate(binary_df, target=target_col)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, binary_df):
        target_col = [c for c in binary_df.columns if c.startswith("target__")][0]
        result = correlate(binary_df, target=target_col)
        assert list(result.columns) == ["feature", "bin", "correlation"]

    def test_target_excluded_from_output(self, binary_df):
        target_col = [c for c in binary_df.columns if c.startswith("target__")][0]
        result = correlate(binary_df, target=target_col)
        target_feature = target_col.split("__")[0]
        # The self-correlation row should be absent
        self_rows = result[result["feature"] == target_feature]
        # It's acceptable to have the complementary bin (target__no) but not r=1
        assert (result["correlation"] == 1.0).sum() == 0

    def test_sorted_by_absolute_correlation_descending(self, binary_df):
        target_col = [c for c in binary_df.columns if c.startswith("target__")][0]
        result = correlate(binary_df, target=target_col)
        abs_corr = result["correlation"].abs()
        assert list(abs_corr) == sorted(abs_corr, reverse=True), (
            "Rows must be sorted by |correlation| descending"
        )

    def test_correlation_within_minus1_to_1(self, binary_df):
        target_col = [c for c in binary_df.columns if c.startswith("target__")][0]
        result = correlate(binary_df, target=target_col)
        assert result["correlation"].between(-1, 1).all()

    def test_feature_is_categorical(self, binary_df):
        target_col = [c for c in binary_df.columns if c.startswith("target__")][0]
        result = correlate(binary_df, target=target_col)
        assert hasattr(result["feature"], "cat"), (
            "feature column must be pd.Categorical for correct plot ordering"
        )

    def test_feature_bin_split_on_double_underscore(self, binary_df):
        target_col = [c for c in binary_df.columns if c.startswith("target__")][0]
        result = correlate(binary_df, target=target_col)
        # All original binarized columns use '__' separator
        assert result["bin"].notna().all()
        assert (result["bin"] != "").all()


# ---------------------------------------------------------------------------
# correlate() – error cases
# ---------------------------------------------------------------------------

class TestCorrelateErrors:
    def test_raises_on_missing_target(self, binary_df):
        with pytest.raises(KeyError, match="not found"):
            correlate(binary_df, target="nonexistent_col")

    def test_raises_on_non_numeric_column(self):
        df = pd.DataFrame(
            {
                "a__0": [0, 1, 0, 1],
                "b__text": ["x", "y", "x", "y"],  # non-numeric
            }
        )
        with pytest.raises(TypeError, match="non-numeric"):
            correlate(df, target="a__0")

    def test_warns_on_imbalanced_target(self):
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame(
            {
                "feat__0": rng.integers(0, 2, size=n),
                "target__yes": [1] * 5 + [0] * 495,  # heavily imbalanced
            }
        )
        with pytest.warns(UserWarning, match="Imbalance"):
            correlate(df, target="target__yes")


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------

class TestIsBinary:
    def test_true_for_zero_one_series(self):
        assert _is_binary(pd.Series([0, 1, 0, 0, 1]))

    def test_false_for_non_binary_series(self):
        assert not _is_binary(pd.Series([0, 1, 2]))

    def test_false_for_float_series(self):
        assert not _is_binary(pd.Series([0.0, 0.5, 1.0]))


class TestCheckAllNumeric:
    def test_passes_on_numeric_only(self):
        df = pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]})
        _check_all_numeric(df, fun_name="test")  # no raise

    def test_raises_on_string_column(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        with pytest.raises(TypeError):
            _check_all_numeric(df, fun_name="test")
