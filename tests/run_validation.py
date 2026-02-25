"""
run_validation.py
-----------------
Standalone validation script that exercises all three pipeline steps
without requiring pytest.  Run with:

    PYTHONPATH=src python tests/run_validation.py

Exits with code 0 on success, 1 on any failure.
"""

from __future__ import annotations

import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from funnel_correlation_py.binarize import (
    binarize,
    _check_data_types,
    _check_missing,
    _fix_low_cardinality_numeric,
    _fix_high_skew_numeric,
    _lump_rare_categories,
    _drop_zero_variance,
)
from funnel_correlation_py.correlate import correlate, _is_binary, _check_all_numeric
from funnel_correlation_py.plot import plot_correlation_funnel, _check_column_names


PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def _run(name: str, fn) -> None:
    try:
        fn()
        results.append((PASS, name, ""))
    except Exception:
        results.append((FAIL, name, traceback.format_exc().strip().splitlines()[-1]))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(seed=42)
N = 200

RAW_DF = pd.DataFrame(
    {
        "age":     RNG.integers(18, 70, size=N),
        "balance": RNG.uniform(-500, 5000, size=N),
        "job":     RNG.choice(["admin", "tech", "sales", "retired"], size=N),
        "target":  RNG.choice(["yes", "no"], size=N),
    }
)


# ---------------------------------------------------------------------------
# binarize tests
# ---------------------------------------------------------------------------

def t_binarize_output_is_dataframe():
    result = binarize(RAW_DF)
    assert isinstance(result, pd.DataFrame), "Expected DataFrame"

def t_binarize_all_integers():
    result = binarize(RAW_DF)
    assert all(result.dtypes == int), f"Non-int dtypes: {result.dtypes.unique()}"

def t_binarize_values_are_0_or_1():
    result = binarize(RAW_DF)
    unique = set(result.values.flatten())
    assert unique.issubset({0, 1}), f"Unexpected values: {unique}"

def t_binarize_double_underscore_separator():
    result = binarize(RAW_DF)
    bad = [c for c in result.columns if "__" not in c]
    assert not bad, f"Columns without '__': {bad}"

def t_binarize_raises_on_missing():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    try:
        binarize(df)
        raise AssertionError("Expected ValueError not raised")
    except ValueError as exc:
        assert "Missing Values" in str(exc)

def t_binarize_raises_on_datetime():
    df = pd.DataFrame({"dt": pd.to_datetime(["2021-01-01", "2022-06-15"])})
    try:
        binarize(df)
        raise AssertionError("Expected TypeError not raised")
    except TypeError as exc:
        assert "Unacceptable" in str(exc)

def t_binarize_thresh_infreq_creates_other():
    cats = ["common"] * 95 + ["rare_a"] + ["rare_b"] + ["common"] * 3
    df = pd.DataFrame({"cat": cats})
    result = binarize(df, thresh_infreq=0.02)
    other_cols = [c for c in result.columns if "-OTHER" in c]
    assert len(other_cols) >= 1, "Expected '-OTHER' column"

def t_binarize_one_hot_false_fewer_cols():
    df = pd.DataFrame({"cat": ["a", "b", "c"] * 10})
    full  = binarize(df, one_hot=True)
    dummy = binarize(df, one_hot=False)
    assert dummy.shape[1] <= full.shape[1]

def t_fix_low_cardinality():
    df = pd.DataFrame({"flag": [0, 1] * 5})
    result = _fix_low_cardinality_numeric(df, thresh=6)
    assert result["flag"].dtype.name in ("category", "object")

def t_fix_high_skew():
    data = [0] * 98 + [100, 200]
    df = pd.DataFrame({"skewed": data})
    result = _fix_high_skew_numeric(df, unique_limit=2)
    assert result["skewed"].dtype.name in ("category", "object")

def t_lump_rare_categories():
    cats = ["a"] * 80 + ["b"] * 15 + ["rare"] * 5
    df = pd.DataFrame({"cat": cats})
    result = _lump_rare_categories(df, thresh=0.06, other_name="-OTHER")
    assert "-OTHER" in result["cat"].values

def t_drop_zero_variance():
    df = pd.DataFrame({"constant": [1, 1, 1], "varying": [0, 1, 0]})
    result = _drop_zero_variance(df)
    assert "constant" not in result.columns
    assert "varying" in result.columns


# ---------------------------------------------------------------------------
# correlate tests
# ---------------------------------------------------------------------------

BINARY_DF = binarize(RAW_DF)
TARGET_COL = [c for c in BINARY_DF.columns if c.startswith("target__yes")][0]

def t_correlate_output_is_dataframe():
    result = correlate(BINARY_DF, target=TARGET_COL)
    assert isinstance(result, pd.DataFrame)

def t_correlate_columns():
    result = correlate(BINARY_DF, target=TARGET_COL)
    assert list(result.columns) == ["feature", "bin", "correlation"]

def t_correlate_sorted_by_abs_descending():
    result = correlate(BINARY_DF, target=TARGET_COL)
    abs_corr = result["correlation"].abs().tolist()
    assert abs_corr == sorted(abs_corr, reverse=True)

def t_correlate_values_within_minus1_1():
    result = correlate(BINARY_DF, target=TARGET_COL)
    assert result["correlation"].between(-1, 1).all()

def t_correlate_feature_is_categorical():
    result = correlate(BINARY_DF, target=TARGET_COL)
    assert hasattr(result["feature"], "cat")

def t_correlate_raises_on_missing_target():
    try:
        correlate(BINARY_DF, target="nonexistent_col")
        raise AssertionError("Expected KeyError")
    except KeyError:
        pass

def t_correlate_warns_on_imbalance():
    df = pd.DataFrame({
        "feat__0":    [0, 1] * 50,
        "target__yes": [1] * 4 + [0] * 96,
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        correlate(df, target="target__yes")
    imbalance_warns = [w for w in caught if "Imbalance" in str(w.message)]
    assert len(imbalance_warns) >= 1

def t_is_binary_true():
    assert _is_binary(pd.Series([0, 1, 0, 1]))

def t_is_binary_false():
    assert not _is_binary(pd.Series([0, 1, 2]))

def t_check_all_numeric_raises_on_string():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    try:
        _check_all_numeric(df, fun_name="test")
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


# ---------------------------------------------------------------------------
# plot tests (matplotlib only - no plotly needed)
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

CORR_DF = correlate(BINARY_DF, target=TARGET_COL)

def t_plot_returns_matplotlib_figure():
    fig = plot_correlation_funnel(CORR_DF, interactive=False)
    assert isinstance(fig, matplotlib.figure.Figure)

def t_plot_has_one_axes():
    fig = plot_correlation_funnel(CORR_DF)
    assert len(fig.axes) == 1

def t_plot_x_limits():
    limits = (-0.5, 0.5)
    fig = plot_correlation_funnel(CORR_DF, limits=limits)
    ax = fig.axes[0]
    lo, hi = ax.get_xlim()
    assert abs(lo - limits[0]) < 0.01 and abs(hi - limits[1]) < 0.01

def t_plot_raises_on_missing_columns():
    bad = pd.DataFrame({"feature": ["a"], "wrong": [0.1]})
    try:
        plot_correlation_funnel(bad)
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "Missing required columns" in str(exc)

def t_check_column_names_passes():
    df = pd.DataFrame({"feature": [], "bin": [], "correlation": []})
    _check_column_names(df, fun_name="test")

def t_check_column_names_raises():
    df = pd.DataFrame({"feature": [], "bin": []})
    try:
        _check_column_names(df, fun_name="test")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Full integration test
# ---------------------------------------------------------------------------

def t_full_pipeline_marketing_synthetic():
    """End-to-end test using the synthetic marketing dataset."""
    from funnel_correlation_py.data import _synthetic_marketing
    df = _synthetic_marketing(n_rows=300).drop(columns=["ID"])
    binary  = binarize(df)
    target  = [c for c in binary.columns if c.startswith("TERM_DEPOSIT__yes")][0]
    corr    = correlate(binary, target=target)
    fig     = plot_correlation_funnel(corr, interactive=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert corr.shape[0] > 0


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

TESTS = [
    # binarize
    t_binarize_output_is_dataframe,
    t_binarize_all_integers,
    t_binarize_values_are_0_or_1,
    t_binarize_double_underscore_separator,
    t_binarize_raises_on_missing,
    t_binarize_raises_on_datetime,
    t_binarize_thresh_infreq_creates_other,
    t_binarize_one_hot_false_fewer_cols,
    t_fix_low_cardinality,
    t_fix_high_skew,
    t_lump_rare_categories,
    t_drop_zero_variance,
    # correlate
    t_correlate_output_is_dataframe,
    t_correlate_columns,
    t_correlate_sorted_by_abs_descending,
    t_correlate_values_within_minus1_1,
    t_correlate_feature_is_categorical,
    t_correlate_raises_on_missing_target,
    t_correlate_warns_on_imbalance,
    t_is_binary_true,
    t_is_binary_false,
    t_check_all_numeric_raises_on_string,
    # plot
    t_plot_returns_matplotlib_figure,
    t_plot_has_one_axes,
    t_plot_x_limits,
    t_plot_raises_on_missing_columns,
    t_check_column_names_passes,
    t_check_column_names_raises,
    # integration
    t_full_pipeline_marketing_synthetic,
]

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  funnel_correlation_py — validation suite")
    print(f"{'='*60}\n")

    for test_fn in TESTS:
        _run(test_fn.__name__, test_fn)

    passed = sum(1 for r in results if r[0] == PASS)
    failed = sum(1 for r in results if r[0] == FAIL)

    for status, name, msg in results:
        icon = "✓" if status == PASS else "✗"
        suffix = f"  → {msg}" if msg else ""
        print(f"  {icon} {name}{suffix}")

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(results)} tests")
    print(f"{'='*60}\n")

    sys.exit(0 if failed == 0 else 1)
