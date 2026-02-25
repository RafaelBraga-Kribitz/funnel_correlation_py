"""
correlate.py
------------
Computes the Pearson correlation between every binary feature and a chosen
target column in a binarized DataFrame.

Mirrors the behaviour of ``correlate()`` from the R correlationfunnel package.

The output is a tidy DataFrame with columns:

    feature     - original feature name (left of ``__``)
    bin         - bin / level label     (right of ``__``)
    correlation - Pearson r value       (range -1 to 1)

sorted in descending order of |correlation| and with ``feature`` stored as a
Categorical type ordered by correlation strength (required for correct
``plot_correlation_funnel`` y-axis ordering).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def correlate(
    data: pd.DataFrame,
    target: str,
    method: str = "pearson",
) -> pd.DataFrame:
    """Correlate all binary features against a target column.

    Parameters
    ----------
    data:
        Binary (0/1) DataFrame produced by :func:`~funnel_correlation_py.binarize`.
    target:
        Name of the column to use as the response variable (Y).
        Must be a column present in ``data``.
    method:
        Correlation method passed to ``pd.DataFrame.corrwith``.
        Defaults to ``"pearson"``.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns ``feature``, ``bin``, ``correlation``,
        sorted by descending absolute correlation.
        ``feature`` is a ``pd.Categorical`` ordered from highest to lowest
        |correlation| (enables correct funnel-plot y-axis ordering).

    Raises
    ------
    KeyError
        If ``target`` is not a column in ``data``.
    TypeError
        If any column in ``data`` is not numeric.
    Warning
        If the target column is binary and severely imbalanced
        (positive-class proportion < 5 %).

    Examples
    --------
    >>> from funnel_correlation_py import binarize, correlate
    >>> from funnel_correlation_py.data import load_marketing_campaign
    >>> df = load_marketing_campaign().drop(columns=["ID"])
    >>> binary_df = binarize(df)
    >>> corr_df = correlate(binary_df, target="TERM_DEPOSIT__yes")
    """
    # -- Validation ----------------------------------------------------------
    if target not in data.columns:
        raise KeyError(
            f"correlate(): target column '{target}' not found in DataFrame. "
            f"Available columns: {list(data.columns)}"
        )

    _check_all_numeric(data, fun_name="correlate")

    # -- Check class imbalance on binary targets ----------------------------
    y = data[target]
    if _is_binary(y):
        _warn_imbalance(y, col_name=target, thresh=0.05, fun_name="correlate")

    # -- Compute correlations ------------------------------------------------
    # corrwith computes correlation of each column with the target series.
    corr_series = data.corrwith(y, method=method)

    # Remove the self-correlation of the target itself (r = 1.0)
    corr_series = corr_series.drop(labels=[target], errors="ignore")
    corr_series = corr_series.dropna()

    # -- Build tidy output ---------------------------------------------------
    corr_df = (
        corr_series
        .reset_index()
        .rename(columns={"index": "feature_bin", 0: "correlation"})
    )

    # The column produced by reset_index may be named differently in pandas versions
    if "feature_bin" not in corr_df.columns:
        corr_df = corr_df.rename(columns={corr_df.columns[0]: "feature_bin"})
    if "correlation" not in corr_df.columns:
        corr_df = corr_df.rename(columns={corr_df.columns[1]: "correlation"})

    # Split "feature__bin" into two columns
    split = corr_df["feature_bin"].str.split("__", n=1, expand=True)
    corr_df["feature"] = split[0]
    corr_df["bin"] = split[1] if split.shape[1] > 1 else corr_df["feature_bin"]

    corr_df = corr_df.drop(columns=["feature_bin"])
    corr_df = corr_df[["feature", "bin", "correlation"]]

    # -- Sort by absolute correlation ----------------------------------------
    corr_df["abs_corr"] = corr_df["correlation"].abs()
    corr_df = corr_df.sort_values("abs_corr", ascending=False).drop(columns=["abs_corr"])
    corr_df = corr_df.reset_index(drop=True)

    # -- Ordered categorical for feature (for y-axis ordering in plot) -------
    # R uses fct_rev(as_factor(feature)) so the highest correlation lands at top.
    # We replicate this: determine unique features ordered by their max |corr|,
    # then reverse so that pandas/matplotlib plots them top-down.
    feature_order = (
        corr_df
        .groupby("feature")["correlation"]
        .apply(lambda s: s.abs().max())
        .sort_values(ascending=True)   # ascending so top of y-axis = highest corr
        .index
        .tolist()
    )

    corr_df["feature"] = pd.Categorical(
        corr_df["feature"],
        categories=feature_order,
        ordered=True,
    )

    return corr_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_all_numeric(data: pd.DataFrame, fun_name: str) -> None:
    """Raise TypeError if any column is not numeric."""
    non_numeric = [
        col for col in data.columns
        if not pd.api.types.is_numeric_dtype(data[col])
    ]
    if non_numeric:
        raise TypeError(
            f"{fun_name}(): [Unacceptable Columns Detected] The following columns "
            f"contain non-numeric data types: " + ", ".join(non_numeric)
        )


def _is_binary(series: pd.Series) -> bool:
    """Return True if series contains only 0 and 1 values."""
    unique_vals = set(series.dropna().unique())
    return unique_vals.issubset({0, 1})


def _warn_imbalance(
    series: pd.Series,
    col_name: str,
    thresh: float,
    fun_name: str,
) -> None:
    """Emit a UserWarning when the positive class proportion is below ``thresh``."""
    prop_positive = series.sum() / len(series)
    if prop_positive < thresh:
        pct = f"{thresh:.0%}"
        warnings.warn(
            f"{fun_name}(): [Data Imbalance Detected] Consider resampling to "
            f"balance the classes more than {pct}. "
            f"Column with imbalance: {col_name}",
            UserWarning,
            stacklevel=2,
        )
