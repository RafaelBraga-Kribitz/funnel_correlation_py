"""
binarize.py
-----------
Converts a tidy DataFrame with numeric and categorical features into a
fully binary (0/1) DataFrame suitable for correlation-funnel analysis.

Mirrors the behaviour of `binarize()` from the R correlationfunnel package.

Workflow
~~~~~~~~
1. Validate column types  (no datetime, no object-but-not-str, etc.)
2. Raise on missing values
3. Convert bool -> int
4. Convert low-cardinality numerics to categorical
5. Convert highly-skewed numerics to categorical
6. Bin remaining numerics with quantile-based cuts  (step_discretize equivalent)
7. Lump rare categorical levels into a single "OTHER" bucket (step_other equivalent)
8. One-hot encode all categorical / binned columns   (step_dummy equivalent)
9. Drop zero-variance columns                        (step_zv equivalent)
10. Rename binned columns from "feature__0" -> "feature__lo_hi" style labels
"""

from __future__ import annotations

import re
from typing import Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def binarize(
    data: pd.DataFrame,
    n_bins: int = 4,
    thresh_infreq: float = 0.01,
    name_infreq: str = "-OTHER",
    one_hot: bool = True,
) -> pd.DataFrame:
    """Convert a DataFrame with numeric and categorical features into binary format.

    Continuous features are binned into ``n_bins`` quantile-based ranges and
    then one-hot encoded.  Categorical features have rare levels lumped into a
    single ``name_infreq`` bucket before one-hot encoding.  The resulting
    DataFrame contains only 0/1 integer columns with descriptive names like
    ``age__18_35`` or ``job__admin``.

    Parameters
    ----------
    data:
        Input DataFrame.  Must not contain datetime columns or missing values.
    n_bins:
        Number of quantile bins for continuous (numeric) features.
    thresh_infreq:
        Minimum relative frequency for a categorical level to keep its own
        column.  Levels appearing less often are lumped into ``name_infreq``.
    name_infreq:
        Label assigned to rare categorical levels.  Defaults to ``"-OTHER"``.
    one_hot:
        When ``True`` (default) each level gets its own column.
        When ``False`` the first level is dropped (dummy encoding, avoids
        perfect multicollinearity).

    Returns
    -------
    pd.DataFrame
        Binary (0/1) integer DataFrame.  Column names use ``__`` as separator
        between feature name and bin / level label.

    Raises
    ------
    TypeError
        If any column contains an unsupported data type.
    ValueError
        If the DataFrame contains missing values (NaN).

    Examples
    --------
    >>> from funnel_correlation_py import binarize
    >>> from funnel_correlation_py.data import load_marketing_campaign
    >>> df = load_marketing_campaign().drop(columns=["ID"])
    >>> binary_df = binarize(df)
    """
    data = data.copy()

    # -- Validation ----------------------------------------------------------
    _check_data_types(data, fun_name="binarize")
    _check_missing(data, fun_name="binarize")

    # -- Pre-processing fixes ------------------------------------------------
    data = _logical_to_integer(data)
    data = _fix_low_cardinality_numeric(data, thresh=n_bins + 3)
    data = _fix_high_skew_numeric(data, unique_limit=2)

    # -- Separate numeric and categorical columns ----------------------------
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

    pieces: list[pd.DataFrame] = []

    # -- Bin numeric columns -------------------------------------------------
    if numeric_cols:
        binned, bin_label_map = _bin_numeric(data[numeric_cols], n_bins=n_bins)
        ohe_numeric = _one_hot_encode(binned, one_hot=one_hot)
        ohe_numeric = _rename_binned_columns(ohe_numeric, bin_label_map)
        pieces.append(ohe_numeric)

    # -- Encode categorical columns -----------------------------------------
    if cat_cols:
        lumped = _lump_rare_categories(
            data[cat_cols],
            thresh=thresh_infreq,
            other_name=name_infreq,
        )
        ohe_cat = _one_hot_encode(lumped, one_hot=one_hot)
        pieces.append(ohe_cat)

    if not pieces:
        raise ValueError("binarize(): No numeric or categorical columns found.")

    result = pd.concat(pieces, axis=1)

    # -- Drop zero-variance columns ------------------------------------------
    result = _drop_zero_variance(result)

    return result.astype(int)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

ALLOWED_DTYPES = {"int64", "int32", "int16", "int8",
                  "float64", "float32",
                  "bool",
                  "object", "category"}


def _check_data_types(data: pd.DataFrame, fun_name: str) -> None:
    """Raise TypeError if any column has an unsupported dtype."""
    bad_cols = []
    for col in data.columns:
        dtype_name = data[col].dtype.name
        is_numeric = pd.api.types.is_numeric_dtype(data[col])
        is_categorical = isinstance(data[col].dtype, pd.CategoricalDtype)
        is_string = pd.api.types.is_string_dtype(data[col]) and data[col].dtype == object
        is_bool = pd.api.types.is_bool_dtype(data[col])

        if not (is_numeric or is_categorical or is_string or is_bool):
            bad_cols.append(col)

    if bad_cols:
        raise TypeError(
            f"{fun_name}(): [Unacceptable Columns Detected] The following columns "
            f"contain non-numeric or non-categorical data types: "
            + ", ".join(bad_cols)
        )


def _check_missing(data: pd.DataFrame, fun_name: str) -> None:
    """Raise ValueError if any column has NaN values."""
    null_counts = data.isnull().sum()
    bad_cols = null_counts[null_counts > 0].index.tolist()
    if bad_cols:
        raise ValueError(
            f"{fun_name}(): [Missing Values Detected] The following columns contain "
            f"NAs: " + ", ".join(bad_cols)
        )


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

def _logical_to_integer(data: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean columns to integer (0/1)."""
    bool_cols = data.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        data = data.copy()
        data[bool_cols] = data[bool_cols].astype(int)
    return data


def _fix_low_cardinality_numeric(data: pd.DataFrame, thresh: int = 6) -> pd.DataFrame:
    """Convert numeric columns with very few unique values to categorical.

    Mirrors R's ``fix_low_cardinality_numeric()``:  columns with fewer than
    ``thresh`` unique values are treated as categories rather than continuous
    features, because binning them would produce meaningless or duplicate bins.
    """
    numeric_cols = data.select_dtypes(include=["number"]).columns
    if not len(numeric_cols):
        return data

    data = data.copy()
    for col in numeric_cols:
        n_unique = data[col].nunique()
        if 1 < n_unique < thresh:
            data[col] = data[col].astype(str).astype("category")
    return data


def _fix_high_skew_numeric(data: pd.DataFrame, unique_limit: int = 2) -> pd.DataFrame:
    """Convert highly skewed numeric columns (very few unique quantile values) to categorical.

    When most rows share the same value the quantile breaks collapse,
    making binning useless.  Columns with ``<= unique_limit`` unique quantile
    values are converted to categorical.
    """
    numeric_cols = data.select_dtypes(include=["number"]).columns
    if not len(numeric_cols):
        return data

    data = data.copy()
    quantile_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    for col in numeric_cols:
        quantile_values = data[col].quantile(quantile_levels)
        if quantile_values.nunique() <= unique_limit:
            data[col] = data[col].astype(str).astype("category")
    return data


# ---------------------------------------------------------------------------
# Binning (step_discretize equivalent)
# ---------------------------------------------------------------------------

def _bin_numeric(
    data: pd.DataFrame,
    n_bins: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Bin each numeric column into ``n_bins`` quantile-based intervals.

    Returns
    -------
    binned_df:
        DataFrame where each column holds string labels like ``"18.0_35.0"``.
    bin_label_map:
        Mapping from column name to ordered list of bin labels
        (used later to rename OHE columns).
    """
    binned = data.copy()
    bin_label_map: dict[str, list[str]] = {}

    for col in data.columns:
        series = data[col]
        try:
            cut_result, bins = pd.qcut(
                series,
                q=n_bins,
                retbins=True,
                duplicates="drop",
            )
        except ValueError:
            # Fallback: equal-width when quantile cuts fail
            cut_result, bins = pd.cut(
                series,
                bins=n_bins,
                retbins=True,
                duplicates="drop",
            )

        # Build human-readable labels: "lo_hi"
        labels = [
            f"{_fmt_bound(bins[i])}_{_fmt_bound(bins[i + 1])}"
            for i in range(len(bins) - 1)
        ]

        actual_cats = cut_result.cat.categories
        # Trim to however many actual categories were produced
        labels = labels[: len(actual_cats)]

        cut_result = cut_result.cat.rename_categories(labels)
        binned[col] = cut_result.astype(str)
        bin_label_map[col] = labels

    return binned, bin_label_map


def _fmt_bound(value: float) -> str:
    """Format a bin boundary as a compact string (no trailing zeros)."""
    if value == int(value):
        return str(int(value))
    return f"{value:.4g}"


# ---------------------------------------------------------------------------
# Rare-category lumping (step_other equivalent)
# ---------------------------------------------------------------------------

def _lump_rare_categories(
    data: pd.DataFrame,
    thresh: float,
    other_name: str,
) -> pd.DataFrame:
    """Replace rare categorical levels with ``other_name``.

    A level is 'rare' when its relative frequency is below ``thresh``.
    If ``thresh == 0`` every level is kept.

    Categorical dtype columns are cast to ``object`` first so that the
    ``other_name`` string can be injected without a "new category" error.
    """
    data = data.copy()

    # Cast any pd.Categorical columns to plain object so .where() can write
    # the other_name string without needing to add it to the category list.
    for col in data.columns:
        if isinstance(data[col].dtype, pd.CategoricalDtype):
            data[col] = data[col].astype(object)

    for col in data.columns:
        if thresh == 0:
            continue
        freq = data[col].value_counts(normalize=True)
        rare_levels = freq[freq < thresh].index
        if len(rare_levels):
            data[col] = data[col].where(~data[col].isin(rare_levels), other=other_name)

    return data


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------

def _one_hot_encode(data: pd.DataFrame, one_hot: bool) -> pd.DataFrame:
    """One-hot encode all columns in ``data``.

    Parameters
    ----------
    data:
        DataFrame with string / category columns (already binned or lumped).
    one_hot:
        When ``True`` keep all levels; when ``False`` drop the first
        (dummy / reference) level.

    Column naming convention: ``<feature>__<level>`` (double underscore).
    """
    encoded_pieces = []

    for col in data.columns:
        dummies = pd.get_dummies(data[col], prefix=col, prefix_sep="__")

        # Sanitize level names: spaces -> underscores, trim
        dummies.columns = [
            re.sub(r"\s+", "_", c).strip() for c in dummies.columns
        ]

        if not one_hot:
            dummies = dummies.iloc[:, 1:]  # drop first level (reference)

        encoded_pieces.append(dummies)

    return pd.concat(encoded_pieces, axis=1)


# ---------------------------------------------------------------------------
# Post-encoding: rename binned columns to "feature__lo_hi"
# ---------------------------------------------------------------------------

def _rename_binned_columns(
    data: pd.DataFrame,
    bin_label_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Rename auto-generated OHE columns for numeric bins.

    ``pd.get_dummies`` produces names like ``age__18.0_35.0``.  This function
    is a no-op if those names are already correct (which they are when
    ``_bin_numeric`` returns proper string labels), but it ensures consistency
    with the R package's ``<feature>__<lo>_<hi>`` convention.
    """
    # No renaming needed: labels were embedded in the binned DataFrame already.
    return data


# ---------------------------------------------------------------------------
# Zero-variance column removal (step_zv equivalent)
# ---------------------------------------------------------------------------

def _drop_zero_variance(data: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that have the same value in every row."""
    varying = data.columns[data.nunique() > 1]
    return data[varying]
