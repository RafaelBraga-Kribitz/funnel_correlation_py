"""
plot.py
-------
Visualise the output of :func:`~funnel_correlation_py.correlate` as a
"correlation funnel" (tornado plot).

Mirrors ``plot_correlation_funnel()`` from the R correlationfunnel package.

Two output modes
~~~~~~~~~~~~~~~~
* ``interactive=False``  (default) -> ``matplotlib`` figure
* ``interactive=True``             -> ``plotly`` figure (hover labels)
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_correlation_funnel(
    data: pd.DataFrame,
    interactive: bool = False,
    limits: Tuple[float, float] = (-1, 1),
    alpha: float = 1.0,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Correlation Funnel",
) -> object:
    """Plot the correlation funnel.

    Parameters
    ----------
    data:
        Tidy DataFrame with columns ``feature``, ``bin``, ``correlation``
        as returned by :func:`~funnel_correlation_py.correlate`.
    interactive:
        When ``True`` returns a ``plotly.graph_objects.Figure``.
        When ``False`` (default) returns a ``matplotlib.figure.Figure``.
    limits:
        X-axis range.  Defaults to ``(-1, 1)``.
    alpha:
        Point transparency (0 = fully transparent, 1 = fully opaque).
    figsize:
        Width and height of the static matplotlib figure in inches.
    title:
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure | plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If ``data`` does not contain the required columns.

    Examples
    --------
    >>> from funnel_correlation_py import binarize, correlate, plot_correlation_funnel
    >>> from funnel_correlation_py.data import load_marketing_campaign
    >>> df = load_marketing_campaign().drop(columns=["ID"])
    >>> binary_df = binarize(df)
    >>> corr_df   = correlate(binary_df, target="TERM_DEPOSIT__yes")
    >>> fig = plot_correlation_funnel(corr_df)
    >>> fig.tight_layout()
    """
    _check_column_names(data, fun_name="plot_correlation_funnel")

    if interactive:
        return _plot_interactive(data, limits=limits, alpha=alpha, title=title)
    return _plot_static(data, limits=limits, alpha=alpha, figsize=figsize, title=title)


# ---------------------------------------------------------------------------
# Static plot (matplotlib)
# ---------------------------------------------------------------------------

def _plot_static(
    data: pd.DataFrame,
    limits: Tuple[float, float],
    alpha: float,
    figsize: Tuple[int, int],
    title: str,
) -> "plt.Figure":
    """Build a static matplotlib correlation funnel."""
    fig, ax = plt.subplots(figsize=figsize)

    # y-axis: feature as categorical.  If stored as pd.Categorical use the
    # .cat.codes for numeric y positioning so matplotlib respects the order.
    if hasattr(data["feature"], "cat"):
        y_values = data["feature"].cat.codes
        y_labels = data["feature"].cat.categories.tolist()
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=9)
    else:
        # Fallback: derive order from |corr| if feature is plain string
        feature_order = (
            data
            .groupby("feature")["correlation"]
            .apply(lambda s: s.abs().max())
            .sort_values(ascending=True)
            .index
            .tolist()
        )
        feature_to_int = {f: i for i, f in enumerate(feature_order)}
        y_values = data["feature"].map(feature_to_int)
        ax.set_yticks(list(feature_to_int.values()))
        ax.set_yticklabels(feature_to_int.keys(), fontsize=9)

    # Vertical zero line
    ax.axvline(x=0, linestyle="--", color="red", linewidth=1, zorder=1)

    # Scatter points
    ax.scatter(
        data["correlation"],
        y_values,
        color="#2c3e50",
        alpha=alpha,
        zorder=2,
        s=40,
    )

    # Bin labels next to each point (equivalent to geom_text_repel)
    _annotate_bins(ax, data, y_values)

    # Formatting
    ax.set_xlim(limits)
    ax.set_xlabel("Correlation", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.tight_layout()
    return fig


def _annotate_bins(
    ax: "plt.Axes",
    data: pd.DataFrame,
    y_values: pd.Series,
) -> None:
    """Add small bin-label annotations beside each scatter point.

    Points on the positive side get labels to the right; negative side to the
    left.  This is a simple version of geom_text_repel that avoids the heavy
    adjustText dependency for typical funnel sizes.
    """
    for (_, row), y in zip(data.iterrows(), y_values):
        corr = row["correlation"]
        label = str(row["bin"])
        ha = "left" if corr >= 0 else "right"
        offset = 0.02 if corr >= 0 else -0.02
        ax.text(
            corr + offset,
            y,
            label,
            fontsize=7,
            color="#2c3e50",
            ha=ha,
            va="center",
        )


# ---------------------------------------------------------------------------
# Interactive plot (plotly)
# ---------------------------------------------------------------------------

def _plot_interactive(
    data: pd.DataFrame,
    limits: Tuple[float, float],
    alpha: float,
    title: str,
) -> object:
    """Build an interactive plotly correlation funnel."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for interactive plots.  "
            "Install with: pip install plotly"
        ) from exc

    # Build hover text
    hover_text = (
        data["feature"].astype(str)
        + " | bin: " + data["bin"].astype(str)
        + " | r = " + data["correlation"].round(3).astype(str)
    )

    # Determine y-axis order (categorical codes ascending = top of chart = highest |corr|)
    if hasattr(data["feature"], "cat"):
        y_axis_order = data["feature"].cat.categories.tolist()
    else:
        y_axis_order = (
            data
            .groupby("feature")["correlation"]
            .apply(lambda s: s.abs().max())
            .sort_values(ascending=True)
            .index
            .tolist()
        )

    fig = go.Figure()

    # Zero line
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1)

    # Points
    fig.add_trace(
        go.Scatter(
            x=data["correlation"],
            y=data["feature"].astype(str),
            mode="markers+text",
            marker=dict(color="#2c3e50", opacity=alpha, size=8),
            text=data["bin"].astype(str),
            textposition="middle right",
            textfont=dict(size=9, color="#2c3e50"),
            hovertext=hover_text,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(range=list(limits), title="Correlation"),
        yaxis=dict(
            categoryorder="array",
            categoryarray=y_axis_order,
            title="Feature",
        ),
        template="simple_white",
        height=max(400, len(y_axis_order) * 20 + 100),
        showlegend=False,
    )

    return fig


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"feature", "bin", "correlation"}


def _check_column_names(data: pd.DataFrame, fun_name: str) -> None:
    """Raise ValueError if the required columns are not present."""
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            f"{fun_name}(): [Unacceptable Data] Missing required columns: "
            + ", ".join(sorted(missing))
            + ".  Acceptable data is generated by correlate()."
        )
