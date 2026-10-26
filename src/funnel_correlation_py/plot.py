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

Styling
~~~~~~~
Both modes are drawn in the Braga design system through ``bk-viz``
(``bk_theme``, light mode): page surface instead of white, BKribitz type,
hairline structure, points in the IBCS actual role, labels in ink. The theme
is scoped to the figure being built: the caller's global matplotlib rcParams
and plotly default template are left untouched.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

import bk_theme as bk

#: Design-system mode the figures are drawn in.
_THEME = "light"


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
    # Install the theme only for the duration of figure construction, so the
    # caller's rcParams survive. Everything that matplotlib resolves later (at
    # draw time or when ticks are regenerated) is set explicitly below.
    dpi = mpl.rcParams["figure.dpi"]  # the caller's resolution, not the theme's
    with mpl.rc_context():
        bk.apply(_THEME, grid="x")
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.set_facecolor(bk.c("surface"))
        ax.set_facecolor(bk.c("surface"))

        label_fp = bk.font("mono", "regular", "mono")

        # y-axis: feature as categorical.  If stored as pd.Categorical use the
        # .cat.codes for numeric y positioning so matplotlib respects the order.
        if hasattr(data["feature"], "cat"):
            y_values = data["feature"].cat.codes
            y_labels = data["feature"].cat.categories.tolist()
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels, fontproperties=label_fp)
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
            ax.set_yticklabels(feature_to_int.keys(), fontproperties=label_fp)

        # Vertical zero line: a reference, so a dashed structure hairline
        ax.axvline(x=0, linestyle=(0, (3, 3)), color=bk.c("mid"),
                   linewidth=1, zorder=1)

        # Scatter points: one series, the IBCS actual role
        ax.scatter(
            data["correlation"],
            y_values,
            color=bk.c("sc-ac"),
            alpha=alpha,
            linewidths=0,
            zorder=2,
            s=40,
        )

        # Bin labels next to each point (equivalent to geom_text_repel)
        _annotate_bins(ax, data, y_values)

        # Formatting
        ax.set_xlim(limits)
        ax.set_xlabel("Correlation", fontproperties=label_fp, color=bk.c("ink-2"))
        ax.set_ylabel("Feature", fontproperties=label_fp, color=bk.c("ink-2"))
        # The centre title slot keeps ax.get_title() working; the system sets
        # titles flush left, so that artist is moved to the left edge.
        ax.set_title(title, fontproperties=bk.font("display", "bold", "project"),
                     color=bk.c("ink"), loc="center")
        ax.title.set_x(0.0)
        ax.title.set_horizontalalignment("left")

        bk.strip_chrome(ax, keep=("left", "bottom"))
        ax.tick_params(
            length=0,
            labelcolor=bk.c("ink-2"),
            labelsize=bk.SIZE["mono"],
            labelfontfamily=label_fp.get_family(),
        )
        ax.grid(axis="x", color=bk.c("rule-c"), linewidth=1, linestyle="-")
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)

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
    fp = bk.font("mono", "regular", "mono-sm")  # 12px, the system's floor
    for (_, row), y in zip(data.iterrows(), y_values):
        corr = row["correlation"]
        label = str(row["bin"])
        ha = "left" if corr >= 0 else "right"
        offset = 0.02 if corr >= 0 else -0.02
        ax.text(
            corr + offset,
            y,
            label,
            fontproperties=fp,
            color=bk.c("ink-2"),
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

    # Design-system template, passed to this figure only: the caller's plotly
    # default template is not changed.
    # apply() registers the fonts and sets the mode the template is read in;
    # rc_context() keeps it from leaving matplotlib rcParams changed.
    with mpl.rc_context():
        bk.apply(_THEME, grid="x")
        template = bk.plotly_template(register_as="bk", set_default=False)

    fig = go.Figure()

    # Zero line: a reference, so a dashed structure hairline
    fig.add_vline(x=0, line_dash="dash", line_color=bk.c("mid"), line_width=1)

    # Points
    fig.add_trace(
        go.Scatter(
            x=data["correlation"],
            y=data["feature"].astype(str),
            mode="markers+text",
            marker=dict(color=bk.c("sc-ac"), opacity=alpha, size=8),
            text=data["bin"].astype(str),
            textposition="middle right",
            textfont=dict(size=12, color=bk.c("ink-2")),
            hovertext=hover_text,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(range=list(limits), title="Correlation", showgrid=True),
        yaxis=dict(
            showgrid=False,
            categoryorder="array",
            categoryarray=y_axis_order,
            title="Feature",
        ),
        template=template,
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
