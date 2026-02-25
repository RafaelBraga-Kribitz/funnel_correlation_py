"""
Unit tests for funnel_correlation_py.plot
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

from funnel_correlation_py.binarize import binarize
from funnel_correlation_py.correlate import correlate
from funnel_correlation_py.plot import plot_correlation_funnel, _check_column_names


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def corr_df() -> pd.DataFrame:
    """Full pipeline output: binarize -> correlate -> correlation DataFrame."""
    rng = np.random.default_rng(seed=7)
    n = 300
    raw = pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=n),
            "balance": rng.uniform(-200, 8000, size=n),
            "job": rng.choice(["admin", "tech", "sales", "retired"], size=n),
            "target": rng.choice(["yes", "no"], size=n),
        }
    )
    binary_df = binarize(raw)
    target_col = [c for c in binary_df.columns if c.startswith("target__yes")][0]
    return correlate(binary_df, target=target_col)


# ---------------------------------------------------------------------------
# plot_correlation_funnel() – static
# ---------------------------------------------------------------------------

class TestPlotStatic:
    def test_returns_matplotlib_figure(self, corr_df):
        import matplotlib.figure
        fig = plot_correlation_funnel(corr_df, interactive=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_has_one_axes(self, corr_df):
        fig = plot_correlation_funnel(corr_df)
        assert len(fig.axes) == 1

    def test_x_axis_limits_respected(self, corr_df):
        limits = (-0.5, 0.5)
        fig = plot_correlation_funnel(corr_df, limits=limits)
        ax = fig.axes[0]
        assert ax.get_xlim() == pytest.approx(limits, abs=1e-3)

    def test_custom_title(self, corr_df):
        fig = plot_correlation_funnel(corr_df, title="My Custom Title")
        ax = fig.axes[0]
        assert ax.get_title() == "My Custom Title"


# ---------------------------------------------------------------------------
# plot_correlation_funnel() – interactive
# ---------------------------------------------------------------------------

class TestPlotInteractive:
    def test_returns_plotly_figure(self, corr_df):
        try:
            import plotly.graph_objects as go
        except ImportError:
            pytest.skip("plotly not installed")
        fig = plot_correlation_funnel(corr_df, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_plotly_figure_has_data(self, corr_df):
        try:
            import plotly.graph_objects as go
        except ImportError:
            pytest.skip("plotly not installed")
        fig = plot_correlation_funnel(corr_df, interactive=True)
        assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# plot_correlation_funnel() – error cases
# ---------------------------------------------------------------------------

class TestPlotErrors:
    def test_raises_on_missing_columns(self):
        bad_df = pd.DataFrame({"feature": ["a"], "wrong_col": [0.1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            plot_correlation_funnel(bad_df)

    def test_raises_on_empty_dataframe(self):
        empty = pd.DataFrame(columns=["feature", "bin", "correlation"])
        # Should not crash; it may produce an empty figure
        fig = plot_correlation_funnel(empty)
        assert fig is not None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

class TestCheckColumnNames:
    def test_passes_on_correct_columns(self):
        df = pd.DataFrame({"feature": [], "bin": [], "correlation": []})
        _check_column_names(df, fun_name="test")  # no raise

    def test_raises_on_bad_columns(self):
        df = pd.DataFrame({"feature": [], "bin": []})
        with pytest.raises(ValueError, match="Missing required columns"):
            _check_column_names(df, fun_name="test")
