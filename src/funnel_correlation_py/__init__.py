"""
funnel_correlation_py
=====================
Python port of the R `correlationfunnel` package by Business Science.

Three-step EDA workflow
-----------------------
1. binarize(df)                     -> binary DataFrame
2. correlate(binary_df, target)     -> correlation DataFrame
3. plot_correlation_funnel(corr_df) -> static or interactive chart

Example
-------
>>> from funnel_correlation_py import binarize, correlate, plot_correlation_funnel
>>> from funnel_correlation_py.data import load_marketing_campaign
>>>
>>> df = load_marketing_campaign().drop(columns=["ID"])
>>> binary_df = binarize(df)
>>> corr_df   = correlate(binary_df, target="TERM_DEPOSIT__yes")
>>> plot_correlation_funnel(corr_df)
"""

from funnel_correlation_py.binarize import binarize
from funnel_correlation_py.correlate import correlate
from funnel_correlation_py.plot import plot_correlation_funnel

__all__ = ["binarize", "correlate", "plot_correlation_funnel"]
__version__ = "0.1.0"
