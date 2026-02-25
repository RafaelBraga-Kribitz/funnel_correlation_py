"""
data.py
-------
Bundled example datasets that mirror those shipped with the R
correlationfunnel package.

Available loaders
-----------------
* ``load_marketing_campaign()``  - Bank telemarketing dataset (4521 rows)
* ``load_customer_churn()``      - Telco customer churn dataset (7043 rows)

Both datasets are sourced on-the-fly from public CSV mirrors if not already
cached in the package's ``data/raw/`` directory.  A tiny synthetic fallback
is used when the network is unavailable, so the rest of the package still
works in offline environments.
"""

from __future__ import annotations

import io
import os
import warnings
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PKG_ROOT = Path(__file__).parent
_DATA_RAW = _PKG_ROOT.parent.parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_marketing_campaign() -> pd.DataFrame:
    """Load the Bank Marketing dataset.

    The dataset describes direct phone-call marketing campaigns by a
    Portuguese bank.  The binary target variable is ``TERM_DEPOSIT``
    (did the client subscribe to a term deposit? yes/no).

    Source: Moro, S., Cortez, P., & Rita, P. (2014). Decision Support Systems.

    Returns
    -------
    pd.DataFrame
        4521 rows x 17 columns.  Column names are UPPER_SNAKE_CASE to match
        the R package convention.

    Examples
    --------
    >>> from funnel_correlation_py.data import load_marketing_campaign
    >>> df = load_marketing_campaign()
    >>> df.shape
    (4521, 17)
    """
    cache_path = _DATA_RAW / "marketing_campaign.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return _read_marketing(cache_path)

    # Try downloading from UCI ML Repository (semicolon-delimited)
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00222/bank.zip"
    )
    try:
        import urllib.request, zipfile
        zip_path = cache_path.parent / "bank.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("bank.csv") as f:
                raw = f.read().decode("utf-8")
        zip_path.unlink(missing_ok=True)
        cache_path.write_text(raw)
        return _read_marketing(io.StringIO(raw))
    except Exception as exc:
        warnings.warn(
            f"Could not download marketing_campaign dataset ({exc}). "
            "Returning a synthetic mini-dataset for demonstration purposes.",
            UserWarning,
            stacklevel=2,
        )
        return _synthetic_marketing()


def load_customer_churn() -> pd.DataFrame:
    """Load the Telco Customer Churn dataset.

    IBM sample dataset tracking customer service usage and churn status.
    Binary target variable: ``CHURN`` (did the customer leave? Yes/No).

    Returns
    -------
    pd.DataFrame
        Up to 7043 rows x 21 columns.

    Examples
    --------
    >>> from funnel_correlation_py.data import load_customer_churn
    >>> df = load_customer_churn()
    >>> "CHURN" in df.columns
    True
    """
    cache_path = _DATA_RAW / "customer_churn.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return _read_churn(cache_path)

    url = (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv"
    )
    try:
        df = pd.read_csv(url)
        df.columns = [c.upper().replace(" ", "_") for c in df.columns]
        df.to_csv(cache_path, index=False)
        return df
    except Exception as exc:
        warnings.warn(
            f"Could not download customer_churn dataset ({exc}). "
            "Returning a synthetic mini-dataset for demonstration purposes.",
            UserWarning,
            stacklevel=2,
        )
        return _synthetic_churn()


# ---------------------------------------------------------------------------
# Internal readers
# ---------------------------------------------------------------------------

def _read_marketing(source) -> pd.DataFrame:
    """Read raw UCI bank.csv (semicolon-delimited) and normalise column names."""
    df = pd.read_csv(source, sep=";")
    rename_map = {
        "age": "AGE",
        "job": "JOB",
        "marital": "MARITAL",
        "education": "EDUCATION",
        "default": "DEFAULT",
        "balance": "BALANCE",
        "housing": "HOUSING",
        "loan": "LOAN",
        "contact": "CONTACT",
        "day": "DAY",
        "month": "MONTH",
        "duration": "DURATION",
        "campaign": "CAMPAIGN",
        "pdays": "PDAYS",
        "previous": "PREVIOUS",
        "poutcome": "POUTCOME",
        "y": "TERM_DEPOSIT",
    }
    df = df.rename(columns=rename_map)
    df.insert(0, "ID", range(1, len(df) + 1))
    return df


def _read_churn(source) -> pd.DataFrame:
    """Read cached churn CSV."""
    return pd.read_csv(source)


# ---------------------------------------------------------------------------
# Synthetic fallbacks (used offline / in CI without network access)
# ---------------------------------------------------------------------------

def _synthetic_marketing(n_rows: int = 200) -> pd.DataFrame:
    """Return a tiny synthetic dataset that mimics the marketing_campaign schema."""
    import numpy as np
    rng = np.random.default_rng(seed=42)

    df = pd.DataFrame(
        {
            "ID": range(1, n_rows + 1),
            "AGE": rng.integers(18, 75, size=n_rows),
            "JOB": rng.choice(
                ["admin.", "blue-collar", "entrepreneur", "housemaid",
                 "management", "retired", "self-employed", "services",
                 "student", "technician", "unemployed", "unknown"],
                size=n_rows,
            ),
            "MARITAL": rng.choice(["divorced", "married", "single"], size=n_rows),
            "EDUCATION": rng.choice(
                ["primary", "secondary", "tertiary", "unknown"], size=n_rows
            ),
            "DEFAULT": rng.choice(["no", "yes"], size=n_rows, p=[0.98, 0.02]),
            "BALANCE": rng.integers(-500, 10000, size=n_rows),
            "HOUSING": rng.choice(["no", "yes"], size=n_rows),
            "LOAN": rng.choice(["no", "yes"], size=n_rows, p=[0.84, 0.16]),
            "CONTACT": rng.choice(
                ["cellular", "telephone", "unknown"], size=n_rows
            ),
            "DAY": rng.integers(1, 31, size=n_rows),
            "MONTH": rng.choice(
                ["jan", "feb", "mar", "apr", "may", "jun",
                 "jul", "aug", "sep", "oct", "nov", "dec"],
                size=n_rows,
            ),
            "DURATION": rng.integers(0, 800, size=n_rows),
            "CAMPAIGN": rng.integers(1, 10, size=n_rows),
            "PDAYS": rng.integers(-1, 400, size=n_rows),
            "PREVIOUS": rng.integers(0, 5, size=n_rows),
            "POUTCOME": rng.choice(
                ["failure", "other", "success", "unknown"], size=n_rows
            ),
            "TERM_DEPOSIT": rng.choice(["no", "yes"], size=n_rows, p=[0.88, 0.12]),
        }
    )
    return df


def _synthetic_churn(n_rows: int = 200) -> pd.DataFrame:
    """Return a tiny synthetic dataset that mimics the customer_churn schema."""
    import numpy as np
    rng = np.random.default_rng(seed=42)

    df = pd.DataFrame(
        {
            "CUSTOMERID": [f"C{i:04d}" for i in range(1, n_rows + 1)],
            "GENDER": rng.choice(["Female", "Male"], size=n_rows),
            "SENIORCITIZEN": rng.integers(0, 2, size=n_rows),
            "PARTNER": rng.choice(["No", "Yes"], size=n_rows),
            "DEPENDENTS": rng.choice(["No", "Yes"], size=n_rows),
            "TENURE": rng.integers(0, 72, size=n_rows),
            "PHONESERVICE": rng.choice(["No", "Yes"], size=n_rows),
            "MULTIPLELINES": rng.choice(
                ["No", "No phone service", "Yes"], size=n_rows
            ),
            "INTERNETSERVICE": rng.choice(
                ["DSL", "Fiber optic", "No"], size=n_rows
            ),
            "ONLINESECURITY": rng.choice(
                ["No", "No internet service", "Yes"], size=n_rows
            ),
            "TECHSUPPORT": rng.choice(
                ["No", "No internet service", "Yes"], size=n_rows
            ),
            "CONTRACT": rng.choice(
                ["Month-to-month", "One year", "Two year"], size=n_rows
            ),
            "PAPERLESSBILLING": rng.choice(["No", "Yes"], size=n_rows),
            "PAYMENTMETHOD": rng.choice(
                ["Bank transfer (automatic)", "Credit card (automatic)",
                 "Electronic check", "Mailed check"],
                size=n_rows,
            ),
            "MONTHLYCHARGES": rng.uniform(18, 120, size=n_rows).round(2),
            "TOTALCHARGES": rng.uniform(0, 8000, size=n_rows).round(2),
            "CHURN": rng.choice(["No", "Yes"], size=n_rows, p=[0.73, 0.27]),
        }
    )
    return df
