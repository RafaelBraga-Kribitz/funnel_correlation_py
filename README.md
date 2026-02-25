# funnel-correlation-py

Python port of the R [`correlationfunnel`](https://github.com/business-science/correlationfunnel) package by Business Science.

Speeds up exploratory data analysis (EDA) by surfacing the strongest
feature-target relationships through a three-step binary correlation workflow.

---

## Three-step workflow

```
raw DataFrame
  -> binarize(df)                    # numeric + categorical -> binary (0/1)
  -> correlate(binary_df, target)    # Pearson r of each feature vs target
  -> plot_correlation_funnel(corr)   # tornado chart, strongest predictors on top
```

---

## Quick start

```python
from funnel_correlation_py import binarize, correlate, plot_correlation_funnel
from funnel_correlation_py.data import load_marketing_campaign

# 1. Load
df = load_marketing_campaign().drop(columns=["ID"])

# 2. Binarize
binary_df = binarize(df, n_bins=4, thresh_infreq=0.01)

# 3. Correlate (pick the binary target column)
corr_df = correlate(binary_df, target="TERM_DEPOSIT__yes")

# 4. Plot (static)
fig = plot_correlation_funnel(corr_df, limits=(-0.4, 0.4))
fig.tight_layout()

# 4b. Plot (interactive, requires plotly)
fig_interactive = plot_correlation_funnel(corr_df, interactive=True)
fig_interactive.show()
```

---

## Installation

```bash
# From the project root
pip install -e .

# With interactive (plotly) support
pip install -e ".[interactive]"
```

---

## API reference

### `binarize(data, n_bins=4, thresh_infreq=0.01, name_infreq="-OTHER", one_hot=True)`

Converts a tidy DataFrame to binary (0/1) format.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | required | Input table (no datetime, no NaN) |
| `n_bins` | `int` | `4` | Quantile bins for numeric features |
| `thresh_infreq` | `float` | `0.01` | Min frequency to keep a categorical level |
| `name_infreq` | `str` | `"-OTHER"` | Label for lumped rare levels |
| `one_hot` | `bool` | `True` | `True` = all levels; `False` = drop first (dummy) |

**What it does per column type:**

| Column dtype | Transformation |
|--------------|---------------|
| `float` / `int` (high cardinality) | Quantile binning -> one-hot encode |
| `float` / `int` (low cardinality) | Treated as categorical -> one-hot encode |
| `object` / `category` | Rare levels lumped -> one-hot encode |
| `bool` | Cast to `int`, then binned |
| Constant column | Dropped (zero variance) |

Column names use `__` as separator: `age__18_35`, `job__admin`.

---

### `correlate(data, target, method="pearson")`

Computes Pearson r between every binary feature and `target`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | required | Binary DataFrame from `binarize()` |
| `target` | `str` | required | Column name of the response variable |
| `method` | `str` | `"pearson"` | Passed to `pd.DataFrame.corrwith` |

Returns a tidy DataFrame with columns `feature`, `bin`, `correlation`,
sorted descending by `|correlation|`.  `feature` is a `pd.Categorical`
ordered for correct y-axis positioning in plots.

Emits a `UserWarning` when the positive-class proportion is below 5%.

---

### `plot_correlation_funnel(data, interactive=False, limits=(-1,1), alpha=1.0, figsize=(10,8), title="Correlation Funnel")`

Tornado-style plot of correlation strength.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | required | Output of `correlate()` |
| `interactive` | `bool` | `False` | `True` returns plotly figure |
| `limits` | `tuple` | `(-1, 1)` | X-axis range |
| `alpha` | `float` | `1.0` | Point transparency |
| `figsize` | `tuple` | `(10, 8)` | Static figure size (inches) |
| `title` | `str` | `"Correlation Funnel"` | Plot title |

---

## Folder structure

```
funnel_correlation_py/
  README.md
  pyproject.toml
  notebooks/
    01_example_usage.ipynb      <- end-to-end demo
  data/
    raw/                        <- downloaded datasets cached here
  src/
    funnel_correlation_py/
      __init__.py
      binarize.py               <- core transformation
      correlate.py              <- Pearson correlation
      plot.py                   <- static + interactive plots
      data.py                   <- dataset loaders + synthetic fallbacks
      features/
      models/
      evaluation/
      visualization/
      utils/
  tests/
    test_binarize.py
    test_correlate.py
    test_plot.py
    run_validation.py           <- standalone runner (no pytest required)
```

---

## Running validation

```bash
# No pytest required
PYTHONPATH=src python tests/run_validation.py

# With pytest (install first)
PYTHONPATH=src pytest tests/ -v
```

---

## Reproducibility

- Seeds used: `numpy.random.default_rng(seed=42)` in all synthetic data.
- Raw datasets cached in `data/raw/` on first download.
- No business logic in notebooks; all reusable code lives in `src/`.

---

## Credits

Original R package: [Business Science — correlationfunnel](https://github.com/business-science/correlationfunnel)
Paper: Duan et al., 2014 — *Selecting the right correlation measure for binary data*. ACM TKDD.
