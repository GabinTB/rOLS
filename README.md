# rOLS

Vectorized rolling and expanding regression for multi-target, multi-factor time series.

Built for performance at scale: hundreds of regressors against hundreds of targets, over thousands of time steps, without loops.

Adapted for applications where dynamic relationships matter most: estimating rolling betas in finance to isolate idiosyncratic sensitivity to narrative factors; tracking time-varying price elasticities in economics to capture structural shifts; attributing regional temperature anomalies in climate science by orthogonalizing forcing factors; and adaptively filtering signals in real time. Designed for speed and scalability, it enables precise, loop-free analysis across domains where traditional methods fall short.

| Metric | Value |
|---|---|
| PyPI Version | [![PyPI version](https://img.shields.io/pypi/v/rols)](https://pypi.org/project/rols/) |
| Python Versions | [![Python versions](https://img.shields.io/pypi/pyversions/rols)](https://pypi.org/project/rols/) |
| License | [![License](https://img.shields.io/pypi/l/rols)](https://github.com/GabinTB/rOLS/blob/main/LICENSE) |
| Downloads | [![Downloads](https://img.shields.io/pypi/dm/rols)](https://pypi.org/project/rols/) |
| GitHub Stars | [![Stars](https://img.shields.io/github/stars/GabinTB/rOLS?style=social)](https://github.com/GabinTB/rOLS) |

---

## Overview

rOLS estimates time-varying sensitivities (betas) between a set of factors and a set of targets using a rolling or expanding window. At each time step, it fits a fresh regression on the most recent observations, giving you a full time series of betas, signals, and R² for each (factor, target) pair.

It supports:
- **OLS and Ridge** regression (controlled by a single `lambda_` parameter)
- **Multiple controls** — partialled out via Frisch-Waugh-Lovell, keeping the per-factor math univariate regardless of how many controls you add
- **Rolling Gram-Schmidt orthogonalization** — factors and/or controls can be orthogonalized within their group before estimation, so each beta reflects incremental explanatory power
- **HAC standard errors** — Newey-West robust SEs, computed on demand
- **Expanding windows** as an alternative to fixed rolling windows
- **Lagged signals** to avoid look-ahead bias

---

## Installation

```bash
pip install rols
```

Requires Python 3.10+ and numpy / pandas.

---

## Quick start

```python
import pandas as pd
import pandas_datareader as pdr
import pandas_datareader.data as web
from rols import RollingOLS

# Loading some factors
factor_df = web.DataReader(['CPIAUCSL', 'CPILFESL'], 'fred', start=start, ).pct_change().dropna()
factors = factor_df.columns.tolist()

# Loading some targets
asset_df = web.DataReader('12_Industry_Portfolios', 'famafrench', start=start, )[1]
asset_df.index = asset_df.index.to_timestamp()
assets = asset_df.columns.tolist()

# Loading some controls
control_df = pdr.get_data_famafrench("F-F_Research_Data_Factors", start=start, )[0].div(100.0).drop(columns=["RF"])
control_df.index = control_df.index.to_timestamp()
controls = control_df.columns.tolist()

# Merge data into one dataframe aligned by date
df = pd.merge(factor_df, asset_df, left_index=True, right_index=True, how='left').ffill()
df = pd.merge(df, control_df, left_index=True, right_index=True, how='left').ffill()

# Running the roling regression
ols = RollingOLS(window=12, expanding=False, lambda_=0.0)
ols.fit(factors=df[factors], controls=df[controls], orthogonalize_controls=True, orthogonalize_factors=True)
result = ols.transform(assets=df[assets])

# Plot some results
for f in factors:
    result.get_beta(f).plot(title=f)
```

---

## API

### `RollingOLS(...)` — constructor

| Parameter | Default | Description |
|---|---|---|
| `window` | `252` | Rolling window length |
| `min_periods` | `window` | Minimum observations to produce a result |
| `expanding` | `False` | Use expanding window instead of rolling |
| `lambda_` | `0.0` | Ridge regularization. `0` = standard OLS |
| `adj_r2` | `False` | Compute adjusted R² |
| `lag_signal` | `False` | Use `beta_{t-1} * factor_t` instead of `beta_t * factor_t` |
| `hac_lags` | `None` | Newey-West lags for HAC SE. `None` disables HAC |
| `dtype` | `"float32"` | Input dtype. Use `"float64"` for higher precision |
| `asset_chunk_size` | `100` | Controls peak memory during residualization |

---

### `.fit(factors, controls=None, orthogonalize_factors=False, orthogonalize_controls=False)`

Fits the model on the regressors side. Residualizes factors against controls (Frisch-Waugh step 1) if controls are provided.

```python
# No controls
ols.fit(df[["f1", "f2", "f3"]])

# With controls
ols.fit(df[["f1", "f2"]], controls=df[["ctrl1", "ctrl2"]])

# With rolling orthogonalization
# f1 is untouched, f2 is orthogonalized against f1, f3 against f1 and f2
ols.fit(df[["f1", "f2", "f3"]], orthogonalize_factors=True)
```

**Column order matters for orthogonalization** — place higher-priority factors first. Each factor's beta will then reflect its incremental contribution beyond all preceding ones.

---

### `.transform(targets, return_control_betas=False)`

Projects targets onto the fitted factor structure and returns a `RollingOLSResult`.

```python
result = ols.transform(df[["y1", "y2", "y3"]])

# With control betas (more expensive)
result = ols.transform(df[["y1", "y2"]], return_control_betas=True)
```

The fitted model can be reused on different target sets without re-fitting:

```python
ols.fit(df[["f1", "f2"]], controls=df[["ctrl1"]])
result_a = ols.transform(df[group_a])
result_b = ols.transform(df[group_b])
```

---

### `.fit_transform(factors, targets, controls=None, ...)`

Convenience one-liner when you don't need to reuse the fitted model.

```python
result = RollingOLS(window=60).fit_transform(
    df[["f1", "f2"]],
    df[["y1", "y2"]],
    controls=df[["ctrl1"]],
    orthogonalize_factors=True,
)
```

---

### `RollingOLSResult` — getters

All results are indexed by time (rows) and target (columns).

```python
result.get_beta("f1")          # DataFrame (T x N_targets)
result.get_signal("f1")        # beta_t * factor_t (or lagged)
result.get_r2("f1")            # rolling R²
result.get_residuals("f1")     # regression residuals

result.get_se("f1")            # Newey-West SE — requires hac_lags
result.get_tstat("f1")         # beta / SE

result.get_control_beta("f1", "ctrl1")  # requires return_control_betas=True
```

**Long format** — useful for downstream analysis, filtering, or plotting:

```python
result.to_long("f1")                    # date, target, beta, signal, r2
result.to_long("f1", include_se=True)   # + se, t_stat
result.to_long_all()                    # all factors stacked
```

---

## Examples

### Ridge regression

```python
# lambda_ > 0 adds λI to X'X before solving
# stabilizes estimation when factors are correlated
ols = RollingOLS(window=120, lambda_=1e-3)
result = ols.fit(df[["f1", "f2", "f3"]]).transform(df[targets])
```

### HAC standard errors

```python
import numpy as np

# Common rule of thumb for lag selection: floor(T^(1/3))
hac_lags = int(np.floor(len(df) ** (1/3)))

ols = RollingOLS(window=120, hac_lags=hac_lags)
result = ols.fit(df[["f1", "f2"]]).transform(df[targets])

se    = result.get_se("f1")      # Newey-West SE
tstat = result.get_tstat("f1")  # t-statistics
```

### Orthogonalization with importance ordering

When factors have a natural priority order, orthogonalization ensures each beta measures incremental contribution beyond higher-priority factors.

```python
# f1 is the primary factor — left untouched
# f2 is orthogonalized against f1
# f3 is orthogonalized against f1 and f2
ols = RollingOLS(window=120)
ols.fit(
    df[["f1", "f2", "f3"]],
    orthogonalize_factors=True,
)
result = ols.transform(df[targets])
```

### Expanding window

```python
ols = RollingOLS(window=30, min_periods=30, expanding=True)
result = ols.fit(df[["f1"]]).transform(df[targets])
```

### Lagged signal (avoiding look-ahead)

```python
# beta estimated at t-1, multiplied by factor at t
ols = RollingOLS(window=60, lag_signal=True)
result = ols.fit(df[["f1"]]).transform(df[targets])
signal = result.get_signal("f1")
```

---

## Design notes

**Frisch-Waugh-Lovell** — when controls are provided, rOLS residualizes both factors and targets against controls before running the per-factor regression. This is mathematically equivalent to the full joint regression but keeps the inner loop univariate, making it fast regardless of how many controls are added.

**Stride tricks** — the rolling window matrix operations use `numpy.lib.stride_tricks.as_strided` to build zero-copy sliding window views, avoiding explicit loops over time for the fixed-window case.

**Memory** — asset residualization is chunked (`asset_chunk_size`) to bound peak memory when the number of targets is large. Reduce this value if you hit memory limits.

**HAC caching** — standard errors are computed lazily and cached on first call to `get_se()`. Calling it multiple times for the same factor incurs no extra cost.