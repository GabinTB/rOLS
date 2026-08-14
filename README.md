# rOLS

Vectorized rolling and expanding regression for multi-target, multi-factor time series.

Built for performance at scale: hundreds of regressors against hundreds of targets, over thousands of time steps, without loops.

Adapted for applications where dynamic relationships matter most: estimating rolling betas in finance to isolate idiosyncratic sensitivity to narrative factors; tracking time-varying price elasticities in economics to capture structural shifts; attributing regional temperature anomalies in climate science to forcing factors; and adaptively filtering signals in real time. Designed for speed and scalability, it enables precise, loop-free analysis across domains where traditional methods fall short.

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
ols.fit(factors=df[factors], controls=df[controls])
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
| `ewma_halflife` | `None` | Exponentially weight observations within each window (half-life in periods). `None` = equal weighting. Not compatible with `expanding=True` |
| `adj_r2` | `False` | Compute adjusted R² |
| `lag_signal` | `False` | Use `beta_{t-1} * factor_t` instead of `beta_t * factor_t` |
| `hac_lags` | `None` | Newey-West lags for HAC SE. `None` disables HAC |
| `dtype` | `"float32"` | DataFrame storage dtype (see note below). Use `"float64"` for higher precision |
| `asset_chunk_size` | `100` | Controls peak memory during residualization |
| `warn_singular` | `True` | Warn (RuntimeWarning) on singular windows. Set `False` to suppress |

---

### `.fit(factors, controls=None)`

Fits the model on the regressors side. Residualizes factors against controls (Frisch-Waugh step 1) if controls are provided.

```python
# No controls
ols.fit(df[["f1", "f2", "f3"]])

# With controls
ols.fit(df[["f1", "f2"]], controls=df[["ctrl1", "ctrl2"]])

```

---

### `.transform(targets)`

Projects targets onto the fitted factor structure and returns a `RollingOLSResult`.

```python
result = ols.transform(df[["y1", "y2", "y3"]])
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
)
```

---

### `RollingOLSResult` — getters

All results are indexed by time (rows) and target (columns).

```python
result.get_beta("f1")          # DataFrame (T x N_targets)
result.get_signal("f1")        # beta_t * factor_t (or lagged)
result.get_r2("f1")            # rolling R²
result.get_residuals("f1")     # regression residuals (FWL step 3)
result.get_factor_adjusted_returns()    # controls removed only (FWL step 2)

result.get_se("f1")            # Newey-West SE — requires hac_lags
result.get_tstat("f1")         # beta / SE

result.get_control_beta("f1", "ctrl1")  # requires return_control_betas=True

result.get_factor_mimicking_returns("f1")    # Series (T,) — cross-sectional λ_t
result.get_all_factor_mimicking_returns()    # DataFrame (T x K)
```

`get_factor_adjusted_returns()` returns asset returns with only the **controls**
partialled out (`e_it = r_it - B_t' * ctrl_t`, FWL step 2) — not specific to any
factor, so it takes no argument. This differs from `get_residuals(factor)`, which
additionally removes the narrative `factor` (FWL step 3). If no controls were
provided at `fit()`, it returns the original asset returns.

`get_control_beta(factor, control)` returns the control's **joint** rolling beta —
its coefficient from the full regression, recovered via Frisch-Waugh-Lovell
(each control partialled out against all other controls), not a univariate
marginal beta. The value does not depend on `factor` (control betas are shared
across factors). Requires `return_control_betas=True` on `transform()`/`fit_transform()`.

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

### EWMA observation weighting

By default every observation in a window counts equally. When recent data should
carry more weight — e.g. narrative-beta estimation in finance, where the latest
behaviour matters most — set `ewma_halflife` to weight observations
exponentially. An observation `ewma_halflife` periods in the past gets half the
weight of the most recent one.

```python
# ~3-month half-life inside a 1-year window
ols = RollingOLS(window=252, ewma_halflife=63)
result = ols.fit(df[["f1", "f2"]]).transform(df[targets])
```

The weighting flows through the betas, R², and the Frisch-Waugh residualization
(weighted least squares per window). NaN rows are dropped per window and the
surviving weights renormalized to sum to 1, so missing data does not distort the
scheme. `ewma_halflife` cannot be combined with `expanding=True` (an expanding
window has no fixed length to precompute weights over), and HAC standard errors
are still computed with equal weights.

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

### Cross-sectional factor mimicking returns

rOLS is usually run in the **time-series direction** (factors as regressors,
asset returns as targets). Flip the orientation and it runs a **cross-sectional**
regression instead: at each date the assets are the observations, their factor
betas are the regressors, and their returns are the single target. The estimated
coefficient `λ_t` is then the **factor mimicking return** `g_t` — the return per
unit of factor exposure in the cross-section — the central quantity in a
Fama-MacBeth / pure-factor portfolio pipeline.

```python
# Cross-sectional: assets as observations, betas as regressors, returns as target
# asset_betas : (T x K) — each column is a factor's cross-sectional exposure
# asset_return: (T x 1) — the single target column
ols = RollingOLS(window=1)               # or expanding for a growing panel
ols.fit(asset_betas)                     # K factors as regressors
result = ols.transform(asset_return[["returns"]])   # single target column

g     = result.get_factor_mimicking_returns("f1")   # Series (T,) — λ_t for f1
g_all = result.get_all_factor_mimicking_returns()   # DataFrame (T x K)
```

The single-target requirement is enforced: `get_factor_mimicking_returns()`
raises `RuntimeError` if `transform()` was called with more than one target
column (i.e. used in the time-series direction). Downstream steps — factor
timing regressions, spanning tests, factor covariance estimation — consume the
`g_t` series directly.

---

## Design notes

**Frisch-Waugh-Lovell** — when controls are provided, rOLS residualizes both factors and targets against controls before running the per-factor regression. This is mathematically equivalent to the full joint regression but keeps the inner loop univariate, making it fast regardless of how many controls are added.

**Stride tricks** — the rolling window matrix operations use `numpy.lib.stride_tricks.as_strided` to build zero-copy sliding window views, avoiding explicit loops over time for the fixed-window case.

**Memory** — asset residualization is chunked (`asset_chunk_size`) to bound peak memory when the number of targets is large. Reduce this value if you hit memory limits.

**Precision (`dtype`)** — `dtype` controls the storage precision of the input and intermediate pandas DataFrames only. Internal matrix operations (gram matrix accumulation and the linear solve) always run in **float64** regardless of this setting, because `np.linalg.solve` loses accuracy in float32 for ill-conditioned windows. So `float32` reduces DataFrame memory but does not change the numerical precision of the regression itself.

**HAC caching** — standard errors are computed lazily and cached on first call to `get_se()`. Calling it multiple times for the same factor incurs no extra cost.
