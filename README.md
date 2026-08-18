# rOLS

Vectorized rolling and expanding regression for multi-target, multi-factor time series.

Built for performance at panel scale: hundreds of targets over thousands of time
steps, without Python loops over time. See [Memory and scale](#memory-and-scale)
below for what actually bounds that claim.

Adapted for applications where dynamic relationships matter most: estimating
rolling betas in finance to isolate idiosyncratic sensitivity to narrative
factors; tracking time-varying price elasticities in economics to capture
structural shifts; attributing regional temperature anomalies in climate
science to forcing factors; and adaptively filtering signals in real time.

| Metric | Value |
|---|---|
| PyPI Version | [![PyPI version](https://img.shields.io/pypi/v/rols)](https://pypi.org/project/rols/) |
| Python Versions | [![Python versions](https://img.shields.io/pypi/pyversions/rols)](https://pypi.org/project/rols/) |
| License | [![License](https://img.shields.io/pypi/l/rols)](https://github.com/GabinTB/rOLS/blob/main/LICENSE) |
| Downloads | [![Downloads](https://img.shields.io/pypi/dm/rols)](https://pypi.org/project/rols/) |
| GitHub Stars | [![Stars](https://img.shields.io/github/stars/GabinTB/rOLS?style=social)](https://github.com/GabinTB/rOLS) |

> **v0.3.0 is a correctness release.** An independent audit found that v0.2.1
> and earlier estimated a statistically inconsistent model — see
> [Migration from v0.2.x](#migration-from-v02x) and
> [`CHANGELOG.md`](CHANGELOG.md). If you have v0.2.x estimates in production,
> re-run them; do not treat this as a routine version bump.

---

## The estimator

For target `i`, endpoint `t`, and the window `W_t` of the last `window`
observations ending at `t`, rOLS solves one weighted least-squares problem on
the complete-case rows of `W_t`:

```
y_i,s = α_i,t + c_s'γ_i,t + f_s'β_i,t + ε_i,s,t,     s ∈ S_i,t ⊆ W_t
```

- `f_s` are the factors of interest (one rolling beta each), `c_s` the always-in
  controls, `α_i,t` the intercept.
- `S_i,t` is the complete-case subset of `W_t`: rows where the target, every
  control, and every regressor in the selected model are simultaneously
  finite. Missing values are dropped, never filled — see
  [Missing data](#missing-data).
- Every reported quantity for `(i, t)` — beta, intercept, residual, R², SE —
  comes from **this one fit**. Nothing is assembled from other endpoints'
  fits, and nothing here is fabricated cross-sectional information: `f_s'β`
  is one asset's own time-series exposure, not a return earned by a
  portfolio.

This is a **time-series** rolling regression. See [Out of scope](#out-of-scope)
for what it is deliberately not.

The full statistical specification — window semantics, Frisch-Waugh-Lovell,
Ridge normalization, EWMA weighting, R² variants, HAC inference, the index
contract — is [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md). This README is
a practical guide to the same estimator; the specification is the source of
truth when they disagree.

rOLS supports:
- **OLS and Ridge** regression (`lambda_`), normalized so the same value has
  the same effective strength regardless of window length, EWMA half-life, or
  complete-case sample size
- **Multiple controls**, partialled out via within-window Frisch-Waugh-Lovell
  — mathematically equivalent to the joint solve, kept as a fast path only
  where it provably matches it (OLS, `lambda_ == 0`)
- **Batched or joint multi-factor models** (`mode`) — marginal-given-controls
  screening betas, or a proper multivariate fit
- **HAC standard errors** (Newey-West), computed on demand from the same fit
  as the reported beta
- **Expanding windows** as an alternative to fixed rolling windows
- **EWMA observation weighting** within each window
- **Lagged signals** to avoid look-ahead bias
- **Sparse estimation cadence** (`estimate_every`) to cut cost on large panels

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
factor_df = web.DataReader(['CPIAUCSL', 'CPILFESL'], 'fred', start=start).pct_change().dropna()
factors = factor_df.columns.tolist()

# Loading some targets
asset_df = web.DataReader('12_Industry_Portfolios', 'famafrench', start=start)[1]
asset_df.index = asset_df.index.to_timestamp()
assets = asset_df.columns.tolist()

# Loading some controls
control_df = pdr.get_data_famafrench("F-F_Research_Data_Factors", start=start)[0].div(100.0).drop(columns=["RF"])
control_df.index = control_df.index.to_timestamp()
controls = control_df.columns.tolist()

# Merge data into one dataframe aligned by date — rOLS requires identical,
# unique, monotonic indexes; do not rely on implicit alignment (see
# "Index contract" below).
df = pd.merge(factor_df, asset_df, left_index=True, right_index=True, how='left').ffill()
df = pd.merge(df, control_df, left_index=True, right_index=True, how='left').ffill()

# Running the rolling regression
ols = RollingOLS(window=12, expanding=False, lambda_=0.0)
ols.fit(factors=df[factors], controls=df[controls])
result = ols.transform(assets=df[assets])

# Plot some results
for f in factors:
    result.get_beta(f).plot(title=f)
```

See [`examples/fama_french_factors.ipynb`](examples/fama_french_factors.ipynb)
for a runnable, end-to-end version covering `mode`, full vs partial R², and
`estimate_every`.

---

## API

### `RollingOLS(...)` — constructor

| Parameter | Default | Description |
|---|---|---|
| `window` | `252` | Rolling window length |
| `min_periods` | `window` | Minimum complete-case rows to produce a result |
| `expanding` | `False` | Use expanding window instead of rolling |
| `fit_intercept` | `True` | Fit an explicit intercept column (not centering) |
| `mode` | `"batched"` | `"batched"`: one model per factor, marginal-given-controls. `"joint"`: one model with every factor, mutually controlled. See [Batched vs joint](#batched-vs-joint-mode) |
| `warn_correlated_factors` | `True` | Warn once when `mode="batched"` and any factor pair has sample `\|correlation\| > 0.3` |
| `lambda_` | `0.0` | Ridge strength on the normalized objective. `0` = OLS |
| `penalize_controls` | `True` | Penalize controls too when `lambda_ > 0` |
| `ewma_halflife` | `None` | Exponentially weight observations within each window (half-life in periods). `None` = equal weighting. Not compatible with `expanding=True` |
| `adj_r2` | `False` | Report adjusted R² from `get_r2`/`get_partial_r2` instead of R² |
| `lag_signal` | `False` | Use `beta_{t-1} * factor_t` instead of `beta_t * factor_t` |
| `hac_lags` | `None` | Newey-West lags for HAC SE. `None` disables HAC (`get_se`/`get_tstat` raise) |
| `denom_tol` | `1e-12` | Threshold below which a variance/SST is treated as zero (NaN out, not `inf`) |
| `dtype` | `"float32"` | DataFrame storage dtype (see [Precision](#precision-dtype)). Solve arithmetic is always float64 |
| `asset_chunk_size` | `100` | Targets processed per chunk during residualization; bounds peak memory |
| `cache_size` | `1` | Factors retained in each on-demand result cache |
| `warn_singular` | `True` | Warn once on singular or ill-conditioned windows (affected estimates become NaN, or become numerically unreliable but finite for ill-conditioning — see [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md)) |
| `cond_warn_threshold` | `1e10` | Warn when `cond(X'X)` for a window's design exceeds this |
| `estimate_every` | `1` | Estimate only every `k`-th endpoint, or the last observation per pandas offset period — see [Sparse cadence](#sparse-cadence-estimate_every) |

---

### `.fit(factors, controls=None)`

Fits the model on the regressors side. Residualizes factors against controls
(Frisch-Waugh step 1) if controls are provided and the fast path applies.

```python
# No controls
ols.fit(df[["f1", "f2", "f3"]])

# With controls
ols.fit(df[["f1", "f2"]], controls=df[["ctrl1", "ctrl2"]])
```

---

### `.transform(targets, return_control_betas=False)`

Projects targets onto the fitted factor structure and returns a
`RollingOLSResult`.

```python
result = ols.transform(df[["y1", "y2", "y3"]])
```

The fitted model can be reused on different target sets without re-fitting:

```python
ols.fit(df[["f1", "f2"]], controls=df[["ctrl1"]])
result_a = ols.transform(df[group_a])
result_b = ols.transform(df[group_b])
```

`return_control_betas=True` additionally computes and stores each control's
joint rolling beta, retrievable via `result.get_control_beta(factor, control)`.

---

### `RollingOLSResult` — getters

All results are indexed by time (rows) and target (columns).

```python
result.get_beta("f1")          # DataFrame (T x N_targets)
result.get_signal("f1")        # beta_t * factor_t (or lagged)
result.get_r2("f1")            # full-model R²
result.get_partial_r2("f1")    # f1's incremental R² over the model without it
result.get_residuals("f1")     # endpoint regression residuals
result.get_factor_adjusted_returns()    # controls removed only (FWL step 2)

result.get_se("f1")            # Newey-West SE — requires hac_lags
result.get_tstat("f1")         # beta / SE

result.get_control_beta("f1", "ctrl1")  # requires return_control_betas=True
result.get_dof("f1")           # residual degrees of freedom
result.get_n_used("f1")        # complete-case row count per endpoint
result.mode                    # "batched" or "joint" — how this result was fit
```

`transform()` materializes betas, intercepts, and observation counts. Signals,
R², residuals, standard errors, and t-statistics are computed when requested.
The residual, R², and standard-error caches retain at most `cache_size`
factors — see [Memory and scale](#memory-and-scale).

Every factor getter accepts an optional target subset:

```python
result.get_beta("f1", assets=["AAPL", "MSFT"])
result.get_se("f1", assets=["AAPL", "MSFT"])
```

For whole-panel work, iterate one factor at a time so inference frames can be
released as the loop advances:

```python
for factor, beta in result.iter_beta():
    process(factor, beta)

for factor, se in result.iter_se():
    process(factor, se)
```

`get_factor_adjusted_returns()` returns target values with only the
**controls** partialled out (`e_it = r_it - γ_t'c_t`, FWL step 2) — not
specific to any factor, so it takes no argument. This differs from
`get_residuals(factor)`, which additionally removes the named `factor` (FWL
step 3). If no controls were provided at `fit()`, it returns the original
target values.

`get_control_beta(factor, control)` returns the control's rolling coefficient
from the fit that also contains `factor`. **In batched mode this depends on
which `factor` you name**: each factor defines its own joint model
`y ~ 1 + controls + factor`, so a control correlated with two different
factors gets two different coefficients. In joint mode there is one shared
model, so the value is the same for every `factor` argument. Requires
`return_control_betas=True` on `transform()`/`fit_transform()`.

**Long format** — useful for downstream analysis, filtering, or plotting:

```python
result.to_long("f1")                    # date, target, beta, signal, r2
result.to_long("f1", include_se=True)   # + se, t_stat
result.to_long_all()                    # all factors stacked
```

---

## Batched vs joint mode

Given `factors = ["f1", "f2", "f3"]`, it is natural to expect the multivariate
model `y = α + β₁f₁ + β₂f₂ + β₃f₃ + ε`. That is `mode="joint"`.

`mode="batched"` (the **default**, for backward compatibility) instead fits
**three separate regressions**, `y ~ 1 + controls + f_j` for each `j`. Each
`β_j` is conditional on the controls but not on the other factors — a
legitimate estimator for signal screening, but if `f1` and `f2` are
correlated, each beta will silently absorb variation attributable to the
other. This is why `warn_correlated_factors=True` (the default) warns once
when batched-mode factors are correlated above `|ρ| > 0.3`.

```python
ols = RollingOLS(window=60, mode="joint")
result = ols.fit(df[["f1", "f2", "f3"]]).transform(df[targets])
# result.get_beta("f1") is now conditional on f2 and f3 too
```

The two modes coincide exactly for a single factor, or for factors that are
mutually orthogonal on the estimation sample. Which mode is *faster* depends
on `lambda_`, not on which is statistically correct — see
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md#batched-vs-joint) for measured
numbers:

- **`lambda_ == 0` (OLS):** batched uses the Frisch-Waugh fast path, sharing
  one controls-only projection and one GEMM across every factor — measured
  faster than joint at panel scale, more so as K grows.
- **`lambda_ > 0` (Ridge):** FWL does not commute with the penalty, so batched
  falls back to K separate joint-equivalent solves. Joint mode is exactly one
  such solve, so it is substantially cheaper — measured ~5x faster at K=20.

So: default to batched for OLS screening, prefer joint whenever factors are
correlated (for correctness), and prefer joint under Ridge regardless (it is
both more correct *and* faster there).

---

## Missing data

One rule, applied uniformly: a row is used for target `i` at endpoint `t` iff
the intercept design, every control, every factor in the selected model, and
the target are simultaneously finite. Rows failing that test are dropped, not
filled — rOLS never imputes. A result is emitted at `t` iff at least
`min_periods` such rows survive.

Missingness in target `i` never affects the sample or result for target `j`.
In batched mode, missingness in one factor only invalidates the model that
uses it — other factors are unaffected. In joint mode, missingness in any
factor invalidates the shared model for every factor.

See [`docs/SPECIFICATION.md §6`](docs/SPECIFICATION.md#6-missing-data) for the
formal statement.

---

## R² variants

`get_r2(factor)` is the **full model's** R² — in batched mode, the model
`y ~ 1 + controls + factor`; in joint mode, the one shared model, identical
across every `factor` argument.

`get_partial_r2(factor)` is `factor`'s **incremental** contribution:
`(SSR_reduced - SSR_full) / SSR_reduced`, where the reduced model drops
`factor` and keeps everything else, evaluated on the full model's
complete-case sample. This is the number to read when factors are correlated
— `get_r2` conflates the whole model's fit with one factor's contribution to
it.

`adj_r2=True` on the constructor makes both accessors report the adjusted
statistic, using effective sample size (`n_eff`, which equals the raw
complete-case count under equal weighting and something smaller under EWMA)
rather than a raw row count.

---

## Examples

### Ridge regression

```python
# lambda_ > 0 penalizes the normalized, standardized objective
# stabilizes estimation when factors are correlated; effective strength is
# invariant to window length, EWMA half-life, and complete-case sample size
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
tstat = result.get_tstat("f1")   # t-statistics
```

Each standard error is computed from the same current-window fit as its beta.
The sandwich uses the full design, including the intercept and controls, the
same complete-case rows, Bartlett lag weights, and the estimator's observation
weights. Computation is lazy and streams one endpoint at a time.

### EWMA observation weighting

By default every observation in a window counts equally. When recent data
should carry more weight — e.g. narrative-beta estimation in finance, where
the latest behaviour matters most — set `ewma_halflife` to weight observations
exponentially. An observation `ewma_halflife` periods in the past gets half
the weight of the most recent one.

```python
# ~3-month half-life inside a 1-year window
ols = RollingOLS(window=252, ewma_halflife=63)
result = ols.fit(df[["f1", "f2"]]).transform(df[targets])
```

The weighting flows through the betas, R², HAC standard errors, and the
Frisch-Waugh residualization (weighted least squares per window). NaN rows are
dropped per window and the surviving weights renormalized to sum to 1, so
missing data does not distort the scheme. `ewma_halflife` cannot be combined
with `expanding=True` (an expanding window has no fixed length to precompute
weights over).

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

### Sparse cadence (`estimate_every`)

On a large panel, re-solving every single endpoint is often more resolution
than needed. `estimate_every` restricts the solver to a coarser cadence — an
integer step count, or a pandas offset alias (e.g. `"W-FRI"`) — while every
kept window still contains every underlying observation (`hac_lags` remains
measured in observations, not cadence steps).

```python
# Re-estimate weekly instead of daily
ols = RollingOLS(window=252, estimate_every="W-FRI")
result = ols.fit(df[["f1"]]).transform(df[targets])
```

`get_*` accessors return the full index with NaN at skipped endpoints, so
downstream code expecting a dense index keeps working. `iter_beta()` and
`iter_se()` yield compact frames — computed endpoints only.

---

## Out of scope

rOLS is a **time-series** rolling regression: factors are regressors and
targets are the dependent series across sequential time observations. It does
not provide:

- **Cross-sectional factor-return estimation** — where assets are the
  observations at each date, not the targets, and a factor return is
  recovered period-by-period across the asset cross-section. This needs a
  different data model (date × asset × factor), specification, and oracle.
  The `get_factor_mimicking_returns()` accessors that briefly existed in
  v0.2.1's development were removed (F13): they renamed a time-series rolling
  beta and performed no such estimation.
- **Factor-mimicking portfolio construction** — the cross-sectional problem
  above, not a time-series one.
- **Sequential Gram-Schmidt or other factor orthogonalization** — this is
  preprocessing, not estimation, and under a rolling basis it changes the
  statistical object at every endpoint (a coefficient change can reflect
  shifting factor correlations rather than shifting target sensitivity). Apply
  it to your inputs before calling `fit()` if you need it; `mode="joint"` is
  the standard way to estimate mutually-controlled factor effects without it.
- **Panel estimators with entity or time fixed effects.**
- **Implicit data alignment, resampling, imputation, or calendar conversion**
  — see [Index contract](#index-contract) and [Missing data](#missing-data).
  Align and handle missingness in your inputs; rOLS raises rather than
  guessing.

File a feature request if you need one of these; each is its own estimator
with its own specification, not a mode flag on this one.

---

## Index contract

Factors, controls, and targets must have indexes that are identical in length,
labels, order, and type; unique; and monotonically increasing. Violations
raise `ValueError` before any array conversion or estimation — rOLS does not
sort, deduplicate, reindex, join, or drop labels implicitly. Align your inputs
first, e.g.:

```python
df = pd.concat([factors, controls, targets], axis=1).dropna()
```

---

## Memory and scale

`estimate_memory()` reports the persistent and on-demand cost **before**
fitting, from input shapes alone:

```python
memory = RollingOLS(window=252, cache_size=1).estimate_memory(
    targets=df[targets],
    factors=df[factors],
    controls=df[controls],
)
print(memory["total"])
print(memory["note"])
```

Concretely, at the benchmark harness's `large` grid (`T=5040`, 2300 targets, 50
factors, 3 controls, `window=252`):

| Quantity | Cost |
|---|---|
| One accessor's full-index output (e.g. one `get_beta(f)` call) | ≈ 93 MB |
| Persistent betas (or intercepts) for all 50 factors | ≈ 4.6 GB |
| Total persistent footprint (`estimate_memory()["total"]`, `cache_size=1`) | ≈ 9.7 GB |

The multiplier that matters is **per factor, per retained quantity**. Calling
`get_beta`, `get_r2`, `get_residuals`, and `get_se` for every factor and
keeping every result alive at once costs roughly `4 × 50 × 93 MB ≈ 18 GB` on
this grid — before the input panel itself. This is why `cache_size` defaults
to `1` and why `iter_beta()` / `iter_se()` exist: they yield one factor's
frame at a time so the previous one can be released, instead of accumulating
`O(K)` frames.

`estimate_every` reduces this multiplicatively: skipping 4 out of every 5
endpoints cuts every per-frame cost roughly fivefold, at the cost of coarser
resolution.

Missing values in a factor split its sufficient statistics into
factor-specific complete-case patterns and may increase memory use relative to
the clean-data figures above. See
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md#memory) for the full breakdown and
the `structural` vs `scattered` NaN-pattern cases.

---

## Precision (`dtype`)

`dtype` controls the storage precision of the input and intermediate pandas
DataFrames only. Internal matrix operations (gram matrix accumulation and the
linear solve) always run in **float64** regardless of this setting, because
`np.linalg.solve`/QR lose accuracy in float32 for ill-conditioned windows.
`get_*` accessor outputs are likewise always float64 — `dtype` reduces input
storage memory, it does not change the numerical precision or the output
dtype of the regression itself.

---

## Migration from v0.2.x

v0.2.1 and earlier estimated a statistically inconsistent model. The table
below is what to expect when re-running old code against v0.3.0 — see
`CHANGELOG.md` and [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) for the
full detail behind each row.

| v0.2.x behaviour | v0.3.0 behaviour | What to expect |
|---|---|---|
| Factor betas used centred `cov/var`; control residualization and HAC used through-origin systems | One consistent model per fit, `fit_intercept=True` by default | Betas, residuals, and R² now describe the same regression; numbers change |
| With controls, a second rolling regression re-rolled first-pass residuals | One direct current-window joint (or FWL) solve | Warm-up halves: first estimate at `min_periods`, not `2 × min_periods` |
| `lambda_` had no effect without controls; with controls it penalized only the control residualization step | Single penalized joint solve on the full design, normalized so strength is invariant to window/EWMA/sample size | Ridge now actually shrinks; `lambda_` values are not comparable to v0.2.x |
| HAC SEs built from historical endpoints' own residuals | HAC computed from the *same* current-window fit as the reported beta | SEs change; some previously-finite SEs may now be NaN with a warning instead of an inaccurate number |
| `orthogonalize_factors` / `orthogonalize_controls` on `fit()` | Removed | Apply orthogonalization to your inputs before calling `fit()`, or use `mode="joint"` |
| No `mode` parameter; multi-factor was implicitly batched | `mode="batched"` (default, unchanged behaviour) or `mode="joint"` | No code change required; consider `mode="joint"` if your factors are correlated |
| `get_control_beta` omitted the named factor from the residualization set | Control beta comes from the joint fit that includes the named factor | Values change; batched-mode control betas now correctly vary by factor |
| `get_factor_mimicking_returns()` / `get_all_factor_mimicking_returns()` | Removed (F13) | Renamed a time-series rolling beta; see [Out of scope](#out-of-scope) |
| No input validation on `window`, `min_periods`, `lambda_`, etc. | Invalid constructor arguments raise `ValueError` at construction | Code passing invalid values now fails fast instead of producing silent NaNs |
| Factors, controls, and targets were aligned positionally (NumPy) or by label (pandas), depending on the internal path taken | Index must be unique, monotonically increasing, and identical across all three DataFrames; a `ValueError` is raised at fit time otherwise | Code passing permuted, duplicate, or mismatched indexes now raises instead of silently returning mispaired results |

---

## Design notes

**Frisch-Waugh-Lovell** — when controls are provided and `lambda_ == 0`, rOLS
residualizes both factors and targets against `[1, controls]` using the
*current window's own* projection, then solves the residualized univariate
regression. This is exactly equivalent to the direct joint solve (proven by
FWL and enforced by a differential test to `1e-10`) but shares one
factorization and one GEMM across every factor and target. `lambda_ > 0`
always routes to the direct joint solve — Ridge does not commute with FWL
residualization, so no fast path is used for it.

**Pattern grouping** — within one window, targets are grouped by their exact
complete-case mask, and each distinct group's design is factorized once and
solved for every target sharing that mask as a block of right-hand sides. This
degrades gracefully to per-target solves when every target has a unique
pattern (fully scattered missingness) and is a large win for the realistic
case (structural entry/exit, most targets sharing the all-present pattern).
See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for the cost model.

**Stride tricks** — the rolling window matrix operations use
`numpy.lib.stride_tricks.as_strided` to build zero-copy sliding window views,
avoiding explicit loops over time for the fixed-window case.

**HAC caching** — standard errors are computed lazily, one endpoint at a time,
and cached on first call to `get_se()`. Calling it multiple times for the same
factor incurs no extra cost. Use `iter_se()` to process all factors while
keeping the factor cache bounded by `cache_size`.

For the cost model behind these — where time actually goes, when the FWL fast
path applies, why joint is not automatically cheaper, and why rank-1 window
updating was considered and rejected — see
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

---

## Further reading

- [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) — the full statistical
  specification; the executable scalar oracle in `tests/oracle.py` implements
  it directly, and every optimized path is validated against that oracle by a
  differential test.
- [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) — cost model, memory
  arithmetic, and measured benchmark numbers.
- [`CHANGELOG.md`](CHANGELOG.md) — what changed in v0.3.0 and why.
- [`.claude/audits/20260813/`](.claude/audits/20260813/) — the independent
  audit that drove this release, including the finding each CHANGELOG entry
  references.
