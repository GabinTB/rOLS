# rOLS

**Vectorized rolling and expanding regression for multi-target, multi-factor time series.**

Built for performance at panel scale: hundreds of targets over thousands of
time steps, without Python loops over time.

Adapted for applications where dynamic relationships matter most: estimating
rolling betas in finance to isolate idiosyncratic sensitivity to narrative
factors; tracking time-varying price elasticities in economics to capture
structural shifts; attributing regional temperature anomalies in climate
science to forcing factors; and adaptively filtering signals in real time.

[![PyPI version](https://img.shields.io/pypi/v/rols)](https://pypi.org/project/rols/)
[![Python versions](https://img.shields.io/pypi/pyversions/rols)](https://pypi.org/project/rols/)
[![License](https://img.shields.io/pypi/l/rols)](https://github.com/GabinTB/rOLS/blob/main/LICENSE)

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
  control, and every regressor in the selected model are simultaneously finite.
- Every reported quantity for `(i, t)` — beta, intercept, residual, R², SE —
  comes from **this one fit**.

This is a **time-series** rolling regression. See the full
[statistical specification](reference/specification.md) for details.

---

## Features

- **OLS and Ridge** regression (`lambda_`), normalized so the same value has
  the same effective strength regardless of window length
- **Multiple controls**, partialled out via within-window Frisch-Waugh-Lovell
- **Batched or joint multi-factor models** (`mode`)
- **HAC standard errors** (Newey-West), computed on demand
- **Expanding windows** as an alternative to fixed rolling windows
- **EWMA observation weighting** within each window
- **Lagged signals** to avoid look-ahead bias
- **Sparse estimation cadence** (`estimate_every`)

---

## Quick install

```bash
pip install rols
```

Requires Python 3.10+ and numpy / pandas.

---

## Quick start

```python
from rols import RollingOLS

ols = RollingOLS(window=60, mode="joint")
ols.fit(df[["f1", "f2"]], controls=df[["ctrl1"]])
result = ols.transform(df[targets])

result.get_beta("f1")       # DataFrame (T × N_targets)
result.get_se("f1")         # Newey-West SE (requires hac_lags)
result.get_r2("f1")         # full-model R²
```

See the [Getting Started](guide/getting-started.md) guide for a complete walkthrough.

---

## Next steps

- [Getting Started](guide/getting-started.md) — installation, first regression, Fama-French example
- [Usage Patterns](guide/usage-patterns.md) — batched vs joint, missing data, Ridge, HAC, EWMA, and more
- [API Reference](api/rolling-ols.md) — full parameter and method documentation
- [Statistical Specification](reference/specification.md) — the formal estimator definition
- [Performance Guide](reference/performance.md) — cost model and benchmark numbers
