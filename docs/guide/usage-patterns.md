# Usage Patterns

## Batched vs joint mode

When you supply more than one factor, you must pass `mode` explicitly.
rOLS raises `ValueError` otherwise — the two estimands differ whenever factors
are correlated.

```python
# ValueError: 3 factors were supplied but `mode` was not specified.
ols = RollingOLS(window=60)
ols.fit(df[["f1", "f2", "f3"]])

# OK — explicit choice:
ols = RollingOLS(window=60, mode="joint")
result = ols.fit(df[["f1", "f2", "f3"]]).transform(df[targets])
```

`mode="joint"` fits the multivariate model `y = α + β₁f₁ + β₂f₂ + β₃f₃ + ε`
once — each beta is conditional on the controls **and** on every other factor.

`mode="batched"` fits **three separate regressions**, `y ~ 1 + controls + f_j`
for each `j`. Each `β_j` is conditional on the controls but **not** on the
other factors. Use `warn_correlated_factors=True` (the default) to get a
one-time warning when batched-mode factors are correlated above `|ρ| > 0.3`.

Which mode is *faster* depends on `lambda_`:

- **`lambda_ == 0` (OLS):** batched uses the Frisch-Waugh fast path — measured
  faster than joint at panel scale.
- **`lambda_ > 0` (Ridge):** joint mode is substantially cheaper (~5× faster
  at K=20). See the [Performance Guide](../reference/performance.md) for
  measured numbers.

---

## Missing data

One rule, applied uniformly: a row is used for target `i` at endpoint `t` iff
the intercept design, every control, every factor in the selected model, and
the target are simultaneously finite. Rows failing that test are dropped, not
filled — rOLS never imputes. A result is emitted at `t` iff at least
`min_periods` such rows survive.

Missingness in target `i` never affects the sample or result for target `j`.
See [Specification §6](../reference/specification.md#6-missing-data) for the
formal statement.

---

## R² variants

`get_r2(factor)` is the **full model's** R². `get_partial_r2(factor)` is
`factor`'s **incremental** contribution: the number to read when factors are
correlated.

`adj_r2=True` on the constructor makes both accessors report the adjusted
statistic. Under Ridge, these are descriptive fit metrics — penalized residuals
are not orthogonal to the regressors, so `get_partial_r2` can be negative.

---

## Ridge regression

```python
# lambda_ > 0 penalizes the normalized, standardized objective
ols = RollingOLS(window=120, lambda_=1e-3, mode="joint")
result = ols.fit(df[["f1", "f2", "f3"]]).transform(df[targets])
```

The penalty strength is invariant to window length, EWMA half-life, and
complete-case sample size.

---

## HAC standard errors

```python
import numpy as np

# Common rule of thumb: floor(T^(1/3))
hac_lags = int(np.floor(len(df) ** (1/3)))

ols = RollingOLS(window=120, hac_lags=hac_lags, mode="joint")
result = ols.fit(df[["f1", "f2"]]).transform(df[targets])

se    = result.get_se("f1")
tstat = result.get_tstat("f1")
```

Each standard error is computed from the same current-window fit as its beta.

!!! warning "HAC SEs are time-series only"
    rOLS computes a Newey-West sandwich per asset that corrects for serial
    correlation. It does **not** correct for cross-sectional dependence across
    assets. For panel-robust inference, use clustered standard errors or the
    Driscoll-Kraay estimator via an external tool.

!!! note "Ridge inference caveat"
    When `lambda_ > 0`, the sandwich estimates variability of β̂_λ around the
    penalized pseudo-true parameter β_λ — not the OLS coefficient β₀. See
    [Specification §10](../reference/specification.md) for the estimand.

---

## EWMA observation weighting

```python
# ~3-month half-life inside a 1-year window
ols = RollingOLS(window=252, ewma_halflife=63, mode="joint")
result = ols.fit(df[["f1", "f2"]]).transform(df[targets])
```

The weighting flows through betas, R², HAC standard errors, and the
Frisch-Waugh residualization. `ewma_halflife` cannot be combined with
`expanding=True`.

---

## Expanding window

```python
ols = RollingOLS(window=30, min_periods=30, expanding=True)
result = ols.fit(df[["f1"]]).transform(df[targets])
```

---

## Lagged signal (avoiding look-ahead)

```python
ols = RollingOLS(window=60, lag_signal=True)
result = ols.fit(df[["f1"]]).transform(df[targets])
signal = result.get_signal("f1")  # beta_{t-1} * factor_t
```

---

## Sparse cadence (`estimate_every`)

On a large panel, re-solving every single endpoint is often more resolution
than needed.

```python
# Re-estimate weekly instead of daily
ols = RollingOLS(window=252, estimate_every="W-FRI")
result = ols.fit(df[["f1"]]).transform(df[targets])
```

`get_*` accessors return the full index with NaN at skipped endpoints.
`iter_beta()` and `iter_se()` yield compact frames — computed endpoints only.

---

## Precision

The `precision` parameter controls both storage and computation dtypes:

| Mode | Storage dtype | Compute dtype | Use case |
|---|---|---|---|
| `"double"` (default) | float64 | float64 | Maximum numerical fidelity |
| `"mixed"` | float32 | float64 | Halves memory for large panels |
| `"single"` | float32 | float32 | Smallest footprint (~4-digit accuracy) |

```python
model = RollingOLS(window=60, precision="mixed")
```

---

## Memory and scale

`estimate_memory()` reports the persistent and on-demand cost **before**
fitting:

```python
memory = RollingOLS(window=252, cache_size=1).estimate_memory(
    targets=df[targets],
    factors=df[factors],
    controls=df[controls],
)
print(memory["total"])
```

The multiplier that matters is **per factor, per retained quantity**. Use
`cache_size=1` (the default) and `iter_beta()` / `iter_se()` to keep memory
bounded. `estimate_every` reduces cost multiplicatively.

See the [Performance Guide](../reference/performance.md) for the full breakdown.
