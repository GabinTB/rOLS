# Migration from v0.2.x

v0.2.1 and earlier estimated a statistically inconsistent model. The table
below is what to expect when re-running old code against v0.3.x — see
[`CHANGELOG.md`](https://github.com/GabinTB/rOLS/blob/main/CHANGELOG.md) and
the [Statistical Specification](../reference/specification.md) for the full
detail behind each row.

| v0.2.x behaviour | v0.3.x behaviour | What to expect |
|---|---|---|
| Factor betas used centred `cov/var`; control residualization and HAC used through-origin systems | One consistent model per fit, `fit_intercept=True` by default | Betas, residuals, and R² now describe the same regression; numbers change |
| With controls, a second rolling regression re-rolled first-pass residuals | One direct current-window joint (or FWL) solve | Warm-up halves: first estimate at `min_periods`, not `2 × min_periods` |
| `lambda_` had no effect without controls; with controls it penalized only the control residualization step | Single penalized joint solve on the full design, normalized so strength is invariant to window/EWMA/sample size | Ridge now actually shrinks; `lambda_` values are not comparable to v0.2.x |
| HAC SEs built from historical endpoints' own residuals | HAC computed from the *same* current-window fit as the reported beta | SEs change; some previously-finite SEs may now be NaN with a warning instead of an inaccurate number |
| `orthogonalize_factors` / `orthogonalize_controls` on `fit()` | Removed | Apply orthogonalization to your inputs before calling `fit()`, or use `mode="joint"` |
| No `mode` parameter; multi-factor was implicitly batched | `mode="batched"` (default, unchanged behaviour) or `mode="joint"` | No code change required; consider `mode="joint"` if your factors are correlated |
| `get_control_beta` omitted the named factor from the residualization set | Control beta comes from the joint fit that includes the named factor | Values change; batched-mode control betas now correctly vary by factor |
| `get_factor_mimicking_returns()` / `get_all_factor_mimicking_returns()` | Removed (F13) | Renamed a time-series rolling beta; not a cross-sectional estimator |
| No input validation on `window`, `min_periods`, `lambda_`, etc. | Invalid constructor arguments raise `ValueError` at construction | Code passing invalid values now fails fast instead of producing silent NaNs |
| Factors, controls, and targets aligned positionally or by label depending on path | Index must be unique, monotonically increasing, and identical; `ValueError` at fit time otherwise | Code passing permuted, duplicate, or mismatched indexes now raises |

## v0.3.x to v0.4.0

v0.4.0 replaces the `dtype` parameter with a three-mode `precision` policy:

| Old | New | Effect |
|---|---|---|
| `dtype="float64"` | `precision="double"` | float64 storage, float64 compute (new default) |
| `dtype="float32"` | `precision="mixed"` | float32 storage, float64 compute |
| *(new)* | `precision="single"` | float32 storage, float32 compute |

`cond_warn_threshold` is now `None` by default (auto-selected: `1e10` for
float64 compute, `1e5` for float32 compute).
