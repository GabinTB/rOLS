# Changelog

All notable changes to rOLS are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ewma_halflife` parameter on `RollingOLS`: exponentially weight observations
  within each window so recent data carries more weight. Flows through betas,
  R², and the Frisch-Waugh residualization (weighted least squares per window).
  Not compatible with `expanding=True`. HAC SEs remain equal-weighted. (#1)
- `get_factor_adjusted_returns()` on `RollingOLSResult`: exposes the FWL step 2
  output (asset returns with only the controls partialled out), distinct from
  `get_residuals(factor)` which also removes the factor (step 3). (#3)
- `get_factor_mimicking_returns(factor)` and `get_all_factor_mimicking_returns()`
  on `RollingOLSResult`: named accessors for the cross-sectional use case where
  the rolling beta is the factor mimicking return. (#5)
- `get_control_beta(factor, control)` reinstated and reimplemented correctly via
  Frisch-Waugh-Lovell — returns the joint (not univariate marginal) control beta.
  Enabled with `return_control_betas=True` on `transform()`/`fit_transform()`.
  Control betas are factor-independent and computed once, shared across factors. (#9)
- `warn_singular` parameter on `RollingOLS` (default `True`): emit a single
  aggregated `RuntimeWarning` when rolling windows are singular, with affected
  estimates set to NaN. Set `False` to suppress. (#12)
- Test suite covering estimators, model, results, and integration. (#11)

### Changed
- NaN-robust residualization rewritten with an intermediate vectorized path:
  when X is clean and NaNs are present only in the targets (the typical large
  asset-panel case), residualization is vectorized over time with an O(N) loop
  over assets instead of the O(T*N) per-column loop. Produces identical results
  with a large speedup at scale. (#2)
- HAC standard error computation vectorized over the time axis via stride tricks;
  the Python loop is now O(n_lags) instead of O(T). Expanding-window path still
  loops. Results identical to the previous implementation. (#4)

### Fixed
- `_solve_batch` no longer lets `inf` from near-singular solves propagate into
  betas, signals, and R². Results are written in place and sanitized to NaN. (#6)
- Adjusted R² no longer divides by zero when `n_obs <= 2`; the denominator is
  guarded so those windows yield NaN instead of `inf`. (#7)
- HAC standard errors are now isolated per asset: a NaN in one asset's residuals
  no longer forces NaN SEs for all assets in that window. A NaN in the factor
  still invalidates the whole window. (#8)
- NaN handling in rolling residualization no longer contaminates the whole panel
  when a single asset has missing values; NaNs in one target column do not affect
  other columns. NaNs in regressors still invalidate the window. Earlier
  "all-or-nothing" behaviour produced all-NaN results on sparse panels.

### Documentation
- Clarified that `dtype` controls pandas DataFrame storage only; internal matrix
  operations always run in float64 for numerical stability. (#10)
- README updated with EWMA, `warn_singular`, factor-adjusted returns, control
  betas, factor mimicking returns, and the precision note.

## [0.1.2] - 2026-04-14
### Added
- Python 3.10–3.14 classifiers and project metadata.

## [0.1.1] - 2026-04-02
### Fixed
- Packaging metadata corrections.

## [0.1.0] - 2026-04-02
### Added
- Initial release: vectorized rolling/expanding OLS and Ridge regression for
  multi-target, multi-factor time series. Frisch-Waugh-Lovell control
  partialling, rolling Gram-Schmidt orthogonalization, Newey-West HAC standard
  errors, lagged signals, and long-format output.

[Unreleased]: https://github.com/GabinTB/rOLS/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/GabinTB/rOLS/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/GabinTB/rOLS/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/GabinTB/rOLS/releases/tag/v0.1.0
