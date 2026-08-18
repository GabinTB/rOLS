# Changelog

All notable changes to rOLS are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ewma_halflife` parameter on `RollingOLS`: exponentially weight observations
  within each window so recent data carries more weight. Flows through betas,
  R², HAC inference, and the Frisch-Waugh residualization (weighted least
  squares per window). Not compatible with `expanding=True`. (#1)
- `get_factor_adjusted_returns()` on `RollingOLSResult`: exposes the FWL step 2
  output (asset returns with only the controls partialled out), distinct from
  `get_residuals(factor)` which also removes the factor (step 3). (#3)
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
- HAC standard errors are computed lazily and streamed per endpoint so
  current-window residuals are not retained across time or factors.

### Removed
- `get_factor_mimicking_returns(factor)` and `get_all_factor_mimicking_returns()`
  on `RollingOLSResult`. These accessors claimed to produce cross-sectional
  Fama-MacBeth factor returns but performed no cross-sectional estimation: they
  renamed the single column of an ordinary time-series rolling beta. The
  documented `window=1` usage is degenerate and the 2-D rolling frame cannot
  represent the date × asset × factor structure required. Cross-sectional
  estimation is out of scope and tracked separately. (F13)
- Rolling Gram-Schmidt orthogonalization: `orthogonalize_factors` and
  `orthogonalize_controls` on `fit()`/`fit_transform()`, and
  `get_raw_exposure_signal()`. Orthogonalization is preprocessing, produces a
  time-varying basis under a rolling window, and is subsumed by `mode="joint"`.
  Apply it to the factors before calling `fit()` if needed. (F11, F12)

### Fixed
- HAC standard errors now use residuals from the beta's own current-window fit,
  the full design and intercept in both bread and scores, the exact complete-case
  sample, and the estimator's EWMA weights. Non-positive variances and invalid
  bread produce NaN with one aggregated warning rather than infinite t-stats.
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
  betas, and the precision note.

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
