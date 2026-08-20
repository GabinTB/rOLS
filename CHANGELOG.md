# Changelog

All notable changes to rOLS are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-08-20

### Breaking Changes
- **`dtype` parameter replaced by `precision`** (#25). The old `dtype` parameter
  (`"float32"` / `"float64"`) is replaced by a three-mode `precision` policy:
  `"double"` (float64/float64, the new default), `"mixed"` (float32 storage /
  float64 compute), and `"single"` (float32/float32). `cond_warn_threshold`
  default is now `None` (auto-selected: `1e10` for float64 compute, `1e5` for
  float32 compute). All internal array allocations respect the chosen precision.
  Migration: `dtype="float64"` → `precision="double"`,
  `dtype="float32"` → `precision="mixed"`.

---

## [0.3.2] — 2026-08-19

### Added
- `tests/test_fwl_conditioning.py`: 10 tests across 5 groups validating the FWL
  rank-condition heuristic under near-collinear and rank-deficient designs (#20).

### Fixed / Documented
- HAC SEs are time-series only — documented the panel cross-sectional dependence
  limitation in `get_se`/`get_tstat` docstrings, `docs/SPECIFICATION.md` §13,
  and `README.md` (#19).

---

## [0.3.1] — 2026-08-18

### Added
- Ridge effective degrees of freedom `df_eff = tr[G(G+P)⁻¹]` used in
  adjusted R² and residual-dof, replacing the raw parameter count (#15).
- `get_se` / `get_tstat` docstrings and `SPECIFICATION.md` §10 now define
  the Ridge inference estimand (β_λ vs β₀) and the nominal-coverage caveat (#16).
- `tests/test_hac_reference.py`: five structurally independent validations
  of the Ridge HAC sandwich (#17).

### Breaking Changes
- `mode` is now required when supplying more than one factor to `fit()`;
  omitting it raises `ValueError` with a message that names both options.
  Single-factor calls and calls with an explicit `mode` are unaffected (#18).

---

## [0.3.0] — 2026-08-18

**A correctness release.** An independent four-reviewer audit (13 August 2026) found that v0.2.1 did not estimate the model its
documentation described: the intercept convention was internally
inconsistent, `lambda_` had no effect without controls, rolling
Frisch-Waugh-Lovell and Gram-Schmidt used nested rather than window-wise
projections, and HAC standard errors did not correspond to the reported
coefficients. **Estimates from v0.2.1 and earlier should not be relied upon**
— re-run them, do not treat this as a routine version bump. Every fix below is
covered by a differential test against a from-scratch scalar oracle
(`tests/oracle.py`), not just against the library's own prior output. See
[`docs/SPECIFICATION.md`](docs/SPECIFICATION.md) for the estimator this
release implements and [README.md § Migration from v0.2.x](README.md#migration-from-v02x)
for a practical before/after table.

### Fixed
- **Intercept convention (F1).** Factor betas, R², and HAC inference now come
  from one consistent model per fit — an explicit intercept column when
  `fit_intercept=True` (the default), a through-origin model when `False` —
  instead of centred betas/R² alongside a through-origin residualization and
  HAC system.
- **Nested-rolling FWL (F2).** Controls are now partialled out using the
  *current window's own* projection, solved directly (or via a proven-equal
  FWL fast path), instead of re-rolling a second regression over
  already-rolled historical residuals. The double warm-up this caused is
  gone: with controls, the first estimate now appears at `min_periods`, not
  `2 × min_periods`.
- **HAC standard errors did not match the reported beta (F3, F20).** SEs are
  now computed from the same current-window fit as the beta they describe,
  using the full design (intercept and controls included in both bread and
  scores), the exact complete-case sample, and — under `ewma_halflife` — the
  same observation weights as the coefficient estimator. Previously HAC used
  equal weighting even when the beta was EWMA-weighted, and its residual
  window mixed residuals from different historical fits.
- **`lambda_` had no effect without controls, and only shrank a
  residualization step with controls (F4, F5).** Ridge is now a single
  penalized solve on the joint design of the selected model, normalized so
  `lambda_` has the same effective strength regardless of window length,
  EWMA half-life, or complete-case sample size.
- **Missing-target numerator/denominator mismatch (F6).** A beta's numerator
  and denominator (and every other reported quantity) now come from the exact
  same complete-case row set — previously the numerator used the
  factor-target paired sample while the denominator could include additional
  factor observations the target didn't have.
- **Silent input misalignment (F7).** Factors, controls, and targets must now
  have identical, unique, monotonically increasing indexes, checked with
  `ValueError` before any array conversion. Previously low-level NumPy paths
  paired rows positionally while pandas paths aligned by label, so
  same-length permuted inputs could silently combine two different row
  pairings.
- **EWMA renormalization changed the effective Ridge penalty (F8).** Weights
  are renormalized to sum to one on the complete-case sample after row drops,
  under both equal and EWMA weighting, so `lambda_`'s meaning is preserved
  under missingness.
- **Signal used a transformed factor while beta was estimated on a different
  one (F9).** With controls now residualized only through the joint/FWL
  solve (no separate orthogonalized factor representation), `get_signal`
  multiplies beta by the same factor values the beta was estimated on.
- **`get_control_beta` omitted the named factor from its own residualization
  set (F10).** A control's coefficient now comes from the joint fit that
  includes the requested factor. In batched mode this means a control's beta
  **can vary by which factor is named** when the two are correlated — the
  previous "factor-independent" claim was true only in the special case of
  mutual orthogonality, not in general.
- **Squared condition number, silent ill-conditioning (F15).** The solver no
  longer forms the normal equations and calls `np.linalg.solve` directly (which
  squares `cond(X)` and raises only on exact numerical singularity); it now
  factorizes the design (QR, augmented with `√penalty` for Ridge) and warns
  once, aggregated, when `cond(X'X)` exceeds `cond_warn_threshold` (new
  parameter, default `1e10`).
- **Adjusted R² degrees of freedom, and full vs partial R² conflation
  (F16).** Adjusted R² now uses the correct slope count including controls
  (previously hardcoded to one slope) and effective sample size under EWMA
  (previously an integer row count). `get_r2` and the new `get_partial_r2`
  are now two distinct, correctly defined quantities — see
  [README § R² variants](README.md#r²-variants).
- **Out-of-range index in the regressor-NaN fallback path (F17).** Loop
  bounds in `rolling_residualize`'s regressor-NaN and vectorized-target-NaN
  paths no longer go out of bounds when `T < window - 1`.
- **NaN contract mismatch between docs and code (F18).** The documented
  contract and the implementation now agree: a row is used iff the intercept
  design, every control, every factor in the selected model, and the target
  are simultaneously finite (complete-case within window); a result is
  emitted iff at least `min_periods` such rows survive.
- `_solve_batch` no longer lets `inf` from near-singular solves propagate into
  betas, signals, and R² — results are sanitized to NaN in place.
- HAC standard errors are isolated per target: a NaN in one target's residuals
  no longer forces NaN SEs for every target in that window.

### Breaking Changes
- **`mode` is now required for multi-factor calls.** Passing more than one
  factor to `fit()` without `mode` raises `ValueError`. Previously, `mode`
  defaulted to `"batched"` silently, meaning every existing multi-factor
  caller received marginal (not mutually controlled) betas without having
  chosen that estimand. The error message names both options and explains
  their difference. Single-factor calls are unaffected: `mode` remains
  optional there (the two modes coincide for one factor). Migration: add
  `mode="batched"` to preserve the old numbers exactly, or switch to
  `mode="joint"` for the mutually-controlled estimand.

### Changed
- **Multi-factor semantics made explicit (F14).** New `mode` parameter:
  `"batched"` fits one model per factor — marginal given controls, not
  mutually controlled; `"joint"` fits one model with every factor, mutually
  controlled. Previously the library ran batched regressions while describing
  itself as multi-factor without naming the distinction.
  `warn_correlated_factors=True` (default) now warns once when
  `mode="batched"` factors are correlated above `|ρ| > 0.3`.
- **Core performance rebuild — no numeric change.** Targets sharing an exact
  complete-case pattern within a window are now factorized once and solved
  as a block of right-hand sides, and the (now correctly window-wise) FWL
  residualization shares one controls-only projection and one GEMM across
  every factor. Every result is unchanged to `1e-10` against the pre-rebuild
  path; see [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for measured cost.
- **Lazy evaluation and bounded memory.** Residuals, R², partial R², and
  standard errors are now computed on demand rather than eagerly, with a
  bounded per-factor cache (new `cache_size` parameter, default `1`). New
  `iter_beta()` / `iter_se()` process every factor without retaining all of
  them at once.
- **Sparse estimation cadence.** New `estimate_every` parameter (integer step
  count or pandas offset alias) restricts solving to a coarser cadence while
  windows still contain every underlying observation; `get_*` accessors
  return the full index with NaN at skipped endpoints.
- **Input validation (F19).** `window`, `min_periods`, `lambda_`,
  `ewma_halflife`, `hac_lags`, and their cross-constraints (e.g.
  `min_periods > window` in rolling mode) are validated at construction with
  `ValueError` naming the offending value, instead of failing silently or
  deep in a NumPy call.

### Added
- `fit_intercept` (default `True`), `mode`, `warn_correlated_factors`,
  `penalize_controls`, `cond_warn_threshold`, `cache_size`, `estimate_every`
  parameters on `RollingOLS`.
- `get_intercept(factor)`, `get_partial_r2(factor)`, `get_dof(factor)`,
  `get_n_used(factor)` on `RollingOLSResult`.
- `iter_beta()` / `iter_se()` for bounded-memory iteration over all factors.
- `result.mode` — the mode a result was computed with.
- `RollingOLS.estimate_memory(targets, factors, controls)` — persistent and
  on-demand memory cost from input shapes, before fitting.
- `tests/oracle.py` — a deliberately slow, from-scratch scalar reference
  implementation; every optimized path is validated against it by a
  differential test, closing the "oracle was the library's own loop" gap
  (F22) that let the above defects ship undetected.
- `benchmarks/` — a wall-time and peak-memory harness with a captured v0.2.1
  speed baseline (speed reference only — v0.2.1's estimates remain
  statistically incorrect and are never a correctness reference).

### Removed
- **`get_factor_mimicking_returns(factor)` and
  `get_all_factor_mimicking_returns()` on `RollingOLSResult` (F13).** These
  accessors claimed to produce cross-sectional Fama-MacBeth factor returns
  but performed no cross-sectional estimation — they renamed the single
  column of an ordinary time-series rolling beta. Cross-sectional estimation
  needs its own date × asset × factor data model, specification, and oracle;
  it is out of scope for this library. See
  [README § Out of scope](README.md#out-of-scope).
- **Rolling Gram-Schmidt orthogonalization (F11, F12).**
  `orthogonalize_factors` / `orthogonalize_controls` on `fit()`/
  `fit_transform()`, and `get_raw_exposure_signal()`. Like the FWL defect
  above, orthogonalization was nested-rolling rather than window-wise, and
  interacted badly with a fixed Ridge penalty since it also changed factor
  scale at every endpoint. Apply orthogonalization to your inputs before
  calling `fit()` if you need it; `mode="joint"` is the standard way to
  estimate mutually-controlled factor effects without it.

### Documentation
- `README.md`, `docs/SPECIFICATION.md` (new), `docs/PERFORMANCE.md` (new),
  and docstrings across `rols/*.py` rewritten against the implementation as
  it now stands, including a v0.2.x migration table and an explicit
  out-of-scope section (cross-sectional estimation, factor-mimicking
  portfolios, orthogonalization, panel fixed effects).
- `examples/fama_french_factors.ipynb` updated to the current API
  (`fit_intercept`, `mode`, `estimate_every`, `get_r2`/`get_partial_r2`,
  removed orthogonalization parameters) and re-run end to end.

## [0.2.1] - 2026-06-14

> **Superseded.** An independent audit round (13 August 2026) found the
> estimator did not match its documentation: the intercept convention was
> internally inconsistent, `lambda_` had no effect without controls, rolling
> FWL and Gram-Schmidt used nested rather than window-wise projections, and
> HAC standard errors did not correspond to the reported coefficients.
> **Estimates from v0.2.1 and earlier should not be relied upon.** See

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

[0.4.0]: https://github.com/GabinTB/rOLS/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/GabinTB/rOLS/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/GabinTB/rOLS/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/GabinTB/rOLS/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/GabinTB/rOLS/compare/v0.1.2...v0.2.1
[0.1.2]: https://github.com/GabinTB/rOLS/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/GabinTB/rOLS/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/GabinTB/rOLS/releases/tag/v0.1.0
