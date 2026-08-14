# rOLS statistical specification

This document defines the estimators that rOLS must implement from version
0.3.0 onward. The scalar oracle in `tests/oracle.py` is the executable reference
for this specification. Optimized implementations are valid only when
differential tests show that they reproduce the oracle.

The central rule is simple: every quantity reported for one endpoint and one
target must come from one fit on one complete-case sample.

## Notation

At observation `s`:

- `y_{i,s}` is target `i`.
- `f_s` is the vector of factors.
- `c_s` is the vector of controls.
- `x_s` is the design row formed from the regressors used by the selected
  estimation mode.
- `W_t` is the observation window ending at endpoint `t`.
- `S_{i,t}` is the complete-case subset used for target `i` at endpoint `t`.

Quantities indexed by `t` are estimated at endpoint `t`. Subscripts `s,t` mean
that an observation at `s` is evaluated using the fit estimated at `t`.

## 1. Base rolling regression

For target `i` and endpoint `t`, rOLS estimates

$$
y_{i,s}
= \alpha_{i,t}
+ f_s^\top \beta_{i,t}
+ c_s^\top \gamma_{i,t}
+ \varepsilon_{i,s,t},
\qquad s \in S_{i,t} \subseteq W_t.
$$

The default is `fit_intercept=True`. The design then contains an explicit
intercept column. Centering is not used as a substitute for that column.

With `fit_intercept=False`, rOLS estimates a through-origin model. Every
dependent quantity must then use uncentred moments, including coefficients,
fitted values, residuals, total sum of squares, R-squared, and HAC inference.

The estimand is the coefficient vector of the selected model on the current
window's complete-case population.

> **v0.2.1 deviates:** factor betas and R-squared use centred covariance and
> variance, while control residualization and HAC use through-origin systems.
> The resulting coefficients, residuals, and inference do not describe one
> regression model.

## 2. Window semantics

For a rolling model with window length `L`,

$$
W_t = \{\max(0, t-L+1), \ldots, t\}.
$$

For an expanding model,

$$
W_t = \{0, \ldots, t\}.
$$

Results are stamped at endpoint `t`. The parameter `min_periods` is the minimum
number of rows in `S_{i,t}`, not the minimum number of calendar positions in
`W_t`. A result is emitted if and only if

$$
|S_{i,t}| \geq \texttt{min_periods}.
$$

All quantities reported at `t` use only observations in `W_t` and the same fit
estimated for `t`. Historical endpoint estimates are never reused as if they
were observations in the current fit.

> **v0.2.1 deviates:** with controls, it first constructs a series of endpoint
> residuals from historical windows and then applies another rolling regression
> to that series. With `window=252` and `min_periods=252`, the first beta appears
> at the 503rd observation instead of the 252nd.

## 3. Controls and Frisch-Waugh-Lovell

The direct current-window joint solve is the authoritative implementation. For
ordinary least squares, the factor coefficient may also be described through
the Frisch-Waugh-Lovell (FWL) theorem.

Let `N_t` contain the nuisance regressors in the current complete-case window.
It contains the controls and, when `fit_intercept=True`, the intercept. Define

$$
M_{N,t}
= I - N_t(N_t^\top N_t)^{-1}N_t^\top,
$$

$$
\widetilde F_t = M_{N,t}F_t,
\qquad
\widetilde y_{i,t} = M_{N,t}y_{i,t},
$$

and

$$
\widehat\beta_{i,t}
= (\widetilde F_t^\top\widetilde F_t)^{-1}
  \widetilde F_t^\top\widetilde y_{i,t}.
$$

Every row in the current window is residualized by the same projection
`M_{N,t}`. A series whose row `s` was residualized using a projection estimated
at `s` is a different estimator and is not permitted.

No FWL fast path is part of the version 0.3.0 remediation. A later optimization
may add one only if it uses the current-window projection and matches the direct
joint solve in differential tests. FWL is not used for penalized regression.

> **v0.2.1 deviates:** each historical residual uses its own endpoint's control
> projection. Re-rolling those residuals is not current-window FWL and causes
> the doubled warm-up.

## 4. Multiple-factor modes

rOLS supports two explicit modes.

### Batched mode

`mode="batched"` is the default. For each factor `j`, rOLS estimates a separate
model:

$$
y_{i,s}
= \alpha_{i,j,t}
+ c_s^\top\gamma_{i,j,t}
+ f_{j,s}\beta_{i,j,t}
+ \varepsilon_{i,j,s,t}.
$$

The coefficient `beta_{i,j,t}` is conditional on the controls but not on the
other factors. This mode is suitable for screening or estimating separate
factor exposures. It is not a multivariate factor model.

### Joint mode

`mode="joint"` estimates all factors in one model:

$$
y_{i,s}
= \alpha_{i,t}
+ c_s^\top\gamma_{i,t}
+ \sum_{j=1}^{K} f_{j,s}\beta_{i,j,t}
+ \varepsilon_{i,s,t}.
$$

Each factor coefficient is conditional on the controls and every other factor.
The two modes coincide for a single factor and, under ordinary least squares,
for factors that are mutually orthogonal on the estimation sample.

The complete-case set contains the regressors used by the selected model. In
batched mode, this means the controls and factor `j`. In joint mode, it means
the controls and all factors.

> **v0.2.1 deviates:** it implements batched independent regressions but
> describes the package as multi-factor without naming that distinction. It
> does not provide an explicit joint mode.

## 5. Ridge regression

Ridge is a single penalized solve on the joint design of the selected model. It
is never implemented by penalized residualization followed by an unpenalized
factor regression.

For one target and endpoint, let non-negative weights over `S_{i,t}` satisfy

$$
\sum_{s \in S_{i,t}} w_{i,s,t} = 1.
$$

The estimator minimizes the normalized objective

$$
\widehat\theta_{i,t}
= \arg\min_\theta
\left[
\sum_{s \in S_{i,t}}
w_{i,s,t}(y_{i,s}-x_s^\top\theta)^2
+ \theta^\top P\theta
\right].
$$

The penalty matrix is diagonal in the standardized solve coordinates. Its
entries follow these rules:

- The intercept is never penalized.
- Factors are penalized with `lambda_` when `lambda_ > 0`.
- Controls are penalized with `lambda_` by default.
- `penalize_controls=False` sets the control penalty entries to zero.

When `fit_intercept=True`, penalized regressors are centered and scaled to unit
weighted variance within `S_{i,t}`. When `fit_intercept=False`, they are scaled
by their weighted root mean square without centering, which preserves the
through-origin estimand. Coefficients are rescaled to the original regressor
units before they are returned.

Normalized loss makes `lambda_` comparable across window lengths, weighting
schemes, and complete-case sample sizes. Standardization makes it comparable
across regressor units. The effective sample size does not rescale the penalty.

At `lambda_=0`, the result must equal the corresponding ordinary least-squares
fit to numerical tolerance.

> **v0.2.1 deviates:** `lambda_` has no effect when controls are absent. With
> controls, it penalizes the control residualizations but not the final factor
> coefficient, which is neither joint Ridge nor FWL. It also applies the same
> bare penalty to unnormalized unweighted moments and normalized EWMA moments.

## 6. Missing data

Missing data follow one complete-case rule within each window and target. Define

$$
S_{i,t}
= \{s \in W_t:
y_{i,s}\text{ and every regressor in the selected model are finite}\}.
$$

The intercept is finite by construction. Every coefficient, weighted mean,
fitted value, residual, sum of squares, R-squared quantity, degree of freedom,
and inference quantity uses `S_{i,t}` and no other sample.

Rows with missing targets or regressors are dropped. Missing values are never
filled or converted to zero. Missingness in target `i` does not change the
sample or result for target `j`.

> **v0.2.1 deviates:** the high-level beta numerator uses the factor-target
> paired sample, while the denominator may use additional factor observations.
> Its documentation also says regressor NaNs invalidate a whole window while
> some implementation paths drop only the affected rows.

## 7. EWMA weighting

For EWMA half-life `h`, define

$$
\alpha = 1 - 2^{-1/h},
\qquad
a_{s,t} = (1-\alpha)^{t-s}.
$$

After applying the complete-case mask, normalize the surviving weights:

$$
w_{i,s,t}
= \frac{a_{s,t}}
       {\sum_{r \in S_{i,t}} a_{r,t}},
\qquad s \in S_{i,t}.
$$

Equal weighting uses `w_{i,s,t}=1/|S_{i,t}|`. Weighted and unweighted paths
therefore use the same normalized-moment convention. Renormalization after a
row drop preserves the meaning of `lambda_` under missingness.

The effective sample size is

$$
n_{\mathrm{eff},i,t}
= \frac{\left(\sum_s w_{i,s,t}\right)^2}
       {\sum_s w_{i,s,t}^2}
= \frac{1}{\sum_s w_{i,s,t}^2}.
$$

`n_eff` enters degrees-of-freedom quantities. It does not multiply or divide
the Ridge penalty.

> **v0.2.1 deviates:** EWMA moments sum to one while unweighted moments are raw
> sums, so the same `lambda_` has a different effective strength. Adjusted
> R-squared also uses integer row counts rather than effective sample size.

## 8. Window-wise orthogonalization

Orthogonalization is sequential and local to the current window. Within a
factor or control group, column `j` is projected on the already
orthogonalized columns `1,...,j-1` using the basis estimated for endpoint `t`.
When `fit_intercept=True`, the intercept belongs to the orthogonalization basis.

For orthogonalized columns `q_{j,t}` on the current estimation rows,

$$
q_{k,t}^\top W_t q_{j,t} = 0,
\qquad k < j.
$$

For equal weighting, this reduces to the ordinary inner product. Column order
sets priority. Insufficient data or a singular basis produces NaN; rOLS never
substitutes raw values for failed orthogonalized values.

When Ridge and orthogonalization are combined, standardization occurs after
orthogonalization. Returned coefficients are rescaled to the original units of
the orthogonalized regressors.

The basis changes with `t`. A change in an orthogonalized beta can therefore
reflect a change in factor correlation, a change in target sensitivity, or
both. rOLS does not currently provide a static full-sample basis.

> **v0.2.1 deviates:** it retains one endpoint residual from each historical
> projection and uses the resulting mixed-basis series in later windows. It
> also fills failed orthogonalized values with raw observations, so the output
> is partly transformed and partly raw.

## 9. R-squared and degrees of freedom

### Full-model R-squared

For the selected model, define

$$
\operatorname{SSR}_{i,t}
= \sum_{s \in S_{i,t}} w_{i,s,t}\widehat\varepsilon_{i,s,t}^2.
$$

With an intercept,

$$
\operatorname{SST}_{i,t}
= \sum_{s \in S_{i,t}} w_{i,s,t}
  (y_{i,s}-\overline y_{i,t})^2.
$$

Without an intercept,

$$
\operatorname{SST}_{i,t}
= \sum_{s \in S_{i,t}} w_{i,s,t}y_{i,s}^2.
$$

The full-model statistic is

$$
R^2_{\mathrm{full},i,t}
= 1 - \frac{\operatorname{SSR}_{i,t}}
           {\operatorname{SST}_{i,t}}.
$$

`get_r2(factor)` returns this full-model statistic for the model containing
`factor`. In batched mode, that is the factor-specific model. In joint mode,
the full-model value is shared across factors.

### Partial R-squared

For factor `j`, fit a reduced model that removes `j` and retains the remaining
regressors. On the same complete-case sample as the full model, define

$$
R^2_{\mathrm{partial},i,j,t}
= \frac{\operatorname{SSR}_{-j,i,t}
        -\operatorname{SSR}_{\mathrm{full},i,t}}
       {\operatorname{SSR}_{-j,i,t}}.
$$

`get_partial_r2(factor)` returns this incremental statistic.

### Adjusted R-squared

Let `p` be the number of estimated slope coefficients. With an intercept,

$$
\overline R^2
= 1-(1-R^2)
  \frac{n_{\mathrm{eff}}-1}{n_{\mathrm{eff}}-p-1}.
$$

Without an intercept, the uncentred adjustment is

$$
\overline R^2
= 1-(1-R^2)
  \frac{n_{\mathrm{eff}}}{n_{\mathrm{eff}}-p}.
$$

The adjusted statistic is NaN when its residual degrees of freedom are not
positive. Ridge uses the same reported convention; it does not claim an
effective-degrees-of-freedom adjustment based on the smoothing matrix.

> **v0.2.1 deviates:** `get_r2` reports a residual-on-residual partial
> R-squared as if it were full-model R-squared. Its adjusted formula hardcodes
> one slope and uses integer counts under EWMA.

## 10. HAC inference

HAC inference uses the same current-window fit, complete-case sample, design,
weights, and coefficient convention as the reported estimate. Let `z_s` be the
exact solve-coordinate design row, including the intercept when enabled and any
orthogonalization or standardization. Let

$$
\widehat u_{i,s,t}
= y_{i,s}-z_s^\top\widehat\theta_{i,t}
$$

be the residual from the fit estimated at endpoint `t`. Define the weighted
score

$$
\psi_{i,s,t}
= w_{i,s,t} z_s\widehat u_{i,s,t}.
$$

For `L = hac_lags`, the Bartlett weight at lag `l` is

$$
b_l = 1-\frac{l}{L+1}.
$$

The meat of the sandwich is

$$
\widehat S_{i,t}
= \sum_s \psi_{i,s,t}\psi_{i,s,t}^\top
+ \sum_{l=1}^{L} b_l
  \sum_s
  \left(
  \psi_{i,s,t}\psi_{i,s-l,t}^\top
  +\psi_{i,s-l,t}\psi_{i,s,t}^\top
  \right),
$$

where each lag sum contains only pairs inside the ordered complete-case sample.
Let

$$
A_{i,t}=Z_t^\top W_{i,t}Z_t+P.
$$

The solve-coordinate covariance is

$$
\widehat{\operatorname{Var}}(\widehat\theta_{i,t})
= \frac{n_{\mathrm{eff},i,t}}
       {n_{\mathrm{eff},i,t}-k}
  A_{i,t}^{-1}\widehat S_{i,t}A_{i,t}^{-1},
$$

where `k` is the number of estimated coefficients, including the intercept when
present. The statistic is undefined when `n_eff <= k`. Covariances are mapped
back to original coefficient units after a standardized Ridge solve.

This weighted sandwich is the normative HAC estimator. Until it is implemented,
`get_se()` must raise when EWMA is active. An equal-weight approximation may be
exposed only under a separate explicit name. It must never be returned silently
by `get_se()`.

The implementation returns NaN, with one aggregated warning, when the bread is
singular or when a negative estimated variance would otherwise produce a zero
standard error and an infinite t-statistic.

> **v0.2.1 deviates:** its HAC window contains endpoint residuals produced by
> different historical fits, and its bread does not match the centred beta
> model. Under EWMA it silently combines weighted coefficients with
> equal-weight inference.

## 11. Signals

Let `f_used` denote the factor column that appears in the authoritative joint
design, expressed in returned coefficient units. Controls alone do not transform
this factor. Without factor orthogonalization,

$$
f^{\mathrm{used}}_{j,s}=f^{\mathrm{raw}}_{j,s}.
$$

With factor orthogonalization, `f_used` is the current-window orthogonalized
factor. Internal Ridge standardization does not change the returned signal
because coefficients are rescaled before use.

The default signal accessor returns the fitted factor term:

$$
\operatorname{signal}_{i,j,t}
= \widehat\beta_{i,j,t}f^{\mathrm{used}}_{j,t}.
$$

With `lag_signal=True`, it returns

$$
\operatorname{signal}_{i,j,t}
= \widehat\beta_{i,j,t-1}f^{\mathrm{used}}_{j,t}.
$$

`get_raw_exposure_signal(factor)` returns the same coefficient multiplied by
the untransformed factor. The two accessors coincide when factor
orthogonalization is disabled, including when controls are present. When
orthogonalization is enabled, their difference is

$$
\widehat\beta_{i,j,t}
\left(f^{\mathrm{raw}}_{j,t}-f^{\mathrm{used}}_{j,t}\right).
$$

> **v0.2.1 deviates:** its beta is computed from a control-residualized and
> possibly orthogonalized factor series, while its signal always multiplies the
> raw factor. Under the version 0.3.0 direct joint solve, controls no longer
> create a separate factor representation; only explicit factor
> orthogonalization does.

## 12. Index and alignment contract

Factors, controls, and targets must have indexes that are:

1. identical in length, labels, order, and index type;
2. unique; and
3. monotonically increasing.

Violations raise `ValueError` before any array conversion or estimation. rOLS
does not sort, deduplicate, reindex, join, or drop labels implicitly. Users must
align inputs before calling the estimator.

> **v0.2.1 deviates:** low-level NumPy operations pair rows by position while
> later pandas operations align by label. Same-length permuted inputs can
> therefore combine two different row pairings without an error.

## 13. Out of scope

rOLS is a rolling or expanding time-series regression library. It does not
provide:

- cross-sectional factor-return or Fama-MacBeth estimation, which requires an
  explicit date by asset by factor data model;
- factor-mimicking portfolio construction;
- panel estimators with entity or time effects; or
- implicit data alignment, resampling, imputation, or calendar conversion.

A future cross-sectional estimator requires its own data model, specification,
oracle, and tests. Renaming a time-series beta does not identify a
cross-sectional factor return.

> **v0.2.1 deviates:** `get_factor_mimicking_returns()` and its README example
> label a single-target time-series beta as a cross-sectional factor return.
> No cross-sectional estimation occurs, and the documented `window=1` example
> is degenerate under sample covariance.
