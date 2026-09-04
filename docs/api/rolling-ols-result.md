# RollingOLSResult

The result container returned by `RollingOLS.transform()` and
`RollingOLS.fit_transform()`. All getters return DataFrames indexed by time
(rows) and target (columns).

::: rols.RollingOLSResult
    options:
      members:
        - get_beta
        - get_intercept
        - get_signal
        - get_r2
        - get_partial_r2
        - get_fitted_values
        - get_residuals
        - get_factor_adjusted_returns
        - get_control_beta
        - get_se
        - get_tstat
        - get_dof
        - get_n_used
        - iter_beta
        - iter_se
        - to_long
        - to_long_all
