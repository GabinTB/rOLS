"""
rOLS: rolling OLS/Ridge regression library
========

RollingOLS: sklearn/statsmodels-style rolling time-series OLS (or Ridge).

Design
------
- fit(factors, controls=None)   : Frisch-Waugh step — residualize factors
- transform(assets)             : project assets, compute betas/signals/R²
- fit_transform(...)            : convenience one-liner

Frisch-Waugh-Lovell partitioning keeps per-factor math univariate regardless
of how many controls are added, and is numerically equivalent to the full
joint regression.

Ridge regularization (lambda_ > 0) adds λI to X'X before solving — stabilizes
estimation when regressors are collinear, at the cost of shrinking betas toward
zero. Set lambda_=0.0 for standard OLS (default).

Rolling Gram-Schmidt orthogonalization can be applied independently to factors
and/or controls. Column order determines priority: first column is untouched,
subsequent columns are orthogonalized against all preceding ones. Use this
when regressors have a natural importance ordering (e.g. evergreen narratives
before transient ones) and you want each beta to represent incremental
explanatory power beyond higher-priority regressors.

HAC standard errors (Newey-West) are computed on demand via result.get_se(factor).
Set hac_lags on the constructor to enable this.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd

from .estimators import rolling_residualize, rolling_gram_schmidt, hac_se
from .results import RollingOLSResult


def _rolling_cov_series_df(
    s: pd.Series,
    df: pd.DataFrame,
    window: int,
    min_periods: int,
    expanding: bool,
) -> pd.DataFrame:
    base = df.expanding(min_periods=min_periods) if expanding else df.rolling(window, min_periods=min_periods)
    return base.cov(s)


def _rolling_var(
    x: pd.Series | pd.DataFrame,
    window: int,
    min_periods: int,
    expanding: bool,
) -> pd.Series | pd.DataFrame:
    base = x.expanding(min_periods=min_periods) if expanding else x.rolling(window, min_periods=min_periods)
    return base.var()


class RollingOLS:
    """
    Vectorized rolling (or expanding) time-series OLS / Ridge regression.

    Supports any number of assets, factors, and control variables.
    Designed as a generic rolling regression library — not specific to any domain.

    Parameters
    ----------
    window : int
        Rolling window length in observations.
    min_periods : int, optional
        Minimum observations to produce a result. Defaults to window.
    expanding : bool
        Use expanding window instead of rolling.
    lambda_ : float
        Ridge regularization strength. 0.0 = standard OLS (default).
        Adding a small value (e.g. 1e-4) stabilizes collinear regressors.
    adj_r2 : bool
        Compute adjusted R² instead of R².
    lag_signal : bool
        If True, signal = beta_{t-1} * factor_t (avoids look-ahead bias).
        If False, signal = beta_t * factor_t (contemporaneous).
    hac_lags : int, optional
        Number of lags for Newey-West HAC standard errors.
        If None (default), HAC is disabled and get_se() will raise.
        A common rule of thumb: floor(T^(1/3)) or floor(4*(T/100)^(2/9)).
    denom_tol : float
        Threshold below which rolling variance is treated as zero (NaN out).
    dtype : str
        Input dtype. 'float32' saves memory; 'float64' for higher precision.
    asset_chunk_size : int
        Number of assets processed per chunk during residualization.
        Lower values reduce peak memory at the cost of slightly more overhead.

    Examples
    --------
    Basic usage — no controls:

    >>> ols = RollingOLS(window=252)
    >>> result = ols.fit(df[["f1", "f2"]]).transform(df[["AAPL", "MSFT"]])
    >>> result.get_beta("f1")      # DataFrame (T x N_assets)
    >>> result.get_signal("f1")
    >>> result.get_r2("f1")

    With controls and Ridge:

    >>> ols = RollingOLS(window=252, lambda_=1e-4, hac_lags=5)
    >>> ols.fit(df[["f1", "f2"]], controls=df[["Mkt-RF", "SMB"]])
    >>> result = ols.transform(df[["AAPL", "MSFT"]])
    >>> result.get_se("f1")        # Newey-West SE
    >>> result.get_tstat("f1")

    With orthogonalization (factors ordered by importance):

    >>> ols.fit(
    ...     df[["evergreen_1", "evergreen_2", "transient_1"]],
    ...     controls=df[["Mkt-RF"]],
    ...     orthogonalize_factors=True,
    ...     orthogonalize_controls=False,
    ... )

    Long format output:

    >>> result.to_long("f1")       # date, asset, beta, signal, r2
    >>> result.to_long_all()       # + factor column
    >>> result.to_long("f1", include_se=True)   # + se, t_stat
    """

    def __init__(
        self,
        window: int = 252,
        min_periods: Optional[int] = None,
        expanding: bool = False,
        lambda_: float = 0.0,
        adj_r2: bool = False,
        lag_signal: bool = False,
        hac_lags: Optional[int] = None,
        denom_tol: float = 1e-12,
        dtype: str = "float32",
        asset_chunk_size: int = 100,
    ) -> None:
        self.window          = window
        self.min_periods     = min_periods if min_periods is not None else window
        self.expanding       = expanding
        self.lambda_         = lambda_
        self.adj_r2          = adj_r2
        self.lag_signal      = lag_signal
        self.hac_lags        = hac_lags
        self.denom_tol       = denom_tol
        self.dtype           = dtype
        self.asset_chunk_size = asset_chunk_size

        self._is_fitted       = False
        self._factor_cols:    List[str] = []
        self._control_cols:   List[str] = []
        self._factors_raw:    Optional[pd.DataFrame] = None  # original, for signal
        self._factor_resids:  Optional[pd.DataFrame] = None  # after FWL step 1
        self._controls_fitted: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        factors: pd.DataFrame,
        controls: Optional[pd.DataFrame] = None,
        orthogonalize_factors: bool = False,
        orthogonalize_controls: bool = False,
    ) -> "RollingOLS":
        """
        Fit the model on the regressors side (Frisch-Waugh step 1).

        Optionally orthogonalizes factors and/or controls via rolling
        Gram-Schmidt before residualization. Column order determines
        orthogonalization priority — first column is untouched, each
        subsequent column is orthogonalized against all preceding ones.

        Parameters
        ----------
        factors : pd.DataFrame
            Regressors of interest. Each column gets its own rolling beta.
            e.g. df[["narrative_1", "narrative_2"]]
        controls : pd.DataFrame, optional
            Always-in regressors to partial out (e.g. df[["Mkt-RF", "SMB"]]).
            If None, no partialling out — pure factor regression.
        orthogonalize_factors : bool
            Apply rolling Gram-Schmidt within the factors group.
            First factor is untouched; each subsequent factor is orthogonalized
            against all previous ones. Use when factors have an importance
            ordering and you want each beta to reflect incremental contribution.
        orthogonalize_controls : bool
            Apply rolling Gram-Schmidt within the controls group.
            Useful when controls are correlated (e.g. multiple style factors).

        Returns
        -------
        self
        """
        factors = factors.astype(self.dtype)
        self._factors_raw = factors  # kept for signal computation

        if orthogonalize_factors and factors.shape[1] > 1:
            factors = rolling_gram_schmidt(
                factors,
                window=self.window,
                min_periods=self.min_periods,
                expanding=self.expanding,
            ).astype(self.dtype)

        self._factor_cols = factors.columns.tolist()

        if controls is not None:
            controls = controls.astype(self.dtype)

            if orthogonalize_controls and controls.shape[1] > 1:
                controls = rolling_gram_schmidt(
                    controls,
                    window=self.window,
                    min_periods=self.min_periods,
                    expanding=self.expanding,
                ).astype(self.dtype)

            self._control_cols    = controls.columns.tolist()
            self._controls_fitted = controls

            self._factor_resids = rolling_residualize(
                y=factors,
                X=controls,
                window=self.window,
                min_periods=self.min_periods,
                expanding=self.expanding,
                ridge_lambda=self.lambda_,
            )
        else:
            self._control_cols    = []
            self._controls_fitted = None
            self._factor_resids   = factors

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(
        self,
        assets: pd.DataFrame,
        # return_control_betas: bool = False,
    ) -> RollingOLSResult:
        """
        Project assets onto fitted factor structure.

        Computes rolling betas, signals, R², and optionally residuals for HAC.

        Parameters
        ----------
        assets : pd.DataFrame
            Target returns. e.g. df[["AAPL", "MSFT", "GOOG"]]

        Returns
        -------
        RollingOLSResult
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        asset_cols = assets.columns.tolist()
        assets     = assets.astype(self.dtype)

        # Frisch-Waugh step 2: residualize assets against controls (chunked)
        if self._controls_fitted is not None:
            chunks = [
                asset_cols[i: i + self.asset_chunk_size]
                for i in range(0, len(asset_cols), self.asset_chunk_size)
            ]
            asset_resids = pd.concat(
                [
                    rolling_residualize(
                        y=assets[chunk],
                        X=self._controls_fitted,
                        window=self.window,
                        min_periods=self.min_periods,
                        expanding=self.expanding,
                        ridge_lambda=self.lambda_,
                    )
                    for chunk in chunks
                ],
                axis=1,
            )
        else:
            asset_resids = assets

        result = RollingOLSResult(
            factor_cols=self._factor_cols,
            asset_cols=asset_cols,
            index=assets.index,
            lag_signal=self.lag_signal,
            window=self.window,
            min_periods=self.min_periods,
            expanding=self.expanding,
            hac_lags=self.hac_lags,
        )

        # Precompute asset residual variance once — shared across all factors
        var_y = _rolling_var(asset_resids, self.window, self.min_periods, self.expanding)

        for fac in self._factor_cols:
            f_resid = self._factor_resids[fac]

            cov_af    = _rolling_cov_series_df(f_resid, asset_resids, self.window, self.min_periods, self.expanding)
            var_f     = _rolling_var(f_resid, self.window, self.min_periods, self.expanding)
            var_f_safe = var_f.where(var_f.abs() > self.denom_tol)

            # Beta
            beta = cov_af.div(var_f_safe, axis=0)

            # Signal — always uses raw (non-orthogonalized) factor values
            f_orig = self._factors_raw[fac]
            signal = beta.shift(1).mul(f_orig, axis=0) if self.lag_signal else beta.mul(f_orig, axis=0)

            # R²
            r2 = (cov_af ** 2).div(var_f_safe.values[:, None] * var_y, axis=0)
            if self.adj_r2:
                n_obs = (
                    asset_resids.expanding(min_periods=self.min_periods).count()
                    if self.expanding
                    else asset_resids.rolling(self.window, min_periods=self.min_periods).count()
                )
                r2 = 1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - 2)

            # Residuals — needed for HAC SE on demand
            reg_resids = asset_resids - beta.mul(f_resid, axis=0)

            result._betas[fac]     = beta
            result._signals[fac]   = signal
            result._r2[fac]        = r2
            result._residuals[fac] = reg_resids
            result._factor_values[fac] = f_resid

            # if return_control_betas and self._controls_fitted is not None:
            #     result._control_betas[fac] = self._compute_control_betas(assets)

        return result

    # ------------------------------------------------------------------
    # fit_transform
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        factors: pd.DataFrame,
        assets: pd.DataFrame,
        controls: Optional[pd.DataFrame] = None,
        orthogonalize_factors: bool = False,
        orthogonalize_controls: bool = False,
        # return_control_betas: bool = False,
    ) -> RollingOLSResult:
        """
        Convenience: fit() then transform() in one call.

        Parameters mirror fit() and transform() — see their docstrings.
        """
        return (
            self
            .fit(factors, controls, orthogonalize_factors, orthogonalize_controls)
            .transform(assets)#, return_control_betas)
        )

    # def _compute_control_betas(self, assets: pd.DataFrame) -> dict:
    #     """Univariate rolling beta of each control against assets (marginal, not joint)."""
    #     out = {}
    #     for ctrl in self._control_cols:
    #         c         = self._controls_fitted[ctrl]
    #         cov_ac    = _rolling_cov_series_df(c, assets, self.window, self.min_periods, self.expanding)
    #         var_c     = _rolling_var(c, self.window, self.min_periods, self.expanding)
    #         var_c_safe = var_c.where(var_c.abs() > self.denom_tol)
    #         out[ctrl]  = cov_ac.div(var_c_safe, axis=0)
    #     return out
