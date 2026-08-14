"""
rOLS: rolling OLS/Ridge regression library
========

RollingOLS: sklearn/statsmodels-style rolling time-series OLS (or Ridge).

Design
------
- fit(factors, controls=None)   : validate and store the regressor groups
- transform(assets)             : solve each factor model on every current window
- fit_transform(...)            : convenience one-liner

Each factor is fitted with the controls in one direct joint solve. Every
reported quantity for an endpoint and target comes from that same fit.

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

import numpy as np
import pandas as pd

from .estimators import (
    JointFitResult,
    rolling_gram_schmidt,
    rolling_joint_solve,
    rolling_residualize,
)
from .results import RollingOLSResult

_INDEX_CONTRACT_GUIDANCE = (
    "rOLS requires an identical, unique, monotonic index across factors, controls, "
    "and targets. Align inputs before calling, e.g. via "
    "df = pd.concat([factors, controls, assets], axis=1).dropna()."
)


def _validate_index(
    *frames: pd.DataFrame | pd.Series,
    names: list[str],
) -> None:
    """Enforce identical, unique, monotonically increasing input indexes."""
    if len(frames) != len(names):
        raise ValueError("frames and names must have the same length")
    if not frames:
        return

    for frame, name in zip(frames, names, strict=True):
        index = frame.index
        if not index.is_unique:
            duplicate_labels = index[index.duplicated()].unique().tolist()
            raise ValueError(
                f"index for '{name}' contains duplicate labels {duplicate_labels!r}. "
                f"{_INDEX_CONTRACT_GUIDANCE}"
            )

    validation_names = " and ".join(f"'{name}'" for name in names)
    for frame, name in zip(frames, names, strict=True):
        if not frame.index.is_monotonic_increasing:
            raise ValueError(
                f"index for '{name}' is not monotonically increasing while validating "
                f"{validation_names}. {_INDEX_CONTRACT_GUIDANCE}"
            )

    reference_index = frames[0].index
    reference_name = names[0]
    for frame, name in zip(frames[1:], names[1:], strict=True):
        index = frame.index
        if type(reference_index) is type(index) and reference_index.equals(index):
            continue

        common_length = min(len(reference_index), len(index))
        mismatch_position = next(
            (
                position
                for position in range(common_length)
                if not pd.Index([reference_index[position]], tupleize_cols=False).equals(
                    pd.Index([index[position]], tupleize_cols=False)
                )
            ),
            None,
        )
        if mismatch_position is not None:
            divergence = (
                f"differ from position {mismatch_position} "
                f"({reference_name}[{mismatch_position}]={reference_index[mismatch_position]!r}, "
                f"{name}[{mismatch_position}]={index[mismatch_position]!r})"
            )
        elif len(reference_index) != len(index):
            divergence = f"first differ at position {common_length}, where one index has ended"
        else:
            divergence = (
                "contain the same labels but use different index types "
                f"({type(reference_index).__name__} and {type(index).__name__})"
            )
        raise ValueError(
            f"index mismatch between '{reference_name}' and '{name}': lengths "
            f"{len(reference_index)} and {len(index)}; {divergence}. "
            f"{_INDEX_CONTRACT_GUIDANCE}"
        )


def _rolling_cov_series_df(
    s: pd.Series,
    df: pd.DataFrame,
    window: int,
    min_periods: int,
    expanding: bool,
) -> pd.DataFrame:
    base = (
        df.expanding(min_periods=min_periods)
        if expanding
        else df.rolling(window, min_periods=min_periods)
    )
    return base.cov(s)


def _rolling_var(
    x: pd.Series | pd.DataFrame,
    window: int,
    min_periods: int,
    expanding: bool,
) -> pd.Series | pd.DataFrame:
    base = (
        x.expanding(min_periods=min_periods)
        if expanding
        else x.rolling(window, min_periods=min_periods)
    )
    return base.var()


def _ewma_weights(halflife: int, window: int) -> np.ndarray:
    """
    Exponential decay weights over a window, oldest-to-newest, summing to 1.

    The weight of an observation `lag` steps before the most recent is
    proportional to (1 - alpha)**lag where alpha = 1 - 2**(-1/halflife), so an
    observation `halflife` steps in the past carries half the weight of the
    newest one. Index 0 is the oldest observation, index window-1 the newest.
    """
    alpha = 1 - np.exp(-np.log(2) / halflife)
    w = (1 - alpha) ** np.arange(window - 1, -1, -1)
    return w / w.sum()


def _ewma_cov_series_df(
    f: pd.Series,
    assets: pd.DataFrame,
    weights: np.ndarray,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    """
    Rolling EWMA-weighted covariance of a Series ``f`` against each column of
    ``assets`` — the weighted analogue of ``_rolling_cov_series_df``.

    Uses biased weighted moments (no Bessel correction): with weights summing
    to 1, cov = sum_t w_t (f_t - f_bar)(a_t - a_bar). The normalization is
    shared with ``_ewma_var`` so beta = cov / var and R² = cov² / (var_f var_a)
    stay consistent. NaN rows (in ``f`` or an asset) are dropped per window and
    per asset, with the surviving weights renormalized to sum to 1.
    """
    f_np = f.to_numpy(dtype=np.float64)
    a_np = assets.to_numpy(dtype=np.float64)
    T, N = a_np.shape
    out = np.full((T, N), np.nan)

    for t in range(min_periods - 1, T):
        L = min(window, t + 1)
        sl = slice(t - L + 1, t + 1)
        w_t = weights[-L:]  # (L,)
        f_w = f_np[sl]  # (L,)
        a_w = a_np[sl]  # (L, N)

        valid = (~np.isnan(f_w))[:, None] & ~np.isnan(a_w)  # (L, N)
        wm = np.where(valid, w_t[:, None], 0.0)  # (L, N)
        wsum = wm.sum(axis=0)  # (N,)
        ok = (valid.sum(axis=0) >= min_periods) & (wsum > 0)
        if not ok.any():
            continue
        wn = wm / np.where(wsum > 0, wsum, 1.0)
        fbar = (wn * np.where(valid, f_w[:, None], 0.0)).sum(axis=0)
        abar = (wn * np.where(valid, a_w, 0.0)).sum(axis=0)
        fc = np.where(valid, f_w[:, None] - fbar, 0.0)
        ac = np.where(valid, a_w - abar, 0.0)
        cov = (wn * fc * ac).sum(axis=0)
        out[t, ok] = cov[ok]

    return pd.DataFrame(out, index=assets.index, columns=assets.columns)


def _ewma_var(
    x: pd.Series | pd.DataFrame,
    weights: np.ndarray,
    window: int,
    min_periods: int,
) -> pd.Series | pd.DataFrame:
    """
    Rolling EWMA-weighted variance — the weighted analogue of ``_rolling_var``.

    Biased weighted variance (weights sum to 1): var = sum_t w_t (x_t - x_bar)².
    Shares its normalization with ``_ewma_cov_series_df``. NaN rows are dropped
    per window/column and the surviving weights renormalized to sum to 1.
    """
    is_series = isinstance(x, pd.Series)
    df = x.to_frame() if is_series else x
    x_np = df.to_numpy(dtype=np.float64)
    T, M = x_np.shape
    out = np.full((T, M), np.nan)

    for t in range(min_periods - 1, T):
        L = min(window, t + 1)
        sl = slice(t - L + 1, t + 1)
        w_t = weights[-L:]  # (L,)
        x_w = x_np[sl]  # (L, M)

        valid = ~np.isnan(x_w)
        wm = np.where(valid, w_t[:, None], 0.0)
        wsum = wm.sum(axis=0)
        ok = (valid.sum(axis=0) >= min_periods) & (wsum > 0)
        if not ok.any():
            continue
        wn = wm / np.where(wsum > 0, wsum, 1.0)
        xbar = (wn * np.where(valid, x_w, 0.0)).sum(axis=0)
        xc = np.where(valid, x_w - xbar, 0.0)
        var = (wn * xc * xc).sum(axis=0)
        out[t, ok] = var[ok]

    if is_series:
        return pd.Series(out[:, 0], index=df.index, name=x.name)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


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
    fit_intercept : bool
        Include an explicit intercept in every window fit. Defaults to True.
    lambda_ : float
        Ridge regularization strength. 0.0 = standard OLS (default).
        Adding a small value (e.g. 1e-4) stabilizes collinear regressors.
    ewma_halflife : int, optional
        If set, observations within each window are exponentially weighted so
        recent data carries more weight: an observation `ewma_halflife` steps
        in the past gets half the weight of the most recent one. Affects the
        rolling betas, R², and the Frisch-Waugh residualization. If None
        (default), all observations are weighted equally and the equal-weight
        fast paths are used unchanged (zero performance impact). Cannot be
        combined with expanding=True — expanding windows have variable length,
        so the weight vector cannot be precomputed. Note: HAC standard errors
        are always computed with equal weights (see hac_se).
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
        Storage dtype for input/intermediate pandas DataFrames. 'float32' saves
        memory; 'float64' for higher precision. Note: this controls DataFrame
        storage only — internal matrix operations (gram matrix accumulation and
        the linear solve) always run in float64 for numerical stability,
        regardless of this setting. See rolling_residualize for details.
    asset_chunk_size : int
        Number of assets processed per chunk during residualization.
        Lower values reduce peak memory at the cost of slightly more overhead.
    warn_singular : bool
        If True (default), emit a RuntimeWarning when one or more rolling
        windows are singular (collinear regressors or degenerate windows),
        with the affected estimates set to NaN. Set False to suppress these
        warnings when singular windows are expected (e.g. short warm-ups).

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
        min_periods: int | None = None,
        expanding: bool = False,
        fit_intercept: bool = True,
        lambda_: float = 0.0,
        adj_r2: bool = False,
        lag_signal: bool = False,
        hac_lags: int | None = None,
        denom_tol: float = 1e-12,
        dtype: str = "float32",
        asset_chunk_size: int = 100,
        warn_singular: bool = True,
        ewma_halflife: int | None = None,
    ) -> None:
        if ewma_halflife is not None and expanding:
            raise ValueError(
                "ewma_halflife cannot be combined with expanding=True: expanding "
                "windows have variable length, so the EWMA weight vector cannot "
                "be precomputed."
            )

        self.window = window
        self.min_periods = min_periods if min_periods is not None else window
        self.expanding = expanding
        self.fit_intercept = fit_intercept
        self.lambda_ = lambda_
        self.ewma_halflife = ewma_halflife
        self.adj_r2 = adj_r2
        self.lag_signal = lag_signal
        self.hac_lags = hac_lags
        self.denom_tol = denom_tol
        self.dtype = dtype
        self.asset_chunk_size = asset_chunk_size
        self.warn_singular = warn_singular

        self._is_fitted = False
        self._factor_cols: list[str] = []
        self._control_cols: list[str] = []
        self._index: pd.Index | None = None
        self._factors_raw: pd.DataFrame | None = None  # original, for signal
        self._factors_fitted: pd.DataFrame | None = None
        self._controls_fitted: pd.DataFrame | None = None

    def _weights(self) -> np.ndarray | None:
        """EWMA observation weights for one full window, or None for equal weights."""
        if self.ewma_halflife is None:
            return None
        return _ewma_weights(self.ewma_halflife, self.window)

    def _solve_targets(self, targets: pd.DataFrame, design: pd.DataFrame) -> JointFitResult:
        """Run the joint solver in target-column chunks and combine its outputs."""
        chunks = [
            targets.columns[i : i + self.asset_chunk_size]
            for i in range(0, targets.shape[1], self.asset_chunk_size)
        ]
        fits = [
            rolling_joint_solve(
                y=targets.loc[:, chunk],
                X=design,
                window=self.window,
                min_periods=self.min_periods,
                expanding=self.expanding,
                fit_intercept=self.fit_intercept,
                weights=self._weights(),
                warn_singular=self.warn_singular,
            )
            for chunk in chunks
        ]
        return JointFitResult(
            coef=np.concatenate([fit.coef for fit in fits], axis=2),
            intercept=np.concatenate([fit.intercept for fit in fits], axis=1),
            resid_endpoint=np.concatenate([fit.resid_endpoint for fit in fits], axis=1),
            ssr=np.concatenate([fit.ssr for fit in fits], axis=1),
            sst=np.concatenate([fit.sst for fit in fits], axis=1),
            n_used=np.concatenate([fit.n_used for fit in fits], axis=1),
            n_eff=np.concatenate([fit.n_eff for fit in fits], axis=1),
        )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        factors: pd.DataFrame,
        controls: pd.DataFrame | None = None,
        orthogonalize_factors: bool = False,
        orthogonalize_controls: bool = False,
    ) -> RollingOLS:
        """
        Validate and store the regressors used by each window fit.

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
        frames = [factors] if controls is None else [factors, controls]
        names = ["factors"] if controls is None else ["factors", "controls"]
        _validate_index(*frames, names=names)

        factors = factors.astype(self.dtype)
        self._index = factors.index.copy()
        self._factors_raw = factors  # kept for signal computation

        if orthogonalize_factors and factors.shape[1] > 1:
            factors = rolling_gram_schmidt(
                factors,
                window=self.window,
                min_periods=self.min_periods,
                expanding=self.expanding,
                warn_singular=self.warn_singular,
            ).astype(self.dtype)

        self._factor_cols = factors.columns.tolist()
        self._factors_fitted = factors

        if controls is not None:
            controls = controls.astype(self.dtype)

            if orthogonalize_controls and controls.shape[1] > 1:
                controls = rolling_gram_schmidt(
                    controls,
                    window=self.window,
                    min_periods=self.min_periods,
                    expanding=self.expanding,
                    warn_singular=self.warn_singular,
                ).astype(self.dtype)

            self._control_cols = controls.columns.tolist()
            self._controls_fitted = controls
        else:
            self._control_cols = []
            self._controls_fitted = None

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(
        self,
        assets: pd.DataFrame,
        return_control_betas: bool = False,
    ) -> RollingOLSResult:
        """
        Project assets onto fitted factor structure.

        Computes rolling betas, signals, R², and optionally residuals for HAC.

        Parameters
        ----------
        assets : pd.DataFrame
            Target returns. e.g. df[["AAPL", "MSFT", "GOOG"]]
        return_control_betas : bool
            If True (and controls were passed to fit()), also compute each
            control's joint rolling beta via Frisch-Waugh-Lovell partitioning,
            accessible through result.get_control_beta(). More expensive.

        Returns
        -------
        RollingOLSResult
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")
        if self._index is None:
            raise RuntimeError("Fitted index is unavailable. Call fit() again before transform().")
        fitted_index = pd.DataFrame(index=self._index)
        _validate_index(fitted_index, assets, names=["factors", "assets"])

        asset_cols = assets.columns.tolist()
        assets = assets.astype(self.dtype)

        if self._factors_fitted is None or self._factors_raw is None:
            raise RuntimeError("Fitted factors are unavailable. Call fit() again.")

        # Preserve the controls-only residual accessor using a separate direct
        # current-window model. It is not used to estimate factor coefficients.
        if self._controls_fitted is not None:
            controls_only_fit = self._solve_targets(assets, self._controls_fitted)
            asset_resids = pd.DataFrame(
                controls_only_fit.resid_endpoint,
                index=assets.index,
                columns=assets.columns,
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

        # Controls-only endpoint residuals, retained for the existing accessor.
        result._factor_adjusted_returns = asset_resids

        for fac in self._factor_cols:
            design_parts = []
            if self._controls_fitted is not None:
                design_parts.append(self._controls_fitted)
            design_parts.append(self._factors_fitted[[fac]])
            design = pd.concat(design_parts, axis=1)
            fit = self._solve_targets(assets, design)

            beta = pd.DataFrame(fit.coef[:, -1, :], index=assets.index, columns=assets.columns)
            intercept = pd.DataFrame(fit.intercept, index=assets.index, columns=assets.columns)
            residuals = pd.DataFrame(fit.resid_endpoint, index=assets.index, columns=assets.columns)
            n_used = pd.DataFrame(fit.n_used, index=assets.index, columns=assets.columns)
            residual_dof = fit.n_eff - design.shape[1] - int(self.fit_intercept)
            dof = pd.DataFrame(residual_dof, index=assets.index, columns=assets.columns)
            r2_values = np.divide(
                fit.ssr,
                fit.sst,
                out=np.full_like(fit.ssr, np.nan),
                where=fit.sst > self.denom_tol,
            )
            r2_values = 1.0 - r2_values
            if self.adj_r2:
                numerator_dof = fit.n_eff - int(self.fit_intercept)
                adjustment = np.divide(
                    numerator_dof,
                    residual_dof,
                    out=np.full_like(residual_dof, np.nan),
                    where=residual_dof > 0,
                )
                r2_values = 1.0 - (1.0 - r2_values) * adjustment
            r2 = pd.DataFrame(r2_values, index=assets.index, columns=assets.columns)

            f_orig = self._factors_raw[fac]
            signal = (
                beta.shift(1).mul(f_orig, axis=0) if self.lag_signal else beta.mul(f_orig, axis=0)
            )

            result._betas[fac] = beta
            result._intercepts[fac] = intercept
            result._signals[fac] = signal
            result._r2[fac] = r2
            result._residuals[fac] = residuals
            result._dof[fac] = dof
            result._n_used[fac] = n_used
            result._factor_values[fac] = self._factors_fitted[fac]

        # Control betas via FWL — independent of factor, so computed once and
        # shared across all factors. For each control, partial it (and the
        # assets) against all OTHER controls, then rolling univariate OLS.
        control_betas: dict = {}
        weights = self._weights()
        if return_control_betas and self._controls_fitted is not None:
            for ctrl in self._control_cols:
                other_controls = [c for c in self._control_cols if c != ctrl]
                if other_controls:
                    asset_resid_j = rolling_residualize(
                        y=assets,
                        X=self._controls_fitted[other_controls],
                        window=self.window,
                        min_periods=self.min_periods,
                        expanding=self.expanding,
                        ridge_lambda=self.lambda_,
                        warn_singular=self.warn_singular,
                        weights=weights,
                    )
                    ctrl_j_resid = rolling_residualize(
                        y=self._controls_fitted[[ctrl]],
                        X=self._controls_fitted[other_controls],
                        window=self.window,
                        min_periods=self.min_periods,
                        expanding=self.expanding,
                        ridge_lambda=self.lambda_,
                        warn_singular=self.warn_singular,
                        weights=weights,
                    )[ctrl]
                else:
                    asset_resid_j = assets
                    ctrl_j_resid = self._controls_fitted[ctrl]

                if weights is not None:
                    cov_ac = _ewma_cov_series_df(
                        ctrl_j_resid, asset_resid_j, weights, self.window, self.min_periods
                    )
                    var_c = _ewma_var(ctrl_j_resid, weights, self.window, self.min_periods)
                else:
                    cov_ac = _rolling_cov_series_df(
                        ctrl_j_resid, asset_resid_j, self.window, self.min_periods, self.expanding
                    )
                    var_c = _rolling_var(
                        ctrl_j_resid, self.window, self.min_periods, self.expanding
                    )
                var_c_safe = var_c.where(var_c.abs() > self.denom_tol)
                control_betas[ctrl] = cov_ac.div(var_c_safe, axis=0)

            # Same dict shared across all factors
            for fac in self._factor_cols:
                result._control_betas[fac] = control_betas

        return result

    # ------------------------------------------------------------------
    # fit_transform
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        factors: pd.DataFrame,
        assets: pd.DataFrame,
        controls: pd.DataFrame | None = None,
        orthogonalize_factors: bool = False,
        orthogonalize_controls: bool = False,
        return_control_betas: bool = False,
    ) -> RollingOLSResult:
        """
        Convenience: fit() then transform() in one call.

        Parameters mirror fit() and transform() — see their docstrings.
        """
        return self.fit(factors, controls, orthogonalize_factors, orthogonalize_controls).transform(
            assets, return_control_betas=return_control_betas
        )
