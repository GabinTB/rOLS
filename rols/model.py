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

Ridge regularization uses a normalized weighted loss and standardizes penalized
regressors within each complete-case window. Factors are always penalized when
lambda_ > 0; controls are penalized by default. The intercept is never penalized.

HAC standard errors (Newey-West) are computed on demand via result.get_se(factor).
Set hac_lags on the constructor to enable this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .estimators import (
    BatchedFitResult,
    JointFitResult,
    _warn_ill_conditioned,
    _warn_singular,
    rolling_fwl_solve,
    rolling_joint_solve,
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
        Ridge strength for the normalized weighted objective. Penalized
        regressors are standardized within each complete-case window before
        solving and returned in their original units. 0.0 gives OLS.
    penalize_controls : bool
        Penalize controls along with factors when ``lambda_ > 0``. Defaults to
        True. Set to False to treat controls as unpenalized nuisance regressors.
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
        This also controls ill-conditioning warnings.
    cond_warn_threshold : float
        Warn when the condition number of a window's weighted design Gram matrix
        exceeds this value. The default is 1e10. This threshold applies to
        ``cond(X'X)``, which is approximately ``cond(X) ** 2``.

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
        penalize_controls: bool = True,
        adj_r2: bool = False,
        lag_signal: bool = False,
        hac_lags: int | None = None,
        denom_tol: float = 1e-12,
        dtype: str = "float32",
        asset_chunk_size: int = 100,
        warn_singular: bool = True,
        ewma_halflife: int | None = None,
        cond_warn_threshold: float = 1e10,
    ) -> None:
        if ewma_halflife is not None and expanding:
            raise ValueError(
                "ewma_halflife cannot be combined with expanding=True: expanding "
                "windows have variable length, so the EWMA weight vector cannot "
                "be precomputed."
            )
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative")
        if not np.isfinite(cond_warn_threshold) or cond_warn_threshold <= 0:
            raise ValueError("cond_warn_threshold must be finite and positive")

        self.window = window
        self.min_periods = min_periods if min_periods is not None else window
        self.expanding = expanding
        self.fit_intercept = fit_intercept
        self.lambda_ = lambda_
        self.penalize_controls = penalize_controls
        self.ewma_halflife = ewma_halflife
        self.adj_r2 = adj_r2
        self.lag_signal = lag_signal
        self.hac_lags = hac_lags
        self.denom_tol = denom_tol
        self.dtype = dtype
        self.asset_chunk_size = asset_chunk_size
        self.warn_singular = warn_singular
        self.cond_warn_threshold = cond_warn_threshold

        self._is_fitted = False
        self._factor_cols: list[str] = []
        self._control_cols: list[str] = []
        self._index: pd.Index | None = None
        self._factors: pd.DataFrame | None = None
        self._controls_fitted: pd.DataFrame | None = None

    def _weights(self) -> np.ndarray | None:
        """EWMA observation weights for one full window, or None for equal weights."""
        if self.ewma_halflife is None:
            return None
        return _ewma_weights(self.ewma_halflife, self.window)

    def _penalty_matrix(self, n_controls: int, n_factors: int) -> np.ndarray | None:
        """Build the diagonal penalty in standardized solve coordinates."""
        if self.lambda_ == 0:
            return None
        n_slopes = n_controls + n_factors
        penalty = np.zeros((n_slopes + int(self.fit_intercept),) * 2)
        slope_offset = int(self.fit_intercept)
        if self.penalize_controls:
            control_positions = np.arange(slope_offset, slope_offset + n_controls)
            penalty[control_positions, control_positions] = self.lambda_
        factor_start = slope_offset + n_controls
        factor_positions = np.arange(factor_start, factor_start + n_factors)
        penalty[factor_positions, factor_positions] = self.lambda_
        return penalty

    def _solve_targets(
        self,
        targets: pd.DataFrame,
        design: pd.DataFrame,
        penalty: np.ndarray | None = None,
    ) -> JointFitResult:
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
                penalty=penalty,
                weights=self._weights(),
                warn_singular=False,
                cond_warn_threshold=self.cond_warn_threshold,
            )
            for chunk in chunks
        ]
        combined = JointFitResult(
            coef=np.concatenate([fit.coef for fit in fits], axis=2),
            intercept=np.concatenate([fit.intercept for fit in fits], axis=1),
            resid_endpoint=np.concatenate([fit.resid_endpoint for fit in fits], axis=1),
            ssr=np.concatenate([fit.ssr for fit in fits], axis=1),
            sst=np.concatenate([fit.sst for fit in fits], axis=1),
            n_used=np.concatenate([fit.n_used for fit in fits], axis=1),
            n_eff=np.concatenate([fit.n_eff for fit in fits], axis=1),
            n_singular=sum(fit.n_singular for fit in fits),
            n_ill_conditioned=sum(fit.n_ill_conditioned for fit in fits),
        )
        if self.warn_singular:
            _warn_singular(combined.n_singular)
            _warn_ill_conditioned(combined.n_ill_conditioned, self.cond_warn_threshold)
        return combined

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        factors: pd.DataFrame,
        controls: pd.DataFrame | None = None,
    ) -> RollingOLS:
        """
        Validate and store the regressors used by each window fit.

        Parameters
        ----------
        factors : pd.DataFrame
            Regressors of interest. Each column gets its own rolling beta.
            e.g. df[["narrative_1", "narrative_2"]]
        controls : pd.DataFrame, optional
            Always-in regressors to partial out (e.g. df[["Mkt-RF", "SMB"]]).
            If None, no partialling out — pure factor regression.
        Returns
        -------
        self
        """
        frames = [factors] if controls is None else [factors, controls]
        names = ["factors"] if controls is None else ["factors", "controls"]
        _validate_index(*frames, names=names)

        factors = factors.astype(self.dtype)
        self._index = factors.index.copy()
        self._factor_cols = factors.columns.tolist()
        self._factors = factors

        if controls is not None:
            controls = controls.astype(self.dtype)
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

        if self._factors is None:
            raise RuntimeError("Fitted factors are unavailable. Call fit() again.")

        fwl_fit: BatchedFitResult | None = None
        if self.lambda_ == 0:
            fwl_fit = rolling_fwl_solve(
                y=assets,
                factors=self._factors,
                controls=self._controls_fitted,
                window=self.window,
                min_periods=self.min_periods,
                expanding=self.expanding,
                fit_intercept=self.fit_intercept,
                weights=self._weights(),
                warn_singular=self.warn_singular,
                cond_warn_threshold=self.cond_warn_threshold,
            )

        # Preserve the controls-only residual accessor using a separate direct
        # current-window model when factor missingness prevents sharing the FWL
        # nuisance projection. It is not used to estimate factor coefficients.
        controls_only_fit: JointFitResult | None = None
        if self._controls_fitted is not None:
            if fwl_fit is not None and np.isfinite(self._factors.to_numpy()).all():
                controls_only_residuals = fwl_fit.nuisance_resid_endpoint
            else:
                controls_only_fit = self._solve_targets(
                    assets,
                    self._controls_fitted,
                    penalty=self._penalty_matrix(len(self._control_cols), n_factors=0),
                )
                controls_only_residuals = controls_only_fit.resid_endpoint
            asset_resids = pd.DataFrame(
                controls_only_residuals,
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
        result._path = "fwl" if self.lambda_ == 0 else "joint"

        # Controls-only endpoint residuals, retained for the existing accessor.
        result._factor_adjusted_returns = asset_resids

        direct_control_betas: dict[str, dict[str, pd.DataFrame]] = {}
        reduced_fit_cache: dict[bytes, JointFitResult] = {}
        if controls_only_fit is not None:
            all_finite = np.ones(len(assets), dtype=bool)
            reduced_fit_cache[np.packbits(all_finite).tobytes()] = controls_only_fit
        for factor_position, fac in enumerate(self._factor_cols):
            design_parts = []
            if self._controls_fitted is not None:
                design_parts.append(self._controls_fitted)
            design_parts.append(self._factors[[fac]])
            design = pd.concat(design_parts, axis=1)
            if fwl_fit is None:
                fit = self._solve_targets(
                    assets,
                    design,
                    penalty=self._penalty_matrix(len(self._control_cols), n_factors=1),
                )
                beta_values = fit.coef[:, -1, :]
                intercept_values = fit.intercept
                residual_values = fit.resid_endpoint
                n_used_values = fit.n_used
                n_eff_values = fit.n_eff
                fit_ssr = fit.ssr
                fit_sst = fit.sst
            else:
                beta_values = fwl_fit.factor_coef[:, factor_position, :]
                intercept_values = fwl_fit.intercept[:, factor_position, :]
                residual_values = fwl_fit.resid_endpoint[:, factor_position, :]
                n_used_values = fwl_fit.n_used[:, factor_position, :]
                n_eff_values = fwl_fit.n_eff[:, factor_position, :]
                fit_ssr = fwl_fit.ssr[:, factor_position, :]
                fit_sst = fwl_fit.sst[:, factor_position, :]

            beta = pd.DataFrame(beta_values, index=assets.index, columns=assets.columns)
            intercept = pd.DataFrame(
                intercept_values,
                index=assets.index,
                columns=assets.columns,
            )
            residuals = pd.DataFrame(
                residual_values,
                index=assets.index,
                columns=assets.columns,
            )
            n_used = pd.DataFrame(n_used_values, index=assets.index, columns=assets.columns)
            residual_dof = n_eff_values - design.shape[1] - int(self.fit_intercept)
            dof = pd.DataFrame(residual_dof, index=assets.index, columns=assets.columns)
            r2_values = np.divide(
                fit_ssr,
                fit_sst,
                out=np.full_like(fit_ssr, np.nan),
                where=fit_sst > self.denom_tol,
            )
            r2_values = 1.0 - r2_values

            if fwl_fit is not None:
                reduced_ssr = fwl_fit.reduced_ssr[:, factor_position, :]
            elif self._controls_fitted is None:
                reduced_ssr = fit_sst
            else:
                factor_is_finite = pd.Series(
                    np.isfinite(self._factors[fac].to_numpy()),
                    index=assets.index,
                )
                factor_mask_key = np.packbits(factor_is_finite.to_numpy()).tobytes()
                reduced_fit = reduced_fit_cache.get(factor_mask_key)
                if reduced_fit is None:
                    reduced_targets = assets.where(factor_is_finite, axis=0)
                    reduced_fit = self._solve_targets(
                        reduced_targets,
                        self._controls_fitted,
                        penalty=self._penalty_matrix(len(self._control_cols), n_factors=0),
                    )
                    reduced_fit_cache[factor_mask_key] = reduced_fit
                reduced_ssr = reduced_fit.ssr
            partial_r2_values = np.divide(
                reduced_ssr - fit_ssr,
                reduced_ssr,
                out=np.full_like(fit_ssr, np.nan),
                where=reduced_ssr > self.denom_tol,
            )

            if self.adj_r2:
                numerator_dof = n_eff_values - int(self.fit_intercept)
                adjustment = np.divide(
                    numerator_dof,
                    residual_dof,
                    out=np.full_like(residual_dof, np.nan),
                    where=residual_dof > 0,
                )
                r2_values = 1.0 - (1.0 - r2_values) * adjustment
                partial_r2_values = 1.0 - (1.0 - partial_r2_values) * adjustment
            r2 = pd.DataFrame(r2_values, index=assets.index, columns=assets.columns)
            partial_r2 = pd.DataFrame(
                partial_r2_values,
                index=assets.index,
                columns=assets.columns,
            )

            factor = self._factors[fac]
            signal = (
                beta.shift(1).mul(factor, axis=0) if self.lag_signal else beta.mul(factor, axis=0)
            )

            result._betas[fac] = beta
            result._intercepts[fac] = intercept
            result._signals[fac] = signal
            result._r2[fac] = r2
            result._partial_r2[fac] = partial_r2
            result._residuals[fac] = residuals
            result._dof[fac] = dof
            result._n_used[fac] = n_used
            result._factor_values[fac] = factor
            if return_control_betas and self._controls_fitted is not None:
                if fwl_fit is None:
                    control_coef = fit.coef
                else:
                    control_coef = fwl_fit.nuisance_coef[:, factor_position, :, :]
                direct_control_betas[fac] = {
                    control: pd.DataFrame(
                        control_coef[:, control_position, :],
                        index=assets.index,
                        columns=assets.columns,
                    )
                    for control_position, control in enumerate(self._control_cols)
                }

        if direct_control_betas:
            result._control_betas = direct_control_betas

        return result

    # ------------------------------------------------------------------
    # fit_transform
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        factors: pd.DataFrame,
        assets: pd.DataFrame,
        controls: pd.DataFrame | None = None,
        return_control_betas: bool = False,
    ) -> RollingOLSResult:
        """
        Convenience: fit() then transform() in one call.

        Parameters mirror fit() and transform() — see their docstrings.
        """
        return self.fit(factors, controls).transform(
            assets, return_control_betas=return_control_betas
        )
