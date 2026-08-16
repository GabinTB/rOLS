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

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .estimators import (
    BatchedFitResult,
    JointFitResult,
    PatternSufficientStatistics,
    _warn_ill_conditioned,
    _warn_singular,
    rolling_fwl_solve,
    rolling_joint_solve,
)
from .results import FactorStatistics, RollingOLSResult

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
    cache_size : int
        Maximum number of factors retained in each lazy result cache. Defaults
        to 1 so residual and inference access stays bounded for large panels.

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
        cache_size: int = 1,
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
        if not isinstance(cache_size, int) or isinstance(cache_size, bool):
            raise TypeError("cache_size must be an integer")
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")

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
        self.cache_size = cache_size

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

    def estimate_memory(
        self,
        targets: pd.DataFrame,
        factors: pd.DataFrame,
        controls: pd.DataFrame | None = None,
    ) -> dict[str, int | str]:
        """Estimate persistent and per-cache memory before fitting a panel."""
        n_observations = len(targets)
        n_targets = targets.shape[1]
        n_factors = factors.shape[1]
        n_controls = 0 if controls is None else controls.shape[1]
        float_bytes = np.dtype(np.float64).itemsize
        factor_frame_bytes = n_observations * n_targets * n_factors * float_bytes
        target_frame_bytes = n_observations * n_targets * float_bytes

        factor_validity = np.isfinite(factors.to_numpy(dtype=np.float64))
        if n_factors:
            packed = np.packbits(factor_validity, axis=0)
            n_factor_patterns = len(
                {packed[:, position].tobytes() for position in range(n_factors)}
            )
        else:
            n_factor_patterns = 0
        pattern_target_bytes = n_observations * n_targets * max(1, n_factor_patterns) * float_bytes

        estimates: dict[str, int | str] = {
            "betas": factor_frame_bytes,
            "intercepts": factor_frame_bytes,
            "n_used": pattern_target_bytes,
            "cross_products": factor_frame_bytes,
            "pattern_statistics": 3 * pattern_target_bytes
            + n_observations * n_factors * float_bytes,
            "retained_inputs": n_observations * (n_targets + n_factors + n_controls) * float_bytes,
            "on_demand_per_frame": target_frame_bytes,
            "on_demand_cache_bytes": self.cache_size * target_frame_bytes,
            "note": (
                "Each lazy quantity costs cache_size * T * N * 8 bytes. "
                "Factor NaNs split sufficient-statistic patterns and increase storage."
            ),
        }
        estimates["total"] = sum(
            value
            for key, value in estimates.items()
            if isinstance(value, int)
            and key not in {"on_demand_per_frame", "on_demand_cache_bytes"}
        )
        return estimates

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

    def _finalize_statistics(
        self,
        r2_values: np.ndarray,
        partial_r2_values: np.ndarray,
        n_eff_values: np.ndarray,
        n_used_values: np.ndarray,
        assets: pd.DataFrame,
        n_controls: int,
    ) -> FactorStatistics:
        """Apply dof adjustment and wrap one factor's derived statistics."""
        residual_dof = n_eff_values - (n_controls + 1) - int(self.fit_intercept)
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
        frame_kwargs = {"index": assets.index, "columns": assets.columns}
        return FactorStatistics(
            r2=pd.DataFrame(r2_values, **frame_kwargs),
            partial_r2=pd.DataFrame(partial_r2_values, **frame_kwargs),
            dof=pd.DataFrame(residual_dof, **frame_kwargs),
            n_used=pd.DataFrame(n_used_values, **frame_kwargs),
        )

    def _statistics_from_arrays(
        self,
        full_ssr: np.ndarray,
        raw_sst: np.ndarray,
        reduced_ssr: np.ndarray,
        n_eff: np.ndarray,
        n_used: np.ndarray,
        assets: pd.DataFrame,
        n_controls: int,
    ) -> FactorStatistics:
        """Derive full and partial R² from one lazily recomputed joint fit."""
        r2_values = 1.0 - np.divide(
            full_ssr,
            raw_sst,
            out=np.full_like(full_ssr, np.nan),
            where=raw_sst > self.denom_tol,
        )
        partial_r2_values = np.divide(
            reduced_ssr - full_ssr,
            reduced_ssr,
            out=np.full_like(full_ssr, np.nan),
            where=reduced_ssr > self.denom_tol,
        )
        return self._finalize_statistics(
            r2_values,
            partial_r2_values,
            n_eff,
            n_used,
            assets,
            n_controls,
        )

    def _statistics_from_patterns(
        self,
        factor_position: int,
        beta: pd.DataFrame,
        statistics: tuple[PatternSufficientStatistics, ...],
        assets: pd.DataFrame,
        n_controls: int,
        selected_target_positions: np.ndarray | None = None,
    ) -> FactorStatistics:
        """Derive statistics from exact FWL complete-case pattern records."""
        output_shape = beta.shape
        r2_values = np.full(output_shape, np.nan)
        partial_r2_values = np.full(output_shape, np.nan)
        n_eff_values = np.full(output_shape, np.nan)
        n_used_values = np.full(output_shape, np.nan)
        beta_values = beta.to_numpy(dtype=np.float64, copy=False)
        all_targets = selected_target_positions is None
        if all_targets:
            selected_target_positions = np.arange(assets.shape[1])
        output_positions = {
            int(original): local for local, original in enumerate(selected_target_positions)
        }

        for pattern in statistics:
            matches = np.flatnonzero(pattern.factor_positions == factor_position)
            if matches.size == 0:
                continue
            local_factor = int(matches[0])
            if all_targets:
                selected_in_pattern = slice(None)
                target_positions = pattern.target_positions
                local_output_positions = target_positions
            else:
                selected_in_pattern = np.array(
                    [
                        position
                        for position, original in enumerate(pattern.target_positions)
                        if int(original) in output_positions
                    ],
                    dtype=int,
                )
                if selected_in_pattern.size == 0:
                    continue
                target_positions = pattern.target_positions[selected_in_pattern]
                local_output_positions = np.array(
                    [output_positions[int(position)] for position in target_positions],
                    dtype=int,
                )
            endpoint_beta = beta_values[pattern.endpoint, local_output_positions]
            explained_ss = endpoint_beta**2 * pattern.denominators[local_factor]
            reduced_ssr = pattern.reduced_ssr[selected_in_pattern]
            raw_sst = pattern.raw_sst[selected_in_pattern]
            full_ssr = reduced_ssr - explained_ss
            r2_values[pattern.endpoint, local_output_positions] = 1.0 - np.divide(
                full_ssr,
                raw_sst,
                out=np.full_like(full_ssr, np.nan),
                where=raw_sst > self.denom_tol,
            )
            partial_r2_values[pattern.endpoint, local_output_positions] = np.divide(
                reduced_ssr - full_ssr,
                reduced_ssr,
                out=np.full_like(full_ssr, np.nan),
                where=reduced_ssr > self.denom_tol,
            )
            n_eff_values[pattern.endpoint, local_output_positions] = np.where(
                np.isfinite(endpoint_beta),
                pattern.n_eff,
                np.nan,
            )
            n_used_values[pattern.endpoint, local_output_positions] = np.where(
                np.isfinite(endpoint_beta),
                pattern.n_used,
                np.nan,
            )

        return self._finalize_statistics(
            r2_values,
            partial_r2_values,
            n_eff_values,
            n_used_values,
            assets,
            n_controls,
        )

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

        Computes rolling betas, intercepts, and observation counts. Signals,
        fit statistics, residuals, and inference quantities are evaluated on
        demand by the returned result.

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

        factors_snapshot = self._factors
        controls_snapshot = self._controls_fitted
        weights_snapshot = self._weights()
        window = self.window
        min_periods = self.min_periods
        expanding = self.expanding
        fit_intercept = self.fit_intercept
        cond_warn_threshold = self.cond_warn_threshold
        asset_chunk_size = self.asset_chunk_size
        n_controls = len(self._control_cols)
        asset_positions = {asset: position for position, asset in enumerate(asset_cols)}
        factor_penalty = self._penalty_matrix(n_controls, n_factors=1)
        controls_penalty = self._penalty_matrix(n_controls, n_factors=0)

        def selected_targets(selected_assets: Sequence[str] | None) -> pd.DataFrame:
            if selected_assets is None:
                return assets
            return assets.loc[:, list(selected_assets)]

        def solve_snapshot(
            targets: pd.DataFrame,
            design: pd.DataFrame,
            penalty: np.ndarray | None,
        ) -> JointFitResult:
            """Recompute one lazy quantity without consulting mutable model state."""
            chunks = [
                targets.columns[position : position + asset_chunk_size]
                for position in range(0, targets.shape[1], asset_chunk_size)
            ]
            fits = [
                rolling_joint_solve(
                    y=targets.loc[:, chunk],
                    X=design,
                    window=window,
                    min_periods=min_periods,
                    expanding=expanding,
                    fit_intercept=fit_intercept,
                    penalty=penalty,
                    weights=weights_snapshot,
                    warn_singular=False,
                    cond_warn_threshold=cond_warn_threshold,
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
                n_singular=sum(fit.n_singular for fit in fits),
                n_ill_conditioned=sum(fit.n_ill_conditioned for fit in fits),
            )

        def factor_design(factor: str) -> pd.DataFrame:
            parts = [] if controls_snapshot is None else [controls_snapshot]
            return pd.concat([*parts, factors_snapshot[[factor]]], axis=1)

        fwl_fit: BatchedFitResult | None = None
        if self.lambda_ == 0:
            fwl_fit = rolling_fwl_solve(
                y=assets,
                factors=factors_snapshot,
                controls=controls_snapshot,
                window=window,
                min_periods=min_periods,
                expanding=expanding,
                fit_intercept=fit_intercept,
                weights=weights_snapshot,
                warn_singular=self.warn_singular,
                cond_warn_threshold=cond_warn_threshold,
                params_only=True,
                return_nuisance_coef=return_control_betas,
            )

        result = RollingOLSResult(
            factor_cols=self._factor_cols,
            asset_cols=asset_cols,
            index=assets.index,
            lag_signal=self.lag_signal,
            window=self.window,
            min_periods=self.min_periods,
            expanding=self.expanding,
            hac_lags=self.hac_lags,
            cache_size=self.cache_size,
        )
        result._path = "fwl" if self.lambda_ == 0 else "joint"

        direct_control_betas: dict[str, dict[str, pd.DataFrame]] = {}
        for factor_position, fac in enumerate(self._factor_cols):
            if fwl_fit is None:
                fit = self._solve_targets(
                    assets,
                    factor_design(fac),
                    penalty=factor_penalty,
                )
                beta_values = fit.coef[:, -1, :]
                intercept_values = fit.intercept
                n_used_values = fit.n_used
            else:
                assert fwl_fit.factor_coef is not None
                assert fwl_fit.intercept is not None
                beta_values = fwl_fit.factor_coef[:, factor_position, :]
                intercept_values = fwl_fit.intercept[:, factor_position, :]

            beta = pd.DataFrame(beta_values, index=assets.index, columns=assets.columns)
            intercept = pd.DataFrame(
                intercept_values,
                index=assets.index,
                columns=assets.columns,
            )
            result._betas[fac] = beta
            result._intercepts[fac] = intercept
            if fwl_fit is None:
                result._n_used[fac] = pd.DataFrame(
                    n_used_values,
                    index=assets.index,
                    columns=assets.columns,
                )
            result._factor_values[fac] = factors_snapshot[fac]
            if return_control_betas and controls_snapshot is not None:
                if fwl_fit is None:
                    control_coef = fit.coef
                else:
                    assert fwl_fit.nuisance_coef is not None
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

        def derive_joint_statistics(
            factor: str,
            selected_assets: Sequence[str] | None,
        ) -> FactorStatistics:
            targets = selected_targets(selected_assets)
            fit = solve_snapshot(targets, factor_design(factor), factor_penalty)
            if controls_snapshot is None:
                reduced_ssr = fit.sst
            else:
                factor_is_finite = pd.Series(
                    np.isfinite(factors_snapshot[factor].to_numpy()),
                    index=assets.index,
                )
                reduced_fit = solve_snapshot(
                    targets.where(factor_is_finite, axis=0),
                    controls_snapshot,
                    controls_penalty,
                )
                reduced_ssr = reduced_fit.ssr
            return self._statistics_from_arrays(
                fit.ssr,
                fit.sst,
                reduced_ssr,
                fit.n_eff,
                fit.n_used,
                targets,
                n_controls,
            )

        if fwl_fit is not None:
            result._sufficient_statistics = fwl_fit.sufficient_statistics
            factor_positions = {
                factor: position for position, factor in enumerate(self._factor_cols)
            }

            def derive_fwl_statistics(
                factor: str,
                selected_assets: Sequence[str] | None,
            ) -> FactorStatistics:
                targets = selected_targets(selected_assets)
                selected_positions = (
                    None
                    if selected_assets is None
                    else np.array([asset_positions[asset] for asset in selected_assets], dtype=int)
                )
                return self._statistics_from_patterns(
                    factor_position=factor_positions[factor],
                    beta=result._betas[factor].loc[:, targets.columns],
                    statistics=fwl_fit.sufficient_statistics,
                    assets=targets,
                    n_controls=n_controls,
                    selected_target_positions=selected_positions,
                )

            result._statistics_loader = derive_fwl_statistics
        else:
            result._statistics_loader = derive_joint_statistics

        def load_residuals(
            factor: str,
            selected_assets: Sequence[str] | None,
        ) -> pd.DataFrame:
            targets = selected_targets(selected_assets)
            if fwl_fit is not None:
                factor_fit = rolling_fwl_solve(
                    y=targets,
                    factors=factors_snapshot[[factor]],
                    controls=controls_snapshot,
                    window=window,
                    min_periods=min_periods,
                    expanding=expanding,
                    fit_intercept=fit_intercept,
                    weights=weights_snapshot,
                    warn_singular=False,
                    cond_warn_threshold=cond_warn_threshold,
                    params_only=True,
                    return_nuisance_coef=False,
                    residuals_only=True,
                )
                assert factor_fit.resid_endpoint is not None
                residual_values = factor_fit.resid_endpoint[:, 0, :]
            else:
                fit = solve_snapshot(targets, factor_design(factor), factor_penalty)
                residual_values = fit.resid_endpoint
            return pd.DataFrame(residual_values, index=targets.index, columns=targets.columns)

        def load_factor_adjusted_returns(
            selected_assets: Sequence[str] | None,
        ) -> pd.DataFrame:
            targets = selected_targets(selected_assets)
            if controls_snapshot is None:
                return targets
            controls_fit = solve_snapshot(targets, controls_snapshot, controls_penalty)
            return pd.DataFrame(
                controls_fit.resid_endpoint,
                index=targets.index,
                columns=targets.columns,
            )

        result._residual_loader = load_residuals
        result._factor_adjusted_loader = load_factor_adjusted_returns

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
