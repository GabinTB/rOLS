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

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from .estimators import (
    BatchedFitResult,
    JointFitResult,
    PatternSufficientStatistics,
    _warn_ill_conditioned,
    _warn_singular,
    rolling_fwl_solve,
    rolling_hac_se,
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


def _warn_correlated_factors(factors: pd.DataFrame, threshold: float = 0.3) -> None:
    """Warn once when batched-mode factors have high pairwise |correlation|.

    Correlation is computed on the full sample as a usage hint — it is not
    per-window and does not affect estimation.
    """
    corr = factors.corr()
    cols = factors.columns.tolist()
    n_pairs = sum(
        1
        for i in range(len(cols))
        for j in range(i + 1, len(cols))
        if abs(corr.iloc[i, j]) > threshold
    )
    if n_pairs > 0:
        warnings.warn(
            f"{n_pairs} factor pair(s) have |correlation| > {threshold} in batched mode. "
            "Betas are marginal given controls and do not control for the other factors. "
            "Use mode='joint' for a multivariate model.",
            UserWarning,
            stacklevel=4,
        )


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
    mode : {"batched", "joint"} or None
        How multiple factors are modeled.

        **Required when more than one factor is supplied.**  rOLS raises
        ``ValueError`` in ``fit()`` if ``factors.shape[1] > 1`` and ``mode``
        was not given — the two estimands differ whenever factors are
        correlated, and the library does not choose silently.

        ``"batched"`` fits a separate model ``y ~ 1 + controls + factor`` per
        factor — each beta is conditional on the controls but **not** on the
        other factors, so correlated factors will each absorb variation
        attributable to the others.

        ``"joint"`` fits one model with every factor,
        ``y ~ 1 + controls + factor_1 + ... + factor_K``, so each beta is
        conditional on the controls and every other factor — the statistically
        safer choice whenever factors are correlated.

        Which is faster depends on ``lambda_``: under OLS (``lambda_ == 0``),
        batched uses the FWL fast path and is at or faster than joint at panel
        scale. Under Ridge (``lambda_ > 0``), FWL is invalid, so batched
        degrades to K separate joint-equivalent solves — joint mode is then
        substantially cheaper. See ``docs/PERFORMANCE.md`` for measured
        numbers. The two modes coincide for a single factor or for mutually
        orthogonal factors. Stored on the result as ``result.mode``.

        ``None`` (default) is accepted only for single-factor calls; it
        resolves to ``"batched"`` internally and is stored as such on the
        result. Passing ``None`` with more than one factor raises
        ``ValueError``.
    warn_correlated_factors : bool
        If True (default), emit one ``UserWarning`` when ``mode="batched"``
        with more than one factor and any factor pair has sample
        ``|correlation| > 0.3``, pointing at ``mode="joint"``. The correlation
        is computed once on the full sample as a usage hint — it is not
        per-window and does not affect estimation. Set False to suppress.
    lambda_ : float
        Ridge strength for the normalized weighted objective. Penalized
        regressors are standardized to unit weighted variance within each
        complete-case window before solving and rescaled back to original
        units on return.  This means ``lambda_`` has the **same** effective
        strength regardless of window length, EWMA half-life, or the number
        of complete rows in a window — the gram diagonal is always 1 in the
        solve coordinates.  ``0.0`` gives OLS.

        **Inference note.** When ``lambda_ > 0``, ``get_se()`` and
        ``get_tstat()`` estimate the sampling variability of the penalized
        estimator around the penalized pseudo-true parameter β_λ, not the
        unpenalized population coefficient β₀.  See ``get_se`` for details.
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
        so the weight vector cannot be precomputed. HAC standard errors use the
        same window weights as the coefficient estimator.
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
    estimate_every : int or str
        Estimate only selected endpoints. An integer counts backwards from the
        final observation; a pandas offset alias selects the last observation
        present in each period. Windows still contain the original observations,
        and ``hac_lags`` remains measured in observations, not cadence steps.
        Public ``get_*`` accessors return the full index with NaN at skipped
        endpoints; ``iter_beta()`` and ``iter_se()`` yield compact frames.

    Examples
    --------
    Basic usage — no controls:

    >>> ols = RollingOLS(window=252, mode="joint")
    >>> result = ols.fit(df[["f1", "f2"]]).transform(df[["AAPL", "MSFT"]])
    >>> result.get_beta("f1")      # DataFrame (T x N_assets)
    >>> result.get_signal("f1")
    >>> result.get_r2("f1")

    With controls and Ridge:

    >>> ols = RollingOLS(window=252, lambda_=1e-4, hac_lags=5, mode="joint")
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
        estimate_every: int | str = 1,
        mode: Literal["batched", "joint"] | None = None,
        warn_correlated_factors: bool = True,
    ) -> None:
        if mode not in ("batched", "joint", None):
            raise ValueError(f"mode must be 'batched', 'joint', or None, got {mode!r}")
        if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
            raise ValueError(f"window must be a positive integer, got {window!r}")
        if min_periods is not None and (
            not isinstance(min_periods, int) or isinstance(min_periods, bool) or min_periods <= 0
        ):
            raise ValueError(f"min_periods must be a positive integer, got {min_periods!r}")
        _resolved_min_periods = min_periods if min_periods is not None else window
        if not expanding and _resolved_min_periods > window:
            raise ValueError(
                f"min_periods={_resolved_min_periods!r} exceeds window={window!r} in rolling mode; "
                "use expanding=True or set min_periods <= window"
            )
        if ewma_halflife is not None and expanding:
            raise ValueError(
                "ewma_halflife cannot be combined with expanding=True: expanding "
                "windows have variable length, so the EWMA weight vector cannot "
                "be precomputed."
            )
        if ewma_halflife is not None and (
            not isinstance(ewma_halflife, int)
            or isinstance(ewma_halflife, bool)
            or ewma_halflife <= 0
        ):
            raise ValueError(f"ewma_halflife must be a positive integer, got {ewma_halflife!r}")
        if lambda_ < 0:
            raise ValueError(f"lambda_ must be non-negative, got {lambda_!r}")
        if hac_lags is not None and (
            not isinstance(hac_lags, int) or isinstance(hac_lags, bool) or hac_lags < 0
        ):
            raise ValueError(f"hac_lags must be a non-negative integer, got {hac_lags!r}")
        if not np.isfinite(cond_warn_threshold) or cond_warn_threshold <= 0:
            raise ValueError("cond_warn_threshold must be finite and positive")
        if not isinstance(cache_size, int) or isinstance(cache_size, bool):
            raise TypeError("cache_size must be an integer")
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        if isinstance(estimate_every, bool) or not isinstance(estimate_every, (int, str)):
            raise ValueError(f"estimate_every={estimate_every!r} must be a positive int or offset")
        if isinstance(estimate_every, int):
            if estimate_every < 1:
                raise ValueError(f"estimate_every={estimate_every!r} must be at least 1")
        else:
            try:
                pd.tseries.frequencies.to_offset(estimate_every)
            except ValueError as error:
                raise ValueError(
                    f"estimate_every={estimate_every!r} is not a valid pandas offset alias"
                ) from error

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
        self.estimate_every = estimate_every
        self.mode = mode
        self.warn_correlated_factors = warn_correlated_factors

        self._is_fitted = False
        self._factor_cols: list[str] = []
        self._control_cols: list[str] = []
        self._index: pd.Index | None = None
        self._factors: pd.DataFrame | None = None
        self._controls_fitted: pd.DataFrame | None = None

    def _estimation_positions(self, index: pd.Index) -> np.ndarray:
        """Return full-index positions retained by the configured cadence."""
        n_observations = len(index)
        if self.estimate_every == 1:
            return np.arange(n_observations, dtype=np.intp)
        first_valid = self.min_periods - 1
        if n_observations <= first_valid:
            return np.empty(0, dtype=np.intp)
        if isinstance(self.estimate_every, int):
            positions = np.arange(
                n_observations - 1,
                first_valid - 1,
                -self.estimate_every,
                dtype=np.intp,
            )
            return positions[::-1].copy()
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError(f"estimate_every={self.estimate_every!r} requires a DatetimeIndex")
        position_series = pd.Series(np.arange(n_observations, dtype=np.intp), index=index)
        try:
            period_endpoints = position_series.resample(self.estimate_every).last().dropna()
        except ValueError as error:
            raise ValueError(
                f"estimate_every={self.estimate_every!r} is not a valid pandas offset alias"
            ) from error
        positions = period_endpoints.to_numpy(dtype=np.intp)
        return positions[positions >= first_valid]

    def _weights(self) -> np.ndarray | None:
        """EWMA observation weights for one full window, or None for equal weights."""
        if self.ewma_halflife is None:
            return None
        return _ewma_weights(self.ewma_halflife, self.window)

    def _solver_endpoint_positions(self, index: pd.Index) -> np.ndarray | None:
        """Keep the default solver path untouched when every endpoint is requested."""
        if self.estimate_every == 1:
            return None
        return self._estimation_positions(index)

    def estimate_memory(
        self,
        targets: pd.DataFrame,
        factors: pd.DataFrame,
        controls: pd.DataFrame | None = None,
    ) -> dict[str, int | str]:
        """Estimate persistent and per-cache memory before fitting a panel."""
        n_observations = len(targets)
        n_stored_endpoints = len(self._estimation_positions(targets.index))
        n_targets = targets.shape[1]
        n_factors = factors.shape[1]
        n_controls = 0 if controls is None else controls.shape[1]
        float_bytes = np.dtype(np.float64).itemsize
        factor_frame_bytes = n_stored_endpoints * n_targets * n_factors * float_bytes
        target_frame_bytes = n_stored_endpoints * n_targets * float_bytes

        factor_validity = np.isfinite(factors.to_numpy(dtype=np.float64))
        if n_factors:
            packed = np.packbits(factor_validity, axis=0)
            n_factor_patterns = len(
                {packed[:, position].tobytes() for position in range(n_factors)}
            )
        else:
            n_factor_patterns = 0
        pattern_target_bytes = (
            n_stored_endpoints * n_targets * max(1, n_factor_patterns) * float_bytes
        )

        estimates: dict[str, int | str] = {
            "betas": factor_frame_bytes,
            "intercepts": factor_frame_bytes,
            "n_used": pattern_target_bytes,
            "pattern_statistics": 3 * pattern_target_bytes
            + n_stored_endpoints * n_factors * float_bytes,
            "retained_inputs": n_observations * (n_targets + n_factors + n_controls) * float_bytes,
            "on_demand_per_frame": target_frame_bytes,
            "on_demand_cache_bytes": self.cache_size * target_frame_bytes,
            "note": (
                "Each retained lazy quantity costs cache_size * n_selected * N * 8 bytes. "
                "Full-index get_* output is expanded transiently on access. "
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
                endpoint_positions=self._solver_endpoint_positions(targets.index),
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
            df_eff=(
                np.concatenate([fit.df_eff for fit in fits], axis=1)
                if all(fit.df_eff is not None for fit in fits)
                else None
            ),
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
        df_eff_values: np.ndarray | None = None,
    ) -> FactorStatistics:
        """Apply dof adjustment and wrap one factor's derived statistics.

        When ``df_eff_values`` is provided (Ridge or any path that computed
        effective dof), the residual degrees of freedom use it directly:
        ``n_eff - df_eff``.  For OLS paths that do not supply it the classic
        formula ``n_eff - (n_controls + 1) - fit_intercept`` is used; the two
        are numerically identical when ``lambda_ == 0``.
        """
        if df_eff_values is not None:
            residual_dof = n_eff_values - df_eff_values
        else:
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
        df_eff: np.ndarray | None = None,
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
            df_eff_values=df_eff,
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

        # FWL path always uses lambda_ == 0; df_eff equals the raw parameter
        # count.  Compute analytically so _finalize_statistics uses the same
        # formula as the joint path (no branch on lambda_).
        n_params = float(n_controls + 1 + int(self.fit_intercept))
        df_eff_values = np.where(np.isfinite(n_eff_values), n_params, np.nan)
        return self._finalize_statistics(
            r2_values,
            partial_r2_values,
            n_eff_values,
            n_used_values,
            assets,
            n_controls,
            df_eff_values=df_eff_values,
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

        n_factors_fit = len(self._factor_cols)
        if n_factors_fit > 1 and self.mode is None:
            raise ValueError(
                f"{n_factors_fit} factors were supplied but `mode` was not specified."
                " rOLS does not\nchoose an estimand for you when multiple factors"
                " are present:\n\n"
                '  mode="batched"  fits y ~ 1 + controls + factor_j separately for'
                " each factor.\n"
                "                  Each beta is conditional on the controls but NOT"
                " on the other\n"
                "                  factors, so correlated factors each absorb"
                " variation\n"
                "                  attributable to the others.\n\n"
                '  mode="joint"    fits y ~ 1 + controls + factor_1 + ... +'
                " factor_K once.\n"
                "                  Each beta is conditional on the controls AND"
                " every other\n"
                "                  factor.\n\n"
                "The two coincide when factors are mutually orthogonal on the"
                " estimation sample.\n"
                "Pass mode explicitly to continue."
            )

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
            control's joint rolling beta, accessible through
            result.get_control_beta(). In batched mode this is one joint fit
            per factor, so a control's beta can vary by which factor is
            named (see get_control_beta's docstring); in joint mode there is
            one fit shared by every factor, so it does not vary. More
            expensive in batched mode: one additional set of coefficients
            per factor rather than one shared set.

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
        denom_tol = self.denom_tol
        hac_lags = self.hac_lags
        warn_singular = self.warn_singular
        cond_warn_threshold = self.cond_warn_threshold
        asset_chunk_size = self.asset_chunk_size
        n_controls = len(self._control_cols)
        n_factors = len(self._factor_cols)
        # Resolve sentinel: mode=None is only allowed for single-factor calls
        # (guaranteed by fit()); resolve to "batched" for all downstream logic.
        effective_mode = self.mode if self.mode is not None else "batched"
        asset_positions = {asset: position for position, asset in enumerate(asset_cols)}
        factor_penalty = self._penalty_matrix(n_controls, n_factors=1)
        controls_penalty = self._penalty_matrix(n_controls, n_factors=0)
        estimated_positions = self._estimation_positions(assets.index)
        solver_endpoint_positions = self._solver_endpoint_positions(assets.index)
        stored_index = assets.index[estimated_positions]

        def selected_targets(selected_assets: Sequence[str] | None) -> pd.DataFrame:
            if selected_assets is None:
                return assets
            return assets.loc[:, list(selected_assets)]

        def stored_targets(selected_assets: Sequence[str] | None) -> pd.DataFrame:
            return selected_targets(selected_assets).iloc[estimated_positions]

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
                    endpoint_positions=solver_endpoint_positions,
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
                df_eff=(
                    np.concatenate([fit.df_eff for fit in fits], axis=1)
                    if all(fit.df_eff is not None for fit in fits)
                    else None
                ),
                n_singular=sum(fit.n_singular for fit in fits),
                n_ill_conditioned=sum(fit.n_ill_conditioned for fit in fits),
            )

        def factor_design(factor: str) -> pd.DataFrame:
            parts = [] if controls_snapshot is None else [controls_snapshot]
            return pd.concat([*parts, factors_snapshot[[factor]]], axis=1)

        # Warn once when batched mode is used with materially correlated factors.
        if effective_mode == "batched" and self.warn_correlated_factors and n_factors > 1:
            _warn_correlated_factors(factors_snapshot)

        # FWL shortcut applies only in batched mode (lambda_==0).
        fwl_fit: BatchedFitResult | None = None
        if effective_mode == "batched" and self.lambda_ == 0:
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
                endpoint_positions=solver_endpoint_positions,
            )

        # Joint mode: one solve with all factors together.
        joint_all_design: pd.DataFrame | None = None
        joint_all_penalty: np.ndarray | None = None
        joint_all_fit: JointFitResult | None = None
        if effective_mode == "joint":
            _parts: list[pd.DataFrame] = [] if controls_snapshot is None else [controls_snapshot]
            joint_all_design = pd.concat([*_parts, factors_snapshot], axis=1)
            joint_all_penalty = self._penalty_matrix(n_controls, n_factors)
            joint_all_fit = self._solve_targets(assets, joint_all_design, joint_all_penalty)

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
            estimated_positions=estimated_positions,
        )
        result._path = "fwl" if (effective_mode == "batched" and self.lambda_ == 0) else "joint"
        result.mode = effective_mode

        direct_control_betas: dict[str, dict[str, pd.DataFrame]] = {}
        for factor_position, fac in enumerate(self._factor_cols):
            if joint_all_fit is not None:
                # Joint mode: extract per-factor coefficient from the shared fit.
                beta_values = joint_all_fit.coef[:, n_controls + factor_position, :]
                intercept_values = joint_all_fit.intercept
                n_used_values = joint_all_fit.n_used
            elif fwl_fit is None:
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

            beta = pd.DataFrame(beta_values, index=stored_index, columns=assets.columns)
            intercept = pd.DataFrame(
                intercept_values,
                index=stored_index,
                columns=assets.columns,
            )
            result._betas[fac] = beta
            result._intercepts[fac] = intercept
            if fwl_fit is None:
                result._n_used[fac] = pd.DataFrame(
                    n_used_values,
                    index=stored_index,
                    columns=assets.columns,
                )
            result._factor_values[fac] = factors_snapshot[fac]
            if return_control_betas and controls_snapshot is not None:
                if joint_all_fit is not None:
                    # Controls are the first n_controls columns of the shared fit.
                    control_coef = joint_all_fit.coef
                elif fwl_fit is None:
                    control_coef = fit.coef
                else:
                    assert fwl_fit.nuisance_coef is not None
                    control_coef = fwl_fit.nuisance_coef[:, factor_position, :, :]
                direct_control_betas[fac] = {
                    control: pd.DataFrame(
                        control_coef[:, control_position, :],
                        index=stored_index,
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
                stored_targets(selected_assets),
                n_controls,
                df_eff=fit.df_eff,
            )

        if joint_all_fit is not None:
            # --- Joint mode statistics ---
            _joint_ssr = joint_all_fit.ssr
            _joint_sst = joint_all_fit.sst
            _joint_n_eff = joint_all_fit.n_eff
            _joint_n_used = joint_all_fit.n_used
            _joint_df_eff = joint_all_fit.df_eff  # effective dof (Ridge-aware)
            _factor_cols_snapshot = list(self._factor_cols)
            # Fallback integer dof for adj-R² when df_eff is unavailable:
            # (n_controls + K) slopes + intercept → same as the old n_controls trick.
            _n_controls_for_dof = n_controls + n_factors - 1
            # Complete-case mask for all factors (restricts the reduced model to
            # the same rows used by the full joint model).
            _joint_regressor_mask = factors_snapshot.notna().all(axis=1)

            def derive_joint_mode_statistics(
                factor: str,
                selected_assets: Sequence[str] | None,
            ) -> FactorStatistics:
                if selected_assets is None:
                    full_ssr = _joint_ssr
                    sst = _joint_sst
                    n_eff = _joint_n_eff
                    n_used = _joint_n_used
                    df_eff = _joint_df_eff
                else:
                    sel = [asset_positions[a] for a in selected_assets]
                    full_ssr = _joint_ssr[:, sel]
                    sst = _joint_sst[:, sel]
                    n_eff = _joint_n_eff[:, sel]
                    n_used = _joint_n_used[:, sel]
                    df_eff = _joint_df_eff[:, sel] if _joint_df_eff is not None else None

                # Reduced model for partial R²: all factors except this one,
                # restricted to the full model's complete-case rows so the two
                # R² values are comparable.
                other_factors = [f for f in _factor_cols_snapshot if f != factor]
                if not other_factors and controls_snapshot is None:
                    reduced_ssr = sst
                else:
                    _r_parts: list[pd.DataFrame] = (
                        [] if controls_snapshot is None else [controls_snapshot]
                    )
                    if other_factors:
                        _r_parts.append(factors_snapshot[other_factors])
                    reduced_design = pd.concat(_r_parts, axis=1)
                    reduced_penalty = self._penalty_matrix(n_controls, len(other_factors))
                    _targets = selected_targets(selected_assets)
                    reduced_fit = solve_snapshot(
                        _targets.where(_joint_regressor_mask, axis=0),
                        reduced_design,
                        reduced_penalty,
                    )
                    if selected_assets is None:
                        reduced_ssr = reduced_fit.ssr
                    else:
                        reduced_ssr = reduced_fit.ssr[:, sel]

                return self._statistics_from_arrays(
                    full_ssr,
                    sst,
                    reduced_ssr,
                    n_eff,
                    n_used,
                    stored_targets(selected_assets),
                    _n_controls_for_dof,
                    df_eff=df_eff,
                )

            result._statistics_loader = derive_joint_mode_statistics
        elif fwl_fit is not None:
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
                    assets=stored_targets(selected_assets),
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
            if joint_all_fit is not None:
                # Joint mode: all factors share one model; residuals are the same
                # regardless of which factor is queried.
                fit = solve_snapshot(targets, joint_all_design, joint_all_penalty)
                residual_values = fit.resid_endpoint
            elif fwl_fit is not None:
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
                    endpoint_positions=solver_endpoint_positions,
                )
                assert factor_fit.resid_endpoint is not None
                residual_values = factor_fit.resid_endpoint[:, 0, :]
            else:
                fit = solve_snapshot(targets, factor_design(factor), factor_penalty)
                residual_values = fit.resid_endpoint
            return pd.DataFrame(residual_values, index=stored_index, columns=targets.columns)

        def load_standard_errors(
            factor: str,
            selected_assets: Sequence[str] | None,
        ) -> pd.DataFrame:
            if hac_lags is None:
                raise RuntimeError("HAC standard errors require hac_lags to be set.")
            targets = selected_targets(selected_assets)
            if joint_all_fit is not None:
                # rolling_hac_se extracts the SE for the last design column, so
                # reorder to place the requested factor last.  All factors share
                # the same lambda_, so reordering does not affect the penalty.
                other_factors = [f for f in self._factor_cols if f != factor]
                _hac_parts: list[pd.DataFrame] = (
                    [] if controls_snapshot is None else [controls_snapshot]
                )
                if other_factors:
                    _hac_parts.append(factors_snapshot[other_factors])
                _hac_parts.append(factors_snapshot[[factor]])
                fac_last_design = pd.concat(_hac_parts, axis=1)
                hac_penalty = self._penalty_matrix(n_controls, n_factors)
                return rolling_hac_se(
                    y=targets,
                    X=fac_last_design,
                    window=window,
                    min_periods=min_periods,
                    expanding=expanding,
                    n_lags=hac_lags,
                    fit_intercept=fit_intercept,
                    penalty=hac_penalty,
                    weights=weights_snapshot,
                    denom_tol=denom_tol,
                    warn_invalid=warn_singular,
                    endpoint_positions=solver_endpoint_positions,
                )
            return rolling_hac_se(
                y=targets,
                X=factor_design(factor),
                window=window,
                min_periods=min_periods,
                expanding=expanding,
                n_lags=hac_lags,
                fit_intercept=fit_intercept,
                penalty=factor_penalty,
                weights=weights_snapshot,
                denom_tol=denom_tol,
                warn_invalid=warn_singular,
                endpoint_positions=solver_endpoint_positions,
            )

        def load_factor_adjusted_returns(
            selected_assets: Sequence[str] | None,
        ) -> pd.DataFrame:
            targets = selected_targets(selected_assets)
            if controls_snapshot is None:
                return targets.iloc[estimated_positions]
            controls_fit = solve_snapshot(targets, controls_snapshot, controls_penalty)
            return pd.DataFrame(
                controls_fit.resid_endpoint,
                index=stored_index,
                columns=targets.columns,
            )

        result._residual_loader = load_residuals
        result._se_loader = load_standard_errors
        result._factor_adjusted_loader = load_factor_adjusted_returns
        result._target_loader = stored_targets

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
