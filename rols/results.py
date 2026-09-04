"""Lazy result container for rolling OLS estimates."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
import pandas as pd

from .estimators import PatternSufficientStatistics

_CacheValue = TypeVar("_CacheValue")


@dataclass(frozen=True)
class FactorStatistics:
    """Derived model statistics for one factor."""

    r2: pd.DataFrame
    partial_r2: pd.DataFrame
    dof: pd.DataFrame
    n_used: pd.DataFrame


@dataclass
class RollingOLSResult:
    """Store primary estimates and derive secondary outputs on demand."""

    factor_cols: list[str]
    asset_cols: list[str]
    index: pd.Index
    lag_signal: bool
    window: int
    min_periods: int
    expanding: bool
    hac_lags: int | None
    mode: str = "batched"
    precision: str = "double"
    storage_dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float64))
    compute_dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float64))
    cache_size: int = 1
    estimated_positions: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.intp))

    # Primary outputs are ready when transform() returns.
    _betas: dict[str, pd.DataFrame] = field(default_factory=dict)
    _intercepts: dict[str, pd.DataFrame] = field(default_factory=dict)
    _n_used: dict[str, pd.DataFrame] = field(default_factory=dict)

    # Optional eager output requested explicitly by transform().
    _control_betas: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)

    # Inputs and compact statistics used by lazy accessors.
    _factor_values: dict[str, pd.Series] = field(default_factory=dict)
    _sufficient_statistics: tuple[PatternSufficientStatistics, ...] = field(default_factory=tuple)
    _statistics_loader: Callable[[str, Sequence[str] | None], FactorStatistics] | None = field(
        default=None,
        repr=False,
    )
    _residual_loader: Callable[[str, Sequence[str] | None], pd.DataFrame] | None = field(
        default=None, repr=False
    )
    _se_loader: Callable[[str, Sequence[str] | None], pd.DataFrame] | None = field(
        default=None, repr=False
    )
    _factor_adjusted_loader: Callable[[Sequence[str] | None], pd.DataFrame] | None = field(
        default=None, repr=False
    )
    _target_loader: Callable[[Sequence[str] | None], pd.DataFrame] | None = field(
        default=None, repr=False
    )

    # Every factor-sized lazy cache is bounded independently by cache_size.
    _statistics_cache: OrderedDict[str, FactorStatistics] = field(
        default_factory=OrderedDict,
        repr=False,
    )
    _residual_cache: OrderedDict[str, pd.DataFrame] = field(
        default_factory=OrderedDict,
        repr=False,
    )
    _se_cache: OrderedDict[str, pd.DataFrame] = field(default_factory=OrderedDict, repr=False)
    _factor_adjusted_returns: pd.DataFrame | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.cache_size, int) or isinstance(self.cache_size, bool):
            raise TypeError("cache_size must be an integer")
        if self.cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        positions = np.asarray(self.estimated_positions)
        if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
            raise ValueError("estimated_positions must be a one-dimensional integer array")
        self.estimated_positions = positions.astype(np.intp, copy=False)

    def _check_factor(self, factor: str) -> None:
        if factor not in self.factor_cols:
            raise KeyError(f"Factor '{factor}' not found. Available: {self.factor_cols}")

    @staticmethod
    def _select_assets(
        values: pd.DataFrame,
        assets: Sequence[str] | None,
    ) -> pd.DataFrame:
        if assets is None:
            return values
        return values.loc[:, list(assets)]

    def _full_index(self, values: pd.DataFrame) -> pd.DataFrame:
        """Expand a compact endpoint frame without retaining the padded copy."""
        return values if values.index.equals(self.index) else values.reindex(self.index)

    def _remember(
        self,
        cache: OrderedDict[str, _CacheValue],
        factor: str,
        value: _CacheValue,
    ) -> _CacheValue:
        cache[factor] = value
        cache.move_to_end(factor)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)
        return value

    def _statistics_for(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> FactorStatistics:
        self._check_factor(factor)
        cached = self._statistics_cache.get(factor)
        if cached is not None:
            self._statistics_cache.move_to_end(factor)
            if assets is None:
                return cached
            return FactorStatistics(
                r2=self._select_assets(cached.r2, assets),
                partial_r2=self._select_assets(cached.partial_r2, assets),
                dof=self._select_assets(cached.dof, assets),
                n_used=self._select_assets(cached.n_used, assets),
            )
        if self._statistics_loader is None:
            raise RuntimeError("Derived statistics are not available.")
        statistics = self._statistics_loader(factor, assets)
        if assets is not None:
            return statistics
        return self._remember(self._statistics_cache, factor, statistics)

    def get_beta(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Coefficient on ``factor`` from the model selected by ``mode``.

        In batched mode (the default), this is ``factor``'s coefficient from
        ``y ~ 1 + controls + factor`` — conditional on the controls, not on
        the other factors. In joint mode it is ``factor``'s coefficient from
        the single model containing every factor. Computed eagerly at
        ``transform()``, not cached lazily. NaN at any endpoint where fewer
        than ``min_periods`` complete-case rows are available.
        """
        self._check_factor(factor)
        return self._full_index(self._select_assets(self._betas[factor], assets))

    def get_intercept(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Intercept from the same fit as ``get_beta(factor)``.

        Zero-filled (not NaN) when ``fit_intercept=False`` was used, since a
        through-origin model has no intercept term to be missing. Computed
        eagerly at ``transform()``; NaN wherever ``get_beta(factor)`` is NaN.
        """
        self._check_factor(factor)
        return self._full_index(self._select_assets(self._intercepts[factor], assets))

    def get_signal(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Factor term in the fitted model, derived from beta and factor values.

        With sparse cadence and ``lag_signal=True``, a beta estimated at endpoint
        ``t`` contributes to the signal at observation ``t + 1``. The signal's
        non-missing positions are therefore shifted one observation beyond the
        estimated endpoint positions, preserving the existing no-lookahead lag.
        """
        self._check_factor(factor)
        beta = self.get_beta(factor, assets)
        factor_values = self._factor_values[factor]
        return (
            beta.shift(1).mul(factor_values, axis=0)
            if self.lag_signal
            else beta.mul(factor_values, axis=0)
        )

    def get_r2(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """R² of the full model containing ``factor``, on its complete-case sample.

        In batched mode this is ``y ~ 1 + controls + factor``'s R²; in joint
        mode it is the single all-factors model's R², identical across every
        ``factor`` argument. Reports the adjusted statistic instead when
        ``adj_r2=True`` was passed to the constructor. Derived lazily and
        cached per factor (bounded by ``cache_size``). NaN wherever SST is at
        or below ``denom_tol`` (e.g. a constant target) or the adjusted
        statistic's residual degrees of freedom are not positive.
        """
        return self._full_index(self._statistics_for(factor, assets).r2)

    def get_partial_r2(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Incremental R² that ``factor`` contributes over the model without it.

        ``(SSR_reduced - SSR_full) / SSR_reduced``, both sums of squares
        computed on the full model's complete-case sample so the two models
        are compared on identical data. This is the informative number when
        factors are correlated — unlike ``get_r2``, it isolates one factor's
        marginal contribution rather than reporting the whole model's fit.
        Adjusted when ``adj_r2=True``. Derived lazily and cached per factor.
        """
        return self._full_index(self._statistics_for(factor, assets).partial_r2)

    def get_fitted_values(
        self,
        factor: str | None = None,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Endpoint fitted value from the model.

        Evaluates the contemporaneous fit: ``ŷ_t = α̂_t + c_t^T γ̂_t + f_t^T β̂_t``.
        Satisfies the invariant ``y = ŷ + ê`` regardless of ``lag_signal``.

        Parameters
        ----------
        factor:
            Factor name. In batched mode, selects which factor's model to use
            (required if there is more than one factor). In joint mode, ignored
            because there is only one joint model.
        assets:
            Optional subset of asset columns to return.
        """
        if self.mode == "joint":
            if factor is not None:
                import warnings

                warnings.warn(
                    "The `factor` argument to get_fitted_values() is ignored in joint mode.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            factor_to_load = self.factor_cols[0] if self.factor_cols else None
        else:
            if factor is None:
                if len(self.factor_cols) > 1:
                    raise ValueError(
                        "get_fitted_values() requires a `factor` argument in batched mode "
                        "when there is more than one factor."
                    )
                factor_to_load = self.factor_cols[0] if self.factor_cols else None
            else:
                factor_to_load = factor

        if self._target_loader is None:
            raise RuntimeError("Fitted values are not available.")

        y = self._full_index(self._target_loader(assets))
        # get_residuals requires a valid string. If factor_to_load is None, let it raise KeyError normally.
        # factor_to_load shouldn't be None unless factor_cols is empty, which is caught by _check_factor.
        return y - self.get_residuals(factor_to_load, assets)

    def get_residuals(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Endpoint residual ``y_t - fitted_t`` from the model containing ``factor``.

        One residual per endpoint — the current window's own fit evaluated
        at its own last row, not a full in-window residual series. NaN
        wherever ``get_beta(factor)`` is NaN. Recomputed lazily per factor and
        held in a bounded cache (``cache_size``); requesting an evicted factor
        recomputes it rather than raising.
        """
        self._check_factor(factor)
        residuals = self._residual_cache.get(factor)
        if residuals is not None:
            self._residual_cache.move_to_end(factor)
            return self._full_index(self._select_assets(residuals, assets))
        if self._residual_loader is None:
            raise RuntimeError("Residuals are not available.")
        residuals = self._residual_loader(factor, assets)
        if assets is not None:
            return self._full_index(residuals)
        compact = self._remember(self._residual_cache, factor, residuals)
        return self._full_index(compact)

    def get_dof(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Residual degrees of freedom for the model containing ``factor``.

        ``n_eff - df_eff``, where ``n_eff`` is the effective sample size
        (equals ``n_used`` under equal weighting, smaller under EWMA) and
        ``df_eff = tr[G (G + P)^{-1}]`` is the effective model degrees of
        freedom.  For OLS (``lambda_ == 0``), ``df_eff`` equals the raw
        parameter count (slopes + intercept) so this is identical to the
        classical ``n_eff - p - fit_intercept``.  For Ridge, ``df_eff < p``
        and the shrinkage is reflected in a larger residual dof.
        Derived lazily and cached per factor.
        """
        return self._full_index(self._statistics_for(factor, assets).dof)

    def get_n_used(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Complete-case row count in the window backing ``get_beta(factor)``.

        Counts raw observations, not the EWMA-weighted effective sample size
        (see ``get_dof`` for that). NaN wherever no estimate was produced.
        """
        self._check_factor(factor)
        if factor in self._n_used:
            return self._full_index(self._select_assets(self._n_used[factor], assets))
        return self._full_index(self._statistics_for(factor, assets).n_used)

    def get_factor_adjusted_returns(
        self,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return controls-residualized targets, computed on first access."""
        if self._factor_adjusted_returns is not None:
            return self._full_index(self._select_assets(self._factor_adjusted_returns, assets))
        if self._factor_adjusted_loader is None:
            raise RuntimeError("factor_adjusted_returns not available.")
        values = self._factor_adjusted_loader(assets)
        if assets is None:
            self._factor_adjusted_returns = values
        return self._full_index(values)

    def get_control_beta(
        self,
        factor: str,
        control: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Rolling coefficient of *control* from the joint model that includes *factor*.

        In batched mode each factor defines a separate model::

            y  ~  1  +  controls  +  factor

        so the coefficient on *control* is the coefficient from the joint model
        that also contains *factor*.  When *factor* and *control* are correlated,
        this value **varies across factors** — it is not a property of the control
        alone.  (When every factor is orthogonal to every control within each
        window, the values coincide across factors, but that is the special case,
        not the rule.)

        The coefficient is read directly from the joint solve; the ``return_control_betas``
        flag gates storage (a DataFrame per factor × control pair), not computation.

        Parameters
        ----------
        factor:
            Factor name; must be one of the columns passed to ``transform()``.
        control:
            Control name; must be one of the controls columns.
        assets:
            Optional subset of asset columns to return.

        Returns
        -------
        pd.DataFrame
            Shape ``(T, N_assets)``.  Rows before the first complete window
            are ``NaN``.
        """
        self._check_factor(factor)
        if not self._control_betas:
            raise RuntimeError(
                "Control betas were not computed. Pass return_control_betas=True "
                "to transform() (and fit() with controls) to enable get_control_beta()."
            )
        if control not in self._control_betas[factor]:
            available = list(self._control_betas[factor])
            raise KeyError(f"Control '{control}' not found. Available controls: {available}")
        return self._full_index(self._select_assets(self._control_betas[factor][control], assets))

    def _standard_errors_for(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return compact HAC output, using the bounded LRU cache."""
        self._check_factor(factor)
        if self.hac_lags is None:
            raise RuntimeError(
                "HAC standard errors require hac_lags to be set on RollingOLS. "
                "e.g. RollingOLS(window=252, hac_lags=5)"
            )
        standard_errors = self._se_cache.get(factor)
        if standard_errors is not None:
            self._se_cache.move_to_end(factor)
            return self._select_assets(standard_errors, assets)
        if self._se_loader is None:
            raise RuntimeError("HAC standard errors are not available.")
        standard_errors = self._se_loader(factor, assets)
        if assets is not None:
            return standard_errors
        return self._remember(self._se_cache, factor, standard_errors)

    def get_se(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Newey-West HAC standard errors, with NaN at skipped endpoints.

        The sandwich estimates the sampling variability of the fixed-penalty
        estimator **β̂_λ** around the penalized pseudo-true parameter **β_λ**.
        It does **not** correct regularization bias relative to the unpenalized
        population coefficient β₀, and it treats the penalty as fixed rather
        than estimated or selected from the data.  Intervals should not be read
        as nominal-coverage confidence intervals for β₀, and selecting
        ``lambda_`` by cross-validation or any data-driven procedure invalidates
        the nominal level further.  For OLS (``lambda_ == 0``), β_λ = β₀ and
        this distinction collapses.

        **Time-series only.** The sandwich corrects for serial correlation in
        each asset's own residuals.  It does not correct for cross-sectional
        dependence across assets.  Aggregating per-asset t-statistics in a
        correlated panel overstates significance; see
        ``docs/reference/specification.md`` §13 for panel-robust alternatives.

        Requires ``hac_lags`` to be set on the constructor (raises otherwise).
        Computation is lazy and streams one endpoint at a time.
        """
        return self._full_index(self._standard_errors_for(factor, assets))

    def get_tstat(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """HAC t-statistic: ``beta / se``, with NaN where undefined.

        See ``get_se`` for the estimand and the time-series-only caveat.
        The t-statistic for Ridge should not be compared against OLS critical
        values without accounting for the bias of β̂_λ relative to β₀.
        """
        tstat = self.get_beta(factor, assets).div(self.get_se(factor, assets))
        return tstat.replace([float("inf"), float("-inf")], float("nan"))

    def iter_beta(
        self,
        assets: Sequence[str] | None = None,
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        """Yield compact betas at computed endpoints for high-throughput iteration."""
        for factor in self.factor_cols:
            yield factor, self._select_assets(self._betas[factor], assets)

    def iter_se(
        self,
        assets: Sequence[str] | None = None,
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        """Compute and yield compact HAC SEs at computed endpoints."""
        for factor in self.factor_cols:
            yield factor, self._standard_errors_for(factor, assets)

    def to_long(self, factor: str, include_se: bool = False) -> pd.DataFrame:
        """Return beta, signal, and R² for one factor in long format."""
        self._check_factor(factor)
        out = pd.DataFrame(
            {
                "beta": self.get_beta(factor).stack(),
                "signal": self.get_signal(factor).stack(),
                "r2": self.get_r2(factor).stack(),
            }
        )
        out.index.names = ["date", "asset"]
        if include_se:
            out["se"] = self.get_se(factor).stack()
            out["t_stat"] = self.get_tstat(factor).stack()
        return out.reset_index()

    def to_long_all(self, include_se: bool = False) -> pd.DataFrame:
        """Return long-format output for every factor."""
        parts = []
        for factor in self.factor_cols:
            values = self.to_long(factor, include_se=include_se)
            values.insert(2, "factor", factor)
            parts.append(values)
        return pd.concat(parts, ignore_index=True)
