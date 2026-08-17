"""Lazy result container for rolling OLS estimates."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

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
    cache_size: int = 1

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
        """Rolling beta, optionally restricted to selected assets."""
        self._check_factor(factor)
        return self._select_assets(self._betas[factor], assets)

    def get_intercept(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Rolling intercept for the model containing ``factor``."""
        self._check_factor(factor)
        return self._select_assets(self._intercepts[factor], assets)

    def get_signal(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Factor term in the fitted model, derived from beta and factor values."""
        self._check_factor(factor)
        beta = self._select_assets(self._betas[factor], assets)
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
        """Full-model rolling R², or adjusted R²."""
        return self._statistics_for(factor, assets).r2

    def get_partial_r2(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Factor partial R², or its adjusted form."""
        return self._statistics_for(factor, assets).partial_r2

    def get_residuals(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Endpoint residuals, recomputed per factor and held in a bounded cache."""
        self._check_factor(factor)
        residuals = self._residual_cache.get(factor)
        if residuals is not None:
            self._residual_cache.move_to_end(factor)
            return self._select_assets(residuals, assets)
        if self._residual_loader is None:
            raise RuntimeError("Residuals are not available.")
        residuals = self._residual_loader(factor, assets)
        if assets is not None:
            return residuals
        return self._remember(self._residual_cache, factor, residuals)

    def get_dof(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Residual degrees of freedom based on effective sample size."""
        return self._statistics_for(factor, assets).dof

    def get_n_used(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Complete-case observation count for the model containing ``factor``."""
        self._check_factor(factor)
        if factor in self._n_used:
            return self._select_assets(self._n_used[factor], assets)
        return self._statistics_for(factor, assets).n_used

    def get_factor_adjusted_returns(
        self,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return controls-residualized targets, computed on first access."""
        if self._factor_adjusted_returns is not None:
            return self._select_assets(self._factor_adjusted_returns, assets)
        if self._factor_adjusted_loader is None:
            raise RuntimeError("factor_adjusted_returns not available.")
        values = self._factor_adjusted_loader(assets)
        if assets is None:
            self._factor_adjusted_returns = values
        return values

    def get_control_beta(
        self,
        factor: str,
        control: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Joint rolling beta of a control requested during transform()."""
        self._check_factor(factor)
        if not self._control_betas:
            raise RuntimeError(
                "Control betas were not computed. Pass return_control_betas=True "
                "to transform() (and fit() with controls) to enable get_control_beta()."
            )
        if control not in self._control_betas[factor]:
            available = list(self._control_betas[factor])
            raise KeyError(f"Control '{control}' not found. Available controls: {available}")
        return self._select_assets(self._control_betas[factor][control], assets)

    def get_factor_mimicking_returns(self, factor: str) -> pd.Series:
        """Return the single-target beta series for cross-sectional use."""
        beta = self.get_beta(factor)
        if beta.shape[1] != 1:
            raise RuntimeError(
                "get_factor_mimicking_returns() requires transform() to be called "
                "with a single target column. "
                f"Got {beta.shape[1]} columns: {beta.columns.tolist()}"
            )
        return beta.iloc[:, 0].rename(factor)

    def get_all_factor_mimicking_returns(self) -> pd.DataFrame:
        """Return all single-target factor mimicking series."""
        return pd.concat(
            {factor: self.get_factor_mimicking_returns(factor) for factor in self.factor_cols},
            axis=1,
        )

    def get_se(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Newey-West HAC standard errors held in a bounded LRU cache."""
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

    def get_tstat(
        self,
        factor: str,
        assets: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """HAC t-statistic, derived as beta divided by standard error."""
        tstat = self.get_beta(factor, assets).div(self.get_se(factor, assets))
        return tstat.replace([float("inf"), float("-inf")], float("nan"))

    def iter_beta(
        self,
        assets: Sequence[str] | None = None,
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        """Yield each factor's beta without materialising a combined panel."""
        for factor in self.factor_cols:
            yield factor, self.get_beta(factor, assets)

    def iter_se(
        self,
        assets: Sequence[str] | None = None,
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        """Compute and yield one factor's HAC standard errors at a time."""
        for factor in self.factor_cols:
            yield factor, self.get_se(factor, assets)

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
