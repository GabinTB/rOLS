"""
rOLS result container
==========

Result container returned by RollingOLS.transform().

All DataFrames stored internally are (T x N_assets).
Getters return views — no copying.
HAC standard errors and t-stats are computed on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd

from .estimators import hac_se


@dataclass
class RollingOLSResult:
    """
    Stores fitted results for all factors after transform().

    Parameters
    ----------
    factor_cols  : list of factor names (same order as fit())
    asset_cols   : list of asset names (same order as transform())
    index        : time index
    lag_signal   : whether signals use beta_{t-1} * factor_t
    window       : rolling window (needed for on-demand HAC)
    min_periods  : minimum periods (needed for on-demand HAC)
    expanding    : expanding window flag (needed for on-demand HAC)
    hac_lags     : number of Newey-West lags; None means HAC unavailable
    """
    factor_cols: List[str]
    asset_cols:  List[str]
    index:       pd.Index
    lag_signal:  bool
    window:      int
    min_periods: int
    expanding:   bool
    hac_lags:    Optional[int]

    # {factor -> DataFrame(T x N_assets)}
    _betas:     Dict[str, pd.DataFrame] = field(default_factory=dict)
    _signals:   Dict[str, pd.DataFrame] = field(default_factory=dict)
    _r2:        Dict[str, pd.DataFrame] = field(default_factory=dict)
    _residuals: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # {factor -> {control -> DataFrame(T x N_assets)}}
    _control_betas: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)

    # cache for HAC SE: {factor -> DataFrame(T x N_assets)}
    _se_cache:  Dict[str, pd.DataFrame] = field(default_factory=dict)

    # raw factor values needed for HAC: {factor -> Series}
    _factor_values: Dict[str, pd.Series] = field(default_factory=dict)

    # validate factor name and raise KeyError if not found
    def _check_factor(self, factor: str) -> None:
        if factor not in self.factor_cols:
            raise KeyError(
                f"Factor '{factor}' not found. Available: {self.factor_cols}"
            )

    # ------------------------------------------------------------------
    # Core getters
    # ------------------------------------------------------------------

    def get_beta(self, factor: str) -> pd.DataFrame:
        """Rolling beta for all assets. Shape: (T, N_assets)."""
        self._check_factor(factor)
        return self._betas[factor]

    def get_signal(self, factor: str) -> pd.DataFrame:
        """
        Factor signal: beta_t * factor_t, or beta_{t-1} * factor_t if lag_signal=True.
        Shape: (T, N_assets).
        """
        self._check_factor(factor)
        return self._signals[factor]

    def get_r2(self, factor: str) -> pd.DataFrame:
        """Rolling R² (or adjusted R²) for all assets. Shape: (T, N_assets)."""
        self._check_factor(factor)
        return self._r2[factor]

    def get_residuals(self, factor: str) -> pd.DataFrame:
        """
        Rolling regression residuals: y_resid_t - beta_t * f_resid_t.
        Shape: (T, N_assets). Used internally for HAC.
        """
        self._check_factor(factor)
        return self._residuals[factor]

    def get_control_beta(self, factor: str, control: str) -> pd.DataFrame:
        """
        Beta of a control variable from the joint regression.
        Only available if return_control_betas=True was passed to transform().
        Shape: (T, N_assets).
        """
        self._check_factor(factor)
        if factor not in self._control_betas or control not in self._control_betas[factor]:
            raise KeyError(
                f"Control beta for ('{factor}', '{control}') not found. "
                "Pass return_control_betas=True to transform()."
            )
        return self._control_betas[factor][control]

    # ------------------------------------------------------------------
    # HAC standard errors (on demand)
    # ------------------------------------------------------------------

    def get_se(self, factor: str) -> pd.DataFrame:
        """
        Newey-West HAC standard errors for beta estimates.
        Shape: (T, N_assets).

        Requires hac_lags to be set on RollingOLS constructor.
        Results are cached after first call.
        """
        self._check_factor(factor)
        if self.hac_lags is None:
            raise RuntimeError(
                "HAC standard errors require hac_lags to be set on RollingOLS. "
                "e.g. RollingOLS(window=252, hac_lags=5)"
            )
        if factor not in self._se_cache:
            self._se_cache[factor] = hac_se(
                residuals=self._residuals[factor],
                factor_values=self._factor_values[factor],
                window=self.window,
                min_periods=self.min_periods,
                expanding=self.expanding,
                n_lags=self.hac_lags,
            )
        return self._se_cache[factor]

    def get_tstat(self, factor: str) -> pd.DataFrame:
        """
        HAC t-statistics: beta / SE. Shape: (T, N_assets).
        Requires hac_lags to be set on RollingOLS constructor.
        """
        self._check_factor(factor)
        se = self.get_se(factor)
        return self._betas[factor].div(se)

    # ------------------------------------------------------------------
    # Long-format output
    # ------------------------------------------------------------------

    def to_long(self, factor: str, include_se: bool = False) -> pd.DataFrame:
        """
        Long-format results for a single factor.

        Parameters
        ----------
        factor     : factor name
        include_se : if True, also include se and t_stat columns
                     (requires hac_lags to be set)

        Returns
        -------
        pd.DataFrame with columns: date, asset, beta, signal, r2
        (plus se, t_stat if include_se=True)
        """
        self._check_factor(factor)
        beta   = self._betas[factor].stack()
        signal = self._signals[factor].stack()
        r2     = self._r2[factor].stack()
        out = pd.DataFrame({"beta": beta, "signal": signal, "r2": r2})
        out.index.names = ["date", "asset"]

        if include_se:
            se     = self.get_se(factor).stack()
            tstat  = self.get_tstat(factor).stack()
            out["se"]     = se
            out["t_stat"] = tstat

        return out.reset_index()

    def to_long_all(self, include_se: bool = False) -> pd.DataFrame:
        """
        Long-format results for all factors combined.

        Returns
        -------
        pd.DataFrame with columns: date, asset, factor, beta, signal, r2
        (plus se, t_stat if include_se=True)
        """
        parts = []
        for fac in self.factor_cols:
            df = self.to_long(fac, include_se=include_se)
            df.insert(2, "factor", fac)
            parts.append(df)
        return pd.concat(parts, ignore_index=True)
