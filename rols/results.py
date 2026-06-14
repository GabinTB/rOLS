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
    # Control betas do not depend on the factor, so the inner dict is shared
    # across all factors. Populated only when transform(return_control_betas=True).
    _control_betas: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)

    # cache for HAC SE: {factor -> DataFrame(T x N_assets)}
    _se_cache:  Dict[str, pd.DataFrame] = field(default_factory=dict)

    # raw factor values needed for HAC: {factor -> Series}
    _factor_values: Dict[str, pd.Series] = field(default_factory=dict)

    # asset returns residualized against controls (FWL step 2): DataFrame(T x N_assets)
    _factor_adjusted_returns: Optional[pd.DataFrame] = field(default=None)

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

        This is the FWL step 3 output — asset returns with BOTH the controls and
        the narrative `factor` removed. For asset returns with only the controls
        removed (FWL step 2), use get_factor_adjusted_returns() instead.
        """
        self._check_factor(factor)
        return self._residuals[factor]

    def get_factor_adjusted_returns(self) -> pd.DataFrame:
        """
        Asset returns residualized against controls (FWL step 2 output).
        Shape: (T, N_assets).

        This is e_it = r_it - B_t' * ctrl_t — asset returns with the effect of
        control variables removed. It is NOT the same as get_residuals(factor),
        which further removes the narrative factor (FWL step 3).

        If no controls were provided at fit(), returns the original asset returns.
        """
        if self._factor_adjusted_returns is None:
            raise RuntimeError("factor_adjusted_returns not available.")
        return self._factor_adjusted_returns

    def get_control_beta(self, factor: str, control: str) -> pd.DataFrame:
        """
        Joint rolling beta of a control variable, recovered via Frisch-Waugh-Lovell.

        This is the control's coefficient from the full joint regression (each
        control partialled out against all other controls), NOT a univariate
        marginal beta. The value does not depend on `factor` — control betas are
        shared across all factors — but a factor name is required for a consistent
        getter signature and validation.

        Only available if `return_control_betas=True` was passed to transform().
        Shape: (T, N_assets).
        """
        self._check_factor(factor)
        if not self._control_betas:
            raise RuntimeError(
                "Control betas were not computed. Pass return_control_betas=True "
                "to transform() (and fit() with controls) to enable get_control_beta()."
            )
        if control not in self._control_betas[factor]:
            available = list(self._control_betas[factor].keys())
            raise KeyError(
                f"Control '{control}' not found. Available controls: {available}"
            )
        return self._control_betas[factor][control]

    # ------------------------------------------------------------------
    # Factor mimicking returns (cross-sectional use case)
    # ------------------------------------------------------------------

    def get_factor_mimicking_returns(self, factor: str) -> pd.Series:
        """
        Time series of the estimated factor mimicking return for `factor`.
        Shape: (T,) Series indexed by date.

        Only meaningful when rOLS is used in the cross-sectional direction:
          - factors = asset-level factor betas (N assets x K factors)
          - targets = asset returns (N assets x 1)

        In this case, the beta from get_beta(factor) is the cross-sectional
        regression coefficient lambda_t — the return per unit of factor exposure
        in the cross-section at each date t.

        This is the g_t series in a pure-factor mimicking portfolio framework.

        NOTE: requires transform() to have been called with a single target column.
        Raises RuntimeError if multiple target columns were used.
        """
        self._check_factor(factor)
        beta_df = self._betas[factor]
        if beta_df.shape[1] != 1:
            raise RuntimeError(
                "get_factor_mimicking_returns() requires transform() to be called "
                "with a single target column. "
                f"Got {beta_df.shape[1]} columns: {beta_df.columns.tolist()}"
            )
        return beta_df.iloc[:, 0].rename(factor)

    def get_all_factor_mimicking_returns(self) -> pd.DataFrame:
        """
        All factor mimicking return series as a (T, K) DataFrame.
        Column names match factor names from fit().
        See get_factor_mimicking_returns() for full documentation.
        """
        return pd.concat(
            {f: self.get_factor_mimicking_returns(f) for f in self.factor_cols},
            axis=1,
        )

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
