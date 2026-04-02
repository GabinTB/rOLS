"""
rOLS estimator functions
=============

Low-level rolling estimators. All functions operate on numpy arrays
or pandas DataFrames and are independent of the model class.

Functions
---------
rolling_residualize   : rolling OLS/Ridge residualization (Frisch-Waugh step)
rolling_gram_schmidt  : rolling Gram-Schmidt orthogonalization within a group
hac_se                : Newey-West HAC standard errors from residuals
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Zero-copy sliding window view: (T, d) -> (T - window + 1, window, d).
    Uses stride tricks — do not write to the output array.
    """
    T, d = arr.shape
    n = T - window + 1
    shape   = (n, window, d)
    strides = (arr.strides[0], arr.strides[0], arr.strides[1])
    return as_strided(arr, shape=shape, strides=strides)


def _solve_batch(XtX: np.ndarray, XtY: np.ndarray) -> np.ndarray:
    """
    Batch solve XtX[i] @ beta[i] = XtY[i].
    Falls back element-wise on singular windows.
    Returns betas with NaN where solve failed.
    """
    n, k, N = XtY.shape
    betas = np.full((n, k, N), np.nan)
    try:
        betas = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        for i in range(n):
            try:
                betas[i] = np.linalg.solve(XtX[i], XtY[i])
            except np.linalg.LinAlgError:
                pass
    return betas


# ---------------------------------------------------------------------------
# Rolling OLS / Ridge residualization
# ---------------------------------------------------------------------------

def rolling_residualize(
    y: pd.DataFrame,
    X: pd.DataFrame,
    window: int,
    min_periods: int,
    expanding: bool,
    ridge_lambda: float = 0.0,
) -> pd.DataFrame:
    """
    Compute rolling OLS (or Ridge) residuals: y_t - X_t @ beta_t for each t.

    Ridge adds lambda * I to X'X before solving, shrinking betas toward zero.
    Set ridge_lambda=0.0 for standard OLS (default).

    Rolling window is fully vectorized via stride tricks.
    Expanding window uses a loop (variable window size precludes stride tricks).

    Parameters
    ----------
    y            : (T, N) DataFrame — targets
    X            : (T, k) DataFrame — regressors
    window       : rolling window length
    min_periods  : minimum observations to produce a result
    expanding    : use expanding window instead of rolling
    ridge_lambda : Ridge regularization strength (0.0 corresponds to OLS)

    Returns
    -------
    pd.DataFrame, same shape/index/columns as y
    """
    y_np = y.to_numpy(dtype=np.float64)
    X_np = X.to_numpy(dtype=np.float64)
    T, N = y_np.shape
    k    = X_np.shape[1]
    resid = np.full((T, N), np.nan)
    ridge_term = ridge_lambda * np.eye(k)

    if expanding:
        for t in range(min_periods - 1, T):
            Xw, yw = X_np[:t + 1], y_np[:t + 1]
            if np.isnan(Xw).any() or np.isnan(yw).any():
                continue
            XtX = Xw.T @ Xw + ridge_term
            XtY = Xw.T @ yw
            try:
                resid[t] = y_np[t] - X_np[t] @ np.linalg.solve(XtX, XtY)
            except np.linalg.LinAlgError:
                pass
    else:
        n_windows = T - window + 1
        if n_windows <= 0:
            return pd.DataFrame(resid, index=y.index, columns=y.columns)

        Xw = _make_windows(X_np, window)  # (n, window, k)
        yw = _make_windows(y_np, window)  # (n, window, N)

        has_nan = np.isnan(Xw).any(axis=(1, 2)) | np.isnan(yw).any(axis=(1, 2))
        valid   = ~has_nan

        XtX = np.einsum('twi,twj->tij', Xw, Xw)           # (n, k, k)
        XtX[valid] += ridge_term                            # Ridge: add λI
        XtY = np.einsum('twi,twn->tin', Xw, yw)            # (n, k, N)

        betas = np.full((n_windows, k, N), np.nan)
        if valid.any():
            betas[valid] = _solve_batch(XtX[valid], XtY[valid])

        t_idx  = np.arange(n_windows) + window - 1
        fitted = np.einsum('ti,tin->tn', X_np[t_idx], betas)
        resid[t_idx] = np.where(has_nan[:, None], np.nan, y_np[t_idx] - fitted)

        # Handle min_periods < window — fill early windows with a loop
        if min_periods < window:
            for t in range(min_periods - 1, window - 1):
                Xw_t, yw_t = X_np[:t + 1], y_np[:t + 1]
                if np.isnan(Xw_t).any() or np.isnan(yw_t).any():
                    continue
                XtX_t = Xw_t.T @ Xw_t + ridge_term
                try:
                    resid[t] = y_np[t] - X_np[t] @ np.linalg.solve(XtX_t, Xw_t.T @ yw_t)
                except np.linalg.LinAlgError:
                    pass

    return pd.DataFrame(resid, index=y.index, columns=y.columns)


# ---------------------------------------------------------------------------
# Rolling Gram-Schmidt orthogonalization
# ---------------------------------------------------------------------------

def rolling_gram_schmidt(
    X: pd.DataFrame,
    window: int,
    min_periods: int,
    expanding: bool,
) -> pd.DataFrame:
    """
    Rolling Gram-Schmidt orthogonalization within a group of regressors.

    At each time t, fits a rolling window and orthogonalizes X[:, j] against
    X[:, 0:j] using their rolling covariance structure. Column order determines
    priority: the first column is untouched, subsequent columns are orthogonalized
    against all previous ones.

    This means each column's residual represents incremental variation
    beyond the higher-priority columns — useful when columns have a natural
    importance ordering (e.g. evergreen narratives first, transient ones last).

    Parameters
    ----------
    X           : (T, k) DataFrame of regressors
    window      : rolling window length
    min_periods : minimum observations
    expanding   : use expanding window

    Returns
    -------
    pd.DataFrame, same shape/index/columns as X
    """
    cols = X.columns.tolist()
    if len(cols) == 1:
        return X.copy()  # single column: nothing to orthogonalize

    result = X.astype(np.float64).copy()

    # Sequentially orthogonalize column j against columns 0..j-1
    # by regressing column j on the already-orthogonalized predecessors
    for j in range(1, len(cols)):
        y   = result[[cols[j]]]       # (T, 1) — current column
        Xprev = result[cols[:j]]      # (T, j) — already orthogonalized predecessors

        resid = rolling_residualize(
            y=y,
            X=Xprev,
            window=window,
            min_periods=min_periods,
            expanding=expanding,
            ridge_lambda=0.0,
        )
        # Where residualization produced NaN (warm-up), keep original values
        result[cols[j]] = resid[cols[j]].fillna(X[cols[j]])

    return result


# ---------------------------------------------------------------------------
# HAC (Newey-West) standard errors
# ---------------------------------------------------------------------------

def hac_se(
    residuals: pd.DataFrame,
    factor_values: pd.Series,
    window: int,
    min_periods: int,
    expanding: bool,
    n_lags: int,
) -> pd.DataFrame:
    """
    Newey-West HAC standard errors for rolling univariate OLS betas.

    For each asset and each time t, computes SE(beta_t) using the residuals
    within the rolling window, corrected for autocorrelation up to n_lags.

    The sandwich estimator is:
        Var(beta) = (X'X)^{-1} * S * (X'X)^{-1}
    where S is the Newey-West long-run variance of X * eps.

    Parameters
    ----------
    residuals     : (T, N) DataFrame — regression residuals per asset
    factor_values : (T,) Series — the factor (regressor) values
    window        : rolling window length
    min_periods   : minimum observations
    expanding     : use expanding window
    n_lags        : number of lags for Newey-West (typically floor(T^(1/3)))

    Returns
    -------
    pd.DataFrame of standard errors, same shape as residuals
    """
    resid_np  = residuals.to_numpy(dtype=np.float64)   # (T, N)
    f_np      = factor_values.to_numpy(dtype=np.float64)  # (T,)
    T, N      = resid_np.shape
    se        = np.full((T, N), np.nan)

    def _nw_se_window(f_w: np.ndarray, e_w: np.ndarray) -> np.ndarray:
        """
        Newey-West SE for one window.
        f_w : (n_obs,) factor values
        e_w : (n_obs, N) residuals
        Returns SE vector of shape (N,).
        """
        n_obs = len(f_w)
        # score: (n_obs, N) — x_t * eps_t
        score = f_w[:, None] * e_w                  # (n_obs, N)
        xx    = f_w @ f_w                            # scalar: X'X

        # Newey-West long-run variance of score
        # S = Gamma(0) + sum_{l=1}^{L} w_l * (Gamma(l) + Gamma(l)')
        # For univariate x: S is scalar per asset
        S = np.einsum('ti,ti->i', score, score) / n_obs  # Gamma(0), shape (N,)

        for lag in range(1, n_lags + 1):
            w = 1.0 - lag / (n_lags + 1)            # Bartlett weight
            gamma = np.einsum('ti,ti->i', score[lag:], score[:-lag]) / n_obs
            S += 2 * w * gamma

        # Var(beta) = S / (X'X)^2 * n_obs  (sandwich, scaled)
        var_beta = S * n_obs / (xx ** 2)
        return np.sqrt(np.maximum(var_beta, 0.0))

    if expanding:
        for t in range(min_periods - 1, T):
            f_w = f_np[:t + 1]
            e_w = resid_np[:t + 1]
            if np.isnan(f_w).any() or np.isnan(e_w).any():
                continue
            if len(f_w) <= n_lags:
                continue
            se[t] = _nw_se_window(f_w, e_w)
    else:
        for t in range(window - 1, T):
            start = t - window + 1
            f_w   = f_np[start:t + 1]
            e_w   = resid_np[start:t + 1]
            if np.isnan(f_w).any() or np.isnan(e_w).any():
                continue
            if len(f_w) <= n_lags:
                continue
            se[t] = _nw_se_window(f_w, e_w)

        # Handle min_periods < window
        if min_periods < window:
            for t in range(min_periods - 1, window - 1):
                f_w = f_np[:t + 1]
                e_w = resid_np[:t + 1]
                if np.isnan(f_w).any() or np.isnan(e_w).any():
                    continue
                if len(f_w) <= n_lags:
                    continue
                se[t] = _nw_se_window(f_w, e_w)

    return pd.DataFrame(se, index=residuals.index, columns=residuals.columns)
