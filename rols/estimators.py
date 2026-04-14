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


def _residualize_single(
    y_col: np.ndarray,
    X_np: np.ndarray,
    T: int,
    window: int,
    min_periods: int,
    ridge_term: np.ndarray,
    x_row_valid: np.ndarray,
) -> np.ndarray:
    """
    NaN-robust rolling OLS residuals for a single target column.

    Drops rows within each window where either X or y is NaN,
    then requires at least min_periods clean rows to produce a result.

    Parameters
    ----------
    y_col        : (T,) array — single target column
    X_np         : (T, k) array — regressors
    T            : number of time steps
    window       : rolling window length
    min_periods  : minimum clean rows required
    ridge_term   : (k, k) ridge regularization matrix
    x_row_valid  : (T,) bool — rows where X has no NaN (precomputed)

    Returns
    -------
    (T,) array of residuals, NaN where insufficient clean data
    """
    resid_col = np.full(T, np.nan)
    n_windows = T - window + 1

    for t in range(n_windows):
        start, end = t, t + window
        t_idx = end - 1

        # skip if y is NaN at the prediction point
        if np.isnan(y_col[t_idx]):
            continue

        y_w = y_col[start:end]
        row_ok = x_row_valid[start:end] & ~np.isnan(y_w)

        if row_ok.sum() < min_periods:
            continue

        Xw_c = X_np[start:end][row_ok]
        yw_c = y_w[row_ok]

        XtX = Xw_c.T @ Xw_c + ridge_term
        try:
            beta_t = np.linalg.solve(XtX, Xw_c.T @ yw_c)
            resid_col[t_idx] = y_col[t_idx] - X_np[t_idx] @ beta_t
        except np.linalg.LinAlgError:
            pass

    # Handle min_periods < window — early windows
    if min_periods < window:
        for t in range(min_periods - 1, window - 1):
            if np.isnan(y_col[t]):
                continue
            y_w = y_col[:t + 1]
            row_ok = x_row_valid[:t + 1] & ~np.isnan(y_w)
            if row_ok.sum() < min_periods:
                continue
            Xw_c = X_np[:t + 1][row_ok]
            yw_c = y_w[row_ok]
            XtX = Xw_c.T @ Xw_c + ridge_term
            try:
                beta_t = np.linalg.solve(XtX, Xw_c.T @ yw_c)
                resid_col[t] = y_col[t] - X_np[t] @ beta_t
            except np.linalg.LinAlgError:
                pass

    return resid_col


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

    NaN handling
    ------------
    NaNs in X invalidate the entire window (no regressor → no regression).
    NaNs in y are handled per-column: rows with NaN are dropped within the
    window before solving, and min_periods applies to the remaining clean rows.
    This means NaNs in one target column never contaminate other columns.

    Fast path: if neither X nor y contain any NaNs, uses fully vectorized
    stride-based computation. Falls back to a per-column loop otherwise.

    Parameters
    ----------
    y            : (T, N) DataFrame — targets
    X            : (T, k) DataFrame — regressors
    window       : rolling window length
    min_periods  : minimum clean observations to produce a result
    expanding    : use expanding window instead of rolling
    ridge_lambda : Ridge regularization strength (0.0 = OLS)

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
        # Expanding window — loop required regardless (variable size)
        # Per-column NaN handling: drop rows with NaN in X or y_j
        x_row_valid = ~np.isnan(X_np).any(axis=1)  # (T,)
        for t in range(min_periods - 1, T):
            X_end = X_np[:t + 1]
            y_end = y_np[:t + 1]
            x_ok  = x_row_valid[:t + 1]

            for j in range(N):
                if np.isnan(y_np[t, j]):
                    continue
                row_ok = x_ok & ~np.isnan(y_end[:, j])
                if row_ok.sum() < min_periods:
                    continue
                Xw_c = X_end[row_ok]
                yw_c = y_end[row_ok, j]
                XtX  = Xw_c.T @ Xw_c + ridge_term
                try:
                    beta_t = np.linalg.solve(XtX, Xw_c.T @ yw_c)
                    resid[t, j] = y_np[t, j] - X_np[t] @ beta_t
                except np.linalg.LinAlgError:
                    pass

    elif not (np.isnan(X_np).any() or np.isnan(y_np).any()):
        # Fast path: no NaNs anywhere — fully vectorized via stride tricks
        n_windows = T - window + 1
        if n_windows <= 0:
            return pd.DataFrame(resid, index=y.index, columns=y.columns)

        Xw = _make_windows(X_np, window)  # (n, window, k)
        yw = _make_windows(y_np, window)  # (n, window, N)

        # X-only NaN check (y is clean by construction here)
        has_nan_X = np.isnan(Xw).any(axis=(1, 2))
        valid     = ~has_nan_X

        XtX = np.einsum('twi,twj->tij', Xw, Xw)
        XtX[valid] += ridge_term
        XtY = np.einsum('twi,twn->tin', Xw, yw)

        betas = np.full((n_windows, k, N), np.nan)
        if valid.any():
            betas[valid] = _solve_batch(XtX[valid], XtY[valid])

        t_idx  = np.arange(n_windows) + window - 1
        fitted = np.einsum('ti,tin->tn', X_np[t_idx], betas)
        resid[t_idx] = np.where(has_nan_X[:, None], np.nan, y_np[t_idx] - fitted)

        if min_periods < window:
            for t in range(min_periods - 1, window - 1):
                Xw_t, yw_t = X_np[:t + 1], y_np[:t + 1]
                if np.isnan(Xw_t).any():
                    continue
                XtX_t = Xw_t.T @ Xw_t + ridge_term
                try:
                    resid[t] = y_np[t] - X_np[t] @ np.linalg.solve(XtX_t, Xw_t.T @ yw_t)
                except np.linalg.LinAlgError:
                    pass

    else:
        # NaN-robust path: per-column loop
        # NaNs in X invalidate the row for all columns.
        # NaNs in y are handled per column — one column's NaNs don't affect others.
        x_row_valid = ~np.isnan(X_np).any(axis=1)  # (T,) — shared across columns

        for j in range(N):
            resid[:, j] = _residualize_single(
                y_col=y_np[:, j],
                X_np=X_np,
                T=T,
                window=window,
                min_periods=min_periods,
                ridge_term=ridge_term,
                x_row_valid=x_row_valid,
            )

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
        return X.copy()

    result = X.astype(np.float64).copy()

    for j in range(1, len(cols)):
        y_col  = result[[cols[j]]]
        Xprev  = result[cols[:j]]

        resid = rolling_residualize(
            y=y_col,
            X=Xprev,
            window=window,
            min_periods=min_periods,
            expanding=expanding,
            ridge_lambda=0.0,
        )
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
    resid_np = residuals.to_numpy(dtype=np.float64)
    f_np     = factor_values.to_numpy(dtype=np.float64)
    T, N     = resid_np.shape
    se       = np.full((T, N), np.nan)

    def _nw_se_window(f_w: np.ndarray, e_w: np.ndarray) -> np.ndarray:
        n_obs = len(f_w)
        score = f_w[:, None] * e_w
        xx    = f_w @ f_w
        S     = np.einsum('ti,ti->i', score, score) / n_obs
        for lag in range(1, n_lags + 1):
            w     = 1.0 - lag / (n_lags + 1)
            gamma = np.einsum('ti,ti->i', score[lag:], score[:-lag]) / n_obs
            S    += 2 * w * gamma
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