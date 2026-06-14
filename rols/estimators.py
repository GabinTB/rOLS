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

import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided


def _warn_singular(n: int) -> None:
    """Emit a single aggregated RuntimeWarning for n singular windows."""
    if n <= 0:
        return
    warnings.warn(
        f"{n} singular window(s) — affected estimates set to NaN. "
        "This usually means collinear regressors or a degenerate window; "
        "consider adding Ridge regularization (lambda_ > 0).",
        RuntimeWarning,
        stacklevel=3,
    )


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


def _solve_batch(XtX: np.ndarray, XtY: np.ndarray, warn_singular: bool = True) -> np.ndarray:
    """
    Batch solve XtX[i] @ beta[i] = XtY[i].
    Falls back element-wise on singular windows.
    Returns betas with NaN where solve failed.

    If warn_singular is True, emits a single aggregated RuntimeWarning
    summarizing how many windows were singular.
    """
    n, k, N = XtY.shape
    betas = np.full((n, k, N), np.nan)
    n_singular = 0
    try:
        result = np.linalg.solve(XtX, XtY)
        result[~np.isfinite(result)] = np.nan
        betas[...] = result
    except np.linalg.LinAlgError:
        for i in range(n):
            try:
                b = np.linalg.solve(XtX[i], XtY[i])
                b[~np.isfinite(b)] = np.nan
                betas[i] = b
            except np.linalg.LinAlgError:
                n_singular += 1
    if warn_singular:
        _warn_singular(n_singular)
    return betas


def _residualize_single(
    y_col: np.ndarray,
    X_np: np.ndarray,
    T: int,
    window: int,
    min_periods: int,
    ridge_term: np.ndarray,
    x_row_valid: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
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
    weights      : (window,) array — per-position observation weights
        (oldest-to-newest), or None for equal weighting. After NaN rows are
        masked out, the surviving weights are renormalized to sum to 1.

    Returns
    -------
    (resid_col, n_singular) : (T,) array of residuals (NaN where insufficient
        clean data or the solve was singular) and the count of singular windows.
    """
    resid_col = np.full(T, np.nan)
    n_singular = 0
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

        if weights is not None:
            w_c = weights[row_ok]
            w_c = w_c / w_c.sum()
            XtX = Xw_c.T @ (Xw_c * w_c[:, None]) + ridge_term
            rhs = Xw_c.T @ (yw_c * w_c)
        else:
            XtX = Xw_c.T @ Xw_c + ridge_term
            rhs = Xw_c.T @ yw_c
        try:
            beta_t = np.linalg.solve(XtX, rhs)
            resid_col[t_idx] = y_col[t_idx] - X_np[t_idx] @ beta_t
        except np.linalg.LinAlgError:
            n_singular += 1

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
            if weights is not None:
                w_c = weights[-(t + 1):][row_ok]
                w_c = w_c / w_c.sum()
                XtX = Xw_c.T @ (Xw_c * w_c[:, None]) + ridge_term
                rhs = Xw_c.T @ (yw_c * w_c)
            else:
                XtX = Xw_c.T @ Xw_c + ridge_term
                rhs = Xw_c.T @ yw_c
            try:
                beta_t = np.linalg.solve(XtX, rhs)
                resid_col[t] = y_col[t] - X_np[t] @ beta_t
            except np.linalg.LinAlgError:
                n_singular += 1

    return resid_col, n_singular


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
    warn_singular: bool = True,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Compute rolling OLS (or Ridge) residuals: y_t - X_t @ beta_t for each t.

    Ridge adds lambda * I to X'X before solving, shrinking betas toward zero.
    Set ridge_lambda=0.0 for standard OLS (default).

    Observation weighting
    ---------------------
    If ``weights`` (length ``window``, oldest-to-newest) is provided, each
    window is solved as a weighted least squares problem: the gram matrix
    accumulates X'WX and X'Wy instead of X'X and X'y. This is how EWMA
    observation weighting (``RollingOLS(ewma_halflife=...)``) is threaded
    through the Frisch-Waugh residualization. When rows are dropped for NaN
    handling, the surviving weights are renormalized to sum to 1 so the
    weighting scheme is unaffected by missing data. ``weights=None`` (default)
    is equal weighting and is bit-for-bit identical to the unweighted path.
    Not supported with ``expanding=True``.

    NaN handling
    ------------
    NaNs in X invalidate the entire window (no regressor → no regression).
    NaNs in y are handled per-column: rows with NaN are dropped within the
    window before solving, and min_periods applies to the remaining clean rows.
    This means NaNs in one target column never contaminate other columns.

    Three rolling paths are selected automatically:

    1. No NaNs anywhere — fully vectorized stride-based computation (fast path).
    2. NaNs only in y, X clean — vectorized over windows with an O(N) loop over
       assets (intermediate path). X'X is recomputed per asset because NaN rows
       in y are dropped, but T is fully vectorized. This is the typical case for
       large asset panels (e.g. index constituents entering/leaving over time).
    3. NaNs in X — per-column, per-window loop (``_residualize_single``).

    Note
    ----
    Internal matrix operations (gram matrix accumulation and the linear solve)
    always use float64 for numerical stability, regardless of the input dtype.
    np.linalg.solve loses accuracy in float32 for ill-conditioned matrices, so
    inputs are upcast here. The RollingOLS ``dtype`` parameter controls pandas
    DataFrame storage only — it does not change the precision of the solve.

    Parameters
    ----------
    y            : (T, N) DataFrame — targets
    X            : (T, k) DataFrame — regressors
    window       : rolling window length
    min_periods  : minimum clean observations to produce a result
    expanding    : use expanding window instead of rolling
    ridge_lambda : Ridge regularization strength (0.0 = OLS)
    warn_singular : if True (default), emit a single aggregated RuntimeWarning
        when one or more windows are singular (estimates set to NaN). Set False
        to suppress (e.g. when singular warm-up windows are expected).
    weights      : (window,) array of per-position observation weights
        (oldest-to-newest), or None for equal weighting. Renormalized over the
        surviving rows after NaN masking. Not supported with expanding=True.

    Returns
    -------
    pd.DataFrame, same shape/index/columns as y
    """
    if weights is not None and expanding:
        raise ValueError("weights are not supported with expanding=True")
    y_np = y.to_numpy(dtype=np.float64)
    X_np = X.to_numpy(dtype=np.float64)
    T, N = y_np.shape
    k    = X_np.shape[1]
    resid = np.full((T, N), np.nan)
    ridge_term = ridge_lambda * np.eye(k)
    n_singular = 0

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
                    n_singular += 1

    elif not np.isnan(X_np).any() and not np.isnan(y_np).any():
        # Fast path: no NaNs anywhere — fully vectorized via stride tricks
        n_windows = T - window + 1
        if n_windows <= 0:
            return pd.DataFrame(resid, index=y.index, columns=y.columns)

        Xw = _make_windows(X_np, window)  # (n, window, k)
        yw = _make_windows(y_np, window)  # (n, window, N)

        # X-only NaN check (y is clean by construction here)
        has_nan_X = np.isnan(Xw).any(axis=(1, 2))
        valid     = ~has_nan_X

        if weights is not None:
            # Weighted gram matrix: X'WX and X'Wy. Apply weights to one side
            # of the einsum so the accumulation sums w_t * x_t * (.)_t.
            Xw_w = Xw * weights[None, :, None]   # (n, window, k)
            XtX  = np.einsum('twi,twj->tij', Xw_w, Xw)
            XtY  = np.einsum('twi,twn->tin', Xw_w, yw)
        else:
            XtX = np.einsum('twi,twj->tij', Xw, Xw)
            XtY = np.einsum('twi,twn->tin', Xw, yw)
        XtX[valid] += ridge_term

        betas = np.full((n_windows, k, N), np.nan)
        if valid.any():
            # _solve_batch emits its own aggregated warning for the batch.
            betas[valid] = _solve_batch(XtX[valid], XtY[valid], warn_singular=warn_singular)

        t_idx  = np.arange(n_windows) + window - 1
        fitted = np.einsum('ti,tin->tn', X_np[t_idx], betas)
        resid[t_idx] = np.where(has_nan_X[:, None], np.nan, y_np[t_idx] - fitted)

        if min_periods < window:
            for t in range(min_periods - 1, window - 1):
                Xw_t, yw_t = X_np[:t + 1], y_np[:t + 1]
                if np.isnan(Xw_t).any():
                    continue
                if weights is not None:
                    w_t = weights[-(t + 1):]
                    w_t = w_t / w_t.sum()
                    XtX_t = Xw_t.T @ (Xw_t * w_t[:, None]) + ridge_term
                    rhs_t = Xw_t.T @ (yw_t * w_t[:, None])
                else:
                    XtX_t = Xw_t.T @ Xw_t + ridge_term
                    rhs_t = Xw_t.T @ yw_t
                try:
                    resid[t] = y_np[t] - X_np[t] @ np.linalg.solve(XtX_t, rhs_t)
                except np.linalg.LinAlgError:
                    n_singular += 1

    elif not np.isnan(X_np).any():
        # Intermediate vectorized NaN-robust path: X is clean, NaNs only in y.
        #
        # Because NaN rows in y_j are dropped from the regression (matching
        # _residualize_single), the gram matrix X'X differs per asset and is
        # NOT shared. But T is fully vectorized inside the per-asset loop via
        # stride tricks + einsum, so the Python loop is O(N) instead of the
        # O(T * N) of the per-column fallback. At MSCI-World scale (~2300 assets
        # with NaNs always present) this is the path that runs, giving a large
        # speedup over the per-column loop while producing identical results.
        n_windows = T - window + 1
        if n_windows <= 0:
            return pd.DataFrame(resid, index=y.index, columns=y.columns)

        Xw = _make_windows(X_np, window)  # (n_windows, window, k)
        yw = _make_windows(y_np, window)  # (n_windows, window, N)
        t_idx = np.arange(n_windows) + window - 1

        for j in range(N):
            yw_j        = yw[:, :, j]            # (n_windows, window)
            valid_j     = ~np.isnan(yw_j)        # (n_windows, window)
            valid_count = valid_j.sum(axis=1)    # (n_windows,)
            # The prediction point is the last row of each window; if y_j is
            # NaN there the window is skipped (matching _residualize_single).
            pred_valid  = valid_j[:, -1]
            sufficient  = (valid_count >= min_periods) & pred_valid
            if not sufficient.any():
                continue

            # Zero out the X (and y) rows where y_j is NaN so they drop out of
            # both X'X and X'y for this asset.
            Xw_masked = np.where(valid_j[:, :, None], Xw, 0.0)   # (n_windows, window, k)
            yw_masked = np.where(valid_j, yw_j, 0.0)             # (n_windows, window)

            if weights is not None:
                # Per-window weights restricted to the surviving (non-NaN) rows,
                # renormalized to sum to 1. Insufficient windows have zero sum
                # but are filtered out by `sufficient`, so the divide is guarded.
                wm    = np.where(valid_j, weights[None, :], 0.0)       # (n_windows, window)
                wsum  = wm.sum(axis=1, keepdims=True)                  # (n_windows, 1)
                wn    = np.divide(wm, wsum, out=np.zeros_like(wm), where=wsum > 0)
                Xw_w  = Xw_masked * wn[:, :, None]                    # (n_windows, window, k)
                XtX_j = np.einsum('twi,twj->tij', Xw_w, Xw_masked)    # (n_windows, k, k)
                XtY_j = np.einsum('twi,tw->ti', Xw_w, yw_masked)[:, :, None]
            else:
                XtX_j = np.einsum('twi,twj->tij', Xw_masked, Xw_masked)  # (n_windows, k, k)
                XtY_j = np.einsum('twi,tw->ti', Xw_masked, yw_masked)[:, :, None]
            XtX_j[sufficient] += ridge_term

            betas_j = np.full((n_windows, k, 1), np.nan)
            # Aggregate the singular warning once for the whole call rather than
            # emitting one per asset.
            betas_j[sufficient] = _solve_batch(
                XtX_j[sufficient], XtY_j[sufficient], warn_singular=False
            )
            n_singular += int(
                (sufficient & np.isnan(betas_j[:, :, 0]).any(axis=1)).sum()
            )

            fitted = np.einsum('ti,ti->t', X_np[t_idx], betas_j[:, :, 0])
            resid[t_idx, j] = np.where(sufficient, y_np[t_idx, j] - fitted, np.nan)

        # Early windows (min_periods < window) use variable-size expanding
        # windows — handled per asset exactly as in _residualize_single.
        if min_periods < window:
            for t in range(min_periods - 1, window - 1):
                for j in range(N):
                    if np.isnan(y_np[t, j]):
                        continue
                    y_w    = y_np[:t + 1, j]
                    row_ok = ~np.isnan(y_w)
                    if row_ok.sum() < min_periods:
                        continue
                    Xw_c  = X_np[:t + 1][row_ok]
                    yw_c  = y_w[row_ok]
                    if weights is not None:
                        w_c = weights[-(t + 1):][row_ok]
                        w_c = w_c / w_c.sum()
                        XtX_t = Xw_c.T @ (Xw_c * w_c[:, None]) + ridge_term
                        rhs_t = Xw_c.T @ (yw_c * w_c)
                    else:
                        XtX_t = Xw_c.T @ Xw_c + ridge_term
                        rhs_t = Xw_c.T @ yw_c
                    try:
                        beta_t = np.linalg.solve(XtX_t, rhs_t)
                        resid[t, j] = y_np[t, j] - X_np[t] @ beta_t
                    except np.linalg.LinAlgError:
                        n_singular += 1

    else:
        # NaN-robust per-column fallback: NaNs present in X.
        # NaNs in X invalidate the row for all columns.
        # NaNs in y are handled per column — one column's NaNs don't affect others.
        x_row_valid = ~np.isnan(X_np).any(axis=1)  # (T,) — shared across columns

        for j in range(N):
            resid[:, j], col_singular = _residualize_single(
                y_col=y_np[:, j],
                X_np=X_np,
                T=T,
                window=window,
                min_periods=min_periods,
                ridge_term=ridge_term,
                x_row_valid=x_row_valid,
                weights=weights,
            )
            n_singular += col_singular

    if warn_singular:
        _warn_singular(n_singular)

    return pd.DataFrame(resid, index=y.index, columns=y.columns)


# ---------------------------------------------------------------------------
# Rolling Gram-Schmidt orthogonalization
# ---------------------------------------------------------------------------

def rolling_gram_schmidt(
    X: pd.DataFrame,
    window: int,
    min_periods: int,
    expanding: bool,
    warn_singular: bool = True,
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
            warn_singular=warn_singular,
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

    Note
    ----
    HAC standard errors are computed with equal weights regardless of
    ``ewma_halflife``. EWMA-weighted HAC is not yet implemented, so SEs from a
    model fitted with EWMA observation weighting still treat every observation
    in the window equally.

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

    def _fill_window(t: int, f_w: np.ndarray, e_w: np.ndarray) -> None:
        # Factor NaN invalidates the whole window — no regressor, no SE.
        if np.isnan(f_w).any():
            return
        if len(f_w) <= n_lags:
            return
        # Residual NaNs are handled per-asset: only contaminated columns are
        # left NaN, clean columns get a valid SE (mirrors rolling_residualize).
        asset_nan = np.isnan(e_w).any(axis=0)
        if asset_nan.all():
            return
        valid = ~asset_nan
        se[t, valid] = _nw_se_window(f_w, e_w[:, valid])

    if expanding:
        for t in range(min_periods - 1, T):
            _fill_window(t, f_np[:t + 1], resid_np[:t + 1])
    else:
        for t in range(window - 1, T):
            start = t - window + 1
            _fill_window(t, f_np[start:t + 1], resid_np[start:t + 1])

        if min_periods < window:
            for t in range(min_periods - 1, window - 1):
                _fill_window(t, f_np[:t + 1], resid_np[:t + 1])

    return pd.DataFrame(se, index=residuals.index, columns=residuals.columns)